"""Lossless document tokenization and fixed-length next-token blocks."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from src.paths import TASK as ROOT, ARTIFACTS as ALL_ARTIFACTS

ARTIFACTS = ALL_ARTIFACTS / "training"
DATA = ARTIFACTS / "data"
TOKENIZER = ALL_ARTIFACTS / "tokenizers/selected"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    """Prepare only train/validation; reuse a split only after checksum verification."""
    DATA.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    tokenizer.no_padding()
    tokenizer.no_truncation()
    assert tokenizer.get_vocab_size() == 32768
    assert tokenizer.token_to_id("[EOS]") == 3
    sources = {"train": sorted((ALL_ARTIFACTS / "corpus/train").glob("*.parquet")),
               "validation": [ALL_ARTIFACTS / "corpus/validation.parquet"]}
    manifest = {"tokenizer_sha256": sha256(TOKENIZER / "tokenizer.json"),
                "dtype": "<u2", "eos_id": 3, "bos_added": False,
                "normalization": None, "seed": 42, "splits": {}}
    for split, files in sources.items():
        source_records = [{"path": str(Path("artifacts") / p.relative_to(ALL_ARTIFACTS)), "sha256": sha256(p)} for p in files]
        signature = {"sources": source_records, "tokenizer_sha256": manifest["tokenizer_sha256"]}
        split_manifest = DATA / f"{split}.json"
        if split_manifest.exists():
            record = read_json(split_manifest)
            if record["signature"] != signature:
                raise ValueError(f"Changed inputs for {split}; choose a new data directory")
            for name, checksum in record["checksums"].items():
                if sha256(DATA / name) != checksum:
                    raise ValueError(f"Corrupted prepared data: {name}")
            manifest["splits"][split] = record
            continue
        binary = DATA / f"{split}.bin"
        index = DATA / f"{split}_documents.parquet"
        schema = pa.schema([("id", pa.string()), ("offset", pa.uint64()),
                            ("n_tokens", pa.uint32()), ("n_chars", pa.uint32())])
        offset, documents = 0, 0
        lengths = []
        with binary.with_suffix(".tmp").open("wb") as output, pq.ParquetWriter(
                index.with_suffix(".tmp"), schema, compression="zstd") as writer:
            for source in files:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=128, columns=["id", "text"]):
                    values = batch.to_pydict()
                    encodings = tokenizer.encode_batch(values["text"], add_special_tokens=False)
                    rows = []
                    for doc_id, text, encoding in zip(values["id"], values["text"], encodings):
                        ids = np.asarray(encoding.ids, dtype="<u2")
                        if not len(ids) or ids.min() < 4 or ids.max() >= 32768:
                            raise ValueError(f"Empty or unexpected special token in {doc_id}")
                        if documents % 10000 == 0 and tokenizer.decode(encoding.ids) != text:
                            raise ValueError(f"Round-trip failure: {doc_id}")
                        ids.tofile(output)
                        np.asarray([3], dtype="<u2").tofile(output)
                        rows.append({"id": doc_id, "offset": offset,
                                     "n_tokens": len(ids), "n_chars": len(text)})
                        lengths.append(len(ids))
                        offset += len(ids) + 1
                        documents += 1
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                print(f"{split}: {documents:,} documents, {offset:,} tokens", flush=True)
        assert documents == (1_000_000 if split == "train" else 5000)
        binary.with_suffix(".tmp").replace(binary)
        index.with_suffix(".tmp").replace(index)
        record = {"signature": signature, "documents": documents, "tokens": offset,
                  "content_tokens": offset - documents, "targets": offset - 1,
                  "document_lengths": dict(zip(["min", "p50", "p90", "p95", "p99", "max"],
                                               np.quantile(lengths, [0, .5, .9, .95, .99, 1]).tolist())),
                  "checksums": {p.name: sha256(p) for p in [binary, index]}}
        write_json(split_manifest, record)
        manifest["splits"][split] = record
    write_json(DATA / "manifest.json", manifest)
    return manifest


class TokenBlocks:
    """Adjacent blocks share one input/target token; no next-token target is dropped."""

    def __init__(self, split, length=2048):
        self.tokens = np.memmap(DATA / f"{split}.bin", dtype="<u2", mode="r")
        self.length = length
        self.targets = len(self.tokens) - 1

    def __len__(self):
        return (self.targets + self.length - 1) // self.length

    def batch(self, indices, device):
        x = np.zeros((len(indices), self.length), dtype=np.int64)
        y = np.full_like(x, -100)
        for row, index in enumerate(indices):
            start = int(index) * self.length
            size = min(self.length, self.targets - start)
            if size <= 0:
                raise IndexError(index)
            x[row, :size] = self.tokens[start:start + size]
            y[row, :size] = self.tokens[start + 1:start + size + 1]
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def target_count(self, indices):
        return int(np.minimum(self.length, self.targets - np.asarray(indices) * self.length).sum())


def verify_data(length=2048):
    manifest = read_json(DATA / "manifest.json")
    for split, record in manifest["splits"].items():
        blocks = TokenBlocks(split, length)
        assert len(blocks.tokens) == record["tokens"]
        index = pq.read_table(DATA / f"{split}_documents.parquet").to_pydict()
        offsets = np.asarray(index["offset"], dtype=np.int64)
        lengths = np.asarray(index["n_tokens"], dtype=np.int64)
        assert np.all(blocks.tokens[offsets + lengths] == 3)
        assert np.array_equal(offsets[1:], offsets[:-1] + lengths[:-1] + 1)
        assert int(offsets[-1] + lengths[-1] + 1) == len(blocks.tokens)
        assert blocks.target_count(np.arange(len(blocks))) == record["targets"]
        x, y = blocks.batch([0, len(blocks) - 1], "cpu")
        assert torch.equal(x[0, 1:], y[0, :-1])
        assert int((y[-1] != -100).sum()) == (blocks.targets - 1) % length + 1
        for name, checksum in record["checksums"].items():
            assert sha256(DATA / name) == checksum, name
    return {"verified": True, "splits": {s: r["tokens"] for s, r in manifest["splits"].items()}}
