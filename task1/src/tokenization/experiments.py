"""Fixed-data, lossless ByteLevel BPE/Unigram comparison; no language-model training."""

import os

os.environ["RAYON_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import argparse
import importlib.metadata
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.data.corpus import ARTIFACTS, SEED, files_for, sha256_file, text_hash, write_json

DIRECTORY = ARTIFACTS / "tokenizers"
SIZES = [8192, 16384, 32768, 65536]
ALGORITHMS = ["bpe", "unigram"]
SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
BATCH_SIZE = 128
SPLITS = ["train", "top_dev", "validation"]
SELECTION = {"model_parameters": 300_000_000, "hidden_size": 1024,
             "tied_weights": True, "embedding_budget_fraction": .15, "token_tolerance": .05}
EXAMPLES = [
    "Българският език има богата история. Учениците четат книги в библиотеката.",
    "учител учителка учители училище училищата",
    "На 25.10.2026 г. цената е 123,45 лв., а отстъпката — 15%.",
    "Вижте https://example.bg/path?q=София&n=42 или пишете на info@example.bg.",
    "София / Sofia: Python и машинно обучение, hello world!",
    "  два  интервала\tтаб\nнов ред\r\nкрай  ",
    "й и\u0306; ѝ и\u0300; café cafe\u0301; 😀🧑🏽‍💻 中文 العربية",
    "Литералите [PAD] [UNK] [BOS] [EOS] са обикновен текст.",
    "", " \t\n\r\n", "\x00\x01\u200b\ufeffкрай",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def candidate_name(algorithm, size):
    return f"{algorithm}_{size}"


def sample_path(split):
    return DIRECTORY / "samples" / f"{split}.parquet"


def text_batches(split):
    for batch in pq.ParquetFile(sample_path(split)).iter_batches(batch_size=BATCH_SIZE, columns=["text"]):
        yield batch.column(0).to_pylist()


def prepare():
    """Read selected row groups without overwriting the corpus audit samples."""
    manifest_path = DIRECTORY / "samples/manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["corpus_manifest_sha256"] != sha256_file(ARTIFACTS / "corpus/manifest.json"):
            raise ValueError("Tokenizer samples belong to another corpus manifest")
        for record in manifest["files"]:
            if sha256_file(DIRECTORY / record["path"]) != record["sha256"]:
                raise ValueError(f"Changed tokenizer sample: {record['path']}")
        print("Fixed tokenizer samples already verified", flush=True)
        return
    indices = np.random.default_rng(SEED).choice(1_000_000, 55_000, replace=False)
    selected = np.sort(indices)
    frames, offset = [], 0
    for path in files_for("top"):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            lo, hi = np.searchsorted(selected, [offset, offset + count])
            if hi > lo:
                table = parquet.read_row_group(group, columns=["id", "text"])
                frame = table.take(pa.array(selected[lo:hi] - offset)).to_pandas()
                frame["source_row"] = selected[lo:hi]
                frames.append(frame)
            offset += count
        print(f"Sampled source {path.name}", flush=True)
    if offset != 1_000_000:
        raise ValueError("Unexpected training corpus size")
    combined = pd.concat(frames, ignore_index=True).set_index("source_row").loc[indices].reset_index()
    validation = pq.read_table(ARTIFACTS / "corpus/validation.parquet", columns=["id", "text"]).to_pandas()
    validation["source_row"] = np.arange(len(validation))
    parts = {"train": combined.iloc[:50_000].copy(), "top_dev": combined.iloc[50_000:].copy(),
             "validation": validation}
    seen_ids, seen_hashes, records = set(), set(), []
    for split, frame in parts.items():
        frame["content_hash"] = frame.text.map(lambda text: text_hash(text).hex())
        frame["n_chars"] = frame.text.str.len()
        frame["n_words"] = frame.text.map(lambda text: len(text.split()))
        frame["n_bytes"] = frame.text.map(lambda text: len(text.encode("utf-8")))
        ids, hashes = set(frame.id), set(frame.content_hash)
        if len(ids) != len(frame) or len(hashes) != len(frame) or seen_ids & ids or seen_hashes & hashes:
            raise ValueError(f"Duplicate/overlapping sample: {split}")
        if len(frame) != (50_000 if split == "train" else 5000) or frame.n_chars.min() < 1:
            raise ValueError(f"Unexpected sample: {split}")
        seen_ids.update(ids)
        seen_hashes.update(hashes)
        path = sample_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd", row_group_size=1000)
        records.append({"split": split, "path": str(path.relative_to(DIRECTORY)), "rows": len(frame),
                        "characters": int(frame.n_chars.sum()), "utf8_bytes": int(frame.n_bytes.sum()),
                        "sha256": sha256_file(path)})
    write_json(manifest_path, {"seed": SEED, "method": "Uniform global rows without replacement; first 50k train, next 5k dev",
                              "corpus_manifest_sha256": sha256_file(ARTIFACTS / "corpus/manifest.json"),
                              "disjoint_ids_and_normalized_hashes": True, "test_accessed": False, "files": records})
    print("Prepared fixed 50k / 5k / 5k tokenizer samples", flush=True)


def configuration(algorithm, size):
    common = {"vocab_size": size, "special_tokens": SPECIALS, "initial_alphabet": ALPHABET,
              "show_progress": True}
    specific = ({"min_frequency": 2, "max_token_length": 64} if algorithm == "bpe" else
                {"unk_token": "[UNK]", "max_piece_length": 64, "shrinking_factor": .75, "n_sub_iterations": 2})
    return {"algorithm": algorithm, "trainer": dict(common, **specific), "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
            "decoder": "ByteLevel", "threads": 8, "special_strings_are_plain_text": True,
            "automatic_special_tokens": False, "padding": False, "truncation": False,
            "sampling_seed": SEED, "training_bitwise_determinism_guaranteed": False,
            "samples_manifest_sha256": sha256_file(DIRECTORY / "samples/manifest.json"),
            "versions": {name: importlib.metadata.version(name) for name in ["tokenizers", "transformers", "numpy", "pyarrow"]}}


def train(algorithm, size):
    directory = DIRECTORY / "candidates" / candidate_name(algorithm, size)
    config = configuration(algorithm, size)
    if (directory / "training.json").exists():
        report = read_json(directory / "training.json")
        if config != read_json(directory / "config.json") or sha256_file(directory / "tokenizer.json") != report["sha256"]:
            raise ValueError(f"Existing candidate does not match requested configuration: {directory.name}")
        print(f"Already trained: {directory.name}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", config)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", dropout=None) if algorithm == "bpe" else models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = (trainers.BpeTrainer if algorithm == "bpe" else trainers.UnigramTrainer)(**config["trainer"])
    started = time.perf_counter()
    tokenizer.train_from_iterator(text_batches("train"), trainer=trainer, length=50_000)
    seconds = time.perf_counter() - started
    tokenizer.encode_special_tokens = True
    if tokenizer.get_vocab_size() != size:
        raise ValueError(f"Requested {size}, got {tokenizer.get_vocab_size()} tokens")
    if [tokenizer.token_to_id(token) for token in SPECIALS] != list(range(4)):
        raise ValueError("Unexpected special-token IDs")
    path = directory / "tokenizer.json"
    tokenizer.save(str(path))
    write_json(directory / "training.json", {"candidate": directory.name, "seconds": seconds,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "actual_vocab_size": tokenizer.get_vocab_size(), "sha256": sha256_file(path)})
    print(f"Trained {directory.name}: {seconds:.1f}s", flush=True)


def load_tokenizer(name):
    tokenizer = Tokenizer.from_file(str(DIRECTORY / "candidates" / name / "tokenizer.json"))
    # This runtime switch is not serialized by tokenizers; set it on every load.
    tokenizer.encode_special_tokens = True
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "report", "run", "selected"])
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--vocab-size", type=int, choices=SIZES)
    args = parser.parse_args()
    if args.command == "selected":
        prepare()
        train("bpe", 32768)
        from src.tokenization.metrics import evaluate
        from src.tokenization.report import export_selected
        evaluate("bpe_32768")
        export_selected("bpe_32768")
        write_json(DIRECTORY / "selected_recipe.json", {
            "candidate": "bpe_32768", "reason": "Fixed recipe selected in the original eight-tokenizer experiment",
            "new_comparison_performed": False, "samples": read_json(DIRECTORY / "samples/manifest.json")})
    elif args.command == "prepare":
        prepare()
    elif args.command in {"train", "evaluate"}:
        if args.algorithm is None or args.vocab_size is None:
            parser.error("train/evaluate require --algorithm and --vocab-size")
        if args.command == "train":
            train(args.algorithm, args.vocab_size)
        else:
            from src.tokenization.metrics import evaluate
            evaluate(candidate_name(args.algorithm, args.vocab_size))
    elif args.command == "report":
        from src.tokenization.report import report
        report()
    else:
        prepare()
        for algorithm in ALGORITHMS:
            for size in SIZES:
                for command in ["train", "evaluate"]:
                    subprocess.run([sys.executable, "-m", "src.tokenization.experiments", command, "--algorithm", algorithm,
                                    "--vocab-size", str(size)], check=True)
        from src.tokenization.report import report
        report()


if __name__ == "__main__":
    main()
