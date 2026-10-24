"""Streaming measurements on the fixed tokenizer samples."""

import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.corpus import sha256_file, write_json
from src.tokenization.experiments import (ALPHABET, BATCH_SIZE, DIRECTORY, EXAMPLES, SPECIALS,
                                   SPLITS, load_tokenizer, read_json, sample_path, text_batches)


def quantiles(values):
    values = np.asarray(values)
    result = {"mean": float(values.mean()), "max": float(values.max())}
    result.update(zip(["p50", "p90", "p95", "p99"], map(float, np.quantile(values, [.5, .9, .95, .99]))))
    return result


def byte_decoder():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    remaining = [byte for byte in range(256) if byte not in visible]
    return {chr(code): byte for byte, code in zip(visible + remaining, visible + list(range(256, 256 + len(remaining))))}


BYTE_DECODER = byte_decoder()


def readable_piece(piece):
    if piece in SPECIALS:
        return piece
    raw = bytes(BYTE_DECODER[char] for char in piece)
    try:
        return raw.decode("utf-8").replace(" ", "␠").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    except UnicodeDecodeError:
        return f"<bytes:{raw.hex(' ')}>"


def frequency_summary(counts, kinds):
    total = int(counts.sum())
    positive = counts[counts > 0] / total
    entropy = float(-(positive * np.log(positive)).sum())
    result = {"entropy_nats": entropy, "effective_vocabulary": float(np.exp(entropy)),
              "top100_mass": float(np.sort(counts)[-100:].sum() / total),
              "top1000_mass": float(np.sort(counts)[-1000:].sum() / total)}
    result["by_kind"] = {}
    for kind in ["learned", "byte", "special"]:
        values = counts[kinds == kind]
        result["by_kind"][kind] = {"types": len(values), "unused": int((values == 0).sum()),
            "unused_fraction": float((values == 0).mean()), "rare_1_9": int(((values > 0) & (values < 10)).sum()),
            "rare_1_9_fraction": float(((values > 0) & (values < 10)).mean()),
            "token_mass": float(values.sum() / total)}
    return result


def measure_split(tokenizer, split, vocabulary):
    meta = pq.read_table(sample_path(split), columns=["id", "content_hash", "n_chars", "n_words", "n_bytes"]).to_pandas()
    counts = np.zeros(len(vocabulary), dtype=np.int64)
    lengths, mismatches, mismatch_examples = [], 0, []
    started = time.perf_counter()
    for texts in text_batches(split):
        encoded = tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        ids = [encoding.ids for encoding in encoded]
        lengths.extend(map(len, ids))
        flat = np.fromiter((token for document in ids for token in document), dtype=np.int64)
        counts += np.bincount(flat, minlength=len(counts))
        if split != "train":
            decoded = tokenizer.decode_batch(ids, skip_special_tokens=False)
            for text, reconstructed in zip(texts, decoded):
                if text != reconstructed:
                    mismatches += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append({"original": repr(text[:300]), "decoded": repr(reconstructed[:300])})
    if len(lengths) != len(meta):
        raise ValueError(f"Incomplete measurement: {split}")
    meta["n_tokens"] = lengths
    meta["tokens_per_1000_chars"] = meta.n_tokens / meta.n_chars * 1000
    total = int(counts.sum())
    result = {"documents": len(meta), "tokens": total, "words": int(meta.n_words.sum()),
              "characters": int(meta.n_chars.sum()), "utf8_bytes": int(meta.n_bytes.sum()),
              "tokens_per_word": total / int(meta.n_words.sum()),
              "characters_per_token": int(meta.n_chars.sum()) / total,
              "bytes_per_token": int(meta.n_bytes.sum()) / total,
              "length": quantiles(meta.n_tokens), "density": quantiles(meta.tokens_per_1000_chars),
              "longer_than_context": {str(size): float(meta.n_tokens.gt(size).mean()) for size in [256, 512, 1024]},
              "unk_tokens": int(counts[tokenizer.token_to_id("[UNK]")]),
              "unk_fraction": int(counts[tokenizer.token_to_id("[UNK]")]) / total,
              "roundtrip_checked": split != "train", "roundtrip_mismatches": mismatches if split != "train" else None,
              "roundtrip_examples": mismatch_examples, "measurement_seconds": time.perf_counter() - started,
              "frequency": frequency_summary(counts, vocabulary.kind.to_numpy())}
    return meta, counts, result


def benchmark(tokenizer):
    texts = pq.read_table(sample_path("top_dev"), columns=["text"]).column(0).to_pylist()
    batches = [texts[start:start + BATCH_SIZE] for start in range(0, len(texts), BATCH_SIZE)]
    seconds, token_totals = [], []
    for run in range(4):
        started, total = time.perf_counter(), 0
        for batch in batches:
            encoded = tokenizer.encode_batch_fast(batch, add_special_tokens=False)
            total += sum(len(item) for item in encoded)
        elapsed = time.perf_counter() - started
        if run:
            seconds.append(elapsed)
            token_totals.append(total)
    if len(set(token_totals)) != 1:
        raise ValueError("Non-deterministic inference")
    median = float(np.median(seconds))
    return {"threads": 8, "batch_size": BATCH_SIZE, "warmups": 1, "runs": 3,
            "seconds": seconds, "median_seconds": median,
            "utf8_mb_per_second": sum(len(text.encode('utf-8')) for text in texts) / 1e6 / median,
            "tokens_per_second": token_totals[0] / median,
            "disk_io_in_timing": False, "method": "encode_batch_fast; no offsets; content only"}


def evaluate(name):
    directory = DIRECTORY / "candidates" / name
    summary_path = directory / "metrics.json"
    if summary_path.exists():
        previous = read_json(summary_path)
        if previous["tokenizer_sha256"] != sha256_file(directory / "tokenizer.json"):
            raise ValueError("Measurements belong to a different tokenizer")
        print(f"Already measured: {name}", flush=True)
        return
    tokenizer = load_tokenizer(name)
    vocabulary = pd.DataFrame(sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]), columns=["piece", "token_id"])
    vocabulary["kind"] = vocabulary.piece.map(lambda piece: "special" if piece in SPECIALS else "byte" if piece in ALPHABET else "learned")
    vocabulary["readable"] = vocabulary.piece.map(readable_piece)
    results = {}
    for split in SPLITS:
        frame, counts, results[split] = measure_split(tokenizer, split, vocabulary)
        frame.to_parquet(directory / f"{split}_documents.parquet", index=False)
        vocabulary[f"count_{split}"] = counts
        print(f"Measured {name} / {split}: {results[split]['tokens']:,} tokens", flush=True)
    vocabulary.to_parquet(directory / "vocabulary.parquet", index=False)
    examples = []
    for text in EXAMPLES:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        reconstructed = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        examples.append({"text": text, "ids": encoded.ids, "pieces": encoded.tokens,
                         "readable_pieces": [readable_piece(piece) for piece in encoded.tokens],
                         "roundtrip": text == reconstructed, "unk_tokens": encoded.ids.count(tokenizer.token_to_id("[UNK]"))})
    write_json(directory / "examples.json", examples)
    write_json(summary_path, {"candidate": name, "tokenizer_sha256": sha256_file(directory / "tokenizer.json"),
               "splits": results, "benchmark": benchmark(tokenizer),
               "edge_cases": {"count": len(examples), "passed": all(row["roundtrip"] and not row["unk_tokens"] for row in examples)}})
    print(f"Completed evaluation: {name}", flush=True)
