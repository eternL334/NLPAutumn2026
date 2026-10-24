"""Reproducible corpus selection and uniform row sampling; no model inference."""

import hashlib
import importlib.metadata
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.download import HQ_REVISION, RAW_REVISION, ROOT
from src.paths import ARTIFACTS, RAW

SEED = 42
OUTPUT_COLUMNS = ["id", "text", "url", "dump", "quality_score"]
SOURCE_COLUMNS = OUTPUT_COLUMNS + ["language_score"]
SCHEMA = pa.schema([(column, pa.float64() if column == "quality_score" else pa.string())
                    for column in OUTPUT_COLUMNS])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def text_hash(text):
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).digest() if normalized else None


def files_for(name):
    locations = {
        "raw": RAW / "data/bul_Cyrl/train",
        "hq": ARTIFACTS / "sources/hq/bul_Cyrl",
        "official_test": ARTIFACTS / "sources/raw_test/data/bul_Cyrl/test",
        "top": ARTIFACTS / "corpus/train",
    }
    files = sorted(locations[name].glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for {name}: {locations[name]}")
    return files


def batches(files, columns=SOURCE_COLUMNS, batch_size=4096):
    for path in files:
        parquet = pq.ParquetFile(path)
        available = [column for column in columns if column in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=available):
            yield batch.to_pydict()


def output_table(frame):
    return pa.Table.from_pandas(frame.reindex(columns=OUTPUT_COLUMNS), schema=SCHEMA, preserve_index=False)


def prepare_holdout():
    directory = ARTIFACTS / "corpus"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([pd.DataFrame(batch) for batch in batches(files_for("official_test"))], ignore_index=True)
    original_count = len(frame)
    frame["content_hash"] = frame.text.map(text_hash)
    frame = frame.dropna(subset=["content_hash"]).sort_values("id").drop_duplicates("content_hash")
    if len(frame) < 10_000:
        raise ValueError("Fewer than 10,000 unique held-out documents")
    positions = np.random.default_rng(SEED).permutation(len(frame))[:10_000]
    selected = frame.iloc[positions].copy()
    for split, part in [("validation", selected.iloc[:5000]), ("test", selected.iloc[5000:])]:
        pq.write_table(output_table(part), directory / f"{split}.parquet", compression="zstd", row_group_size=1000)
    pq.write_table(pa.Table.from_pandas(selected[["id", "content_hash"]], preserve_index=False), directory / "holdout_index.parquet")
    return {"official_rows": original_count, "unique_nonempty_rows": len(frame), "validation": 5000, "test": 5000}


def index_hq():
    path = ARTIFACTS / "hq_index.parquet"
    schema = pa.schema([
        ("source_row", pa.int64()), ("id", pa.string()), ("content_hash", pa.binary(32)),
        ("quality_score", pa.float64()), ("language_score", pa.float64()),
        ("n_chars", pa.int64()), ("n_words", pa.int64()), ("n_bytes", pa.int64()),
    ])
    offset = 0
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for batch in batches(files_for("hq"), ["id", "text", "quality_score", "language_score"]):
            texts = batch["text"]
            n = len(texts)
            records = {
                "source_row": np.arange(offset, offset + n), "id": batch["id"],
                "content_hash": [text_hash(text) for text in texts],
                "quality_score": batch["quality_score"], "language_score": batch["language_score"],
                "n_chars": [len(text) for text in texts],
                "n_words": [len(text.split()) for text in texts],
                "n_bytes": [len(text.encode("utf-8")) for text in texts],
            }
            writer.write_table(pa.Table.from_pydict(records, schema=schema))
            offset += n
            if offset // 131072 != (offset - n) // 131072:
                print(f"Indexed {offset:,} HQ documents", flush=True)
    return path


def select_top(index_path, count=1_000_000):
    frame = pq.read_table(index_path).to_pandas()
    counts = {"published_hq": len(frame)}
    counts["empty_removed"] = int(frame.content_hash.isna().sum())
    frame = frame.dropna(subset=["content_hash"])
    if not np.isfinite(frame.quality_score).all():
        raise ValueError("Non-finite quality scores")
    frame = frame.sort_values(["quality_score", "id"], ascending=[False, True], kind="stable")
    before = len(frame)
    frame = frame.drop_duplicates("content_hash")
    counts["exact_duplicates_removed"] = before - len(frame)
    counts["repeated_ids_observed"] = int(frame.id.duplicated().sum())
    heldout = pq.read_table(ARTIFACTS / "corpus/holdout_index.parquet").to_pandas()
    overlaps = frame.id.isin(heldout.id) | frame.content_hash.isin(heldout.content_hash)
    counts["holdout_overlaps_removed"] = int(overlaps.sum())
    frame = frame.loc[~overlaps]
    counts["eligible"] = len(frame)
    if len(frame) < count:
        raise ValueError(f"Only {len(frame)} eligible documents")
    selected = frame.iloc[:count].copy()
    selected["rank"] = np.arange(1, count + 1)
    selected["output_position"] = np.random.default_rng(SEED).permutation(count)
    pq.write_table(pa.Table.from_pandas(selected, preserve_index=False), ARTIFACTS / "selection.parquet", compression="zstd")
    return selected, counts


def export_train(selected):
    """Partition the selected rows on disk, then sort one 100k-row shard at a time."""
    stage = ARTIFACTS / "staging"
    stage.mkdir(exist_ok=True)
    directory = ARTIFACTS / "corpus/train"
    directory.mkdir(parents=True, exist_ok=True)
    stage_schema = SCHEMA.append(pa.field("output_position", pa.int64()))
    writers = [pq.ParquetWriter(stage / f"{i:02}.parquet", stage_schema, compression="zstd") for i in range(10)]
    positions = dict(zip(selected.source_row, selected.output_position))
    offset = 0
    try:
        for batch in batches(files_for("hq"), OUTPUT_COLUMNS):
            n = len(batch["id"])
            take = [i for i in range(n) if offset + i in positions]
            if take:
                frame = pd.DataFrame(batch).iloc[take].copy()
                frame["output_position"] = [positions[offset + i] for i in take]
                for shard, part in frame.groupby(frame.output_position // 100_000):
                    table = pa.Table.from_pandas(part, schema=stage_schema, preserve_index=False)
                    writers[int(shard)].write_table(table)
            offset += n
    finally:
        for writer in writers:
            writer.close()
    for shard in range(10):
        path = stage / f"{shard:02}.parquet"
        table = pq.read_table(path).sort_by("output_position").drop(["output_position"])
        if table.num_rows != 100_000:
            raise ValueError(f"Unexpected shard length: {shard}: {table.num_rows}")
        pq.write_table(table, directory / f"{shard:02}.parquet", compression="zstd", row_group_size=1000)
        path.unlink()
        print(f"Exported train shard {shard + 1}/10", flush=True)
    stage.rmdir()


def uniform_sample(name, count=10_000, seed=SEED):
    """Sample global row indices, reading each touched row group only once."""
    files = files_for(name)
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    indices = np.sort(np.random.default_rng(seed).choice(total, count, replace=False))
    records = []
    offset = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = [column for column in SOURCE_COLUMNS if column in parquet.schema_arrow.names]
        for group in range(parquet.num_row_groups):
            n = parquet.metadata.row_group(group).num_rows
            lo, hi = np.searchsorted(indices, [offset, offset + n])
            if hi > lo:
                table = parquet.read_row_group(group, columns=columns)
                table = table.take(pa.array(indices[lo:hi] - offset))
                rows = table.to_pylist()
                for row, index in zip(rows, indices[lo:hi]):
                    row["source_row"] = int(index)
                    records.append(row)
            offset += n
    frame = pd.DataFrame(records)
    for column in SOURCE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan if column.endswith("score") else None
    frame["content_hash"] = frame.text.map(text_hash)
    if name == "top":
        metadata = pq.read_table(ARTIFACTS / "selection.parquet", columns=["id", "language_score"]).to_pandas()
        frame = frame.drop(columns="language_score").merge(metadata, on="id", validate="one_to_one")
    path = ARTIFACTS / "samples" / f"{name}.parquet"
    path.parent.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
    print(f"Sampled {len(frame):,}/{total:,} {name} documents", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(selected, counts, holdout):
    files = sorted((ARTIFACTS / "corpus").glob("train/*.parquet"))
    files += [ARTIFACTS / f"corpus/{split}.parquet" for split in ["validation", "test"]]
    totals = {key: int(selected[column].sum()) for key, column in
              [("characters", "n_chars"), ("whitespace_words", "n_words"), ("utf8_bytes", "n_bytes")]}
    manifest = {
        "seed": SEED, "raw_revision": RAW_REVISION, "hq_revision": HQ_REVISION,
        "sources": {"raw": "HuggingFaceFW/fineweb-2", "hq": "epfml/FineWeb2-HQ-Plus", "config": "bul_Cyrl"},
        "source_rows": {name: sum(pq.ParquetFile(path).metadata.num_rows for path in files_for(name))
                        for name in ["hq", "official_test"]},
        "normalization_for_hash_only": "NFC, CRLF/CR to LF, strip; original text preserved",
        "selection": "quality_score descending, id ascending; empty/duplicate/heldout removal before top 1000000",
        "train_shuffle": "numpy.default_rng(seed).permutation assigns output positions to score-ranked rows",
        "holdout_selection": "Drop empty/hash duplicates, sort id, seeded permutation: first 5000 validation, next 5000 test",
        "counts": counts, "holdout": holdout, "train_documents": len(selected),
        "quality_threshold": float(selected.quality_score.min()),
        "last_selected_id": selected.iloc[-1].id,
        "train_totals": totals, "tokens": None,
        "software": {package: importlib.metadata.version(package) for package in ["pyarrow", "numpy", "pandas", "huggingface_hub"]},
        "files": [{"path": str(path.relative_to(ARTIFACTS)), "rows": pq.ParquetFile(path).metadata.num_rows,
                   "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }
    raw_files = sorted((RAW / "data/bul_Cyrl/train").glob("*.parquet"))
    if raw_files:
        manifest["source_rows"]["raw"] = sum(pq.ParquetFile(p).metadata.num_rows for p in raw_files)
    write_json(ARTIFACTS / "corpus/manifest.json", manifest)
    return manifest


def prepare():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest_path = ARTIFACTS / "corpus/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["raw_revision"], manifest["hq_revision"]) != (RAW_REVISION, HQ_REVISION):
            raise ValueError("Corpus source revisions changed; choose a new artifact directory")
        for item in manifest["files"]:
            if sha256_file(ARTIFACTS / item["path"]) != item["sha256"]:
                raise ValueError(f"Changed corpus file: {item['path']}")
        print("Existing corpus checksums verified", flush=True)
        return
    for name, expected in [("hq", 2_561_554), ("official_test", 36_933)]:
        actual = sum(pq.ParquetFile(path).metadata.num_rows for path in files_for(name))
        if actual != expected:
            raise ValueError(f"Incomplete or unexpected source {name}: {actual} != {expected}")
    holdout = prepare_holdout()
    index_path = index_hq()
    selected, counts = select_top(index_path)
    export_train(selected)
    make_manifest(selected, counts, holdout)


def verify():
    """Validate the actual delivered texts, not only the selection metadata."""
    manifest = json.loads((ARTIFACTS / "corpus/manifest.json").read_text())
    selected = pq.read_table(ARTIFACTS / "selection.parquet").to_pandas().sort_values("output_position")
    expected_ids = selected.id.to_numpy()
    expected_hashes = selected.content_hash.to_numpy()
    expected_scores = selected.quality_score.to_numpy()
    offset = 0
    totals = {"characters": 0, "whitespace_words": 0, "utf8_bytes": 0}
    for batch in batches(files_for("top"), OUTPUT_COLUMNS):
        n = len(batch["id"])
        hashes = [text_hash(text) for text in batch["text"]]
        if not np.array_equal(batch["id"], expected_ids[offset:offset + n]):
            raise ValueError("Exported order/IDs differ from shuffled selection")
        if hashes != list(expected_hashes[offset:offset + n]):
            raise ValueError("Exported text hashes differ from source selection")
        if not np.array_equal(batch["quality_score"], expected_scores[offset:offset + n]):
            raise ValueError("Exported scores differ from source selection")
        for text in batch["text"]:
            totals["characters"] += len(text)
            totals["whitespace_words"] += len(text.split())
            totals["utf8_bytes"] += len(text.encode("utf-8"))
        offset += n
    if offset != 1_000_000 or selected.id.nunique() != offset or selected.content_hash.nunique() != offset:
        raise ValueError("Train count or uniqueness failed")
    if totals != manifest["train_totals"]:
        raise ValueError("Text totals differ from manifest")
    id_sets, hash_sets = {"train": set(selected.id)}, {"train": set(selected.content_hash)}
    split_totals = {"train": {"documents": offset, **totals}}
    for split in ["validation", "test"]:
        frame = pq.read_table(ARTIFACTS / f"corpus/{split}.parquet").to_pandas()
        hashes = frame.text.map(text_hash)
        if len(frame) != 5000 or hashes.isna().any() or hashes.nunique() != 5000 or frame.id.nunique() != 5000:
            raise ValueError(f"Invalid held-out split: {split}")
        if frame.quality_score.notna().any():
            raise ValueError("Held-out scores must be null")
        id_sets[split], hash_sets[split] = set(frame.id), set(hashes)
        split_totals[split] = {"documents": len(frame), "characters": int(frame.text.map(len).sum()),
                               "whitespace_words": int(frame.text.map(lambda text: len(text.split())).sum()),
                               "utf8_bytes": int(frame.text.map(lambda text: len(text.encode('utf-8'))).sum())}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if id_sets[left] & id_sets[right] or hash_sets[left] & hash_sets[right]:
            raise ValueError(f"Split overlap: {left}, {right}")
    for item in manifest["files"]:
        path = ARTIFACTS / item["path"]
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False):
            raise ValueError(f"Unexpected schema: {path}")
        if parquet.metadata.num_rows != item["rows"] or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Manifest mismatch: {path}")
    # Reconstruct the ranking independently from the stored full HQ index.
    index = pq.read_table(ARTIFACTS / "hq_index.parquet", columns=["id", "content_hash", "quality_score"]).to_pandas()
    ranked = index.dropna(subset=["content_hash"]).sort_values(["quality_score", "id"], ascending=[False, True], kind="stable")
    ranked = ranked.drop_duplicates("content_hash")
    heldout_ids = id_sets["validation"] | id_sets["test"]
    heldout_hashes = hash_sets["validation"] | hash_sets["test"]
    ranked = ranked.loc[~ranked.id.isin(heldout_ids) & ~ranked.content_hash.isin(heldout_hashes)]
    if list(ranked.id.iloc[:1_000_000]) != list(selected.sort_values("rank").id):
        raise ValueError("Delivered selection is not the top million eligible documents")
    report = {"passed": True, "splits": split_totals, "exact_overlap": 0,
              "source_hashes_match": True, "ranking_verified": True, "file_checksums_verified": True,
              "near_duplicates_checked": False}
    write_json(ARTIFACTS / "corpus/verification.json", report)
    return report


if __name__ == "__main__":
    prepare()
