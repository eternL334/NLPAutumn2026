"""Fetch only the pinned Bulgarian source files needed by the experiment."""

import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download
from src.paths import ARTIFACTS, RAW, TASK as ROOT

RAW_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
HQ_REVISION = "146690ef88e86e6d728a03ea6f6d85037d8cf9fa"


def download(include_raw=False):
    sources = [
        ("HuggingFaceFW/fineweb-2", RAW_REVISION, "data/bul_Cyrl/test/*", ARTIFACTS / "sources/raw_test"),
        ("epfml/FineWeb2-HQ-Plus", HQ_REVISION, "bul_Cyrl/*.parquet", ARTIFACTS / "sources/hq"),
    ]
    if include_raw:
        sources.append(("HuggingFaceFW/fineweb-2", RAW_REVISION, "data/bul_Cyrl/train/*", RAW))
    for repo, revision, pattern, directory in sources:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=pattern,
            local_dir=directory,
            max_workers=4,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-raw", action="store_true", help="Also download raw train for the corpus comparison")
    download(parser.parse_args().include_raw)


if __name__ == "__main__":
    main()
