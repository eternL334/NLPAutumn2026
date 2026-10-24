"""Paths shared by the editable package, scripts and notebooks."""

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
TASK = PACKAGE.parent
REPOSITORY = TASK.parent
CONFIGS = TASK / "configs"
ARTIFACTS = Path(os.environ.get("MONOLM_ARTIFACTS", TASK / "artifacts")).expanduser().resolve()
legacy_raw = REPOSITORY / "fineweb2_raw"
default_raw = legacy_raw if legacy_raw.exists() and "MONOLM_ARTIFACTS" not in os.environ else ARTIFACTS / "sources/raw_train"
RAW = Path(os.environ.get("MONOLM_RAW_DATA", default_raw)).expanduser().resolve()
