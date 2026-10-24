"""Atomic Accelerate checkpoints and local experiment records."""

import importlib.metadata
import os
import shutil
import uuid
from pathlib import Path

from src.training.data import ARTIFACTS, TOKENIZER, read_json, sha256, write_json


def versions():
    result = {}
    for name in ["torch", "transformers", "accelerate", "flash-attn", "flash-attn-4", "nvidia-cutlass-dsl",
                 "trackio", "tokenizers", "numpy", "pyarrow", "triton"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class Progress:
    def __init__(self, total_steps):
        self.step = 0
        self.cursor = 0
        self.tokens = 0
        self.elapsed = 0.0
        self.best_loss = float("inf")
        self.total_steps = total_steps

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        self.__dict__.update(state)


class Recorder:
    def __init__(self, directory, config, resume=None):
        # Explicitly local even if the parent shell has remote Trackio defaults.
        for key in ["TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_BUCKET_ID", "TRACKIO_WEBHOOK_URL"]:
            os.environ.pop(key, None)
        os.environ["TRACKIO_DIR"] = str(ARTIFACTS / "trackio")
        import trackio
        self.trackio = trackio
        self.directory = Path(directory)
        self.attempt = uuid.uuid4().hex[:12]
        self.file = (self.directory / f"metrics-{self.attempt}.jsonl").open("a", buffering=1)
        self.run = trackio.init(project="monolm", name=f"{self.directory.name}-{self.attempt}",
                                group=self.directory.name, config={**config, "resume": str(resume) if resume else None},
                                embed=False, auto_log_gpu=True)

    def log(self, step, metrics):
        import json
        self.file.write(json.dumps({"step": step, **metrics}, ensure_ascii=False, allow_nan=False) + "\n")
        self.trackio.log(metrics, step=step)

    def close(self):
        self.file.close()
        self.trackio.finish()


def save_checkpoint(accelerator, directory, progress, signature, *, best=False):
    root = Path(directory) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{progress.step:06d}-{uuid.uuid4().hex[:6]}"
    temporary = root / (name + ".tmp")
    target = root / name
    accelerator.save_state(str(temporary), safe_serialization=True)
    write_json(temporary / "metadata.json", {"signature": signature, "step": progress.step,
                                             "tokens": progress.tokens, "cursor": progress.cursor})
    # Completed metadata/checksums distinguish a committed checkpoint from an interrupted write.
    for path in temporary.iterdir():
        if path.is_file():
            if not path.stat().st_size:
                raise IOError(f"Empty checkpoint file: {path}")
            with path.open("rb") as source:
                os.fsync(source.fileno())
    checksums = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
    write_json(temporary / "checksums.json", checksums)
    temporary.rename(target)
    write_json(root / "last.json", {"path": name})
    if best:
        write_json(root / "best.json", {"path": name})
    protected = {read_json(root / p)["path"] for p in ["last.json", "best.json"] if (root / p).exists()}
    checkpoints = sorted((p for p in root.glob("step-*") if not p.name.endswith(".tmp")),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    protected.update(p.name for p in checkpoints[:2])
    for old in checkpoints:
        if old.name not in protected:
            shutil.rmtree(old)
    return target


def load_checkpoint(accelerator, checkpoint, signature):
    checkpoint = Path(checkpoint)
    metadata = read_json(checkpoint / "metadata.json")
    if metadata["signature"] != signature:
        raise ValueError("Checkpoint configuration, data or dependency versions differ")
    for name, checksum in read_json(checkpoint / "checksums.json").items():
        if sha256(checkpoint / name) != checksum:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    accelerator.load_state(str(checkpoint))


def export_model(accelerator, model, directory):
    directory = Path(directory)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(directory, safe_serialization=True)
    for source in TOKENIZER.iterdir():
        if source.is_file():
            shutil.copy2(source, directory / source.name)
    write_json(directory / "generation_config.json", {
        "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0,
        "do_sample": True, "temperature": .8, "top_p": .95,
        "max_new_tokens": 128, "suppress_tokens": [0, 1, 2], "use_cache": True,
    })
