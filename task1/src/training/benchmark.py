"""Calibrate in fresh processes so failed CUDA allocations cannot poison later checks."""

import argparse
import copy
import math
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.data import ARTIFACTS, ROOT, TOKENIZER, TokenBlocks, read_json, verify_data, write_json
from src.training.state import export_model, load_checkpoint, save_checkpoint, versions


def _worker(mode, config, directory):
    from src.training.train import evaluate, setup, update

    if mode == "inference":
        model = AutoModelForCausalLM.from_pretrained(
            directory / "export", local_files_only=True, dtype=torch.float32,
            attn_implementation=config["runtime"]["attention"]).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        x = tokenizer("България е страна", add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = model.generate(**x, max_new_tokens=8, do_sample=False, use_cache=True)
        assert y.shape[1] > x.input_ids.shape[1]
        assert model.lm_head.weight is model.model.embed_tokens.weight
        return {"full_model_export_loaded": True, "tied_weights": True,
                "generated_tokens": int(y.shape[1] - x.input_ids.shape[1]),
                "flash_attention_kv_cache": True}

    accelerator, model, optimizer, scheduler, progress = setup(config)
    length = config["training"]["sequence_length"]
    blocks = TokenBlocks("train", length)
    count = config["training"]["batch_tokens"] // length
    indices = np.arange(count)
    if mode == "measure":
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        update(accelerator, model, optimizer, scheduler, blocks, indices, config)
        torch.cuda.synchronize()
        warmup = time.monotonic() - start
        durations = []
        for _ in range(2):
            start = time.monotonic()
            metrics = update(accelerator, model, optimizer, scheduler, blocks, indices, config)
            torch.cuda.synchronize()
            durations.append(time.monotonic() - start)
        seconds = float(np.median(durations))
        return {"warmup_seconds": warmup, "step_seconds": seconds,
                "tokens_per_second": metrics["tokens"] / seconds, "loss": metrics["loss"],
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "free_gib": torch.cuda.mem_get_info()[0] / 2**30,
                "calibration": {"device": torch.cuda.get_device_name(),
                                "capability": list(torch.cuda.get_device_capability()),
                                "total_memory": torch.cuda.get_device_properties(0).total_memory,
                                "torch_cuda": torch.version.cuda, "versions": versions()}}

    validation = TokenBlocks("validation", length)
    signature = {"config": config, "versions": versions(), "purpose": "isolated GPU checkpoint check"}
    if mode == "checkpoint":
        metrics = update(accelerator, model, optimizer, scheduler, blocks, indices, config)
        progress.step, progress.cursor, progress.tokens = 1, count, metrics["tokens"]
        scores = evaluate(accelerator, model, validation, np.arange(4), config)
        checkpoint = save_checkpoint(accelerator, directory, progress, signature)
        return {"checkpoint": str(checkpoint), "validation_loss": scores["loss"]}

    saved = read_json(directory / "checkpoint.json")
    load_checkpoint(accelerator, saved["checkpoint"], signature)
    assert (progress.step, progress.cursor) == (1, count)
    scores = evaluate(accelerator, model, validation, np.arange(4), config)
    assert abs(scores["loss"] - saved["validation_loss"]) < 1e-5
    metrics = update(accelerator, model, optimizer, scheduler, blocks,
                     np.arange(progress.cursor, progress.cursor + count), config)
    export_model(accelerator, model, directory / "export")
    return {"checkpoint_loss_before": saved["validation_loss"],
            "checkpoint_loss_after": scores["loss"], "resumed_loss": metrics["loss"],
            "resumed_tokens": metrics["tokens"], "attention": config["runtime"]["attention"],
            "device": torch.cuda.get_device_name(), "bf16": True, "separate_processes": True}


def _launch(mode, config_path, directory, deadline):
    """All CUDA work lives in the child; a timeout or driver error ends that process."""
    directory.mkdir(parents=True, exist_ok=True)
    result_path, log_path = directory / f"{mode}.json", directory / f"{mode}.log"
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("GPU check budget exhausted; no further process was started")
    print(f"Calibration: {mode} ({directory.name}); log: {log_path}", flush=True)
    with log_path.open("w") as log:
        try:
            process = subprocess.run(
                [sys.executable, "-m", "src.training.benchmark", mode,
                 "--config", str(config_path), "--directory", str(directory)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=False)
        except subprocess.TimeoutExpired as exc:
            write_json(result_path, {"status": "timeout", "log": str(log_path)})
            raise RuntimeError(f"GPU check reached its time limit; see {log_path}") from exc
    result = read_json(result_path) if result_path.exists() else {
        "status": "error", "error": f"Worker exited with code {process.returncode}"}
    if process.returncode != 0 and result["status"] == "ok":
        result.update(status="error", error=f"Worker exited with code {process.returncode}")
    result["log"] = str(log_path)
    return result


def _require_success(result):
    if result["status"] != "ok":
        raise RuntimeError(f"Calibration failed: {result.get('error', result['status'])}. "
                           f"See {result['log']}. No recommended configuration was published.")


def benchmark(config):
    from src.training.checks import gpu_budget

    verification = verify_data(config["training"]["sequence_length"])
    write_json(ARTIFACTS / "data_verification.json", verification)
    directory = ARTIFACTS / "calibrations" / uuid.uuid4().hex[:12]
    write_json(directory / "source_config.json", config)
    options = config.get("benchmark", {})
    records = []
    with gpu_budget() as remaining:
        deadline = time.monotonic() + remaining
        for checkpointing in options.get("activation_checkpointing", [False, True]):
            for microbatch in options.get("microbatches", [1, 2, 4]):
                # Leave time to verify the selected checkpoint, resume and export.
                if time.monotonic() >= deadline - 160:
                    break
                candidate = copy.deepcopy(config)
                candidate["training"]["planned_steps"] = 100
                runtime = {"microbatch": microbatch, "activation_checkpointing": checkpointing, "compile": False}
                candidate["runtime"].update(runtime)
                case = directory / f"batch-{microbatch}-checkpoint-{int(checkpointing)}"
                write_json(case / "config.json", candidate)
                result = _launch("measure", case / "config.json", case, deadline)
                record = {**runtime, **result}
                records.append(record)
                write_json(directory / "benchmark.json", records)
                print({k: v for k, v in record.items() if k != "calibration"}, flush=True)
                if result["status"] != "out_of_memory":
                    _require_success(result)
        margin = options.get("min_free_gib", 1)
        eligible = [r for r in records if r["status"] == "ok" and r["free_gib"] >= margin]
        if not eligible:
            raise RuntimeError(f"No measured configuration leaves {margin} GiB free within the GPU check budget; "
                               f"see {directory}")
        selected = max(eligible, key=lambda r: r["tokens_per_second"])
        selected_runtime = {k: selected[k] for k in ["microbatch", "activation_checkpointing", "compile"]}
        check_config = copy.deepcopy(config)
        check_config["runtime"].update(selected_runtime)
        check_config["training"]["planned_steps"] = 100
        checks = directory / "checks"
        write_json(checks / "config.json", check_config)
        results = {}
        for mode in ["checkpoint", "resume", "inference"]:
            results[mode] = _launch(mode, checks / "config.json", checks, deadline)
            _require_success(results[mode])
        selected["resume_valid"] = True

    final = copy.deepcopy(config)
    final["runtime"].update(selected_runtime)
    settings = final["training"]
    length = settings["sequence_length"]
    epoch_steps = math.ceil(len(TokenBlocks("train", length)) / (settings["batch_tokens"] // length))
    budget_steps = math.floor(settings["budget_fraction"] * settings["wall_seconds"] / selected["step_seconds"])
    settings["planned_steps"] = min(epoch_steps, max(1, budget_steps))
    final["calibration"] = {**selected["calibration"], "artifacts": str(directory)}
    write_json(directory / "benchmark.json", records)
    write_json(ARTIFACTS / "benchmark.json", records)
    write_json(ARTIFACTS / "checks/gpu.json", results["resume"])
    write_json(ARTIFACTS / "checks/inference.json", results["inference"])
    write_json(ARTIFACTS / "budget.json", {"selected": selected, "epoch_steps": epoch_steps,
               "planned_steps": settings["planned_steps"],
               "estimated_training_hours": settings["planned_steps"] * selected["step_seconds"] / 3600,
               "estimated_epoch_hours": epoch_steps * selected["step_seconds"] / 3600,
               "compile": "Not measured; eager Flash Attention selected within bounded GPU checks"})
    write_json(ARTIFACTS / "recommended_config.json", final)
    print(f"Calibration complete: batch={selected['microbatch']}, checkpointing={selected['activation_checkpointing']}, "
          f"{selected['tokens_per_second']:.0f} tokens/s, {settings['planned_steps']} planned steps", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["measure", "checkpoint", "resume", "inference"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = {"status": "ok", **_worker(args.mode, read_json(args.config), args.directory)}
    except Exception as exc:
        result = {"status": "out_of_memory" if isinstance(exc, torch.OutOfMemoryError) else "error",
                  "error": str(exc)}
        traceback.print_exc()
    write_json(args.directory / f"{args.mode}.json", result)
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
