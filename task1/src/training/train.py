"""Single-GPU pretraining with Accelerate, Flash Attention and local Trackio."""

import argparse
import gc
import math
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer

from src.training.model import create_model, create_optimizer, create_scheduler
from src.training.data import ARTIFACTS, DATA, ROOT, TOKENIZER, TokenBlocks, read_json, sha256, verify_data, write_json
from src.training.state import Progress, Recorder, export_model, load_checkpoint, save_checkpoint, versions
from src.paths import CONFIGS, PACKAGE

PROMPTS = ["България е страна", "София е известна с", "За да приготвим хляб,",
           "Момичето отвори прозореца и", "Машинното обучение е", "Според учените, водата"]


def setup(config, *, cpu=False):
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    set_seed(config["seed"])
    accelerator = Accelerator(mixed_precision="no" if cpu else "bf16", cpu=cpu)
    if accelerator.num_processes != 1:
        raise ValueError("This training pipeline supports exactly one process/GPU")
    if not cpu and (accelerator.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    if not cpu and config.get("calibration", {}).get("device", torch.cuda.get_device_name()) != torch.cuda.get_device_name():
        raise ValueError("This configuration was calibrated on another GPU; run benchmark on this machine")
    runtime = config["runtime"]
    model = create_model(config["model"], attention="sdpa" if cpu else runtime["attention"],
                         checkpointing=runtime["activation_checkpointing"])
    optimizer = create_optimizer(model, config["training"], fused=not cpu)
    scheduler = create_scheduler(optimizer, config["training"], config["training"]["planned_steps"])
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if runtime["compile"]:
        model = torch.compile(model, dynamic=False)
    progress = Progress(config["training"]["planned_steps"])
    accelerator.register_for_checkpointing(progress)
    return accelerator, model, optimizer, scheduler, progress


def update(accelerator, model, optimizer, scheduler, blocks, indices, config):
    """Normalize accumulated SUM losses by the exact number of valid targets.

    Accumulation is explicit; Accelerator.gradient_accumulation_steps stays at 1,
    so backward must not divide these already-normalized losses a second time.
    """
    microbatch = config["runtime"]["microbatch"]
    targets = blocks.target_count(indices)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=accelerator.device)
    model.train()
    for start in range(0, len(indices), microbatch):
        x, y = blocks.batch(indices[start:start + microbatch], accelerator.device)
        output = model(input_ids=x, use_cache=False)
        loss = F.cross_entropy(output.logits.float().reshape(-1, config["model"]["vocab_size"]),
                               y.reshape(-1), ignore_index=-100, reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss; optimizer was not updated")
        accelerator.backward(loss / targets)
        loss_sum += loss.detach()
        del output, loss
    norm = accelerator.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Non-finite gradient norm; optimizer was not updated")
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(loss_sum / targets), "grad_norm": float(norm), "lr": lr, "tokens": targets}


@torch.no_grad()
def evaluate(accelerator, model, blocks, indices, config):
    model.eval()
    total = torch.zeros((), device=accelerator.device)
    count = blocks.target_count(indices)
    microbatch = config["runtime"]["microbatch"]
    for start in range(0, len(indices), microbatch):
        x, y = blocks.batch(indices[start:start + microbatch], accelerator.device)
        logits = model(input_ids=x, use_cache=False).logits
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                 ignore_index=-100, reduction="sum")
    loss = float(total / count)
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    return {"loss": loss, "perplexity": math.exp(min(loss, 80)), "tokens": count}


@torch.no_grad()
def generate_examples(accelerator, model, config):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    rows = []
    devices = [accelerator.device.index or 0] if accelerator.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), accelerator.autocast():
        torch.manual_seed(config["seed"] + 100)
        for prompt in PROMPTS:
            x = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(accelerator.device)
            output = unwrapped.generate(**x, max_new_tokens=128, do_sample=True, temperature=.8,
                                        top_p=.95, suppress_tokens=[0, 1, 2], use_cache=True)
            rows.append({"prompt": prompt, "continuation": tokenizer.decode(
                output[0, x.input_ids.shape[1]:], skip_special_tokens=True)})
    return rows


def run(config, directory, *, resume=None, stop_after_steps=None, full_validation=True, samples=True):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError(f"Run directory is nonempty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    settings = config["training"]
    length = settings["sequence_length"]
    if settings["planned_steps"] < 1 or config["runtime"]["microbatch"] < 1:
        raise ValueError("planned_steps and microbatch must be positive")
    if settings["batch_tokens"] % length:
        raise ValueError("batch_tokens must be divisible by sequence_length")
    if length > config["model"]["max_position_embeddings"]:
        raise ValueError("Sequence length exceeds the configured context")
    verify_data(length)
    if sha256(TOKENIZER / "tokenizer.json") != read_json(DATA / "manifest.json")["tokenizer_sha256"]:
        raise ValueError("Selected tokenizer differs from the prepared data")
    batch_blocks = settings["batch_tokens"] // length
    blocks, validation = TokenBlocks("train", length), TokenBlocks("validation", length)
    order = np.random.default_rng(config["seed"]).permutation(len(blocks))
    eval_indices = np.sort(np.random.default_rng(config["seed"] + 1).choice(
        len(validation), min(settings["eval_blocks"], len(validation)), replace=False))
    signature = {"config": config, "data_sha256": sha256(DATA / "manifest.json"),
                 "allocator": os.environ["PYTORCH_ALLOC_CONF"],
                 "tokenizer": {p.name: sha256(p) for p in TOKENIZER.iterdir() if p.is_file()},
                 "versions": versions(), "code": {name: sha256(PACKAGE / "training" / name) for name in
                                                   ["train.py", "model.py", "data.py", "state.py"]}}
    if resume and read_json(Path(resume) / "metadata.json")["signature"] != signature:
        raise ValueError("Resume configuration, code, dependencies or data differ from the checkpoint")
    write_json(directory / "config.json", config)
    write_json(directory / "signature.json", signature)
    write_json(directory / "validation_blocks.json", eval_indices.tolist())
    accelerator, model, optimizer, scheduler, progress = setup(config)
    if resume:
        load_checkpoint(accelerator, resume, signature)
    base_elapsed = progress.elapsed
    recorder = Recorder(directory, config, resume)
    stopping = []
    previous_handlers = {}
    for sig in [signal.SIGINT, signal.SIGTERM]:
        previous_handlers[sig] = signal.signal(sig, lambda number, frame: stopping.append(number))
    last_save = time.monotonic()
    log_start = last_save
    recent_seconds = 0.0
    window_loss, window_tokens = 0.0, 0
    reason = "planned_steps"
    try:
        initial = evaluate(accelerator, model, validation, eval_indices, config)
        recorder.log(progress.step, {f"validation/{k}": v for k, v in initial.items()})
        if progress.step == 0:
            progress.best_loss = initial["loss"]
            progress.elapsed = base_elapsed + time.monotonic() - start
            save_checkpoint(accelerator, directory, progress, signature, best=True)
            last_save = time.monotonic()
        while progress.step < progress.total_steps and progress.cursor < len(order):
            elapsed = base_elapsed + time.monotonic() - start
            if stopping or elapsed + recent_seconds >= settings["wall_seconds"] - settings["shutdown_reserve_seconds"]:
                reason = "signal" if stopping else "time_budget"
                break
            if stop_after_steps is not None and progress.step >= stop_after_steps:
                reason = "short_check"
                break
            indices = order[progress.cursor:progress.cursor + batch_blocks]
            step_start = time.monotonic()
            metrics = update(accelerator, model, optimizer, scheduler, blocks, indices, config)
            recent_seconds = time.monotonic() - step_start
            progress.step += 1
            progress.cursor += len(indices)
            progress.tokens += metrics["tokens"]
            window_tokens += metrics["tokens"]
            window_loss += metrics["loss"] * metrics["tokens"]
            progress.elapsed = base_elapsed + time.monotonic() - start
            final_step = progress.step == progress.total_steps or progress.cursor == len(order)
            finishing = final_step or progress.step == stop_after_steps or bool(stopping)
            if progress.step % settings["log_steps"] == 0 or final_step or progress.step == stop_after_steps:
                record = {"train/loss": window_loss / window_tokens, "train/lr": metrics["lr"],
                          "train/grad_norm": metrics["grad_norm"], "train/tokens": progress.tokens,
                          "performance/tokens_per_second": window_tokens / (time.monotonic() - log_start),
                          "performance/step_seconds": recent_seconds, "elapsed_seconds": progress.elapsed,
                          "gpu/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                recorder.log(progress.step, record)
                print(f"step={progress.step} tokens={progress.tokens:,} loss={record['train/loss']:.4f} "
                      f"tok/s={record['performance/tokens_per_second']:.0f}", flush=True)
                window_tokens, window_loss = 0, 0.0
                log_start = time.monotonic()
            best = False
            if progress.step % settings["eval_steps"] == 0 and not finishing:
                scores = evaluate(accelerator, model, validation, eval_indices, config)
                recorder.log(progress.step, {f"validation/{k}": v for k, v in scores.items()})
                best = scores["loss"] < progress.best_loss
                progress.best_loss = min(progress.best_loss, scores["loss"])
            if samples and progress.step % settings["generate_steps"] == 0 and not finishing:
                rows = generate_examples(accelerator, model, config)
                write_json(directory / f"samples-{progress.step:06d}.json", rows)
                recorder.trackio.log({"samples": recorder.trackio.Table(
                    columns=["prompt", "continuation"],
                    data=[[r["prompt"], r["continuation"]] for r in rows])}, step=progress.step)
            if not finishing and (best or time.monotonic() - last_save >= settings["checkpoint_seconds"]):
                progress.elapsed = base_elapsed + time.monotonic() - start
                save_checkpoint(accelerator, directory, progress, signature, best=best)
                last_save = time.monotonic()
        scores = evaluate(accelerator, model, validation, eval_indices, config)
        best = scores["loss"] < progress.best_loss or not (directory / "checkpoints/best.json").exists()
        progress.best_loss = min(progress.best_loss, scores["loss"])
        progress.elapsed = base_elapsed + time.monotonic() - start
        checkpoint = save_checkpoint(accelerator, directory, progress, signature, best=best)
        recorder.log(progress.step, {f"validation/{k}": v for k, v in scores.items()})
        if full_validation and reason == "planned_steps":
            scores = evaluate(accelerator, model, validation, np.arange(len(validation)), config)
            recorder.log(progress.step, {f"full_validation/{k}": v for k, v in scores.items()})
        if samples and reason != "signal":
            write_json(directory / f"samples-{progress.step:06d}.json", generate_examples(accelerator, model, config))
        if reason == "planned_steps":
            export_model(accelerator, model, directory / "export")
        summary = {"step": progress.step, "tokens": progress.tokens, "cursor": progress.cursor,
                   "elapsed_seconds": base_elapsed + time.monotonic() - start, "reason": reason,
                   "checkpoint": str(checkpoint), "validation": scores}
        write_json(directory / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(directory / "failure.json", {"step": progress.step, "cursor": progress.cursor,
                                               "error": repr(exc)})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        recorder.close()
        del model, optimizer, scheduler
        accelerator.free_memory()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "benchmark", "train", "export"])
    parser.add_argument("--config", type=Path, default=CONFIGS / "train_l40.json")
    parser.add_argument("--run", type=Path, default=ARTIFACTS / "runs/monolm-346m")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    if args.resume is not None and args.resume.suffix == ".json":
        args.resume = args.resume.parent / read_json(args.resume)["path"]
    if args.command == "prepare":
        from src.training.data import prepare, verify_data
        prepare()
        print(verify_data())
    elif args.command in {"check", "benchmark"}:
        from src.training.checks import check, benchmark
        (check if args.command == "check" else benchmark)(read_json(args.config))
    elif args.command == "export":
        if args.resume is None:
            parser.error("export requires --resume CHECKPOINT")
        config = read_json(args.resume / "metadata.json")["signature"]["config"]
        accelerator, model, _, _, _ = setup(config)
        signature = read_json(args.resume / "metadata.json")["signature"]
        load_checkpoint(accelerator, args.resume, signature)
        export_model(accelerator, model, args.run)
    else:
        config = read_json(args.config)
        if "planned_steps" not in config["training"] or "calibration" not in config:
            parser.error("Run benchmark on this machine first, then use training/recommended_config.json")
        print(run(config, args.run, resume=args.resume, stop_after_steps=args.stop_after_steps))


if __name__ == "__main__":
    main()
