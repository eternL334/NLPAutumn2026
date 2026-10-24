"""Bounded functional checks and throughput measurements, never a final run."""

import copy
import gc
import math
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.model import create_model, parameter_report
from src.training.data import ARTIFACTS, DATA, ROOT, TOKENIZER, TokenBlocks, read_json, verify_data, write_json
from src.training.state import Recorder, export_model, load_checkpoint, save_checkpoint, versions

CHECKS = ARTIFACTS / "checks"


@contextmanager
def gpu_budget():
    path = ARTIFACTS / "gpu_check_budget.json"
    budget = read_json(path) if path.exists() else {"limit_seconds": 900, "used_seconds": 0.0}
    remaining = budget["limit_seconds"] - budget["used_seconds"]
    if remaining < 30:
        raise RuntimeError("The authorized 15-minute GPU check budget is exhausted")
    start = time.monotonic()
    try:
        yield remaining
    finally:
        budget["used_seconds"] += time.monotonic() - start
        write_json(path, budget)


def check(config):
    from src.training.train import setup, update
    CHECKS.mkdir(parents=True, exist_ok=True)
    with torch.device("meta"):
        full = create_model(config["model"], attention="eager")
    report = parameter_report(full)
    assert sum(row["parameters"] for row in report) == 346096576
    assert full.lm_head.weight is full.model.embed_tokens.weight
    write_json(ARTIFACTS / "parameters.json", report)
    del full

    # Small CPU model checks resume independently of nondeterministic GPU kernels.
    tiny = copy.deepcopy(config)
    tiny["model"].update(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         max_position_embeddings=64)
    tiny["training"].update(sequence_length=32, batch_tokens=128, planned_steps=4)
    tiny["runtime"].update(microbatch=1, activation_checkpointing=False, compile=False)
    accelerator, model, optimizer, scheduler, progress = setup(tiny, cpu=True)
    blocks = TokenBlocks.__new__(TokenBlocks)
    blocks.tokens = np.random.default_rng(19).integers(4, 32768, size=258, dtype=np.uint16)
    blocks.length, blocks.targets = 32, 257
    first = update(accelerator, model, optimizer, scheduler, blocks, np.arange(4), tiny)
    progress.step, progress.cursor, progress.tokens = 1, 4, 128
    signature = {"config": tiny, "purpose": "CPU resume check"}
    checkpoint = save_checkpoint(accelerator, CHECKS, progress, signature)
    second = update(accelerator, model, optimizer, scheduler, blocks, np.arange(4, 9), tiny)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
    expected_lr = scheduler.get_last_lr()
    load_checkpoint(accelerator, checkpoint, signature)
    assert (progress.step, progress.cursor, progress.tokens) == (1, 4, 128)
    resumed = update(accelerator, model, optimizer, scheduler, blocks, np.arange(4, 9), tiny)
    max_error = max(float((v - expected[k]).abs().max()) for k, v in model.state_dict().items())
    assert max_error == 0 and second == resumed
    assert scheduler.get_last_lr() == expected_lr
    assert expected_random[0] == random.random() and expected_random[1] == float(np.random.rand())
    assert torch.equal(expected_random[2], torch.rand(3))
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, expected_optimizer["state"][key][name])
    x, y = blocks.batch([8], "cpu")
    assert int((y != -100).sum()) == 1
    model.eval()
    x, _ = blocks.batch([0], "cpu")
    with torch.no_grad():
        original = model(x).logits
        changed = x.clone()
        changed[:, 16:] = (changed[:, 16:] + 123) % 32768
        assert torch.equal(original[:, :16], model(changed).logits[:, :16])
    export_model(accelerator, model, CHECKS / "tiny_export")
    reloaded = AutoModelForCausalLM.from_pretrained(CHECKS / "tiny_export", local_files_only=True)
    with torch.no_grad():
        assert torch.equal(original, reloaded(x).logits)
        generated = reloaded.generate(x[:, :4], max_new_tokens=4, do_sample=False)
        assert generated.shape[1] > 4
    tokenizer = AutoTokenizer.from_pretrained(CHECKS / "tiny_export", local_files_only=True)
    reference = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    edge = "България 😀 и\u0306\t\n[EOS] [PAD]"
    assert tokenizer.encode(edge) == reference.encode(edge)
    assert tokenizer.decode(tokenizer.encode(edge)) == edge
    recorder = Recorder(CHECKS, {"purpose": "functional check", "versions": versions()})
    recorder.log(0, {"check/loss": first["loss"]})
    recorder.log(1, {"check/resume_max_weight_error": max_error})
    recorder.close()
    del model, optimizer, scheduler, reloaded, expected, expected_optimizer
    accelerator.free_memory()
    del accelerator
    gc.collect()
    result = {"cpu_resume_exact": True, "rng_restored": True, "optimizer_restored": True,
              "scheduler_restored": True, "causal": True, "padding_excluded": True,
              "export_roundtrip": True, "tokenizer_roundtrip": True, "trackio_logged": True,
              "parameters": 346096576, "versions": versions()}
    write_json(CHECKS / "functional.json", result)
    print(result, flush=True)


def benchmark(config):
    from src.training.train import evaluate, setup, update
    verification = verify_data(config["training"]["sequence_length"])
    write_json(ARTIFACTS / "data_verification.json", verification)
    records = []
    original_config = copy.deepcopy(config)
    config = copy.deepcopy(config)
    config["training"]["planned_steps"] = 100
    length = config["training"]["sequence_length"]
    blocks, validation = TokenBlocks("train", length), TokenBlocks("validation", length)
    indices = np.arange(config["training"]["batch_tokens"] // length)
    with gpu_budget() as remaining:
        deadline = time.monotonic() + remaining
        accelerator, model, optimizer, scheduler, progress = setup(config)
        unwrapped = accelerator.unwrap_model(model)
        successful_updates = 0
        for checkpointing in [False, True]:
            if checkpointing:
                unwrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                unwrapped.gradient_checkpointing_disable()
            for microbatch in config.get("benchmark", {}).get("microbatches", [1, 2, 4]):
                if time.monotonic() > deadline - 160:
                    break
                config["runtime"].update(microbatch=microbatch, activation_checkpointing=checkpointing)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                record = {"microbatch": microbatch, "activation_checkpointing": checkpointing, "compile": False}
                try:
                    warmup = time.monotonic()
                    update(accelerator, model, optimizer, scheduler, blocks, indices, config)
                    successful_updates += 1
                    torch.cuda.synchronize()
                    record["warmup_seconds"] = time.monotonic() - warmup
                    durations = []
                    for _ in range(2):
                        start = time.monotonic()
                        metrics = update(accelerator, model, optimizer, scheduler, blocks, indices, config)
                        successful_updates += 1
                        torch.cuda.synchronize()
                        durations.append(time.monotonic() - start)
                    record.update(status="ok", step_seconds=float(np.median(durations)),
                                  tokens_per_second=metrics["tokens"] / float(np.median(durations)),
                                  loss=metrics["loss"], peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                                  free_gib=torch.cuda.mem_get_info()[0] / 2**30)
                except torch.OutOfMemoryError:
                    record["status"] = "out_of_memory"
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                records.append(record)
                write_json(ARTIFACTS / "benchmark.json", records)
                print(record, flush=True)
        margin = config.get("benchmark", {}).get("min_free_gib", 1)
        eligible = [r for r in records if r["status"] == "ok" and r["free_gib"] >= margin
                    and r.get("resume_valid", True)]
        if not eligible:
            raise RuntimeError(f"No measured configuration leaves the required {margin} GiB VRAM margin")
        selected = max(eligible, key=lambda r: r["tokens_per_second"])
        config["runtime"].update({k: selected[k] for k in ["microbatch", "activation_checkpointing", "compile"]})
        if selected["activation_checkpointing"]:
            unwrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            unwrapped.gradient_checkpointing_disable()
        # Check the full-sized state, export, and one resumed optimizer update.
        progress.step, progress.cursor = successful_updates, len(indices)
        progress.tokens = successful_updates * blocks.target_count(indices)
        signature = {"config": config, "purpose": "full-size GPU checkpoint check"}
        CHECKS.mkdir(parents=True, exist_ok=True)
        checkpoint = save_checkpoint(accelerator, CHECKS / "full_model", progress, signature)
        validation_score = evaluate(accelerator, model, validation, np.arange(4), config)
        load_checkpoint(accelerator, checkpoint, signature)
        after_load = evaluate(accelerator, model, validation, np.arange(4), config)
        assert abs(after_load["loss"] - validation_score["loss"]) < 1e-5
        metrics = update(accelerator, model, optimizer, scheduler, blocks, indices, config)
        export_model(accelerator, model, CHECKS / "full_model/export")
        write_json(CHECKS / "gpu.json", {"checkpoint_loss_before": validation_score["loss"],
                                        "checkpoint_loss_after": after_load["loss"],
                                        "resumed_loss": metrics["loss"], "attention": config["runtime"]["attention"],
                                        "device": torch.cuda.get_device_name(), "bf16": True})
        accelerator.free_memory()
        del accelerator, model, optimizer, scheduler, unwrapped
        gc.collect()
        torch.cuda.empty_cache()
        # Validate the exported inference model after releasing optimizer/activation memory.
        exported = AutoModelForCausalLM.from_pretrained(
            CHECKS / "full_model/export", local_files_only=True,
            dtype=torch.float32, attn_implementation=config["runtime"]["attention"]).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        x = tokenizer("България е страна", add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = exported.generate(**x, max_new_tokens=8, do_sample=False, use_cache=True)
        assert y.shape[1] > x.input_ids.shape[1]
        write_json(CHECKS / "inference.json", {"full_model_export_loaded": True,
                   "tied_weights": exported.lm_head.weight is exported.model.embed_tokens.weight,
                   "generated_tokens": int(y.shape[1] - x.input_ids.shape[1]), "flash_attention_kv_cache": True})
        del exported, x, y
        gc.collect()
        torch.cuda.empty_cache()
    # Do not use any benchmark weights for the final run.
    final = original_config
    final["runtime"].update(config["runtime"])
    settings = final["training"]
    epoch_steps = math.ceil(len(blocks) / len(indices))
    budget_steps = math.floor(settings["budget_fraction"] * settings["wall_seconds"] / selected["step_seconds"])
    settings["planned_steps"] = min(epoch_steps, max(1, budget_steps))
    final["calibration"] = {"device": torch.cuda.get_device_name(),
                            "capability": list(torch.cuda.get_device_capability()),
                            "total_memory": torch.cuda.get_device_properties(0).total_memory,
                            "torch_cuda": torch.version.cuda, "versions": versions()}
    write_json(ARTIFACTS / "recommended_config.json", final)
    write_json(ARTIFACTS / "budget.json", {"selected": selected, "epoch_steps": epoch_steps,
                                         "planned_steps": settings["planned_steps"],
                                         "estimated_training_hours": settings["planned_steps"] * selected["step_seconds"] / 3600,
                                         "estimated_epoch_hours": epoch_steps * selected["step_seconds"] / 3600,
                                         "compile": "Not measured; eager Flash Attention selected within bounded GPU checks"})
