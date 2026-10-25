"""Bounded functional checks and throughput measurements, never a final run."""

import copy
import gc
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.model import create_model, parameter_report
from src.training.data import ARTIFACTS, TOKENIZER, TokenBlocks, read_json, write_json
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
    from src.training.benchmark import benchmark as calibrate
    return calibrate(config)
