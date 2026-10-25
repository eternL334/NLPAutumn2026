"""Sequential, resumable inference with a separate bounded GPU budget."""

import argparse
import json
import os
import re
import subprocess
import sys
import time

from src.paths import PACKAGE, TASK
from src.training.data import read_json, sha256, write_json
from src.evaluation.benchmark import ROOT, GENERATION, item_seed, read_jsonl, verify_frozen, write_jsonl


def training_processes():
    import psutil
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info["cmdline"] or []
        if any(cmd[i:i + 3] == ["-m", "src", "train"] for i in range(len(cmd) - 2)):
            found.append(p.pid)
    return found


def generate(model_name="all"):
    manifest = verify_frozen()
    for name, expected in manifest["generation_code"].items():
        if sha256(PACKAGE / "evaluation" / name) != expected:
            raise ValueError(f"Frozen generation implementation changed: {name}")
    if training_processes():
        raise RuntimeError("Training is active; inference deferred without touching the training process")
    ledger_path = ROOT / "gpu_budget.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {
        "limit_seconds": 1800, "used_seconds": 0.0, "segments": []}
    for model in (["monolm", "qwen"] if model_name == "all" else [model_name]):
        path = ROOT / "generations" / f"{model}.jsonl"
        if path.exists() and len(read_jsonl(path)) == 100:
            print(f"{model}: 100 cached generations", flush=True)
            continue
        remaining = ledger["limit_seconds"] - ledger["used_seconds"]
        if remaining < 10:
            print("GPU budget exhausted; partial generations preserved", flush=True)
            return
        if training_processes():
            raise RuntimeError("Training started; inference deferred")
        start = time.monotonic()
        log_path = ROOT / "generations" / f"{model}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        status = "interrupted"
        try:
            with log_path.open("a") as log:
                process = subprocess.run([sys.executable, "-m", "src.evaluation.benchmark_generation",
                                          "--worker", model, "--seconds", str(remaining - 2)],
                                         cwd=TASK, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=remaining, check=False)
            status = "ok" if process.returncode == 0 else "failed"
            if process.returncode:
                raise RuntimeError(f"Inference failed; see {log_path}")
        except subprocess.TimeoutExpired:
            status = "time_budget"
            print(f"Time limit reached; cached outputs preserved. See {log_path}", flush=True)
            return
        finally:
            elapsed = time.monotonic() - start
            ledger["used_seconds"] += elapsed
            ledger["segments"].append({"model": model, "elapsed_seconds": elapsed, "status": status})
            write_json(ledger_path, ledger)
        count = len(read_jsonl(path)) if path.exists() else 0
        print(f"{model}: {count}/100 generations; GPU budget {ledger['used_seconds']:.1f}/1800 seconds", flush=True)
        if count < 100:
            return


def worker(name, seconds):
    start = time.monotonic()
    deadline = start + seconds
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model_path = ROOT / "models" / name
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model.generation_config.do_sample = False
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    suppressed = [i for i in tokenizer.all_special_ids if i not in eos_ids]
    path = ROOT / "generations" / f"{name}.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    completed = {r["id"] for r in rows}
    assert len(completed) == len(rows)
    manifest_hash = sha256(ROOT / "manifest.json")
    assert all(r["manifest_sha256"] == manifest_hash and r["model"] == name for r in rows)

    class Stop(StoppingCriteria):
        def __init__(self, prefix, task):
            self.prefix, self.task, self.reason = prefix, task, None

        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() >= deadline:
                self.reason = "time_budget"
                return True
            if self.task == "continuation":
                text = tokenizer.decode(input_ids[0, self.prefix:], skip_special_tokens=True)
                if len(re.findall(r"\S+", text)) > 80:
                    self.reason = "word_limit"
                    return True
            return False

    for item in read_jsonl(ROOT / "items.jsonl"):
        if item["id"] in completed:
            continue
        if time.monotonic() >= deadline - 1:
            break
        seed = item_seed(item["id"])
        torch.manual_seed(seed)
        inputs = tokenizer(item["prompt"], add_special_tokens=False, return_tensors="pt").to("cuda")
        prefix = inputs.input_ids.shape[1]
        options = GENERATION[item["task"]]
        assert prefix + options["max_new_tokens"] <= 2048
        stop = Stop(prefix, item["task"])
        kwargs = {"do_sample": item["task"] == "continuation", "max_new_tokens": options["max_new_tokens"],
                  "eos_token_id": eos, "pad_token_id": pad, "suppress_tokens": suppressed,
                  "num_beams": 1, "repetition_penalty": 1.0, "use_cache": True,
                  "stopping_criteria": StoppingCriteriaList([stop])}
        if item["task"] == "continuation":
            kwargs.update(temperature=.8, top_p=.95, top_k=0)
        began = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        ids = output[0, prefix:].tolist()
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        if item["task"] == "continuation":
            words = list(re.finditer(r"\S+", raw))
            response = raw[:words[79].end()] if len(words) > 80 else raw
        else:
            response = next((line.strip() for line in raw.splitlines() if line.strip()), "")
        row = {"id": item["id"], "family": item["family"], "task": item["task"], "model": name,
               "seed": seed, "raw_output": raw, "response": response,
               "input_tokens": prefix, "output_tokens": len(ids), "words": len(response.split()),
               "characters": len(response), "seconds": time.monotonic() - began,
               "stop_reason": stop.reason or ("eos" if ids and ids[-1] in eos_ids else "token_limit"),
               "manifest_sha256": manifest_hash}
        if stop.reason == "time_budget":
            write_json(ROOT / "generations" / f"{name}_incomplete.json", row)
            break
        rows.append(row)
        write_jsonl(path, rows)
        print(json.dumps({k: row[k] for k in ["id", "output_tokens", "words", "seconds", "stop_reason"]}), flush=True)
    write_json(ROOT / "generations" / f"{name}_runtime.json", {
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "attention": "sdpa",
        "dtype": "bfloat16", "count": len(rows), "elapsed_seconds": time.monotonic() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["monolm", "qwen"], required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    worker(args.worker, args.seconds)
