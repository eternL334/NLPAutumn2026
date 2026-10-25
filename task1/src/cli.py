"""Small command dispatcher; each stage owns its arguments and implementation."""

import argparse
import importlib
import sys

COMMANDS = {
    "download": "Download pinned Bulgarian HQ/test sources; --include-raw adds the comparison corpus",
    "corpus": "Build or verify the top-million corpus",
    "tokenizer": "selected: fixed BPE-32k recipe; run: reproduce all eight experiments",
    "tokenize": "Encode the full train/validation corpus",
    "check": "Bounded functional checks on a small CPU model",
    "benchmark": "Calibrate batch size and training duration on this GPU",
    "train": "Start or resume training explicitly",
    "export": "Export a checkpoint for Transformers inference",
    "infer": "Generate a continuation from a local exported model",
    "doctor": "Inspect the environment; --cuda also checks Flash Attention backward",
    "stats": "Prepare three uniform samples and compute inexpensive corpus statistics",
    "audit": "Prepare blind packets or aggregate externally supplied assessments",
    "evaluate": "Prepare, generate, judge packets and summarize the Bulgarian benchmark",
}


def main():
    parser = argparse.ArgumentParser(description="Bulgarian MonoLM pipeline")
    parser.add_argument("command", choices=COMMANDS, help="; ".join(f"{k}: {v}" for k, v in COMMANDS.items()))
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    command, arguments = sys.argv[1], sys.argv[2:]
    if command not in COMMANDS:
        parser.error(f"Unknown command: {command}")
    if command in {"download", "tokenizer", "infer", "doctor", "evaluate"}:
        module = {"download": "data.download", "tokenizer": "tokenization.experiments",
                  "infer": "inference", "doctor": "environment", "evaluate": "evaluation.benchmark"}[command]
        sys.argv = [f"monolm {command}", *arguments]
        importlib.import_module(f"src.{module}").main()
    elif command in {"tokenize", "check", "benchmark", "train", "export"}:
        sys.argv = ["src", "prepare" if command == "tokenize" else command, *arguments]
        importlib.import_module("src.training.train").main()
    elif command == "corpus":
        stage = argparse.ArgumentParser(description=COMMANDS[command])
        stage.add_argument("action", choices=["build", "verify"], nargs="?", default="build")
        action = stage.parse_args(arguments).action
        from src.data.corpus import prepare, verify
        if action == "build":
            prepare()
        print(verify())
    elif command == "stats":
        argparse.ArgumentParser(description=COMMANDS[command]).parse_args(arguments)
        from src.data.corpus import ARTIFACTS, uniform_sample
        from src.data.statistics import calculate, figures
        for name in ["raw", "hq", "top"]:
            if not (ARTIFACTS / "samples" / f"{name}.parquet").exists():
                uniform_sample(name)
        calculate()
        figures()
    else:
        stage = argparse.ArgumentParser(description=COMMANDS[command])
        stage.add_argument("action", choices=["prepare", "summarize"])
        stage.add_argument("--rubric", choices=["strict", "failures"], default="failures")
        args = stage.parse_args(arguments)
        from src.evaluation import audit, failures
        from src.paths import ARTIFACTS
        module = audit if args.rubric == "strict" else failures
        if args.action == "prepare":
            if not (ARTIFACTS / "audit/design.json").exists():
                audit.prepare_packets()
            if args.rubric == "failures" and not (ARTIFACTS / "audit_failures/design.json").exists():
                failures.prepare_packets()
        else:
            module.summarize()
            if args.rubric == "failures":
                failures.plot()
