"""Report dependencies and check the selected GPU attention implementation."""

import argparse
import json
import platform

from src.paths import ARTIFACTS, CONFIGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["l40", "5080"], default="l40")
    parser.add_argument("--cuda", action="store_true", help="Run a small Flash Attention forward/backward")
    args = parser.parse_args()
    import torch
    from src.training.state import versions
    report = {"python": platform.python_version(), "versions": versions(),
              "artifacts": str(ARTIFACTS), "profile": args.profile, "torch_cuda": torch.version.cuda}
    if args.cuda:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 is unavailable; check nvidia-smi and the installed PyTorch wheel")
        config = json.loads((CONFIGS / f"train_{args.profile}.json").read_text())
        backend = config["runtime"]["attention"]
        if backend == "flash_attention_2":
            from flash_attn import flash_attn_func
        else:
            from flash_attn.cute import flash_attn_func
        q = torch.randn(1, 128, 15, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 128, 5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        output = flash_attn_func(q, k, v, causal=True)
        output = output[0] if isinstance(output, tuple) else output
        output.float().square().mean().backward()
        if not all(bool(x.grad.isfinite().all()) for x in [q, k, v]):
            raise RuntimeError("Non-finite Flash Attention gradients")
        report.update(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                      memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                      attention=backend, flash_backward_verified=True)
    print(json.dumps(report, indent=2))
