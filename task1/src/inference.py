"""Generate from an exported local model; never download pretrained weights."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    attention = "sdpa" if args.device == "cpu" else (
        "flash_attention_4" if torch.cuda.get_device_capability()[0] >= 10 else "flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, attn_implementation=attention,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32).to(args.device).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    if inputs.input_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        parser.error("Prompt plus generated tokens exceeds the model context")
    kwargs = {"do_sample": args.temperature > 0}
    if args.temperature > 0:
        kwargs.update(temperature=args.temperature, top_p=.95)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, suppress_tokens=[0, 1, 2],
                                use_cache=True, **kwargs)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
