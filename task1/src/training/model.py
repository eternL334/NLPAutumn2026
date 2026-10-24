"""MonoLM: a randomly initialized, exportable Qwen3 decoder configuration."""

import math

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def create_model(config, *, attention="flash_attention_4", checkpointing=False):
    model_config = Qwen3Config(
        **config, hidden_act="silu", attention_bias=False, attention_dropout=0.0,
        use_sliding_window=False, use_cache=False, tie_word_embeddings=True,
        rms_norm_eps=1e-5, initializer_range=0.02,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pad_token_id=0, bos_token_id=2, eos_token_id=3,
    )
    model_config._attn_implementation = attention
    model = Qwen3ForCausalLM(model_config)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def create_optimizer(model, config, *, fused=True):
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
         "weight_decay": config["weight_decay"]},
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config["learning_rate"], betas=(.9, .95), eps=1e-8, fused=fused)


def create_scheduler(optimizer, config, total_steps):
    warmup = max(1, math.ceil(total_steps * config["warmup_fraction"]))
    floor = config["min_learning_rate"] / config["learning_rate"]

    def multiplier(step):
        # LambdaLR evaluates step=0 before the first update: use a nonzero first LR.
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup - 1))
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def parameter_report(model):
    rows = []
    for name, module in [("embedding / output", model.model.embed_tokens),
                         *[(f"layer {i + 1}", layer) for i, layer in enumerate(model.model.layers)],
                         ("final norm", model.model.norm)]:
        n = sum(p.numel() for p in module.parameters())
        rows.append({"module": name, "parameters": n, "weights_fp32_bytes": 4 * n,
                     "gradients_fp32_bytes": 4 * n, "adam_moments_fp32_bytes": 8 * n,
                     "weights_bf16_bytes": 2 * n})
    return rows
