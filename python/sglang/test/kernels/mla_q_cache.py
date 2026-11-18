"""Inputs and the split-chain reference for the fused MLA Q/cache tests."""

import torch


def _inputs(tokens: int, cache_dtype=torch.bfloat16, output_dtype=None):
    torch.manual_seed(23)
    heads = 12
    q_nope = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    q_pe = torch.randn(
        tokens, heads, 64, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_nope = torch.randn(
        tokens, 1, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_pe = torch.randn(tokens, 1, 64, device="cuda", dtype=torch.bfloat16).contiguous()
    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "k_nope": k_nope,
        "k_pe": k_pe,
        "kv_cache": torch.zeros(tokens + 4, 1, 576, device="cuda", dtype=cache_dtype),
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
        # The cos/sin tables have one identity row, so every position must be 0.
        "positions": torch.zeros(tokens, device="cuda", dtype=torch.int64),
        "k_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "q_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "cos_cache": torch.ones(1, 32, device="cuda", dtype=torch.bfloat16),
        "sin_cache": torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16),
        "out": torch.empty(
            tokens,
            heads,
            576,
            device="cuda",
            dtype=cache_dtype if output_dtype is None else output_dtype,
        ),
    }


def _reference(values):
    return (
        torch.cat((values["q_nope"], values["q_pe"]), dim=-1).to(values["out"].dtype),
        torch.cat((values["k_nope"], values["k_pe"]), dim=-1).to(
            values["kv_cache"].dtype
        ),
    )
