#!/usr/bin/env python3
"""Check whether SM100 TMEM output fragments are accidentally duplicated."""

from __future__ import annotations

import os

import torch
from sgl_kernel.flash_mla import flash_mla_sparse_fwd

from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import (
    _sparse_mla_q8kv8_prefill_fwd_sm100_trusted,
)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise SystemExit("Safety check failed: use CUDA_VISIBLE_DEVICES=7")
    if torch.cuda.device_count() != 1:
        raise SystemExit("Safety check failed: expose exactly one GPU")

    torch.manual_seed(1234)
    device = torch.device("cuda:0")
    q = (torch.randn(1, 8, 512, device=device) * 0.25).to(torch.float8_e4m3fn)
    kv = (torch.randn(128, 1, 512, device=device) * 0.25).to(torch.float8_e4m3fn)
    indices = torch.arange(128, dtype=torch.int32, device=device).reshape(1, 1, 128)
    scale = torch.ones((), dtype=torch.float32, device=device)
    sink = torch.zeros(32, dtype=torch.float32, device=device)
    lengths = torch.full((1,), 128, dtype=torch.int32, device=device)
    out = torch.empty(1, 8, 512, dtype=torch.bfloat16, device=device)
    max_logits = torch.empty(1, 32, dtype=torch.float32, device=device)
    lse = torch.empty_like(max_logits)

    _sparse_mla_q8kv8_prefill_fwd_sm100_trusted(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=512**-0.5,
        q_scale=scale,
        kv_scale=scale,
        attn_sink=sink,
        topk_length=lengths,
        out=out,
        max_logits=max_logits,
        lse=lse,
        active_heads=8,
    )
    torch.cuda.synchronize()

    q_padded = torch.zeros(1, 64, 512, dtype=torch.bfloat16, device=device)
    q_padded[:, :8].copy_(q.to(torch.bfloat16))
    sink_padded = torch.zeros(64, dtype=torch.float32, device=device)
    golden, _, _ = flash_mla_sparse_fwd(
        q=q_padded,
        kv=kv.to(torch.bfloat16),
        indices=indices,
        sm_scale=512**-0.5,
        d_v=512,
        attn_sink=sink_padded,
        topk_length=lengths,
    )
    torch.cuda.synchronize()

    segments = out.float().reshape(8, 8, 64)
    print("pairwise max_abs_diff for head0 64-column segments:")
    for lhs in range(8):
        values = []
        for rhs in range(8):
            values.append((segments[0, lhs] - segments[0, rhs]).abs().max().item())
        print(" ".join(f"{value:.7f}" for value in values))

    golden_segments = golden[:, :8].float().reshape(8, 8, 64)
    print("actual-to-golden cosine for head0 64-column segments:")
    for lhs in range(8):
        values = []
        for rhs in range(8):
            values.append(
                torch.nn.functional.cosine_similarity(
                    segments[0, lhs], golden_segments[0, rhs], dim=0
                ).item()
            )
        print(" ".join(f"{value:+.5f}" for value in values))


if __name__ == "__main__":
    main()
