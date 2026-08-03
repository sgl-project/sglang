#!/usr/bin/env python3
"""
Kernel-level Lean vs Standard (SplitK) decode-attention benchmark for the
Qwen2.5-7B-Instruct head configuration (GQA: 28 query heads / 4 KV heads, head_dim 128).

Single-GPU (tp=1) shapes, batch=1, context sweep. Reports per-call kernel latency,
speedup, correctness (cosine similarity vs SplitK), and whether the production
auto-gate would enable Lean at that shape.

Usage: python3 benchmark/lean_kernel_qwen_gqa.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from sglang.kernels.ops.attention.decode_attention import (
    _LEAN_BLOCK_M,
    _lean_decode_launch_params,
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    lean_decode_seqlen_gate,
)

# Qwen2.5-7B-Instruct (tp=1)
H_Q, H_KV, D, D_V = 28, 4, 128, 128
KVG = H_Q // H_KV  # 7
MAX_KV_SPLITS = 8


def bench(fn, warmup=15, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def run(B, S):
    dev, dt = "cuda", torch.float16
    sm = 1.0 / (D**0.5)
    tot = B * S

    # Pre-allocate Lean buffers (reused across multiple runs at this shape)
    total_programs, _, _ = _lean_decode_launch_params(H_KV, KVG)
    lean_Mp = torch.empty(
        (total_programs, _LEAN_BLOCK_M), dtype=torch.float32, device=dev
    )
    lean_Lp = torch.empty(
        (total_programs, _LEAN_BLOCK_M), dtype=torch.float32, device=dev
    )
    lean_Op = torch.empty(
        (total_programs, _LEAN_BLOCK_M, D_V), dtype=torch.float32, device=dev
    )
    lean_locks = torch.zeros((total_programs,), dtype=torch.int32, device=dev)

    kvi = torch.arange(0, (B + 1) * S, step=S, device=dev, dtype=torch.int32)
    kvx = torch.arange(0, tot, device=dev, dtype=torch.int32)
    q = torch.randn(B, H_Q, D, dtype=dt, device=dev)
    k = torch.randn(tot, H_KV, D, dtype=dt, device=dev)
    v = torch.randn(tot, H_KV, D_V, dtype=dt, device=dev)
    attn_logits = torch.empty(
        (B, H_Q, MAX_KV_SPLITS, D_V), dtype=torch.float32, device=dev
    )
    attn_lse = torch.empty((B, H_Q, MAX_KV_SPLITS), dtype=torch.float32, device=dev)
    nks = torch.full((B,), MAX_KV_SPLITS, dtype=torch.int32, device=dev)

    # Standard SplitK (grouped) baseline. sm_scale_withk = sm * k_scale (k_scale=1), v_scale=1.
    o_std = torch.zeros(B, H_Q, D_V, dtype=dt, device=dev)
    std_ms = bench(
        lambda: decode_attention_fwd_grouped(
            q, k, v, o_std, kvi, kvx, attn_logits, attn_lse, nks, MAX_KV_SPLITS, sm, 1.0
        )
    )

    # Lean
    attn_logits2 = torch.empty_like(attn_logits)
    attn_lse2 = torch.empty_like(attn_lse)
    o_lean = torch.zeros(B, H_Q, D_V, dtype=dt, device=dev)
    lean_ms = bench(
        lambda: decode_attention_fwd(
            q,
            k,
            v,
            o_lean,
            kvi,
            kvx,
            attn_logits2,
            attn_lse2,
            nks,
            MAX_KV_SPLITS,
            sm,
            1.0,
            1.0,
            enable_lean=True,
            lean_Mp=lean_Mp,
            lean_Lp=lean_Lp,
            lean_Op=lean_Op,
            lean_locks=lean_locks,
        )
    )

    cos = torch.nn.functional.cosine_similarity(
        o_lean.flatten().float(), o_std.flatten().float(), dim=0
    ).item()
    gate = lean_decode_seqlen_gate(H_Q, KVG, B, B * S, is_mla=False)
    return std_ms, lean_ms, cos, gate


def main():
    print(
        f"\nQwen2.5-7B GQA kernel bench  (H_Q={H_Q}, H_KV={H_KV}, kv_group={KVG}, D={D})"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 92)
    print(
        f"{'context':>9} | {'SplitK (ms)':>12} | {'Lean (ms)':>11} | {'speedup':>8} | {'auto-gate':>9} | {'cos':>7}"
    )
    print("-" * 92)
    for S in [2048, 8192, 16384, 32768, 65536, 131072]:
        std, lean, cos, gate = run(1, S)
        label = f"{S//1024}K"
        print(
            f"{label:>9} | {std:>12.3f} | {lean:>11.3f} | {std/lean:>7.2f}x | "
            f"{'ON' if gate else 'OFF':>9} | {cos:>7.4f}"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()
