#!/usr/bin/env python3
"""
Kernel-level Lean vs Standard (SplitK) decode-attention sweep, both GQA head
configs, full batch x context grid, at the shipped 1xCU persistent grid.

Generalizes benchmark/lean_kernel_qwen_gqa.py (which is Qwen batch=1 only) to
sweep Qwen2.5-7B (28Q/4KV) and Llama-3.1-8B (32Q/8KV) over
batch in {1,2,4,8,16,32} x context in {8K,16K,32K,64K,128K}, reporting per-call
kernel latency, speedup (std / lean), cosine parity vs SplitK, and whether the
eager auto-gate would enable Lean. Writes a CSV for the PR tables.

Grid is whatever SGLANG_FORCE_LEAN_GRID_CU_MULT resolves to (default 1.0 = one
CTA per CU); set it to A/B other grids without a rebuild.

Usage: python3 benchmark/lean_kernel_sweep.py [out.csv]
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

MODELS = [
    ("qwen2.5-7b", 28, 4),
    ("llama3.1-8b", 32, 8),
]
D = D_V = 128
MAX_KV_SPLITS = 8
BATCHES = [1, 2, 4, 8, 16, 32]
CONTEXTS = [8192, 16384, 32768, 65536, 131072]


def bench(fn, warmup=15, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def run(H_Q, H_KV, B, S):
    dev, dt = "cuda", torch.float16
    kvg = H_Q // H_KV
    sm = 1.0 / (D**0.5)
    tot = B * S

    total_programs, _, _ = _lean_decode_launch_params(H_KV, kvg)
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

    o_std = torch.zeros(B, H_Q, D_V, dtype=dt, device=dev)
    std_ms = bench(
        lambda: decode_attention_fwd_grouped(
            q, k, v, o_std, kvi, kvx, attn_logits, attn_lse, nks, MAX_KV_SPLITS, sm, 1.0
        )
    )

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
    gate = lean_decode_seqlen_gate(H_Q, kvg, B, B * S, is_mla=False)
    return std_ms, lean_ms, cos, gate


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "grid_out/kernel_sweep_1xcu.csv"
    mult = float(os.environ.get("SGLANG_FORCE_LEAN_GRID_CU_MULT", "1.0"))
    print(f"\nKernel sweep  GPU={torch.cuda.get_device_name(0)}  grid_mult={mult}")
    print("=" * 78)
    rows = ["model,H_Q,H_KV,batch,context,std_ms,lean_ms,speedup,cos,gate"]
    for name, H_Q, H_KV in MODELS:
        print(f"\n{name} ({H_Q}Q/{H_KV}KV)")
        print(
            f"{'batch':>5} {'ctx':>6} {'std_ms':>9} {'lean_ms':>9} {'speedup':>8} {'gate':>5} {'cos':>7}"
        )
        for B in BATCHES:
            for S in CONTEXTS:
                # b32 x 128K on the 8-KV-head config exceeds the microbench's single
                # contiguous KV tensor (faults the GPU); real serving uses a paged pool.
                if H_KV == 8 and B == 32 and S == 131072:
                    print(
                        f"{B:>5} {S // 1024:>5}K {'skipped (contiguous-KV limit)':>30}"
                    )
                    rows.append(f"{name},{H_Q},{H_KV},{B},{S},,,,skip,")
                    continue
                try:
                    std, lean, cos, gate = run(H_Q, H_KV, B, S)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{B:>5} {S // 1024:>5}K {'OOM':>9}")
                    rows.append(f"{name},{H_Q},{H_KV},{B},{S},,,,OOM,")
                    continue
                sp = std / lean
                print(
                    f"{B:>5} {S // 1024:>5}K {std:>9.3f} {lean:>9.3f} {sp:>7.2f}x {('ON' if gate else 'OFF'):>5} {cos:>7.4f}"
                )
                rows.append(
                    f"{name},{H_Q},{H_KV},{B},{S},{std:.4f},{lean:.4f},{sp:.4f},{cos:.4f},{int(gate)}"
                )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
