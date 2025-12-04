# -*- coding: utf-8 -*-
"""
Microbenchmark: fused vs baseline (emulated) for MLA RoPE + FP8 + KV write.
Uses the sgl_kernel.mla_rope_quantize_fp8_fused extension.
"""
import time

import torch
from sgl_kernel import mla_rope_quantize_fp8_fused


def run_one(nnz=1024, Dn=512, Dr=64, iters=200, warmup=20, device="cuda"):
    torch.manual_seed(0)

    q_nope = torch.randn(nnz, Dn, device=device, dtype=torch.float16)
    q_rope = torch.randn(nnz, Dr, device=device, dtype=torch.float16)
    k_nope = torch.randn(nnz, Dn, device=device, dtype=torch.float16)
    k_rope = torch.randn(nnz, Dr, device=device, dtype=torch.float16)

    max_seq = max(2048, nnz)
    t = torch.linspace(0, 1, steps=max_seq, device=device, dtype=torch.float32)[:, None]
    idx = torch.arange(Dr, device=device, dtype=torch.float32)[None, :]
    freqs = 0.1 * (idx + 1.0)
    cos = torch.cos(t * freqs)
    sin = torch.sin(t * freqs)
    cos_sin = torch.cat([cos, sin], dim=1)
    pos_ids = torch.randint(
        low=0, high=max_seq, size=(nnz,), device=device, dtype=torch.long
    )

    slots = nnz + 8
    loc = torch.arange(nnz, device=device, dtype=torch.long)

    q_out = torch.empty(nnz, Dn + Dr, device=device, dtype=torch.uint8)
    k_nope_out = torch.empty(nnz, Dn, device=device, dtype=torch.uint8)
    k_rope_out = torch.empty(nnz, Dr, device=device, dtype=torch.uint8)
    kv_base = torch.zeros(slots, Dn + Dr, device=device, dtype=torch.uint8)

    # baselines
    def baseline():
        mla_rope_quantize_fp8_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin,
            pos_ids,
            False,
            q_out,
            k_nope_out,
            k_rope_out,
            None,
            None,
        )
        kv_base.zero_()
        kv_base[loc, :Dn] = k_nope_out
        kv_base[loc, Dn:] = k_rope_out

    kv_fused = torch.zeros_like(kv_base)

    def fused():
        mla_rope_quantize_fp8_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin,
            pos_ids,
            False,
            q_out,
            None,
            None,
            kv_fused,
            loc,
        )

    # warmup
    for _ in range(warmup):
        baseline()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        baseline()
    torch.cuda.synchronize()
    t1 = time.time()
    baseline_ms = (t1 - t0) * 1000.0 / iters

    for _ in range(warmup):
        fused()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fused()
    torch.cuda.synchronize()
    t1 = time.time()
    fused_ms = (t1 - t0) * 1000.0 / iters

    return baseline_ms, fused_ms, baseline_ms / fused_ms


if __name__ == "__main__":
    print("MLA RoPE + FP8 Quantization + KV Cache Write Fusion Benchmark")
    print("=" * 70)
    print("Config: Dn=512, Dr=64, iters=1000, warmup=100")
    print("=" * 70)

    # Test larger batch sizes and more iterations for stable measurements
    for nnz in [1024, 4096, 8192, 16384, 32768]:
        b, f, s = run_one(nnz=nnz, iters=1000, warmup=100)
        if b > 0:
            speedup_pct = (s - 1.0) * 100
            print(
                f"nnz={nnz:5d} | baseline={b:7.3f} ms | fused={f:7.3f} ms | speedup x{s:4.2f} ({speedup_pct:+5.1f}%)"
            )
