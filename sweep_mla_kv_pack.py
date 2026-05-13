"""Sweep (vec_n, block_s, num_warps) for mla_kv_pack_quantize_fp8."""

import sys
import time

import torch
import triton.testing

from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8

QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
NUM_HEADS = 32
DTYPE = torch.bfloat16
DEVICE = "cuda"

# Focused on mid-bs range where there's still headroom to HBM theoretical peak.
BATCH_SIZES = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]

# Limit config space — block_s × num_warps × vec_n combos that make sense:
# CTAs = cdiv(bs, block_s) × num_heads; per-CTA work = block_s × (24 K-vecs + 16 V-vecs)
# at vec_n=8 or (12 + 8) at vec_n=16. Restrict to reasonable thread counts.
CONFIGS = []
for vec_n in (8, 16):
    for block_s in (1, 2, 4, 8, 16, 32, 64):
        for num_warps in (1, 2, 4, 8):
            # Skip configs with way too many threads for the per-CTA work.
            vecs_per_token = (128 + 64 + 128) // vec_n
            if block_s * vecs_per_token < num_warps * 32:
                continue
            CONFIGS.append((vec_n, block_s, num_warps))
print(f"Total configs: {len(CONFIGS)}", flush=True)


def bench_one(bs, vec_n, block_s, num_warps):
    k_nope = torch.randn(
        (bs, NUM_HEADS, QK_NOPE), dtype=DTYPE, device=DEVICE
    )
    k_pe = torch.randn((bs, 1, QK_ROPE), dtype=DTYPE, device=DEVICE)
    v = torch.randn((bs, NUM_HEADS, V_HEAD), dtype=DTYPE, device=DEVICE)
    k_out = torch.empty(
        (bs, NUM_HEADS, QK_NOPE + QK_ROPE),
        dtype=torch.float8_e4m3fn,
        device=DEVICE,
    )
    v_out = torch.empty(
        (bs, NUM_HEADS, V_HEAD), dtype=torch.float8_e4m3fn, device=DEVICE
    )

    def fn():
        mla_kv_pack_quantize_fp8(
            k_nope,
            k_pe,
            v,
            k_out=k_out,
            v_out=v_out,
            vec_n=vec_n,
            block_s=block_s,
            num_warps=num_warps,
        )

    # Warmup + check that this config doesn't blow up.
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    except Exception as e:
        return None

    ms = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5])
    return float(ms) * 1000.0  # us


def main():
    best = {}
    for bs in BATCH_SIZES:
        rows = []
        t0 = time.time()
        for vec_n, block_s, num_warps in CONFIGS:
            us = bench_one(bs, vec_n, block_s, num_warps)
            if us is None:
                continue
            rows.append((us, vec_n, block_s, num_warps))
        rows.sort()
        print(f"\n--- bs={bs} ({time.time()-t0:.1f}s) top 5 ---", flush=True)
        for us, vec_n, block_s, num_warps in rows[:5]:
            print(
                f"  vec_n={vec_n} block_s={block_s} warps={num_warps} -> {us:.3f} us",
                flush=True,
            )
        best[bs] = rows[0]
    print("\n=== Best per bs ===", flush=True)
    for bs, (us, vn, bsv, nw) in best.items():
        print(f"  bs={bs:>5}: vec_n={vn} block_s={bsv} warps={nw} -> {us:.3f} us", flush=True)


if __name__ == "__main__":
    main()
