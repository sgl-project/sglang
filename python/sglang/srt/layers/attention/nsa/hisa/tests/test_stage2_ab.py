"""Stage 2 (ragged_pool_mqa / pool_mqa_attn_return_logits_fp8) A/B:
tilelang vs triton at all K.

Stage 2 input is the [num_pool, D] blocked_k from stage 1 — K only enters
the kernel via num_pool. The tilelang variant (custom_ops.py:760-846) and
triton variant (kernels.py:_ragged_pool_mqa_kernel) both produce the
[seq_len, num_pool] f32 logits with -inf / +inf masks for outside-[ks,ke)
positions and force_maintain at boundary positions.

Both kernels are fed the SAME blocked_k_fp8 + blocked_k_scale (produced by
the new tilelang grouped/vanilla mean_pool, which we just made the default).
This isolates stage 2 perf from stage 1 differences.

Note: tilelang kernel internally chunks num_pool with block_N=256, so it
reads 256 rows per chunk regardless of cu_k_e_max. We bench at num_pool
>= 256 to avoid that OOB-read hazard polluting timing.
"""
from __future__ import annotations

import time
import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    ragged_pool_mqa_triton,
)


DEVICE = torch.device("cuda")
H, D = 64, 128


def make_inputs(seq_q, seq_kv, K):
    """Generate Q, K, weights, cu_seqlen + run stage 1 to get blocked_k."""
    torch.manual_seed(0)
    q = torch.randn(seq_q, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(seq_kv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(seq_q, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(seq_q, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(seq_kv // 4, seq_kv, seq_q, device=DEVICE).to(torch.int32)

    # Stage 1: tilelang for both (since we just made it default).
    if K < 64:
        bk, bks = fp8_native_block_mean_pooling_grouped_interface(k_fp8, k_scale, K)
    else:
        bk, bks = fp8_native_block_mean_pooling_interface(k_fp8, k_scale, K)

    cu_ks_blk = cu_ks // K
    cu_ke_blk = (cu_ke + K - 1) // K
    return q, bk, bks, weights, cu_ks, cu_ke, cu_ks_blk, cu_ke_blk


def bench_one(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def _verdict(diff_max_rel: float) -> str:
    return "OK" if diff_max_rel <= 1.0 / 16 else "FAIL"


def correctness(seq_q, seq_kv, K):
    q, bk, bks, w, cu_ks, cu_ke, cu_ks_blk, cu_ke_blk = make_inputs(seq_q, seq_kv, K)

    triton_logits = ragged_pool_mqa_triton(
        q_fp8=q, blocked_k_fp8=bk, blocked_k_scale=bks,
        weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        k_block_size=K,
    )
    tilelang_logits = pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q, blocked_kv_fp8=bk, blocked_kv_scale=bks,
        kv_block_size=K, weights_f32=w,
        cu_seqlen_blocked_ks=cu_ks_blk, cu_seqlen_blocked_ke=cu_ke_blk,
    )

    # Both produce [seq_q, num_pool] f32 with -inf/+inf masks. Compare:
    # 1) inf masks must match exactly
    # 2) finite logits diff at fp8-strict tolerance
    inf_t = torch.isinf(triton_logits)
    inf_l = torch.isinf(tilelang_logits)
    inf_match = (inf_t == inf_l).all().item()

    finite_mask = torch.isfinite(triton_logits) & torch.isfinite(tilelang_logits)
    # NaN-safe diff: only compute differences where both finite. Plain
    # ``(t - l).abs() * mask`` propagates ``nan = inf - inf`` through.
    if finite_mask.any():
        t_finite = triton_logits[finite_mask]
        l_finite = tilelang_logits[finite_mask]
        diff = (t_finite - l_finite).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        max_val = max(
            t_finite.abs().max().item(),
            l_finite.abs().max().item(),
            1e-9,
        )
        rel = max_abs / max_val
    else:
        max_abs = mean_abs = rel = 0.0

    return inf_match, _verdict(rel), max_abs, mean_abs, rel


def main():
    print("=" * 110)
    print("Stage 2 CORRECTNESS — triton (ragged_pool_mqa) vs tilelang (pool_mqa_attn_return_logits_fp8)")
    print("=" * 110)
    print(f"{'sq':>4} {'skv':>6} {'K':>4} {'num_pool':>8} | inf | verdict | max|Δ|       mean|Δ|     max|Δ|/max|val|")
    print("-" * 110)

    cfgs = []
    # K=128: skv=32K → num_pool=256 (just at the block_N=256 boundary)
    # K=64: skv=16K → num_pool=256
    # K=32: skv=8K → num_pool=256
    # K=16: skv=4K → num_pool=256
    # K=8: skv=2K → num_pool=256
    for sq, skv, K in [
        (32, 32768, 128),
        (32, 16384, 64),
        (32,  8192, 32),
        (32,  4096, 16),
        (32,  2048,  8),
        (256, 32768, 128),
        (256, 16384, 64),
        (256,  8192, 32),
        (256,  4096, 16),
        (256,  2048,  8),
    ]:
        try:
            inf_m, verdict, mx, mn, rel = correctness(sq, skv, K)
            num_pool = (skv + K - 1) // K
            inf_str = "OK" if inf_m else "MISMATCH"
            print(
                f"{sq:>4} {skv:>6} {K:>4} {num_pool:>8} | {inf_str:>3} | {verdict:>7} |"
                f" {mx:>10.3e}  {mn:>10.3e}  {rel:>10.3e}"
            )
        except Exception as e:
            num_pool = (skv + K - 1) // K
            print(f"{sq:>4} {skv:>6} {K:>4} {num_pool:>8} | ERROR: {type(e).__name__}: {str(e)[:60]}")

    print()
    print("=" * 80)
    print("Stage 2 SPEED — wall-time per call (μs), avg over 300 iters")
    print("=" * 80)
    print(f"{'sq':>4} {'skv':>6} {'K':>4} {'num_pool':>8} | tilelang  triton   tl/triton  winner")
    print("-" * 80)
    for sq, skv, K in [
        (32, 32768, 128),
        (32, 16384,  64),
        (32,  8192,  32),
        (32,  4096,  16),
        (32,  2048,   8),
        (256, 32768, 128),
        (256, 16384,  64),
        (256,  8192,  32),
        (256,  4096,  16),
        (256,  2048,   8),
    ]:
        q, bk, bks, w, cu_ks, cu_ke, cu_ks_blk, cu_ke_blk = make_inputs(sq, skv, K)

        tilelang_fn = lambda: pool_mqa_attn_return_logits_fp8_interface(
            q_fp8=q, blocked_kv_fp8=bk, blocked_kv_scale=bks,
            kv_block_size=K, weights_f32=w,
            cu_seqlen_blocked_ks=cu_ks_blk, cu_seqlen_blocked_ke=cu_ke_blk,
        )
        triton_fn = lambda: ragged_pool_mqa_triton(
            q_fp8=q, blocked_k_fp8=bk, blocked_k_scale=bks,
            weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
            k_block_size=K,
        )
        try:
            tl_us = bench_one(tilelang_fn)
            tr_us = bench_one(triton_fn)
            ratio = tl_us / tr_us
            winner = "tilelang" if ratio < 1.0 else "triton"
            num_pool = (skv + K - 1) // K
            print(
                f"{sq:>4} {skv:>6} {K:>4} {num_pool:>8} | {tl_us:>7.2f}  {tr_us:>7.2f}   "
                f"{ratio:>5.2f}x   {winner}"
            )
        except Exception as e:
            print(f"{sq:>4} {skv:>6} {K:>4} | ERROR: {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
