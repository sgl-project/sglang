"""Prefill stage 2 (pool MQA) A/B bench: tilelang vs DeepGEMM.

Aligned with ``test_e2e_simulated.py``: same input generator
(``make_prefill_inputs``), same skv sweep (8K..128K, 16 chunks), same H, D,
sq=8192. So the per-chunk numbers here multiplied by 61 layers × B should
roughly match the prefill ms in the e2e bench.

Why the standalone-vs-e2e gap: the previous version of this bench tested
only ctx=128K with ``cu_ke = arange(ctx-sq+1, ctx+1)`` (all rows touch
~full ctx) — that's the worst-case point, where DG wins ~2x. Averaged
across 16 chunks at skv ∈ {8K,16K,...,128K} with ``cu_ke = linspace
(skv/4, skv, sq)`` (mirroring the e2e workload), the relative savings
drop to ~16-24% per chunk because shorter skv means less GEMM work and
less savings.

Bench compares 4 paths per (skv, K):
  A.  tilelang full (legacy production): pool_mqa + clean+maintain (fused)
  B0. DeepGEMM raw (clean_logits=False)              — pure GEMM cost
  B1. DeepGEMM + clean (clean_logits=True)           — GEMM + internal -inf
  B2. B1 + stride-aware ``force_maintain_logits_triton`` on a sliced view
      (no .contiguous() copy)                        — current production

Output column ``skv pad`` shows DG's SM-aligned row-stride padding (=
``stride(0) - n_blocks``); padding is constant across skv at a given K
(typically 256 elems for n_blocks > 256).
"""
from __future__ import annotations

import torch
import deep_gemm

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_mean_pooling_interface,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    force_maintain_logits_triton,
)
# Reuse e2e's input generator so the two scripts cannot drift.
from sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated import (
    H, D, PREFILL_CHUNK, N_PREFILL_CHUNKS,
    make_prefill_inputs,
)


DEVICE = torch.device("cuda")


def prep_pool_inputs(sq, skv, K):
    """Pool the e2e prefill inputs once and reuse across A/B paths."""
    q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
    n_blocks = (skv + K - 1) // K
    bk, bks = fp8_native_block_mean_pooling_interface(k_fp8, k_scale, K)
    bk = bk[:n_blocks].contiguous()
    bks = bks[:n_blocks].contiguous()
    cu_ks_blk = cu_ks // K
    cu_ke_blk = (cu_ke + K - 1) // K
    return q, bk, bks, w, cu_ks_blk, cu_ke_blk, n_blocks


def run_tilelang(q, bk, bks, K, w, cu_ks_blk, cu_ke_blk):
    return pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q, blocked_kv_fp8=bk, blocked_kv_scale=bks,
        kv_block_size=K, weights_f32=w,
        cu_seqlen_blocked_ks=cu_ks_blk,
        cu_seqlen_blocked_ke=cu_ke_blk,
        clean_logits=True, force_maintain=True,
    )


def run_deepgemm_raw(q, bk, bks, w, cu_ks_blk, cu_ke_blk):
    return deep_gemm.fp8_mqa_logits(
        q, (bk, bks), w, cu_ks_blk, cu_ke_blk, clean_logits=False,
    )


def run_deepgemm_clean(q, bk, bks, w, cu_ks_blk, cu_ke_blk):
    return deep_gemm.fp8_mqa_logits(
        q, (bk, bks), w, cu_ks_blk, cu_ke_blk, clean_logits=True,
    )


def run_deepgemm_clean_view(q, bk, bks, w, cu_ks_blk, cu_ke_blk, n_blocks):
    """Production path: DG + slice view + stride-aware force_maintain."""
    out = deep_gemm.fp8_mqa_logits(
        q, (bk, bks), w, cu_ks_blk, cu_ke_blk, clean_logits=True,
    )[:, :n_blocks]
    force_maintain_logits_triton(out, cu_ks_blk, cu_ke_blk)
    return out


def compare_valid_region(a, b, n_blocks, label):
    """Compare logits in [:, :n_blocks] only, ignoring +inf force_maintain markers
    (tilelang A applies them; raw DG B0/B1 do not)."""
    a_valid = a[:, :n_blocks]
    b_valid = b[:, :n_blocks]
    posinf_mask = torch.isposinf(a_valid)
    a_clean = torch.where(posinf_mask, b_valid, a_valid)
    diff = (a_clean - b_valid).abs()
    finite = torch.isfinite(a_clean) & torch.isfinite(b_valid)
    if finite.any():
        max_abs = diff[finite].max().item()
        scale = a_clean[finite].abs().max().item()
        rel = max_abs / max(scale, 1e-6)
    else:
        max_abs, rel = float("nan"), float("nan")
    a_neginf = torch.isneginf(a_valid)
    b_neginf = torch.isneginf(b_valid)
    neginf_match = (a_neginf == b_neginf).all().item()
    print(f"  {label}: -inf_layout_match={neginf_match}  "
          f"max_abs={max_abs:.4f}  rel={rel:.4e}")


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def main():
    sq = PREFILL_CHUNK
    skv_list = [(i + 1) * PREFILL_CHUNK for i in range(N_PREFILL_CHUNKS)]
    print(f"prefill stage 2 A/B: tilelang vs DeepGEMM "
          f"(aligned with test_e2e_simulated)")
    print(f"sq={sq}  H={H}  D={D}  cu_ke=linspace(skv/4, skv, sq)")
    print("=" * 110)

    for K in (16, 32, 64, 128):
        print()
        print(f"--- K={K} ---")
        print(f"{'skv':>6} {'n_blk':>6} {'pad':>4} | "
              f"{'A: tile':>9} {'B0: dg':>8} {'B1: +cln':>10} {'B2: +ftm':>10} | "
              f"{'B2/A':>6} {'A-B2':>7}")
        print("-" * 110)
        sum_a = 0.0
        sum_b2 = 0.0
        for skv in skv_list:
            q, bk, bks, w, cu_ks_blk, cu_ke_blk, n_blocks = prep_pool_inputs(sq, skv, K)

            # Correctness sanity at the largest skv only (cheap, deterministic).
            if skv == skv_list[-1]:
                ref = run_tilelang(q, bk, bks, K, w, cu_ks_blk, cu_ke_blk)
                b1 = run_deepgemm_clean(q, bk, bks, w, cu_ks_blk, cu_ke_blk)
                compare_valid_region(ref, b1, n_blocks, f"K={K} skv={skv}: A vs B1")
                pad = b1.stride(0) - n_blocks
            else:
                # Probe pad cheaply once.
                tmp = run_deepgemm_clean(q, bk, bks, w, cu_ks_blk, cu_ke_blk)
                pad = tmp.stride(0) - n_blocks
                del tmp

            t_a = bench(lambda: run_tilelang(q, bk, bks, K, w, cu_ks_blk, cu_ke_blk))
            t_b0 = bench(lambda: run_deepgemm_raw(q, bk, bks, w, cu_ks_blk, cu_ke_blk))
            t_b1 = bench(lambda: run_deepgemm_clean(q, bk, bks, w, cu_ks_blk, cu_ke_blk))
            t_b2 = bench(lambda: run_deepgemm_clean_view(
                q, bk, bks, w, cu_ks_blk, cu_ke_blk, n_blocks,
            ))

            sum_a += t_a
            sum_b2 += t_b2
            print(f"{skv:>6} {n_blocks:>6} {pad:>4} | "
                  f"{t_a:>9.1f} {t_b0:>8.1f} {t_b1:>10.1f} {t_b2:>10.1f} | "
                  f"{t_b2/t_a:>5.2f}x {t_a-t_b2:>+7.1f}")

            del q, bk, bks, w, cu_ks_blk, cu_ke_blk
            torch.cuda.empty_cache()

        print("-" * 110)
        print(f"  16-chunk SUM (μs/req): A={sum_a:>7.0f}  B2={sum_b2:>7.0f}  "
              f"saved={sum_a-sum_b2:>+7.0f}  ratio={sum_b2/sum_a:.2f}x")


if __name__ == "__main__":
    main()
