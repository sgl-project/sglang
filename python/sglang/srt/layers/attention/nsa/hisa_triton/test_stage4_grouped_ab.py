"""Stage 4 (block_sparse_mqa_return_logits) grouped A/B at K<64:
tilelang grouped vs triton grouped.

The non-grouped tilelang kernel sets ``block_N = T.min(block_N, K)``, so at
K<64 the GEMM tile becomes ``[K, D]×[D, H]`` — a fragmenting WGMMA shape.
Both candidate kernels here pack ``G = block_N // K`` consecutive topk
indices into one full ``[block_N=64, D]`` GEMM tile to restore tensor-core
utilisation.

Compares:
  * fp8_native_block_sparse_mqa_attn_return_logits_grouped_interface (tilelang)
  * block_sparse_mqa_triton (triton, dispatches to grouped kernel for K<64)

We feed both kernels the SAME inputs (q, k, k_scale, topk_block_index,
weights, cu_seqlen_ks/ke). Both produce ``[seq_q, topk * K]`` f32 with
``-inf`` at out-of-[ks,ke) positions.

Bounds-of-inputs note: both grouped kernels mask invalid topk-ids via the
k_row >= 0 path (tilelang uses the ``-1`` sentinel from interface padding;
triton uses k_rows < seq_kv directly), so we don't need padding to match.
"""
from __future__ import annotations

import time
import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_sparse_mqa_attn_return_logits_grouped_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    block_sparse_mqa_triton,
)


DEVICE = torch.device("cuda")
H, D = 64, 128


def make_inputs(seq_q, seq_kv, K, topk):
    """Generate q/k/k_scale/weights/cu_seqlen + a valid topk_block_index."""
    torch.manual_seed(0)
    q = torch.randn(seq_q, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(seq_kv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(seq_q, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(seq_q, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(seq_kv // 4, seq_kv, seq_q, device=DEVICE).to(torch.int32)

    # Random valid topk_block_indices in [0, seq_kv // K).
    num_blocks = seq_kv // K
    topk_block_index = torch.randint(
        0, num_blocks, (seq_q, topk), device=DEVICE, dtype=torch.int64,
    )
    return q, k_fp8, k_scale, topk_block_index, weights, cu_ks, cu_ke


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


def correctness(seq_q, seq_kv, K, topk):
    q, kfp8, ks, topk_idx, w, cu_ks, cu_ke = make_inputs(seq_q, seq_kv, K, topk)

    triton_logits = block_sparse_mqa_triton(
        q_fp8=q, k_fp8=kfp8, k_scale=ks,
        topk_block_index=topk_idx,
        kv_block_size=K, weights=w,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    tilelang_logits = fp8_native_block_sparse_mqa_attn_return_logits_grouped_interface(
        q=q, k=kfp8, k_scale=ks,
        topk_block_index=topk_idx,
        kv_block_size=K, weights=w,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )

    inf_t = torch.isinf(triton_logits)
    inf_l = torch.isinf(tilelang_logits)
    inf_match = (inf_t == inf_l).all().item()

    finite_mask = torch.isfinite(triton_logits) & torch.isfinite(tilelang_logits)
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
    print("Stage 4 grouped CORRECTNESS — tilelang grouped vs triton grouped (K<64)")
    print("=" * 110)
    print(f"{'sq':>4} {'skv':>6} {'K':>4} {'topk':>5} | inf | verdict | max|Δ|       mean|Δ|     max|Δ|/max|val|")
    print("-" * 110)

    # Shapes mimic real prefill: bigger skv at smaller K so the topk
    # candidate space is meaningful. Triton grouped requires
    # topk % (256 // K) == 0; tilelang grouped requires topk % (64 // K) == 0
    # but pads internally — the test passes the unpadded topk to both.
    cfgs = [
        (32,  4096,  8, 512),
        (32,  4096, 16, 256),
        (32,  4096, 32, 128),
        (32,  4096, 64,  64),
        (256, 16384,  8, 512),
        (256, 16384, 16, 256),
        (256, 16384, 32, 128),
        (256, 16384, 64,  64),
    ]
    for sq, skv, K, topk in cfgs:
        try:
            inf_m, verdict, mx, mn, rel = correctness(sq, skv, K, topk)
            inf_str = "OK" if inf_m else "MISMATCH"
            print(
                f"{sq:>4} {skv:>6} {K:>4} {topk:>5} | {inf_str:>3} | {verdict:>7} |"
                f" {mx:>10.3e}  {mn:>10.3e}  {rel:>10.3e}"
            )
        except Exception as e:
            print(f"{sq:>4} {skv:>6} {K:>4} {topk:>5} | ERROR: {type(e).__name__}: {str(e)[:60]}")

    print()
    print("=" * 80)
    print("Stage 4 grouped SPEED — wall-time per call (μs), avg over 300 iters")
    print("=" * 80)
    print(f"{'sq':>4} {'skv':>6} {'K':>4} {'topk':>5} | tilelang  triton   tl/triton  winner")
    print("-" * 80)
    for sq, skv, K, topk in cfgs:
        q, kfp8, ks, topk_idx, w, cu_ks, cu_ke = make_inputs(sq, skv, K, topk)

        tilelang_fn = lambda: fp8_native_block_sparse_mqa_attn_return_logits_grouped_interface(
            q=q, k=kfp8, k_scale=ks,
            topk_block_index=topk_idx,
            kv_block_size=K, weights=w,
            cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )
        triton_fn = lambda: block_sparse_mqa_triton(
            q_fp8=q, k_fp8=kfp8, k_scale=ks,
            topk_block_index=topk_idx,
            kv_block_size=K, weights=w,
            cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )
        try:
            tl_us = bench_one(tilelang_fn)
            tr_us = bench_one(triton_fn)
            ratio = tl_us / tr_us
            winner = "tilelang" if ratio < 1.0 else "triton"
            print(
                f"{sq:>4} {skv:>6} {K:>4} {topk:>5} | {tl_us:>7.2f}  {tr_us:>7.2f}   "
                f"{ratio:>5.2f}x   {winner}"
            )
        except Exception as e:
            print(f"{sq:>4} {skv:>6} {K:>4} {topk:>5} | ERROR: {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
