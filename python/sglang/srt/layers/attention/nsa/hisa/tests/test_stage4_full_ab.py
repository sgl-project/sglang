"""Stage 4 (block_sparse_mqa) FULL A/B across K∈{8,16,32,64,128}.

Tilelang side dispatches:
  K < 64  → ``..._grouped_interface`` (G=block_N/K consecutive topks
            packed into a [block_N, D] WGMMA tile)
  K >= 64 → ``..._interface``         (vanilla, block_N == K already
            >= WGMMA min)

Triton side: ``block_sparse_mqa_triton`` (auto-dispatches to grouped
for K<128, original for K=128).

Both produce ``[seq_q, topk * K]`` f32 with -inf mask outside [ks, ke).
We compare correctness (fp8-strict) + wall-time at sq∈{32, 256}.
"""
from __future__ import annotations

import time
import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_block_sparse_mqa_attn_return_logits_grouped_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    _block_sparse_mqa_grouped_kernel,
    _block_sparse_mqa_kernel,
    block_sparse_mqa_triton,
)


def triton_grouped_forced_call(q_fp8, k_fp8, k_scale, topk_block_index, K, weights,
                               cu_seqlen_ks, cu_seqlen_ke, GEMM_TILE=256):
    """Force the grouped kernel at any K (including K=128). GROUP_SIZE = GEMM_TILE/K."""
    seq_len, H_, D_ = q_fp8.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_block_index.shape[-1])
    GROUP_SIZE = GEMM_TILE // K
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE  # ceil — kernel handles tail
    logits = torch.empty(
        (seq_len, topk * K), device=q_fp8.device, dtype=torch.float32,
    )
    grid = (seq_len, num_chunks)
    _block_sparse_mqa_grouped_kernel[grid](
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights,
        cu_seqlen_ks, cu_seqlen_ke,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        k_fp8.stride(0), k_fp8.stride(1),
        k_scale.stride(0),
        topk_block_index.stride(0), topk_block_index.stride(1),
        logits.stride(0), logits.stride(1),
        weights.stride(0), weights.stride(1),
        seq_kv,
        topk,
        HEADS=H_, DIM=D_,
        KV_BLOCK_SIZE=K,
        GROUP_SIZE=GROUP_SIZE,
    )
    return logits


def triton_vanilla_call(q_fp8, k_fp8, k_scale, topk_block_index, K, weights,
                        cu_seqlen_ks, cu_seqlen_ke):
    """Triton vanilla: per-CTA = one (seq, K-block) pair, BLOCK_N=K.
    Each CTA loads K rows from ONE topk index — no grouping.
    """
    seq_len, H_, D_ = q_fp8.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_block_index.shape[-1])
    logits = torch.empty(
        (seq_len, topk * K), device=q_fp8.device, dtype=torch.float32,
    )
    BLOCK_N = K  # one CTA per topk block, no sub-splitting
    grid = (seq_len, topk * (K // BLOCK_N))
    _block_sparse_mqa_kernel[grid](
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights,
        cu_seqlen_ks, cu_seqlen_ke,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        k_fp8.stride(0), k_fp8.stride(1),
        k_scale.stride(0),
        topk_block_index.stride(0), topk_block_index.stride(1),
        logits.stride(0), logits.stride(1),
        weights.stride(0), weights.stride(1),
        seq_kv,
        HEADS=H_, DIM=D_,
        KV_BLOCK_SIZE=K,
        BLOCK_N=BLOCK_N,
        SUBS_PER_TOPK=K // BLOCK_N,
    )
    return logits


DEVICE = torch.device("cuda")
H, D = 64, 128


def tilelang_grouped(q, k, k_scale, topk_idx, K, weights, cu_ks, cu_ke):
    """Grouped tilelang (block_N=128, G=block_N/K). Skips at K=128 (G=1 trivial)."""
    return fp8_native_block_sparse_mqa_attn_return_logits_grouped_interface(
        q=q, k=k, k_scale=k_scale, topk_block_index=topk_idx,
        kv_block_size=K, weights=weights,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )


def tilelang_vanilla(q, k, k_scale, topk_idx, K, weights, cu_ks, cu_ke):
    """Vanilla tilelang (block_N = T.min(128, K))."""
    return fp8_native_block_sparse_mqa_attn_return_logits_interface(
        q=q, k=k, k_scale=k_scale, topk_block_index=topk_idx,
        kv_block_size=K, weights=weights,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )


def triton_grouped(q, k, k_scale, topk_idx, K, weights, cu_ks, cu_ke):
    """Force the grouped kernel at ANY K (including K=128, GEMM_TILE=256, G=2)."""
    return triton_grouped_forced_call(q, k, k_scale, topk_idx, K, weights, cu_ks, cu_ke)


def triton_vanilla(q, k, k_scale, topk_idx, K, weights, cu_ks, cu_ke):
    """Force the legacy ``_block_sparse_mqa_kernel`` (no grouping) at any K."""
    return triton_vanilla_call(q, k, k_scale, topk_idx, K, weights, cu_ks, cu_ke)


def make_inputs(sq, skv, K, topk):
    torch.manual_seed(0)
    q = torch.randn(sq, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(skv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(skv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(sq, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(sq, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(skv // 4, skv, sq, device=DEVICE).to(torch.int32)
    num_blocks = skv // K
    topk_idx = torch.randint(
        0, num_blocks, (sq, topk), device=DEVICE, dtype=torch.int64,
    )
    return q, k_fp8, k_scale, topk_idx, weights, cu_ks, cu_ke


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def run(sq, skv, K, topk):
    q, kfp8, ks, topk_idx, w, cu_ks, cu_ke = make_inputs(sq, skv, K, topk)

    def try_bench(fn):
        try:
            return bench(lambda: fn(q, kfp8, ks, topk_idx, K, w, cu_ks, cu_ke))
        except Exception as e:
            return f"ERR:{type(e).__name__}"

    trv_us = try_bench(triton_vanilla)
    trg_us = try_bench(triton_grouped)  # forced grouped at all K
    tlv_us = try_bench(tilelang_vanilla)
    tlg_us = try_bench(tilelang_grouped) if K < 128 else "(=van)"
    return trv_us, trg_us, tlv_us, tlg_us


def main():
    print("=" * 110)
    print("Stage 4 4-way A/B — triton-vanilla vs triton-grouped vs tilelang-vanilla vs tilelang-grouped")
    print("=" * 110)
    print(f"{'sq':>4} {'skv':>6} {'K':>4} {'topk':>5} | "
          f"tr-van   tr-grp   tl-van   tl-grp  | best/worst")
    print("-" * 110)

    # Production chunked-prefill: sq=8192, sweep skv∈{8K, 16K, 32K, 64K},
    # topk = min(2048, skv/K) (same dispatch as HisaIndexer).
    cfgs = []
    for skv in (8192, 16384, 32768, 65536):
        for K in (8, 16, 32, 64, 128):
            num_blocks = skv // K
            topk = min(2048, num_blocks)
            cfgs.append((8192, skv, K, topk))
    for sq, skv, K, topk in cfgs:
        trv_us, trg_us, tlv_us, tlg_us = run(sq, skv, K, topk)

        def fmt(x):
            return f"{x:>7.2f}" if isinstance(x, float) else f"{x:>7s}"

        floats = [t for t in (trv_us, trg_us, tlv_us, tlg_us) if isinstance(t, float)]
        if floats:
            ratio = max(floats) / min(floats)
            ratio_str = f"{ratio:>5.2f}x"
        else:
            ratio_str = "  n/a"
        print(
            f"{sq:>4} {skv:>6} {K:>4} {topk:>5} | "
            f"{fmt(trv_us)}  {fmt(trg_us)}  {fmt(tlv_us)}  {fmt(tlg_us)}  | "
            f"{ratio_str}"
        )


if __name__ == "__main__":
    main()
