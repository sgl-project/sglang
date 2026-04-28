"""v4 orchestrator — paged pool-K cache (v3 layout) + triton hotspot kernels.

Same control flow as ``fp8_native_hierarchy_paged_mqa_logits_with_pool_cache_v3``
in ``hisa/custom_ops.py``:

    1) tail_only_v3   (tilelang, cheap, ~10 μs)
    2) block-MQA      (→ triton batch_decode_pool_mqa_v3_triton,  1-3.5× faster)
    3) torch.topk     (unchanged)
    4) sparse-paged   (→ triton sparse_paged_mqa_triton,  6-15× faster)

Per-step on decode (B=10, ctx=65K): v4 saves ~12 ms / step vs v3 at the
indexer level (steady-state total ~5 ms vs v3's ~17 ms). Correctness: fp8
ULP drift ≤ 2.6% rel, topk-2048 IoU ≥ 0.997 vs tilelang — within fp8
accumulation noise, no e2e regression expected (verify via
``SGLANG_HISA_VERIFY=1`` when flipping the default).
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
    fp8_native_paged_mean_pooling_tail_only_v3_interface,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_v3_triton,
    batch_pool_mqa_triton,
    block_mean_pooling_triton,
    block_sparse_mqa_triton,
    paged_mean_pooling_triton,
    ragged_pool_mqa_triton,
    sparse_paged_mqa_triton,
    tail_only_v3_triton,
)


def fp8_native_hierarchy_paged_mqa_logits_with_pool_cache_v4(
    q_fp8: torch.Tensor,                # [B, 1, H, D] fp8
    kv_cache_fp8: torch.Tensor,         # [num_blocks, paged_block_size, 1, D+4] uint8
    pool_k_pages: torch.Tensor,         # [num_pool_pages_global, pool_page_size * (D+4)] uint8
    pool_page_tables: torch.Tensor,     # [B, max_pool_pages] i32
    weights: torch.Tensor,              # [B*1, H] f32
    context_lens: torch.Tensor,         # [B] i32 — raw seq_len per request
    block_tables: torch.Tensor,         # [B, max_kv_blocks] i32
    k_block_size: int,
    pool_page_size: int,
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1) Refresh tail pool block in place. Dispatch: tilelang for k>=paged
    # (the well-tested production path), SK16 triton for k<paged where
    # tilelang would assert pooling%paged==0.
    paged_block_size = kv_cache_fp8.shape[1]
    if k_block_size < paged_block_size:
        num_phys = kv_cache_fp8.shape[0]
        kv_cache_flat = kv_cache_fp8.view(num_phys, -1)
        tail_only_v3_triton(
            kv_cache_flat=kv_cache_flat,
            context_lens=context_lens,
            block_tables=block_tables,
            pool_page_tables=pool_page_tables,
            pool_k_pages=pool_k_pages,
            k_block_size=k_block_size,
            paged_block_size=paged_block_size,
            pool_page_size=pool_page_size,
        )
    else:
        fp8_native_paged_mean_pooling_tail_only_v3_interface(
            kv_cache=kv_cache_fp8, context_lens=context_lens,
            block_tables=block_tables,
            pool_page_tables=pool_page_tables,
            pool_k_pages=pool_k_pages,
            k_block_size=k_block_size,
            pool_page_size=pool_page_size,
        )

    # 2) Block-MQA — triton port of paged pool_k_pages reader.
    num_pool_blocks_per_req = (context_lens + k_block_size - 1) // k_block_size
    block_k_indexer_score = batch_decode_pool_mqa_v3_triton(
        q_fp8=q_fp8,
        pool_k_pages=pool_k_pages,
        pool_page_tables=pool_page_tables,
        weights_f32=weights,
        context_lens_pool=num_pool_blocks_per_req,
        pool_page_size=pool_page_size,
    )  # [B, 1, max_pool_pages * pool_page_size] f32

    # 3) Top-k over pool blocks — torch native.
    topk_block_indices = torch.topk(
        block_k_indexer_score,
        k=min(block_topk, block_k_indexer_score.shape[-1]),
        dim=-1, sorted=False,
    ).indices

    # 4) Sparse paged MQA — triton port of the decode hotspot.
    block_sparse_k_indexer_score = sparse_paged_mqa_triton(
        q_fp8=q_fp8,
        kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_block_indices,
        kv_block_size=k_block_size,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_tables,
    )
    return block_sparse_k_indexer_score, topk_block_indices


def fp8_native_hierarchy_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,                # [B, 1, H, D] fp8
    kv_cache_fp8: torch.Tensor,         # [num_blocks, paged_block_size, 1, D+4] uint8
    weights: torch.Tensor,              # [B*1, H] f32
    context_lens: torch.Tensor,         # [B] i32 — raw seq_len per request
    block_tables: torch.Tensor,         # [B, max_kv_blocks] i32
    k_block_size: int,
    block_topk: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-triton fp8 hierarchy MQA, no pool cache. Mirrors
    ``fp8_native_hierarchy_paged_mqa_logits`` (tilelang) but every kernel
    is the SK1..SK12 triton variant, so it works for k_block_size in
    {8, 16, 32, 64, 128} (tilelang ones break for k<64).

    Used by ``_get_topk_paged`` whenever ``k_block_size < 64``, regardless
    of pool-cache env-var state — tilelang would assert there.

    Flow (matches v1 baseline, no cache):
      1) paged mean-pool — ``paged_mean_pooling_triton`` (SK2)
      2) block-MQA score — ``batch_pool_mqa_triton`` (SK10) on contiguous blocked_k
      3) torch.topk
      4) sparse paged MQA — ``sparse_paged_mqa_triton`` (SK3/SK12)

    Returns ``(block_sparse_logits[B, 1, topk*k_block_size],
    topk_block_indices[B, 1, topk] int64)``.
    """
    B, seq_q, H, D = q_fp8.shape
    assert seq_q == 1, "decode expects q_len=1"
    max_num_pool = (max_seq_len + k_block_size - 1) // k_block_size

    # 1) Fresh paged mean-pool (no cache).
    blocked_k_fp8, blocked_k_scale, num_pool_blocks = paged_mean_pooling_triton(
        max_num_pooling_blocks=max_num_pool,
        kv_cache=kv_cache_fp8,
        context_lens=context_lens,
        block_tables=block_tables,
        k_block_size=k_block_size,
    )  # blocked_k: [B, max_num_pool, D] fp8 ; scale: [B, max_num_pool] f32

    # 2) Block-MQA on contiguous blocked_k.
    block_k_indexer_score = batch_pool_mqa_triton(
        q_fp8=q_fp8,
        blocked_k_fp8=blocked_k_fp8,
        blocked_k_scale=blocked_k_scale,
        weights_f32=weights,
        context_lens=num_pool_blocks,
    )  # [B, 1, max_num_pool] f32

    # 3) Top-k over pool blocks.
    topk_block_indices = torch.topk(
        block_k_indexer_score,
        k=min(block_topk, block_k_indexer_score.shape[-1]),
        dim=-1, sorted=False,
    ).indices  # [B, 1, topk] int64

    # 4) Sparse paged MQA on the chosen K-blocks.
    block_sparse_k_indexer_score = sparse_paged_mqa_triton(
        q_fp8=q_fp8,
        kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_block_indices,
        kv_block_size=k_block_size,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_tables,
    )  # [B, 1, topk*k_block_size] f32
    return block_sparse_k_indexer_score, topk_block_indices


def fp8_native_hierarchy_mqa_logits_triton(
    q_fp8: torch.Tensor,                             # [seq, H, D] fp8
    kv: tuple[torch.Tensor, torch.Tensor],           # (k_fp8 [N, D] fp8, k_scale [N, 4] uint8 OR [N] f32)
    weights: torch.Tensor,                           # [seq, H] f32
    cu_seqlen_ks: torch.Tensor,                      # [seq] i32
    cu_seqlen_ke: torch.Tensor,                      # [seq] i32
    k_block_size: int,
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed-stage ragged prefill hierarchy MQA. Mirrors the tilelang
    ``fp8_native_hierarchy_mqa_logits`` but routes around its K<64 OOB by
    using bounds-safe stage-1 variants. Stages 1-2 are tilelang at all K
    (per A/B in test_grouped_mean_pool / test_stage2_ab); stage 4 is split
    by K (test_stage4_full_ab at sq=8192, skv∈{8K..64K}).

    Flow:
      1) ragged mean-pool — tilelang. ``..._grouped_interface`` when K<64
         (one CTA per block_N=64 tokens, no over-read), ``..._interface``
         when K>=64 (vanilla, safe because pool block already >= block_N).
      2) ragged block-MQA — tilelang ``pool_mqa_attn_return_logits_fp8`` on
         the [num_pool, D] blocked_k from stage 1.
      3) ``torch.topk`` on bf16 logits.
      4) ragged sparse-MQA — split:
         - K<128: triton ``block_sparse_mqa_triton`` (grouped path,
           GEMM_TILE=256, GROUP_SIZE=256/K). Beats tilelang 1.4-8x at
           sq=8192: tilelang's per-iter G TMA loads pay setup tax that
           dominates at small K, while triton's single [256, D] gather
           load + m=256 WGMMA tile saturates HBM and tensor cores.
         - K=128: tilelang ``..._interface`` (vanilla, block_N=128).
           Wins ~15% vs triton in this range — TMA bulk load (16KB/shot)
           tops out HBM where triton's 256-row gather can't, and the
           cdll-thin launcher amortises across the whole sq=8192 chunk.

    Returns ``(block_sparse_logits[seq, topk*K], topk_block_indices[seq, topk] int64)``.
    """
    k_fp8, k_scales = kv
    # k_scales arrives from get_index_k_scale_buffer as uint8 [N, 4] (= one
    # f32 packed). Triton kernels want f32. Mirror the cast in custom_ops.py.
    if k_scales.dtype == torch.uint8:
        k_scales = k_scales.view(torch.float32)
    if k_scales.ndim == 2:
        assert k_scales.shape[1] == 1, (
            f"k_scales should be [N] or [N, 1], got {k_scales.shape}"
        )
        k_scales = k_scales.squeeze(1)

    # 1) Mean-pool ragged K → tilelang. Bench (test_grouped_mean_pool.py) shows
    # tilelang stage 1 is 1.8x faster than the triton equivalent at all K
    # (8..128) due to lower per-launch Python overhead (~9μs vs triton's
    # ~24μs). The vanilla tilelang kernel has a boundary OOB at K<block_N=64,
    # so we route those to the bounds-safe grouped variant.
    if k_block_size < 64:
        blocked_k_fp8, blocked_k_scale = fp8_native_block_mean_pooling_grouped_interface(
            k_fp8, k_scales, k_block_size,
        )
    else:
        blocked_k_fp8, blocked_k_scale = fp8_native_block_mean_pooling_interface(
            k_fp8, k_scales, k_block_size,
        )

    # 2) Block-MQA on blocked_k → tilelang. Per-stage A/B (test_stage2_ab.py)
    # shows tilelang's pool_mqa_attn_return_logits_fp8 is 1.6x faster than
    # ragged_pool_mqa_triton at all K (18.6μs vs 29.7μs wall-time): tilelang
    # has lower launch overhead AND its [ks_min, ke_max] union skip + clean+
    # maintain-as-separate-launch combination beats the triton fused mask
    # approach end-to-end. Tilelang takes blocked cu_seqlen, so we have to
    # add back 3 host-side PyTorch launches (~5-6μs cost) — net still ~7μs
    # positive per orchestrator call. Refactor TODO: move ``// K`` into the
    # tilelang kernel via a K_BLOCK_SIZE param to recover those.
    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
    block_k_indexer_score = pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q_fp8,
        blocked_kv_fp8=blocked_k_fp8,
        blocked_kv_scale=blocked_k_scale,
        kv_block_size=k_block_size,
        weights_f32=weights,
        cu_seqlen_blocked_ks=cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke=cu_seqlen_blocked_ke,
    )

    # 3) Top-k over pool blocks. bf16 + sorted=False matches the tilelang
    # path (~40% faster than f32 on long row, ordering doesn't matter for
    # downstream sparse-MQA).
    topk_actual = min(block_topk, block_k_indexer_score.shape[-1])
    topk_block_indices = torch.topk(
        block_k_indexer_score.bfloat16(),
        k=topk_actual,
        dim=-1, sorted=False,
    ).indices  # [seq, topk_actual] int64

    # 4) Sparse-MQA on raw K — split dispatch by K (sweep at sq=8192,
    # skv∈{8K,16K,32K,64K}, test_stage4_full_ab.py):
    #   K<128: triton grouped is 1.4-8x faster than tilelang. tilelang's
    #     per-iter G TMA loads pay setup tax that dominates at small K
    #     (TMA descriptor overhead ~constant, transfer time ∝ K rows);
    #     triton's [256, D] gather load + m=256 WGMMA tile wins.
    #   K=128: tilelang vanilla wins ~15% (skv=8K..64K). At full K
    #     bandwidth TMA bulk load (16KB/shot) tops out HBM where triton's
    #     256-row gather can't, and the cdll-thin launcher amortises
    #     across the whole sq=8192 chunk too.
    if k_block_size == 128:
        block_sparse_logits = fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q=q_fp8,
            k=k_fp8,
            k_scale=k_scales,
            topk_block_index=topk_block_indices,
            kv_block_size=k_block_size,
            weights=weights,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
        )  # [seq, topk_actual * k_block_size] f32
    else:
        block_sparse_logits = block_sparse_mqa_triton(
            q_fp8=q_fp8,
            k_fp8=k_fp8,
            k_scale=k_scales,
            topk_block_index=topk_block_indices,
            kv_block_size=k_block_size,
            weights=weights,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
        )

    return block_sparse_logits, topk_block_indices
