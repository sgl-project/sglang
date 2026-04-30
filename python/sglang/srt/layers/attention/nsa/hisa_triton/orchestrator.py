"""Default decode/prefill orchestrators — paged pool-K cache + triton hotspots.

Same control flow as the tilelang fallback
``fp8_native_hierarchy_paged_mqa_logits_tilelang_with_pool_cache`` in
``hisa/custom_ops.py``, but stages 2 and 4 swap to triton ports:

    1) tail_only       (tilelang for K>=paged_block, triton for K<paged_block)
    2) block-MQA       (→ triton batch_decode_pool_mqa_triton, 1-3.5x faster)
    3) torch.topk      (unchanged)
    4) sparse-paged    (→ triton sparse_paged_mqa_triton, 6-15x faster)

Per-step on decode (B=10, ctx=65K): ~12 ms / step vs the all-tilelang
fallback at the indexer level (steady-state ~5 ms vs ~17 ms). Correctness:
fp8 ULP drift <= 2.6% rel, topk-2048 IoU >= 0.997 vs tilelang — within fp8
accumulation noise, no e2e regression expected (verify with
``SGLANG_HISA_VERIFY=1`` when flipping the default).
"""
from __future__ import annotations

import os

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import (
    MAX_TOPK as _FAST_TOPK_MAX,
    fast_topk_runtime,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_triton,
    batch_pool_mqa_triton,
    block_mean_pooling_triton,
    block_sparse_mqa_triton,
    paged_mean_pooling_triton,
    ragged_pool_mqa_triton,
    sparse_paged_mqa_triton,
)


# Stage-3 fast_topk dispatch.
#   prefill K∈{16,32}  → -3 to -6% e2e at production block_topk = 8192//K
#                        (test_e2e_simulated.py, ctx=128K). Stage 4 is huge
#                        per-call so CPU launch latency overlaps cleanly.
#   prefill K∈{64,128} → loses kernel-side at small block_topk; stays on
#                        torch.topk(bf16).
#   decode all K       → kernel wins ~4× in isolation; eager-pipeline
#                        bench shows ~3-6 μs/step regression because
#                        stage-3 GPU (9 μs) is shorter than stage-4
#                        launch latency (CUDA-graph A/B confirms it: under
#                        capture+replay, fast_topk wins ~25 μs/step).
#                        Wired live for production A/B — sglang's
#                        piecewise-graph behavior may differ from our
#                        eager bench. Toggle off via env var if it
#                        regresses real-world.
# Output i32 — both downstream wrappers accept it (no .to() cast).
_FAST_TOPK_DISABLE = os.environ.get("SGLANG_HISA_FAST_TOPK_DISABLE", "0") == "1"


def _stage3_topk_prefill(
    score_2d: torch.Tensor, block_topk: int, k_block_size: int,
) -> torch.Tensor:
    """Prefill stage 3: [seq, L] f32 → [seq, topk] (i32 fast / i64 torch).

    ``topk = min(block_topk, L)`` enforces fast_topk's ``L >= topk`` contract
    on short prefill chunks (where ``L < block_topk``). Stage 2 already writes
    -inf at invalid block positions, so the radix select picks correctly
    without per-row lengths.
    """
    topk = min(block_topk, score_2d.shape[-1])
    if _FAST_TOPK_DISABLE or k_block_size not in (16, 32) or topk > _FAST_TOPK_MAX:
        # bf16 path: ~40% faster than f32 torch.topk on long rows.
        return torch.topk(
            score_2d.bfloat16(), k=topk, dim=-1, sorted=False,
        ).indices
    return fast_topk_runtime(score_2d, topk)


def _stage3_topk_decode(
    score_3d: torch.Tensor, block_topk: int,
) -> torch.Tensor:
    """Decode stage 3: [B, 1, L] f32 → [B, 1, topk] (i32 fast / i64 torch).

    ``topk = min(block_topk, L)`` enforces fast_topk's ``L >= topk`` contract.
    In decode L is the fixed pool buffer width (max_pool_pages * pool_page_size),
    so the clamp is a no-op in production. Stage 2 (``_batch_decode_pool_mqa_kernel``)
    writes -inf at all ``pool_idx >= context_lens_pool``, so the radix select
    picks correctly over the full row.
    """
    topk = min(block_topk, score_3d.shape[-1])
    if _FAST_TOPK_DISABLE or topk > _FAST_TOPK_MAX:
        return torch.topk(score_3d, k=topk, dim=-1, sorted=False).indices
    B, S, L = score_3d.shape
    return fast_topk_runtime(score_3d.view(B * S, L), topk).view(B, S, topk)


def fp8_native_hierarchy_paged_mqa_logits(
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
    # 1) Tail-pool refresh is *skipped*. Stage 2's force_maintain at
    # ``pool_idx == k_e - 1`` (kernels.py: _batch_decode_pool_mqa_kernel)
    # always assigns +inf to the tail block — so the tail's actual mean-pool
    # value never affects topk selection. Stage 4 reads raw KV cache for the
    # selected blocks, not pool_k_pages, so a stale tail in pool_k_pages
    # also doesn't contaminate the final logits. Saves ~7-10 μs/layer per
    # decode step. ``update_pool_for_completed_blocks_*`` (called from the
    # store-side hook in pool_k_cache.py) still keeps non-tail blocks fresh.

    # 2) Block-MQA — triton port of paged pool_k_pages reader.
    num_pool_blocks_per_req = (context_lens + k_block_size - 1) // k_block_size
    block_k_indexer_score = batch_decode_pool_mqa_triton(
        q_fp8=q_fp8,
        pool_k_pages=pool_k_pages,
        pool_page_tables=pool_page_tables,
        weights_f32=weights,
        context_lens_pool=num_pool_blocks_per_req,
        pool_page_size=pool_page_size,
    )  # [B, 1, max_pool_pages * pool_page_size] f32

    # 3) Top-k over pool blocks.
    topk_block_indices = _stage3_topk_decode(block_k_indexer_score, block_topk)

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


def fp8_native_hierarchy_paged_mqa_logits_no_pool_cache(
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
    ``fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy`` (tilelang) but every kernel
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
    topk_block_indices = _stage3_topk_decode(block_k_indexer_score, block_topk)

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


def fp8_native_hierarchy_mqa_logits(
    q_fp8: torch.Tensor,                             # [seq, H, D] fp8
    kv: tuple[torch.Tensor, torch.Tensor],           # (k_fp8 [N, D] fp8, k_scale [N, 4] uint8 OR [N] f32)
    weights: torch.Tensor,                           # [seq, H] f32
    cu_seqlen_ks: torch.Tensor,                      # [seq] i32
    cu_seqlen_ke: torch.Tensor,                      # [seq] i32
    k_block_size: int,
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed-stage ragged prefill hierarchy MQA. Mirrors the tilelang
    ``fp8_native_hierarchy_mqa_logits_tilelang_legacy`` but routes around its K<64 OOB by
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

    # 3) Top-k over pool blocks. fast_topk_runtime at K∈{16,32} (1.45-1.83×),
    # torch.topk(bf16) at K∈{64,128} (fast_topk's setup tax > sort savings
    # at small topk). Helper above hides the dispatch and the L>=topk clamp.
    topk_block_indices = _stage3_topk_prefill(
        block_k_indexer_score, block_topk, k_block_size,
    )  # [seq, min(block_topk, L)] int32 (fast_topk) or int64 (torch.topk)

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
        )  # [seq, min(block_topk, L) * k_block_size] f32
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
