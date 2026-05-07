"""Default decode/prefill orchestrators — paged pool-K cache + triton hotspots.

Pipeline:

    1) tail_only       (tilelang for K>=paged_block, triton for K<paged_block)
    2) block-MQA       (→ triton batch_decode_pool_mqa_triton)
    3) fast_topk_runtime
    4) sparse-paged    (→ triton sparse_paged_mqa_triton)

Per-step on decode (B=10, ctx=65K): ~12 ms / step at the indexer level.
Correctness: fp8 ULP drift <= 2.6% rel, topk-2048 IoU >= 0.997 vs reference
tilelang baseline — within fp8 accumulation noise, no e2e regression expected.
"""

from __future__ import annotations

import deep_gemm
import torch

from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import (
    MAX_TOPK as _FAST_TOPK_MAX,
)
from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import (
    fast_topk_runtime,
)
from sglang.srt.layers.attention.nsa.hisa.tilelang_kernels import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    block_sparse_mqa_triton,
    clean_and_force_maintain_logits_decode_triton,
    force_maintain_logits_triton,
    sparse_paged_mqa_triton,
)

# Stage 3: fast_topk_runtime for all K (prefill + decode); only falls back
# to torch.topk(bf16) when topk > _FAST_TOPK_MAX (out of fast_topk's range).
# Production block_topk = 8192//K ∈ {512, 256, 128, 64}, well within
# fast_topk capacity. Output is i32; downstream wrappers accept i32 / i64.


def _stage3_topk_prefill(
    score_2d: torch.Tensor,
    block_topk: int,
) -> torch.Tensor:
    """Prefill stage 3: [seq, L] f32 → [seq, topk] (i32 fast / i64 torch).

    ``topk = min(block_topk, L)`` enforces fast_topk's ``L >= topk`` contract
    on short prefill chunks (where ``L < block_topk``). Stage 2 already writes
    -inf at invalid block positions, so the radix select picks correctly
    without per-row lengths.
    """
    topk = min(block_topk, score_2d.shape[-1])
    if topk > _FAST_TOPK_MAX:
        # bf16 path: ~40% faster than f32 torch.topk on long rows.
        return torch.topk(
            score_2d.bfloat16(),
            k=topk,
            dim=-1,
            sorted=False,
        ).indices
    return fast_topk_runtime(score_2d, topk)


def _stage3_topk_decode(
    score_3d: torch.Tensor,
    block_topk: int,
) -> torch.Tensor:
    """Decode stage 3: [B, 1, L] f32 → [B, 1, topk] (i32 fast / i64 torch).

    ``topk = min(block_topk, L)`` enforces fast_topk's ``L >= topk`` contract.
    In decode L is the fixed pool buffer width (max_pool_pages * pool_page_size),
    so the clamp is a no-op in production. Stage 2 (``_batch_decode_pool_mqa_kernel``)
    writes -inf at all ``pool_idx >= context_lens_pool``, so the radix select
    picks correctly over the full row.
    """
    topk = min(block_topk, score_3d.shape[-1])
    if topk > _FAST_TOPK_MAX:
        return torch.topk(score_3d, k=topk, dim=-1, sorted=False).indices
    B, S, L = score_3d.shape
    return fast_topk_runtime(score_3d.view(B * S, L), topk).view(B, S, topk)


def fp8_native_hierarchy_paged_mqa_logits(
    q_fp8: torch.Tensor,  # [B, 1, H, D] fp8
    kv_cache_fp8: torch.Tensor,  # [num_blocks, paged_block_size, 1, D+4] uint8
    pool_k_pages: torch.Tensor,  # [num_pool_pages_global, pool_page_size * (D+4)] uint8
    pool_page_tables: torch.Tensor,  # [B, max_pool_pages] i32
    weights: torch.Tensor,  # [B*1, H] f32
    context_lens: torch.Tensor,  # [B] i32 — raw seq_len per request
    block_tables: torch.Tensor,  # [B, max_kv_blocks] i32
    k_block_size: int,
    pool_page_size: int,
    block_topk: int,
    max_seq_len: int,  # max ctx in tokens (= block_tables.shape[1] * 64 in production)
    schedule_metadata: torch.Tensor,  # DG pool-domain schedule, computed by caller (graph-stable buffer)
    paged_block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1) Tail-pool refresh is *skipped*. Stage 2's force_maintain at
    # ``pool_idx == k_e - 1`` (clean_and_force_maintain_logits_decode_triton
    # below) always assigns +inf to the tail block — so the tail's actual
    # mean-pool value never affects topk selection. Stage 4 reads raw KV
    # cache for the selected blocks, not pool_k_pages, so a stale tail in
    # pool_k_pages also doesn't contaminate the final logits.

    # 2) Block-MQA via DG ``fp8_paged_mqa_logits``. pool_k_pages is page-level
    # SoA (bytes [0, ps*D) = fp8, bytes [ps*D, ps*(D+4)) = f32 scales),
    # identical to sglang's main index_k kv-cache layout (sgl_kernel
    # fused_store_index_cache.cuh:70-72). DG handles SoA via TWO TMA
    # descriptors (``tensor_map_kv`` + ``tensor_map_kv_scales``) reading
    # different sub-regions of the same buffer — the ``[N_pp, ps, 1, D+4]``
    # 4D view is just for DG to compute page-level pitch.
    #
    # ``schedule_metadata`` is computed once per forward by the caller
    # (HisaIndexer; mirrors nsa_indexer.py:478-483 — getattr-fallback)
    # so that all 61 layers share a stable buffer captured into the
    # cuda graph. ``clean_and_force_maintain_logits_decode_triton`` takes
    # only ke (no cu_ks tensor) — saves the per-call zeros_like alloc.
    #
    # Speed (test_dg_decode.py speed_compare, schedule precomputed):
    #   eager full orch: ~8% slower than triton stage-2
    #   graph-replay:    0.74-0.98× of triton (worst at B=1 short ctx;
    #                    near-tied at B=32 ctx≥64K, the production sweet spot)
    # Trade: simpler maintenance (DG vs hand-tuned triton kernel), at
    # ~5-10% decode-stage cost in the favorable shapes; weigh vs e2e.
    num_pool_blocks_per_req = (context_lens + k_block_size - 1) // k_block_size
    pool_k_view = pool_k_pages.view(
        pool_k_pages.shape[0],
        pool_page_size,
        1,
        q_fp8.shape[-1] + 4,
    )
    weights_2d = weights if weights.dim() == 2 else weights.view(-1, weights.shape[-1])
    # Max pool blocks across the batch, derived from the main-KV table capacity:
    # max_seq_len tokens / k_block_size, rounded up. Tighter than reading from
    # pool_page_tables.shape (which carries the pool allocator's outer padding).
    max_pool_seq_len = (max_seq_len + k_block_size - 1) // k_block_size
    # DeepGEMM release-0426 requires context_lens of shape [batch_size, next_n];
    # hisa paged decode uses next_n=1, so unsqueeze the 1D pool-block count.
    num_pool_blocks_per_req_2d = (
        num_pool_blocks_per_req
        if num_pool_blocks_per_req.dim() == 2
        else num_pool_blocks_per_req.unsqueeze(-1)
    )
    # NOTE: clean_logits=False because clean_logits=True routes to the SM90
    # smxx_clean_logits path which asserts 1D context_lens (attention.hpp:381
    # "not is_context_lens_2d") and is incompatible with the 2D requirement
    # at :352. The clean_and_force_maintain triton kernel below replaces it
    # — it writes -inf at out-of-range positions AND the +inf sentinels.
    block_k_indexer_score = deep_gemm.fp8_paged_mqa_logits(
        q_fp8,  # [B, 1, H, D]
        pool_k_view,  # [N_pp, pool_page_size, 1, D+4]
        weights_2d,  # [B, H]
        num_pool_blocks_per_req_2d,  # [B, 1] i32 — pool-block "context_lens"
        pool_page_tables,  # [B, max_pp] i32
        schedule_metadata,
        max_pool_seq_len,
        clean_logits=False,
    )  # [B*1, max_pool_seq_len] f32 (raw, must be cleaned below)
    clean_and_force_maintain_logits_decode_triton(
        block_k_indexer_score,
        num_pool_blocks_per_req,
    )
    block_k_indexer_score = block_k_indexer_score.unsqueeze(
        1
    )  # [B, 1, max_pool_seq_len]

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


def fp8_native_hierarchy_mqa_logits(
    q_fp8: torch.Tensor,  # [seq, H, D] fp8
    kv: tuple[
        torch.Tensor, torch.Tensor
    ],  # (k_fp8 [N, D] fp8, k_scale [N, 4] uint8 OR [N] f32)
    weights: torch.Tensor,  # [seq, H] f32
    cu_seqlen_ks: torch.Tensor,  # [seq] i32
    cu_seqlen_ke: torch.Tensor,  # [seq] i32
    k_block_size: int,
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed-stage ragged prefill hierarchy MQA. Stages 1-2 are tilelang
    at all K (per A/B in test_grouped_mean_pool / test_stage2_ab); stage 4
    is split by K (test_stage4_full_ab at sq=8192, skv∈{8K..64K}).

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
        assert (
            k_scales.shape[1] == 1
        ), f"k_scales should be [N] or [N, 1], got {k_scales.shape}"
        k_scales = k_scales.squeeze(1)

    # 1) Mean-pool ragged K → tilelang. Bench (test_grouped_mean_pool.py) shows
    # tilelang stage 1 is 1.8x faster than the triton equivalent at all K
    # (8..128) due to lower per-launch Python overhead (~9μs vs triton's
    # ~24μs). The vanilla tilelang kernel has a boundary OOB at K<block_N=64,
    # so we route those to the bounds-safe grouped variant.
    if k_block_size < 64:
        blocked_k_fp8, blocked_k_scale = (
            fp8_native_block_mean_pooling_grouped_interface(
                k_fp8,
                k_scales,
                k_block_size,
            )
        )
    else:
        blocked_k_fp8, blocked_k_scale = fp8_native_block_mean_pooling_interface(
            k_fp8,
            k_scales,
            k_block_size,
        )

    # 2) Block-MQA on blocked_k → DeepGEMM ``fp8_mqa_logits``. Per-stage A/B
    # (bench_pool_mqa_deepgemm_vs_tilelang.py) shows DeepGEMM beats tilelang
    # 2.0-2.3x at sq=8192, ctx=128K across all K∈{16,32,64,128} including
    # the .contiguous() copy (e.g. K=16: 2609μs → 1277μs). DeepSeek tuned this
    # exact GEMM shape for the original NSA indexer; pooled K reuses the same
    # ``Q [seq,H,D] @ K^T [N,D]`` path with weights+ReLU+sum-over-H fused in.
    # ``clean_logits=True`` writes -inf outside [ks, ke), matching HISA's
    # clean_logits_(). force_maintain (+inf at ks and ke-1) is not in DG, so
    # we run a tiny stride-aware triton post-pass.
    #
    # Quirk: DG output has SM-aligned row stride padding (e.g. n_blocks=8192 →
    # stride=8448). We slice [:, :n_blocks] as a view (no copy) — fast_topk_v2
    # and torch.topk both read stride(0) correctly; the post-pass kernel is
    # also stride-aware. Avoids a ~165μs .contiguous() copy at K=16.
    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
    n_blocks = blocked_k_fp8.shape[0]
    block_k_indexer_score = deep_gemm.fp8_mqa_logits(
        q_fp8,
        (blocked_k_fp8, blocked_k_scale),
        weights,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
        clean_logits=True,
    )[:, :n_blocks]
    force_maintain_logits_triton(
        block_k_indexer_score,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )

    # 3) Top-k over pool blocks. fast_topk_runtime at K∈{16,32} (1.45-1.83×),
    # torch.topk(bf16) at K∈{64,128} (fast_topk's setup tax > sort savings
    # at small topk). Helper above hides the dispatch and the L>=topk clamp.
    topk_block_indices = _stage3_topk_prefill(
        block_k_indexer_score, block_topk
    )  # [seq, min(block_topk, L)] int32 (fast_topk) or int64 (torch.topk)

    # 4) Sparse-MQA on raw K — all K via triton ``block_sparse_mqa_triton``:
    #   K∈{16,32}:  persistent (GEMM_TILE=256 GROUP_SIZE=GEMM_TILE/K K_CHUNKS=16 w8)
    #   K=64:       grouped (GEMM_TILE=256 GROUP_SIZE=4)
    #   K=128:      persistent (GEMM_TILE=128 GROUP_SIZE=1 K_CHUNKS=64 w4 s3)
    #               — wins 1.35-1.38× over tilelang at sq=8192 skv∈{16K..128K}.
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
