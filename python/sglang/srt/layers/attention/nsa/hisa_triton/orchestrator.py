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
    fp8_native_paged_mean_pooling_tail_only_v3_interface,
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
    """All-triton ragged prefill hierarchy MQA. Mirrors
    ``fp8_native_hierarchy_mqa_logits`` (tilelang) but uses triton kernels
    that handle ``k_block_size in {8, 16, 32, 64, 128}`` correctly.

    Used by ``_get_topk_ragged`` whenever ``k_block_size < 64``: tilelang's
    ``fp8_native_block_mean_pooling`` does ``T.copy(K[s:s+block_N=64], ...)``
    even when the pool block is K<64 wide, which OOB-reads up to ``block_N - K``
    rows past the K tensor at boundary CTAs (last few pool blocks). Most of
    the time those reads land in mapped pages, but under sustained prefill
    (long contexts + chunked prefill at K=16) they eventually hit an
    unmapped page → Xid 13 → silent server death.

    Flow (matches v1 baseline, no cache):
      1) ragged mean-pool — ``block_mean_pooling_triton`` (SK1/SK6 grouped
         when K<64; legacy when K>=64). Loads block_N=64 tokens once and
         reshapes to [G, K, D] for per-K mean — no over-read.
      2) ragged block-MQA — ``ragged_pool_mqa_triton`` (new). Fuses
         clean_logits + force_maintain into the GEMM post-process.
      3) ``torch.topk`` on bf16 logits.
      4) ragged sparse-MQA — ``block_sparse_mqa_triton`` (SK4/SK11 grouped
         GEMM_TILE=256 when K<128).

    Returns ``(block_sparse_logits[seq, topk*K], topk_block_indices[seq, topk] int64)``
    matching the tilelang wrapper's shapes.
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

    # 1) Mean-pool ragged K → [num_pool, D] fp8 + [num_pool] f32 scale.
    blocked_k_fp8, blocked_k_scale = block_mean_pooling_triton(
        k_fp8=k_fp8, k_scale=k_scales, k_block_size=k_block_size,
    )

    # 2) Block-MQA on blocked_k → [seq, num_pool] f32 (with -inf/+inf masks).
    # Pass raw token-space cu_seqlen + k_block_size; the kernel divides
    # internally so we avoid 3-4 host-side PyTorch elementwise launches
    # (floor_divide × 2, add, optional .to(int32)) that otherwise add
    # ~30-50μs to the orchestrator at small sq.
    block_k_indexer_score = ragged_pool_mqa_triton(
        q_fp8=q_fp8,
        blocked_k_fp8=blocked_k_fp8,
        blocked_k_scale=blocked_k_scale,
        weights=weights,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        k_block_size=k_block_size,
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

    # 4) Sparse-MQA on raw K. block_sparse_mqa_triton's grouped path (K<128)
    # requires ``topk % (256 // K) == 0``. In production, block_topk=512
    # and num_pool >= 512 → topk=512 satisfies the divisibility for all
    # K in {8,16,32,64} (GROUP_SIZE ∈ {32,16,8,4}). On short warmup
    # inputs num_pool < block_topk and topk = num_pool may not align;
    # pad with -1 (kernel masks via k_rows < 0 → -inf logits) and slice
    # back after the call.
    group_size_sparse = 256 // k_block_size if k_block_size < 128 else 1
    topk_padded = (
        (topk_actual + group_size_sparse - 1) // group_size_sparse
    ) * group_size_sparse
    if topk_padded > topk_actual:
        pad_n = topk_padded - topk_actual
        pad = torch.full(
            (topk_block_indices.shape[0], pad_n),
            fill_value=-1,
            device=topk_block_indices.device,
            dtype=topk_block_indices.dtype,
        )
        topk_for_kernel = torch.cat([topk_block_indices, pad], dim=-1)
    else:
        topk_for_kernel = topk_block_indices

    block_sparse_logits = block_sparse_mqa_triton(
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        k_scale=k_scales,
        topk_block_index=topk_for_kernel,
        kv_block_size=k_block_size,
        weights=weights,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
    )  # [seq, topk_padded * k_block_size] f32

    if topk_padded > topk_actual:
        block_sparse_logits = block_sparse_logits[
            ..., : topk_actual * k_block_size
        ].contiguous()

    return block_sparse_logits, topk_block_indices
