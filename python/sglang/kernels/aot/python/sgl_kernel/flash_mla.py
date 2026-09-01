import dataclasses
from typing import Optional, Tuple

import torch

try:
    from sgl_kernel import flashmla_ops  # triggers TORCH extension registration
except Exception as _e:
    _flashmla_import_error = _e
else:
    _flashmla_import_error = None

_IMPORT_ERROR = ImportError(
    "Failed to load sgl_kernel.flashmla_ops extension. Ensure CUDA Driver >= 12.4"
)


@dataclasses.dataclass
class FlashMLASchedMeta:
    """Tile scheduler metadata for the newer FlashMLA Python API."""

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]
        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False
    config: Optional[Config] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def get_mla_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_tokens_per_head_k: Optional[int] = None,
    num_heads_k: Optional[int] = None,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_q_tokens_per_head_k: Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
        num_heads_k: The number of k heads.
        num_heads_q: The number of q heads. This argument is optional when sparse attention is not enabled
        is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
        topk: If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error

    if cache_seqlens is None:
        return FlashMLASchedMeta(), None

    assert num_q_tokens_per_head_k is not None
    assert num_heads_k is not None

    if is_fp8_kvcache and topk is None:
        return torch.ops.sgl_kernel.get_mla_decoding_metadata_dense_fp8.default(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
        )
    return torch.ops.sgl_kernel.get_mla_decoding_metadata.default(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor | FlashMLASchedMeta,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        descale_q: (batch_size), torch.float32. Descaling factors for Q, used for fp8 quantization.
        descale_k: (batch_size), torch.float32. Descaling factors for K, used for fp8 quantization.
        is_fp8_kvcache: bool. Whether the k_cache and v_cache are in fp8 format. For the format of FP8 KV cache, please refer to README.md
        indices: (batch_size, seq_len_q, topk), torch.int32. If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to. Invalid indices should be set to -1 or numbers >= total_seq_len_kv. For details about how to set up `indices`, please refer to README.md.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        return _flash_mla_with_kvcache_sched_meta(
            q=q,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            sched_meta=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
        )

    assert num_splits is not None
    assert block_table is not None
    assert cache_seqlens is not None
    assert attn_sink is None
    assert extra_k_cache is None
    assert extra_indices_in_kvcache is None
    assert topk_length is None
    assert extra_topk_length is None
    if indices is not None:
        assert causal == False, "causal must be `false` if sparse attention is enabled."
    assert (descale_q is None) == (
        descale_k is None
    ), "descale_q and descale_k should be both None or both not None"

    if indices is None and q.element_size() == 1:
        out, softmax_lse = torch.ops.sgl_kernel.fwd_kvcache_mla_fp8.default(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            tile_scheduler_metadata,
            num_splits,
            descale_q,
            descale_k,
        )
    else:
        out, softmax_lse = torch.ops.sgl_kernel.fwd_kvcache_mla.default(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            tile_scheduler_metadata,
            num_splits,
            is_fp8_kvcache,
            indices,
            attn_sink,
            extra_k_cache,
            extra_indices_in_kvcache,
            topk_length,
            extra_topk_length,
        )
    return out, softmax_lse


def _flash_mla_with_kvcache_sched_meta(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    sched_meta: FlashMLASchedMeta,
    num_splits: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    is_fp8_kvcache: bool,
    indices: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert num_splits is None, "num_splits must be None with FlashMLASchedMeta"

    topk = indices.shape[-1] if indices is not None else None
    extra_page_block_size = (
        extra_k_cache.shape[1] if extra_k_cache is not None else None
    )
    extra_topk = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )

    if not sched_meta.have_initialized:
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            b=q.shape[0],
            s_q=q.shape[1],
            h_q=q.shape[2],
            page_block_size=k_cache.shape[1],
            h_k=k_cache.shape[2],
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            topk=topk,
            extra_page_block_size=extra_page_block_size,
            extra_topk=extra_topk,
        )
    else:
        helper_msg = (
            " Input arguments are inconsistent with FlashMLASchedMeta. Reuse a "
            "scheduler only for matching tensor shapes and sparse settings."
        )
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], helper_msg
        assert sched_meta.config.s_q == q.shape[1], helper_msg
        assert sched_meta.config.h_q == q.shape[2], helper_msg
        assert sched_meta.config.page_block_size == k_cache.shape[1], helper_msg
        assert sched_meta.config.h_k == k_cache.shape[2], helper_msg
        assert sched_meta.config.causal == causal, helper_msg
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, helper_msg
        assert sched_meta.config.topk == topk, helper_msg
        assert (
            sched_meta.config.extra_page_block_size == extra_page_block_size
        ), helper_msg
        assert sched_meta.config.extra_topk == extra_topk, helper_msg

    if topk is not None:
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True for sparse attention"
        out, lse, new_tile_scheduler_metadata, new_num_splits = (
            torch.ops.sgl_kernel.sparse_decode_fwd.default(
                q,
                k_cache,
                indices,
                topk_length,
                attn_sink,
                sched_meta.tile_scheduler_metadata,
                sched_meta.num_splits,
                extra_k_cache,
                extra_indices_in_kvcache,
                extra_topk_length,
                head_dim_v,
                softmax_scale,
            )
        )
    else:
        assert block_table is not None and cache_seqlens is not None
        assert attn_sink is None
        assert extra_k_cache is None
        assert extra_indices_in_kvcache is None
        assert topk_length is None
        assert extra_topk_length is None
        out, lse, new_tile_scheduler_metadata, new_num_splits = (
            torch.ops.sgl_kernel.dense_decode_fwd.default(
                q,
                k_cache,
                head_dim_v,
                cache_seqlens,
                block_table,
                softmax_scale,
                causal,
                sched_meta.tile_scheduler_metadata,
                sched_meta.num_splits,
            )
        )

    sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
    sched_meta.num_splits = new_num_splits
    return out, lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512

    Returns:
        (output, max_logits, lse)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error

    results = torch.ops.sgl_kernel.sparse_prefill_fwd.default(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    return results
