from dataclasses import dataclass, field
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


@dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA
    """

    @dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]

        extra_page_block_size: Optional[int] = None
        extra_topk: Optional[int] = None

    have_initialized: bool = False
    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = (
        None  # (num_sm_parts, DecodingSchedMetaSize/sizeof(int)), dtype torch.int32.
    )
    num_splits: Optional[torch.Tensor] = None  # (batch_size + 1), dtype torch.int32.


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty FlashMLASchedMeta object. The actual scheduling metadata
    will be generated during the first invocation of flash_mla_with_kvcache.

    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_q_tokens_per_head_k: Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
        num_heads_k: The number of k heads.
        num_heads_q: The number of q heads. This argument is optional when sparse attention is not enabled
        is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
        topk: If not None, sparse attention will be enabled.

    Returns:
        A tuple of (FlashMLASchedMeta, None). Only the first element is useful.
    """
    return FlashMLASchedMeta(), None


def get_mla_decoding_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get metadata for dense FP8 decoding.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, DecodingSchedMetaSize/sizeof(int)), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error
    return torch.ops.sgl_kernel.get_mla_decoding_metadata_dense_fp8.default(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FlashMLA attention with KV cache.

    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: FlashMLASchedMeta object returned by get_mla_metadata.
        num_splits: Must be None (for API compatibility).
        softmax_scale: float. The scale of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        descale_q: (batch_size), torch.float32. Descaling factors for Q.
        descale_k: (batch_size), torch.float32. Descaling factors for K.
        is_fp8_kvcache: bool. Whether the k_cache is in fp8 format.
        indices: (batch_size, seq_len_q, topk), torch.int32. For sparse attention.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error

    sched_meta = tile_scheduler_metadata
    assert isinstance(
        sched_meta, FlashMLASchedMeta
    ), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None, use FlashMLASchedMeta instead"

    topk = indices.shape[-1] if indices is not None else None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not sched_meta.have_initialized:
        # Initialize the tile scheduler metadata during the first invocation
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
            extra_page_block_size=None,
            extra_topk=None,
        )
    else:
        # Check consistency with sched_meta
        assert sched_meta.config is not None
        helper_msg = " Input arguments are inconsistent with sched_meta."
        assert sched_meta.config.b == q.shape[0], "batch_size mismatch." + helper_msg
        assert sched_meta.config.s_q == q.shape[1], "seq_len_q mismatch." + helper_msg
        assert sched_meta.config.h_q == q.shape[2], "num_heads_q mismatch." + helper_msg
        assert sched_meta.config.page_block_size == k_cache.shape[1], (
            "page_block_size mismatch." + helper_msg
        )
        assert sched_meta.config.h_k == k_cache.shape[2], (
            "num_heads_k mismatch." + helper_msg
        )
        assert sched_meta.config.causal == causal, "causal mismatch." + helper_msg
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, (
            "is_fp8_kvcache mismatch." + helper_msg
        )
        assert sched_meta.config.topk == topk, "topk mismatch." + helper_msg

    # Call the underlying kernel
    if indices is None and q.element_size() == 1:
        # FP8 dense decode path
        out, softmax_lse = torch.ops.sgl_kernel.fwd_kvcache_mla_fp8.default(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            descale_q,
            descale_k,
        )
    else:
        # BF16/FP16 dense or sparse decode path
        out, softmax_lse = torch.ops.sgl_kernel.fwd_kvcache_mla.default(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            is_fp8_kvcache,
            indices,
        )

    return out, softmax_lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32.
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512

    Returns:
        (output, max_logits, lse)
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error
    results = torch.ops.sgl_kernel.sparse_prefill_fwd.default(
        q, kv, indices, sm_scale, d_v
    )
    return results
