from typing import Optional, Tuple

import torch

try:
    from . import flashmla_ops  # triggers TORCH extension registration
except Exception as _e:
    _flashmla_import_error = _e
else:
    _flashmla_import_error = None

_IMPORT_ERROR = ImportError(
    "Failed to load sgl_kernel.flashmla_ops extension. Ensure CUDA Driver >= 12.4"
)


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
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
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
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
        is_fp8_kvcache: bool. Whether the k_cache and v_cache are in fp8 format. For the format of FP8 KV cache, please refer to README.md
        indices: (batch_size, seq_len_q, topk), torch.int32. If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to. Invalid indices should be set to -1 or numbers >= total_seq_len_kv. For details about how to set up `indices`, please refer to README.md.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        assert causal == False, "causal must be `false` if sparse attention is enabled."
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
    results = torch.ops.sgl_kernel.sparse_prefill_fwd.default(
        q, kv, indices, sm_scale, d_v
    )
    return results
