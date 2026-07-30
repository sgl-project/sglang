from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# Sparse attention utils
def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    block_offset = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    column_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    column_index = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    torch.ops.sgl_kernel.convert_vertical_slash_indexes.default(
        block_count,
        block_offset,
        column_count,
        column_index,
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        context_size,
        block_size_M,
        block_size_N,
        causal,
    )
    return block_count, block_offset, column_count, column_index


def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    # [N_HEADS] : different head use different number of indices
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    block_offset = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    column_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    column_index = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    torch.ops.sgl_kernel.convert_vertical_slash_indexes_mergehead.default(
        block_count,
        block_offset,
        column_count,
        column_index,
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        vertical_indices_count,
        slash_indices_count,
        context_size,
        block_size_M,
        block_size_N,
        causal,
    )
    return block_count, block_offset, column_count, column_index


def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = torch.ops.sgl_kernel.fwd_sparse.default(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_varlen_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_varlen_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = torch.ops.sgl_kernel.varlen_fwd_sparse.default(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out
