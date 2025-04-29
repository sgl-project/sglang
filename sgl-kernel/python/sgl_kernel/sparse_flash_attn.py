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
    r"""
    Compute attention with vertical and slash sparsity patterns.

    This function implements sparse attention using the "vertical" and "slash" patterns as described in
    [MInference 1.0](https://arxiv.org/abs/2407.02490), optimized for long-context LLM pre-filling.
    The interface closely matches `flash_attn_func`, with four additional arguments to specify sparsity patterns:
    `block_count` and `block_offset` for slash sparsity; `column_count` and `column_index` for vertical sparsity.
    For more details, see Appendix C.4.2 of the paper.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (batch_size, seqlen, nheads, headdim).
    k : torch.Tensor
        Key tensor of shape (batch_size, seqlen, nheads_k, headdim).
    v : torch.Tensor
        Value tensor of shape (batch_size, seqlen, nheads_k, headdim).
    block_count : torch.Tensor
        For slash sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M)).
        Specifies the number of nonzero blocks per row block.
    block_offset : torch.Tensor
        For slash sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M), NNZ_S).
        Specifies the block indices of nonzero blocks for each row block.
    column_count : torch.Tensor
        For vertical sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M)).
        Specifies the number of nonzero columns per block.
    column_index : torch.Tensor
        For vertical sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M), NNZ_V).
        Specifies the column indices of nonzero columns for each block.
    dropout_p : float
        Dropout probability for attention weights.
    softmax_scale : float, optional
        Scaling factor applied to QK^T before softmax, defaults to 1 / sqrt(headdim).
    causal : bool, optional
        Whether to apply a causal mask (for autoregressive modeling). Default: False.
    alibi_slopes : torch.Tensor, optional
        (nheads,) or (batch_size, nheads), fp32. Adds a bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|) to attention scores.
    deterministic : bool, optional
        Use deterministic backward pass. Slightly slower and uses more memory. Forward is always deterministic.
    return_attn_probs : bool, optional
        If True, returns attention probabilities for testing (not guaranteed to be correctly scaled).

    Returns
    -------
    out : torch.Tensor
        Output tensor of shape (batch_size, seqlen, nheads, headdim).
    softmax_lse : torch.Tensor, optional
        (batch_size, nheads, seqlen). The logsumexp of each row of the attention score matrix QK^T * scaling.
        Returned if `return_attn_probs` or similar debugging flag is set.

    Notes
    -----
    - The vertical and slash sparsity patterns are designed to reduce computation and memory usage in long-context LLMs,
      as detailed in MInference 1.0 ([arXiv:2407.02490](https://arxiv.org/abs/2407.02490)), Appendix C.4.2.
    - The function supports dropout, ALiBi positional bias, and both deterministic and non-deterministic backward passes.
    - The sparsity patterns are specified using the provided block-related arguments, enabling dynamic and efficient
      sparse attention calculation on GPU.
    - This function is intended for expert users familiar with advanced attention mechanisms and the MInference approach.

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
    r"""
    Compute attention with vertical and slash sparsity patterns (variable-length sequences).

    This function implements sparse attention using "vertical" and "slash" patterns for batches of
    variable-length sequences, as described in [MInference 1.0](https://arxiv.org/abs/2407.02490), Appendix C.4.2.
    The interface closely follows `flash_attn_varlen_func`, with four additional arguments to specify sparsity:
    `block_count` and `block_offset` for slash sparsity; `column_count` and `column_index` for vertical sparsity.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (total_q, nheads, headdim), where total_q is the sum of all query token counts in the batch.
    k : torch.Tensor
        Key tensor of shape (total_k, nheads_k, headdim), where total_k is the sum of all key token counts in the batch.
    v : torch.Tensor
        Value tensor of shape (total_k, nheads_k, headdim).
    block_count : torch.Tensor
        For slash sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M)).
        Number of nonzero blocks per row block.
    block_offset : torch.Tensor
        For slash sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M), NNZ_S).
        Block indices of nonzero blocks for each row block.
    column_count : torch.Tensor
        For vertical sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M)).
        Number of nonzero columns per block.
    column_index : torch.Tensor
        For vertical sparsity: (batch_size, nheads, ceil(seqlen / BLOCK_M), NNZ_V).
        Column indices of nonzero columns for each block.
    cu_seqlens_q : torch.Tensor
        (batch_size + 1,), dtype torch.int32. Cumulative sequence lengths for queries, used to index into `q`.
    cu_seqlens_k : torch.Tensor
        (batch_size + 1,), dtype torch.int32. Cumulative sequence lengths for keys, used to index into `k`/`v`.
    max_seqlen_q : int
        Maximum query sequence length in the batch.
    max_seqlen_k : int
        Maximum key sequence length in the batch.
    dropout_p : float
        Dropout probability for attention weights.
    softmax_scale : float, optional
        Scaling factor applied to QK^T before softmax. Defaults to 1 / sqrt(headdim).
    causal : bool, optional
        Whether to apply a causal mask (for autoregressive modeling). Default: False.
    softcap : float, optional
        If > 0, activates softcapping attention.
    alibi_slopes : torch.Tensor, optional
        (nheads,) or (batch_size, nheads), fp32. Adds a bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|) to attention scores.
    deterministic : bool, optional
        Use deterministic backward pass (slower, more memory). Forward pass is always deterministic.
    return_attn_probs : bool, optional
        If True, returns attention probabilities for testing (not guaranteed to be correctly scaled).

    Returns
    -------
    out : torch.Tensor
        Output tensor of shape (total_q, nheads, headdim).
    softmax_lse : torch.Tensor, optional
        (nheads, total_q_seqlen). The logsumexp of each row of the attention score matrix QK^T * scaling.
        Returned if `return_attn_probs` or similar debugging flag is set.

    Notes
    -----
    - Supports both "vertical" and "slash" sparsity patterns, as detailed in MInference 1.0 ([arXiv:2407.02490](https://arxiv.org/abs/2407.02490)), Appendix C.4.2.
    - Handles batches of variable-length sequences using `cu_seqlens_q` and `cu_seqlens_k`.
    - Sparsity patterns are provided via block-related arguments for efficient GPU-based inference.
    - Dropout, ALiBi positional bias, causal masking, and optional softcapping are supported.
    - Intended for use in large-scale LLM prefill acceleration with MInference.

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
