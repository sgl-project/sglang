"""Expand NPU pooled attention indices and append uncompressed tails."""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_pool_topk_and_append_tail_batched_npu_kernel(
    pool_indices_ptr,
    seq_lens_per_token_ptr,
    out_ptr,
    stride_pi0: tl.constexpr,
    stride_pi1: tl.constexpr,
    stride_o0: tl.constexpr,
    stride_o1: tl.constexpr,
    n_real,
    N_POOL_TOPK: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    TAIL_LEN: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Expand pool-level topk indices to token-level and append tail tokens.

    Each program processes BLOCK_M rows at a time with a stride of
    NUM_CORES * BLOCK_M. Phase 1 expands a row block in parallel; phase 2
    preserves the per-row layout while unrolling with static_range.
    """
    pid = tl.program_id(0)

    for r0 in range(pid * BLOCK_M, n_real, NUM_CORES * BLOCK_M):
        row_offs = r0 + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        row_mask = row_offs < n_real

        # Phase 1: expand a BLOCK_M batch from pool indices to token indices.
        p_offs = tl.arange(0, N_POOL_TOPK)
        pid_val = tl.load(
            pool_indices_ptr
            + row_offs[:, None] * stride_pi0
            + p_offs[None, :] * stride_pi1,
            mask=row_mask[:, None],
            other=0,
        ).to(tl.int32)  # [BLOCK_M, N_POOL_TOPK]

        base = pid_val * POOL_SIZE  # [BLOCK_M, N_POOL_TOPK]
        o_offs = tl.arange(0, POOL_SIZE)  # [POOL_SIZE]

        # [BLOCK_M, P, N] -> [BLOCK_M, N, P] -> [BLOCK_M, N * P]
        tmp = o_offs[None, :, None] + base[:, None, :]  # [BLOCK_M, P, N]
        tmp = tl.trans(tmp, [0, 2, 1])  # [BLOCK_M, N, P]
        result = tl.reshape(tmp, (BLOCK_M, N_POOL_TOPK * POOL_SIZE))

        col_offs = tl.arange(0, N_POOL_TOPK * POOL_SIZE)
        tl.store(
            out_ptr + row_offs[:, None] * stride_o0 + col_offs[None, :] * stride_o1,
            result,
            mask=row_mask[:, None],
        )

        # Phase 2: append each row's uncompressed tail.
        for m in tl.static_range(BLOCK_M):
            r = r0 + m
            valid = r < n_real
            seq_len = tl.load(
                seq_lens_per_token_ptr + r,
                mask=valid,
                other=0,
            ).to(tl.int32)
            pool_len = seq_len // POOL_SIZE
            tail_start = pool_len * POOL_SIZE
            tail_count = seq_len - tail_start

            t_offs = tl.arange(0, TAIL_LEN)
            tail_vals = tl.where(t_offs < tail_count, tail_start + t_offs, -1)
            tl.store(
                out_ptr + r * stride_o0 + TOPK * stride_o1 + t_offs * stride_o1,
                tail_vals,
                mask=valid,
            )


def _expand_pool_topk_and_append_tail_batched_npu(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
) -> torch.Tensor:
    """Expand NPU pool-level top-k indices and append uncompressed tails.

    Args:
        pool_indices: Selected pool positions, shaped [>=n_real, n_pool_topk].
        seq_lens_per_token: Request length for each query, shaped [>=n_real].
        n_real: Number of valid query rows; later rows are masked.
        num_q_padded: Output row count; padded rows contain -1.
        pool_size: Number of tokens represented by one pool.
        index_topk: Token-level top-k, equal to n_pool_topk * pool_size.

    Returns:
        Token indices shaped [num_q_padded, index_topk + pool_size - 1].
    """
    from sglang.kernels.ops.kvcache.kpool_index_npu import (
        _default_triton_num_programs,
    )

    n_pool_topk = pool_indices.shape[1]
    tail_len = pool_size - 1
    out_cols = index_topk + tail_len
    # Align rows to eight int32 values (32 bytes).
    out_cols_pad = (out_cols + 7) // 8 * 8

    # Let the kernel mask rows beyond n_real so callers need not slice inputs.
    n_real = min(n_real, pool_indices.shape[0], seq_lens_per_token.shape[0])

    assert pool_indices.ndim == 2
    assert pool_indices.shape[0] >= n_real
    assert seq_lens_per_token.ndim == 1
    assert seq_lens_per_token.shape[0] >= n_real
    assert num_q_padded >= n_real
    assert pool_indices.dtype == torch.int32
    assert seq_lens_per_token.dtype == torch.int32
    assert n_pool_topk * pool_size == index_topk

    # Prefill padding rows with -1 and align the physical column count.
    out = torch.full(
        (num_q_padded, out_cols_pad), -1, dtype=torch.int32, device=pool_indices.device
    )

    assert out.stride(1) == 1

    if n_real > 0:
        num_cores = _default_triton_num_programs(pool_indices.device, n_real)
        _expand_pool_topk_and_append_tail_batched_npu_kernel[(num_cores,)](
            pool_indices,
            seq_lens_per_token,
            out,
            pool_indices.stride(0),
            pool_indices.stride(1),
            out.stride(0),
            out.stride(1),
            n_real,
            N_POOL_TOPK=n_pool_topk,
            POOL_SIZE=pool_size,
            TOPK=index_topk,
            TAIL_LEN=tail_len,
            NUM_CORES=num_cores,
            BLOCK_M=4,
        )

    return out[:, :out_cols]
