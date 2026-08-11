from __future__ import annotations

import torch
import triton
import triton.language as tl

_TOP_K = 32


@triton.jit
def _top32_local_sum_kernel(
    probs_ptr,
    local_values_ptr,
    partial_sums_ptr,
    vocab_size,
    num_chunks: tl.constexpr,
    TOP_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    row_base = row * vocab_size
    chunk_base = chunk * CHUNK_SIZE
    offsets = chunk_base + tl.arange(0, CHUNK_SIZE)
    valid = offsets < vocab_size
    values = tl.load(
        probs_ptr + row_base + offsets,
        mask=valid,
        other=float("-inf"),
    )
    row_sum = tl.sum(tl.where(valid, values, 0.0), axis=0)
    top_values = tl.topk(values, TOP_K, dim=0)

    top_values = tl.sort(top_values, dim=0, descending=True)
    output_offsets = (row * num_chunks + chunk) * TOP_K + tl.arange(0, TOP_K)
    tl.store(local_values_ptr + output_offsets, top_values)
    tl.store(partial_sums_ptr + row * num_chunks + chunk, row_sum)


@triton.jit
def _top32_merge_pivot_kernel(
    local_values_ptr,
    partial_sums_ptr,
    top_ps_ptr,
    top_values_ptr,
    row_sums_ptr,
    pivots_ptr,
    normalizers_ptr,
    fast_path_ptr,
    fallback_ptr,
    num_chunks: tl.constexpr,
    TOP_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = (row * num_chunks) * TOP_K + tl.arange(0, TOP_K)
    top_values = tl.load(local_values_ptr + offsets)
    top_values = tl.topk(top_values, TOP_K, dim=0)
    row_sum = tl.load(partial_sums_ptr + row * num_chunks)

    for chunk in range(1, num_chunks):
        top_values = tl.bitonic_merge(top_values)
        offsets = (row * num_chunks + chunk) * TOP_K + tl.arange(0, TOP_K)
        values = tl.load(local_values_ptr + offsets)
        top_values = tl.maximum(top_values, tl.topk(values, TOP_K, dim=0))
        row_sum += tl.load(partial_sums_ptr + row * num_chunks + chunk)

    top_values = tl.sort(top_values, dim=0, descending=True)
    top_p = tl.load(top_ps_ptr + row)
    budget = row_sum - (1.0 - top_p)
    prefix_mass = tl.cumsum(top_values, axis=0) - top_values
    within = prefix_mass <= budget
    position = tl.maximum(tl.sum(within.to(tl.int32), axis=0) - 1, 0)
    positions = tl.arange(0, TOP_K)
    pivot = tl.sum(tl.where(positions == position, top_values, 0.0), axis=0)
    normalizer = tl.sum(tl.where(top_values >= pivot, top_values, 0.0), axis=0)
    last_value = tl.sum(tl.where(positions == TOP_K - 1, top_values, 0.0), axis=0)
    last_within = (
        tl.sum(
            tl.where(positions == TOP_K - 1, within.to(tl.int32), 0),
            axis=0,
        )
        != 0
    )
    fast_path = (~last_within) & (last_value < pivot) & (normalizer > 0)

    output_offsets = row * TOP_K + positions
    tl.store(top_values_ptr + output_offsets, top_values)
    tl.store(row_sums_ptr + row, row_sum)
    tl.store(pivots_ptr + row, pivot)
    tl.store(normalizers_ptr + row, normalizer)
    tl.store(fast_path_ptr + row, fast_path)
    tl.atomic_or(fallback_ptr, (~fast_path).to(tl.int32))


def top_p_select_hierarchical_triton(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    chunk_size: int = 1024,
    num_warps: int = 4,
):
    assert probs.ndim == 2 and probs.is_contiguous()
    assert probs.dtype == torch.float32 and probs.is_cuda
    assert top_ps.shape == (probs.shape[0],)
    assert top_ps.dtype == torch.float32 and top_ps.is_cuda
    assert chunk_size in (512, 1024, 2048)

    rows, vocab_size = probs.shape
    num_chunks = triton.cdiv(vocab_size, chunk_size)
    local_values = torch.empty(
        (rows, num_chunks, _TOP_K),
        dtype=torch.float32,
        device=probs.device,
    )
    partial_sums = torch.empty(
        (rows, num_chunks),
        dtype=torch.float32,
        device=probs.device,
    )
    top_values = torch.empty(
        (rows, _TOP_K),
        dtype=torch.float32,
        device=probs.device,
    )
    row_sums = torch.empty(rows, dtype=torch.float32, device=probs.device)
    pivots = torch.empty_like(row_sums)
    normalizers = torch.empty_like(row_sums)
    fast_path = torch.empty(rows, dtype=torch.bool, device=probs.device)
    fallback = torch.zeros((), dtype=torch.int32, device=probs.device)

    _top32_local_sum_kernel[(rows, num_chunks)](
        probs,
        local_values,
        partial_sums,
        vocab_size,
        num_chunks=num_chunks,
        TOP_K=_TOP_K,
        CHUNK_SIZE=chunk_size,
        num_warps=num_warps,
    )
    _top32_merge_pivot_kernel[(rows,)](
        local_values,
        partial_sums,
        top_ps,
        top_values,
        row_sums,
        pivots,
        normalizers,
        fast_path,
        fallback,
        num_chunks=num_chunks,
        TOP_K=_TOP_K,
        num_warps=num_warps,
    )
    return top_values, row_sums, pivots, normalizers, fast_path, fallback


__all__ = ["top_p_select_hierarchical_triton"]
