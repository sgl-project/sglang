from __future__ import annotations

import torch
import triton
import triton.language as tl

_TOP_K = 32


@triton.jit
def _top32_local_kernel(
    probs_ptr,
    local_values_ptr,
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
    top_values = tl.topk(values, TOP_K, dim=0)

    top_values = tl.sort(top_values, dim=0, descending=True)
    output_offsets = (row * num_chunks + chunk) * TOP_K + tl.arange(0, TOP_K)
    tl.store(local_values_ptr + output_offsets, top_values)


@triton.jit
def _top32_merge_kernel(
    local_values_ptr,
    top_values_ptr,
    num_chunks: tl.constexpr,
    TOP_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = (row * num_chunks) * TOP_K + tl.arange(0, TOP_K)
    top_values = tl.load(local_values_ptr + offsets)
    top_values = tl.topk(top_values, TOP_K, dim=0)

    for chunk in range(1, num_chunks):
        top_values = tl.bitonic_merge(top_values)
        offsets = (row * num_chunks + chunk) * TOP_K + tl.arange(0, TOP_K)
        values = tl.load(local_values_ptr + offsets)
        top_values = tl.maximum(top_values, tl.topk(values, TOP_K, dim=0))

    top_values = tl.sort(top_values, dim=0, descending=True)
    positions = tl.arange(0, TOP_K)
    output_offsets = row * TOP_K + positions
    tl.store(top_values_ptr + output_offsets, top_values)


def top_p_select_hierarchical_triton(
    probs: torch.Tensor,
    *,
    chunk_size: int = 1024,
    num_warps: int = 4,
):
    assert probs.ndim == 2 and probs.is_contiguous()
    assert probs.dtype == torch.float32 and probs.is_cuda
    assert chunk_size in (512, 1024, 2048)

    rows, vocab_size = probs.shape
    num_chunks = triton.cdiv(vocab_size, chunk_size)
    local_values = torch.empty(
        (rows, num_chunks, _TOP_K),
        dtype=torch.float32,
        device=probs.device,
    )
    top_values = torch.empty(
        (rows, _TOP_K),
        dtype=torch.float32,
        device=probs.device,
    )

    _top32_local_kernel[(rows, num_chunks)](
        probs,
        local_values,
        vocab_size,
        num_chunks=num_chunks,
        TOP_K=_TOP_K,
        CHUNK_SIZE=chunk_size,
        num_warps=num_warps,
    )
    _top32_merge_kernel[(rows,)](
        local_values,
        top_values,
        num_chunks=num_chunks,
        TOP_K=_TOP_K,
        num_warps=num_warps,
    )
    return top_values


__all__ = ["top_p_select_hierarchical_triton"]
