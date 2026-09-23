"""Adler-32 over PD transfer descriptors using Triton-Ascend.

The byte stream is ordered by descriptor, then by the supplied item indices.
Only the reduced checksum is copied to the host, not the KV cache itself.
"""

import torch
import triton
import triton.language as tl

_MOD_ADLER = 65521
_BLOCK_SIZE = 1024


@triton.jit
def _adler32_partial_kernel(
    base,
    indices,
    partials,
    stride: tl.constexpr,
    num_bytes,
    remaining_bytes,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_bytes
    rows = tl.load(indices + offsets // stride, mask=valid, other=0).to(tl.int64)
    ptr = base.to(tl.pointer_type(tl.uint8))
    values = tl.load(ptr + rows * stride + offsets % stride, valid, other=0).to(
        tl.int32
    )

    # For a stream of N bytes x[j]:
    # a = 1 + sum(x[j]); b = N + sum((N - j) * x[j]), modulo 65521.
    # Reduce each product before summation to keep the vector reduction in
    # int32: BLOCK * (65521 - 1) < 2**31, even for multi-GiB streams.
    weights = ((remaining_bytes - offsets) % 65521).to(tl.int32)
    a = tl.sum(values, 0) % 65521
    b = tl.sum((weights * values) % 65521, 0) % 65521
    tl.store(partials + block * 2, a)
    tl.store(partials + block * 2 + 1, b)


def adler32_strided_checksum(
    data_ptrs: list[int],
    strides: list[int],
    indices: list[torch.Tensor],
) -> int:
    """Match the CUDA strided checksum's ordered raw-byte semantics.

    Each descriptor addresses contiguous items at ``base + index * stride``.
    Pointer lifetimes and index bounds are owned by the KV pool, as on CUDA.
    This diagnostic path synchronizes when returning its Python integer.
    """
    if not (len(data_ptrs) == len(strides) == len(indices)):
        raise ValueError("data_ptrs, strides and indices must have the same length")
    if not indices:
        return 1
    device = indices[0].device
    if device.type != "npu":
        raise ValueError("NPU checksum requires NPU index tensors")

    sizes = []
    for base, stride, idx in zip(data_ptrs, strides, indices):
        if idx.device != device or idx.ndim != 1 or not idx.is_contiguous():
            raise ValueError("indices must be contiguous 1D tensors on the same NPU")
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must have int32 or int64 dtype")
        if stride < 0 or (stride > 0 and idx.numel() > 0 and base <= 0):
            raise ValueError("nonempty descriptors require a valid pointer and stride")
        sizes.append(idx.numel() * stride)

    total_bytes = sum(sizes)
    if total_bytes == 0:
        return 1
    blocks = [triton.cdiv(size, _BLOCK_SIZE) for size in sizes]
    partials = torch.empty((sum(blocks), 2), dtype=torch.int32, device=device)
    block_offset = 0
    byte_offset = 0
    for base, stride, idx, size, count in zip(
        data_ptrs, strides, indices, sizes, blocks
    ):
        if count:
            _adler32_partial_kernel[(count,)](
                base,
                idx,
                partials[block_offset:],
                stride,
                size,
                total_bytes - byte_offset,
                BLOCK=_BLOCK_SIZE,
            )
        block_offset += count
        byte_offset += size

    sums = partials.sum(dim=0, dtype=torch.int64).cpu().tolist()
    a = (1 + sums[0]) % _MOD_ADLER
    b = (total_bytes + sums[1]) % _MOD_ADLER
    return (b << 16) | a
