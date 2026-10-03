"""Adler-32 over PD transfer descriptors using Triton-Ascend.

The byte stream is ordered by descriptor, then by the supplied item indices.
Only the reduced checksum is copied to the host, not the KV cache itself.
"""

import torch
import triton
import triton.language as tl

_MOD_ADLER = 65521
_BLOCK_SIZE = 1024
# Arbitrary; picked to stay at core scale rather than measured on A5. Ascend
# re-runs a grid wider than the device's cores, so each program loops instead.
_MAX_PROGRAMS = 64


@triton.jit
def _adler32_partial_kernel(
    base,
    indices,
    partials,
    num_items,
    stride,
    items_per_program,
    remaining_mod,
    stride_mod,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    first = pid * items_per_program
    last = tl.minimum(first + items_per_program, num_items)
    # The int-to-pointer cast stays outside both loops: an in-loop cast is a
    # known Ascend hazard (vllm-ascend vllm_ascend/ops/triton/batch_memcpy.py).
    ptr = base.to(tl.pointer_type(tl.uint8))
    lane = tl.arange(0, BLOCK)

    # For a stream of N bytes x[j]:
    # a = 1 + sum(x[j]); b = N + sum((N - j) * x[j]), modulo 65521.
    # Every term is reduced before it accumulates, so the lanes stay in int32:
    # BLOCK * (65521 - 1) < 2**31, even for multi-GiB streams.
    pos_mod = ((first.to(tl.int64) * stride) % 65521).to(tl.int32)
    a = tl.zeros([BLOCK], dtype=tl.int32)
    b = tl.zeros([BLOCK], dtype=tl.int32)
    for item in range(first, last):
        row = tl.load(indices + item).to(tl.int64)
        row_ptr = ptr + row * stride
        for start in range(0, stride, BLOCK):
            offsets = start + lane
            valid = offsets < stride
            values = tl.load(row_ptr + offsets, mask=valid, other=0).to(tl.int32)
            # 2 * 65521 keeps the weight positive before the truncating modulo.
            weights = (remaining_mod - pos_mod - offsets % 65521 + 131042) % 65521
            a = (a + values) % 65521
            b = (b + (weights * values) % 65521) % 65521
        pos_mod = (pos_mod + stride_mod) % 65521
    tl.store(partials + pid * 2, tl.sum(a, 0) % 65521)
    tl.store(partials + pid * 2 + 1, tl.sum(b, 0) % 65521)


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
    plans = []
    for idx, size in zip(indices, sizes):
        if size == 0:
            plans.append((0, 0))
            continue
        items = idx.numel()
        per_program = triton.cdiv(items, min(items, _MAX_PROGRAMS))
        plans.append((per_program, triton.cdiv(items, per_program)))

    partials = torch.empty(
        (sum(count for _, count in plans), 2), dtype=torch.int32, device=device
    )
    program_offset = 0
    byte_offset = 0
    for base, stride, idx, size, (per_program, count) in zip(
        data_ptrs, strides, indices, sizes, plans
    ):
        if count:
            _adler32_partial_kernel[(count,)](
                base,
                idx,
                partials[program_offset:],
                idx.numel(),
                stride,
                per_program,
                (total_bytes - byte_offset) % _MOD_ADLER,
                stride % _MOD_ADLER,
                BLOCK=_BLOCK_SIZE,
            )
        program_offset += count
        byte_offset += size

    sums = partials.sum(dim=0, dtype=torch.int64).cpu().tolist()
    a = (1 + sums[0]) % _MOD_ADLER
    b = (total_bytes + sums[1]) % _MOD_ADLER
    return (b << 16) | a
