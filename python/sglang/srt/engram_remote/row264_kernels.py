"""Bit-preserving E4M3/E8M0 lookup from a contiguous264-byte row.

Raw pointers allow the same consumer arithmetic for registered Host and GPU
backing. query_order=True decodes the RDMA result at the original query slot.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(
    base, ids, output, lo, hi, reps, QUERY_ORDER: tl.constexpr, MAPPED: tl.constexpr
):
    q = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids + q).to(tl.int64)
    owned = (idx >= lo) & (idx < hi)
    row = q if QUERY_ORDER else tl.where(owned, idx - lo, 0)
    if MAPPED:
        row = tl.load(reps + q, mask=owned, other=0)
        owned = owned & (row >= 0) & (row < tl.num_programs(0))
    byte = base.to(tl.int64).to(tl.pointer_type(tl.uint8))
    fp8 = byte.to(tl.pointer_type(tl.float8e4nv))
    x = tl.arange(0, 256)
    value = tl.load(fp8 + row * 264 + x, mask=owned, other=0.0).to(tl.float32)
    exp = tl.load(byte + row * 264 + 256 + x // 32, mask=owned, other=0).to(tl.int32)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exp == 0, 2.0**-127, scale)
    result = tl.where(owned, value * scale, 0.0)
    tl.store(output + q * 256 + x, result.to(tl.bfloat16))


def gather_row264(
    pointer,
    ids,
    output,
    row_lo=0,
    row_hi=2**62,
    query_order=False,
    representatives=None,
):
    if ids.dtype != torch.int64 or not ids.is_cuda or not ids.is_contiguous():
        raise ValueError("row264 IDs must be contiguous CUDA int64")
    if (
        output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or output.numel() != ids.numel() * 256
    ):
        raise ValueError("row264 output must be contiguous BF16 with256 columns")
    if representatives is not None and (
        representatives.dtype != torch.int64
        or representatives.shape != ids.shape
        or representatives.device != ids.device
    ):
        raise ValueError("invalid mapped row indices")
    if ids.numel():
        _gather[(ids.numel(),)](
            pointer,
            ids,
            output,
            row_lo,
            row_hi,
            representatives if representatives is not None else ids,
            QUERY_ORDER=query_order,
            MAPPED=representatives is not None,
            num_warps=4,
        )
    return output
