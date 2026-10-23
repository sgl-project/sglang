"""Triton gather of DeepSeek-V4.1 engram rows: fp8 e4m3 payload, e8m0 block scales.

The table pointers arrive as raw addresses so one kernel serves a device table, a
pinned host table, or (on Grace-Blackwell, through ATS) a plain host mapping. The
output is bf16 computed as fp32(row) * 2**(exp - 127) then rounded once, which is
the arithmetic of the torch lookup it replaces.
"""

import torch
import triton
import triton.language as tl

# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127


@triton.jit
def _engram_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    IDS_COLS: tl.constexpr,
    IDS_ROW_STRIDE: tl.constexpr,
    IDS_COL_STRIDE: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx_offset = (row // IDS_COLS) * IDS_ROW_STRIDE + (row % IDS_COLS) * IDS_COL_STRIDE
    idx = tl.load(ids_ptr + idx_offset).to(tl.int64)
    # The table holds rows [row_lo, row_hi); an id outside it is not read and
    # comes out as zeros, which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    vals = tl.load(w + local * DIM + offs, mask=owned, other=0.0).to(tl.float32)
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
        tl.int32
    )
    # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = tl.where(owned, vals * scale, 0.0)
    tl.store(out_ptr + row * DIM + offs, out.to(tl.bfloat16))


def engram_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
) -> torch.Tensor:
    """Gather 1D or 2D integer ``ids`` into contiguous ``out`` [ids.numel(), dim].

    Strided indices are read in logical row-major order without a repack.

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes and ``scale_ptr``
    [rows, dim // block_size] e8m0 bytes for global rows [row_lo, row_hi); both
    may live in device or host memory. Ids outside the range produce zero rows.
    """
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
    assert dim % block_size == 0, (dim, block_size)
    assert (
        ids.is_cuda and ids.ndim in (1, 2) and ids.dtype in (torch.int32, torch.int64)
    )
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
    n = ids.numel()
    if n:
        _engram_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            out,
            row_lo,
            row_hi,
            DIM=dim,
            BLK=block_size,
            IDS_COLS=ids.shape[1] if ids.ndim == 2 else 1,
            IDS_ROW_STRIDE=ids.stride(0),
            IDS_COL_STRIDE=ids.stride(1) if ids.ndim == 2 else 0,
            E8M0_ZERO=_E8M0_ZERO,
        )
    return out
