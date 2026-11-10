"""Triton gather of DeepSeek-V4.1 engram rows: fp8 e4m3 or packed MXFP4 (E2M1) payload,
e8m0 block scales.

The table pointers arrive as raw addresses so one kernel serves a device table, a
pinned host table, or (on Grace-Blackwell, through ATS) a plain host mapping.

Dequant is fp32(row) * 2**(exp - 127) then a single round to the activation
dtype. Hopper/Blackwell keep bf16. SM70 has no BF16 and no Triton ``fp8e4nv``,
so Volta loads the payload as uint8, converts e4m3fn in software, and stores
fp16.
"""

import torch
import triton
import triton.language as tl

# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127


def _sm70_gather() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7


@triton.jit
def _dequant_rows(
    w_base,
    s_base,
    row,
    offs,
    mask,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    MXFP4: tl.constexpr,
    SM70: tl.constexpr = False,
):
    # Dequantized fp32 values at columns `offs` of table rows `row`, whose payload
    # and e8m0 scales start at byte addresses `w_base` and `s_base`. MXFP4 rows
    # hold two E2M1 values per byte, the even column in the low nibble.
    if MXFP4:
        byte = tl.load(
            w_base.to(tl.pointer_type(tl.uint8)) + row * (DIM // 2) + offs // 2,
            mask=mask,
            other=0,
        ).to(tl.int32)
        nib = tl.where(offs % 2 == 0, byte & 0xF, byte >> 4)
        # Magnitude code m: 0 -> 0.0, 1 -> 0.5 (fp32 bits 252 << 22), and for
        # m >= 2 the fp32 bits are (m + 252) << 22; every value is exact.
        m = nib & 0x7
        bits = tl.where(m >= 2, (m + 252) << 22, m * (252 << 22))
        # Set the sign bit: before Triton 3.8, -x is 0 - x, which turns -0 into +0.
        vals = (bits | ((nib & 0x8) << 28)).to(tl.float32, bitcast=True)
    elif SM70:
        # SM70 Triton's fp8e4nv is unsupported; uint8 + software e4m3fn matches
        # torch.float() of float8_e4m3fn (including denorms: man * 2**-9).
        w = w_base.to(tl.int64).to(tl.pointer_type(tl.uint8))
        raw = tl.load(w + row * DIM + offs, mask=mask, other=0).to(tl.int32)
        sign = (raw >> 7) & 1
        exp = (raw >> 3) & 0xF
        man = raw & 7
        denorm = man.to(tl.float32) * (2.0**-9)
        norm = (1.0 + man.to(tl.float32) * 0.125) * tl.math.exp2(
            exp.to(tl.float32) - 7.0
        )
        vals = tl.where(exp == 0, denorm, norm)
        vals = tl.where((exp == 0) & (man == 0), 0.0, vals)
        # e4m3fn: exp=15, man=7 is NaN (no Inf).
        vals = tl.where((exp == 15) & (man == 7), float("nan"), vals)
        vals = tl.where(sign == 1, -vals, vals)
    else:
        vals = tl.load(
            w_base.to(tl.pointer_type(tl.float8e4nv)) + row * DIM + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    exps = tl.load(
        s_base.to(tl.pointer_type(tl.uint8)) + row * (DIM // BLK) + offs // BLK,
        mask=mask,
        other=0,
    ).to(tl.int32)
    # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    return vals * scale


def unpack_mxfp4_rows(packed: torch.Tensor) -> torch.Tensor:
    """[..., dim // 2] uint8 E2M1 pairs (even column in the low nibble) ->
    [..., dim] e4m3, bit-exact: the torch twin of the MXFP4 decode above."""
    nibble = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)
    # E2M1 magnitudes 0, 0.5, 1, ..., 6 in e4m3: 0x00, 0x30, then 4 * m + 48.
    magnitude = nibble & 0x7
    bits = torch.where(magnitude >= 2, magnitude * 4 + 48, magnitude * 48)
    return (bits | ((nibble & 0x8) << 4)).view(torch.float8_e4m3fn)


@triton.jit
def _engram_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    dst_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    HAS_DST: tl.constexpr,
    MXFP4: tl.constexpr,
    OUT_BF16: tl.constexpr = True,
    SM70: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    dst = tl.load(dst_ptr + row).to(tl.int64) if HAS_DST else row
    # The table holds rows [row_lo, row_hi); an id outside it is not read and
    # comes out as zeros, which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    offs = tl.arange(0, DIM)
    vals = _dequant_rows(
        w_ptr.to(tl.int64),
        s_ptr.to(tl.int64),
        local,
        offs,
        owned,
        DIM,
        BLK,
        E8M0_ZERO,
        MXFP4,
        SM70,
    )
    out = tl.where(owned, vals, 0.0)
    if OUT_BF16:
        tl.store(out_ptr + dst * DIM + offs, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + dst * DIM + offs, out.to(tl.float16))


def engram_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
    sort_ids: bool = False,
    mxfp4: bool = False,
) -> torch.Tensor:
    """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] fp16/bf16, contiguous).

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes, or with ``mxfp4``
    [rows, dim // 2] bytes of E2M1 pairs (even column in the low nibble), and
    ``scale_ptr`` [rows, dim // block_size] e8m0 bytes for global rows
    [row_lo, row_hi); both may live in device or host memory. Ids outside the
    range produce zero rows.

    ``sort_ids`` reads the rows in address order and writes each to its original
    position. Random rows of a host table larger than the GPU's translation reach
    miss on nearly every row; sorted, neighbouring programs share a page. SM70
    requires ``out.dtype == torch.float16``.
    """
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
    assert dim % block_size == 0, (dim, block_size)
    assert (
        ids.is_cuda and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    )
    assert out.is_contiguous()
    sm70 = _sm70_gather()
    if sm70:
        assert out.dtype == torch.float16, (
            f"SM70 engram_gather writes fp16, not {out.dtype}"
        )
    else:
        assert out.dtype == torch.bfloat16
    assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
    n = ids.numel()
    if n:
        dst = None
        if sort_ids:
            ids, dst = torch.sort(ids)
        _engram_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            dst if dst is not None else ids,
            out,
            row_lo,
            row_hi,
            DIM=dim,
            BLK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
            HAS_DST=dst is not None,
            MXFP4=mxfp4,
            OUT_BF16=out.dtype == torch.bfloat16,
            SM70=sm70,
        )
    return out


@triton.jit
def _engram_rows_to_token_major_kernel(
    src_ptr,
    dst_ptr,
    n,
    WORLD: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per (token, source rank): move that rank's CHUNK values.
    pid = tl.program_id(0).to(tl.int64)
    token, rank = pid // WORLD, pid % WORLD
    src = src_ptr + (rank * n + token) * CHUNK
    dst = dst_ptr + (token * WORLD + rank) * CHUNK
    for start in range(0, CHUNK, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < CHUNK
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def engram_rows_to_token_major(recv: torch.Tensor, world: int) -> torch.Tensor:
    """Reorder an all-to-all's output, ``recv`` ([world * n, chunk], contiguous,
    one block of n rows per source rank), into [n, world * chunk]: each token's
    chunks in source-rank order. Several times faster than transpose + reshape."""
    assert recv.dim() == 2 and recv.is_contiguous(), (recv.shape, recv.stride())
    assert recv.shape[0] % world == 0, (recv.shape, world)
    n, chunk = recv.shape[0] // world, recv.shape[1]
    out = recv.new_empty((n, world * chunk))
    if n and chunk:
        _engram_rows_to_token_major_kernel[(n * world,)](
            recv,
            out,
            n,
            WORLD=world,
            CHUNK=chunk,
            BLOCK=min(1024, triton.next_power_of_2(chunk)),
        )
    return out
