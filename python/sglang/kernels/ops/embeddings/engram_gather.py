"""Triton gather of DeepSeek-V4.1 engram rows: fp8 e4m3 or packed MXFP4 (E2M1) payload,
e8m0 block scales.

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
def _row_values(row_addr, offs, mask, PACKED_FP4: tl.constexpr):
    # fp32 values at columns `offs` of the rows starting at byte address `row_addr`.
    # Packed rows hold two E2M1 values per byte, the even column in the low nibble.
    if PACKED_FP4:
        byte = tl.load(
            (row_addr + offs // 2).to(tl.pointer_type(tl.uint8)), mask=mask, other=0
        ).to(tl.int32)
        nib = tl.where(offs % 2 == 0, byte & 0xF, byte >> 4)
        # Magnitude code m: 0 -> 0.0, 1 -> 0.5 (fp32 bits 252 << 22), and for
        # m >= 2 the fp32 bits are (m + 252) << 22; every value is exact.
        m = nib & 0x7
        bits = tl.where(m >= 2, (m + 252) << 22, m * (252 << 22))
        # Set the sign bit: before Triton 3.8, -x is 0 - x, which turns -0 into +0.
        vals = (bits | ((nib & 0x8) << 28)).to(tl.float32, bitcast=True)
    else:
        vals = tl.load(
            (row_addr + offs).to(tl.pointer_type(tl.float8e4nv)), mask=mask, other=0.0
        ).to(tl.float32)
    return vals


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
    PACKED_FP4: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    dst = tl.load(dst_ptr + row).to(tl.int64) if HAS_DST else row
    # The table holds rows [row_lo, row_hi); an id outside it is not read and
    # comes out as zeros, which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    row_bytes = DIM // 2 if PACKED_FP4 else DIM
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    vals = _row_values(w_ptr.to(tl.int64) + local * row_bytes, offs, owned, PACKED_FP4)
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
        tl.int32
    )
    # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = tl.where(owned, vals * scale, 0.0)
    tl.store(out_ptr + dst * DIM + offs, out.to(tl.bfloat16))


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
    packed_fp4: bool = False,
) -> torch.Tensor:
    """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] bf16, contiguous).

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes, or with ``packed_fp4``
    [rows, dim // 2] bytes of E2M1 pairs (even column in the low nibble), and
    ``scale_ptr`` [rows, dim // block_size] e8m0 bytes for global rows
    [row_lo, row_hi); both may live in device or host memory. Ids outside the
    range produce zero rows.

    ``sort_ids`` reads the rows in address order and writes each to its original
    position. Random rows of a host table larger than the GPU's translation reach
    miss on nearly every row; sorted, neighbouring programs share a page.
    """
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
    assert dim % block_size == 0, (dim, block_size)
    assert (
        ids.is_cuda and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    )
    assert out.is_contiguous() and out.dtype == torch.bfloat16
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
            PACKED_FP4=packed_fp4,
        )
    return out


@triton.jit
def _engram_gather_peer_kernel(
    ids_ptr,
    ids_stride,
    out_ptr,
    w_bias_ptr,
    s_bias_ptr,
    starts_ptr,
    n_cols,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    ROWS: tl.constexpr,
    LOG2_WORLD: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    PACKED_FP4: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    col_ok = cols < n_cols
    idx = tl.load(ids_ptr + tok * ids_stride + cols, mask=col_ok, other=0).to(tl.int64)
    # Owner = the last shard whose first row is <= idx; starts is padded to a
    # power of two with int64 max, so the search never runs past a real shard.
    owner = tl.zeros([ROWS], dtype=tl.int32)
    for step in tl.static_range(LOG2_WORLD):
        probe = owner + (1 << (LOG2_WORLD - 1 - step))
        start = tl.load(starts_ptr + probe)
        owner = tl.where(start <= idx, probe, owner)
    # Bases are pre-biased by each shard's first row, so a global id indexes them.
    row_bytes = DIM // 2 if PACKED_FP4 else DIM
    w_row = tl.load(w_bias_ptr + owner) + idx * row_bytes
    s_row = tl.load(s_bias_ptr + owner) + idx * (DIM // BLK)
    offs = tl.arange(0, DIM)
    s = (s_row[:, None] + (offs // BLK)[None, :]).to(tl.pointer_type(tl.uint8))
    mask = col_ok[:, None]
    vals = _row_values(w_row[:, None], offs[None, :], mask, PACKED_FP4)
    exps = tl.load(s, mask=mask, other=0).to(tl.int32)
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    # -0 becomes +0, as the zero-fill + sum of the sharded lookup leaves it, so the
    # two lookups agree bit for bit.
    out = vals * scale
    out = tl.where(out == 0.0, 0.0, out)
    dst = out_ptr + (tok * n_cols + cols)[:, None].to(tl.int64) * DIM + offs[None, :]
    tl.store(dst, out.to(tl.bfloat16), mask=mask)


def engram_gather_peer(
    ids: torch.Tensor,
    out: torch.Tensor,
    w_bias: torch.Tensor,
    s_bias: torch.Tensor,
    starts: torch.Tensor,
    dim: int,
    block_size: int,
    packed_fp4: bool = False,
) -> torch.Tensor:
    """Gather ``ids`` ([T, C] int, unit column stride) into ``out`` ([T, C, dim]
    bf16, contiguous), reading each id from the shard that owns it.

    ``starts`` ([P] int64, P a power of two) holds each shard's first global row,
    padded with int64 max; ``w_bias`` / ``s_bias`` ([P] int64) are each shard's
    row / e8m0 scale base address minus its first row's offset, so any rank's
    shard (local or a peer mapping over NVLink) is addressed by the global id.
    Rows are fp8 e4m3, or with ``packed_fp4`` E2M1 pairs as in ``engram_gather``.
    """
    assert ids.dim() == 2 and ids.stride(1) == 1, (ids.shape, ids.stride())
    assert ids.dtype in (torch.int32, torch.int64)
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    assert out.shape == (*ids.shape, dim), (out.shape, ids.shape)
    world = starts.numel()
    assert world & (world - 1) == 0, world
    t, c = ids.shape
    if t and c:
        rows = 8
        _engram_gather_peer_kernel[(t, triton.cdiv(c, rows))](
            ids,
            ids.stride(0),
            out,
            w_bias,
            s_bias,
            starts,
            c,
            DIM=dim,
            BLK=block_size,
            ROWS=rows,
            LOG2_WORLD=world.bit_length() - 1,
            E8M0_ZERO=_E8M0_ZERO,
            PACKED_FP4=packed_fp4,
            num_warps=4,
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
