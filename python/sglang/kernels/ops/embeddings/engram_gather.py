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
    dst_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    HAS_DST: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    dst = tl.load(dst_ptr + row).to(tl.int64) if HAS_DST else row
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
) -> torch.Tensor:
    """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] bf16, contiguous).

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes and ``scale_ptr``
    [rows, dim // block_size] e8m0 bytes for global rows [row_lo, row_hi); both
    may live in device or host memory. Ids outside the range produce zero rows.

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
    w_row = tl.load(w_bias_ptr + owner) + idx * DIM
    s_row = tl.load(s_bias_ptr + owner) + idx * (DIM // BLK)
    offs = tl.arange(0, DIM)
    w = (w_row[:, None] + offs[None, :]).to(tl.pointer_type(tl.float8e4nv))
    s = (s_row[:, None] + (offs // BLK)[None, :]).to(tl.pointer_type(tl.uint8))
    mask = col_ok[:, None]
    vals = tl.load(w, mask=mask, other=0.0).to(tl.float32)
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
) -> torch.Tensor:
    """Gather ``ids`` ([T, C] int, unit column stride) into ``out`` ([T, C, dim]
    bf16, contiguous), reading each id from the shard that owns it.

    ``starts`` ([P] int64, P a power of two) holds each shard's first global row,
    padded with int64 max; ``w_bias`` / ``s_bias`` ([P] int64) are each shard's fp8
    row / e8m0 scale base address minus its first row's offset, so any rank's
    shard (local or a peer mapping over NVLink) is addressed by the global id.
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
            num_warps=4,
        )
    return out
