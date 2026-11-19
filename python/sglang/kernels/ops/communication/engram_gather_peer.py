"""Triton gather of DeepSeek-V4.1 engram rows from the rank that owns each row.

Every rank's shard (local, or a peer mapping over NVLink) is addressed through
pre-biased base pointers, so one kernel reads a row wherever it lives. Rows are
dequantized as in ``embeddings.engram_gather``.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.embeddings.engram_gather import _E8M0_ZERO, _dequant_rows


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
    MXFP4: tl.constexpr,
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
    offs = tl.arange(0, DIM)
    mask = col_ok[:, None]
    out = _dequant_rows(
        tl.load(w_bias_ptr + owner)[:, None],
        tl.load(s_bias_ptr + owner)[:, None],
        idx[:, None],
        offs[None, :],
        mask,
        DIM,
        BLK,
        E8M0_ZERO,
        MXFP4,
    )
    # -0 becomes +0, as the zero-fill + sum of the sharded lookup leaves it, so the
    # two lookups agree bit for bit.
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
    mxfp4: bool = False,
) -> torch.Tensor:
    """Gather ``ids`` ([T, C] int, unit column stride) into ``out`` ([T, C, dim]
    bf16, contiguous), reading each id from the shard that owns it.

    ``starts`` ([P] int64, P a power of two) holds each shard's first global row,
    padded with int64 max; ``w_bias`` / ``s_bias`` ([P] int64) are each shard's
    row / e8m0 scale base address minus its first row's offset, so any rank's
    shard (local or a peer mapping over NVLink) is addressed by the global id.
    Rows are fp8 e4m3, or with ``mxfp4`` E2M1 pairs as in ``engram_gather``.
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
            MXFP4=mxfp4,
            num_warps=4,
        )
    return out
