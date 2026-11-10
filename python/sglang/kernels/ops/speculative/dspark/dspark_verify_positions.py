from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_verify_positions_cmp_padding_one_kernel(
    positions_ptr,
    live_seq_lens_ptr,
    dst_ptr,
    n_req,
    n_draft,
    ratio,
    dst_size,
    BLOCK: tl.constexpr,
):
    """Fused verify compressor-position fill for uniform (non-ragged) verify.

    Equivalent to the CPU reference in
    ``_fill_verify_positions_cmp_padding_one``: for every request with a
    live prefix, token ``t`` (1-based) completes a compressed group when
    ``(live + t) % ratio == 0``; the emitted value is that token's
    position shifted to the group start (``position + 1 - ratio``).
    A single program visits elements in row-major order, so a scalar
    running rank reproduces the stable ``nonzero`` compaction order, and
    the destination zero-fill is fused into the same launch.
    """
    num_dst_blocks = tl.cdiv(dst_size, BLOCK)
    for b in range(0, num_dst_blocks):
        offs = b * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            dst_ptr + offs, tl.zeros([BLOCK], dtype=tl.int64), mask=offs < dst_size
        )

    rank = tl.zeros([], dtype=tl.int32)
    total = n_req * n_draft
    for b in range(0, tl.cdiv(total, BLOCK)):
        offs = b * BLOCK + tl.arange(0, BLOCK)
        within = offs < total
        req = offs // n_draft
        t = offs % n_draft + 1
        live = tl.load(live_seq_lens_ptr + req, mask=within, other=0)
        position = tl.load(positions_ptr + offs, mask=within, other=0)
        keep = within & (live > 0) & (((live + t) % ratio) == 0)
        ranks = rank + tl.cumsum(keep.to(tl.int32), 0) - 1
        rank += tl.sum(keep.to(tl.int32), 0)
        in_bounds = keep & (ranks < dst_size)
        dst_offsets = tl.where(in_bounds, ranks, 0)
        values = tl.where(in_bounds, position + (1 - ratio), 0)
        tl.store(dst_ptr + dst_offsets, values, mask=in_bounds)


def fill_verify_positions_cmp_padding_one_fused(
    positions: torch.Tensor,
    live_seq_lens: torch.Tensor,
    dst: torch.Tensor,
    ratio: int,
    n_draft: int,
) -> bool:
    """Try the single-kernel triton fast path; return True when it ran.

    Falls back (returns False) on CPU tensors — where unit tests exercise the
    aten reference chain.
    """
    if positions.device.type != "npu":
        return False
    request_num = positions.shape[0] // int(n_draft)
    if live_seq_lens.numel() < request_num:
        # Let the aten path raise its size-mismatch error.
        return False
    _fill_verify_positions_cmp_padding_one_kernel[(1,)](
        positions,
        live_seq_lens,
        dst,
        request_num,
        int(n_draft),
        int(ratio),
        dst.numel(),
        BLOCK=256,
    )
    return True
