"""Fused DFlash post-verify bookkeeping (ROCm K3 decode path).

* ``verify_argmax``: torch.argmax(logits, -1) for the [bs * block, vocab] fp32
  verify logits via the NaN-aware two-stage split kernels of row_argmax.py
  (identical result: NaNs win, first index on ties).
* ``mamba_track_steps``: one kernel for the ~15 elementwise int64 ops that
  ``DFlashWorkerV2._update_target_mamba_state_after_verify`` issues to derive
  ``last_correct_step_indices`` and ``mamba_steps_to_track``. Pure integer
  math, bit-identical.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.row_argmax import (
    _medium_argmax_final_kernel,
    _medium_argmax_partial_kernel,
)


def verify_argmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    rows, n = x.shape
    out = torch.empty((rows,), dtype=torch.int64, device=x.device)
    if rows == 0:
        return out
    block = 4096 if rows <= 256 else 8192
    splits = triton.cdiv(n, block)
    pv = torch.empty((rows, splits), dtype=torch.float32, device=x.device)
    pi = torch.empty((rows, splits), dtype=torch.int32, device=x.device)
    _medium_argmax_partial_kernel[(rows, splits)](
        x, pv, pi, n, x.stride(0), splits, block, num_warps=4
    )
    _medium_argmax_final_kernel[(rows,)](
        pv, pi, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out


@triton.jit
def _mamba_track_kernel(
    commit_ptr,
    pre_ptr,
    post_ptr,
    last_out,
    steps_out,
    n,
    interval,
    HAS_TRACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    commit = tl.load(commit_ptr + offs, mask=m, other=0).to(tl.int64)
    tl.store(last_out + offs, commit - 1, mask=m)
    if HAS_TRACK:
        pre = tl.load(pre_ptr + offs, mask=m, other=0).to(tl.int64)
        post = tl.load(post_ptr + offs, mask=m, other=0).to(tl.int64)
        # torch floor-division on int64 (operands may be negative only in
        # padding rows; keep floor semantics anyway).
        qpre = pre // interval
        qpre = tl.where(
            (pre % interval != 0) & ((pre < 0) != (interval < 0)), qpre - 1, qpre
        )
        qpost = post // interval
        qpost = tl.where(
            (post % interval != 0) & ((post < 0) != (interval < 0)), qpost - 1, qpost
        )
        to_track = qpre != qpost
        tracking_point = qpost * interval
        ith = tl.maximum(tracking_point - pre - 1, 0)
        can = to_track & (ith < commit)
        tl.store(steps_out + offs, tl.where(can, ith, -1), mask=m)


def mamba_track_steps(commit_lens, seq_lens_pre, seq_lens_post, interval, has_track):
    """Returns (last_correct_step_indices int64, mamba_steps_to_track int64 | None)."""
    n = commit_lens.shape[0]
    dev = commit_lens.device
    last = torch.empty((n,), dtype=torch.int64, device=dev)
    steps = torch.empty((n,), dtype=torch.int64, device=dev) if has_track else last
    if n:
        BLOCK = 128
        _mamba_track_kernel[(triton.cdiv(n, BLOCK),)](
            commit_lens,
            seq_lens_pre if has_track else commit_lens,
            seq_lens_post if has_track else commit_lens,
            last,
            steps,
            n,
            int(interval) if has_track else 1,
            HAS_TRACK=bool(has_track),
            BLOCK=BLOCK,
            num_warps=1,
        )
    return last, (steps if has_track else None)


@triton.jit
def _compact_lens_kernel(seq_ptr, dpl_out, suffix_out, n, W, PAGE, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(seq_ptr + offs, mask=m, other=0).to(tl.int64)
    s32 = s.to(tl.int32)
    vis = tl.where(s32 > W, W, s32).to(tl.int64)
    if PAGE > 1:
        vs = s - vis
        r = vs % PAGE
        r = tl.where((r != 0) & ((r < 0) != (PAGE < 0)), r + PAGE, r)
        aligned = vs - r
        dpl = (s - aligned).to(tl.int32)
    else:
        dpl = vis.to(tl.int32)
    tl.store(dpl_out + offs, dpl, mask=m)
    tl.store(suffix_out + offs, s - dpl.to(tl.int64), mask=m)


def compact_draft_lens(seq_lens, window, page_size):
    """(draft_prefix_lens int32, suffix_start int64) exactly as
    DFlashWorkerV2._compute_compact_draft_seq_lens + the suffix_start of
    _rebuild_compact_draft_cache, in one launch."""
    n = seq_lens.shape[0]
    dpl = torch.empty((n,), dtype=torch.int32, device=seq_lens.device)
    suffix = torch.empty((n,), dtype=torch.int64, device=seq_lens.device)
    if n:
        BLOCK = 128
        _compact_lens_kernel[(triton.cdiv(n, BLOCK),)](
            seq_lens,
            dpl,
            suffix,
            n,
            int(window),
            int(page_size),
            BLOCK=BLOCK,
            num_warps=1,
        )
    return dpl, suffix
