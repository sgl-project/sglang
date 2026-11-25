"""Capped SwiGLU with the route weight folded in, BF16 out, in one launch: the
activation between the two GEMMs of the UINT4-AWQ Quark experts on ROCm. Only the
launch is new: the math is the native `_capped_swiglu` device function."""

import torch
import triton
import triton.language as tl

# Private helpers of the native SwiGLU kernels, imported rather than copied.
from sglang.kernels.ops.gemm.inkling2_marlin_sm90 import (
    SWIGLU_BLOCK,
    _split_sum,
    _splits,
)
from sglang.kernels.ops.moe.inkling2_moe import _capped_swiglu


@triton.jit
def _swiglu_bf16(
    X,
    Ids,
    Gammas,
    Y,
    start,
    num_local,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    split_stride=0,
    SPLITS: tl.constexpr = 1,
):
    row = tl.program_id(0).to(tl.int64)
    expert = tl.load(Ids + row) - start
    if (expert < 0) | (expert >= num_local):
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    # Split-K partials sum to what the BF16 reduce would have stored.
    gate = _split_sum(X, row * 2 * N + 2 * cols, mask, split_stride, SPLITS, True)
    up = _split_sum(X, row * 2 * N + 2 * cols + 1, mask, split_stride, SPLITS, True)
    gate, up = gate.to(tl.float32), up.to(tl.float32)
    y = _capped_swiglu(gate, up, tl.load(Gammas + row), GATE_CAP, UP_CAP)
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask)


def swiglu_bf16(acc, ids, gammas, start, num_local, caps):
    """Capped SwiGLU times the route weight per w13 output row (source order), BF16
    out; rows of other ranks' experts stay unset. acc may hold FP32 split-K partials."""
    acc, splits = _splits(acc)
    rows, width = acc.shape[0], acc.shape[1] // 2
    y = torch.empty((rows, width), dtype=torch.bfloat16, device=acc.device)
    if rows:
        _swiglu_bf16[(rows, triton.cdiv(width, SWIGLU_BLOCK))](
            acc,
            ids,
            gammas,
            y,
            start,
            num_local,
            width,
            SWIGLU_BLOCK,
            *caps,
            split_stride=acc.numel() if splits > 1 else 0,
            SPLITS=splits,
            enable_fp_fusion=False,
        )
    return y
