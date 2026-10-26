"""Muse Spark target verify: the split-KV merge of the verify kernels' FP32 partials fused with the attention
output's per-head RMSNorm and sigmoid gate, one launch instead of the merge kernel plus the norm-gate kernel.

Partials ``[T, H, S, 64]`` with max / denom ``[T, H, S]`` as the swa_verify_splitkv / gqa_verify_splitkv(_2kv)
partials kernels write them. The merged row is rounded to BF16, as the merge kernel stores it, before the norm.
"""

import torch
import triton
import triton.language as tl

HEAD_DIM = 64


class VerifyReduceDeferral:
    """Hand-off between a model that consumes the partials itself and the attention backend: while ``requested``
    the verify kernels skip their merge and leave ``state`` = (partial, maximum, denom, v_descale)."""

    def __init__(self):
        self.requested = False
        self.state = None

    def take(self):
        state, self.state = self.state, None
        return state


DEFERRAL = VerifyReduceDeferral()


@triton.jit
def _reduce_norm_gate(
    PARTIAL,
    MAXIMUM,
    DENOM,
    V_SCALE,
    G,
    W,
    Y,
    eps,
    N,
    H,
    stride_g0,
    stride_g1,
    HEADS_PER_GROUP: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    # BLOCK_M (token, head) rows per program; row r = token * H + head.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, 64)
    index = rows[:, None] * SPLITS + s[None, :]
    maximum = tl.load(MAXIMUM + index, mask=row_mask[:, None], other=float("-inf"))
    denom = tl.load(DENOM + index, mask=row_mask[:, None], other=0.0)
    global_max = tl.max(tl.where(denom > 0, maximum, float("-inf")), axis=1)
    global_max = tl.where(global_max > float("-inf"), global_max, 0.0)
    weights = tl.where(denom > 0, tl.exp2(maximum - global_max[:, None]), 0.0)
    total = tl.sum(denom * weights, axis=1)
    partial = tl.load(
        PARTIAL + index[:, :, None] * 64 + d[None, None, :],
        mask=row_mask[:, None, None],
        other=0.0,
    )
    accum = tl.sum(partial * weights[:, :, None], axis=1)
    inverse = tl.where(total > 0, 1.0 / total, 0.0)
    if V_SCALE is not None:
        inverse *= tl.load(V_SCALE)
    x = (accum * inverse[:, None]).to(tl.bfloat16).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / 64 + eps)
    x = x * rstd[:, None]
    if HAS_WEIGHT:
        x = x * tl.load(W + d).to(tl.float32)[None, :]
    if HAS_GATE:
        token = rows // H
        head = rows % H
        offset = (head // HEADS_PER_GROUP) * stride_g1 + (head % HEADS_PER_GROUP) * 64
        g = tl.load(
            G + token[:, None] * stride_g0 + offset[:, None] + d[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        x = x * tl.sigmoid(g)
    tl.store(Y + rows[:, None] * 64 + d[None, :], x, mask=row_mask[:, None])


def verify_reduce_norm_gate(state, out_shape, out_dtype, gate, eps, weight=None):
    """Merged, RMS-normalized (``weight`` optional) and sigmoid-gated attention output ``out_shape``
    (``[T, H * 64]``) from the deferred verify ``state``; ``gate`` is ``None`` or the grouped
    ``[T, groups, heads_per_group * 64]`` gate view."""
    partial, maximum, denom, v_scale = state
    tokens, heads, splits = maximum.shape
    n = tokens * heads
    y = torch.empty(out_shape, device=partial.device, dtype=out_dtype)
    if gate is None:
        stride_g0 = stride_g1 = 0
        heads_per_group = heads
    else:
        heads_per_group = gate.shape[-1] // HEAD_DIM
        stride_g0, stride_g1 = gate.stride(0), gate.stride(1)
    block_m = 8
    _reduce_norm_gate[(triton.cdiv(n, block_m),)](
        partial,
        maximum,
        denom,
        v_scale,
        gate,
        weight,
        y,
        eps,
        n,
        heads,
        stride_g0,
        stride_g1,
        heads_per_group,
        splits,
        block_m,
        weight is not None,
        gate is not None,
        num_warps=4,
    )
    return y
