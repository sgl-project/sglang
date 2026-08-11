# SPDX-License-Identifier: Apache-2.0
"""Z-Image-specific bit-exact per-head RMSNorm kernel."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _qk_rmsnorm_native_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    token_stride,
    nheads,
    n_rows,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    rows_per_prog: tl.constexpr,
):
    """Per-head RMSNorm replicating the eager ZImageRMSNorm kernel chain bit-for-bit.

    The eager path (`ZImageRMSNorm.forward` on a `[rows, head_dim]` bf16 tensor)
    lowers to aten pow/mean/rsqrt/mul kernels. For bf16 rows of 128, aten's
    reduce_kernel vectorizes with 16-byte loads (8 bf16 elements per lane):
    lane t serially accumulates contiguous elements 8t .. 8t+7 in fp32
    (squares rounded to bf16 first, matching aten::pow), then a shfl-down
    butterfly tree combines the 32 lane partials (halves 16/8/4/2/1). Every
    intermediate below is rounded to bf16 at exactly the same points as the
    aten kernels, so the output satisfies `torch.equal` against the eager path
    (verified over 9M+ random rows across scales 0.005-200).
    """
    prog = tl.program_id(0)
    row_offs = tl.arange(0, rows_per_prog)
    rows = prog * rows_per_prog + row_offs
    row_mask = rows < n_rows
    tokens = rows // nheads
    heads = rows % nheads
    base = x_ptr + tokens * token_stride + heads * head_dim

    offs = tl.arange(0, head_dim)
    x = tl.load(base[:, None] + offs[None, :], mask=row_mask[:, None], other=0.0)
    sq = (x * x).to(tl.bfloat16)

    # Lane t's serial accumulation of contiguous elements 8t..8t+7: peel the
    # 8-element groups apart with ordered tl.split chains (loads stay one
    # coalesced pass) and add them in exact serial order in fp32.
    g = tl.reshape(sq, (rows_per_prog, head_dim // 8, 4, 2), can_reorder=False)
    p0, p1 = tl.split(g)  # elements (0,2,4,6) / (1,3,5,7)
    p00, p01 = tl.split(
        tl.reshape(p0, (rows_per_prog, head_dim // 8, 2, 2), can_reorder=False)
    )  # (0,4) / (2,6)
    p10, p11 = tl.split(
        tl.reshape(p1, (rows_per_prog, head_dim // 8, 2, 2), can_reorder=False)
    )  # (1,5) / (3,7)
    s0, s4 = tl.split(
        tl.reshape(p00, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s2, s6 = tl.split(
        tl.reshape(p01, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s1, s5 = tl.split(
        tl.reshape(p10, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s3, s7 = tl.split(
        tl.reshape(p11, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    acc = tl.reshape(s0, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s1, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s2, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s3, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s4, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s5, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s6, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s7, (rows_per_prog, head_dim // 8)).to(tl.float32)
    # shfl-down butterfly over the 16 lane partials (lanes 16..31 hold the
    # additive identity in aten's 32-lane warp, so their stage is an exact
    # no-op): acc[i] += acc[i + half] for half in (8, 4, 2, 1).
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 8), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 4), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 2), can_reorder=False), axis=1)
    ssum = tl.sum(acc, axis=1)
    ms = (ssum / head_dim).to(tl.bfloat16)
    # aten adds the python-float eps in fp32 opmath, then rsqrt rounds to bf16.
    rstd = tl.rsqrt((ms.to(tl.float32) + eps).to(tl.bfloat16).to(tl.float32)).to(
        tl.bfloat16
    )

    weight = tl.load(weight_ptr + offs)
    y = ((x.to(tl.float32) * rstd.to(tl.float32)[:, None]).to(tl.bfloat16) * weight).to(
        tl.bfloat16
    )
    tl.store(
        y_ptr + rows[:, None] * head_dim + offs[None, :], y, mask=row_mask[:, None]
    )


def _qk_head_token_stride(x: torch.Tensor, head_dim: int) -> int | None:
    """Token stride for a `[B, S, H, head_dim]` view whose head block is packed.

    Accepts the strided views produced by slicing a fused qkv projection: the
    last dim must be contiguous, heads packed (`stride(-2) == head_dim`), and
    tokens uniformly strided across the flattened `(B, S)` dims.
    """
    if x.dim() != 4 or x.shape[-1] != head_dim:
        return None
    if x.stride(-1) != 1 or x.stride(-2) != head_dim:
        return None
    token_stride = x.stride(1)
    if x.shape[0] > 1 and x.stride(0) != x.shape[1] * token_stride:
        return None
    return token_stride


def can_use_qk_rmsnorm_native(
    x: torch.Tensor, weight: torch.Tensor, head_dim: int
) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.numel() > 0
        and head_dim == 128
        and weight.shape == (head_dim,)
        and weight.is_contiguous()
        and _qk_head_token_stride(x, head_dim) is not None
    )


def zimage_qk_rmsnorm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Fused, bit-exact ZImageRMSNorm over q/k heads; returns a contiguous tensor.

    Reads strided `[B, S, H, head_dim]` views (e.g. fused-qkv slices) directly,
    absorbing the `.contiguous()` materialization the eager path needs.
    Returns None when the input is not supported (caller falls back).
    """
    head_dim = x.shape[-1]
    if not can_use_qk_rmsnorm_native(x, weight, head_dim):
        return None
    token_stride = _qk_head_token_stride(x, head_dim)
    if token_stride is None:
        return None
    nheads = x.shape[2]
    n_rows = x.shape[0] * x.shape[1] * nheads
    rows_per_prog = 8
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    with torch.get_device_module().device(x.device):
        _qk_rmsnorm_native_kernel[grid](
            y,
            x,
            weight,
            token_stride,
            nheads,
            n_rows,
            head_dim,
            eps,
            rows_per_prog=rows_per_prog,
            num_warps=8,
        )
    return y


__all__ = ["can_use_qk_rmsnorm_native", "zimage_qk_rmsnorm_native"]
