"""Forward RMSNorm on gfx950: one row per program, FP32 statistics, ``x * rstd * weight`` in FP32.

``norm_fwd`` with ``norm='rms'`` and ``rmsnorm`` on top of it, as in ``sm100.py``. The sm100 kernels do not compile
on gfx950: their pinned layouts assume 32-lane warps.

FP16/BF16 rows of 4096 features take a TLX kernel; other rows take a plain kernel with 8 waves above 2048 features and
4 below. The choice depends only on the row width and dtype, so a row's result does not depend on the batch.

The TLX kernel maps elements to its 8 waves as the 8-wave plain kernel does (8 contiguous elements per lane, 512 per
wave), so each wave's partial sum is the plain kernel's, and it adds the 8 partials in the plain kernel's tree order:
the two kernels are bitwise equal at 4096 features. The plain kernel combines the partials with two LDS round trips and
two barriers (write, reduce in one wave, write, read); the TLX kernel pins a layout in which every lane holds all 8
partials, so it needs one round trip and one barrier. It also loads the weight with the row, before the reduction.

Rows of up to 16384 features. LayerNorm, residual, bias and post-norm addend inputs, output buffers and autograd are not
provided on this arch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import UnsupportedOp

MAX_FEATURES = 16384
TLX_FEATURES = (4096, )
# Element offsets at or past this need 64-bit row offsets.
INT32_OFFSETS = 1 << 31


@triton.jit
def _rmsnorm_kernel(X, W, Y, RSTD, stride_x, eps, N: tl.constexpr, BLOCK_N: tl.constexpr, HAS_WEIGHT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x) / N + eps)
    y = x * rstd
    if HAS_WEIGHT:
        y = y * tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + offs, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rmsnorm_tlx_kernel(X, W, Y, RSTD, stride_x, eps, N: tl.constexpr, HAS_WEIGHT: tl.constexpr, ROW64: tl.constexpr):
    # An [8, 512] tile: tile row = wave, 8 contiguous elements per lane.
    tile: tl.constexpr = tlx.layout(shape=((64, 8), (8, )), stride=((8, 512), (1, )))
    replicated: tl.constexpr = tlx.layout(shape=((64, 8), (8, )), stride=((0, 0), (1, )))
    row = tl.program_id(0).to(tl.int64) if ROW64 else tl.program_id(0)
    offs = tlx.require_layout(tl.arange(0, 8)[:, None] * (N // 8) + tl.arange(0, N // 8)[None, :], tile)
    if HAS_WEIGHT:
        w = tl.load(W + offs).to(tl.float32)
    x = tl.load(X + row * stride_x + offs).to(tl.float32)
    partials = tlx.require_layout(tl.sum(x * x, axis=1), replicated)
    # The plain kernel's cross-wave tree: ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)).
    pairs = tl.sum(tl.reshape(partials, (2, 4)), axis=0)
    quads = tl.sum(tl.reshape(pairs, (2, 2)), axis=0)
    rstd = tl.rsqrt(tl.sum(quads, axis=0) / N + eps)
    y = x * rstd
    if HAS_WEIGHT:
        y = y * w
    tl.store(Y + row * N + offs, y.to(Y.dtype.element_ty))
    tl.store(RSTD + row, rstd)


def use_tlx_kernel(N: int, dtype) -> bool:
    return N in TLX_FEATURES and dtype in (torch.float16, torch.bfloat16)


def norm_fwd(x, weight=None, bias=None, *, norm='layer', eps=1e-06, residual=None, post_norm_addend=None,
             out_dtype=None, residual_dtype=None, store_stats=True, residual_norm_weight=None, residual_norm_bias=None,
             residual_norm_rstd=None, residual_norm_mean=None, out=None, mean_out=None, rstd_out=None,
             return_residual=None):
    if norm != 'rms':
        raise UnsupportedOp("tlx.ops.norm_fwd on gfx950 supports only norm='rms'")
    unsupported = dict(bias=bias, residual=residual, post_norm_addend=post_norm_addend, residual_dtype=residual_dtype,
                       residual_norm_weight=residual_norm_weight, residual_norm_bias=residual_norm_bias,
                       residual_norm_rstd=residual_norm_rstd, residual_norm_mean=residual_norm_mean, out=out,
                       mean_out=mean_out, rstd_out=rstd_out)
    given = [name for name, value in unsupported.items() if value is not None]
    if return_residual:
        given.append('return_residual')
    if out_dtype not in (None, x.dtype):
        given.append('out_dtype')
    if given:
        raise UnsupportedOp(f"tlx.ops.norm_fwd on gfx950 does not support {', '.join(given)}")
    M, N = x.shape
    if N > MAX_FEATURES:
        raise UnsupportedOp(f"tlx.ops.norm_fwd on gfx950 keeps a row in registers: at most {MAX_FEATURES} features")
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    if M:
        if use_tlx_kernel(N, x.dtype):
            row64 = (M - 1) * max(x.stride(0), N) + N >= INT32_OFFSETS
            _rmsnorm_tlx_kernel[(M, )](x, weight, y, rstd, x.stride(0), float(eps), N, weight is not None, row64,
                                       num_warps=8, num_stages=1)
        else:
            block_n = triton.next_power_of_2(N)
            _rmsnorm_kernel[(M, )](x, weight, y, rstd, x.stride(0), float(eps), N, block_n, weight is not None,
                                   num_warps=8 if block_n >= 4096 else 4)
    return (y, None, None, rstd if store_stats else None)


def rmsnorm(x, weight, bias=None, residual=None, post_norm_addend=None, eps=1e-05):
    y, _, _, rstd = norm_fwd(x, weight, bias, norm='rms', eps=eps, residual=residual, post_norm_addend=post_norm_addend)
    return (y, None, rstd)
