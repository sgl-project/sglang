# SPDX-License-Identifier: Apache-2.0
"""Bit-exact fused LayerNorm + adaLN modulate Triton kernels for Intel XPU.

Two eager chains, each replaced by one launch that reproduces the eager result
bit for bit on XPU (``torch.equal``):

- ``xpu_layernorm_modulate``: ``LN(x) * (1 + scale) + shift``
- ``xpu_gated_resnorm``: ``r = residual + update * gate`` followed by the same
  LN + modulate on ``r``; returns ``(out, r)``

The CUDA kernels in ``kda_kernels/layernorm_modulate_triton.py`` replicate
CUDA aten's reduction with inline PTX, so they cannot run here. These kernels
instead replicate torch-xpu-ops ``VectorizedLayerNormKernelFunctor`` for bf16
rows (``src/ATen/native/xpu/sycl/LayerNormKernels.cpp``, ``compute_stats``):

- Work-group of 1024 work-items (32 sub-groups of SIMD 32), which torch-xpu
  selects whenever ``N / 4`` exceeds the 1024 max work-group size.
- Work-item ``t`` Welford-pushes the 4-element vectors ``t, t + 1024, ...``,
  each scalar as ``mean' = fma(delta, rcp(count), mean)``,
  ``m2' = fma(delta, val - mean', m2)`` with ``rcp`` being
  ``sycl::native::recip``, read from a per-device table of integer-count
  reciprocals (``_native_recip_table``).
- Lanes fold with ``shift_group_left`` offsets 16..1 (self = lower lane), then
  sub-groups fold with the same pairing through SLM, via ``WelfordCombine``
  with contracted multiply-adds.
- ``rstd = rsqrt(div_rn(m2, N) + eps)``; ``y = bf16(rstd * (x - mean))``; the
  modulate chain rounds to bf16 after each op like the eager kernels.

Measured bit-exact on Arc Pro B70 (torch 2.13.0+xpu). Other devices or torch
versions may pick a different work-group size, so callers must verify
``torch.equal`` against the eager chain once and fall back on mismatch.
"""

from __future__ import annotations

import os

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

try:
    from triton.language.extra.intel import libdevice  # type: ignore
except ImportError:  # non-XPU Triton build
    # The kernels below reference ``libdevice``, but Triton resolves a jit
    # function's globals when it compiles it, and these only compile on first
    # launch -- which the ``is_xpu`` checks in the ``can_use_*`` predicates
    # prevent off XPU. Importing this module must stay free of XPU-only
    # dependencies: the package facade resolves ``_EXPORTS`` eagerly on
    # attribute access, so a hard import here would make Intel Triton an
    # import-time requirement for every caller of the facade, on every platform.
    libdevice = None

# torch-xpu's vectorized LayerNorm geometry for rows with N / 4 > 1024.
_WG = 1024
_SIMD = 32
_VEC = 4
# 8-byte alignment is what torch-xpu requires to take the vectorized kernel for bf16.
_ALIGN = _VEC * 2
# Triton threads per program; the result does not depend on it (layout only).
_NUM_WARPS = int(os.environ.get("SGLANG_XPU_LN_MOD_NUM_WARPS", "4"))


@triton.jit
def _welford_combine(b_mean, b_m2, b_cnt, a_mean, a_m2, a_cnt, rcp_ptr):
    # torch-xpu WelfordCombine(dataB=self, dataA=other).
    delta = b_mean - a_mean
    cnt = a_cnt + b_cnt
    coef = tl.load(rcp_ptr + cnt.to(tl.int32))
    n_a = a_cnt * coef
    n_b = b_cnt * coef
    mean = tl.fma(n_a, a_mean, n_b * b_mean)
    m2 = tl.fma(delta * delta * a_cnt, n_b, a_m2 + b_m2)
    pos = cnt > 0
    return tl.where(pos, mean, 0.0), tl.where(pos, m2, 0.0), cnt


@triton.jit
def _fold_step(mean, m2, cnt, rcp_ptr, ROWS: tl.constexpr, W: tl.constexpr):
    # One tree level over the last dim: lower half is self, upper half is other.
    ml, mu = tl.split(tl.permute(tl.reshape(mean, (ROWS, 2, W // 2)), (0, 2, 1)))
    sl, su = tl.split(tl.permute(tl.reshape(m2, (ROWS, 2, W // 2)), (0, 2, 1)))
    cl, cu = tl.split(tl.permute(tl.reshape(cnt, (ROWS, 2, W // 2)), (0, 2, 1)))
    return _welford_combine(ml, sl, cl, mu, su, cu, rcp_ptr)


@triton.jit
def _fold_32(mean, m2, cnt, rcp_ptr, ROWS: tl.constexpr):
    mean, m2, cnt = _fold_step(mean, m2, cnt, rcp_ptr, ROWS, 32)
    mean, m2, cnt = _fold_step(mean, m2, cnt, rcp_ptr, ROWS, 16)
    mean, m2, cnt = _fold_step(mean, m2, cnt, rcp_ptr, ROWS, 8)
    mean, m2, cnt = _fold_step(mean, m2, cnt, rcp_ptr, ROWS, 4)
    mean, m2, cnt = _fold_step(mean, m2, cnt, rcp_ptr, ROWS, 2)
    return mean, m2, cnt


@triton.jit
def _welford_push(v, live, mean, m2, cnt, rcp_ptr):
    new_cnt = cnt + 1.0
    rcp = tl.load(rcp_ptr + new_cnt.to(tl.int32))
    delta = v - mean
    new_mean = tl.fma(delta, rcp, mean)
    new_m2 = tl.fma(delta, v - new_mean, m2)
    return (
        tl.where(live, new_mean, mean),
        tl.where(live, new_m2, m2),
        tl.where(live, new_cnt, cnt),
    )


@triton.jit
def _load_vec4(ptr, offs, live):
    # One aligned 4-element vector per work-item, split into its scalars in order.
    v = tl.load(
        ptr + offs[:, None] + tl.arange(0, 4)[None, :], mask=live[:, None], other=0.0
    )
    lo, hi = tl.split(tl.reshape(v, (v.shape[0], 2, 2)))  # (k0, k2), (k1, k3)
    v0, v2 = tl.split(lo)
    v1, v3 = tl.split(hi)
    return v0, v1, v2, v3


@triton.jit
def _gated_value(r_ptr, u_ptr, g_ptr, offs, live):
    # Eager ``residual + update * gate``: two bf16-rounded ops.
    r0, r1, r2, r3 = _load_vec4(r_ptr, offs, live)
    u0, u1, u2, u3 = _load_vec4(u_ptr, offs, live)
    g0, g1, g2, g3 = _load_vec4(g_ptr, offs, live)
    x0 = (
        r0.to(tl.float32)
        + (u0.to(tl.float32) * g0.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    x1 = (
        r1.to(tl.float32)
        + (u1.to(tl.float32) * g1.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    x2 = (
        r2.to(tl.float32)
        + (u2.to(tl.float32) * g2.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    x3 = (
        r3.to(tl.float32)
        + (u3.to(tl.float32) * g3.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    return x0, x1, x2, x3


@triton.jit
def _modulate(x, s, b, mean, rstd):
    y = (rstd * (x.to(tl.float32) - mean)).to(tl.bfloat16).to(tl.float32)
    one_s = (1.0 + s.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    p = (y * one_s).to(tl.bfloat16).to(tl.float32)
    return (p + b.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _store_vec4(ptr, offs, live, v0, v1, v2, v3):
    # Inverse of _load_vec4: one 4-element vector per work-item.
    quad = tl.reshape(tl.join(tl.join(v0, v2), tl.join(v1, v3)), (v0.shape[0], 4))
    tl.store(ptr + offs[:, None] + tl.arange(0, 4)[None, :], quad, mask=live[:, None])


@triton.jit
def _xpu_ln_modulate_kernel(
    x_ptr,
    u_ptr,
    g_ptr,
    s_ptr,
    b_ptr,
    o_ptr,
    ro_ptr,
    rcp_ptr,
    eps,
    N: tl.constexpr,
    WG: tl.constexpr,
    SIMD: tl.constexpr,
    GATED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_off = row * N
    t = tl.arange(0, WG)
    NVEC: tl.constexpr = N // 4
    ITERS: tl.constexpr = (NVEC + WG - 1) // WG

    mean = tl.zeros((WG,), tl.float32)
    m2 = tl.zeros((WG,), tl.float32)
    cnt = tl.zeros((WG,), tl.float32)
    for j in tl.static_range(ITERS):
        vec = j * WG + t
        live = vec < NVEC
        offs = row_off + vec * 4
        if GATED:
            x0, x1, x2, x3 = _gated_value(x_ptr, u_ptr, g_ptr - row_off, offs, live)
            _store_vec4(ro_ptr, offs, live, x0, x1, x2, x3)
        else:
            x0, x1, x2, x3 = _load_vec4(x_ptr, offs, live)
        mean, m2, cnt = _welford_push(x0.to(tl.float32), live, mean, m2, cnt, rcp_ptr)
        mean, m2, cnt = _welford_push(x1.to(tl.float32), live, mean, m2, cnt, rcp_ptr)
        mean, m2, cnt = _welford_push(x2.to(tl.float32), live, mean, m2, cnt, rcp_ptr)
        mean, m2, cnt = _welford_push(x3.to(tl.float32), live, mean, m2, cnt, rcp_ptr)

    # Lane fold inside each sub-group, then the sub-group fold.
    SG: tl.constexpr = WG // SIMD
    mean, m2, cnt = _fold_32(
        tl.reshape(mean, (SG, SIMD)),
        tl.reshape(m2, (SG, SIMD)),
        tl.reshape(cnt, (SG, SIMD)),
        rcp_ptr,
        SG,
    )
    mean, m2, cnt = _fold_32(
        tl.reshape(mean, (1, SG)),
        tl.reshape(m2, (1, SG)),
        tl.reshape(cnt, (1, SG)),
        rcp_ptr,
        1,
    )
    mean = tl.sum(mean, axis=1)
    var = libdevice.div_rn(tl.sum(m2, axis=1), tl.full((1,), N, tl.float32))
    rstd = libdevice.rsqrt(var + eps)
    mean = tl.reshape(mean, (1,))
    rstd = tl.reshape(rstd, (1,))

    for j in tl.static_range(ITERS):
        vec = j * WG + t
        live = vec < NVEC
        offs = row_off + vec * 4
        if GATED:
            x0, x1, x2, x3 = _gated_value(x_ptr, u_ptr, g_ptr - row_off, offs, live)
        else:
            x0, x1, x2, x3 = _load_vec4(x_ptr, offs, live)
        s0, s1, s2, s3 = _load_vec4(s_ptr - row_off, offs, live)
        b0, b1, b2, b3 = _load_vec4(b_ptr - row_off, offs, live)
        _store_vec4(
            o_ptr,
            offs,
            live,
            _modulate(x0, s0, b0, mean, rstd),
            _modulate(x1, s1, b1, mean, rstd),
            _modulate(x2, s2, b2, mean, rstd),
            _modulate(x3, s3, b3, mean, rstd),
        )


@triton.jit
def _native_recip_table_kernel(out_ptr, BLOCK: tl.constexpr):
    # Only this exact configuration (one program, BLOCK 8192, 4 warps) was verified to match
    # sycl::native::recip for every count 1..8191 on Arc Pro B70; fast_dividef lowers to
    # something else in other layouts, so the LN kernels read the table instead.
    i = tl.arange(0, BLOCK)
    tl.store(
        out_ptr + i,
        libdevice.fast_dividef(tl.full((BLOCK,), 1.0, tl.float32), i.to(tl.float32)),
    )


_RECIP_TABLES: dict[torch.device, torch.Tensor] = {}
_RECIP_TABLE_SIZE = 8192


def _native_recip_table(device: torch.device) -> torch.Tensor:
    """``sycl::native::recip(float(i))`` for i < 8192, built once per device."""
    table = _RECIP_TABLES.get(device)
    if table is None:
        table = torch.empty(_RECIP_TABLE_SIZE, dtype=torch.float32, device=device)
        _native_recip_table_kernel[(1,)](table, BLOCK=_RECIP_TABLE_SIZE, num_warps=4)
        _RECIP_TABLES[device] = table
    return table


def _row_vector(
    t: torch.Tensor, hidden: int, device: torch.device
) -> torch.Tensor | None:
    """A contiguous per-channel ``[hidden]`` view of scale/shift/gate, else None."""
    if not (
        isinstance(t, torch.Tensor)
        and t.dtype is torch.bfloat16
        and t.device == device
        and t.numel() == hidden
        and t.shape[-1] == hidden
        and t.is_contiguous()
        and t.data_ptr() % _ALIGN == 0
    ):
        return None
    return t.reshape(hidden)


def _eligible_rows(x: torch.Tensor) -> bool:
    hidden = x.shape[-1]
    return (
        x.is_xpu
        and x.dtype is torch.bfloat16
        and x.numel() > 0
        and x.is_contiguous()
        and x.data_ptr() % _ALIGN == 0
        and hidden % _VEC == 0
        and hidden // _VEC > _WG
        and hidden < _RECIP_TABLE_SIZE
    )


def can_use_xpu_layernorm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    if not _eligible_rows(x):
        return False
    hidden = x.shape[-1]
    return (
        _row_vector(scale, hidden, x.device) is not None
        and _row_vector(shift, hidden, x.device) is not None
    )


def xpu_layernorm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    out = torch.empty_like(x)
    s = _row_vector(scale, hidden, x.device)
    b = _row_vector(shift, hidden, x.device)
    _xpu_ln_modulate_kernel[(rows,)](
        x,
        x,
        x,
        s,
        b,
        out,
        out,
        _native_recip_table(x.device),
        float(eps),
        N=hidden,
        WG=_WG,
        SIMD=_SIMD,
        GATED=False,
        num_warps=_NUM_WARPS,
    )
    return out


def can_use_xpu_gated_resnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    if not (
        _eligible_rows(residual)
        and isinstance(update, torch.Tensor)
        and update.shape == residual.shape
        and update.dtype is residual.dtype
        and update.device == residual.device
        and update.is_contiguous()
        and update.data_ptr() % _ALIGN == 0
    ):
        return False
    hidden = residual.shape[-1]
    return all(
        _row_vector(t, hidden, residual.device) is not None
        for t in (gate, scale, shift)
    )


def xpu_gated_resnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = residual.shape[-1]
    rows = residual.numel() // hidden
    out = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    g = _row_vector(gate, hidden, residual.device)
    s = _row_vector(scale, hidden, residual.device)
    b = _row_vector(shift, hidden, residual.device)
    _xpu_ln_modulate_kernel[(rows,)](
        residual,
        update,
        g,
        s,
        b,
        out,
        residual_out,
        _native_recip_table(residual.device),
        float(eps),
        N=hidden,
        WG=_WG,
        SIMD=_SIMD,
        GATED=True,
        num_warps=_NUM_WARPS,
    )
    return out, residual_out


__all__ = [
    "can_use_xpu_gated_resnorm",
    "can_use_xpu_layernorm_modulate",
    "xpu_gated_resnorm",
    "xpu_layernorm_modulate",
]
