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

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.environ import envs

try:
    from triton.language.extra.intel import libdevice  # type: ignore
except ImportError:  # non-XPU Triton build
    # The kernels only compile on launch, which selection prevents off XPU; a hard
    # import here would make Intel Triton a requirement of the facade everywhere.
    libdevice = None

# torch-xpu's vectorized LayerNorm geometry for rows with N / 4 > 1024.
_WG = 1024
_SIMD = 32
_VEC = 4
# 8-byte alignment is what torch-xpu requires to take the vectorized kernel for bf16.
_ALIGN = _VEC * 2


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
    seq_len,
    g_stride,
    s_stride,
    b_stride,
    eps,
    N: tl.constexpr,
    WG: tl.constexpr,
    SIMD: tl.constexpr,
    GATED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_off = row * N
    # Modulation rows are per batch; offsets below are relative to row_off.
    batch = row // seq_len
    g_row = g_ptr + batch * g_stride - row_off
    s_row = s_ptr + batch * s_stride - row_off
    b_row = b_ptr + batch * b_stride - row_off
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
            x0, x1, x2, x3 = _gated_value(x_ptr, u_ptr, g_row, offs, live)
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
            x0, x1, x2, x3 = _gated_value(x_ptr, u_ptr, g_row, offs, live)
        else:
            x0, x1, x2, x3 = _load_vec4(x_ptr, offs, live)
        s0, s1, s2, s3 = _load_vec4(s_row, offs, live)
        b0, b1, b2, b3 = _load_vec4(b_row, offs, live)
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


def can_use_xpu_layernorm_modulate(dtype: torch.dtype, hidden: int) -> bool:
    """Select the torch-xpu BF16 vectorized LayerNorm reduction replicated above."""
    return (
        libdevice is not None
        and dtype is torch.bfloat16
        and hidden % _VEC == 0
        and _WG < hidden // _VEC
        and hidden < _RECIP_TABLE_SIZE
    )


def _rows_ok(x: torch.Tensor) -> bool:
    # torch-xpu only takes the replicated vectorized path for aligned rows.
    return (
        x.is_xpu
        and x.dim() == 3
        and x.numel() > 0
        and x.is_contiguous()
        and x.data_ptr() % _ALIGN == 0
        and can_use_xpu_layernorm_modulate(x.dtype, x.shape[-1])
    )


def _mod_rows(t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    # (batch, hidden) rows with a packed last dim, possibly strided views of a
    # chunked adaLN projection; a (batch, 1, hidden) view squeezes to the same.
    batch, _, hidden = x.shape
    rows = t.squeeze(1) if t.dim() == 3 and t.shape[1] == 1 else t
    if (
        rows.shape != (batch, hidden)
        or rows.stride(1) != 1
        or rows.dtype is not x.dtype
        or rows.device != x.device
    ):
        return None
    return rows, rows.stride(0) if batch > 1 else hidden


def _update_ok(update: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        update.shape == residual.shape
        and update.dtype is residual.dtype
        and update.device == residual.device
        and update.is_contiguous()
        and update.data_ptr() % _ALIGN == 0
    )


def _checked_rows(x: torch.Tensor, name: str) -> None:
    if not _rows_ok(x):
        raise RuntimeError(
            f"{name} must be a contiguous, 8-byte aligned BF16 XPU [B, S, D] "
            "tensor with 4096 < D < 8192 and D divisible by 4"
        )


def _checked_mod(
    t: torch.Tensor, x: torch.Tensor, name: str
) -> tuple[torch.Tensor, int]:
    mod = _mod_rows(t, x)
    if mod is None:
        raise RuntimeError(f"{name} must have packed BF16 [B, D] rows on x's device")
    return mod


def _launch(x, update, gate, scale, shift, out, residual_out, eps, *, gated):
    (gate, g_stride), (scale, s_stride), (shift, b_stride) = gate, scale, shift
    batch, seq_len, hidden = x.shape
    _xpu_ln_modulate_kernel[(batch * seq_len,)](
        x,
        update,
        gate,
        scale,
        shift,
        out,
        residual_out,
        _native_recip_table(x.device),
        seq_len,
        g_stride,
        s_stride,
        b_stride,
        float(eps),
        N=hidden,
        WG=_WG,
        SIMD=_SIMD,
        GATED=gated,
        num_warps=envs.SGLANG_XPU_LN_MOD_NUM_WARPS.get(),
    )


def xpu_layernorm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    """``LN(x) * (1 + scale) + shift`` for BF16 XPU ``[B, S, D]``, bit-exact vs eager."""
    _checked_rows(x, "x")
    scale_rows = _checked_mod(scale, x, "scale")
    shift_rows = _checked_mod(shift, x, "shift")
    out = torch.empty_like(x)
    _launch(x, x, scale_rows, scale_rows, shift_rows, out, out, eps, gated=False)
    return out


def can_defer_xpu_gated_residual(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> bool:
    """Whether ``residual + update * gate`` can be left to the next block's norm."""
    return (
        not torch.compiler.is_compiling()
        and _rows_ok(residual)
        and _update_ok(update, residual)
        and _mod_rows(gate, residual) is not None
    )


def xpu_gated_resnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``r = residual + update * gate`` then LN+modulate of ``r``; returns ``(out, r)``."""
    _checked_rows(residual, "residual")
    if not _update_ok(update, residual):
        raise RuntimeError(
            "update must match residual's shape, dtype, device and layout"
        )
    gate_rows = _checked_mod(gate, residual, "gate")
    scale_rows = _checked_mod(scale, residual, "scale")
    shift_rows = _checked_mod(shift, residual, "shift")
    out = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    _launch(
        residual,
        update,
        gate_rows,
        scale_rows,
        shift_rows,
        out,
        residual_out,
        eps,
        gated=True,
    )
    return out, residual_out


__all__ = [
    "can_defer_xpu_gated_residual",
    "can_use_xpu_layernorm_modulate",
    "xpu_gated_resnorm",
    "xpu_layernorm_modulate",
]
