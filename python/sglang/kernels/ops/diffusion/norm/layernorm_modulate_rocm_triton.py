# SPDX-License-Identifier: Apache-2.0
"""ROCm BF16 LayerNorm + modulation for FLUX.1's 3072-wide rows.

FP32 Welford statistics with explicit BF16 eager epilogue rounding. The
reduction follows aten's 256-thread ROCm vectorized LayerNorm. Exactness is
version-dependent: model callers must verify against the live aten dispatch
using BitExactFusionGate, or explicitly use a quality-gated fusion site.
No learned affine parameters, residual addition, or RMSNorm are supported.
"""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _is_validated_rocm_device(device_index: int) -> bool:
    """Only gfx90a has numerical and performance qualification for this backend."""
    if not torch.version.hip:
        return False
    arch = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx90a"


@triton.jit
def _combine_welford(lower, lower_m2, lower_count, upper, upper_m2, upper_count):
    # Balanced reductions of 256 equal-count lanes: the two coefficients
    # are exactly 1/2 at every level. Let Triton lower the reduction instead
    # of materializing reshaped intermediate trees in registers/shared memory.
    delta = lower - upper
    mean = tl.fma(upper, 0.5, lower * 0.5)
    m2 = tl.fma((delta * delta) * upper_count, 0.5, upper_m2 + lower_m2)
    return mean, m2, lower_count + upper_count


@triton.jit
def _row_moments(
    X, row, row_valid, D: tl.constexpr, ROWS: tl.constexpr, VARIANCE_FMA: tl.constexpr
):
    # aten ROCm vectorized LayerNorm: 256 threads, four adjacent elements
    # per vector, serial Welford pushes, then four 64-lane wave reductions.
    lanes = tl.arange(0, 256)
    mean = tl.full((ROWS, 256), 0.0, tl.float32)
    m2 = tl.full((ROWS, 256), 0.0, tl.float32)
    for i in tl.static_range(D // 1024):
        for v in tl.static_range(4):
            value = tl.load(
                X + row[:, None] * D + i * 1024 + lanes[None, :] * 4 + v,
                row_valid[:, None],
                other=0,
            ).to(tl.float32)
            delta = value - mean
            mean = tl.fma(delta, 1.0 / (i * 4 + v + 1), mean)
            if VARIANCE_FMA:
                m2 = tl.fma(delta, value - mean, m2)
            else:
                m2 = m2 + delta * (value - mean)
    count = tl.full((ROWS, 256), D // 256, tl.float32)
    mean, m2, _ = tl.reduce((mean, m2, count), 1, _combine_welford)
    return mean, m2 / D


@triton.jit
def _layernorm_modulate_rocm_kernel(
    X,
    SCALE,
    SHIFT,
    Y,
    SEQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE_BATCH: tl.constexpr,
    SHIFT_BATCH: tl.constexpr,
    ROWS: tl.constexpr = 1,
    NROWS: tl.constexpr = 0,
    VARIANCE_FMA: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = (row < NROWS) if NROWS else tl.full((ROWS,), True, tl.int1)
    mean, variance = _row_moments(X, row, row_valid, D, ROWS, VARIANCE_FMA)
    batch = row // SEQ
    rstd = tl.rsqrt(variance + EPS)
    # Stream the epilogue in small tiles: retaining a full padded row plus
    # scale/shift intermediates creates excessive register pressure on gfx90a.
    for chunk in range(D // 256):
        col = chunk * 256 + tl.arange(0, 256)
        valid = row_valid[:, None]
        x = tl.load(X + row[:, None] * D + col[None, :], valid, other=0).to(tl.float32)
        normalized = ((x - mean[:, None]) * rstd[:, None]).to(tl.bfloat16)
        scale = tl.load(
            SCALE + batch[:, None] * SCALE_BATCH + col[None, :], valid, other=0
        ).to(tl.float32)
        shift = tl.load(
            SHIFT + batch[:, None] * SHIFT_BATCH + col[None, :], valid, other=0
        ).to(tl.float32)
        factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
        product = (normalized.to(tl.float32) * factor).to(tl.bfloat16).to(tl.float32)
        out = (product + shift).to(tl.bfloat16)
        tl.store(Y + row[:, None] * D + col[None, :], out, valid)


def can_use_layernorm_modulate_rocm(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    """Accept contiguous BF16 BLC activations and BxD or Bx1xD modulation.

    Modulation can be a strided view of a larger projection (e.g. FLUX chunk).
    Only the hidden dimension must be contiguous; each batch stride is passed
    independently, without materializing copies in the hot path.
    """
    if (
        not torch.version.hip
        or torch.is_grad_enabled()
        or not x.is_cuda
        or x.ndim != 3
        or x.dtype != torch.bfloat16
        or x.shape[-1] != 3072
        or x.numel() == 0
        or not x.is_contiguous()
        # aten's four-element vectorized reduction requires aligned storage.
        or x.data_ptr() % (4 * x.element_size()) != 0
        or not _is_validated_rocm_device(x.device.index)
    ):
        return False
    batch, _, width = x.shape
    return all(
        t.device == x.device
        and t.dtype == torch.bfloat16
        and t.shape in ((batch, width), (batch, 1, width))
        and t.stride(-1) == 1
        for t in (scale, shift)
    )


def _layernorm_modulate_rocm_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    variance_fma: bool = False,
) -> torch.Tensor:
    """Compute affine-free LN(x) * (1 + scale) + shift in one GPU launch."""
    if not can_use_layernorm_modulate_rocm(x, scale, shift):
        raise ValueError("unsupported input for ROCm LayerNorm + modulation")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    batch, seq, width = x.shape
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _layernorm_modulate_rocm_kernel[(batch * seq,)](
            x,
            scale,
            shift,
            out,
            seq,
            width,
            eps,
            scale.stride(0),
            shift.stride(0),
            VARIANCE_FMA=variance_fma,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@torch.library.custom_op("sglang::layernorm_modulate_rocm", mutates_args=())
def _layernorm_modulate_rocm_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    variance_fma: bool,
) -> torch.Tensor:
    return _layernorm_modulate_rocm_impl(x, scale, shift, eps, variance_fma)


@_layernorm_modulate_rocm_op.register_fake
def _fake_layernorm_modulate_rocm(x, scale, shift, eps, variance_fma):
    return torch.empty_like(x)


def layernorm_modulate_rocm(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    variance_fma: bool = False,
) -> torch.Tensor:
    """Fuse LN/modulation with explicit online-variance contraction policy.

    Native ROCm compiler builds differ in variance multiply-add contraction.
    Model callers must verify a variant against their live reference; the
    default uses separate rounding, while ``variance_fma=True`` contracts it.
    """
    if torch.compiler.is_compiling():
        return _layernorm_modulate_rocm_op(x, scale, shift, eps, variance_fma)
    return _layernorm_modulate_rocm_impl(x, scale, shift, eps, variance_fma)
