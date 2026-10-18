# SPDX-License-Identifier: Apache-2.0
"""MOSSImage RMSNorm: FP32 statistics, BF16 rounding before learned scale.

For widths 128/3072 with at least 16 rows, reproduce ATen's vectorized FP32
mean reduction in one kernel. CUDA Reduce.cuh uses 32 lanes, four accumulators
per lane, sequential fragment accumulation, then descending shuffle offsets.
Other shapes retain ATen's mean with fused pointwise operations around it.
Callers must verify against the live eager chain for each reduction regime.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import cuda_rsqrtf, mul_rn_f32
from sglang.kernels.ops.diffusion.common.platform import is_cuda
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _fold_halves(values, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    # Match shfl_down: first half + second half, largest offset first.
    lo, hi = tl.split(tl.permute(tl.reshape(values, (ROWS, 2, WIDTH)), (0, 2, 1)))
    return lo + hi


@triton.jit
def _ordered_rstd(
    X,
    M: tl.constexpr,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    lanes = tl.arange(0, 32)
    parts = tl.arange(0, 4)
    acc = tl.full((ROWS, 32, 4), 0, tl.float32)
    # ATen vectorizes the already-rounded FP32 squares, four per lane.
    # Unroll to schedule loads ahead of the dependent accumulation chain.
    for chunk in tl.static_range(DIM // 128):
        cols = chunk * 128 + lanes[:, None] * 4 + parts[None, :]
        x = tl.load(
            X + rows[:, None, None] * DIM + cols[None, :, :], rows[:, None, None] < M, 0
        ).to(tl.float32)
        acc = acc + mul_rn_f32(x, x)
    even, odd = tl.split(tl.reshape(acc, (ROWS, 32, 2, 2)))
    a, c = tl.split(even)
    b, d = tl.split(odd)
    total = ((a + b) + c) + d
    total = _fold_halves(total, ROWS, 16)
    total = _fold_halves(total, ROWS, 8)
    total = _fold_halves(total, ROWS, 4)
    total = _fold_halves(total, ROWS, 2)
    total = _fold_halves(total, ROWS, 1)
    mean = mul_rn_f32(tl.reshape(total, (ROWS,)), 1.0 / DIM)
    return cuda_rsqrtf(mean + 1e-6)


@triton.jit
def _ordered_norm_kernel(
    X,
    Weight,
    Out,
    M: tl.constexpr,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    rstd = _ordered_rstd(X, M, DIM, ROWS)

    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] < M) & (cols[None, :] < DIM)
    x = tl.load(X + rows[:, None] * DIM + cols[None, :], mask, 0).to(tl.float32)
    weight = tl.load(Weight + cols, cols < DIM, 0).to(tl.float32)
    normalized = mul_rn_f32(x, rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    tl.store(
        Out + rows[:, None] * DIM + cols[None, :],
        mul_rn_f32(normalized, weight[None, :]),
        mask,
    )


@triton.jit
def _norm_modulate_kernel(
    X,
    Weight,
    Scale,
    Shift,
    Out,
    M: tl.constexpr,
    SEQ: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    SHIFT_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, 4096)
    mask = cols < 3072
    rstd = _ordered_rstd(X, M, 3072, 1)
    x = tl.load(X + row * 3072 + cols, mask, 0).to(tl.float32)
    weight = tl.load(Weight + cols, mask, 0).to(tl.float32)
    norm = mul_rn_f32(x, tl.sum(rstd, 0)).to(tl.bfloat16).to(tl.float32)
    norm = mul_rn_f32(norm, weight).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(Scale + (row // SEQ) * SCALE_STRIDE + cols, mask, 0).to(tl.float32)
    shift = tl.load(Shift + (row // SEQ) * SHIFT_STRIDE + cols, mask, 0).to(tl.float32)
    factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    scaled = mul_rn_f32(norm, factor).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + row * 3072 + cols, scaled + shift, mask)


@triton.jit
def _norm_rope_kernel(X, Weight, Cos, Sin, Out, M: tl.constexpr, HEADS: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    cols = tl.arange(0, 128)
    mask = rows[:, None] < M
    rstd = _ordered_rstd(X, M, 128, 16)
    x = tl.load(X + rows[:, None] * 128 + cols[None, :], mask, 0).to(tl.float32)
    weight = tl.load(Weight + cols).to(tl.float32)
    norm = mul_rn_f32(x, rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    norm = mul_rn_f32(norm, weight[None, :]).to(tl.bfloat16).to(tl.float32)
    lo, hi = tl.split(tl.permute(tl.reshape(norm, (16, 2, 64)), (0, 2, 1)))
    half_cols = tl.arange(0, 64)[None, :]
    rope_offsets = (rows // HEADS)[:, None] * 128 + half_cols
    cos_lo = tl.load(Cos + rope_offsets, mask, 0).to(tl.float32)
    cos_hi = tl.load(Cos + rope_offsets + 64, mask, 0).to(tl.float32)
    sin_lo = tl.load(Sin + rope_offsets, mask, 0).to(tl.float32)
    sin_hi = tl.load(Sin + rope_offsets + 64, mask, 0).to(tl.float32)
    out_lo = mul_rn_f32(lo, cos_lo).to(tl.bfloat16).to(tl.float32) + mul_rn_f32(
        -hi, sin_lo
    ).to(tl.bfloat16).to(tl.float32)
    out_hi = mul_rn_f32(hi, cos_hi).to(tl.bfloat16).to(tl.float32) + mul_rn_f32(
        lo, sin_hi
    ).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + rows[:, None] * 128 + half_cols, out_lo, mask)
    tl.store(Out + rows[:, None] * 128 + half_cols + 64, out_hi, mask)


@triton.jit
def _square_kernel(X, Squared, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    tl.store(Squared + offsets, mul_rn_f32(x, x), offsets < N)


@triton.jit
def _normalize_kernel(
    X, Mean, Weight, Out, N: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask, 0).to(tl.float32)
    mean = tl.load(Mean + offsets // DIM, mask, 0)
    weight = tl.load(Weight + offsets % DIM, mask, 0).to(tl.float32)
    normalized = mul_rn_f32(x, cuda_rsqrtf(mean + 1e-6))
    # This rounding precedes gamma in MOSSImage, unlike conventional RMSNorm.
    normalized = normalized.to(tl.bfloat16).to(tl.float32)
    tl.store(Out + offsets, mul_rn_f32(normalized, weight), mask)


def can_use_rmsnorm_cast_before_weight(
    *,
    device: torch.device,
    dtype: torch.dtype,
    contiguous: bool = True,
    eps: float = 1e-6,
) -> bool:
    return (
        is_cuda()
        and device.type == "cuda"
        and dtype == torch.bfloat16
        and eps == 1e-6
        and contiguous
    )


def can_use_rmsnorm_cast_before_weight_modulate(
    *, device: torch.device, dtype: torch.dtype, dim: int, rows: int
) -> bool:
    return (
        can_use_rmsnorm_cast_before_weight(device=device, dtype=dtype)
        and dim == 3072
        and rows >= 16
        and rows * dim < 2**29
    )


def can_use_rmsnorm_cast_before_weight_rope(
    *, device: torch.device, dtype: torch.dtype, dim: int, rows: int
) -> bool:
    return (
        can_use_rmsnorm_cast_before_weight(device=device, dtype=dtype)
        and dim == 128
        and rows >= 16
        and rows * dim < 2**29
    )


def _valid_rmsnorm_cast_before_weight(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        is_cuda()
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and weight.shape == (x.shape[-1],)
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.numel() > 0
    )


def _fake_rmsnorm_cast_before_weight(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


def _valid_rmsnorm_cast_before_weight_modulate(x, weight, scale, shift):
    return (
        _valid_rmsnorm_cast_before_weight(x, weight)
        and x.ndim == 3
        and x.shape[-1] == 3072
        and 16 <= x.numel() // 3072
        and x.numel() < 2**29
        and scale.shape == shift.shape == (x.shape[0], 3072)
        and scale.dtype == shift.dtype == x.dtype
        and scale.device == shift.device == x.device
        and scale.stride(-1) == shift.stride(-1) == 1
    )


def _valid_rmsnorm_cast_before_weight_rope(x, weight, cos, sin):
    return (
        _valid_rmsnorm_cast_before_weight(x, weight)
        and x.ndim == 4
        and x.shape[-1] == 128
        and 16 <= x.numel() // 128
        and x.numel() < 2**29
        and cos.shape == sin.shape == (x.shape[0] * x.shape[1], 128)
        and cos.dtype == sin.dtype == x.dtype
        and cos.device == sin.device == x.device
        and cos.is_contiguous()
        and sin.is_contiguous()
    )


def _fake_rmsnorm_epilogue(x, weight, first, second):
    return torch.empty_like(x)


@register_custom_op(
    op_name="rmsnorm_cast_before_weight_modulate_bitexact",
    mutates_args=[],
    fake_impl=_fake_rmsnorm_epilogue,
)
def rmsnorm_cast_before_weight_modulate_bitexact(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    if not _valid_rmsnorm_cast_before_weight_modulate(x, weight, scale, shift):
        raise ValueError("Unsupported MOSSImage RMSNorm + modulation inputs.")
    with torch.cuda.device(x.device):
        out = torch.empty_like(x)
        rows = x.numel() // 3072
        _norm_modulate_kernel[(rows,)](
            x,
            weight,
            scale,
            shift,
            out,
            rows,
            x.shape[1],
            scale.stride(0),
            shift.stride(0),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@register_custom_op(
    op_name="rmsnorm_cast_before_weight_rope_bitexact",
    mutates_args=[],
    fake_impl=_fake_rmsnorm_epilogue,
)
def rmsnorm_cast_before_weight_rope_bitexact(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    if not _valid_rmsnorm_cast_before_weight_rope(x, weight, cos, sin):
        raise ValueError("Unsupported MOSSImage RMSNorm + RoPE inputs.")
    with torch.cuda.device(x.device):
        out = torch.empty_like(x)
        rows = x.numel() // 128
        _norm_rope_kernel[(triton.cdiv(rows, 16),)](
            x,
            weight,
            cos,
            sin,
            out,
            rows,
            x.shape[2],
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@register_custom_op(
    op_name="rmsnorm_cast_before_weight_bitexact",
    mutates_args=[],
    fake_impl=_fake_rmsnorm_cast_before_weight,
)
def rmsnorm_cast_before_weight_bitexact(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Match ``(x.float() * rsqrt(mean(x.float()**2) + 1e-6)).bf16() * w``."""
    if not _valid_rmsnorm_cast_before_weight(x, weight):
        raise ValueError("RMSNorm requires contiguous CUDA BF16 inputs and weight.")
    with torch.cuda.device(x.device):
        out = torch.empty_like(x)
        dim = x.shape[-1]
        rows = x.numel() // dim
        # At least 16 rows keeps ATen's reduction width at 32 lanes for
        # these dimensions. Smaller row counts can select a different tree.
        # Larger FP32 workspaces can make TensorIterator split the reduction.
        if dim in (128, 3072) and rows >= 16 and x.numel() < 2**29:
            _ordered_norm_kernel[(triton.cdiv(rows, 4),)](
                x,
                weight,
                out,
                rows,
                dim,
                4,
                triton.next_power_of_2(dim),
                num_warps=4,
                enable_fp_fusion=False,
            )
            return out
        squared = torch.empty_like(x, dtype=torch.float32)
        grid = (triton.cdiv(x.numel(), 1024),)
        _square_kernel[grid](x, squared, x.numel(), 1024, enable_fp_fusion=False)
        mean = squared.mean(-1, keepdim=True)
        _normalize_kernel[grid](
            x, mean, weight, out, x.numel(), x.shape[-1], 1024, enable_fp_fusion=False
        )
    return out
