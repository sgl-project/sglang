# SPDX-License-Identifier: Apache-2.0
"""RMSNorm pointwise fusion preserving Torch's FP32 mean and cast boundaries."""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import cuda_rsqrtf
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _square_fp32_kernel(x_ptr, square_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x_ptr + index, index < N, 0).to(tl.float32)
    tl.store(square_ptr + index, value * value, index < N)


def _mean_square(x: torch.Tensor, block_size: int) -> torch.Tensor:
    squares = torch.empty_like(x, dtype=torch.float32)
    _square_fp32_kernel[(triton.cdiv(x.numel(), block_size),)](
        x, squares, x.numel(), block_size, enable_fp_fusion=False
    )
    return squares.mean(dim=-1, keepdim=True)


@triton.jit
def _rmsnorm_finish_kernel(
    x_ptr,
    variance_ptr,
    weight_ptr,
    out_ptr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    value = tl.load(x_ptr + index, mask, 0).to(tl.float32)
    variance = tl.load(variance_ptr + index // DIM, mask, 0)
    weight = tl.load(weight_ptr + index % DIM, mask, 0).to(tl.float32)
    # eager rounds the normalized activation before multiplying the weight
    normalized = (value * tl.rsqrt(variance + EPS)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + index, normalized.to(tl.float32) * weight, mask)


def _fake_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@register_custom_op(
    op_name="rmsnorm_preserve_reduction",
    mutates_args=[],
    fake_impl=_fake_rmsnorm,
)
def rmsnorm_preserve_reduction(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Preserve aten's mean and cast-before-weight semantics without residuals."""
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim >= 2
        and x.numel() > 0
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
    ):
        raise RuntimeError("invalid input for rmsnorm_preserve_reduction")
    x = x.contiguous()
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        variance = _mean_square(x, 1024)
        _rmsnorm_finish_kernel[(triton.cdiv(x.numel(), 512),)](
            x,
            variance,
            weight,
            out,
            x.numel(),
            x.shape[-1],
            eps,
            512,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _rmsnorm_modulate_finish_kernel(
    X,
    Mean,
    Weight,
    Modulation,
    Out,
    Gate,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    TOKENS: tl.constexpr,
    EPS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    row, column = offsets // WIDTH, offsets % WIDTH
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    mean = tl.load(Mean + row, valid, other=0)
    normalized = x * cuda_rsqrtf(mean + EPS)
    if ADAPTIVE:
        batch = row // TOKENS
        base = batch * (3 * WIDTH) + column
        scale = tl.load(Modulation + base, valid, other=0).to(tl.float32)
        shift = tl.load(Modulation + base + WIDTH, valid, other=0).to(tl.float32)
        normalized = normalized * (1.0 + scale) + shift
        gate = tl.load(Modulation + base + 2 * WIDTH, valid, other=0)
        tl.store(Gate + batch * WIDTH + column, gate, valid & (row % TOKENS == 0))
    elif WEIGHTED:
        weight = tl.load(Weight + column, valid, other=0).to(tl.float32)
        normalized = normalized * (1.0 + weight)
    tl.store(Out + offsets, normalized, valid)


def rmsnorm_modulate_preserve_reduction(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    modulation: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inference RMSNorm with ``1 + weight`` or packed scale/shift/gate modulation.
    Keep Torch's FP32 mean and affine rounding; return owned outputs in x.dtype.
    """
    if (
        x.device.type != "cuda"
        or torch.version.hip is not None
        or x.ndim != 3
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.numel() == 0
    ):
        raise ValueError(
            "rmsnorm_modulate_preserve_reduction requires contiguous CUDA FP16/BF16 [B,T,D]"
        )
    batch, tokens, width = x.shape
    if weight is not None and modulation is not None:
        raise ValueError("Choose weight or adaptive modulation")
    for affine, shape in ((weight, (width,)), (modulation, (batch, 3 * width))):
        if affine is not None and (
            affine.device != x.device
            or affine.shape != shape
            or not affine.is_contiguous()
            or affine.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        ):
            raise ValueError("Invalid normalization affine shape, layout or dtype")
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (x, weight, modulation)
    ):
        raise RuntimeError("rmsnorm_modulate_preserve_reduction is inference-only")
    with torch.cuda.device(x.device):
        mean = _mean_square(x, 256)
        output = torch.empty_like(x)
        gate = (
            None
            if modulation is None
            else torch.empty((batch, 1, width), device=x.device, dtype=x.dtype)
        )
        _rmsnorm_modulate_finish_kernel[(triton.cdiv(x.numel(), 256),)](
            x,
            mean,
            weight,
            modulation,
            output,
            gate,
            x.numel(),
            width,
            tokens,
            eps,
            modulation is not None,
            weight is not None,
            256,
            enable_fp_fusion=False,
        )
    return output, gate
