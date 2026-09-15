# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.quantization.fp8_utils import fp8_dtype_to_triton
from sglang.srt.utils import (
    cdiv,
    cpu_has_amx_support,
    is_cpu,
    is_gfx95_supported,
    is_npu,
    next_power_of_2,
)

_is_npu = is_npu()
_use_cpu = is_cpu() and cpu_has_amx_support()

# Maximum rows per Triton block for layernorm gated kernel
MAX_ROWS_PER_BLOCK = 4
_FP8_E4M3_MAX = 448.0


@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    # PDL: x is the producer's output (e.g. the fused KDA verify kernel, which
    # triggers its dependents right after the o store), so every load sits
    # behind the wait; the launch/prologue overlaps the producer's tail.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(
            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(
            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (
        (b_x - b_mean[:, None]) * b_rstd[:, None]
        if not IS_RMS_NORM
        else b_x * b_rstd[:, None]
    )
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    D: tl.constexpr,  # number of columns in x
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # swish/sigmoid output gate
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = (
        torch.empty((T,), dtype=torch.float, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    if D <= 512:
        BT = 32
        pdl_kwargs = (
            {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
        )
        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_warps=4,
            **pdl_kwargs,
        )
    else:
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_warps=4,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


class LayerNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        x_shape_og = x.shape
        g_shape_og = g.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_gated_fwd(
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=eps,
            residual=residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(residual_out, g, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.activation = activation
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    return LayerNormGatedFunction.apply(
        x,
        g,
        weight,
        bias,
        activation,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True,
    )


@triton.jit
def _rms_norm_gated_per_token_fp8_kernel(
    x,
    g,
    weight,
    output,
    output_scale,
    eps,
    stride_x_m,
    stride_g_m,
    stride_output_m,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_offsets = tl.arange(0, NUM_HEADS)[:, None]
    dim_offsets = tl.arange(0, HEAD_DIM)[None, :]
    offsets = head_offsets * HEAD_DIM + dim_offsets
    x_row = tl.load(x + token_idx * stride_x_m + offsets).to(tl.float32)
    gate_row = tl.load(g + token_idx * stride_g_m + offsets).to(tl.float32)
    norm_weight = tl.load(weight + dim_offsets).to(tl.float32)

    variance = tl.sum(x_row * x_row, axis=1) / HEAD_DIM
    gated = (
        x_row
        * (1.0 / tl.sqrt(variance[:, None] + eps))
        * norm_weight
        * tl.sigmoid(gate_row)
    )
    gated = gated.to(tl.bfloat16).to(tl.float32)
    token_absmax = tl.max(tl.max(tl.abs(gated), axis=1), axis=0)
    scale = tl.maximum(token_absmax, 1e-12) / FP8_MAX
    tl.store(output_scale + token_idx, scale)

    quantized = tl.clamp(gated / scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    tl.store(
        output + token_idx * stride_output_m + offsets,
        quantized.to(tl.uint8, bitcast=True),
    )


def can_use_rms_norm_gated_per_token_fp8(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    return (
        is_gfx95_supported()
        and x.is_cuda
        and g.is_cuda
        and weight.is_cuda
        and x.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.dim() >= 3
        and g.dim() >= 3
        and x.numel() == g.numel()
        and x.shape[-2:] == g.shape[-2:]
        and x.shape[-1] == 128
        and x.shape[-2] in (8, 16)
        and weight.shape == (128,)
        and x.is_contiguous()
        and g.is_contiguous()
        and weight.is_contiguous()
    )


def rms_norm_gated_per_token_fp8(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_rms_norm_gated_per_token_fp8(x, g, weight):
        raise ValueError(
            "gated RMSNorm PTPC bridge requires contiguous gfx950 BF16 "
            "[..., tokens, 8|16, 128] inputs and a contiguous [128] weight"
        )

    num_heads = x.shape[-2]
    num_tokens = x.numel() // (num_heads * 128)
    output = torch.empty(
        (num_tokens, num_heads * 128),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    output_scale = torch.empty(
        (num_tokens, 1),
        dtype=torch.float32,
        device=x.device,
    )
    if num_tokens == 0:
        return output, output_scale

    x_2d = x.view(num_tokens, num_heads * 128)
    g_2d = g.view(num_tokens, num_heads * 128)
    with torch.cuda.device(x.device):
        _rms_norm_gated_per_token_fp8_kernel[(num_tokens,)](
            x=x_2d,
            g=g_2d,
            weight=weight,
            output=output.view(torch.uint8),
            output_scale=output_scale,
            eps=eps,
            stride_x_m=x_2d.stride(0),
            stride_g_m=g_2d.stride(0),
            stride_output_m=output.stride(0),
            NUM_HEADS=num_heads,
            HEAD_DIM=128,
            FP8_DTYPE=fp8_dtype_to_triton(torch.float8_e4m3fn),
            FP8_MAX=_FP8_E4M3_MAX,
            num_warps=8 if num_heads == 16 else 4,
        )
    return output, output_scale


class FusedRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        if _use_cpu:
            assert self.activation == "silu", (
                "CPU rmsnorm_gated currently only supports activation silu"
            )
            return torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(
                x, self.weight, g, self.eps
            )
        else:
            return rms_norm_gated(
                x,
                g,
                self.weight,
                self.bias,
                self.activation,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )
