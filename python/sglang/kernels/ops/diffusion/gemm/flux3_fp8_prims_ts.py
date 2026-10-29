# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 adapters for the PrimsTS FP8 GEMM, with optional SwiGLU."""

from __future__ import annotations

from dataclasses import replace

import torch
from flashinfer.prims_ts import gemm as prims_ts_gemm

from sglang.kernels.ops.diffusion.gemm.flux3_prims_ts_config import (
    delta_kwargs,
    flux3_prims_ts_config,
)
from sglang.kernels.ops.quantization.per_token_quant_fp8 import per_token_quant_fp8

QuantizedActivation = tuple[torch.Tensor, torch.Tensor]


def _flux3_gemm_config(
    x: torch.Tensor,
    *,
    epilogue: str,
    k: int,
    n: int,
    head_dim: int | None = None,
    qkv_scale: bool = False,
) -> prims_ts_gemm.PrimsTsGemmConfig | None:
    """Return the explicit PrimsTS specialization for the FLUX.3 projections."""
    return flux3_prims_ts_config(
        x,
        operand_format="fp8_e4m3",
        epilogue=epilogue,
        k=k,
        n=n,
        head_dim=head_dim,
        qkv_scale=qkv_scale,
    )


def pack_flux3_mlp_weight(
    weight_kn: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[gate | value]`` rows as adjacent ``[value, gate]`` pairs.

    The PrimsTS epilogue computes ``first * silu(second)``. Reversing
    each pair therefore reproduces FLUX.3's ``silu(gate) * value`` convention.
    """
    if weight_kn.ndim != 2:
        raise ValueError(f"weight must be [K, 2H], got {tuple(weight_kn.shape)}")
    k, n = weight_kn.shape
    if n % 2:
        raise ValueError(f"gated MLP weight width must be even, got {n}")
    if weight_scale.shape != (n,):
        raise ValueError(
            f"weight_scale must have shape ({n},), got {tuple(weight_scale.shape)}"
        )

    half = n // 2
    weight_nk = weight_kn.t()
    packed_weight = torch.stack((weight_nk[half:], weight_nk[:half]), dim=1).reshape(
        n, k
    )
    packed_scale = torch.stack(
        (weight_scale[half:], weight_scale[:half]), dim=1
    ).reshape(n)
    return packed_weight.contiguous(), packed_scale.contiguous()


def quantize_flux3_activation(x: torch.Tensor) -> QuantizedActivation:
    """Dynamically quantize BF16 rows for one or more PrimsTS GEMMs."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"FLUX.3 PrimsTS FP8 input must be BF16, got {x.dtype}")
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_fp8 = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(x_2d.shape[0], device=x.device, dtype=torch.float32)
    per_token_quant_fp8(x_2d, x_fp8, x_scale)
    return x_fp8, x_scale


def _quantized_flux3_activation(
    x: torch.Tensor, quantized: QuantizedActivation | None
) -> QuantizedActivation:
    if quantized is None:
        return quantize_flux3_activation(x)
    return quantized


def flux3_fused_mlp_up(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_weight_scale: torch.Tensor,
    quantized: QuantizedActivation | None = None,
    *,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused FP8 GEMM + SwiGLU, quantizing BF16 rows when needed."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"FLUX.3 fused MLP input must be BF16, got {x.dtype}")
    x_fp8, x_scale = _quantized_flux3_activation(x, quantized)
    out = prims_ts_gemm.fp8_linear_swiglu(
        x_fp8,
        packed_weight,
        x_scale,
        packed_weight_scale,
        config=_flux3_gemm_config(
            x, epilogue="swiglu", k=packed_weight.shape[1], n=packed_weight.shape[0]
        ),
        **delta_kwargs(delta),
        skip_check=True,
    )
    return out.view(*x.shape[:-1], out.shape[-1])


def flux3_clustered_fp8_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_weight_scale: torch.Tensor,
    quantized: QuantizedActivation | None = None,
    qkv_scale: torch.Tensor | None = None,
    *,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a PrimsTS FP8 GEMM, optionally scaling packed Q/K/V thirds."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"FLUX.3 PrimsTS FP8 input must be BF16, got {x.dtype}")
    x_fp8, x_scale = _quantized_flux3_activation(x, quantized)
    out = prims_ts_gemm.fp8_linear(
        x_fp8,
        packed_weight,
        x_scale,
        packed_weight_scale,
        qkv_scale=qkv_scale,
        config=_flux3_gemm_config(
            x,
            epilogue="linear",
            k=packed_weight.shape[1],
            n=packed_weight.shape[0],
            qkv_scale=qkv_scale is not None,
        ),
        **delta_kwargs(delta),
        skip_check=True,
    )
    return out.view(*x.shape[:-1], out.shape[-1])


def flux3_clustered_fp8_qkv_qknorm_rope(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    quantized: QuantizedActivation | None = None,
    qkv_scale: torch.Tensor | None = None,
    *,
    separate_qkv_output: bool = False,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse QKV GEMM, QK RMSNorm, and RoPE, quantizing rows when needed."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"FLUX.3 fused QKV input must be BF16, got {x.dtype}")
    x_fp8, x_scale = _quantized_flux3_activation(x, quantized)
    config = _flux3_gemm_config(
        x,
        epilogue="qkv_qknorm_rope",
        k=packed_weight.shape[1],
        n=packed_weight.shape[0],
        head_dim=q_norm_weight.numel(),
        qkv_scale=qkv_scale is not None,
    )
    if separate_qkv_output:
        if config is None:
            major, minor = torch.cuda.get_device_capability(x.device)
            config = prims_ts_gemm.PrimsTsGemmConfig(
                arch=major * 10 + minor,
                operand_format="fp8_e4m3",
                output_format="bf16",
                epilogue="qkv_qknorm_rope",
                has_bias=False,
                head_dim=q_norm_weight.numel(),
                is_neox=False,
                has_qkv_scale=qkv_scale is not None,
            )
        config = replace(
            config,
            separate_qkv_output=True,
            use_tma_store=False,
        )
    out = prims_ts_gemm.fp8_qkv_qknorm_rope(
        x_fp8,
        packed_weight,
        x_scale,
        packed_weight_scale,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_q_heads=packed_weight.shape[0] // (3 * q_norm_weight.numel()),
        num_kv_heads=packed_weight.shape[0] // (3 * k_norm_weight.numel()),
        head_dim=q_norm_weight.numel(),
        qkv_scale=qkv_scale,
        config=config,
        **delta_kwargs(delta),
        skip_check=True,
    )
    if separate_qkv_output:
        return out.view(3, *x.shape[:-1], out.shape[-1] // 3)
    return out.view(*x.shape[:-1], out.shape[-1])
