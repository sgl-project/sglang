# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 adapters for FlashInfer PrimsTS dense NVFP4 GEMMs.

The dense PrimsTS kernels deliberately consume the canonical 128x4 block-scale
layout.  Keep that layout as a flat buffer: reshaping it as conventional
``[rows, K / 16]`` scales loses the swizzle required by the TMA descriptors.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.prims_ts import gemm as prims_ts_gemm

from sglang.kernels.ops.diffusion.gemm.flux3_prims_ts_config import (
    delta_kwargs,
    flux3_prims_ts_config,
)

QuantizedNvfp4Activation = tuple[torch.Tensor, torch.Tensor, float]
PackedNvfp4Activation = tuple[torch.Tensor, torch.Tensor, float, tuple[int, ...]]
_GLOBAL_SCALE_INV_CACHE: dict[tuple[str, int | None, float], torch.Tensor] = {}
_PREPARED_FP4_CACHE: dict[tuple[object, ...], prims_ts_gemm.PreparedFp4Linear] = {}


def _scale_numel(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * 128 * ((k // 16 + 3) // 4) * 4


def _global_scale_inv(device: torch.device, input_global_scale: float) -> torch.Tensor:
    key = (device.type, device.index, input_global_scale)
    scale = _GLOBAL_SCALE_INV_CACHE.get(key)
    if scale is None:
        scale = torch.tensor(
            [1.0 / input_global_scale], device=device, dtype=torch.float32
        )
        _GLOBAL_SCALE_INV_CACHE[key] = scale
    return scale


def quantize_flux3_nvfp4_activation(
    x: torch.Tensor, input_global_scale: float
) -> QuantizedNvfp4Activation:
    """Quantize BF16 rows and return a canonical 128x4 E4M3 scale buffer.

    ``nvfp4_quantize`` receives the reciprocal global scale, while PrimsTS
    receives the dequantization scale.  Returning the exact 128x4 buffer is
    essential: the dense kernel validates its padded element count and reads
    it through 128-by-4 TMA tiles.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(f"FLUX.3 PrimsTS NVFP4 input must be BF16, got {x.dtype}")
    if input_global_scale <= 0:
        raise ValueError(
            f"NVFP4 input global scale must be positive, got {input_global_scale}"
        )
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    if x_2d.shape[1] % 256:
        raise ValueError(
            f"FLUX.3 PrimsTS NVFP4 requires K divisible by 256, got {x_2d.shape[1]}"
        )
    packed, scale_128x4 = nvfp4_quantize(
        x_2d,
        # The CuTe-DSL quantizer takes the reciprocal global scale as a host
        # float; a device tensor would cost four extra tensor ops per call.
        1.0 / input_global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        # The dense GEMM itself is CuTe-DSL.  Using the matching quantizer
        # avoids FlashInfer's optional TRT-LLM quantizer extension and keeps
        # the 128x4 layout native end-to-end on SM100+.
        backend="cute-dsl",
    )
    scale_128x4 = scale_128x4.view(torch.uint8).reshape(-1).contiguous()
    expected = _scale_numel(x_2d.shape[0], x_2d.shape[1])
    if scale_128x4.numel() != expected:
        raise RuntimeError(
            "FlashInfer NVFP4 quantization did not return the required 128x4 "
            f"scale buffer: expected {expected} bytes, got {scale_128x4.numel()}"
        )
    return packed.view(torch.uint8).contiguous(), scale_128x4, input_global_scale


def _quantized_activation(
    x: torch.Tensor,
    input_global_scale: float,
    quantized: QuantizedNvfp4Activation | None,
) -> QuantizedNvfp4Activation:
    if quantized is not None and quantized[2] == input_global_scale:
        return quantized
    return quantize_flux3_nvfp4_activation(x, input_global_scale)


def _prepared_linear(
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    config: prims_ts_gemm.PrimsTsGemmConfig | None,
) -> prims_ts_gemm.PreparedFp4Linear | None:
    # Caching would keep offloaded GPU weights alive; use the one-shot kernel.
    if not weight_packed.is_cuda or getattr(
        weight_packed, "_sglang_layerwise_offloaded", False
    ):
        return None
    key = (
        "linear",
        weight_packed.data_ptr(),
        weight_packed._version,
        weight_scale_128x4.data_ptr(),
        weight_scale_128x4._version,
        input_global_scale,
        weight_global_scale,
        config,
    )
    prepared = _PREPARED_FP4_CACHE.get(key)
    if prepared is None:
        prepared = prims_ts_gemm.prepare_fp4_linear(
            weight_packed,
            weight_scale_128x4,
            input_global_scale,
            weight_global_scale,
            config=config,
        )
        _PREPARED_FP4_CACHE[key] = prepared
    return prepared


def _prepared_swiglu(
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    config: prims_ts_gemm.PrimsTsGemmConfig | None,
    *,
    fp4_output: bool,
    device: torch.device,
) -> prims_ts_gemm.PreparedFp4Linear | None:
    # Caching would keep offloaded GPU weights alive; use the one-shot kernel.
    if not weight_packed.is_cuda or getattr(
        weight_packed, "_sglang_layerwise_offloaded", False
    ):
        return None
    key = (
        "swiglu_fp4" if fp4_output else "swiglu_bf16",
        weight_packed.data_ptr(),
        weight_packed._version,
        weight_scale_128x4.data_ptr(),
        weight_scale_128x4._version,
        input_global_scale,
        weight_global_scale,
        config,
    )
    prepared = _PREPARED_FP4_CACHE.get(key)
    if prepared is None:
        prepared = prims_ts_gemm.prepare_fp4_linear_swiglu(
            weight_packed,
            weight_scale_128x4,
            input_global_scale,
            weight_global_scale,
            out_dtype=torch.uint8 if fp4_output else torch.bfloat16,
            output_quant_scale=(_global_scale_inv(device, 1.0) if fp4_output else None),
            config=config,
        )
        _PREPARED_FP4_CACHE[key] = prepared
    return prepared


def _prepared_qkv(
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    config: prims_ts_gemm.PrimsTsGemmConfig | None,
    qkv_scale: torch.Tensor | None = None,
    *,
    separate_qkv_output: bool = False,
) -> prims_ts_gemm.PreparedFp4Linear | None:
    # Caching would keep offloaded GPU weights alive; use the one-shot kernel.
    if not weight_packed.is_cuda or getattr(
        weight_packed, "_sglang_layerwise_offloaded", False
    ):
        return None
    head_dim = q_norm_weight.numel()
    if separate_qkv_output:
        if config is None:
            major, minor = torch.cuda.get_device_capability(weight_packed.device)
            config = prims_ts_gemm.PrimsTsGemmConfig(
                arch=major * 10 + minor,
                operand_format="nvfp4_e2m1",
                output_format="bf16",
                epilogue="qkv_qknorm_rope",
                has_bias=False,
                head_dim=head_dim,
                is_neox=False,
                tile_k=256,
                has_qkv_scale=qkv_scale is not None,
            )
        config = replace(
            config,
            separate_qkv_output=True,
            use_tma_store=False,
        )
    key = (
        "qkv_qknorm_rope",
        weight_packed.data_ptr(),
        weight_packed._version,
        weight_scale_128x4.data_ptr(),
        weight_scale_128x4._version,
        q_norm_weight.data_ptr(),
        q_norm_weight._version,
        k_norm_weight.data_ptr(),
        k_norm_weight._version,
        input_global_scale,
        weight_global_scale,
        config,
        None if qkv_scale is None else qkv_scale.data_ptr(),
        None if qkv_scale is None else qkv_scale._version,
    )
    prepared = _PREPARED_FP4_CACHE.get(key)
    if prepared is None:
        num_heads = weight_packed.shape[0] // (3 * head_dim)
        args = (
            weight_packed,
            weight_scale_128x4,
            input_global_scale,
            weight_global_scale,
            q_norm_weight,
            k_norm_weight,
        )
        kwargs = dict(
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            is_neox=False,
            config=config,
        )
        if qkv_scale is not None:
            kwargs["qkv_scale"] = qkv_scale
        prepared = prims_ts_gemm.prepare_fp4_qkv_qknorm_rope(*args, **kwargs)
        _PREPARED_FP4_CACHE[key] = prepared
    return prepared


def flux3_nvfp4_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    quantized: QuantizedNvfp4Activation | None = None,
    *,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a FLUX.3 projection entirely through PrimsTS NVFP4 GEMM."""
    a_packed, a_scale_128x4, _ = _quantized_activation(x, input_global_scale, quantized)
    prepared = _prepared_linear(
        weight_packed,
        weight_scale_128x4,
        input_global_scale,
        weight_global_scale,
        flux3_prims_ts_config(
            x,
            operand_format="nvfp4_e2m1",
            epilogue="linear",
            k=weight_packed.shape[1] * 2,
            n=weight_packed.shape[0],
        ),
    )
    if prepared is not None:
        out = prepared(a_packed, a_scale_128x4, **delta_kwargs(delta))
    else:
        out = prims_ts_gemm.fp4_linear(
            a_packed,
            a_scale_128x4,
            input_global_scale,
            weight_packed,
            weight_scale_128x4,
            weight_global_scale,
            **delta_kwargs(delta),
            skip_check=True,
        )
    return out.view(*x.shape[:-1], out.shape[-1])


def flux3_nvfp4_linear_from_packed_activation(
    activation: PackedNvfp4Activation,
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    weight_global_scale: float,
) -> torch.Tensor:
    """Run an NVFP4 projection without requantizing an intermediate result."""
    a_packed, a_scale_128x4, a_global_scale, leading_shape = activation
    prepared = _prepared_linear(
        weight_packed,
        weight_scale_128x4,
        a_global_scale,
        weight_global_scale,
        flux3_prims_ts_config(
            a_packed,
            operand_format="nvfp4_e2m1",
            epilogue="linear",
            k=weight_packed.shape[1] * 2,
            n=weight_packed.shape[0],
            m=math.prod(leading_shape),
        ),
    )
    if prepared is not None:
        out = prepared(a_packed, a_scale_128x4)
    else:
        out = prims_ts_gemm.fp4_linear(
            a_packed,
            a_scale_128x4,
            a_global_scale,
            weight_packed,
            weight_scale_128x4,
            weight_global_scale,
            skip_check=True,
        )
    return out.view(*leading_shape, out.shape[-1])


def flux3_nvfp4_linear_swiglu(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    quantized: QuantizedNvfp4Activation | None = None,
    *,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FLUX.3's gated MLP-up projection with PrimsTS' SwiGLU epilogue."""
    a_packed, a_scale_128x4, _ = _quantized_activation(x, input_global_scale, quantized)
    prepared = _prepared_swiglu(
        weight_packed,
        weight_scale_128x4,
        input_global_scale,
        weight_global_scale,
        flux3_prims_ts_config(
            x,
            operand_format="nvfp4_e2m1",
            epilogue="swiglu",
            k=weight_packed.shape[1] * 2,
            n=weight_packed.shape[0],
        ),
        fp4_output=False,
        device=x.device,
    )
    if prepared is not None:
        out = prepared(a_packed, a_scale_128x4, **delta_kwargs(delta))
    else:
        out = prims_ts_gemm.fp4_linear_swiglu(
            a_packed,
            a_scale_128x4,
            input_global_scale,
            weight_packed,
            weight_scale_128x4,
            weight_global_scale,
            **delta_kwargs(delta),
            skip_check=True,
        )
    return out.view(*x.shape[:-1], out.shape[-1])


def flux3_nvfp4_linear_swiglu_fp4(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    quantized: QuantizedNvfp4Activation | None = None,
) -> PackedNvfp4Activation:
    """Emit packed NVFP4 SwiGLU output for the following MLP-down GEMM.

    A unit output global scale is supplied explicitly, so the PrimsTS API
    returns only the packed payload and canonical 128x4 E4M3 scales.  That
    same known scale is then consumed directly by ``fp4_linear``; no BF16
    materialization or standalone NVFP4 quantization is needed between MLP-up
    and MLP-down.
    """
    a_packed, a_scale_128x4, _ = _quantized_activation(x, input_global_scale, quantized)
    prepared = _prepared_swiglu(
        weight_packed,
        weight_scale_128x4,
        input_global_scale,
        weight_global_scale,
        flux3_prims_ts_config(
            x,
            operand_format="nvfp4_e2m1",
            epilogue="swiglu",
            k=weight_packed.shape[1] * 2,
            n=weight_packed.shape[0],
            output_format="nvfp4_e2m1",
        ),
        fp4_output=True,
        device=x.device,
    )
    if prepared is not None:
        payload, scale_128x4 = prepared(a_packed, a_scale_128x4)
    else:
        payload, scale_128x4 = prims_ts_gemm.fp4_linear_swiglu(
            a_packed,
            a_scale_128x4,
            input_global_scale,
            weight_packed,
            weight_scale_128x4,
            weight_global_scale,
            out_dtype=torch.uint8,
            output_quant_scale=_global_scale_inv(x.device, 1.0),
            skip_check=True,
        )
    return payload, scale_128x4, 1.0, tuple(x.shape[:-1])


def flux3_nvfp4_qkv_qknorm_rope(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_128x4: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    qkv_scale: torch.Tensor | None = None,
    quantized: QuantizedNvfp4Activation | None = None,
    *,
    separate_qkv_output: bool = False,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse NVFP4 QKV GEMM, Q/K RMSNorm, and RoPE through PrimsTS.

    ``qkv_scale`` holds the per-partition Q/K/V weight scales already
    multiplied by ``input_global_scale``; it must be a persistent tensor
    because it keys the prepared-kernel cache.
    """
    a_packed, a_scale_128x4, _ = _quantized_activation(x, input_global_scale, quantized)
    prepared = _prepared_qkv(
        weight_packed,
        weight_scale_128x4,
        input_global_scale,
        weight_global_scale,
        q_norm_weight,
        k_norm_weight,
        flux3_prims_ts_config(
            x,
            operand_format="nvfp4_e2m1",
            epilogue="qkv_qknorm_rope",
            k=weight_packed.shape[1] * 2,
            n=weight_packed.shape[0],
            head_dim=q_norm_weight.numel(),
            qkv_scale=qkv_scale is not None,
        ),
        qkv_scale,
        separate_qkv_output=separate_qkv_output,
    )
    if prepared is not None:
        out = prepared(
            a_packed,
            a_scale_128x4,
            cos_sin=cos_sin_cache,
            positions=positions,
            **delta_kwargs(delta),
        )
    elif separate_qkv_output:
        raise RuntimeError("separate QKV output requires a prepared CUDA projection")
    else:
        head_dim = q_norm_weight.numel()
        out = prims_ts_gemm.fp4_qkv_qknorm_rope(
            a_packed,
            a_scale_128x4,
            input_global_scale,
            weight_packed,
            weight_scale_128x4,
            weight_global_scale,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            num_q_heads=weight_packed.shape[0] // (3 * head_dim),
            num_kv_heads=weight_packed.shape[0] // (3 * head_dim),
            head_dim=head_dim,
            is_neox=False,
            qkv_scale=qkv_scale,
            **delta_kwargs(delta),
            skip_check=True,
        )
    if separate_qkv_output:
        return out.view(3, *x.shape[:-1], out.shape[-1] // 3)
    return out.view(*x.shape[:-1], out.shape[-1])
