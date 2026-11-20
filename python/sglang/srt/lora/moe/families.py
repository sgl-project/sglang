"""MoE LoRA weight families: their vendors, and recognition from config or layer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

logger = logging.getLogger(__name__)

# The first vendor is the default. Quantized families use Marlin with BF16 input.
_VENDORS = {
    "bf16": ("cutedsl", "triton"),
    "fp8": ("marlin", "cutedsl", "triton"),
    "fp8_channel": ("marlin", "cutedsl", "triton"),
    "fp8_tensor": ("marlin", "triton"),
    "nvfp4": ("marlin", "cutedsl"),
    "mxfp4": ("marlin", "cutedsl"),
    "mxfp8": ("marlin", "cutedsl"),
}

# When a vendor lacks a provider for a family, keep the loader's weight layout.
_SAME_LAYOUT_VENDOR = {
    "bf16": "cutedsl",
    "fp8_tensor": "triton",
    "mxfp8": "cutedsl",
}


def resolve_vendor(family: str, vendor: str) -> str:
    vendors = _VENDORS[family]
    if vendor in vendors:
        return vendor
    fallback = _SAME_LAYOUT_VENDOR.get(family)
    if fallback is None:
        raise NotImplementedError(
            f"{family.upper()} MoE LoRA has no base-GEMM vendor for "
            f"--moe-runner-backend lora_{vendor}; use "
            + " or ".join(f"lora_{v}" for v in vendors)
        )
    logger.info_once(
        "MoE LoRA base-GEMM vendor %r serves no %s layers; this layer "
        "family uses %r, which reads the same weight layout",
        vendor,
        family,
        fallback,
    )
    return fallback


# ModelConfig.quantization -> family; native fp8 means the 128-block layout.
_CONFIG_FAMILIES = {
    None: "bf16",
    "fp8": "fp8",
    "mxfp8": "mxfp8",
    "modelopt_fp4": "nvfp4",
    "w8a8_fp8": "fp8_channel",
}
_COMPRESSED_TENSORS_FP8_FAMILIES = {
    "block": "fp8",
    "channel": "fp8_channel",
    "tensor": "fp8_tensor",
}
_COMPRESSED_TENSORS = "compressed-tensors"
# Checkpoint methods allowed through startup; layers check the actual format.
CONFIG_QUANTIZATIONS = (*_CONFIG_FAMILIES, _COMPRESSED_TENSORS, "modelopt_mixed")


def family_of_layer(base_layer: FusedMoE) -> str:
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    quant_method = base_layer.quant_method
    if isinstance(quant_method, UnquantizedFusedMoEMethod):
        if (
            base_layer.w13_weight.dtype != torch.bfloat16
            or base_layer.w2_weight.dtype != torch.bfloat16
        ):
            raise NotImplementedError(
                "MoE LoRA on an unquantized layer requires resident BF16 weights"
            )
        return "bf16"
    # FP16 would fail silently: Marlin takes half, but the stage buffers are BF16.
    if base_layer.params_dtype != torch.bfloat16:
        raise NotImplementedError("Quantized MoE LoRA requires BF16 activations.")
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    if isinstance(quant_method, Fp8MoEMethod):
        # Expert metadata takes precedence over the dense FP8 format.
        if getattr(quant_method, "is_fp4_expert", False):
            return "mxfp4"
        if getattr(quant_method, "use_mxfp8", False):
            return "mxfp8"
        if not quant_method.block_quant:
            raise NotImplementedError(
                "MoE LoRA on FP8 requires 128-block weight quantization; "
                "per-tensor layers are unsupported"
            )
        if (
            not quant_method.use_marlin
            and base_layer.w13_weight.dtype != torch.float8_e4m3fn
        ):
            raise NotImplementedError(
                "MoE LoRA on FP8 requires resident float8_e4m3fn weights"
            )
        return "fp8"
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    if isinstance(quant_method, ModelOptNvFp4FusedMoEMethod):
        return "nvfp4"
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
        Mxfp4MarlinMoEMethod,
    )

    if isinstance(quant_method, Mxfp4MarlinMoEMethod):
        return "mxfp4"
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsFusedMoEMethod,
    )

    if isinstance(quant_method, CompressedTensorsFusedMoEMethod):
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )

        # compressed-tensors delegates the MoE method to the layer's scheme.
        scheme = base_layer.scheme
        if not isinstance(scheme, CompressedTensorsW8A8Fp8MoE):
            raise NotImplementedError(
                "MoE LoRA on compressed-tensors supports the FP8 scheme only; "
                f"this layer's scheme is {type(scheme).__name__}"
            )
        return _COMPRESSED_TENSORS_FP8_FAMILIES[scheme.weight_quant.strategy]
    from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod

    if isinstance(quant_method, W8A8FP8MoEMethod):
        return "fp8_channel"
    raise NotImplementedError(
        "MoE LoRA supports unquantized BF16, 128-block, per-channel or "
        "per-tensor FP8, MXFP8, MXFP4 and NVFP4 MoE; this layer's quant method is "
        f"{type(quant_method).__name__}"
    )
