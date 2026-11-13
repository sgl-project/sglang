"""MoE LoRA weight families: the vendors that serve each one, and how a
family is recognized, provisionally from the checkpoint's config and
authoritatively from the resident layer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

logger = logging.getLogger(__name__)

# The first vendor is the default for each weight family. Every quantized
# family defaults to Marlin, which keeps the activations in BF16 (W4A16 for
# the FP4 families, W8A16 for the FP8 ones); the cutedsl and triton vendors
# quantize the activations to the weights' format and serve on request.
VENDORS = {
    "bf16": ("cutedsl", "triton"),
    "fp8": ("marlin", "cutedsl", "triton"),
    "fp8_channel": ("marlin", "cutedsl", "triton"),
    "fp8_tensor": ("marlin", "triton"),
    "nvfp4": ("marlin", "cutedsl"),
    "mxfp4": ("marlin", "cutedsl"),
    "mxfp8": ("marlin", "cutedsl"),
}

# Marlin reads experts that the loader repacked into tiles under lora_marlin;
# every other vendor reads the checkpoint layout. A runner whose vendor has no
# provider for a family is served by the vendor below, which reads the layout
# the loader left for that runner (BF16 experts are not repacked). Families
# absent here refuse such a runner.
_SAME_LAYOUT_VENDOR = {
    "bf16": "cutedsl",
    "fp8_tensor": "triton",
    "mxfp8": "cutedsl",
}


def default_vendor(family: str) -> str:
    return VENDORS[family][0]


def resolve_vendor(family: str, vendor: str) -> str:
    """The vendor that serves ``family`` under --moe-runner-backend
    lora_<vendor>: itself, or one that reads the same weight layout. Never
    Marlin for another runner: only lora_marlin prepares its tiles."""
    vendors = VENDORS[family]
    if vendor in vendors:
        return vendor
    fallback = _SAME_LAYOUT_VENDOR.get(family)
    if fallback is None:
        raise NotImplementedError(
            f"{family.upper()} MoE LoRA has no base-GEMM vendor for "
            f"--moe-runner-backend lora_{vendor}; use "
            + " or ".join(f"lora_{v}" for v in vendors)
        )
    logger.info(
        "MoE LoRA base-GEMM vendor %r serves no %s layers; this layer "
        "family uses %r, which reads the same weight layout",
        vendor,
        family,
        fallback,
    )
    return fallback


# ModelConfig.quantization (resolved from the checkpoint) -> MoE LoRA weight
# family; FP8 configs that carry MXFP4 experts are the mxfp4 family, and
# native fp8 is the 128-block layout (_check_fp8_expert_blocks).
_CONFIG_FAMILIES = {
    None: "bf16",
    "fp8": "fp8",
    "mxfp8": "mxfp8",
    "modelopt_fp4": "nvfp4",
    "w8a8_fp8": "fp8_channel",
}
# compressed-tensors 8-bit float weights by (strategy, dynamic activations).
_COMPRESSED_TENSORS_FP8_FAMILIES = {
    ("block", True): "fp8",
    ("channel", True): "fp8_channel",
    ("tensor", False): "fp8_tensor",
    ("tensor", True): "fp8_tensor",
}
_COMPRESSED_TENSORS = ("compressed-tensors", "compressed_tensors")
# Every ModelConfig.quantization a lora_* runner accepts; layer attachment
# checks the specific weight layout within each scheme.
CONFIG_QUANTIZATIONS = (*_CONFIG_FAMILIES, *_COMPRESSED_TENSORS)


def _compressed_tensors_family(model_config: Any) -> str | None:
    """The family of a compressed-tensors checkpoint from its config groups.

    Each group quantizes the Linear layers it targets; FusedMoE experts
    load as those projections. Groups that disagree, or that are not 8-bit
    float, stay auto: layer attachment checks the exact scheme.
    """
    families = set()
    for group in (model_config.hf_quant_config.get("config_groups") or {}).values():
        weights = group.get("weights") or {}
        if weights.get("type") != "float" or weights.get("num_bits") != 8:
            return None
        activations = group.get("input_activations") or {}
        families.add(
            _COMPRESSED_TENSORS_FP8_FAMILIES.get(
                (weights.get("strategy"), bool(activations.get("dynamic")))
            )
        )
    return families.pop() if len(families) == 1 else None


def family_of_config(model_config: Any) -> str | None:
    """The family the checkpoint's config implies for the routed experts, or
    None when the config does not settle it. Provisional: it picks a default
    runner before any layer exists, and layer attachment decides."""
    if model_config.is_fp4_experts:
        return "mxfp4"
    if model_config.quantization in _COMPRESSED_TENSORS:
        return _compressed_tensors_family(model_config)
    return _CONFIG_FAMILIES.get(model_config.quantization)


def _require_bf16_activations(base_layer: FusedMoE) -> None:
    if base_layer.params_dtype != torch.bfloat16:
        raise NotImplementedError("FP8 MoE LoRA requires BF16 activations.")


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
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    if isinstance(quant_method, Fp8MoEMethod):
        _require_bf16_activations(base_layer)
        if getattr(quant_method, "use_mxfp8", False):
            return "mxfp8"
        # FP8 configs may carry MXFP4 experts (routed_experts_quant_method).
        if getattr(quant_method, "is_fp4_expert", False):
            return "mxfp4"
        if not quant_method.block_quant:
            raise NotImplementedError(
                "MoE LoRA on FP8 requires 128-block weight quantization; "
                "per-tensor layers are unsupported"
            )
        # lora_marlin repacks the experts into Marlin tiles instead.
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
        from compressed_tensors.quantization import QuantizationStrategy

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
        _require_bf16_activations(base_layer)
        if scheme.block_quant:
            return "fp8"
        # The scheme pairs per-channel weights with dynamic per-token
        # activations, and per-tensor weights with per-tensor activations.
        if scheme.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            return "fp8_channel"
        return "fp8_tensor"
    from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod

    if isinstance(quant_method, W8A8FP8MoEMethod):
        _require_bf16_activations(base_layer)
        return "fp8_channel"
    raise NotImplementedError(
        "MoE LoRA supports unquantized BF16, 128-block, per-channel or "
        "per-tensor FP8, MXFP8, MXFP4 and NVFP4 MoE; this layer's quant method is "
        f"{type(quant_method).__name__}"
    )
