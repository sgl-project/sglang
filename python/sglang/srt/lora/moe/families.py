from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

logger = logging.getLogger(__name__)

# Existing loader layouts. New families are admitted with their loaders.
_VENDORS = {
    "bf16": ("cutedsl", "triton"),
    "fp8": ("cutedsl", "triton"),
    "nvfp4": ("marlin",),
}


def resolve_vendor(family: str, vendor: str | None) -> str:
    vendors = _VENDORS[family]
    if vendor not in vendors:
        if vendor is not None:
            logger.info(
                "MoE LoRA base-GEMM vendor %r serves no %s layers; this layer "
                "family uses its default vendor %r",
                vendor,
                family,
                vendors[0],
            )
        vendor = vendors[0]
    return vendor


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
        if base_layer.params_dtype != torch.bfloat16:
            raise NotImplementedError("FP8 MoE LoRA requires BF16 activations.")
        if not quant_method.block_quant or getattr(quant_method, "use_mxfp8", False):
            raise NotImplementedError(
                "MoE LoRA on FP8 requires 128-block weight quantization; "
                "per-tensor and MXFP8 layers are unsupported"
            )
        if base_layer.w13_weight.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                "MoE LoRA on FP8 requires resident float8_e4m3fn weights; "
                "a Marlin-repacked FP8 layer is unsupported"
            )
        return "fp8"
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    if isinstance(quant_method, ModelOptNvFp4FusedMoEMethod):
        return "nvfp4"
    raise NotImplementedError(
        "MoE LoRA supports unquantized BF16, 128-block FP8, and NVFP4 "
        f"MoE; this layer's quant method is {type(quant_method).__name__}"
    )
