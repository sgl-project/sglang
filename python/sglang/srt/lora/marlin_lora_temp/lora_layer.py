"""LoRA hooks for the experimental Marlin MoE runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.lora.marlin_lora_temp.policy import (
    validate_experimental_sgl_marlin_contract,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput


def init_experimental_sgl_marlin_lora(layer, base_layer) -> None:
    """Store Marlin quantization metadata on the wrapped layer."""
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsFusedMoEMethod,
    )
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    assert isinstance(
        base_layer.quant_method,
        (CompressedTensorsFusedMoEMethod, ModelOptNvFp4FusedMoEMethod),
    ), (
        f"experimental_sgl_marlin requires a quant method exposing "
        f"get_marlin_quant_info, got {type(base_layer.quant_method).__name__}"
    )

    quant_info = base_layer.quant_method.get_marlin_quant_info(base_layer)
    weight_device = quant_info.w13_qweight.device
    device_capability = (
        torch.cuda.get_device_capability(weight_device)
        if weight_device.type == "cuda"
        else (0, 0)
    )
    validate_experimental_sgl_marlin_contract(
        base_layer.moe_runner_config,
        moe_ep_size=int(base_layer.moe_ep_size),
        device_capability=device_capability,
    )

    layer._lora_runner = None
    layer._quant_info = quant_info


def dispatch_experimental_sgl_marlin_lora(
    dispatch_output, quant_info, base_layer, lora_info
) -> StandardCombineInput:
    """Run the experimental marlin MoE-LoRA pipeline for a single layer."""
    from sglang.srt.lora.marlin_lora_temp.moe_runner import (
        fused_experts_experimental_sgl_marlin_lora,
    )

    return fused_experts_experimental_sgl_marlin_lora(
        dispatch_output,
        quant_info,
        base_layer.moe_runner_config,
        lora_info,
    )
