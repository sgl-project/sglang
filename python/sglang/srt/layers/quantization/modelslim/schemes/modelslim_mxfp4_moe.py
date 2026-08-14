"""ModelSlim packed-MXFP4 expert schemes for Ascend NPU."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A4MXFP4MoEMethod,
    NPUW4A8MXFPMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

MXFP_BLOCK_SIZE = 32
MXFP_SCALE_PAIR_SIZE = 2
MXFP4_PACK_FACTOR = 2
# OCP E8M0 reserves 0xFF for NaN.  A valid ModelSlim UE8M0 scale must never
# contain it, so filling the parameter with this byte makes partial/missing
# checkpoint loads observable before the NPU kernel can consume garbage.
MXFP_E8M0_NOT_LOADED = 0xFF


def _mxfp4_moe_weight_shapes(
    weight_prefix: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Derive the checkpoint weight and UE8M0 scale shapes for one projection."""
    if weight_prefix == "w13":
        output_size = 2 * intermediate_size_per_partition
        input_size = hidden_size
    elif weight_prefix == "w2":
        output_size = hidden_size
        input_size = intermediate_size_per_partition
    else:
        raise ValueError(f"weight_prefix must be 'w13' or 'w2', got {weight_prefix!r}.")

    if input_size % MXFP4_PACK_FACTOR != 0:
        raise ValueError(f"MXFP4 input size must be even, got {input_size}.")
    scale_blocks = (input_size + MXFP_BLOCK_SIZE - 1) // MXFP_BLOCK_SIZE
    if scale_blocks % MXFP_SCALE_PAIR_SIZE != 0:
        raise ValueError(
            "MXFP4 grouped matmul pairs adjacent UE8M0 scales, but input size "
            f"{input_size} produces an odd block count {scale_blocks}."
        )

    weight_shape = (
        num_experts,
        output_size,
        input_size // MXFP4_PACK_FACTOR,
    )
    scale_shape = (num_experts, output_size, scale_blocks)
    return weight_shape, scale_shape


class _ModelSlimMXFP4MoESchemeBase(ModelSlimMoEScheme):
    kernel_class = None

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,
    ) -> None:
        if self.kernel_class is None:
            raise TypeError("MXFP4 MoE scheme base cannot be instantiated directly.")
        self.quant_config = quant_config
        self.weight_prefix = weight_prefix
        self.kernel = self.kernel_class(weight_prefix)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        weight_shape, scale_shape = _mxfp4_moe_weight_shapes(
            self.weight_prefix,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        weight = torch.nn.Parameter(
            torch.empty(weight_shape, dtype=torch.uint8), requires_grad=False
        )
        layer.register_parameter(f"{self.weight_prefix}_weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        weight_scale = torch.nn.Parameter(
            torch.full(
                scale_shape,
                MXFP_E8M0_NOT_LOADED,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter(f"{self.weight_prefix}_weight_scale", weight_scale)
        set_weight_attrs(weight_scale, extra_weight_attrs)

        # Microscaling has no zero point. Keep the common AscendQuantInfo field
        # explicit without introducing a checkpoint parameter.
        layer.register_parameter(f"{self.weight_prefix}_weight_offset", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)


class ModelSlimMXFP4MoEScheme(_ModelSlimMXFP4MoESchemeBase):
    """W4A4_MXFP4 experts: packed FP4 weights and dynamic MXFP4 activations."""

    kernel_class = NPUW4A4MXFP4MoEMethod


class ModelSlimMXFP4W4A8MoEScheme(_ModelSlimMXFP4MoESchemeBase):
    """W4A8_MXFP experts: packed FP4 weights and dynamic MXFP8 activations."""

    kernel_class = NPUW4A8MXFPMoEMethod
