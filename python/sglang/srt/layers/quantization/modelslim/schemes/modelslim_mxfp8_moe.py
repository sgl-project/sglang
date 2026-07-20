"""ModelSlim MXFP8 offline scheme for MoE layers on Ascend NPU (SRT).

Loads weights pre-quantised by msmodelslim: float8_e4m3fn weights + uint8 block
scales (block_size=32). The layout transform and the forward pass are delegated
to ``NPUMXFP8MoEMethod`` -- the same kernel the online MXFP8 MoE path uses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUMXFP8MoEMethod
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimMXFP8MoEScheme",
]

# Block (group) size of the msmodelslim MXFP8 export format.
MXFP8_BLOCK_SIZE = 32


class ModelSlimMXFP8MoEScheme(ModelSlimMoEScheme):
    """
    Offline MXFP8 MoE scheme that creates weights for either the
    w13 (gate+up) or w2 (down) projection group.

    Two instances of this class are used per MoE layer:
      - weight_prefix="w13"   → handles the fused gate_proj + up_proj weights
      - weight_prefix="w2"    → handles the down_proj weights

    The float8_e4m3fn weight dtype allocated here is what tells
    ``NPUMXFP8MoEMethod.process_weights_after_loading`` to take its offline
    (re-layout only) branch instead of quantising the weights itself.
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,  # "w13" or "w2"
    ) -> None:
        self.quant_config = quant_config
        self.kernel = NPUMXFP8MoEMethod()
        self.weight_prefix = weight_prefix
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got '{weight_prefix}'"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        # Determine shape based on weight group
        if self.weight_prefix == "w13":
            a_dim = 2 * intermediate_size_per_partition
            b_dim = hidden_size
        else:  # w2
            a_dim = hidden_size
            b_dim = intermediate_size_per_partition

        prefix = self.weight_prefix

        # Create and register weight: [E, N, K] float8_e4m3fn
        weight_name = f"{prefix}_weight"
        weight = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, b_dim, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter(weight_name, weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # Create and register block scale: [E, N, K // 32] uint8 (e8m0)
        scale_name = f"{prefix}_weight_scale"
        scale = torch.nn.Parameter(
            torch.empty(
                num_experts, a_dim, b_dim // MXFP8_BLOCK_SIZE, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # MXFP8 is a pure scale format: the e8m0 block exponent above carries
        # everything, there is no zero point. The int8/int4 schemes do have one,
        # so ModelSlimMoEMethod.apply reads layer.{w13,w2}_weight_offset
        # unconditionally when it builds AscendQuantInfo (where the field is
        # Optional). Register it as None so the attribute exists and resolves to
        # "no offset" rather than raising AttributeError. A None parameter is
        # skipped by named_parameters(), so no weight loader looks for it.
        layer.register_parameter(f"{prefix}_weight_offset", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Delegate weight processing to the NPU kernel for the fixed weight group.
        """
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
