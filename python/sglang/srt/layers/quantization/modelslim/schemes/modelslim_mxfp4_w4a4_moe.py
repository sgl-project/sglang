"""ModelSlim MXFP4 W4A4 offline scheme for MoE layers on Ascend NPU (SRT).

Loads weights pre-quantised by msmodelslim: packed-fp4 uint8 weights + uint8
block scales (block_size=32, E8M0 +127 biased) — the same on-disk format as the
W4A8 MXFP MoE scheme, since both store MXFP4 weights. The activation format is
what differs at inference: W4A4 runs fp4 activations through a fused gmm1
(swiglu + fp4 requant) and an fp4 gmm2, delegated to ``NPUMXFP4W4A4MoEMethod``
— the same kernel the online ``--quantization mxfp4`` experts path uses.

Mirrors vllm-ascend ``AscendW4A4MXFP4DynamicFusedMoEMethod`` (single-level).
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP4W4A4MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

__all__ = [
    "ModelSlimMXFP4W4A4MoEScheme",
]

# Block (group) size of the msmodelslim W4A4_MXFP4 export format.
MXFP4_BLOCK_SIZE = 32
# Two fp4 values packed into one uint8 → K dimension is halved.
MXFP4_PACK_FACTOR = 2


class ModelSlimMXFP4W4A4MoEScheme(ModelSlimMoEScheme):
    """
    Offline MXFP4 W4A4 MoE scheme that creates weights for either the
    w13 (gate+up) or w2 (down) projection group.

    Two instances of this class are used per MoE layer:
      - weight_prefix="w13"   → handles the fused gate_proj + up_proj weights
      - weight_prefix="w2"    → handles the down_proj weights

    The uint8 weight dtype allocated here is what tells
    ``NPUMXFP4W4A4MoEMethod.process_weights_after_loading`` to take its offline
    (re-layout only) branch instead of quantising the weights itself.
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,  # "w13" or "w2"
    ) -> None:
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got '{weight_prefix}'"
            )
        self.quant_config = quant_config
        self.weight_prefix = weight_prefix
        self.kernel = NPUMXFP4W4A4MoEMethod(weight_prefix)

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

        # Determine shape based on weight group.
        # w13: [E, 2*inter, hidden] → packed fp4: [E, 2*inter, hidden//2]
        # w2:  [E, hidden, inter]   → packed fp4: [E, hidden, inter//2]
        # Scale: [E, N, K//32] uint8 (E8M0, block_size=32).
        # Mirrors vllm-ascend AscendW4A4MXFP4DynamicFusedMoEMethod.get_weight /
        # get_dynamic_quant_param exactly.
        if self.weight_prefix == "w13":
            a_dim = 2 * intermediate_size_per_partition
            b_dim = hidden_size
        else:  # w2
            a_dim = hidden_size
            b_dim = intermediate_size_per_partition

        prefix = self.weight_prefix

        # Packed fp4 weight: two fp4 values per uint8 → K dimension // 2.
        weight_name = f"{prefix}_weight"
        weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                a_dim,
                b_dim // MXFP4_PACK_FACTOR,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter(weight_name, weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # E8M0 block scale: one uint8 per block of 32 elements.
        scale_name = f"{prefix}_weight_scale"
        scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                a_dim,
                b_dim // MXFP4_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # W4A4_MXFP4 is a pure scale format with no zero point. The shared
        # ModelSlim MoE apply path still reads this optional field when it builds
        # AscendQuantInfo, so register None to make that contract explicit.
        layer.register_parameter(f"{prefix}_weight_offset", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Delegate weight processing to the NPU kernel for the fixed weight group."""
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
