from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A4Int4MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW4A4Int4MoE",
]


class ModelSlimW4A4Int4MoE(ModelSlimMoEScheme):
    """
    W4A4 integer MoE scheme that creates weights for either the
    w13 (gate+up) or w2 (down) projection group.

    Two instances of this class are used per MoE layer:
      - weight_prefix="w13"   → handles the fused gate_proj + up_proj weights
      - weight_prefix="w2"    → handles the down_proj weights
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,  # "w13" or "w2"
        group_size: int = 0,
    ) -> None:
        self.quant_config = quant_config
        self.kernel = NPUW4A4Int4MoEMethod()
        self.weight_prefix = weight_prefix
        self.group_size = group_size
        self.is_per_channel_weight = group_size == 0
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
        """
        Create and register weight, scale, and offset parameters for the layer.
        Shape depends on the W4A4 packing environment flag and whether the weight
        prefix is "w13" or "w2".
        """
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        # --- compute shapes based on the packing path and prefix ---
        if envs.SGLANG_NPU_W4A4_NEW_PACKING.get():
            if self.weight_prefix == "w13":
                out_features = intermediate_size_per_partition
                in_features = hidden_size
            else:  # w2
                out_features = hidden_size // 2
                in_features = intermediate_size_per_partition

            weight_shape = (num_experts, out_features, in_features)
            scale_shape = (num_experts, 2 * out_features, 1)
        else:
            if self.weight_prefix == "w13":
                a_dim = 2 * intermediate_size_per_partition
                b_dim = hidden_size
            else:  # w2
                a_dim = hidden_size
                b_dim = intermediate_size_per_partition

            weight_shape = (num_experts, a_dim, b_dim)
            scale_shape = (num_experts, a_dim, 1)

        offset_shape = scale_shape  # offset always matches scale

        self._create_weight_params(
            layer,
            self.weight_prefix,
            weight_shape,
            scale_shape,
            offset_shape,
            extra_weight_attrs,
        )

    @staticmethod
    def _create_weight_params(
        layer: torch.nn.Module,
        prefix: str,
        weight_shape: tuple,
        scale_shape: tuple,
        offset_shape: tuple,
        extra_weight_attrs: dict,
    ) -> None:
        """Helper that registers weight, scale, and offset as parameters."""
        # Weight
        weight = torch.nn.Parameter(
            torch.empty(weight_shape, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # Scale
        scale = torch.nn.Parameter(
            torch.empty(scale_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight_scale", scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # Offset
        offset = torch.nn.Parameter(
            torch.empty(offset_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight_offset", offset)
        set_weight_attrs(offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Delegate weight processing to the NPU kernel for the fixed weight group.
        """
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
