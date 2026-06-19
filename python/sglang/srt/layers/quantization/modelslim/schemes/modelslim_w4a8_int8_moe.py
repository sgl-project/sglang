from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A8Int8MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["ModelSlimW4A8Int8MoE"]


class ModelSlimW4A8Int8MoE(ModelSlimMoEScheme):
    """
    W4A8 MoE scheme for a single weight group (w13 or w2).

    Two instances of this class are created per MoE layer:
      - weight_prefix="w13"  → handles gate + up projections
      - weight_prefix="w2"   → handles down projection

    Configuration flags (``is_per_channel_weight``, ``activation_use_clip``)
    are passed to the underlying NPU kernel.
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,
        group_size: int = 0,
        tp_size: int = 1,
        activation_use_clip: bool = False,
    ) -> None:
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got '{weight_prefix}'"
            )
        self.quant_config = quant_config
        self.weight_prefix = weight_prefix
        self.group_size = group_size
        self.tp_size = tp_size
        self.is_per_channel_weight = group_size == 0
        self.activation_use_clip = activation_use_clip
        self.kernel = NPUW4A8Int8MoEMethod(
            is_per_channel_weight=self.is_per_channel_weight,
            activation_use_clip=self.activation_use_clip,
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
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # Determine dimensions based on weight group
        if self.weight_prefix == "w13":
            out_features = intermediate_size_per_partition
            in_features = hidden_size
            bias_last_dim = 1
        else:  # w2
            out_features = hidden_size // 2
            in_features = intermediate_size_per_partition
            bias_last_dim = 16 // self.tp_size

        prefix = self.weight_prefix

        # ---- weight ----
        weight = torch.nn.Parameter(
            torch.empty(num_experts, out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # ---- scale ----

        scale = torch.nn.Parameter(
            torch.empty(num_experts, 2 * out_features, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight_scale", scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # ---- offset ----
        offset = torch.nn.Parameter(
            torch.empty(num_experts, 2 * out_features, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight_offset", offset)
        set_weight_attrs(offset, extra_weight_attrs)

        # ---- per‑group second scale/offset (when not per‑channel) ----
        if not self.is_per_channel_weight:
            scale_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * out_features,
                    in_features // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter(f"{prefix}_weight_scale_second", scale_second)
            set_weight_attrs(scale_second, extra_weight_attrs)

            offset_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * out_features,
                    in_features // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter(f"{prefix}_weight_offset_second", offset_second)
            set_weight_attrs(offset_second, extra_weight_attrs)

        # ---- bias for scale (activation clip path) ----
        # This parameter is always created; the kernel uses it only when activation_use_clip is True.
        scale_bias = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * out_features, bias_last_dim, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_scale_bias", scale_bias)
        set_weight_attrs(scale_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Delegate weight processing to the kernel for the assigned weight group."""
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
