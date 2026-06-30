from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW8A8Mxfp8MoE",
]


class ModelSlimW8A8Mxfp8MoE(ModelSlimMoEScheme):
    """
    W8A8 MXFP8 MoE scheme for NPU.

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
        # Kernel that will process the weights (transpose, reshape scales)
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUW8A8Mxfp8MoEMethod,
        )
        self.kernel = NPUW8A8Mxfp8MoEMethod()
        self.weight_prefix = weight_prefix
        self.group_size = group_size

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
        Create weight and scale parameters in MXFP8 format.
        """
        # --- fixed shapes per prefix, matching the new MXFP8 reference ---
        if self.weight_prefix == "w13":
            out_features = 2 * intermediate_size_per_partition
            in_features = hidden_size
        else:  # w2
            out_features = hidden_size
            in_features = intermediate_size_per_partition

        weight_shape = (num_experts, out_features, in_features)

        # Scale shape: group‑wise scaling; if group_size is 0 fall back to per‑channel (scale per out feature)
        if self.group_size > 0:
            scale_shape = (num_experts, out_features, in_features // self.group_size)
        else:
            scale_shape = (num_experts, out_features, 1)

        self._create_weight_params(
            layer,
            self.weight_prefix,
            weight_shape,
            scale_shape,
            extra_weight_attrs,
        )

    @staticmethod
    def _create_weight_params(
        layer: torch.nn.Module,
        prefix: str,
        weight_shape: tuple,
        scale_shape: tuple,
        extra_weight_attrs: dict,
    ) -> None:
        """Register weight (float8_e4m3fn) and scale (uint8) parameters."""
        # Weight (e4m3fn)
        weight = torch.nn.Parameter(
            torch.empty(weight_shape, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # Scale (uint8, as used by the MXFP8 kernel)
        scale = torch.nn.Parameter(
            torch.empty(scale_shape, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight_scale", scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # No offset in the MXFP8 scheme.

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Delegate weight processing to the NPU kernel for the fixed weight group.
        The kernel will handle transposes and scale reshaping as required.
        """
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
