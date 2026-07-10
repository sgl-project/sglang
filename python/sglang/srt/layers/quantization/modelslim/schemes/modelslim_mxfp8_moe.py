"""ModelSlim MXFP8 offline scheme for MoE layers on Ascend NPU (SRT).

Loads weights pre-quantised by msmodelslim: float8_e4m3fn weights + uint8
block scales (block_size=32). The layout transform (2D scale → 3D
pair-split + transpose) and the forward pass are delegated to
``NPUMXFP8FusedMoEMethod`` (the same kernel used for online MXFP8 MoE).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUMXFP8FusedMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils.common import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DispatchOutput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimMXFP8MoEScheme",
]

MXFP8_BLOCK_SIZE = 32


class ModelSlimMXFP8MoEScheme(ModelSlimMoEScheme):
    """Offline MXFP8 scheme for FusedMoE layers.

    Creates float8_e4m3fn weight placeholders and uint8 block-scale
    placeholders (shape [E, N, K // 32]). ``process_weights_after_loading``
    delegates to ``NPUMXFP8FusedMoEMethod`` which detects the fp8 dtype and
    takes the offline (transpose-only) branch.
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        prefix: str = None,
    ):
        self.quant_config = quant_config
        self.kernel = NPUMXFP8FusedMoEMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported as FMWSS,
        )

        extra_weight_attrs.update(
            {"quant_method": FMWSS.BLOCK.value}
        )

        w13_up_dim = 2 * intermediate_size_per_partition

        # w13 (gate_up): [E, 2I, H] float8_e4m3fn
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, w13_up_dim, hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # w2 (down): [E, H, I] float8_e4m3fn
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # w13 scale: [E, 2I, H // 32] uint8 (block_size=32, msmodelslim layout)
        w13_scale_dim = hidden_size // MXFP8_BLOCK_SIZE
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, w13_up_dim, w13_scale_dim, dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        # w2 scale: [E, H, I // 32] uint8
        w2_scale_dim = intermediate_size_per_partition // MXFP8_BLOCK_SIZE
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, w2_scale_dim, dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: MoeRunnerConfig,
    ) -> None:
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        return self.kernel.apply(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None,
        group_list_type: int,
        group_list: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.kernel.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )
