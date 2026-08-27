from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import ceil_div, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.models.inkling_common.quantization.config import (
        InklingModelOptNvfp4Config,
    )


logger = logging.getLogger(__name__)

MXFP_BLOCK_SIZE = 32
NVFP_BLOCK_SIZE = 16

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


class InklingMoEMethodBase(FusedMoEMethodBase):
    """Protocol for Inkling MoE quant methods"""


class InklingNvfp4MoEMethod(InklingMoEMethodBase):
    def __init__(self, quant_config: InklingModelOptNvfp4Config):
        self.quant_config = quant_config
        self.runner: MoeRunner | None = None
        self.moe_runner_config: MoeRunnerConfig | None = None
        self._srt_trtllm_runner: MoeRunner | None = None
        self._srt_trtllm_runner_config: MoeRunnerConfig | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,  # type: ignore[reportMissingParameterType]
    ):
        from torch.nn.parameter import Parameter

        from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP

        assert isinstance(
            layer, InklingBatchDenseMLP
        ), "InklingNvfp4MoEMethod is only used for InklingBatchDenseMLP (shared experts)"

        w13_up_dim = 2 * intermediate_size_per_partition

        # half shapes for packed uint8 weights
        w13_weight = Parameter(
            torch.empty(num_experts, w13_up_dim, hidden_size // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale_dim1_shape = ceil_div(w13_up_dim, self.quant_config.dim1_group_size)
        w13_scale_dim2_shape = ceil_div(hidden_size, self.quant_config.dim2_group_size)
        w2_scale_dim1_shape = ceil_div(hidden_size, self.quant_config.dim1_group_size)
        w2_scale_dim2_shape = ceil_div(
            intermediate_size_per_partition, self.quant_config.dim2_group_size
        )
        w13_scale = Parameter(
            torch.empty(
                num_experts,
                w13_scale_dim1_shape,
                w13_scale_dim2_shape,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        w2_scale = Parameter(
            torch.empty(
                num_experts,
                w2_scale_dim1_shape,
                w2_scale_dim2_shape,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        w13_scale2 = Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        w2_scale2 = Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        w13_original_shape = Parameter(
            torch.empty(3, dtype=torch.int32),
            requires_grad=False,
        )
        w2_original_shape = Parameter(
            torch.empty(3, dtype=torch.int32),
            requires_grad=False,
        )
        w13_input_amax = Parameter(
            torch.full((1,), float("nan"), dtype=torch.float32),
            requires_grad=False,
        )
        w2_input_amax = Parameter(
            torch.full((1,), float("nan"), dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale", w13_scale)
        layer.register_parameter("w2_scale", w2_scale)
        layer.register_parameter("w13_scale2", w13_scale2)
        layer.register_parameter("w2_scale2", w2_scale2)
        layer.register_parameter("w13_original_shape", w13_original_shape)
        layer.register_parameter("w2_original_shape", w2_original_shape)
        layer.register_parameter("w13_input_amax", w13_input_amax)
        layer.register_parameter("w2_input_amax", w2_input_amax)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w13_scale2, extra_weight_attrs)
        set_weight_attrs(w2_scale2, extra_weight_attrs)
        set_weight_attrs(w13_original_shape, extra_weight_attrs)
        set_weight_attrs(w2_original_shape, extra_weight_attrs)
        set_weight_attrs(w13_input_amax, extra_weight_attrs)
        set_weight_attrs(w2_input_amax, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights for the dense shared-expert NVFP4 path.

        Routed NVFP4 MoE now uses ModelOptNvFp4FusedMoEMethod; this hook is reached
        only for shared experts (InklingBatchDenseMLP), which carry an ``_fp4_strategy``
        and run their own weight preparation.
        """
        if getattr(layer, "_fp4_strategy", None) is not None:
            layer.process_weights_after_loading()

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,  # type: ignore[override]
    ):
        # Kept only to satisfy the FusedMoEMethodBase abstract interface.
        # InklingNvfp4MoEMethod serves the dense shared-expert path (InklingBatchDenseMLP,
        # which uses its own FP4 serving); routed NVFP4 MoE uses
        # ModelOptNvFp4FusedMoEMethod.
        raise NotImplementedError(
            "InklingNvfp4MoEMethod is the dense shared-expert method; routed NVFP4 "
            "MoE uses ModelOptNvFp4FusedMoEMethod."
        )
