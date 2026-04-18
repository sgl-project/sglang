"""Self-contained Marlin FP4 fallback methods for NVFP4 on non-Blackwell GPUs."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.moe_runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import build_fp4_marlin_moe_quant_info
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_fp4_layer_for_marlin,
    prepare_moe_fp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


class ModelOptFp4MarlinLinearMethod(ModelOptFp4LinearMethod):
    """Marlin FP4 fallback for dense Linear layers."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)
        layer.weight_scale_2_marlin = Parameter(weight_scale_2, requires_grad=False)
        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.sglang.apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_scale_2_marlin,
            workspace=layer.marlin_workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )


class ModelOptNvFp4MarlinFusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """Marlin FP4 fallback for Fused MoE layers."""

    # Consumed by ``FusedMoE.__init__`` to skip routed_scaling_factor topk fusion
    # (Marlin applies the scaling during moe_sum_reduce).
    _is_marlin_fallback: bool = True

    def __init__(self, quant_config: ModelOptFp4Config):
        # Skip parent __init__ which enforces is_blackwell_supported().
        # User-facing fallback notice is emitted once in prepare_moe_fp4_layer_for_marlin.
        self.quant_config = quant_config
        self.enable_flashinfer_trtllm_moe = False
        self._cache_permute_indices = {}

    @property
    def load_up_proj_weight_first(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_moe_fp4_layer_for_marlin(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        quant_info = build_fp4_marlin_moe_quant_info(
            layer, self.moe_runner_config.num_experts
        )
        return self.runner.run(dispatch_output, quant_info)
