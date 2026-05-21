"""Self-contained Marlin FP4 fallback schemes for compressed-tensors NVFP4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import build_fp4_marlin_moe_quant_info
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    CompressedTensorsW4A4Fp4,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
    CompressedTensorsW4A4Nvfp4MoE,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_fp4_layer_for_marlin,
    prepare_moe_fp4_layer_for_marlin,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


class CompressedTensorsW4A4MarlinFp4(CompressedTensorsW4A4Fp4):
    """Marlin FP4 fallback scheme for compressed-tensors W4A4 Linear."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: List[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        super().create_weights(
            layer,
            output_partition_sizes,
            input_size_per_partition,
            params_dtype,
            weight_loader,
            **kwargs,
        )
        layer.params_dtype = params_dtype

    def process_weights_after_loading(self, layer) -> None:
        global_scale = layer.weight_global_scale.max().to(torch.float32)
        layer.weight_global_scale = torch.nn.Parameter(
            global_scale, requires_grad=False
        )
        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight_packed",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_global_scale",
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.sglang.apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight_packed,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.marlin_workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )


class CompressedTensorsW4A4Nvfp4MarlinMoE(CompressedTensorsW4A4Nvfp4MoE):
    """Marlin FP4 fallback scheme for compressed-tensors W4A4 MoE."""

    _is_marlin_fallback: bool = True

    def __init__(self):
        # Skip parent __init__ which enforces is_blackwell_supported().
        # User-facing fallback notice is emitted once in prepare_moe_fp4_layer_for_marlin.
        self.group_size = 16
        self.use_flashinfer_trtllm = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.intermediate_size_per_partition = intermediate_size_per_partition

    def process_weights_after_loading(self, layer) -> None:
        # Parent handles packed->weight rename and weight_scale_2 inversion;
        # TRTLLM-specific branches are skipped because use_flashinfer_trtllm=False.
        super().process_weights_after_loading(layer)
        prepare_moe_fp4_layer_for_marlin(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        quant_info = build_fp4_marlin_moe_quant_info(
            layer, self.moe_runner_config.num_experts
        )
        return self.runner.run(dispatch_output, quant_info)
