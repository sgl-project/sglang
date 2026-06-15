# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.srt.layers.quantization.utils import convert_to_channelwise

__all__ = ["CompressedTensorsW8A16Fp8"]

SUPPORTED_STRATEGIES = [
    QuantizationStrategy.CHANNEL,
    QuantizationStrategy.TENSOR,
    QuantizationStrategy.BLOCK,
]


class CompressedTensorsW8A16Fp8(CompressedTensorsLinearScheme):
    def __init__(
        self,
        strategy: str,
        is_static_input_scheme: bool,
        weight_block_size: Optional[List[int]] = None,
    ):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = weight_block_size

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # So if we have a fused module (QKV, MLP) with per tensor scales,
    # we expand each scale to its shard's channels.
    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.BLOCK:
            # compressed-tensors stores block scales as "weight_scale", while
            # the Marlin FP8 block path reads "weight_scale_inv". Rename it and
            # leave the weight in [output, input] layout (size_k_first=False),
            # mirroring the Fp8 block-quant path.
            if self.is_static_input_scheme:
                layer.input_scale = torch.nn.Parameter(
                    layer.input_scale.data, requires_grad=False
                )
            layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
            weight_scale = layer.weight_scale.data
            del layer._parameters["weight_scale"]
            layer.register_parameter(
                "weight_scale_inv",
                torch.nn.Parameter(weight_scale, requires_grad=False),
            )
            prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            return

        if self.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(
                layer.weight_scale, layer.logical_widths
            )
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False)
        else:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )

        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False)

        if self.is_static_input_scheme:
            # required by torch.compile to be torch.nn.Parameter
            layer.input_scale = torch.nn.Parameter(
                layer.input_scale.data, requires_grad=False
            )
        prepare_fp8_layer_for_marlin(layer, size_k_first=True)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_partition_sizes: List[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = (
            self.weight_block_size
            if self.strategy == QuantizationStrategy.BLOCK
            else None
        )

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        elif self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            block_n, block_k = self.weight_block_size
            for output_partition_size in output_partition_sizes:
                assert output_partition_size % block_n == 0, (
                    f"output_partition_size={output_partition_size} is not "
                    f"divisible by block_n={block_n}"
                )
            assert input_size_per_partition % block_k == 0, (
                f"input_size_per_partition={input_size_per_partition} is not "
                f"divisible by block_k={block_k}"
            )
            weight_scale = BlockQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            raise ValueError(
                f"Unsupported weight strategy={self.strategy}, "
                f"supported strategies are {SUPPORTED_STRATEGIES}"
            )

        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
