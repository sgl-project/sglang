# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_ptpc_linear,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    dispatch_w8a8_block_fp8_linear,
    maybe_create_device_identity,
    process_fp8_weight_block_strategy,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)

__all__ = ["CompressedTensorsW8A8Fp8"]

strategy_to_parameter_type = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure
        if self.weight_block_size is not None:
            assert not self.is_static_input_scheme
            self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        maybe_create_device_identity()
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = create_fp8_scale_parameter(
            strategy_to_parameter_type[self.strategy],
            output_partition_sizes,
            input_size_per_partition,
            layer.weight_block_size,
            weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                layer.weight,
                layer.weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            weight = weight.t()

        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                layer.weight, layer.weight_scale, getattr(layer, "input_scale", None)
            )
            weight = weight.t()

        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            weight, weight_scale = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale
            )
            input_scale = None

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # required by torch.compile to be torch.nn.Parameter
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        if input_scale is not None:
            layer.input_scale = Parameter(input_scale.data, requires_grad=False)

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight_block_size is not None:
            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )

        if _use_aiter and self.strategy == QuantizationStrategy.CHANNEL:
            return apply_fp8_ptpc_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )
        else:
            return apply_fp8_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )
