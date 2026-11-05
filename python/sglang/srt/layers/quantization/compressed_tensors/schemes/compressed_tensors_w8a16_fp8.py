# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.parameter import (
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
from sglang.srt.utils import is_cpu, use_intel_amx_backend

__all__ = ["CompressedTensorsW8A16Fp8"]

_is_cpu = is_cpu()

SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]


class CompressedTensorsW8A16Fp8(CompressedTensorsLinearScheme):
    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # So if we have a fused module (QKV, MLP) with per tensor scales,
    # we expand each scale to its shard's channels.
    def process_weights_after_loading(self, layer) -> None:
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

        if self.is_static_input_scheme:
            # required by torch.compile to be torch.nn.Parameter
            layer.input_scale = torch.nn.Parameter(
                layer.input_scale.data, requires_grad=False
            )

        if _is_cpu:
            # No marlin kernel on CPU; apply_weights calls fp8_scaled_mm_cpu
            # instead. Pack the weight to VNNI layout once here (also converts
            # bias to float32) so apply_weights doesn't repack on every call.
            _amx_process_weight_after_loading(layer, ["weight"])
            return

        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False)
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
        if _is_cpu:
            # block_size_N == 1 selects the per-channel (as opposed to block)
            # weight-only scale layout in fp8_scaled_mm_cpu: one scale value
            # per output row (layer.weight_scale is [N, 1]), constant across K.
            # The kernel requires bias to be float32 regardless of x's dtype.
            return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                x,
                layer.weight,
                layer.weight_scale,
                [1, layer.weight.shape[1]],
                bias.float() if bias is not None else None,
                x.dtype,
                use_intel_amx_backend(layer),  # is_vnni
            )
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
