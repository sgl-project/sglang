# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Callable
from typing import Optional

import torch

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_mxfp4_layer_for_marlin,
)
from sglang.srt.layers.utils.common import copy_or_rebind_param

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A16Mxfp4"]

MXFP4_GROUP_SIZE = 32


class _Mxfp4MarlinQuantConfig:
    """Carries the one field `prepare_mxfp4_layer_for_marlin` reads off
    `layer.quant_config`. Setting it is what enables that helper's group_size
    validation, which is skipped entirely when the attribute is absent."""

    group_size = MXFP4_GROUP_SIZE


class CompressedTensorsW4A16Mxfp4(CompressedTensorsLinearScheme):
    """Weight-only MXFP4 via the FP4 Marlin kernel: FP4 weights with E8M0 group
    scales, BF16 activations.

    Serves two config shapes:
      - mxfp4a16, which has no ``input_activations`` at all;
      - mxfp4 (w4a4), where the activation quantization is dropped. Both carry
        byte-identical weights, so one scheme covers them.

    BF16-only: the E8M0 scale decoder is instantiated for bf16 alone (see
    marlin/dequant.h), unlike NVFP4's E4M3 scales which also have an fp16 path.
    """

    def __init__(self, has_input_activations: bool = False):
        # True for a w4a4 checkpoint being served weight-only. Unlike NVFP4,
        # MXFP4 activation quantization carries no global-scale tensor, so there
        # is no extra parameter to register.
        self.has_input_activations = has_input_activations
        self.group_size = MXFP4_GROUP_SIZE

    @classmethod
    def get_min_capability(cls) -> int:
        # FP4 Marlin is a weight-only kernel and needs no FP4 tensor cores.
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        # prepare_mxfp4_layer_for_marlin reads both of these; without
        # params_dtype it raises on a None activation dtype.
        layer.params_dtype = params_dtype
        layer.quant_config = _Mxfp4MarlinQuantConfig()

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # MXFP4 scales are raw E8M0 exponent bytes, not a float8 type.
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # prepare_mxfp4_layer_for_marlin operates on `weight`; compressed-tensors
        # names the packed weight `weight_packed`.
        copy_or_rebind_param(layer, "weight", layer.weight_packed.data)
        del layer.weight_packed

        prepare_mxfp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=None,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
