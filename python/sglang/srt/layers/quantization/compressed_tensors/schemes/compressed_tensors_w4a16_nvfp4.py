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
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.layers.utils.common import copy_or_rebind_param

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A16Fp4"]

NVFP4_GROUP_SIZE = 16


class _Nvfp4MarlinQuantConfig:
    """Carries the one field `prepare_nvfp4_layer_for_marlin` reads off
    `layer.quant_config`. Setting it is what enables that helper's group_size
    validation, which is skipped entirely when the attribute is absent."""

    group_size = NVFP4_GROUP_SIZE


class CompressedTensorsW4A16Fp4(CompressedTensorsLinearScheme):
    """Weight-only NVFP4 via the FP4 Marlin kernel: FP4 weights, FP16/BF16
    activations.

    Serves two config shapes:
      - nvfp4a16, which has no ``input_activations`` at all;
      - nvfp4 (w4a4) on a pre-Blackwell GPU, where the activation quantization
        is dropped and the checkpoint's ``input_global_scale`` goes unused.

    Kept separate from ``CompressedTensorsW4A4Fp4`` rather than folded in behind
    a flag because ``get_min_capability`` is a classmethod resolved on the
    instance's class, so a shared class would also lower the SM100 gate that
    guards the native w4a4 path.
    """

    def __init__(self, has_input_global_scale: bool = False):
        # True for a w4a4 checkpoint being served weight-only: the parameter
        # must still be registered so the loader finds a home for it.
        self.has_input_global_scale = has_input_global_scale
        self.group_size = NVFP4_GROUP_SIZE

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
        # prepare_nvfp4_layer_for_marlin reads both of these; without
        # params_dtype it raises on a None activation dtype.
        layer.params_dtype = params_dtype
        layer.quant_config = _Nvfp4MarlinQuantConfig()

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

        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        if self.has_input_global_scale:
            input_global_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The activation scale is meaningless for a weight-only kernel.
        if self.has_input_global_scale:
            del layer.input_global_scale

        if torch.unique(layer.weight_global_scale).numel() != 1:
            logger.warning(
                "In NVFP4 weight-only linear, weight_global_scale differs across "
                "fused parallel layers. Accuracy may be degraded."
            )
        # compressed-tensors stores the global scale as a divisor (1/scale),
        # while Marlin's prep helper expects the scale itself. Skipping the
        # inversion overflows the bf16 bias multiply to inf, zeroing all logits.
        copy_or_rebind_param(
            layer,
            "weight_global_scale",
            (1 / layer.weight_global_scale.max()).to(torch.float32),
        )

        # prepare_nvfp4_layer_for_marlin operates on `weight`; compressed-tensors
        # names the packed weight `weight_packed`.
        copy_or_rebind_param(layer, "weight", layer.weight_packed.data)
        del layer.weight_packed

        prepare_nvfp4_layer_for_marlin(layer)

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
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
