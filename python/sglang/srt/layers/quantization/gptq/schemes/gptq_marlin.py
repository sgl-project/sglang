# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
    GPTQMarlinLinearKernel,
    MarlinLinearLayerConfig,
)
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)

from .gptq_scheme import GPTQLinearSchemeBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig

__all__ = ["GPTQMarlinLinearScheme"]


class GPTQMarlinLinearScheme(GPTQLinearSchemeBase):
    def __init__(self, quant_config: "GPTQMarlinConfig"):
        self.quant_config = quant_config
        self.kernel = GPTQMarlinLinearKernel(quant_config)

        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition

        self.kernel.kernel_config = MarlinLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
            has_g_idx=self.quant_config.desc_act,
        )

        group_size = (
            self.quant_config.group_size
            if self.quant_config.group_size != -1
            else input_size
        )

        if marlin_repeat_scales_on_all_ranks(
            self.quant_config.desc_act, self.quant_config.group_size, is_row_parallel
        ):
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        g_idx = RowvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )
        else:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        return self.kernel.apply(layer, x, bias)
