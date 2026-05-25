# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter

from .awq_scheme import AWQLinearSchemeBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.awq.awq import AWQConfig

__all__ = ["AWQLinearScheme", "AWQAscendLinearScheme"]


class AWQLinearScheme(AWQLinearSchemeBase):
    def __init__(self, quant_config: "AWQConfig"):
        self.quant_config = quant_config
        self.kernel = self._init_kernel(quant_config)

    def _init_kernel(self, quant_config: "AWQConfig"):
        from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
            AWQLinearKernel,
        )

        return AWQLinearKernel(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        return self.kernel.apply(layer, x, bias)


class AWQAscendLinearScheme(AWQLinearScheme):
    def _init_kernel(self, quant_config: "AWQConfig"):
        from sglang.srt.hardware_backend.npu.quantization.awq_kernels import (
            AWQAscendLinearKernel,
        )

        return AWQAscendLinearKernel(quant_config)
