# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import GPTQLinearKernel
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from sglang.srt.utils import set_weight_attrs

from .gptq_scheme import GPTQLinearSchemeBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig

__all__ = ["GPTQLinearScheme", "GPTQAscendLinearScheme"]


class GPTQLinearScheme(GPTQLinearSchemeBase):
    def __init__(self, quant_config: "GPTQConfig"):
        self.quant_config = quant_config
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"
        self.kernel = self._init_kernel(quant_config)

    def _init_kernel(self, quant_config: "GPTQConfig"):
        return GPTQLinearKernel(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
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
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        group_size = (
            self.quant_config.group_size
            if self.quant_config.group_size != -1
            else input_size
        )
        self.kernel.use_shuffle = True
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if (
            input_size != input_size_per_partition
            and self.quant_config.group_size != -1
        ):
            if self.quant_config.desc_act:
                self.kernel.use_shuffle = False
            else:
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

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
            data=torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )
        qzeros_args = {
            "data": torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }
        if scale_and_zero_input_dim is None:
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
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        return self.kernel.apply(layer, x, bias)


class GPTQAscendLinearScheme(GPTQLinearScheme):
    def _init_kernel(self, quant_config: "GPTQConfig"):
        from sglang.srt.hardware_backend.npu.quantization.gptq_kernels import (
            GPTQLinearAscendKernel,
        )

        return GPTQLinearAscendKernel(quant_config)

    def create_weights(self, layer: torch.nn.Module, **kwargs):
        super().create_weights(layer=layer, **kwargs)
        set_weight_attrs(layer.qzeros, {"pack_factor": self.quant_config.pack_factor})
        set_weight_attrs(layer.qweight, {"pack_factor": self.quant_config.pack_factor})

        if self.quant_config.desc_act:
            raise ValueError(
                "Currently, desc_act (True) is not supported by GPTQ "
                "quantization on npu."
            )
