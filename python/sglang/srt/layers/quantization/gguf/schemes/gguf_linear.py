# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter

from sglang.srt.utils import set_weight_attrs

from .gguf_scheme import (
    GGUFLinearSchemeBase,
    GGUFUninitializedParameter,
    create_padded_weight_param,
)

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.gguf.gguf import GGUFConfig

__all__ = ["GGUFLinearScheme", "GGUFAscendLinearScheme"]


class GGUFLinearScheme(GGUFLinearSchemeBase):
    def __init__(self, quant_config: "GGUFConfig"):
        self.quant_config = quant_config
        self.kernel = self._init_kernel(quant_config)

    def _init_kernel(self, quant_config: "GGUFConfig"):
        from sglang.srt.hardware_backend.gpu.quantization.gguf_kernels import (
            GGUFLinearKernel,
        )

        return GGUFLinearKernel(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.kernel.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)


class GGUFAscendLinearScheme(GGUFLinearScheme):
    def _init_kernel(self, quant_config: "GGUFConfig"):
        from sglang.srt.hardware_backend.npu.quantization.gguf_kernels import (
            GGUFAscendLinearKernel,
        )

        return GGUFAscendLinearKernel(quant_config)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        from sglang.srt.hardware_backend.npu.quantization.gguf_kernels import (
            DEQUANT_TYPES,
            UNQUANTIZED_TYPES,
        )

        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            raise ValueError(
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} in layer."
            )
        create_padded_weight_param(layer)
        self.kernel.process_weights_after_loading(layer)
