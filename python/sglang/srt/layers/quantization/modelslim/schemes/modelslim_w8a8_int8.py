# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Dict, List, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
    NPUW8A8Int8LinearMethod,
)
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme


class ModelSlimW8A8Int8(ModelSlimLinearScheme):
    def __init__(
        self,
        quant_config: Dict[str, any],
        prefix: str,
    ):
        self.quant_config = quant_config
        self.is_dynamic = (
            self.quant_config.get(prefix + ".weight", "") == "W8A8_DYNAMIC"
        )
        if self.is_dynamic:
            self.kernel = NPUW8A8Int8DynamicLinearMethod()
        else:
            self.kernel = NPUW8A8Int8LinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition), dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_offset = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_offset", weight_offset)

        if not self.is_dynamic:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(1, dtype=params_dtype),
                weight_loader=weight_loader,
            )
            input_scale.ignore_warning = True
            layer.register_parameter("input_scale", input_scale)

            input_offset = PerTensorScaleParameter(
                data=torch.empty(1, dtype=params_dtype),
                weight_loader=weight_loader,
            )
            input_offset.ignore_warning = True
            layer.register_parameter("input_offset", input_offset)

            quant_bias = ChannelQuantScaleParameter(
                data=torch.empty(output_size_per_partition, dtype=torch.int32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("quant_bias", quant_bias)

            if params_dtype == torch.bfloat16:
                deq_scale_dtype = torch.float32
            elif params_dtype == torch.float16:
                deq_scale_dtype = torch.int64
            else:
                raise ValueError(f"Unsupported params_dtype: {params_dtype}")
            deq_scale = ChannelQuantScaleParameter(
                data=torch.empty(output_size_per_partition, dtype=deq_scale_dtype),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("deq_scale", deq_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self.kernel.process_weights_after_loading(layer)

    def get_cp_decode_slice_attrs(self, layer, *, row_parallel: bool):
        """Describe the post-load [input, output] layout, not checkpoint axes."""
        if not self.is_dynamic: # 测试的模型是动态w8a8，静态的可能不一样，未测试
            raise ValueError("CP decode attention TP requires dynamic ModelSlim W8A8")

        attrs = [(layer, "weight", 0 if row_parallel else 1)]
        if not row_parallel:
            # Scales/offsets are per output channel. Row shards still produce
            # every output channel, so their scales/offsets must stay complete.
            attrs.extend(
                (layer, name, 0) for name in ("weight_scale", "weight_offset")
            )
        return attrs

    def prepare_cp_decode_weight(self, weight, dim, rank, size):
        from sglang.srt.hardware_backend.npu.utils import (
            NPUACLFormat,
            npu_format_cast,
        )

        # Narrow logical ND axes, then repack the local matrix for the GEMM.
        # A view into a full FRACTAL_NZ tensor is not a packed local weight.
        nd = torch.ops.npu.npu_format_cast(weight, NPUACLFormat.ACL_FORMAT_ND.value)
        width = nd.shape[dim] // size
        # A dim-0 shard can already be contiguous but still aliases a larger
        # allocation. Own the local storage before converting its NZ descriptor.
        local = nd.narrow(dim, rank * width, width).clone(
            memory_format=torch.contiguous_format
        )
        return npu_format_cast(local, NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)
