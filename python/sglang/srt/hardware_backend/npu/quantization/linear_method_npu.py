from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class _NPULinearMethodBase(LinearMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        super().__init__()
        self.quant_config = quant_config


class NPUW8A8Int8LinearMethod(_NPULinearMethodBase):

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

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = torch.ops.npu.npu_quantize(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        return torch.ops.npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)

        expanding_factor = layer.weight.data.shape[0]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )


class NPUW8A8Int8DynamicLinearMethod(_NPULinearMethodBase):

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

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=bias,
            output_dtype=original_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()
