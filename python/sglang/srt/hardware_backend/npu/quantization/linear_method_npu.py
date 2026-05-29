from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class _NPULinearMethodBase(LinearMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW8A8Int8LinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

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


class NPUW8A8Int8DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if isinstance(x, tuple):
            """dynamic_scale is calculated in malprolog kernel"""
            original_dtype = torch.bfloat16
            quant_out, dynamic_scale = x
        else:
            original_dtype = x.dtype
            quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPU_W4A4DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        weight.data = self._w4a4_pack_int4(weight.data)
        weight.data = weight.data.transpose(-2, -1).contiguous()        
        weight.data = npu_format_cast(weight.data)                                          
        weight.data = self._pack_to_int32(weight.data)

    def _pack_int4(self, weight) -> torch.Tensor:
        """
        Pack int4 weight to int8 weight
        @param weight: torch.Tensor, int4 weight
        @return: torch.Tensor, int8 weight
        """
        weight = weight.to(torch.int8)
        e = 0  # number of experts
        if len(weight.shape) == 2:
            k, n = weight.shape
        elif len(weight.shape) == 3:
            e, k, n = weight.shape
        n_new = n // 2 + n % 2
      
        if n_new != n // 2:
            raise AssertionError("n dimension should be even")
      
        weight = weight.reshape(-1, 2)
        weight0 = weight[:, :1]
        weight1 = weight[:, 1:]
      
        weight1_4 = torch.bitwise_left_shift(weight1, 4)
        weight2_4 = weight0 & 0b00001111
      
        weight_add = torch.bitwise_or(weight1_4, weight2_4)
        if e == 0:
            weight_res = weight_add.reshape(k, n_new)
        else:
            weight_res = weight_add.reshape(e, k, n_new)
        return weight_res
  
  
    def _w4a4_pack_int4(self, save_quant_weight):
        """
        Pack int4 weight to int8 weight
        @param save_quant_weight: torch.Tensor, int4 weight
        @return: torch.Tensor, int8 weight
        """
        weight = save_quant_weight.transpose(-1, -2).contiguous()
        packed_weight_tensor = self._pack_int4(weight)
        packed_weight_tensor = packed_weight_tensor.transpose(-1, -2).contiguous()
        return packed_weight_tensor

    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        return weight.view(torch.int32).contiguous()
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )
