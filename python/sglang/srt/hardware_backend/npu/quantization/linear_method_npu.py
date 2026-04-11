from typing import TYPE_CHECKING, Optional, List

import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.utils import is_npu, is_npu_before_atlas_a5

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

_is_npu = is_npu()
_is_npu_before_atlas_a5 = is_npu_before_atlas_a5()
if _is_npu:
    import torch_npu

def fp8_matmul_npu(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None, # if input is dynamic quant, input_scale == None
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if block_size != [128, 128]:
        raise ValueError("fp8_matmul_npu func now only supports block_size == [128, 128]")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError("fp8_matmul_npu func now only supports fp8e4m3")

    orig_shape = input.shape
    k = orig_shape[-1]
    input_2d = input.reshape(-1, k).contiguous()
 
    if _is_npu_before_atlas_a5:
        output_2d = torch.ops.npu.softfp8_w8a16_matmul(input_2d, weight, weight_scale, "bf16")
    else:
        group_sizes = (1, 128, 128)

        x_fp8, x_scale = torch.ops.npu.npu_dynamic_block_quant(
            input_2d,
            dst_type=torch.float8_e4m3fn,
            row_block_size=1,
            col_block_size=128,
        )

        output_2d = torch.ops.npu.npu_quant_matmul(
            x_fp8,
            weight,
            scale=weight_scale,
            pertoken_scale=x_scale,
            output_dtype=torch.bfloat16,
            group_sizes=group_sizes,
        )

    output = output_2d.reshape(*orig_shape[:-1], output_2d.shape[-1])
 
    return output

class _NPULinearMethodBase(LinearMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW8A8MxFp8LinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(-1, -2).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)
        weight_scale = layer.weight_scale.data.transpose(
            -1, -2
        ).contiguous()
        k32, n = weight_scale.shape
        weight_scale = weight_scale.view(k32 // 2, 2, n).permute(0, 2, 1).contiguous()
        layer.weight_scale.data = weight_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight=layer.weight
        weight_scale=layer.weight_scale
        group_sizes = (1, 1, 32)

        orig_shape = x.shape
        K = orig_shape[-1]
        x = x.reshape(-1, K).contiguous()

        x_fp8, x_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x,
            axis=1,
            dst_type=torch.float8_e4m3fn,
            block_size=32,
            scale_alg=None,
        )

        out = torch.ops.npu.npu_quant_matmul(
            x_fp8,
            weight,
            scale=weight_scale,
            pertoken_scale=x_scale,
            output_dtype=torch.bfloat16,
            group_sizes=group_sizes,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu
        )

        return out.reshape(*orig_shape[:-1], out.shape[-1])


class NPUW4A8MxFpLinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2).transpose(0, 1)
        weight_scale = layer.weight_scale.data
        weight_scale = weight_scale.reshape(weight_scale.shape[0], weight_scale.shape[1] // 2, 2).transpose(0, 1)
        layer.weight_scale.data = weight_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight=layer.weight
        weight_scale=layer.weight_scale
        group_sizes = (1, 1, 32)

        orig_shape = x.shape
        K = orig_shape[-1]
        x = x.reshape(-1, K).contiguous()

        x_fp8, x_scale = torch.ops.npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

        out = torch.ops.npu.npu_quant_matmul(
            x_fp8,
            weight,
            scale=weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=torch.bfloat16,
            group_sizes=group_sizes,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
        )
        return out.reshape(*orig_shape[:-1], out.shape[-1])


class NPUW4A4MxFp4LinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(-1, -2).contiguous()
        weight_scale = layer.weight_scale.data.transpose(-1, -2).contiguous()
        k_group, n = weight_scale.shape
        layer.weight_scale.data = (
            weight_scale.view(k_group // 2, 2, n).permute(0, 2, 1).contiguous()
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_shape = x.shape
        k = original_shape[-1]
        x = x.reshape(-1, k).contiguous()

        x_fp4, x_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x,
            dst_type=torch_npu.float4_e2m1fn_x2,
            round_mode="round",
        )

        out = torch.ops.npu.npu_quant_matmul(
            x_fp4,
            layer.weight,
            scale=layer.weight_scale,
            pertoken_scale=x_scale,
            bias=bias,
            output_dtype=x.dtype,
            group_sizes=(1, 1, 32),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
        )

        return out.reshape(*original_shape[:-1], out.shape[-1])


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
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
            layer.weight.data.to(torch.int32)
        )

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
