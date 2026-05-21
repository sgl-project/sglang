from typing import TYPE_CHECKING, Optional

import torch
import torch_npu
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

MXFP8_BLOCK_SIZE = 32
_FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))


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


class NPUMXFP8LinearMethod(_NPULinearMethodBase):
    """Ascend NPU MXFP8 linear method for LLM (SRT) models.

    Online mode: loads FP16/BF16 weights → quantises to MXFP8 at load time.
    Inference: dynamic MXFP8 activation quant + MXFP8 matmul (block_size=32).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise later in process_weights_after_loading
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)

        # Move weight to NPU if needed (cpu offload may have moved it back to CPU)
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Online MXFP8 quantisation of weights (block_size=32)
        qw, w_scale = torch_npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=torch_npu.float8_e4m3fn
        )
        # Pre-transpose to [in, out] for npu_quant_matmul (avoid per-call transpose)
        layer.weight = Parameter(qw.transpose(0, 1).contiguous(), requires_grad=False)
        layer.weight_scale_inv = Parameter(w_scale.transpose(0, 1).contiguous(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch_npu.float8_e4m3fn
        )

        # MXFP8 matmul (weight & scale already transposed at load time)
        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale_inv,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


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
