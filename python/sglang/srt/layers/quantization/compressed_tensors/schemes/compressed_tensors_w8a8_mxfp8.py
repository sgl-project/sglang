from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from sgl_kernel import sgl_per_token_group_quant_8bit
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.utils import is_xpu

__all__ = ["CompressedTensorsW8A8MXFp8"]

_is_xpu = is_xpu()

MXFP8_GROUP_SIZE = 32


class CompressedTensorsW8A8MXFp8(CompressedTensorsLinearScheme):
    def __init__(self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs):
        global MXFP8_GROUP_SIZE
        self.group_size = weight_quant.group_size or MXFP8_GROUP_SIZE
        assert self.group_size == MXFP8_GROUP_SIZE, (
            f"MXFP8 requires group_size=={MXFP8_GROUP_SIZE}, got {self.group_size}"
        )
        assert input_quant is None or input_quant.dynamic, (
            "MXFP8 W8A8 requires dynamically quantized activations"
        )

    @classmethod
    def get_min_capability(cls) -> int:
        if _is_xpu:
            return 20
        # blackwell and up
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        assert input_size_per_partition % self.group_size == 0, (
            f"K={input_size_per_partition} must be a multiple of "
            f"group_size={self.group_size} for MXFP8"
        )
        scale_k = input_size_per_partition // self.group_size

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                scale_k,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_scale_e8m0 = layer.weight_scale.data.contiguous().view(
            torch.float8_e8m0fnu
        )
        layer.weight_scale = Parameter(weight_scale_e8m0, requires_grad=False)
        layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        # layer.weight = Parameter(layer.weight.t().contiguous(), requires_grad=False)

    def _quantize_activation(self, x_2d):
        _FUSED_QUANT_DTYPES = (torch.bfloat16, torch.float16)
        M, K = x_2d.shape
        scale_k = K // self.group_size
        if x_2d.dtype not in _FUSED_QUANT_DTYPES:
            x_2d = x_2d.to(torch.bfloat16)

        fp8_info = torch.finfo(torch.float8_e4m3fn)
        xq = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x_2d.device)

        x_scale = torch.empty(
            M, scale_k, dtype=torch.float8_e8m0fnu, device=x_2d.device
        )

        sgl_per_token_group_quant_8bit(
            input=x_2d,
            output_q=xq,
            output_s=x_scale,
            group_size=self.group_size,
            eps=1e-10,
            fp8_min=fp8_info.min,
            fp8_max=fp8_info.max,
            scale_ue8m0=True,
            enable_v2=True,
        )
        # check x_scale data type
        return xq, x_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()

        xq, x_scale = self._quantize_activation(x_2d)

        output = torch.nn.functional.scaled_mm(
            xq,
            layer.weight,
            x_scale,
            torch.nn.functional.ScalingType.BlockWise1x32,
            layer.weight_scale,
            torch.nn.functional.ScalingType.BlockWise1x32,
            bias=bias,
            output_dtype=orig_dtype,
        )
        return output.reshape(*orig_shape[:-1], output.shape[-1])
