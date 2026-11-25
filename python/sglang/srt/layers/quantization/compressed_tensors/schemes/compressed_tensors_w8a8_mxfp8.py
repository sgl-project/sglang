from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp8_utils import (
    dispatch_w8a8_mxfp8_linear,
    prepare_mxfp8_linear_weight_scale,
    resolve_mxfp8_dense_gemm_backend,
    resolve_mxfp8_linear_weight,
    resolve_mxfp8_linear_weight_scale,
)
from sglang.srt.utils import is_hip, is_xpu

__all__ = ["CompressedTensorsW8A8MXFp8"]

_is_hip = is_hip()
_is_xpu = is_xpu()


class CompressedTensorsW8A8MXFp8(CompressedTensorsLinearScheme):
    def __init__(self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs):
        self.group_size = 32
        backend = resolve_mxfp8_dense_gemm_backend()
        if (
            backend.is_unsupported()
            or backend.is_gfx95()
            or (not backend.is_xpu_scaled_mm() and _is_hip)
        ):
            raise NotImplementedError(
                "CompressedTensorsW8A8MXFp8 requires a CUDA Blackwell (SM100+) "
                "or Intel XPU MXFP8 dense GEMM backend; "
                f"resolved backend={backend.value}."
            )
        self.mxfp8_dense_backend = backend
        self.w8a8_mxfp8_linear = dispatch_w8a8_mxfp8_linear(backend)

    @classmethod
    def get_min_capability(cls) -> int:
        if _is_xpu:
            return 20
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
        layer.params_dtype = params_dtype

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
            data=torch.zeros(
                output_size_per_partition,
                scale_k,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale.format_ue8m0 = True
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_mxfp8_linear_weight_scale(
            layer,
            self.mxfp8_dense_backend,
            layer.weight_scale.data,
            scale_attr="weight_scale",
            params_dtype=layer.params_dtype,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = resolve_mxfp8_linear_weight(layer, self.mxfp8_dense_backend)
        weight_scale, extra_kwargs = resolve_mxfp8_linear_weight_scale(
            layer,
            self.mxfp8_dense_backend,
            scale_attr="weight_scale",
        )
        return self.w8a8_mxfp8_linear(
            input=x,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=None,
            bias=bias,
            **extra_kwargs,
        )
