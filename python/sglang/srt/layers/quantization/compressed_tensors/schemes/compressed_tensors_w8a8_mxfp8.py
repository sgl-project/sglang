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
    resolve_mxfp8_dense_gemm_backend,
)
from sglang.srt.utils import is_xpu

__all__ = ["CompressedTensorsW8A8MXFp8"]

_is_xpu = is_xpu()


class CompressedTensorsW8A8MXFp8(CompressedTensorsLinearScheme):
    def __init__(self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs):
        self.group_size = 32
        self.mxfp8_dense_backend = resolve_mxfp8_dense_gemm_backend()
        # Only the XPU kernel reads the checkpoint scale as-is; the CUDA / ROCm kernels
        # need a load-time scale re-layout this scheme does not perform.
        if not (
            self.mxfp8_dense_backend.is_xpu_scaled_mm()
            or self.mxfp8_dense_backend.is_unsupported()
        ):
            raise NotImplementedError(
                "compressed-tensors MXFP8 W8A8 is only implemented on XPU, got MXFP8 "
                f"backend {self.mxfp8_dense_backend.value}."
            )
        self.w8a8_mxfp8_linear = dispatch_w8a8_mxfp8_linear()

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: torch.xpu.get_device_capability
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
        # Checkpoint layout is kept as-is: weight [N, K] e4m3, weight_scale
        # [N, K // 32] ue8m0 bytes, which xpu_mxfp8_scaled_mm_linear reads directly.
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.w8a8_mxfp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
        )
