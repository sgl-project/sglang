import logging
from functools import lru_cache
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
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.utils import is_cuda, is_xpu

__all__ = ["CompressedTensorsW8A8MXFp8"]

_is_xpu = is_xpu()
_is_cuda = is_cuda()
logger = logging.getLogger(__name__)
MXFP8_GROUP_SIZE = 32


def dequant_mxfp8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode the checkpoint's row-major UE8M0 scale for each group of 32."""
    blocks = weight.float().unflatten(-1, (-1, MXFP8_GROUP_SIZE))
    scales = weight_scale.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return (blocks * scales).flatten(-2).to(out_dtype)


@lru_cache(maxsize=None)
def _xpu_mxfp8_supported(device_index: int) -> bool:
    device = torch.device("xpu", device_index)
    weight = torch.zeros((32, 128), dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    scales = torch.full((32, 4), 127, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )
    scaling = torch.nn.functional.ScalingType.BlockWise1x32
    try:
        torch.nn.functional.scaled_mm(
            weight,
            weight.t(),
            scales,
            scaling,
            scales,
            scaling,
            output_dtype=torch.bfloat16,
        )
        torch.xpu.synchronize(device)
    except (RuntimeError, NotImplementedError) as error:
        logger.info("MXFP8 block-scaled GEMM unavailable on %s: %s", device, error)
        return False
    return True


class CompressedTensorsW8A8MXFp8(CompressedTensorsLinearScheme):
    def __init__(self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs):
        self.group_size = MXFP8_GROUP_SIZE
        if not (_is_cuda or _is_xpu):
            raise NotImplementedError(
                "compressed-tensors MXFP8 dense linear requires CUDA or XPU."
            )
        self.use_native = (
            _is_xpu
            and resolve_mxfp8_dense_gemm_backend().is_xpu_scaled_mm()
            and _xpu_mxfp8_supported(torch.xpu.current_device())
        )
        if self.use_native:
            self.w8a8_mxfp8_linear = dispatch_w8a8_mxfp8_linear()
        else:
            logger.info_once(
                "Using BF16 fallback for compressed-tensors MXFP8 dense linear"
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: torch.xpu.get_device_capability
        if _is_xpu:
            return 20
        return 80

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
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self.use_native:
            copy_or_rebind_param(
                layer,
                "dequantized_weight",
                dequant_mxfp8_weight(
                    layer.weight, layer.weight_scale, self.params_dtype
                ),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_native:
            return torch.nn.functional.linear(x, layer.dequantized_weight, bias)
        return self.w8a8_mxfp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
        )
