# SPDX-License-Identifier: Apache-2.0
"""
CompressedTensorsW4A16Fp4 - Weight-only nvFP4 quantization scheme.

This scheme handles W4A16 (4-bit weight, 16-bit activation) models that use
NVIDIA's nvFP4 (E2M1) format with FP8-E4M3 block scales.

Model config example:
{
    "format": "nvfp4-pack-quantized",
    "weights": {
        "num_bits": 4,
        "strategy": "tensor_group",
        "type": "float",
        "group_size": 16,
        "scale_dtype": "torch.float8_e4m3fn"
    },
    "input_activations": null  # W4A16 = no activation quantization
}
"""
import logging
from collections.abc import Callable
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.modelopt_quant import (
    enable_flashinfer_fp4_gemm,
    fp4_gemm,
    fp4_quantize,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A16Fp4"]


class CompressedTensorsW4A16Fp4(CompressedTensorsScheme):
    """
    W4A16 nvFP4 quantization scheme.

    - Weights: nvFP4 (E2M1, 4-bit) with FP8-E4M3 block scales, group_size=16
    - Activations: BF16/FP16 (dynamically quantized to FP4 during matmul)

    This differs from W4A4 (CompressedTensorsW4A4Fp4) in that:
    - W4A4: Both weights and activations are statically quantized
    - W4A16: Only weights are pre-quantized, activations quantized at runtime
    """

    def __init__(self) -> None:
        self.group_size = 16  # nvFP4 block size

    @classmethod
    def get_min_capability(cls) -> int:
        # Requires SM100 (Blackwell) for nvFP4 hardware support
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Packed FP4 weights (2 values per byte)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global weight scale (per tensor)
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per-block weight scales (FP8-E4M3, one per 16 elements)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # For W4A16, we use a default input scale of 1.0 for dynamic quantization
        # This will be calibrated at runtime
        input_global_scale = PerTensorScaleParameter(
            data=torch.ones(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Use default input scale for W4A16 (dynamic activation quantization)
        # The scale of 1.0 means we quantize activations fresh each forward pass
        device = layer.weight_packed.device
        layer.input_global_scale = Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32),
            requires_grad=False,
        )

        if get_fp4_gemm_runner_backend().is_flashinfer_trtllm():
            # FlashInfer TRTLLM FP4 GEMM requires specific weight layout
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            # Standard CUTLASS layout with swizzled block scales
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(
                layer.weight_packed.data, requires_grad=False
            )

        # Compute alpha for GEMM output scaling
        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply W4A16 nvFP4 quantized linear operation.

        Unlike W4A4, activations are dynamically quantized here since
        input_global_scale=1.0 means we use runtime quantization.
        """
        output_dtype = x.dtype
        w_n, _ = layer.weight_packed.shape
        output_shape = [x.shape[0], w_n]

        # Dynamic quantization of activations to FP4
        x_fp4, x_blockscale = fp4_quantize(x, layer.input_global_scale)

        assert x_fp4.dtype == torch.uint8
        assert layer.weight_packed.dtype == torch.uint8
        assert layer.weight_scale.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        w = layer.weight_packed
        w_blockscale = layer.weight_scale

        if enable_flashinfer_fp4_gemm:
            w = layer.weight_packed.T
            w_blockscale = layer.weight_scale.T

        out = fp4_gemm(
            x_fp4,
            w,
            x_blockscale,
            w_blockscale,
            layer.alpha,
            output_dtype,
            w_n,
        )

        if bias is not None:
            out = out + bias

        return out.view(*output_shape)
