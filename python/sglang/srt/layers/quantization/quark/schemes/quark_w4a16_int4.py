# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.utils import use_gfx1151_kernels

__all__ = ["QuarkW4A16Int4"]


def _gfx1151_w4a16_gemm(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> Optional[torch.Tensor]:
    """gfx1151 WMMA GEMMs for symmetric group-128 weights: the Gluon small-M GEMM for at
    most 16 rows (decode, speculative verify) and the prefill GEMM from 256 rows; None
    elsewhere, leaving the skinny GEMM."""
    from sglang.kernels.ops.quantization import (
        w4a16_prefill_gfx1151,
        w4a16_small_m_gfx1151,
    )

    if w4a16_small_m_gfx1151.covered(x, weight, scale):
        return w4a16_small_m_gfx1151.w4a16_small_m_gemm(x, weight, scale)
    if w4a16_prefill_gfx1151.covered(x, weight, scale):
        return w4a16_prefill_gfx1151.w4a16_prefill_gemm(x, weight, scale)
    return None


class QuarkW4A16Int4(QuarkLinearScheme):
    """Quark weight-only int4 (AWQ-packed, signed codes, per-group scales) on ROCm,
    through the Triton skinny GEMM (gfx1151 WMMA GEMMs where they apply)."""

    SUPPORTED_GROUP_SIZES = (32, 64, 128)

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.group_size = weight_config.get("group_size")
        self.is_sym = bool(weight_config.get("symmetric", False))
        self.pack_factor = 8
        self.use_gfx1151 = (
            use_gfx1151_kernels() and self.is_sym and self.group_size == 128
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"Input size {input_size_per_partition} is not divisible by "
                f"group size {self.group_size}. This can be caused by too "
                "large a tensor parallel size."
            )
        if output_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                f"Output size {output_size_per_partition} is not divisible by "
                f"the INT4 pack factor {self.pack_factor}. This can be caused "
                "by too large a tensor parallel size."
            )

        layer.logical_widths = output_partition_sizes
        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_zero_point = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_zero_point", weight_zero_point)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.kernels.ops.quantization.w4a16_skinny_triton import (
            repack_awq_qzeros_to_skinny,
            repack_awq_w4_to_skinny,
        )

        # Quark stores signed int4 in AWQ's N-packed, interleaved layout. Convert once to
        # unsigned zero-point-8 codes in the [N, K / 8] layout the skinny GEMM reads.
        weight = repack_awq_w4_to_skinny(layer.weight.data.contiguous(), signed=True)
        if self.is_sym:
            weight_zero_point = torch.empty(
                0, dtype=torch.int32, device=layer.weight_zero_point.device
            )
        else:
            weight_zero_point = repack_awq_qzeros_to_skinny(
                layer.weight_zero_point.data.contiguous(), signed=True
            )
        if self.use_gfx1151:
            from sglang.kernels.ops.quantization.w4a16_small_m_gfx1151 import pad_rows

            weight = pad_rows(weight)
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_zero_point = Parameter(weight_zero_point, requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data.t().contiguous(), requires_grad=False
        )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (layer.weight.shape[0],)
        out = None
        if self.use_gfx1151:
            out = _gfx1151_w4a16_gemm(reshaped_x, layer.weight, layer.weight_scale)
        if out is None:
            from sglang.kernels.ops.quantization.w4a16_skinny_triton import (
                w4a16_skinny_gemm,
            )

            out = w4a16_skinny_gemm(
                input=reshaped_x,
                qweight=layer.weight,
                scales=layer.weight_scale,
                group_size=self.group_size,
                qzeros=None if self.is_sym else layer.weight_zero_point,
            )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
