# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import logging
from typing import Any, Callable, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.utils import use_gfx115_kernels

__all__ = ["QuarkW4A16Int4"]

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _gluon_small_m_available() -> bool:
    """The Gluon small-M GEMM needs Triton >= 3.7, whose ``AMDWMMALayout`` takes register
    and warp bases; with an older Triton, or no Gluon, the Triton GEMMs run instead."""
    try:
        from triton.experimental.gluon.language.amd import AMDWMMALayout

        if "reg_bases" not in inspect.signature(AMDWMMALayout).parameters:
            raise ImportError("AMDWMMALayout takes no reg_bases (Triton < 3.7)")
        from sglang.kernels.ops.quantization import w4a16_small_m_gfx1151  # noqa: F401
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        logger.warning(
            "gfx115x Gluon W4A16 GEMM unavailable, using Triton GEMMs: %s", e
        )
        return False
    return True


def _gfx115_w4a16_gemm(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> Optional[torch.Tensor]:
    """gfx115x WMMA GEMMs for symmetric group-128 weights: the Gluon small-M GEMM for at
    most 16 rows (decode, speculative verify) and the prefill GEMM from 256 rows; None
    elsewhere, leaving the skinny GEMM."""
    from sglang.kernels.ops.quantization import w4a16_prefill_gfx1151

    if _gluon_small_m_available():
        from sglang.kernels.ops.quantization import w4a16_small_m_gfx1151

        if w4a16_small_m_gfx1151.covered(x, weight, scale):
            return w4a16_small_m_gfx1151.w4a16_small_m_gemm(x, weight, scale)
    if w4a16_prefill_gfx1151.covered(x, weight, scale):
        return w4a16_prefill_gfx1151.w4a16_prefill_gemm(x, weight, scale)
    return None


class QuarkW4A16Int4(QuarkLinearScheme):
    """Quark weight-only int4 (AWQ-packed, signed codes, per-group scales) on ROCm,
    through the Triton skinny GEMM (gfx115x WMMA GEMMs where they apply)."""

    SUPPORTED_GROUP_SIZES = (32, 64, 128)

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.group_size = weight_config.get("group_size")
        self.pack_factor = 8
        self.use_gfx115 = use_gfx115_kernels() and self.group_size == 128
        if self.use_gfx115:
            _gluon_small_m_available()  # decide (and warn) at load, not in the first forward

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
            pad_rows,
            repack_awq_w4_to_skinny,
        )

        # Quark stores signed int4 in AWQ's N-packed, interleaved layout. Convert once to
        # unsigned zero-point-8 codes in the [N, K / 8] layout the skinny GEMM reads.
        weight = repack_awq_w4_to_skinny(layer.weight.data.contiguous(), signed=True)
        if self.use_gfx115:
            weight = pad_rows(weight)
        layer.weight = Parameter(weight, requires_grad=False)
        # symmetric: the checkpoint's zero points are all zero in signed codes
        layer.weight_zero_point = Parameter(
            torch.empty(0, dtype=torch.int32, device=weight.device),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            layer.weight_scale.data.t().contiguous(), requires_grad=False
        )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (layer.weight.shape[0],)
        out = None
        if self.use_gfx115:
            out = _gfx115_w4a16_gemm(reshaped_x, layer.weight, layer.weight_scale)
        if out is None:
            from sglang.kernels.ops.quantization.w4a16_skinny_triton import (
                w4a16_skinny_gemm,
            )

            out = w4a16_skinny_gemm(
                input=reshaped_x,
                qweight=layer.weight,
                scales=layer.weight_scale,
                group_size=self.group_size,
            )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
