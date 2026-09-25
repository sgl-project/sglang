# SPDX-License-Identifier: Apache-2.0
"""compressed-tensors WNA16 int4 dense linear for Intel XPU."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.hardware_backend.xpu.quantization.int4pack_utils import (
    SUPPORTED_GROUP_SIZES,
    pack_int4_to_uint8,
    unpack_compressed_tensors_qweight,
    xpu_int4pack_mm,
)
from sglang.srt.layers.quantization.utils import replace_parameter

# Symmetric WNA16 stores each int4 code biased by 8 (``uint4b8``). Marlin folds
# that bias into its scales; the native XPU op takes an explicit zero-point.
_UINT4B8_ZERO_POINT = 8


class CompressedTensorsWNA16XPULinearKernel:
    def __init__(self, group_size: int, symmetric: bool, has_g_idx: bool):
        self.group_size = group_size
        self.symmetric = symmetric
        self.has_g_idx = has_g_idx

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"compressed-tensors WNA16 on XPU requires group_size in "
                f"{SUPPORTED_GROUP_SIZES}, got {self.group_size}. The native "
                "XPU INT4 operator does not support this group size "
                "(channelwise/-1 is out of scope)."
            )
        if not self.symmetric:
            raise ValueError(
                "compressed-tensors WNA16 on XPU only supports symmetric "
                "weight quantization; this checkpoint carries a weight "
                "zero-point."
            )
        if self.has_g_idx:
            raise ValueError(
                "compressed-tensors WNA16 on XPU does not support activation "
                "reordering (actorder=group)."
            )

        weight_packed = layer.weight_packed.data  # [N, K // 8] int32
        scales = layer.weight_scale.data  # [N, K // gs]

        codes = unpack_compressed_tensors_qweight(weight_packed)  # [N, K]
        qweight_uint8 = pack_int4_to_uint8(codes)  # [N, K // 2]
        qweight_packed = torch.ops.aten._convert_weight_to_int4pack(qweight_uint8, 8)

        # The native op wants group-major scales and zero-points: [K // gs, N].
        xpu_scales = scales.t().contiguous()
        zero_points = torch.full(
            xpu_scales.shape,
            _UINT4B8_ZERO_POINT,
            dtype=torch.int8,
            device=xpu_scales.device,
        )

        replace_parameter(layer, "weight_packed", qweight_packed)
        layer.register_parameter(
            "xpu_scales",
            torch.nn.Parameter(xpu_scales, requires_grad=False),
        )
        layer.register_parameter(
            "xpu_zero_points",
            torch.nn.Parameter(zero_points, requires_grad=False),
        )
        del layer.weight_scale

        layer.xpu_out_features = codes.shape[0]
        layer.xpu_group_size = self.group_size

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return xpu_int4pack_mm(
            x=x,
            qweight_packed=layer.weight_packed,
            group_size=layer.xpu_group_size,
            scales=layer.xpu_scales,
            zero_points=layer.xpu_zero_points,
            out_features=layer.xpu_out_features,
            bias=bias,
        )
