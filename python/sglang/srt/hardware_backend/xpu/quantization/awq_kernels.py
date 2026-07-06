# SPDX-License-Identifier: Apache-2.0
"""AWQ int4 dense linear for Intel XPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.xpu.quantization.int4pack_utils import (
    AWQ_REVERSE_PACK_ORDER,
    SUPPORTED_GROUP_SIZES,
    build_qscale_and_zeros,
    pack_int4_to_uint8,
    xpu_int4pack_mm,
)
from sglang.srt.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


def _unpack_awq_to_codes(packed: torch.Tensor, rows: int) -> torch.Tensor:
    """Deinterleave AWQ-packed int32 ``[rows, cols]`` into codes ``[rows, cols*8]``.

    Codes are in ``[0, 15]`` and restored to natural (non-interleaved) order.
    """
    t = packed.view(torch.uint8)  # [rows, cols * 4]
    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=t.device)
    t = (t[:, :, None] >> shifter) & 0xF  # [rows, cols * 4, 2]
    t = t.view(-1, 8)[:, AWQ_REVERSE_PACK_ORDER]  # undo interleave
    return t.reshape(rows, -1)  # [rows, cols * 8]


class AWQXPULinearKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        group_size = self.quant_config.group_size
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"AWQ on XPU requires group_size in {SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}. torch _weight_int4pack_mm_xpu does not "
                "support this group size (per-channel/-1 is out of scope)."
            )

        qweight = layer.qweight.data  # [K, N // 8] int32
        qzeros = layer.qzeros.data  # [K // gs, N // 8] int32
        scales = layer.scales.data  # [K // gs, N]

        k = qweight.shape[0]
        n = scales.shape[1]

        # qweight -> [N, K // 2] uint8 (torch int4pack B layout)
        codes = _unpack_awq_to_codes(qweight, k)  # [K, N]
        codes = codes.t().contiguous()  # [N, K]
        qweight_packed = pack_int4_to_uint8(codes)  # [N, K // 2]

        # qzeros -> [K // gs, N] codes, then fold into float zero.
        zp = _unpack_awq_to_codes(qzeros, scales.shape[0])  # [K // gs, N]
        qscale_and_zeros = build_qscale_and_zeros(scales, zp)  # [K // gs, N, 2]

        replace_parameter(layer, "qweight", qweight_packed)
        layer.register_parameter(
            "qscale_and_zeros",
            torch.nn.Parameter(qscale_and_zeros, requires_grad=False),
        )
        # scales/qzeros are now folded into qscale_and_zeros; drop them so they
        # don't linger in state_dict / waste memory.
        del layer.qzeros
        del layer.scales

        layer.xpu_out_features = n
        layer.xpu_group_size = group_size

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return xpu_int4pack_mm(
            x,
            layer.qweight,
            layer.xpu_group_size,
            layer.qscale_and_zeros,
            layer.xpu_out_features,
            bias,
        )
