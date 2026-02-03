"""
Utilities to manage the dequantization of weights.
"""

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)


def dequantize_fp8(
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    block_size: list[int],
    format_ue8m0: bool = False,
) -> torch.Tensor:
    """
    Dequantizes `w_q` to bfloat16.
    """
    if format_ue8m0:
        # TODO this is only needed for Blackwell
        w_s = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])

    w_dequant = block_quant_dequant(
        w_q,
        w_s,
        block_size=block_size,
        dtype=torch.bfloat16,
    )

    return w_dequant
