"""
Common utilities for quark.
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def quantize_fp8_scale_tensorwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    FP8_MAX = 448.0
    scale = w.abs().amax().float() / FP8_MAX
    scaled = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return scaled, scale


def quantize_int4_scale_columnwise(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    S4_MAX = 7
    w_flat = w.reshape(-1, w.shape[-1]).float()
    scale = w_flat.abs().amax(axis=-1) / S4_MAX
    scaled = torch.round(w_flat / scale[:, None]).to(torch.int8).clamp(-S4_MAX, S4_MAX)
    return scaled.reshape(w.shape), scale.reshape(w.shape[:-1])


def pack_int4_to_int32(to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
    if to_pack.ndim > 2:
        raise ValueError(
            "Pack: Only supports tensors with dimensions not greater than 2."
        )

    if reorder:
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    else:
        order_map = [0, 1, 2, 3, 4, 5, 6, 7]
    pack_num = 8
    if to_pack.ndim == 2:
        packed = torch.zeros(
            to_pack.shape[0],
            to_pack.shape[1] // pack_num,
            dtype=torch.int32,
            device=to_pack.device,
        )
        new_c = to_pack.shape[1] // pack_num
        for c in range(new_c):
            for i in range(pack_num):
                # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                packed_col = to_pack[:, c * pack_num + order_map[i]].to(torch.int32)
                packed_col = packed_col & 0x0F
                packed[:, c] = torch.bitwise_or(
                    packed[:, c], torch.bitwise_left_shift(packed_col, i * 4)
                )
    elif to_pack.ndim == 0:
        packed = to_pack.to(torch.int32)
    else:
        packed = torch.zeros(
            to_pack.shape[0] // pack_num, dtype=torch.int32, device=to_pack.device
        )
        new_c = to_pack.shape[0] // pack_num
        for c in range(new_c):
            for i in range(pack_num):
                # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                packed_col = to_pack[c * pack_num + order_map[i]]
                packed_col = packed_col & 0x0F
                packed[c] = torch.bitwise_or(
                    packed[c], torch.bitwise_left_shift(packed_col, i * 4)
                )

    return packed.view(torch.uint32)
