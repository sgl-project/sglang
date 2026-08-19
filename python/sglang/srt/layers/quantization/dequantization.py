"""
Utilities to manage the dequantization of weights.
"""

from typing import Optional

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.utils import set_weight_attrs

NVFP4_BLOCK_SIZE = 16
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


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


def dequantize_nvfp4(
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_s2: Optional[torch.Tensor],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """NVFP4 -> ``out_dtype``. ``w_q``: uint8 [..., out, in/2] packed e2m1
    (low nibble = even idx). ``w_s``: fp8 e4m3 [..., out, in/16] per-block.
    ``w_s2``: optional fp32 per-tensor scalar that multiplies the per-block
    scale (ModelOpt / AMD Quark NVFP4)."""
    device = w_q.device
    *batch, out_dim, half_in = w_q.shape
    in_dim = half_in * 2

    low = (w_q & 0xF).to(torch.int64)
    high = (w_q >> 4).to(torch.int64)
    lut = _FP4_E2M1_LUT.to(device=device, dtype=torch.float32)
    deq = torch.empty(*batch, out_dim, in_dim, dtype=torch.float32, device=device)
    deq[..., 0::2] = lut[low]
    deq[..., 1::2] = lut[high]

    scale = w_s.to(torch.float32)
    if w_s2 is not None:
        scale = scale * w_s2.to(torch.float32)
    scale = scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1)
    return (deq * scale).to(out_dtype)
