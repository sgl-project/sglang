"""
Utilities to manage the dequantization of weights.
"""

from typing import Optional

import torch

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
