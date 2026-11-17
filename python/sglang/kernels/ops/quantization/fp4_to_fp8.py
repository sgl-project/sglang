"""fp4 index vectors (four ue8m0 blocks of 32 e2m1 values) as e4m3 with one
power-of-two scale per vector, for kernels that take fp8 operands."""

from __future__ import annotations

import functools
from typing import Tuple

import torch

# A block 19 or more binades below its vector's largest rounds to zero in e4m3,
# so the table stops there.
_FP4_TO_FP8_SHIFTS = 20


@functools.cache
def _fp4_byte_to_fp8_pair(device: torch.device) -> torch.Tensor:
    """[shift * 256 + byte] int16: the two e4m3 codes of a packed fp4 byte, low
    nibble first, scaled by 2**(6 - shift)."""
    from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

    byte = torch.arange(256)
    pair = DSV4_DEQUANT_FP4_TABLE[torch.stack([byte & 15, byte >> 4], dim=-1)]
    scale = torch.exp2(6.0 - torch.arange(_FP4_TO_FP8_SHIFTS, dtype=torch.float32))
    e4m3 = (scale[:, None, None] * pair).to(torch.float8_e4m3fn)
    return e4m3.view(torch.int16).flatten().to(device)


def widen_fp4_to_fp8(
    payload: torch.Tensor, scales: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp4 vectors (``payload`` [..., 64] int8, ``scales`` [...] int32 of four packed
    ue8m0 block exponents) as e4m3 [..., 128] times an fp32 [...] scale. Exact while
    every block is within 14 binades of the vector's largest: e4m3 has 2**6
    headroom over fp4's 6 and subnormals down to 2**-9."""
    exponents = scales.unsqueeze(-1).view(torch.uint8).int()
    top = exponents.amax(-1, keepdim=True)
    shifts = (top - exponents).clamp_max(_FP4_TO_FP8_SHIFTS - 1)
    # The table entry of each byte: its block's shift, then the byte itself.
    index = payload.view(torch.uint8).unflatten(-1, (4, 16)) + shifts[..., None] * 256
    pairs = _fp4_byte_to_fp8_pair(payload.device).index_select(0, index.flatten())
    fp8 = pairs.view(payload.shape).view(torch.float8_e4m3fn)
    # 2**(top - 127 - 6): subnormal for the smallest exponents, where shifting the
    # exponent into the fp32 bits would wrap.
    top = top.squeeze(-1)
    return fp8, torch.ldexp(torch.ones_like(top, dtype=torch.float32), top - 133)
