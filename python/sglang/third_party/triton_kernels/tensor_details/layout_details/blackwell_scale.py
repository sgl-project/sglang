import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .base import Layout

SWIZZLE_ALIGN_INNER = tl.constexpr(8)
SWIZZLE_SIZE_INNER = tl.constexpr(4)
SWIZZLE_SIZE_OUTER = tl.constexpr(128)


@dataclass
class BlackwellMXScaleLayout(Layout):
    B: int
    ALIGN_K: int
    ALIGN_N: int
    SWIZZLE_K: int
    K_pad: int
    N_pad: int
    name: str = "BLACKWELL_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)
        (
            *self.leading_shape,
            self.K,
            self.N,
        ) = shape
        self.B = math.prod(self.leading_shape)
        self.ALIGN_K = 8
        self.ALIGN_N = 128
        self.SWIZZLE_K = 4
        self.K_pad = (self.K + self.ALIGN_K - 1) // self.ALIGN_K * self.ALIGN_K
        self.N_pad = (self.N + self.ALIGN_N - 1) // self.ALIGN_N * self.ALIGN_N

    def swizzle_data(self, data):
        data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_pad - self.K))
        data = data.transpose(-1, -2).contiguous()
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.ALIGN_N // 32, 32, self.K_pad // self.SWIZZLE_K,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4).contiguous()
        data = data.view(1, self.B * self.N_pad // 128, self.K_pad // self.SWIZZLE_K, 2, 256)
        return data

    def unswizzle_data(self, data):
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.K_pad // self.SWIZZLE_K, 32, self.ALIGN_N // 32,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_pad)
        data = data.transpose(-1, -2)
        return data[..., :self.K, :self.N]

    def swizzle_block_shape(self, block_shape):
        assert block_shape[0] >= 128, f"{block_shape[0]=} must be >= 128"
        return [1, block_shape[0] // 128, block_shape[1] // 4, 2, 256]


@triton.jit
def unswizzle_mx_scale_bw(
    x,
    SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
    SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
    ALIGN_INNER: tl.constexpr = SWIZZLE_ALIGN_INNER,
):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    tl.static_assert(shape_1 % SIZE_OUTER == 0)
    tl.static_assert(shape_1 // SIZE_OUTER <= ALIGN_INNER)
    x = x.reshape(shape_0, (shape_1 // SIZE_OUTER) // SIZE_INNER, 32, SIZE_OUTER // 32, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x
