"""SiLU-and-mul for the Torch MPS eager path (Metal JIT)."""

from __future__ import annotations

import functools

import torch

# One thread per output element. silu is computed in fp32 and rounded to T
# before the multiply, matching F.silu(gate) * up in forward_native.
_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

#define DEFINE_SILU_AND_MUL(T, NAME)                                          \\
kernel void silu_and_mul_##NAME(                                              \\
    device const T* x [[buffer(0)]],                                          \\
    device T* out [[buffer(1)]],                                              \\
    constant uint& d [[buffer(2)]],                                           \\
    uint i [[thread_position_in_grid]]                                        \\
) {                                                                           \\
    uint row = i / d;                                                         \\
    uint col = i - row * d;                                                   \\
    device const T* gate = x + row * 2 * d;                                   \\
    float g = float(gate[col]);                                               \\
    T silu = T(g / (1.0f + exp(-g)));                                         \\
    out[i] = T(float(silu) * float(gate[d + col]));                           \\
}

DEFINE_SILU_AND_MUL(float, float)
DEFINE_SILU_AND_MUL(half, half)
DEFINE_SILU_AND_MUL(bfloat, bfloat)
"""

_DTYPE_SUFFIX = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}


@functools.cache
def _library():
    return torch.mps.compile_shader(_METAL_SOURCE)


def silu_and_mul_mps_supported(x: torch.Tensor) -> bool:
    """Whether :func:`silu_and_mul_mps` handles this input."""
    return (
        x.device.type == "mps"
        and x.dtype in _DTYPE_SUFFIX
        and x.numel() > 0
        and x.shape[-1] % 2 == 0
        # Thread indices are 32-bit.
        and x.numel() < 2**32
    )


def silu_and_mul_mps(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` with ``d = x.shape[-1] // 2``.

    Callers check :func:`silu_and_mul_mps_supported`.
    """
    d = x.shape[-1] // 2
    x = x.contiguous()
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    getattr(_library(), f"silu_and_mul_{_DTYPE_SUFFIX[x.dtype]}")(
        x, out, d, threads=out.numel()
    )
    return out
