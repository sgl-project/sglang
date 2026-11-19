"""Keep the Inkling2 Triton kernels buildable on ROCm: CUDA's programmatic dependent
launch hooks and the PTX approximations exist only on NVIDIA targets."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

IS_HIP = tl.constexpr(torch.version.hip is not None)


def pdl_launch(enabled: bool = True) -> dict:
    # Triton's ROCm launcher rejects the launch_pdl keyword, whatever its value.
    return {} if IS_HIP else {"launch_pdl": enabled}


@triton.jit
def gdc_wait():
    if not IS_HIP:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if not IS_HIP:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def rcp_rn(x):
    """1 / x rounded to nearest: libdevice's on CUDA (subnormals flush to zero),
    which ROCm's libdevice lacks."""
    if IS_HIP:
        return tl.div_rn(1.0, x)
    else:
        return libdevice.rcp_rn(x)


@triton.jit
def approx_f32(x, OP: tl.constexpr):
    """PTX `OP $0, $1` on CUDA; the same function through Triton math on ROCm."""
    if IS_HIP:
        if OP == "ex2.approx.f32" or OP == "ex2.approx.ftz.f32":
            return tl.exp2(x)
        elif OP == "rcp.approx.f32" or OP == "rcp.approx.ftz.f32":
            return 1.0 / x
        elif OP == "rsqrt.approx.f32" or OP == "rsqrt.approx.ftz.f32":
            return tl.math.rsqrt(x)
        else:
            tl.static_assert(OP == "tanh.approx.f32", "unsupported approximation")
            return libdevice.tanh(x)
    else:
        return tl.inline_asm_elementwise(
            OP + " $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
