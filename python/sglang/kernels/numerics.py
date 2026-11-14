# SPDX-License-Identifier: Apache-2.0
"""Numerical primitives shared by bit-exact Triton kernels.

The multiply and reciprocal-square-root primitives are inline asm on purpose:
:func:`mul_rn_f32` is the opaque barrier that keeps LLVM from contracting it
into an FMA with a following add, which is what lets the fused kernels stay
bit-identical to the eager chain.

Triton lowers ``tl.inline_asm_elementwise`` straight into LLVM inline asm, so
the mnemonics must be the target assembler's own syntax. CUDA takes PTX with
``f`` constraints; the AMD backend needs GCN mnemonics with ``v`` (VGPR)
constraints. Handing the PTX form to AMD fails while assembling the kernel::

    error: couldn't allocate output register for constraint 'f'

and that aborts the process instead of raising, so it cannot be caught at the
call site. AMD has one more wrinkle: there is no single instruction for an
IEEE divide (LLVM does not expand the ``div.rn.f32`` pseudo-instruction inside
inline asm), so :func:`div_rn_f32` uses Triton's own fp32 division there, which
lowers to the full ``v_div_scale``/``v_rcp_f32``/``v_div_fixup_f32`` sequence
because ``/`` carries no fast-math flags.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_FLT_MIN = tl.constexpr(1.1754943508222875e-38)

# AMD GPUs reject the PTX constraint set described above (observed on gfx1151).
_AMD = torch.version.hip is not None


@triton.jit
def round_bf16_to_fp32(value):
    """RNE-round fp32 to bf16 precision while keeping an fp32 register."""
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


if _AMD:

    @triton.jit
    def mul_rn_f32(x, y):
        """Opaque correctly-rounded fp32 multiply that blocks FMA contraction."""
        return tl.inline_asm_elementwise(
            asm="v_mul_f32 $0, $1, $2",
            constraints="=v,v,v",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def div_rn_f32(x, y):
        """IEEE correctly-rounded fp32 division."""
        return x / y

    @triton.jit
    def rsqrt_approx_f32(x):
        return tl.inline_asm_elementwise(
            asm="v_rsq_f32 $0, $1",
            constraints="=v,v",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

else:

    @triton.jit
    def mul_rn_f32(x, y):
        """Opaque correctly-rounded fp32 multiply that blocks FMA contraction."""
        return tl.inline_asm_elementwise(
            asm="mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def div_rn_f32(x, y):
        """IEEE correctly-rounded fp32 division."""
        return tl.inline_asm_elementwise(
            asm="div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def rsqrt_approx_f32(x):
        return tl.inline_asm_elementwise(
            asm="rsqrt.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


@triton.jit
def cuda_rsqrtf(x):
    """Match CUDA ``rsqrtf``, including its subnormal rescaling path."""
    is_subnormal = tl.abs(x) < _FLT_MIN
    scaled = tl.where(is_subnormal, x * 16777216.0, x)
    result = rsqrt_approx_f32(scaled)
    return tl.where(is_subnormal, result * 4096.0, result)


__all__ = [
    "cuda_rsqrtf",
    "div_rn_f32",
    "mul_rn_f32",
    "round_bf16_to_fp32",
    "rsqrt_approx_f32",
]
