from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _make_name(*args):
    return "kimi_k3_" + "_".join(str(a) for a in args)


def _fast_math_flags() -> list[str]:
    # Mirrors sgl-kernel's CMake policy: fast-math on SM90, precise on
    # SM100+ (Blackwell needs bit-exact expf), off on HIP (clang rejects).
    if is_hip_runtime():
        return []
    if get_jit_cuda_arch().major >= 10:
        return []
    return ["--use_fast_math"]


@cache_once
def _jit_situ_and_mul_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT SiTU-and-mul module for a given dtype."""
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        _make_name("situ_and_mul"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulKernel<{args}>::run")],
        extra_cuda_cflags=_fast_math_flags(),
    )


def situ_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    """Fused SiTU (SoftCap-GLU) activation: bf16 -> bf16.

    gate_out = beta * tanh(gate / beta) * sigmoid(gate)
    up_out   = linear_beta * tanh(up / linear_beta)  [if linear_beta is not None]
    output   = gate_out * up_out

    Parameters
    ----------
    input       : bf16 CUDA tensor [*, 2*D]
    out         : optional pre-allocated bf16 CUDA tensor [*, D]
    beta        : gate softcap scalar (e.g. 4.0)
    linear_beta : up softcap scalar (e.g. 25.0), or None to skip
    """
    hidden_size = input.shape[-1] // 2
    if out is None:
        out = input.new_empty(*input.shape[:-1], hidden_size)

    # 2D inputs may be row-strided (e.g. a slice of a fused-GEMM output);
    # higher-rank inputs keep the dense-view path.
    if input.dim() == 2 and input.stride(1) == 1:
        input_2d = input
    else:
        input_2d = input.contiguous().view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)

    has_linear_beta = linear_beta is not None
    module = _jit_situ_and_mul_module(input.dtype)
    module.run(
        input_2d,
        out_2d,
        float(beta),
        float(linear_beta) if has_linear_beta else 0.0,
        has_linear_beta,
    )
    return out
