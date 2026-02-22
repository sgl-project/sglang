from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_IS_ROCM: bool = torch.version.hip is not None


@cache_once
def _jit_activation_module(dtype: torch.dtype, act_type: str) -> Module:
    # act_type: "silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "gelu_quick"
    args = make_cpp_args(dtype)
    kernel_name = f"{act_type}<{args}>"

    return load_jit(
        f"activation_{act_type}",
        *args,
        cuda_files=["elementwise/activation.cuh"],
        cuda_wrappers=[(act_type, f"{kernel_name}::run")],
    )


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    if _IS_ROCM:
        import sgl_kernel

        sgl_kernel.silu_and_mul(out, input)
    else:
        module = _jit_activation_module(input.dtype, "silu_and_mul")
        module.silu_and_mul(out, input)
    return out


def gelu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    if _IS_ROCM:
        import sgl_kernel

        sgl_kernel.gelu_and_mul(out, input)
    else:
        module = _jit_activation_module(input.dtype, "gelu_and_mul")
        module.gelu_and_mul(out, input)
    return out


def gelu_tanh_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    if _IS_ROCM:
        import sgl_kernel

        sgl_kernel.gelu_tanh_and_mul(out, input)
    else:
        module = _jit_activation_module(input.dtype, "gelu_tanh_and_mul")
        module.gelu_tanh_and_mul(out, input)
    return out


def gelu_quick(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    if _IS_ROCM:
        import sgl_kernel

        sgl_kernel.gelu_quick(out, input)
    else:
        module = _jit_activation_module(input.dtype, "gelu_quick")
        module.gelu_quick(out, input)
    return out
