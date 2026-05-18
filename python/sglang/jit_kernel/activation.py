from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _fast_math_flags() -> list[str]:
    # Mirrors sgl-kernel's CMake policy: fast-math on SM90, precise on
    # SM100+ (Blackwell needs bit-exact expf), off on HIP (clang rejects).
    if is_hip_runtime():
        return []
    if get_jit_cuda_arch().major >= 10:
        return []
    return ["--use_fast_math"]


@cache_once
def _jit_activation_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "activation",
        *args,
        cuda_files=["elementwise/activation.cuh"],
        extra_cuda_cflags=_fast_math_flags(),
        cuda_wrappers=[
            ("run_activation", f"ActivationKernel<{args}>::run_activation"),
            (
                "run_activation_filtered",
                f"ActivationKernel<{args}>::run_activation_filtered",
            ),
        ],
    )


SUPPORTED_ACTIVATIONS = {"silu", "gelu", "gelu_tanh"}


@register_custom_op(mutates_args=["out"])
def _run_activation_inplace(
    op_name: str, input: torch.Tensor, out: torch.Tensor
) -> None:
    hidden_size = input.shape[-1] // 2
    module = _jit_activation_module(input.dtype)
    input_2d = input.view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)
    module.run_activation(input_2d, out_2d, op_name)


@register_custom_op(mutates_args=["out"])
def _run_activation_filtered_inplace(
    op_name: str,
    input: torch.Tensor,
    out: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_step: int,
) -> None:
    hidden_size = input.shape[-1] // 2
    module = _jit_activation_module(input.dtype)
    input_2d = input.view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)
    module.run_activation_filtered(input_2d, out_2d, expert_ids, expert_step, op_name)


def run_activation(
    op_name: str,
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    """Apply ``op_name`` activation followed by element-wise multiplication.

    When ``expert_ids`` is provided, output rows are skipped for tokens whose
    routed expert id is ``-1``. ``expert_step`` is 1 for per-token routing and
    ``BLOCK_SIZE_M`` for sorted/TMA routing — i.e. ``expert_ids[token_id //
    expert_step]`` is consulted before computing each row.
    """
    assert op_name in SUPPORTED_ACTIVATIONS, f"Unsupported activation: {op_name}"
    hidden_size = input.shape[-1] // 2
    if out is None:
        out = input.new_empty(*input.shape[:-1], hidden_size)
    if expert_ids is None:
        _run_activation_inplace(op_name, input, out)
    else:
        _run_activation_filtered_inplace(op_name, input, out, expert_ids, expert_step)
    return out


def silu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    return run_activation("silu", input, out, expert_ids, expert_step)


def gelu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    return run_activation("gelu", input, out, expert_ids, expert_step)


def gelu_tanh_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    return run_activation("gelu_tanh", input, out, expert_ids, expert_step)
