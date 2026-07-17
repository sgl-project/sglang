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
def activation_module(dtype: torch.dtype) -> Module:
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
            (
                "run_unary_activation",
                f"ActivationKernel<{args}>::run_unary_activation",
            ),
        ],
    )


SUPPORTED_ACTIVATIONS = {"silu", "gelu", "gelu_tanh"}
SUPPORTED_UNARY_ACTIVATIONS = {"relu2"}


@register_custom_op(mutates_args=["out"])
def _run_activation_inplace(
    op_name: str, input: torch.Tensor, out: torch.Tensor
) -> None:
    hidden_size = input.shape[-1] // 2
    module = activation_module(input.dtype)
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
    module = activation_module(input.dtype)
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


@register_custom_op(mutates_args=["out"])
def _run_unary_activation_inplace(
    op_name: str, input: torch.Tensor, out: torch.Tensor
) -> None:
    last = input.shape[-1]
    module = activation_module(input.dtype)
    module.run_unary_activation(input.view(-1, last), out.view(-1, last), op_name)


def run_unary_activation(
    op_name: str,
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a standalone (non-gated) element-wise activation: ``out = act(input)``.

    Unlike :func:`run_activation`, there is no gate/up split — ``input`` and
    ``out`` share the same shape.
    """
    assert op_name in SUPPORTED_UNARY_ACTIVATIONS, (
        f"Unsupported unary activation: {op_name}"
    )
    if out is None:
        out = torch.empty_like(input)
    _run_unary_activation_inplace(op_name, input, out)
    return out


def relu2(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Squared ReLU: ``out = max(0, input) ** 2`` (element-wise)."""
    return run_unary_activation("relu2", input, out)


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


# =============================================================================
# Fused activation + per-token-group FP8 quantization
# =============================================================================


@cache_once
def _jit_activation_quant_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "fused_act_and_mul_quant",
        *args,
        cuda_files=["elementwise/fused_act_and_mul_quant.cuh"],
        extra_cuda_cflags=_fast_math_flags(),
        cuda_wrappers=[
            (
                "run_activation_quant",
                f"ActivationQuantKernel<{args}>::run_activation_quant",
            ),
            (
                "run_activation_quant_filtered",
                f"ActivationQuantKernel<{args}>::run_activation_quant_filtered",
            ),
        ],
    )


@register_custom_op(mutates_args=["output_q", "output_scale"])
def _run_activation_quant_inplace(
    op_name: str,
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    group_size: int,
    scale_ue8m0: bool,
) -> None:
    hidden_size = input.shape[-1] // 2
    module = _jit_activation_quant_module(input.dtype)
    input_2d = input.view(-1, hidden_size * 2)
    output_q_2d = output_q.view(-1, hidden_size)
    num_groups = hidden_size // group_size
    output_scale_2d = output_scale.view(-1, num_groups)
    module.run_activation_quant(
        input_2d, output_q_2d, output_scale_2d, op_name, group_size, scale_ue8m0
    )


@register_custom_op(mutates_args=["output_q", "output_scale"])
def _run_activation_quant_filtered_inplace(
    op_name: str,
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_step: int,
    group_size: int,
    scale_ue8m0: bool,
) -> None:
    hidden_size = input.shape[-1] // 2
    module = _jit_activation_quant_module(input.dtype)
    input_2d = input.view(-1, hidden_size * 2)
    output_q_2d = output_q.view(-1, hidden_size)
    num_groups = hidden_size // group_size
    output_scale_2d = output_scale.view(-1, num_groups)
    module.run_activation_quant_filtered(
        input_2d,
        output_q_2d,
        output_scale_2d,
        expert_ids,
        expert_step,
        op_name,
        group_size,
        scale_ue8m0,
    )


def run_activation_quant(
    op_name: str,
    input: torch.Tensor,
    output_q: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused activation + per-token-group FP8 quantization.

    Computes ``act(gate) * up`` and immediately quantizes the result to FP8
    with per-group scales, saving one global memory round-trip compared to
    calling :func:`run_activation` followed by a separate quantization step.

    Args:
        op_name: Activation type — one of ``"silu"``, ``"gelu"``, ``"gelu_tanh"``.
        input: Input tensor of shape ``[*, 2 * hidden_dim]`` (gate||up concatenated).
        output_q: Optional pre-allocated output for quantized values,
            shape ``[*, hidden_dim]``, dtype ``torch.float8_e4m3fn``.
        output_scale: Optional pre-allocated output for scales,
            shape ``[*, hidden_dim // group_size]``, dtype ``torch.float32``.
        expert_ids: Optional expert routing ids for MoE filtering.
            Rows with ``expert_ids[token // expert_step] == -1`` are skipped.
        expert_step: Stride for expert_ids lookup (1 for per-token, BLOCK_SIZE_M for TMA).
        group_size: Number of elements per quantization group (default 128).
        scale_ue8m0: If True, round scales to power-of-2 (UE8M0 format for DeepGEMM).

    Returns:
        Tuple of ``(output_q, output_scale)``.
    """
    assert op_name in SUPPORTED_ACTIVATIONS, f"Unsupported activation: {op_name}"
    hidden_size = input.shape[-1] // 2
    assert hidden_size % group_size == 0, (
        f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
    )
    output_shape = input.shape[:-1] + (hidden_size,)
    scale_shape = input.shape[:-1] + (hidden_size // group_size,)
    if output_q is None:
        output_q = torch.empty(
            output_shape, dtype=torch.float8_e4m3fn, device=input.device
        )
    if output_scale is None:
        output_scale = torch.empty(
            scale_shape, dtype=torch.float32, device=input.device
        )
    if expert_ids is None:
        _run_activation_quant_inplace(
            op_name, input, output_q, output_scale, group_size, scale_ue8m0
        )
    else:
        _run_activation_quant_filtered_inplace(
            op_name,
            input,
            output_q,
            output_scale,
            expert_ids,
            expert_step,
            group_size,
            scale_ue8m0,
        )
    return output_q, output_scale


def silu_and_mul_quant(
    input: torch.Tensor,
    output_q: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused silu_and_mul + per-token-group FP8 quantization."""
    return run_activation_quant(
        "silu",
        input,
        output_q,
        output_scale,
        expert_ids,
        expert_step,
        group_size,
        scale_ue8m0,
    )


def gelu_and_mul_quant(
    input: torch.Tensor,
    output_q: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gelu_and_mul + per-token-group FP8 quantization."""
    return run_activation_quant(
        "gelu",
        input,
        output_q,
        output_scale,
        expert_ids,
        expert_step,
        group_size,
        scale_ue8m0,
    )


def gelu_tanh_and_mul_quant(
    input: torch.Tensor,
    output_q: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gelu_tanh_and_mul + per-token-group FP8 quantization."""
    return run_activation_quant(
        "gelu_tanh",
        input,
        output_q,
        output_scale,
        expert_ids,
        expert_step,
        group_size,
        scale_ue8m0,
    )
