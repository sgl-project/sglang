from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_causal_conv1d_module(dtype: torch.dtype) -> "Module":
    """JIT-compile the causal_conv1d kernel for a specific input dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "causal_conv1d",
        *args,
        cuda_files=["mamba/causal_conv1d_jit.cuh"],
        cuda_wrappers=[
            ("causal_conv1d_fwd", f"CausalConv1dFwdKernel<{args}>::run"),
            ("causal_conv1d_update", f"CausalConv1dUpdateKernel<{args}>::run"),
        ],
    )


@register_custom_op(
    op_name="sglang_jit_causal_conv1d_fwd",
    mutates_args=["x", "conv_states"],
)
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
) -> None:
    """Forward causal 1-D convolution that writes the result back into ``x`` in place."""
    module = _jit_causal_conv1d_module(x.dtype)
    module.causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
    )


@register_custom_op(
    op_name="sglang_jit_causal_conv1d_update",
    mutates_args=["x", "conv_state"],
)
def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
) -> None:
    """Single-step causal 1-D convolution that updates ``conv_state`` and writes into ``x``."""
    module = _jit_causal_conv1d_module(x.dtype)
    module.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
