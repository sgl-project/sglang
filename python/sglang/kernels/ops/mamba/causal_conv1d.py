"""JIT depthwise causal conv1d: prefill (``fwd``) and decode (``update``).

In-place, like the AOT ops they replace on CUDA: the output is written back into
``x`` and the conv state is advanced in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_causal_conv1d_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: float16, bfloat16, float32"
        )
    # The AOT wheel ships an SM90 build compiled with `-use_fast_math` and a
    # precise-math build for every other arch; match that split so the SiLU
    # epilogue keeps producing the same bits as the op being replaced.
    arch = get_jit_cuda_arch()
    use_fast_math = (arch.major, arch.minor) == (9, 0)
    math_mode = "fast_math" if use_fast_math else "precise_math"
    args = make_cpp_args(dtype)
    return load_jit(
        "causal_conv1d",
        math_mode,
        *args,
        cuda_files=["mamba/causal_conv1d.cuh"],
        cuda_wrappers=[
            ("causal_conv1d_fwd", f"causal_conv1d_fwd<{args}>"),
            ("causal_conv1d_update", f"causal_conv1d_update<{args}>"),
        ],
        extra_cuda_cflags=["--use_fast_math"] if use_fast_math else [],
    )


@register_custom_op(
    op_name="mamba_causal_conv1d_fwd",
    mutates_args=["x", "conv_states"],
)
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
) -> None:
    """Causal depthwise conv1d forward (prefill), written back into ``x``."""
    module = _jit_causal_conv1d_module(x.dtype)
    module.causal_conv1d_fwd(
        x,
        weight,
        bias_,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
    )


@register_custom_op(
    op_name="mamba_causal_conv1d_update",
    mutates_args=["x", "conv_state"],
)
def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
) -> None:
    """Causal depthwise conv1d update (decode), written back into ``x``."""
    module = _jit_causal_conv1d_module(x.dtype)
    module.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias_,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
