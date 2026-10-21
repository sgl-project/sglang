from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_topk_softmax_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_topk_softmax",
        *args,
        cuda_files=["moe/moe_topk_softmax.cuh"],
        cuda_wrappers=[("topk_softmax", f"topk_softmax<{args}>")],
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="moe_topk_softmax_out",
    mutates_args=["topk_weights", "topk_ids", "workspace"],
)
def moe_topk_softmax_out(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    workspace: torch.Tensor,
    renormalize: bool,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor],
) -> None:
    """
    Fused softmax top-k MoE gate (destination-passing style).

    Args:
        topk_weights:  [num_tokens, topk], float32, pre-allocated output
        topk_ids:      [num_tokens, topk], int32,   pre-allocated output
        gating_output: [num_tokens, num_experts], fp32/fp16/bf16
        workspace:     [num_tokens * num_experts] float32 scratch (may be size 1
                       when num_experts is a supported power-of-2 <= 256)
        renormalize:   whether to renormalize weights to sum to 1 per row
        moe_softcapping: tanh softcapping value applied to the logits, 0 disables
        correction_bias: [num_experts] float32 per-expert bias, or None
    """
    module = _jit_moe_topk_softmax_module(gating_output.dtype)
    module.topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        workspace,
        renormalize,
        moe_softcapping,
        correction_bias,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    Fused softmax top-k MoE gate with the same call signature as
    ``sgl_kernel.topk_softmax`` (destination-passing, in-place).

    Args:
        topk_weights:  [num_tokens, topk] float32, written in-place
        topk_ids:      [num_tokens, topk] int32,   written in-place
        gating_output: [num_tokens, num_experts] fp32/fp16/bf16
        renormalize:   whether to renormalize weights to sum to 1 per row
        moe_softcapping: tanh softcapping value applied to the logits, 0 disables
        correction_bias: [num_experts] float32 per-expert bias, or None
    """
    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]

    # The warp-specialized fast path covers power-of-two expert counts up to 512
    # and needs no scratch; everything else goes through the two-pass
    # softmax + top-k path, which writes probabilities to the workspace first.
    # This threshold must stay in sync with the `case 512:` in the .cuh
    # dispatcher and with the AOT host entry in
    # kernels/aot/csrc/moe/moe_topk_softmax_kernels.cu.
    is_pow2 = num_experts != 0 and (num_experts & (num_experts - 1)) == 0
    needs_workspace = not is_pow2 or num_experts > 512
    workspace_size = num_tokens * num_experts if needs_workspace else 1
    workspace = torch.empty(
        workspace_size, dtype=torch.float32, device=gating_output.device
    )

    moe_topk_softmax_out(
        topk_weights,
        topk_ids,
        gating_output,
        workspace,
        renormalize,
        moe_softcapping,
        correction_bias,
    )
