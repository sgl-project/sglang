# Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1rc2/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum
from typing import Optional

import torch

from sglang.srt.utils import direct_register_custom_op, get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1


def rocm_aiter_asm_moe_tkw1_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    fc1_scale: Optional[torch.Tensor] = None,
    fc2_scale: Optional[torch.Tensor] = None,
    fc1_smooth_scale: Optional[torch.Tensor] = None,
    fc2_smooth_scale: Optional[torch.Tensor] = None,
    a16: bool = False,
    per_tensor_quant_scale: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    activation_method: int = ActivationMethod.SILU.value,
) -> torch.Tensor:

    from aiter import ActivationType
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1

    activation = ActivationType(activation_method)

    return asm_moe_tkw1(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        fc1_scale=fc1_scale,
        fc2_scale=fc2_scale,
        fc1_smooth_scale=fc1_smooth_scale,
        fc2_smooth_scale=fc2_smooth_scale,
        a16=a16,
        per_tensor_quant_scale=per_tensor_quant_scale,
        expert_mask=expert_mask,
        activation=activation,
    )


def rocm_aiter_asm_moe_tkw1_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    fc1_scale: Optional[torch.Tensor] = None,
    fc2_scale: Optional[torch.Tensor] = None,
    fc1_smooth_scale: Optional[torch.Tensor] = None,
    fc2_smooth_scale: Optional[torch.Tensor] = None,
    a16: bool = False,
    per_tensor_quant_scale: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    activation_method: int = ActivationMethod.SILU.value,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


if _use_aiter:

    direct_register_custom_op(
        op_name="rocm_aiter_asm_moe_tkw1",
        op_func=rocm_aiter_asm_moe_tkw1_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_asm_moe_tkw1_fake,
    )


def rocm_fused_experts_tkw1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    activation_method = (
        ActivationMethod.SILU if activation == "silu" else ActivationMethod.GELU
    )
    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # w8a8 per-channel quantization
    if per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8:
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert (
            topk_weights.dim() == 2
        ), "`topk_weights` should be in shape (num_tokens, topk)"
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when" " `apply_router_weight_on_input` is True"
        )

        return torch.ops.sglang.rocm_aiter_asm_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale=w1_scale,
            fc2_scale=w2_scale,
            fc1_smooth_scale=None,
            fc2_smooth_scale=None,
            a16=False,
            per_tensor_quant_scale=None,
            expert_mask=None,
            activation_method=activation_method,
        )
    else:
        assert False, "This should not be called."
