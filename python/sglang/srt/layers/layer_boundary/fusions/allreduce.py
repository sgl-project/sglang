# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Select fused completion and read kernels for each consumer."""

from functools import partial
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.layers.layer_boundary.contracts import FusedMlpInput
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_QUANT_READ,
    Fp8Input,
    NormQuantRead,
    apply_aiter_all_reduce_fusion,
    apply_flashinfer_allreduce_fusion,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe import post_experts_reduction_group
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import get_bool_env_var, is_hip

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


def attention_fusions(plan, read=NORM_QUANT_READ) -> Tuple[Callable, ...]:
    """The fused kernels that complete what the previous layer left together
    with the residual update and the input norm, in the order they are tried.
    Each takes (owed, residual, forward_batch, post_residual_addition) and
    returns None when it does not take the batch. They add the residual
    plainly; the boundary tries them only when the update it writes in is a
    plain add. A backend's come first."""
    given = plan.fusions.attention_input(plan) if plan.fusions else ()
    if not hasattr(plan.norm, "forward_with_allreduce_fusion"):
        return given
    fuses_quant = (
        isinstance(read, NormQuantRead)
        and read.fp8_input is not None
        and _use_aiter
        and not get_bool_env_var("SGLANG_DISABLE_FUSED_AR_QUANT", default="false")
        and get_exec().comm.enable_aiter_allreduce_fusion
        and hasattr(plan.norm, "forward_with_allreduce_fusion_quant_per_group")
    )
    return (
        *given,
        partial(
            complete_attention_input,
            plan,
            fuses_quant=fuses_quant,
            keep_bf16=fuses_quant and read.fp8_input is Fp8Input.TUPLE_AND_BF16,
        ),
    )


def complete_attention_input(
    plan,
    owed: UnreducedOutput,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    post_residual_addition: Optional[torch.Tensor],
    *,
    fuses_quant: bool,
    keep_bf16: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Complete the sum the previous layer left, add it to the residual and
    apply the input norm in one aiter or flashinfer kernel; None when the
    kernel does not take this batch. The optional FP8 result follows the consumer read declaration;
    the separate ``quant_format`` path remains the read adapter's responsibility."""
    if (
        not isinstance(owed, UnreducedOutput)
        # The kernel does not add it.
        or post_residual_addition is not None
        # The kernel reduces over the MoE output's group.
        or owed.group is not post_experts_reduction_group()
    ):
        return None
    hidden_states = owed.partial
    if not (
        apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
        or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
    ):
        return None
    if fuses_quant:
        # Falls back to AR+RMSNorm + separate quant internally when the
        # fully-fused kernel cannot service the shape.
        quant_result = plan.norm.forward_with_allreduce_fusion_quant_per_group(
            hidden_states,
            residual,
            use_attn_tp_group=False,
            keep_bf16=keep_bf16,
        )
        if quant_result is not None:
            return quant_result
    return plan.norm.forward_with_allreduce_fusion(
        hidden_states, residual, use_attn_tp_group=False
    )


def ffn_fusions(plan) -> Tuple["FusedMlpInput", ...]:
    """The fused kernels that can take the attention -> FFN steps, in the
    order they are tried. They add the residual plainly; the boundary tries
    them only when the update it writes in is a plain add. A backend's come
    first."""
    given = plan.fusions.ffn_input(plan) if plan.fusions else ()
    if not hasattr(plan.norm, "forward_with_allreduce_fusion"):
        return given
    return (
        *given,
        FusedMlpInput(
            completes=SumGroup.ATTN_TP,
            run=partial(complete_ffn_input, plan),
            preserves_residual=(
                flashinfer_preserves_residual
                if isinstance(plan.norm, (RMSNorm, GemmaRMSNorm))
                and type(plan.norm).forward_with_allreduce_fusion
                in (
                    RMSNorm.forward_with_allreduce_fusion,
                    GemmaRMSNorm.forward_with_allreduce_fusion,
                )
                else None
            ),
        ),
    )


def flashinfer_preserves_residual(value, forward_batch):
    """The CUDA wrapper returns a fresh residual, including backend fallback."""
    return (
        not _use_aiter
        and get_parallel().attn_tp_size > 1
        and apply_flashinfer_allreduce_fusion(value.shape[0])
    )


def complete_ffn_input(
    plan,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The attention-TP all-reduce, residual add and norm in one aiter or
    flashinfer kernel; None when neither takes the batch."""
    if not (
        apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
        or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
    ):
        return None
    return plan.norm.forward_with_allreduce_fusion(
        hidden_states, residual, use_attn_tp_group=True
    )
