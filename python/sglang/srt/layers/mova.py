# Copyright 2023-2026 SGLang Team
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
"""Inference primitives for mixture-of-value attention (MoVA)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.utils import set_weight_attrs
from sglang.srt.utils.custom_op import register_custom_op

_ROUTED_LINEAR_CHUNK_SIZE = 64 * 1024


def _prepare_mova_moe_config(config: dict) -> dict:
    """Copy a generic MoE config and remove options MoVA cannot consume."""

    config = dict(config)
    # The ordinary two-GEMM MoE runner handles USE_TMA separately and removes
    # it before launching the Triton kernel. MoVA reuses only the first GEMM
    # and does not construct TMA descriptors, so forwarding this tuning-only
    # key as a kernel constexpr would fail at launch.
    config.pop("USE_TMA", None)
    return config


def mova_router_topk(
    router_logits: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    *,
    score_func: str,
    top_k: int,
    scaling_factor: float,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply xLLM's selection-only router-bias semantics.

    Scores are computed in fp32. ``router_bias`` changes which value experts
    are selected, but the mixture coefficients are gathered from the unbiased
    scores. Scaling happens after optional top-k renormalization.
    """

    if top_k <= 0 or top_k > router_logits.shape[-1]:
        raise ValueError(
            f"top_k must be in [1, {router_logits.shape[-1]}], got {top_k}"
        )
    if router_logits.is_cuda and (score_func == "sigmoid" or router_bias is None):
        # Reuse SGLang's fused sigmoid/softmax top-k kernels. They implement
        # the same selection-only correction-bias contract and return fp32
        # mixture weights plus int32 expert ids.
        from sglang.srt.layers.moe.topk import fused_topk

        weights, selected = fused_topk(
            hidden_states=router_logits,
            gating_output=router_logits,
            topk=top_k,
            # Native xLLM leaves a top-1 route at its raw probability.
            renormalize=renormalize and top_k > 1,
            correction_bias=router_bias,
            scoring_func=score_func,
        )
        return (weights * scaling_factor).to(router_logits.dtype), selected
    if score_func == "sigmoid":
        scores = torch.sigmoid(router_logits.float())
    elif score_func == "softmax":
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported MoVA router score function: {score_func}")

    selection_scores = scores
    if router_bias is not None:
        selection_scores = selection_scores + router_bias.to(selection_scores)

    selected = torch.topk(selection_scores, top_k, dim=-1).indices
    weights = torch.gather(scores, dim=-1, index=selected)
    if renormalize and top_k > 1:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * scaling_factor
    return weights.to(router_logits.dtype), selected.to(torch.int32)


def routed_linear_reference(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    """Straightforward MoVA value projection used as the correctness oracle."""

    if hidden_states.ndim != 2:
        raise ValueError("MoVA routed linear expects [tokens, hidden] inputs")
    if expert_weights.ndim != 3:
        raise ValueError("MoVA expert weights must be [experts, output, hidden]")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError("MoVA routing weights and expert ids must have equal shape")
    if hidden_states.shape[0] != selected_experts.shape[0]:
        raise ValueError("MoVA route count must match the token count")
    if hidden_states.shape[1] != expert_weights.shape[2]:
        raise ValueError("MoVA input and expert hidden dimensions differ")
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, expert_weights.shape[1]))

    # This deliberately favors clarity over memory use. Production CUDA paths
    # use ``routed_linear`` below and never materialize selected expert weights.
    selected_weights = expert_weights[selected_experts.to(torch.long)]
    projected = torch.einsum("mknh,mh->mkn", selected_weights, hidden_states)
    projected = F.silu(projected)
    return (projected * routing_weights.to(projected).unsqueeze(-1)).sum(dim=1)


def _fake_mova_routed_linear_cuda(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    return hidden_states.new_empty((hidden_states.shape[0], expert_weights.shape[1]))


@register_custom_op(
    op_name="mova_routed_linear_cuda",
    fake_impl=_fake_mova_routed_linear_cuda,
)
def _routed_linear_cuda_chunk(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    # Keep these imports local: CPU model inspection and mapping tests should
    # not initialize Triton or require the CUDA extension.
    import triton.language as tl

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
    from sglang.srt.layers.moe.fused_moe_triton import (
        moe_align_block_size,
        try_get_optimal_moe_config,
    )

    num_tokens = hidden_states.shape[0]
    top_k = selected_experts.shape[1]
    num_experts, output_size, input_size = expert_weights.shape
    # ``try_get_optimal_moe_config`` uses the last dimension of its synthetic
    # second-GEMM shape as N. MoVA has no second GEMM, so describe the desired
    # routed projection output explicitly.
    config = _prepare_mova_moe_config(
        try_get_optimal_moe_config(
            expert_weights.shape,
            (num_experts, input_size, output_size),
            top_k,
            None,
            num_tokens,
        )
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        selected_experts, config["BLOCK_SIZE_M"], num_experts
    )
    projected = torch.empty(
        (num_tokens * top_k, output_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    invoke_fused_moe_kernel(
        hidden_states,
        expert_weights,
        None,
        projected,
        None,
        None,
        None,
        routing_weights,
        selected_experts,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # Routing weights are applied after SiLU.
        top_k,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        filter_expert=False,
    )
    projected = F.silu(projected.view(num_tokens, top_k, output_size))
    return (projected * routing_weights.to(projected).unsqueeze(-1)).sum(dim=1)


def routed_linear(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    """Run routed value projections using SGLang's fused-MoE first GEMM."""

    if not hidden_states.is_cuda:
        return routed_linear_reference(
            hidden_states, expert_weights, routing_weights, selected_experts
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Fused MoVA routed linear supports fp16 and bf16 only")
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, expert_weights.shape[1]))
    if not hidden_states.is_contiguous() or not expert_weights.is_contiguous():
        raise ValueError("Fused MoVA inputs and expert weights must be contiguous")

    outputs = []
    for begin in range(0, hidden_states.shape[0], _ROUTED_LINEAR_CHUNK_SIZE):
        end = min(begin + _ROUTED_LINEAR_CHUNK_SIZE, hidden_states.shape[0])
        outputs.append(
            _routed_linear_cuda_chunk(
                hidden_states[begin:end],
                expert_weights,
                routing_weights[begin:end],
                selected_experts[begin:end],
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class RoutedValueExperts(nn.Module):
    """Persistent output-sharded MoVA value-expert weights."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__()
        if output_size % tp_size:
            raise ValueError(
                f"MoVA value width {output_size} is not divisible by TP={tp_size}"
            )
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = output_size // tp_size
        self.tp_rank = tp_rank
        self.weight = nn.Parameter(
            torch.empty(num_experts, self.output_size_per_partition, input_size),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ) -> None:
        output_begin = self.tp_rank * self.output_size_per_partition
        if loaded_shard_id is None:
            expected = (self.num_experts, self.output_size, self.input_size)
            if tuple(loaded_weight.shape) != expected:
                raise ValueError(
                    f"Packed MoVA value weight must be {expected}, got "
                    f"{tuple(loaded_weight.shape)}"
                )
            local_weight = loaded_weight.narrow(
                1, output_begin, self.output_size_per_partition
            )
            param.data.copy_(local_weight)
            return

        if not 0 <= loaded_shard_id < self.num_experts:
            raise ValueError(f"Invalid MoVA value expert id: {loaded_shard_id}")
        expected = (self.output_size, self.input_size)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"MoVA value expert must be {expected}, got {tuple(loaded_weight.shape)}"
            )
        local_weight = loaded_weight.narrow(
            0, output_begin, self.output_size_per_partition
        )
        param.data[loaded_shard_id].copy_(local_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        return routed_linear(
            hidden_states,
            self.weight,
            routing_weights,
            selected_experts,
        )
