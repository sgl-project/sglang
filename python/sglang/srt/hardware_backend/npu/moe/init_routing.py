"""
NPU MoE init routing components.

Prepare token routing before expert computation. Two API versions are provided:
- v1: legacy routing using ``npu_moe_init_routing``.
- v2: improved routing using ``npu_moe_init_routing_v2``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseInitRouting(ABC):
    """Abstract base for NPU MoE init routing."""

    @abstractmethod
    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: ...


class NPUMoEInitRouting_v1(BaseInitRouting):
    """
    NPU MoE init routing (v1 API).

    Uses ``npu_moe_init_routing`` with a manually constructed ``row_idx`` tensor.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_ids.device)
            .view(topk_ids.shape[1], -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            )
        )
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, None


class NPUMoEInitRouting_v2(BaseInitRouting):
    """
    NPU MoE init routing (v2 API).

    Uses ``npu_moe_init_routing_v2``, which integrates expert token counting.
    """

    def __init__(self, quant_mode: int = -1):
        self.quant_mode = quant_mode

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=self.quant_mode,
            )
        )
        if self.quant_mode == -1:
            pertoken_scale = None
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale


class NPUMoEInitRouting_Quant(BaseInitRouting):
    """
    NPU MoE init routing (Quant API).

    Uses ``npu_moe_init_routing_quant``, which integrates expert token counting.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]

        hidden_states, expanded_row_idx, expert_tokens, _, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_quant(
                hidden_states,
                topk_ids,
                active_num=num_tokens * topk_ids.shape[1],
                expert_num=num_experts,
                expert_tokens_num_mode=1,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale
