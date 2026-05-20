from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    _mask_topk_ids_padded_region,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


class HashTopK(nn.Module):
    def __init__(
        self,
        topk,
        num_experts,
        num_fused_shared_experts,
        vocab_size,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts
        self.score_func = scoring_func
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, topk - num_fused_shared_experts, dtype=torch.int32),
            requires_grad=False,
        )

        assert not apply_routed_scaling_factor_on_output, "not implemented"

    def empty_topk_output(self, device: torch.device):
        topk = self.topk - self.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)

    def _forward_torch(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = torch.nn.functional.softplus(router_logits).sqrt()

        num_token = scores.shape[0]

        topk_ids = torch.zeros(
            (num_token, self.topk), dtype=torch.int32, device=scores.device
        )
        topk_weights = torch.zeros(
            (num_token, self.topk), dtype=scores.dtype, device=scores.device
        )

        if self.num_fused_shared_experts == 1:
            topk_ids[:, :-1] = self.tid2eid[input_ids]
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])

            if self.score_func != "softmax":
                topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)

            topk_ids[:, -1] = torch.randint(
                low=self.num_experts,
                high=self.num_experts + self.num_fused_shared_experts,
                size=(num_token,),
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )

            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / self.routed_scaling_factor
            )
        else:
            topk_ids[:, :] = self.tid2eid[input_ids]
            topk_weights[:, :] = scores.gather(1, topk_ids[:, :])
            if self.score_func != "softmax":
                topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ):
        assert (
            input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0]
        ), f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"

        if envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
            from sglang.jit_kernel.deepseek_v4 import hash_topk

            topk_weights, topk_ids = hash_topk(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                num_fused_shared_experts=self.num_fused_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.score_func,
            )
        else:
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)

        if is_hip():
            topk_weights = topk_weights.to(torch.float32)

        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )
        return topk_output
