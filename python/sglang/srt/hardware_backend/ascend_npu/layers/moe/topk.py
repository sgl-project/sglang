from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, select_experts

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopK, TopKOutput


def topk(
    self: TopK,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    use_grouped_topk: bool,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> TopKOutput:
    topk_native = self.topk_config.torch_native

    if not use_grouped_topk and not topk_native:
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=self.topk_config.top_k,
        )
        topk_weights = topk_weights.to(torch.float32)

        if renormalize:
            topk_weights_sum = (
                topk_weights.sum(dim=-1, keepdim=True)
                if self.topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
            )
            topk_weights = topk_weights / topk_weights_sum

        if expert_location_dispatch_info is not None:
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info
            )
        get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

        return StandardTopKOutput(topk_weights, topk_ids, _)
    if use_grouped_topk and not topk_native and router_logits.shape[-1] == 256:
        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        routed_scaling_factor = self.topk_config.routed_scaling_factor or 1

        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=self.topk_config.top_k,
            bias=self.topk_config.correction_bias.to(torch.float32),
            k_group=self.topk_config.topk_group,
            group_count=self.topk_config.num_expert_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1,
            routed_scaling_factor=routed_scaling_factor,
            eps=float(1e-20),
        )

        if renormalize:
            topk_weights_sum = (
                topk_weights.sum(dim=-1, keepdim=True)
                if self.topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
            )
            topk_weights = topk_weights / topk_weights_sum

        if expert_location_dispatch_info is not None:
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info
            )
        get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

        return StandardTopKOutput(topk_weights, topk_ids, _)
    else:
        self.topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
