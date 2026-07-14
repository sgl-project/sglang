from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    capture_routed_experts_if_allowed,
    select_experts,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def _apply_routed_scaling_after_renorm(
    topk_weights: torch.Tensor,
    topk_config: "TopKConfig",
) -> torch.Tensor:
    """Mirror GPU post-renorm scaling when apply_routed_scaling_factor_on_output is set."""
    if (
        topk_config.renormalize
        and topk_config.apply_routed_scaling_factor_on_output
        and topk_config.routed_scaling_factor is not None
    ):
        return topk_weights * topk_config.routed_scaling_factor
    return topk_weights


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias
    use_npu_moe_gating_top_k = get_bool_env_var("USE_NPU_MOE_GATING_TOP_K")

    # sqrtsoftplus (DSV4 noaux_tc): top-k over (scores + bias); weights from
    # un-biased scores. The custom op fuses softplus/sqrt/topk/gather/norm/cast.
    if topk_config.scoring_func == "sqrtsoftplus":
        if use_npu_moe_gating_top_k:
            routed_scaling_factor = (
                topk_config.routed_scaling_factor
                if topk_config.apply_routed_scaling_factor_on_output
                else 1.0
            )
            topk_weights, topk_ids, _ = torch.ops.custom.npu_moe_gating_top_k(
                x=router_logits.to(torch.float32),
                k=topk_config.top_k,
                bias=(
                    correction_bias.to(torch.float32)
                    if correction_bias is not None
                    else None
                ),
                input_ids=None,
                tid2eid=None,
                routed_scaling_factor=float(routed_scaling_factor),
                norm_type=2,
            )
        else:
            scores = torch.nn.functional.softplus(router_logits.float()).sqrt()
            scores_for_choice = (
                scores + correction_bias.unsqueeze(0).float()
                if correction_bias is not None
                else scores
            )
            _, topk_ids = torch.topk(
                scores_for_choice, k=topk_config.top_k, dim=-1, sorted=False
            )
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = scores.gather(1, topk_ids)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            elif topk_config.routed_scaling_factor is not None:
                topk_weights = topk_weights * topk_config.routed_scaling_factor
        topk_weights = topk_weights.to(torch.float32)

    # Fast path: simple top-k without grouped routing and bias
    elif not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=topk_config.top_k,
        )

        if renormalize:
            topk_weights = l1_norm(
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
        topk_weights = topk_weights.to(torch.float32)

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            # num_expert_group and topk_group in some topk_config without group is None, (not supported by this ops)
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=0,
            # 1 for sigmoid, 0 for softmax
            norm_type=1,
            routed_scaling_factor=(
                topk_config.routed_scaling_factor
                if topk_config.apply_routed_scaling_factor_on_output
                else 1
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # torch native is not yet supported num_token_non_padded
    # Fallback to torch native implementation
    else:
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
