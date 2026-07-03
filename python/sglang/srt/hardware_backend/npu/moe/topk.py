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

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def _mask_padded_tokens(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
) -> None:
    if num_token_non_padded is None:
        return
    indices = torch.arange(topk_ids.shape[0], device=topk_ids.device)
    if isinstance(num_token_non_padded, torch.Tensor):
        num_token_non_padded = num_token_non_padded.to(device=topk_ids.device)
    # NOTE: boolean-index assignment (topk_ids[mask, :] = v) lowers to
    # aclnnNonzeroV2 on Ascend, which has a data-dependent output shape and
    # cannot be captured by NPU graph capture (broke decode cuda-graph init).
    # Use in-place masked_fill_ instead: same semantics, graph-safe (elementwise).
    padding_mask = (indices >= num_token_non_padded).unsqueeze(-1)
    topk_ids.masked_fill_(padding_mask, -1)
    topk_weights.masked_fill_(padding_mask, 0.0)


def _biased_sigmoid_topk_torch_npu(
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = router_logits.to(torch.float32).sigmoid()
    scores_for_choice = scores + topk_config.correction_bias.to(torch.float32)
    _, topk_ids = torch.topk(
        scores_for_choice,
        k=topk_config.top_k,
        dim=-1,
        sorted=False,
    )
    topk_weights = scores.gather(1, topk_ids)

    if topk_config.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if topk_config.apply_routed_scaling_factor_on_output:
            topk_weights = topk_weights * (
                topk_config.routed_scaling_factor
                if topk_config.routed_scaling_factor is not None
                else 1.0
            )

    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    return topk_weights, topk_ids


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
    scoring_func = topk_config.scoring_func

    # Fast path: simple top-k without grouped routing and bias
    if (
        not use_grouped_topk
        and correction_bias is None
        and scoring_func == "softmax"
        and num_token_non_padded is None
    ):
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

    # sqrtsoftplus (DSV4 noaux_tc): the NPU op only scores sigmoid/softmax, so use
    # a torch path. top-k over (scores + bias); weights from un-biased scores.
    elif topk_config.scoring_func == "sqrtsoftplus":
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
        else:
            topk_weights = topk_weights * topk_config.routed_scaling_factor
        topk_weights = topk_weights.to(torch.float32)

    # MiniMax-M3 uses sigmoid routing with correction bias. The bias must only
    # affect expert selection; combine weights come from the original sigmoid
    # scores, then get normalized and scaled.
    elif (
        not use_grouped_topk
        and correction_bias is not None
        and scoring_func == "sigmoid"
        and topk_config.num_fused_shared_experts == 0
    ):
        topk_weights, topk_ids = _biased_sigmoid_topk_torch_npu(
            router_logits, topk_config, num_token_non_padded
        )

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or scoring_func == "sigmoid"
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
                1 if renormalize else topk_config.routed_scaling_factor
            ),
            eps=float(1e-20),
        )
        if renormalize:
            topk_weights = l1_norm(
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
            if topk_config.apply_routed_scaling_factor_on_output:
                topk_weights = topk_weights * (
                    topk_config.routed_scaling_factor
                    if topk_config.routed_scaling_factor is not None
                    else 1.0
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
    _mask_padded_tokens(topk_weights, topk_ids, num_token_non_padded)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
