from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.layers.moe.topk import StandardTopKOutput, select_experts
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    tid2eid: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias
    scoring_func = topk_config.scoring_func
    routed_scaling_factor = topk_config.routed_scaling_factor
    is_hash_layer = topk_config.is_hash_layer

    use_npu_moe_gating_top_k = get_bool_env_var("USE_NPU_MOE_GATING_TOP_K")
    if use_npu_moe_gating_top_k and (is_hash_layer or scoring_func == "sqrtsoftplus"):
        if is_hash_layer:
            bias = None
        else:
            bias = correction_bias
            input_ids = None
            tid2eid = None
        topk_weights, topk_ids, _ = torch.ops.custom.npu_moe_gating_top_k(
            x=router_logits,
            k=topk_config.top_k,
            bias=bias,
            input_ids=input_ids,
            tid2eid=tid2eid,
            routed_scaling_factor=routed_scaling_factor,
            norm_type=2,
        )
    elif is_hash_layer or scoring_func == "sqrtsoftplus":
        assert (
            input_ids.numel() == router_logits.shape[0]
        ), f"{input_ids.numel()} vs {router_logits.shape[0]}"
        if use_grouped_topk:
            raise NotImplementedError(f"grouped topk not implemented")
        hash_idx = tid2eid[input_ids] if is_hash_layer else None

        if scoring_func == "sigmoid":
            scores = router_logits.sigmoid()
        elif scoring_func == "softmax":
            scores = router_logits.softmax(dim=-1, dtype=torch.float32)
        elif scoring_func == "sqrtsoftplus":
            scores = F.softplus(router_logits).sqrt()
        else:
            raise NotImplementedError(
                f"not supported scoring function for MOE gating:{scoring_func}"
            )

        # select top-k experts
        original_scores = scores
        if correction_bias is not None:
            scores = scores + correction_bias.unsqueeze(0)
        _, topk_ids = torch.topk(scores, k=topk_config.top_k, dim=-1, sorted=False)
        topk_ids = hash_idx if hash_idx is not None else topk_ids
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = original_scores.gather(1, topk_ids)

        # norm gate to sum 1
        if scoring_func != "softmax":
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * routed_scaling_factor

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

    # Grouped top-k with correction bias
    elif use_grouped_topk and correction_bias is not None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=correction_bias.to(torch.float32),
            k_group=topk_config.topk_group,
            group_count=topk_config.num_expert_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1,
            routed_scaling_factor=(
                1 if renormalize else topk_config.routed_scaling_factor
            ),
            eps=float(1e-20),
        )

    # npu_moe_gating_top_k is not yet supported custom_routing_function
    # torch native is not yet supported num_token_non_padded
    elif (
        topk_config.custom_routing_function is None
        and num_token_non_padded is not None
        and correction_bias is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=correction_bias.to(torch.float32),
            renorm=0,
            norm_type=1,
            routed_scaling_factor=(
                1 if renormalize else topk_config.routed_scaling_factor
            ),
            eps=float(1e-20),
        )

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
    get_global_experts_capturer().capture(
        layer_id=layer_id,
        topk_ids=topk_ids,
    )

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
