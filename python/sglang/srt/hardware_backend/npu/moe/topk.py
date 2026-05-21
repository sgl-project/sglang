from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import StandardTopKOutput, select_experts
from sglang.srt.layers.moe.utils import is_deepep_class_backend
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


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

    _is_deepep_fusion = (
        topk_config.num_fused_shared_experts > 0 and is_deepep_class_backend()
    )
    # When fusion is enabled, only select routed experts from native op;
    # shared expert column is appended afterwards.
    k = (
        topk_config.top_k - topk_config.num_fused_shared_experts
        if _is_deepep_fusion
        else topk_config.top_k
    )

    # Fast path: simple top-k without grouped routing and bias
    if not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=k,
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
            k=k,
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
            norm_type=1,  # 1 for sigmoid, 0 for softmax
            routed_scaling_factor=(
                1 if renormalize else topk_config.routed_scaling_factor
            ),
            eps=float(1e-20),
        )

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

    # DeepEP fusion post-processing: append shared expert columns, apply
    # EPLB remap (routed only), then remap to interleaved layout.
    if _is_deepep_fusion:
        num_experts = router_logits.shape[-1]
        n = topk_config.num_fused_shared_experts
        topk_ids = torch.cat(
            [
                topk_ids,
                topk_ids.new_full((topk_ids.shape[0], n), num_experts),
            ],
            dim=-1,
        )
        topk_weights = torch.cat(
            [
                topk_weights,
                topk_weights.new_full(
                    (topk_weights.shape[0], n),
                    (
                        1.0 / topk_config.routed_scaling_factor
                        if topk_config.routed_scaling_factor
                        else 1.0
                    ),
                ),
            ],
            dim=-1,
        )

        if expert_location_dispatch_info is not None:
            routed_cols = topk_ids[:, :-n]
            routed_cols = topk_ids_logical_to_physical(
                routed_cols, expert_location_dispatch_info
            )
            topk_ids = torch.cat([routed_cols, topk_ids[:, -n:]], dim=-1)

        if (cap := get_global_experts_capturer()) is not None:
            cap.capture(
                layer_id=layer_id,
                topk_indices=topk_ids,
            )

        from sglang.srt.layers.moe.topk import _remap_topk_for_deepep

        topk_ids, topk_weights = _remap_topk_for_deepep(
            topk_ids,
            topk_weights,
            topk_config.num_fused_shared_experts,
            router_logits.shape[1],
            topk_config,
        )

        get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    else:
        if expert_location_dispatch_info is not None:
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info
            )

        get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
        if (cap := get_global_experts_capturer()) is not None:
            cap.capture(
                layer_id=layer_id,
                topk_indices=topk_ids,
            )

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
