from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import StandardTopKOutput, select_experts
from sglang.srt.layers.moe.utils import is_deepep_class_backend
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


@triton.jit
def _fused_remap_deepep_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    N,
    K,
    K_PLUS_N,
    num_local_routed,
    num_local_experts,
    ep_rank,
    shared_weight,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    cols = tl.arange(0, BLOCK)
    mask = cols < K_PLUS_N
    is_routed = cols < K

    # Clamp load indices to stay within input bounds for masked columns
    safe_cols = tl.where(is_routed, cols, tl.zeros_like(cols))

    # Load routed IDs and apply interleaved remap: id -> id + id // num_local_routed
    routed_ids = tl.load(
        topk_ids_ptr + row * K + safe_cols, mask=is_routed, other=0
    )
    remapped_ids = routed_ids + routed_ids // num_local_routed

    # Shared expert IDs: ep_rank * num_local_experts + num_local_routed + shared_idx
    shared_idx = cols - K
    shared_ids = ep_rank * num_local_experts + num_local_routed + shared_idx

    final_ids = tl.where(is_routed, remapped_ids, shared_ids)
    tl.store(out_ids_ptr + row * K_PLUS_N + cols, final_ids, mask=mask)

    # Copy routed weights; set shared weight to shared_weight
    routed_weights = tl.load(
        topk_weights_ptr + row * K + safe_cols, mask=is_routed, other=0.0
    )
    final_weights = tl.where(is_routed, routed_weights, shared_weight)
    tl.store(out_weights_ptr + row * K_PLUS_N + cols, final_weights, mask=mask)


def _fused_remap_deepep_npu(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_fused_shared_experts: int,
    num_physical_routed_experts: int,
    topk_config: "TopKConfig",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton kernel: expand routed topk with shared experts and remap
    to DeepEP interleaved layout. Replaces torch.cat + _remap_topk_for_deepep."""
    from sglang.srt.distributed.parallel_state import (
        get_moe_expert_parallel_rank,
        get_moe_expert_parallel_world_size,
    )

    if topk_ids.shape[0] == 0:
        return topk_ids, topk_weights

    ep_size = get_moe_expert_parallel_world_size()
    ep_rank = get_moe_expert_parallel_rank()
    num_local_routed = num_physical_routed_experts // ep_size
    num_local_experts = num_local_routed + num_fused_shared_experts

    N = topk_ids.shape[0]
    K = topk_ids.shape[1]
    K_PLUS_N = K + num_fused_shared_experts

    routed_scaling_factor = topk_config.routed_scaling_factor
    shared_weight = (
        1.0 if not routed_scaling_factor else 1.0 / routed_scaling_factor
    )

    out_ids = topk_ids.new_empty((N, K_PLUS_N), dtype=topk_ids.dtype)
    out_weights = topk_weights.new_empty(
        (N, K_PLUS_N), dtype=topk_weights.dtype
    )

    BLOCK = max(32, K_PLUS_N)
    _fused_remap_deepep_kernel[(N,)](
        topk_ids,
        topk_weights,
        out_ids,
        out_weights,
        N=N,
        K=K,
        K_PLUS_N=K_PLUS_N,
        num_local_routed=num_local_routed,
        num_local_experts=num_local_experts,
        ep_rank=ep_rank,
        shared_weight=shared_weight,
        BLOCK=BLOCK,
    )

    return out_ids, out_weights


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

    # DeepEP fusion post-processing: EPLB remap (routed only), then fused
    # kernel to append shared experts and remap to interleaved layout.
    if _is_deepep_fusion:
        n = topk_config.num_fused_shared_experts

        if expert_location_dispatch_info is not None:
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info
            )

        num_physical_routed_experts = (
            expert_location_dispatch_info.num_physical_experts
            if expert_location_dispatch_info is not None
            else router_logits.shape[1]
        )

        topk_ids, topk_weights = _fused_remap_deepep_npu(
            topk_ids,
            topk_weights,
            n,
            num_physical_routed_experts,
            topk_config,
        )

        if (cap := get_global_experts_capturer()) is not None:
            cap.capture(
                layer_id=layer_id,
                topk_indices=topk_ids,
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
