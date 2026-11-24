from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import is_dp_max_padding
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.utils import (
    get_moe_runner_backend,
    has_per_rank_fused_shared_slots,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKConfig


class EpRemapPolicy(msgspec.Struct, frozen=True):
    """Init-static EP local expert remapping and padding-mask eligibility."""

    allow_padded_row_mask: bool = False
    can_fuse_static: bool = False
    ep_size: int = 1
    ep_rank: int = 0
    interleaved: bool = False


def resolve_ep_remap_policy(
    *,
    topk_config: TopKConfig,
    is_cuda: bool,
    use_aiter: bool,
    is_xpu: bool,
    is_expert_remapping_enabled_fn: Callable[[], bool],
) -> EpRemapPolicy:
    disable_fused = envs.SGLANG_DISABLE_FUSED_EP_REMAP.get()
    allow_padded_row_mask = (
        is_cuda and not use_aiter and not is_xpu and not disable_fused
    )
    if not allow_padded_row_mask or topk_config.custom_routing_function is not None:
        return EpRemapPolicy(allow_padded_row_mask=allow_padded_row_mask)
    try:
        # Standalone select_experts unit tests may omit server/parallel config.
        get_server_args()
    except ValueError:
        return EpRemapPolicy(allow_padded_row_mask=allow_padded_row_mask)
    parallel = get_parallel()
    ep_size = parallel.moe_ep_size
    ep_rank = parallel.moe_ep_rank
    interleaved = envs.SGLANG_ENABLE_EP_INTERLEAVED_EXPERTS.get()
    runner_backend = topk_config.runner_backend or get_moe_runner_backend()

    # Lazy import: standard.py imports topk.py at module load time.
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        _keeps_global_expert_ids,
    )

    can_fuse_static = (
        allow_padded_row_mask
        and ep_size > 1
        # DWDP replicates full experts after load and collapses FusedMoE.moe_ep_size=1.
        and parallel.dwdp_size <= 1
        and not topk_config.shared_experts.use_waterfill
        and get_exec().moe.ep_join_mode != "scale"
        and get_moe_a2a_backend().is_none()
        and not _keeps_global_expert_ids(runner_backend)
        and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        and not is_expert_remapping_enabled_fn()
        and not has_per_rank_fused_shared_slots(topk_config.num_shared_topk_slots)
        and not (
            topk_config.num_shared_topk_slots > 0
            and topk_config.fused_shared_experts_scaling_factor is not None
        )
    )
    return EpRemapPolicy(
        allow_padded_row_mask=allow_padded_row_mask,
        can_fuse_static=can_fuse_static,
        ep_size=ep_size,
        ep_rank=ep_rank,
        interleaved=interleaved,
    )


def can_fuse_ep_local_expert_remap(
    *,
    policy: EpRemapPolicy,
    allow_routed_experts_capture: bool,
    num_routed_experts: int,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo],
    routing_overridden: bool,
) -> bool:
    if not policy.can_fuse_static:
        return False
    if num_routed_experts <= 0 or num_routed_experts % policy.ep_size != 0:
        return False
    if expert_location_dispatch_info is not None or routing_overridden:
        return False

    # Lazy import: standard.py imports topk.py at module load time.
    from sglang.srt.layers.moe.token_dispatcher.standard import _MASK_DP_PAD_MOE

    if _MASK_DP_PAD_MOE and is_dp_max_padding():
        return False
    if allow_routed_experts_capture and get_global_experts_capturer() is not None:
        return False
    recorder = get_global_expert_distribution_recorder()
    if recorder.recording or recorder.get_current_pass_count_buffer() is not None:
        return False
    return True


def build_ep_remap_kwargs(*, policy: EpRemapPolicy, num_routed_experts: int) -> dict:
    if policy.interleaved:
        return {"ep_interleaved_size": policy.ep_size, "ep_rank": policy.ep_rank}
    num_local_routed = num_routed_experts // policy.ep_size
    return {
        "ep_start": policy.ep_rank * num_local_routed,
        "ep_end": (policy.ep_rank + 1) * num_local_routed,
    }


def prepare_biased_topk_ep_kwargs(
    *,
    policy: EpRemapPolicy,
    topk_config: TopKConfig,
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    scoring_func: str,
    top_k_for_gate: int,
    num_shared_topk_slots_for_gate: int,
    num_token_non_padded: Optional[torch.Tensor],
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo],
    routing_overridden: bool,
    packed_topk: Optional[torch.Tensor],
    router_logits_partials: Optional[torch.Tensor],
) -> Tuple[dict, bool]:
    if (
        not policy.allow_padded_row_mask
        or packed_topk is not None
        or router_logits_partials is not None
    ):
        return {}, False
    if (
        scoring_func == "sigmoid"
        and correction_bias is not None
        and num_shared_topk_slots_for_gate == 0
    ):
        from sglang.kernels.ops.moe import moe_route_radix

        if correction_bias.stride(0) == 1 and moe_route_radix.covered(
            router_logits, correction_bias, top_k_for_gate
        ):
            return {}, False
    kwargs = {}
    if num_token_non_padded is not None:
        kwargs["mask_padded_rows"] = True
    ep_local_remapped = can_fuse_ep_local_expert_remap(
        policy=policy,
        allow_routed_experts_capture=topk_config.allow_routed_experts_capture,
        num_routed_experts=router_logits.shape[1],
        expert_location_dispatch_info=expert_location_dispatch_info,
        routing_overridden=routing_overridden,
    )
    if ep_local_remapped:
        kwargs.update(
            build_ep_remap_kwargs(
                policy=policy, num_routed_experts=router_logits.shape[1]
            )
        )
    return kwargs, ep_local_remapped


def prepare_softmax_topk_ep_kwargs(
    *,
    policy: EpRemapPolicy,
    topk_config: TopKConfig,
    num_routed_experts: int,
    scoring_func: str,
    num_token_non_padded: Optional[torch.Tensor],
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo],
    routing_overridden: bool,
) -> Tuple[dict, bool, bool]:
    if not policy.allow_padded_row_mask or scoring_func != "softmax":
        return {}, False, False
    kwargs = {}
    padded_rows_masked = False
    if num_token_non_padded is not None:
        kwargs["num_token_non_padded"] = num_token_non_padded
        padded_rows_masked = True
    ep_local_remapped = can_fuse_ep_local_expert_remap(
        policy=policy,
        allow_routed_experts_capture=topk_config.allow_routed_experts_capture,
        num_routed_experts=num_routed_experts,
        expert_location_dispatch_info=expert_location_dispatch_info,
        routing_overridden=routing_overridden,
    )
    if ep_local_remapped:
        kwargs.update(
            build_ep_remap_kwargs(
                policy=policy, num_routed_experts=num_routed_experts
            )
        )
    return kwargs, padded_rows_masked, ep_local_remapped

