"""ROCm/AITER grouped routing with persistent shared-expert columns."""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import has_per_rank_fused_shared_slots
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig

# ROCm/AITER only: pre-fill shared columns once, then write routed columns
# through strided views. Each routing config owns its buffer to avoid aliasing
# between layers. Oversized batches retain the append kernel.
_AITER_TOPK_FUSE_SHARED_MAX_TOKENS_CAP = 131072
_aiter_topk_fuse_shared_max_tokens_cache = None
_aiter_topk_fuse_shared_bufs: dict = {}


def _get_aiter_topk_fuse_shared_max_tokens() -> int:
    """Bound persistent buffer memory using the fixed prefill token budget."""
    global _aiter_topk_fuse_shared_max_tokens_cache
    if _aiter_topk_fuse_shared_max_tokens_cache is None:
        from sglang.srt.runtime_context import get_schedule

        try:
            schedule = get_schedule()
        except ValueError:
            # Reached before the global config is published (a unit test, or
            # offline init). Return the cap without caching it, so a later call
            # still picks up the real value.
            return _AITER_TOPK_FUSE_SHARED_MAX_TOKENS_CAP
        cps = schedule.chunked_prefill_size or 0
        mpt = schedule.max_prefill_tokens or 0
        m = max(int(cps), int(mpt), 8192)  # 8192 floor for tiny configs
        if int(cps) <= 0 and int(mpt) <= 0:
            # chunked prefill disabled -> use the safety cap
            m = _AITER_TOPK_FUSE_SHARED_MAX_TOKENS_CAP
        _aiter_topk_fuse_shared_max_tokens_cache = min(
            m, _AITER_TOPK_FUSE_SHARED_MAX_TOKENS_CAP
        )
    return _aiter_topk_fuse_shared_max_tokens_cache


def _aiter_topk_fuse_shared_ep_is_single() -> bool:
    """Use ordinary append until the MoE parallel config is initialized."""
    try:
        return get_parallel().moe_ep_size == 1
    except (AssertionError, ValueError):
        return False


def _get_aiter_topk_fuse_shared_buf(
    topk_routed: int,
    n_shared: int,
    num_experts: int,
    shared_weight: float,
    device,
    buffer_key=None,
):
    """Persistent [MAX, topk_routed + n_shared] weight/id buffers whose shared
    columns are pre-filled once (id = num_experts + i, weight = shared_weight).
    Fixed max size (>= max prefill batch) so the tensor address is stable across
    CUDA-graph replays."""
    key = (
        topk_routed,
        n_shared,
        num_experts,
        float(shared_weight),
        str(device),
        buffer_key,
    )
    buf = _aiter_topk_fuse_shared_bufs.get(key)
    if buf is None:
        total = topk_routed + n_shared
        M = _get_aiter_topk_fuse_shared_max_tokens()
        w = torch.empty((M, total), dtype=torch.float32, device=device)
        ids = torch.empty((M, total), dtype=torch.int32, device=device)
        ids[:, topk_routed:] = torch.arange(
            num_experts, num_experts + n_shared, dtype=torch.int32, device=device
        ).unsqueeze(0)
        w[:, topk_routed:] = shared_weight
        buf = (w, ids)
        _aiter_topk_fuse_shared_bufs[key] = buf
    return buf


def try_select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    *,
    layer_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
) -> Optional["StandardTopKOutput"]:
    """Return fused routing, or None to use the ordinary routing/append path."""
    config = topk_config
    n_shared = config.num_fused_shared_experts
    if (
        not config.use_grouped_topk
        or config.correction_bias is None
        or n_shared <= 0
        or not _aiter_topk_fuse_shared_ep_is_single()
        or num_token_non_padded is not None
        or expert_location_dispatch_info is not None
        or has_per_rank_fused_shared_slots(n_shared)
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
        or envs.SGLANG_K3_RADIX4_TOPK.get()
        or config.apply_routed_scaling_factor_on_output
        or config.num_expert_group is None
        or config.topk_group is None
        or router_logits.shape[0] > _get_aiter_topk_fuse_shared_max_tokens()
    ):
        return None

    from aiter import biased_grouped_topk

    # Lazy import because the common dispatcher imports this module.
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput,
        capture_routed_experts_if_allowed,
        get_global_expert_distribution_recorder,
    )

    assert hidden_states.shape[0] == router_logits.shape[0]
    token, num_experts = router_logits.shape
    routed_k = config.top_k - n_shared
    shared_weight = config.fused_shared_experts_scaling_factor
    full_w, full_ids = _get_aiter_topk_fuse_shared_buf(
        routed_k,
        n_shared,
        num_experts,
        1.0 if shared_weight is None else shared_weight,
        router_logits.device,
        id(config),
    )
    routed_w = full_w[:token, :routed_k]
    routed_ids = full_ids[:token, :routed_k]
    # AITER honors the output row stride, leaving pre-filled shared columns intact.
    biased_grouped_topk(
        router_logits,
        config.correction_bias.to(dtype=router_logits.dtype),
        routed_w,
        routed_ids,
        config.num_expert_group,
        config.topk_group,
        config.renormalize,
        (
            config.routed_scaling_factor
            if config.routed_scaling_factor is not None
            else 1.0
        ),
    )
    capture_routed_experts_if_allowed(config, layer_id, routed_ids)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=routed_ids)
    return StandardTopKOutput(full_w[:token], full_ids[:token], router_logits)
