"""ROCm/AITER grouped routing with persistent shared-expert columns.

Each routing config owns a buffer whose shared columns are pre-filled once, so
layers never alias; AITER writes the routed columns through strided views.
Batches taller than the buffer keep the append kernel.
"""

from functools import partial
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import has_per_rank_fused_shared_slots
from sglang.srt.runtime_context import (
    get_buffer,
    get_context,
    get_parallel,
    get_resources,
    get_schedule,
)

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig

_MAX_BUFFER_TOKENS = 131072
_MIN_BUFFER_TOKENS = 8192


def _buffer_height() -> int:
    """Rows per buffer: the prefill token budget, floored for tiny configs and capped."""
    schedule = get_schedule()
    budget = max(schedule.chunked_prefill_size or 0, schedule.max_prefill_tokens or 0)
    if budget <= 0:
        return _MAX_BUFFER_TOKENS
    return min(max(budget, _MIN_BUFFER_TOKENS), _MAX_BUFFER_TOKENS)


def _allocate_shared_columns(
    routed_k: int,
    n_shared: int,
    num_experts: int,
    shared_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[height, routed_k + n_shared] weight/id buffers with the shared columns
    pre-filled (id = num_experts + i, weight = shared_weight)."""
    height = _buffer_height()
    total = routed_k + n_shared
    weights = torch.empty((height, total), dtype=torch.float32, device=device)
    ids = torch.empty((height, total), dtype=torch.int32, device=device)
    ids[:, routed_k:] = torch.arange(
        num_experts, num_experts + n_shared, dtype=torch.int32, device=device
    )
    weights[:, routed_k:] = shared_weight
    return weights, ids


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
    n_shared = config.num_shared_topk_slots
    if (
        not config.use_grouped_topk
        or config.correction_bias is None
        or n_shared <= 0
        # Standalone callers that never published config keep the ordinary path.
        or not get_context().is_config_namespace_published("parallel")
        or get_parallel().moe_ep_size != 1
        or num_token_non_padded is not None
        or expert_location_dispatch_info is not None
        or has_per_rank_fused_shared_slots(n_shared)
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
        or envs.SGLANG_K3_RADIX4_TOPK.get()
        or envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.get()
        or config.apply_routed_scaling_factor_on_output
        or config.num_expert_group is None
        or config.topk_group is None
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
    shared_weight = 1.0 if shared_weight is None else float(shared_weight)
    device = router_logits.device
    name = (
        f"aiter_topk_shared_columns:{id(config)}:{routed_k}:{n_shared}:"
        f"{num_experts}:{shared_weight}:{device}"
    )
    # Allocation must not land inside a graph capture; warmup normally allocates first.
    if name not in get_resources().buffers and torch.cuda.is_current_stream_capturing():
        return None
    full_w, full_ids = get_buffer(
        name,
        partial(
            _allocate_shared_columns,
            routed_k=routed_k,
            n_shared=n_shared,
            num_experts=num_experts,
            shared_weight=shared_weight,
            device=device,
        ),
    )
    if token > full_w.shape[0]:
        return None
    routed_w = full_w[:token, :routed_k]
    routed_ids = full_ids[:token, :routed_k]
    # Same kernel inputs as biased_grouped_topk_gpu: an fp32 bias routes in fp32.
    if config.correction_bias.dtype == torch.float32:
        route_logits = router_logits.to(torch.float32)
        route_bias = config.correction_bias
    else:
        route_logits = router_logits
        route_bias = config.correction_bias_for_dtype(router_logits.dtype)
    # AITER honors the output row stride, leaving pre-filled shared columns intact.
    biased_grouped_topk(
        route_logits,
        route_bias,
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
