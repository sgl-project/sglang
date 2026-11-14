"""ROCm/AITER grouped routing with persistent shared-expert columns.

Each routing config owns a buffer whose shared columns are pre-filled once, so
layers never alias; AITER writes the routed columns through strided views.
Batches taller than the buffer keep the append kernel.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import (
    has_per_rank_fused_shared_slots,
    is_tbo_enabled,
)
from sglang.srt.runtime_context import get_context, get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig

_MAX_BUFFER_TOKENS = 131072
_MIN_BUFFER_TOKENS = 8192


def _buffer_height() -> int:
    """Rows per buffer: the prefill token budget, floored and capped. Without a
    budget it is the floor; taller batches keep the append kernel."""
    schedule = get_schedule()
    budget = max(schedule.chunked_prefill_size or 0, schedule.max_prefill_tokens or 0)
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


def _shared_columns(
    config: "TopKConfig", *, routed_k: int, num_experts: int, device: torch.device
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The config's persistent buffers, or None when creating them now would
    land inside a graph capture."""
    n_shared = config.num_shared_topk_slots
    shared_weight = config.fused_shared_experts_scaling_factor
    shared_weight = 1.0 if shared_weight is None else float(shared_weight)
    key = (routed_k, n_shared, num_experts, shared_weight, device)
    cached = config.aiter_shared_columns
    if cached is None:
        # Allocation must not land inside a graph capture; warmup normally allocates first.
        if torch.cuda.is_current_stream_capturing():
            return None
        weights, ids = _allocate_shared_columns(
            routed_k=routed_k,
            n_shared=n_shared,
            num_experts=num_experts,
            shared_weight=shared_weight,
            device=device,
        )
        cached = config.aiter_shared_columns = (key, weights, ids)
    # Captured graphs keep writing the first buffer, so a changed layout falls back
    # instead of replacing it.
    if cached[0] != key:
        return None
    return cached[1], cached[2]


def _radix4_routes(
    router_logits: torch.Tensor,
    route_bias: torch.Tensor,
    *,
    routed_k: int,
    num_expert_group: int,
    topk_group: int,
) -> bool:
    """Whether the ordinary path routes this call with radix4 instead of AITER."""
    if not envs.SGLANG_K3_RADIX4_TOPK.get():
        return False
    from sglang.kernels.ops.moe import moe_route_radix4

    if not moe_route_radix4.available():
        return False
    scores = router_logits
    if scores.dtype != route_bias.dtype:
        # The router casts the logits to the bias dtype; a meta tensor with the
        # cast's layout decides coverage without paying for the cast.
        scores = torch.empty_like(scores, dtype=route_bias.dtype, device="meta")
    return moe_route_radix4.covered(
        scores, route_bias, routed_k, num_expert_group, topk_group
    )


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
        # The other micro-batch may route through this layer before this one's
        # experts read the layer's single buffer.
        or is_tbo_enabled()
        or num_token_non_padded is not None
        or expert_location_dispatch_info is not None
        or has_per_rank_fused_shared_slots(n_shared)
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
        or envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.get()
        # TopK.forward_native (the torch.compile bs=1 route) keeps the ordinary path.
        or config.torch_native
        or config.apply_routed_scaling_factor_on_output
        or config.num_expert_group is None
        or config.topk_group is None
    ):
        return None

    from aiter import biased_grouped_topk

    # Lazy import because the common dispatcher imports this module.
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput,
        aiter_grouped_topk_route_inputs,
        capture_routed_experts_if_allowed,
    )

    assert hidden_states.shape[0] == router_logits.shape[0]
    token, num_experts = router_logits.shape
    routed_k = config.top_k - n_shared
    # The bias select_experts hands the ordinary router: fp32 as is, else the cached cast.
    bias = config.correction_bias
    if bias.dtype != torch.float32:
        bias = config.correction_bias_for_dtype(router_logits.dtype)
    if _radix4_routes(
        router_logits,
        bias,
        routed_k=routed_k,
        num_expert_group=config.num_expert_group,
        topk_group=config.topk_group,
    ):
        return None
    buffers = _shared_columns(
        config, routed_k=routed_k, num_experts=num_experts, device=router_logits.device
    )
    if buffers is None or token > buffers[0].shape[0]:
        return None
    full_w, full_ids = buffers
    routed_w = full_w[:token, :routed_k]
    routed_ids = full_ids[:token, :routed_k]
    route_logits, route_bias = aiter_grouped_topk_route_inputs(router_logits, bias)
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
