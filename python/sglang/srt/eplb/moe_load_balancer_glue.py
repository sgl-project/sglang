"""SGLang-owned glue for EvenMoE, the framework-neutral MoE load balancer.

EvenMoE (https://github.com/ai-dynamo/evenmoe, import name ``evenmoe``) owns
the algorithms (L1 expert placement: EPLB variants, UltraEP; L2 replica
routing: static, LPLB, Waterfill, UltraEP). SGLang stays the authority over
the committed expert placement and the collectives; this module translates
between the two. Everything here is a no-op unless
``--moe-load-balancer-algorithm`` is set, and ``evenmoe`` is only imported then.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)
from sglang.srt.runtime_context import get_context, get_exec

_FORWARD_MODE_TO_STAGE = {
    ForwardMode.EXTEND: "prefill",
    ForwardMode.DLLM_EXTEND: "prefill",
    ForwardMode.DECODE: "decode",
    ForwardMode.MIXED: "mixed",
    ForwardMode.IDLE: "idle",
    ForwardMode.TARGET_VERIFY: "speculative",
    ForwardMode.DRAFT_EXTEND_V2: "speculative",
}


def route_topk_with_mlb(
    *,
    moe_load_balancer,
    layer_id: int,
    topk_output,
    num_tokens: int,
    num_token_non_padded: Optional[torch.Tensor],
    routed_scaling_factor: float,
):
    """Run the configured L2 pipeline and materialize SGLang TopK output.

    ``topk_output`` holds logical expert ids straight from the router; the
    returned output holds physical ids (and the fused shared-expert slot when
    the pipeline routes it) and has been recorded for EPLB statistics and
    routed-experts capture.
    """

    from evenmoe.adapters.sglang import (
        to_routing_request,
        to_sglang_routing_output,
    )

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    context = get_context()
    refresh = context.resources.mlb_placement_refresh
    if refresh is not None:
        # Coupled placement re-plans this layer from the current batch and
        # commits it before routing reads the placement.
        refresh(layer_id, topk_output.topk_ids)
    request = to_routing_request(
        context=context,
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        token_count=(
            num_token_non_padded if num_token_non_padded is not None else num_tokens
        ),
        stage=_current_stage(),
        routed_scaling_factor=routed_scaling_factor,
    )
    decision = moe_load_balancer.route_tokens(request)
    output = to_sglang_routing_output(
        decision,
        context=context,
        router_logits=topk_output.router_logits,
        routed_scaling_factor=routed_scaling_factor,
    )
    resources = context.resources
    # With a balancer, physical ids are recorded here rather than in TopK.
    resources.expert_distribution_recorder.on_select_experts(
        topk_ids=output.recorded_physical_topk_ids
    )
    if resources.experts_capturer is not None:
        resources.experts_capturer.capture(
            layer_id=layer_id, topk_indices=output.recorded_physical_topk_ids
        )
    return StandardTopKOutput(
        topk_weights=output.topk_weights,
        topk_ids=output.topk_ids,
        router_logits=output.router_logits,
    )


def _current_stage() -> Optional[str]:
    """Routing stage of the active forward (ForwardContext.forward_mode)."""
    if not has_forward_context():
        return None
    return _FORWARD_MODE_TO_STAGE.get(get_forward_context().forward_mode)


def get_configured_moe_load_balancer_pipeline():
    """The server's ``--moe-load-balancer-algorithm`` pipeline, or None."""
    return get_moe_load_balancer_pipeline(get_exec().moe.moe_load_balancer_algorithm)


def resolve_moe_load_balancer():
    """The MoELoadBalancer a target (Hash)TopK under construction routes through.

    None unless the pipeline routes after TopK. ModelRunner publishes the
    target's balancer before it builds the model, so it is bound once here
    instead of being attached to every router after loading.
    """
    pipeline = get_configured_moe_load_balancer_pipeline()
    if pipeline is None or not pipeline.capabilities.requires_post_topk_routing:
        return None
    moe_load_balancer = get_context().resources.moe_load_balancer
    if moe_load_balancer is None:
        raise RuntimeError(
            "--moe-load-balancer-algorithm routes after TopK, but no "
            "MoELoadBalancer is published; ModelRunner must create it before "
            "building the model."
        )
    return moe_load_balancer


def get_moe_load_balancer_pipeline(algorithm):
    """Parse ``--moe-load-balancer-algorithm`` into an EvenMoE RoutingPipeline."""
    if algorithm is None:
        return None
    try:
        from evenmoe import RoutingPipeline
    except ImportError as error:
        raise ImportError(
            "--moe-load-balancer-algorithm requires the evenmoe package "
            "(https://github.com/ai-dynamo/evenmoe)."
        ) from error

    return RoutingPipeline.from_value(algorithm)


def uses_coupled_placement(pipeline_or_balancer) -> bool:
    """Whether the routing pipeline brings its own placement policy (today: UltraEP).

    Its placement also produces routing state (a per-source-rank quota), is
    re-planned per layer from live traffic using per-rank statistics, and keeps
    each rank's master experts fixed. Accepts a RoutingPipeline or the
    MoELoadBalancer built from it.
    """
    return (
        pipeline_or_balancer is not None
        and pipeline_or_balancer.placement_policy is not None
    )


def resolve_pipeline_a2a_backend(pipeline, moe_a2a_backend: str) -> str:
    """The ``--moe-a2a-backend`` to run ``pipeline`` with: the requested one if
    supported, else the pipeline's first supported backend SGLang has."""
    from evenmoe.adapters.sglang import resolve_sglang_a2a_backend

    return resolve_sglang_a2a_backend(pipeline.capabilities, moe_a2a_backend)
