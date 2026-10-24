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

from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)
from sglang.srt.runtime_context import get_context, get_exec

_FORWARD_MODE_TO_STAGE = {
    "EXTEND": "prefill",
    "SPLIT_PREFILL": "prefill",
    "DLLM_EXTEND": "prefill",
    "DECODE": "decode",
    "MIXED": "mixed",
    "IDLE": "idle",
    "TARGET_VERIFY": "speculative",
    "DRAFT_EXTEND": "speculative",
    "DRAFT_EXTEND_V2": "speculative",
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
    forward_mode = get_forward_context().forward_mode
    if forward_mode is None:
        return None
    name = getattr(forward_mode, "name", None) or str(forward_mode).upper()
    return _FORWARD_MODE_TO_STAGE.get(name)


def get_configured_moe_load_balancer_pipeline():
    """The server's ``--moe-load-balancer-algorithm`` pipeline, or None."""
    return get_moe_load_balancer_pipeline(get_exec().moe.moe_load_balancer_algorithm)


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
