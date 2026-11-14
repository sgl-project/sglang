"""Per-layer online placement refresh.

The periodic EPLB rebalance re-plans the whole model between forwards
(EPLBManager). A pipeline that refreshes per layer (see
``refreshes_placement_per_layer``) instead re-plans one layer from the batch
that layer is about to route, inside the forward, and commits it before the
layer dispatches. SGLang owns when that happens (the representative-batch
decision below), the collectives, the weight move and the committed placement;
EvenMoE only plans.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import torch

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.runtime_context import get_context, get_parallel

# A refresh only counts batches with at least this many global tokens: smaller
# batches are too noisy to re-plan a layer's replicas from.
_MIN_REPRESENTATIVE_TOKENS = 512


class LayerPlacementRefresher:
    """Re-plans and commits one layer at a time, every ``refresh_interval``
    representative batches per layer."""

    def __init__(
        self,
        *,
        moe_load_balancer,
        get_model: Callable[[], Any],
        get_expert_location_updater: Callable[[], Any],
        refresh_interval: int,
    ):
        from evenmoe.policies.l1 import RefreshGate

        self._moe_load_balancer = moe_load_balancer
        # The model and the updater exist only after ModelRunner builds them.
        self._get_model = get_model
        self._get_expert_location_updater = get_expert_location_updater
        self._refresh_gate = RefreshGate(refresh_interval)
        self._forward_batch = None

    @contextmanager
    def forward_scope(self, forward_batch):
        """Expose the running batch to refresh_layer, which TopK calls without it."""
        self._forward_batch = forward_batch
        try:
            yield
        finally:
            self._forward_batch = None

    def refresh_layer(self, layer_id, logical_topk_ids):
        """Re-plan one layer from its current TopK and commit weights + placement
        before that layer dispatches."""
        if not self._should_refresh_layer(layer_id):
            return

        from evenmoe.adapters.sglang import commit_placement, plan_layer_from_topk

        # EvenMoE gathers every rank's counts over the EP group and plans the layer.
        maps = plan_layer_from_topk(
            self._moe_load_balancer, get_context(), layer_id, logical_topk_ids
        )
        layer_metadata = ExpertLocationMetadata._init_raw(
            ep_size=get_global_expert_location_metadata().ep_size,
            physical_to_logical_map=maps.physical_to_logical_map,
            logical_to_all_physical_map=maps.logical_to_all_physical_map,
            mlb_routing_metadata=maps.routing_metadata,
        )
        # Move this layer's expert weights against the still-live placement.
        self._get_expert_location_updater().update_layer(
            self._get_model().routed_experts_weights_of_layer[layer_id],
            layer_metadata,
            layer_id,
            nnodes=get_parallel().nnodes,
            rank=get_parallel().tp_rank,
        )
        # Publish the new placement so this layer's routing reads it.
        commit_placement(self._moe_load_balancer, get_context(), [layer_id])

    def _should_refresh_layer(self, layer_id) -> bool:
        """Whether this layer re-plans on this pass. Every EP rank must reach the
        same answer, since the refresh runs collectives, so only global batch
        facts decide it."""
        forward_batch = self._forward_batch
        if forward_batch is None or torch.cuda.is_current_stream_capturing():
            return False
        mode = forward_batch.global_forward_mode or forward_batch.forward_mode
        # Decode-only passes (idle DP ranks included) do not count as batches.
        if (
            mode.is_decode() or mode.is_idle()
        ) and not forward_batch.is_extend_in_batch:
            return False
        if not self._refresh_gate.is_due(layer_id):
            return False
        # DP batches list every rank's count (prefer the one before speculative
        # scaling); without DP the batch's real-token count is already global.
        counts = forward_batch.original_global_num_tokens_cpu
        if counts is None:
            counts = forward_batch.global_num_tokens_cpu
        num_tokens = (
            sum(counts)
            if counts is not None
            else forward_batch.global_num_token_non_padded_cpu
        )
        return self._refresh_gate.should_refresh(
            layer_id, representative=num_tokens >= _MIN_REPRESENTATIVE_TOKENS
        )
