# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# ModelSpec -- a model's declaration of the structural seams it exposes.
#
# This is the "type" of a model in the efficiency framework. A new model is
# adapted by writing ONE small spec (capabilities + a few seam accessors), the
# same way cache-dit models register a BlockAdapter pointing at their block
# list. compose() type-checks a technique's required_capabilities against the
# spec's capabilities; if the model does not provide a seam, the technique is
# refused.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sglang.multimodal_gen.runtime.efficiency.technique import Capability


@dataclass
class ModelSpec:
    """Declares the seams a model exposes so generic techniques can plug in.

    Seam accessors are optional; the presence of a capability in
    ``capabilities`` is the contract that the corresponding accessor is set.

    Accessors
    ---------
    get_blocks(transformer) -> iterable of transformer blocks (for BLOCKS).
    prunable_segment(hidden, ctx) -> (start, end) along ``seq_dim`` marking the
        prunable token span (for PRUNABLE_TOKENS); default = whole sequence.
    """

    name: str
    capabilities: frozenset[Capability] = frozenset()

    # seam accessors
    get_blocks: Optional[Callable[[Any], Any]] = None
    prunable_segment: Optional[Callable[[Any, Any], tuple[int, int]]] = None
    seq_dim: int = 1

    # token-prune gather/scatter (model-specific I/O; defaults handle the simple
    # [B, S, C] hidden case). A model whose pruned forward needs MORE than the
    # hidden sliced -- e.g. per-token coords/timestep/masks -- provides these
    # so TokenPrune stays generic.
    #   prune_gather(payload, keep_idx, ctx) -> pruned_payload
    #   prune_scatter(output, keep_idx, full_len, ctx, compensation) -> full_output
    prune_gather: Optional[Callable] = None
    prune_scatter: Optional[Callable] = None

    # sequence-parallel behavior: when True, token-prune selects per-rank-local
    # indices (keeps USP all-to-all shards balanced) instead of a global top-K.
    sp_local_prune: bool = False

    extra: dict = field(default_factory=dict)

    def has(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def missing(self, required: frozenset[Capability]) -> frozenset[Capability]:
        return frozenset(required) - self.capabilities

    def segment(self, hidden, ctx) -> tuple[int, int]:
        """Resolve the prunable token span; default = the whole sequence."""
        if self.prunable_segment is not None:
            return self.prunable_segment(hidden, ctx)
        n = hidden.shape[self.seq_dim]
        return (0, n)
