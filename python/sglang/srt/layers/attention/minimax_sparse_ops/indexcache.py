"""Pure decision logic for MiniMax-M3 decode IndexCache.

IndexCache reuses the sparse block selection (``topk_idx``) across sparse layers
during decode: the lightning indexer is recomputed only on 1 of every ``stride``
sparse layers (the "cadence" layers) and the last selection is reused on the
other ``stride - 1``.

The helpers here are deliberately dependency-free (no torch / triton) so the
cadence policy can be unit-tested on CPU without importing the attention
backend. ``MiniMaxSparseAttnBackend`` imports and drives them at runtime.
"""

from typing import Dict, Iterable


def indexcache_enabled(stride: int) -> bool:
    """Whether IndexCache reuse is active for ``stride``.

    ``stride`` of 0 or 1 means OFF (recompute the indexer on every sparse layer,
    the stock behavior); any value ``> 1`` enables reuse with that cadence.
    """
    return bool(stride and stride > 1)


def indexcache_layer_positions(sparse_layer_ids: Iterable[int]) -> Dict[int, int]:
    """Map each sparse layer id to its execution-order position.

    Positions are 0-based over the sorted sparse layer ids, so a layer is a
    cadence (recompute) layer iff ``position % stride == 0``.
    """
    return {lid: i for i, lid in enumerate(sorted(sparse_layer_ids))}


def indexcache_should_reuse(pos: int, stride: int, has_cached_state: bool) -> bool:
    """Whether the sparse layer at execution position ``pos`` should REUSE the
    cached block selection instead of recomputing the lightning indexer.

    Reuse happens only when IndexCache is enabled, the layer is not a cadence
    layer (``pos % stride != 0``), and a selection has already been cached this
    forward. The first sparse layer of a forward is always position 0 (a cadence
    layer) and has no cached state, so it recomputes -- which is what makes
    within-forward reuse CUDA-graph safe (cadence layers run before reuse ones).
    """
    if not indexcache_enabled(stride):
        return False
    if not has_cached_state:
        return False
    return pos % stride != 0
