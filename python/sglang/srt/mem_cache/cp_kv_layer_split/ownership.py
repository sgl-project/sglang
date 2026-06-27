"""Model-agnostic ownership math for CP KV LayerSplit."""

from __future__ import annotations

from sglang.srt.layers.dp_attention import get_attention_cp_group


def layers_per_cp_rank(num_layers: int, cp_size: int) -> int:
    """Max local KV buffers per CP rank (balanced contiguous split)."""
    assert num_layers > 0 and cp_size > 0
    return (num_layers + cp_size - 1) // cp_size


def _kv_layer_range_for_cp_rank(
    cp_rank: int, cp_size: int, num_layers: int
) -> tuple[int, int]:
    """Balanced contiguous global layer range owned by one CP rank."""
    assert num_layers > 0 and cp_size > 0
    assert 0 <= cp_rank < cp_size

    base = num_layers // cp_size
    remainder = num_layers % cp_size
    start = cp_rank * base + min(cp_rank, remainder)
    end = start + base + (1 if cp_rank < remainder else 0)
    return start, end


def kv_layer_owner(layer_id: int, cp_size: int, num_layers: int) -> int:
    """CP rank that stores KV cache for global transformer layer ``layer_id``."""
    assert num_layers > 0 and cp_size > 0
    assert 0 <= layer_id < num_layers
    base = num_layers // cp_size
    remainder = num_layers % cp_size
    if base == 0:
        return layer_id
    long_layers = (base + 1) * remainder
    if layer_id < long_layers:
        return layer_id // (base + 1)
    return remainder + (layer_id - long_layers) // base


def owns_kv_layer(layer_id: int, cp_rank: int, cp_size: int, num_layers: int) -> bool:
    return kv_layer_owner(layer_id, cp_size, num_layers) == cp_rank


def kv_layer_owner_global_rank(layer_id: int, cp_size: int, num_layers: int) -> int:
    """Global process rank (within the attention-CP group) that owns KV for ``layer_id``."""
    owner_cp = kv_layer_owner(layer_id, cp_size, num_layers)
    return get_attention_cp_group().ranks[owner_cp]


def owned_kv_layer_range(
    cp_rank: int,
    cp_size: int,
    num_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
) -> tuple[int, int]:
    """Return the local PP slice of transformer layers owned by ``cp_rank``."""
    if start_layer >= end_layer_exclusive:
        return start_layer, start_layer

    owner_start, owner_end = _kv_layer_range_for_cp_rank(cp_rank, cp_size, num_layers)
    owned_start = max(start_layer, owner_start)
    owned_end = min(end_layer_exclusive, owner_end)
    if owned_start >= owned_end:
        return start_layer, start_layer
    return owned_start, owned_end


def num_owned_kv_layers(
    cp_rank: int,
    cp_size: int,
    num_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
) -> int:
    """Number of transformer layers whose KV cache is stored on ``cp_rank`` on this GPU."""
    owned_start, owned_end = owned_kv_layer_range(
        cp_rank, cp_size, num_layers, start_layer, end_layer_exclusive
    )
    return owned_end - owned_start


def build_owned_layer_local_index_map(
    cp_rank: int,
    cp_size: int,
    num_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
) -> dict[int, int]:
    """Map owned global layer id to local buffer index."""
    owned_start, owned_end = owned_kv_layer_range(
        cp_rank, cp_size, num_layers, start_layer, end_layer_exclusive
    )
    return {
        layer_id: layer_id - owned_start for layer_id in range(owned_start, owned_end)
    }
