from __future__ import annotations


def resolve_layer_block_ends(block_layers: int, num_layers: int) -> tuple[int, ...]:
    """Exclusive stage-local ends, shared by graph capture and H2D submission."""
    return (*range(block_layers, num_layers, block_layers), num_layers)
