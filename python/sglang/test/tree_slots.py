"""Prefix-tree slot fixtures for caches built without KV storage."""

from typing import Iterable, Optional, Tuple

from sglang.srt.mem_cache.kv.plan import (
    TREE_BLOCK,
    TREE_STATE,
    TREE_TOKEN,
    TREE_WINDOW,
    TreeSlot,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

# A separate family keeps the auxiliary window outside storage-window rules;
# tests supply its driver through component_registry_override.
CUSTOM_WINDOW = "custom_window"

FAMILY_OF = {
    ComponentType.FULL: TREE_TOKEN,
    ComponentType.SWA: TREE_WINDOW,
    ComponentType.AUXILIARY_SWA: CUSTOM_WINDOW,
    ComponentType.MAMBA: TREE_STATE,
    ComponentType.C128: TREE_BLOCK,
}


def tree_slots_of(
    components: Iterable[ComponentType],
    *,
    window: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
) -> Tuple[TreeSlot, ...]:
    """State checkpoints default to the runtime chunk size, as in production."""
    slots = []
    for c in components:
        family = FAMILY_OF[ComponentType(c)]
        numbers = {}
        if family in (TREE_WINDOW, CUSTOM_WINDOW):
            numbers["window"] = window
        if family == TREE_STATE:
            if checkpoint_interval is None:
                from sglang.srt.runtime_context import mamba_cache_chunk_size

                checkpoint_interval = mamba_cache_chunk_size()
            numbers["checkpoint_interval"] = checkpoint_interval
        slots.append(
            TreeSlot(component=ComponentType(c).name, family=family, **numbers)
        )
    return tuple(slots)


def slot_of(params, component: ComponentType) -> TreeSlot:
    name = ComponentType(component).name
    return next(slot for slot in params.tree_slots if slot.component == name)


def index_slots(cache, components: Iterable[ComponentType]) -> None:
    """Initialize only the family lookup for caches built with ``__new__``."""
    cache._index_slots(
        tuple(
            TreeSlot(
                component=ComponentType(c).name, family=FAMILY_OF[ComponentType(c)]
            )
            for c in components
        )
    )
