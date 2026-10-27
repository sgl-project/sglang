"""Prefix-tree slots for tests that build a cache by hand.

Production takes a cache's slots from its KV storage, whose plan gives each
sub-pool's family by its lifetime. A hand-built test names component ids
instead; ``FAMILY_OF`` gives each id the family production gives its sub-pool.
"""

from typing import Iterable, Optional, Tuple

from sglang.srt.mem_cache.kv.plan import (
    TREE_BLOCK,
    TREE_STATE,
    TREE_TOKEN,
    TREE_WINDOW,
    TreeSlot,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

# A tree-local window component (buffer mode's auxiliary window) is no slot of
# the KV storage's: tests give it a family of its own and its class through
# ``component_registry_override``, so the cache's window rules leave it alone.
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
    """The slots of a tree whose components are ``components``, in order. A
    window slot keeps ``window`` tokens (the test's retained window); a state
    slot's checkpoint interval defaults to the runtime's chunk size, where
    production's plan takes it from."""
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
    """The slot ``params`` gives ``component``, for a test that builds a
    component directly."""
    name = ComponentType(component).name
    return next(slot for slot in params.tree_slots if slot.component == name)


def index_slots(cache, components: Iterable[ComponentType]) -> None:
    """Give a hand-built cache (``object.__new__(UnifiedRadixCache)``) the
    family-to-slot table its ``__init__`` builds from its slots, for a tree
    whose components are ``components``. The table names slots only, so the
    slots carry no numbers."""
    cache._index_slots(
        tuple(
            TreeSlot(
                component=ComponentType(c).name, family=FAMILY_OF[ComponentType(c)]
            )
            for c in components
        )
    )
