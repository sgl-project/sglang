"""Prefix-tree slots for tests that build a cache by hand.

Production takes a cache's slots from its KV storage, whose plan gives each
sub-pool's family by its lifetime. A hand-built test names component ids
instead; ``FAMILY_OF`` gives each id the family production gives its sub-pool.
"""

from typing import Iterable, Tuple

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


def tree_slots_of(components: Iterable[ComponentType]) -> Tuple[TreeSlot, ...]:
    """The slots of a tree whose components are ``components``, in order."""
    return tuple(
        TreeSlot(component=ComponentType(c).name, family=FAMILY_OF[ComponentType(c)])
        for c in components
    )
