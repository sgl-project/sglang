"""Prefix-tree slot fixtures for caches built without KV storage."""

from typing import Iterable, Optional, Tuple

import msgspec

from sglang.srt.mem_cache.kv.plan import (
    TREE_BLOCK,
    TREE_STATE,
    TREE_TOKEN,
    TREE_WINDOW,
    TreeSlot,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

# A separate family keeps a second window outside storage-window rules; tests
# supply its driver through component_registry_override.
CUSTOM_WINDOW = "custom_window"
CUSTOM_WINDOW_COMPONENT = ComponentType.of("CUSTOM_WINDOW")

FAMILY_OF = {
    ComponentType.FULL: TREE_TOKEN,
    ComponentType.SWA: TREE_WINDOW,
    CUSTOM_WINDOW_COMPONENT: CUSTOM_WINDOW,
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


def storage_tree_slots(
    storage,
    *,
    window: Optional[int] = None,
    components: Optional[Iterable[ComponentType]] = None,
) -> Tuple[TreeSlot, ...]:
    """The slots a cache over unified ``storage`` keeps, as production builds
    them: the storage's own, a window slot named by its sub-pool. ``window``
    sets a window slot's length where the test chooses one; ``components``
    keeps only the slots of those components' families."""
    families = (
        None
        if components is None
        else {FAMILY_OF[ComponentType(c)] for c in components}
    )
    return tuple(
        (
            msgspec.structs.replace(slot, window=window)
            if window is not None and slot.family == TREE_WINDOW
            else slot
        )
        for slot in storage.tree_slots()
        if families is None or slot.family in families
    )


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


def only_window(cache) -> str:
    """The window sub-pool of a cache keeping one window (the static pools'
    name when the cache keeps no window slot)."""
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    names = [name for name, _ in cache.window_sub_pools()]
    if not names:
        return BaseSWAKVPool.WINDOW_SUB_POOL
    (name,) = names
    return name


def window_evictable(cache) -> int:
    """Evictable tokens of a one-window cache's window."""
    return cache.window_evictable_size(only_window(cache))


def window_protected(cache) -> int:
    """Locked tokens of a one-window cache's window."""
    return cache.window_protected_size(only_window(cache))


def core_window_evictable(core) -> int:
    """Evictable tokens of a tree core's window-family slot."""
    return core.component_evictable_size(core.slot_id(TREE_WINDOW))


def core_window_protected(core) -> int:
    """Locked tokens of a tree core's window-family slot."""
    return core.component_protected_size(core.slot_id(TREE_WINDOW))
