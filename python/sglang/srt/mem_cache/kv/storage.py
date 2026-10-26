"""The unified pool as host and peer transfers see it.

`KVStorage` is what HiCache and PD read: the sub-pools in the order host pools
and transfers list them, each with its external names, its layout view, its
model layers, the device memory a peer registers (`regions`), and how a host
load takes and returns device room. Sub-pools carry names, never a lifetime;
code outside the KV package loops over them instead of naming a combination.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import msgspec

from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.memory.host import page_envelope_pair
from sglang.srt.mem_cache.kv.plan import KVStoragePlan, SubPoolExternal, WireEntry
from sglang.srt.mem_cache.kv.router import KVLayerRouter


class StorageSubPool(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    external: SubPoolExternal
    # Each routed model layer id -> its index in the sub-pool's view.
    layer_indices: Dict[int, int]
    # One slot per request rather than per token: hosts mirror it as state.
    holds_state: bool


class Region(msgspec.Struct, frozen=True, kw_only=True):
    """A sub-pool's device memory as a PD peer registers it: one entry per
    buffer, and per entry the slices a peer of another TP size reads. A whole
    page envelope has no slices: both peers must lay it out identically."""

    ptrs: Tuple[int, ...]
    lens: Tuple[int, ...]
    item_lens: Tuple[int, ...]
    dim_per_tensor: Tuple[int, ...] = ()
    conv_shard_groups: Tuple[Any, ...] = ()
    slice_outer_counts: Tuple[int, ...] = ()
    layer_ids: Tuple[int, ...] = ()
    # The token KV byte layout both peers must share, for views that have one.
    kv_layout: Optional[str] = None


class KVStorage:
    """The plan's sub-pools over one unified buffer, their views and the chain
    allocator that hands out their ids. Token KV views and layer indices come
    from the KV pool's router, the state view from the allocator's state."""

    def __init__(
        self,
        *,
        plan: KVStoragePlan,
        router: KVLayerRouter,
        allocator,
        start_layer: int,
    ):
        self._plan = plan
        self._allocator = allocator
        self.start_layer = start_layer
        state = allocator.state
        self._views: Dict[str, Any] = {}
        sub_pools = []
        for s in plan.owner_first():
            holds_state = isinstance(s.spec, MambaSubPoolSpec)
            if holds_state:
                self._views[s.name] = state.pool
                layer_indices = dict(state.layer_map)
            else:
                self._views[s.name] = router.view(s.name)
                layer_indices = {
                    layer_id: router.route(layer_id).index
                    for layer_id in router.sub_pool(s.name).layer_ids
                }
            sub_pools.append(
                StorageSubPool(
                    name=s.name,
                    external=s.external,
                    layer_indices=layer_indices,
                    holds_state=holds_state,
                )
            )
        self._sub_pools = tuple(sub_pools)

    @property
    def owner(self) -> str:
        """The sub-pool that owns the token id space: the anchor of host
        stacks, and the transfer's KV entries."""
        return self._plan.owner().name

    def sub_pools(self) -> Tuple[StorageSubPool, ...]:
        """Every sub-pool, owner first."""
        return self._sub_pools

    def view(self, sub_pool: str) -> Any:
        return self._views[sub_pool]

    def page_envelope_pair(self) -> Optional[Tuple[str, str]]:
        """The two sub-pools whose host pools may share one page-envelope
        backing, owner first (kv/memory/host.py)."""
        return page_envelope_pair(self._plan)

    def wire_entries(self) -> Tuple[WireEntry, ...]:
        return self._plan.wire_entries()

    def regions(self, sub_pool: str) -> Region:
        view = self._views[sub_pool]
        ptrs, lens, item_lens = view.get_contiguous_buf_infos()
        if isinstance(self._plan.sub_pool(sub_pool).spec, MambaSubPoolSpec):
            return Region(
                ptrs=tuple(ptrs),
                lens=tuple(lens),
                item_lens=tuple(item_lens),
                dim_per_tensor=tuple(view.get_state_dim_per_tensor()),
                conv_shard_groups=tuple(view.get_state_conv_shard_groups()),
                slice_outer_counts=tuple(view.get_state_slice_outer_counts()),
                layer_ids=tuple(view.get_state_layer_ids()),
            )
        spec = self._plan.sub_pool(sub_pool).spec
        return Region(
            ptrs=tuple(ptrs),
            lens=tuple(lens),
            item_lens=tuple(item_lens),
            kv_layout=(
                view.kv_cache_layout if isinstance(spec, MHASubPoolSpec) else None
            ),
        )

    def device_room(self, sub_pool: str) -> int:
        """Free units a host load can take in the sub-pool now."""
        if sub_pool == self._allocator.roles.window:
            return self._allocator.swa_available_size()
        assert sub_pool == self._allocator.roles.state, sub_pool
        return self._allocator.state_slots.schedulable_available_size()

    def host_load_alloc(
        self, sub_pool: str
    ) -> Optional[Tuple[Callable[[int], Any], Callable[[Any], Any]]]:
        """How a host load takes device units in the sub-pool and gives back
        the ones it did not use, or None when loads take the owner's token
        ids. A window binds physical pages before the copy."""
        roles = self._allocator.roles
        if sub_pool == roles.window:
            window = self._allocator.swa_attn_allocator
            return window.alloc_physical, window.cancel_physical_reservation
        if sub_pool == roles.state:
            slots = self._allocator.state_slots
            return slots.alloc, slots.free
        return None
