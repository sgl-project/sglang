"""Routes a model layer to the sub-pool view that stores its KV.

Code outside the KV package reaches a layer's storage through a router: which
sub-pool and part hold it (`route`), the view and the layer id that view's
methods take (`layer`, or `read`, which first waits for a layer-wise load-back),
and the pool's token KV sub-pools in plan order (`sub_pools`). Parts other than
token KV, such as a request's recurrent state, are bound after construction
(`bind_part`) and reached by naming the part. Routes and sub-pools carry names,
never a lifetime.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import msgspec

# The part every token KV view holds today; derived parts get their own names.
KV_PART = "kv"
# A request's recurrent state, one slot per request, in the state sub-pool.
STATE_PART = "state"


class PartRoute(msgspec.Struct, frozen=True, kw_only=True):
    """Where one layer's KV lives."""

    sub_pool: str
    part: str
    # 0-based position of the layer among the layers the part's view holds.
    index: int


class KVLayer(msgspec.Struct, frozen=True, kw_only=True):
    """Reads use ``KVLayerRouter.read`` to wait for layer-wise load-back;
    writes use ``KVLayerRouter.layer`` without waiting."""

    route: PartRoute
    view: Any  # a type from kv.memory.views for the layouts that have one
    # The id the view's per-layer methods take, exactly as the view's callers
    # pass it today: a composite's local index, or a plain pool's model layer id.
    view_layer_id: int
    # Whether the layer's sub-pool is the router's owner, whose write ids are the
    # request's token ids; another token sub-pool writes at the batch's `swa_loc`.
    owner: bool = True
    # A plain pool's own layer: the view is the pool, which takes the write loc
    # as the backend hands it and resolves it itself.
    plain: bool = False

    def key(self, scale: Optional[float] = None):
        if scale is None:
            return self.view.get_key_buffer(self.view_layer_id)
        return self.view.get_key_buffer(self.view_layer_id, scale)

    def value(self, scale: Optional[float] = None):
        if scale is None:
            return self.view.get_value_buffer(self.view_layer_id)
        return self.view.get_value_buffer(self.view_layer_id, scale)

    def kv(self):
        return self.view.get_kv_buffer(self.view_layer_id)

    def kv_scales(self):
        return self.view.get_kv_scale_buffer(self.view_layer_id)

    def store(self, layer, loc_info, cache_k, cache_v, *args, **kwargs) -> None:
        """Writes the layer's K and V at the ids of its sub-pool: the owner's
        (`full_loc`, else `loc`) or the window's (`swa_loc`), as the attention
        backend bound them for the batch."""
        if self.plain:
            self.view.set_kv_buffer(layer, loc_info, cache_k, cache_v, *args, **kwargs)
            return
        from sglang.srt.mem_cache.kv.memory.views import MLAKVView

        write_loc = self._write_loc(layer, loc_info)
        if isinstance(self.view, MLAKVView):
            self.view.set_kv_buffer(
                layer,
                write_loc,
                cache_k,
                cache_v,
                layer_id_override=self.view_layer_id,
            )
            return
        self.view.set_kv_buffer(
            layer,
            write_loc,
            cache_k,
            cache_v,
            *args,
            layer_id_override=self.view_layer_id,
            **kwargs,
        )

    def store_mla(self, layer, loc_info, cache_k_nope, cache_k_rope) -> None:
        if self.plain:
            self.view.set_mla_kv_buffer(layer, loc_info, cache_k_nope, cache_k_rope)
            return
        self.view.set_mla_kv_buffer(
            layer,
            self._write_loc(layer, loc_info),
            cache_k_nope,
            cache_k_rope,
            layer_id_override=self.view_layer_id,
        )

    def _write_loc(self, layer, loc_info):
        # Imported here: memory_pool imports this module for its routers.
        from sglang.srt.mem_cache.memory_pool import (
            KVWriteLoc,
            unwrap_write_loc,
            write_loc_is_physical,
        )

        loc, window_loc, owner_loc = unwrap_write_loc(loc_info)
        if self.owner:
            ids = owner_loc if owner_loc is not None else loc
        else:
            assert window_loc is not None, (
                f"layer {layer.layer_id} writes sub-pool {self.route.sub_pool!r}, "
                "but the attention backend bundled no swa_loc for it"
            )
            ids = window_loc
        return KVWriteLoc(ids, physical=write_loc_is_physical(loc_info))

    def read_mla(self, layer, loc, dst_dtype=None):
        """Read ids arrive translated at their production site."""
        if self.plain:
            return self.view.get_mla_kv_buffer(layer, loc, dst_dtype)
        return self.view.get_mla_kv_buffer(
            layer, loc, dst_dtype, layer_id_override=self.view_layer_id
        )


class PublicSubPool(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    # Model layer ids routed to this sub-pool, in the order the pool declares them.
    layer_ids: Tuple[int, ...]


class KVLayerRouter:
    """Routes the layers of one pool to its sub-pool views.

    Holds the pool for its layer-wise load-back counter, which a host transfer
    registers after construction, and its ``start_layer``. Token KV reads wait
    on that counter whenever it is set; reads of the bound parts wait on it only
    once the host stack restores them layer by layer too
    (`load_parts_layer_wise`)."""

    def __init__(
        self,
        pool,
        *,
        sub_pools: Tuple[PublicSubPool, ...],
        views: Dict[str, Any],
        layers: Dict[int, KVLayer],
    ):
        self._pool = pool
        self._sub_pools = sub_pools
        self._by_name = {sub_pool.name: sub_pool for sub_pool in sub_pools}
        self._views = views
        self._layers = layers
        self._part_layers: Dict[Tuple[int, str], KVLayer] = {}
        self._part_layer_ids: Dict[str, Tuple[int, ...]] = {}
        self._part_views: Dict[str, Any] = {}
        self._parts_load_layer_wise = False

    @classmethod
    def from_layer_maps(
        cls,
        pool,
        layer_maps: Sequence[Tuple[str, Any, Dict[int, int]]],
        *,
        owner: Optional[str] = None,
    ) -> KVLayerRouter:
        """A composite's router: per sub-pool, in plan order, its name, its
        view, and each routed model layer's index in that view, which is also
        the id the composite passes the view today. ``owner`` names the
        sub-pool whose write ids are the request's token ids (default: the
        first)."""
        owner = layer_maps[0][0] if owner is None else owner
        layers: Dict[int, KVLayer] = {}
        for name, view, indices in layer_maps:
            for layer_id, index in indices.items():
                layers[layer_id] = KVLayer(
                    route=PartRoute(sub_pool=name, part=KV_PART, index=index),
                    view=view,
                    view_layer_id=index,
                    owner=name == owner,
                )
        return cls(
            pool,
            sub_pools=tuple(
                PublicSubPool(name=name, layer_ids=tuple(indices))
                for name, _, indices in layer_maps
            ),
            views={name: view for name, view, _ in layer_maps},
            layers=layers,
        )

    def bind_part(
        self,
        *,
        sub_pool: str,
        part: str,
        view: Any,
        indices: Dict[int, int],
        view_layer_ids: Optional[Dict[int, int]] = None,
    ) -> None:
        """Routes a part other than token KV: each model layer's index in the
        part's view, and the id the view's methods take when it is not that
        index. A part held once per request rather than per layer is routed
        at the layer whose load-back restores it."""
        assert part != KV_PART and part not in self._part_layer_ids, part
        for layer_id, index in indices.items():
            self._part_layers[(layer_id, part)] = KVLayer(
                route=PartRoute(sub_pool=sub_pool, part=part, index=index),
                view=view,
                view_layer_id=(
                    index if view_layer_ids is None else view_layer_ids[layer_id]
                ),
            )
        self._part_layer_ids[part] = tuple(sorted(indices))
        self._part_views[part] = view

    def part_indices(self, part: str) -> Dict[int, int]:
        """Each routed layer's index in the bound part's view."""
        return {
            layer_id: self._part_layers[(layer_id, part)].route.index
            for layer_id in self._part_layer_ids.get(part, ())
        }

    def part_view(self, part: str) -> Optional[Any]:
        """The view a bound part's layers share, or None when no part of that
        name is bound."""
        return self._part_views.get(part)

    def load_parts_layer_wise(self) -> None:
        """The host stack restores the bound parts layer by layer, as it does
        token KV: from now on their reads wait on the load-back counter."""
        self._parts_load_layer_wise = True

    def route(self, layer_id: int, part: str = KV_PART) -> PartRoute:
        return self.layer(layer_id, part).route

    def layer(self, layer_id: int, part: str = KV_PART) -> KVLayer:
        """The layer's view, without waiting for a load-back: for writes and
        for what a backend caches at init."""
        if part != KV_PART:
            return self._part_layers[(layer_id, part)]
        return self._layers[layer_id]

    def read(self, layer_id: int, part: str = KV_PART) -> KVLayer:
        """`layer`, once a layer-wise host load-back has brought the layer in."""
        self._wait(layer_id, part)
        return self.layer(layer_id, part)

    def wait_for_part(self, part: str) -> None:
        """Waits until a load-back has brought in every layer of the part on
        this stage; layers land in order, so until its last. A part with no
        layer on this stage (a pipeline stage without state layers) waits for
        nothing."""
        layer_ids = self._part_layer_ids[part]
        if layer_ids:
            self._wait(layer_ids[-1], part)

    def _wait(self, layer_id: int, part: str) -> None:
        counter = self._pool.layer_transfer_counter
        if counter is not None and (part == KV_PART or self._parts_load_layer_wise):
            counter.wait_until(layer_id - self._pool.start_layer)

    def sub_pools(self) -> Tuple[PublicSubPool, ...]:
        """The token KV sub-pools; bound parts are not listed."""
        return self._sub_pools

    def sub_pool(self, name: str) -> Optional[PublicSubPool]:
        """The named sub-pool, or None when the pool has none by that name."""
        return self._by_name.get(name)

    def routes_to(self, sub_pool: str) -> bool:
        """Whether some layer's KV lives in the named sub-pool."""
        found = self.sub_pool(sub_pool)
        return found is not None and len(found.layer_ids) > 0

    def view(self, sub_pool: str) -> Any:
        return self._views[sub_pool]

    def wrapped_view(self) -> Optional[Any]:
        """The one token KV view the pool wraps, when all its token KV lives in
        a single sub-pool view other than the pool itself (its other layers
        hold none); None for a pool that is its own view or spreads its layers
        over several sub-pools."""
        if len(self.sub_pools()) != 1:
            return None
        view = self.view(self.sub_pools()[0].name)
        return None if view is self._pool else view

    def layer_ids(self) -> Tuple[int, ...]:
        """Every routed model layer id, ascending."""
        return tuple(sorted(lid for sp in self.sub_pools() for lid in sp.layer_ids))


class PlainPoolRouter(KVLayerRouter):
    """Lazily route model layer ids without requiring a layer count;
    the pool applies its own ``start_layer`` offset."""

    def __init__(self, pool, name: str):
        super().__init__(pool, sub_pools=(), views={name: pool}, layers={})
        self._name = name

    def layer(self, layer_id: int, part: str = KV_PART) -> KVLayer:
        if part != KV_PART:
            return super().layer(layer_id, part)
        kv = self._layers.get(layer_id)
        if kv is None:
            kv = self._layers[layer_id] = KVLayer(
                route=PartRoute(
                    sub_pool=self._name,
                    part=KV_PART,
                    index=layer_id - self._pool.start_layer,
                ),
                view=self._pool,
                view_layer_id=layer_id,
                plain=True,
            )
        return kv

    def sub_pools(self) -> Tuple[PublicSubPool, ...]:
        start = self._pool.start_layer
        return (
            PublicSubPool(
                name=self._name,
                layer_ids=tuple(range(start, start + self._pool.layer_num)),
            ),
        )

    def sub_pool(self, name: str) -> Optional[PublicSubPool]:
        return self.sub_pools()[0] if name == self._name else None
