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
    route: PartRoute
    view: Any  # a type from kv.memory.views for the layouts that have one
    # The id the view's per-layer methods take, exactly as the view's callers
    # pass it today: a composite's local index, or a plain pool's model layer id.
    view_layer_id: int


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
        cls, pool, layer_maps: Sequence[Tuple[str, Any, Dict[int, int]]]
    ) -> KVLayerRouter:
        """A composite's router: per sub-pool, in plan order, its name, its
        view, and each routed model layer's index in that view, which is also
        the id the composite passes the view today."""
        layers: Dict[int, KVLayer] = {}
        for name, view, indices in layer_maps:
            for layer_id, index in indices.items():
                layers[layer_id] = KVLayer(
                    route=PartRoute(sub_pool=name, part=KV_PART, index=index),
                    view=view,
                    view_layer_id=index,
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

    def layer_ids(self) -> Tuple[int, ...]:
        """Every routed model layer id, ascending."""
        return tuple(sorted(lid for sp in self.sub_pools() for lid in sp.layer_ids))


class PlainPoolRouter(KVLayerRouter):
    """A plain pool's router: one sub-pool whose view is the pool itself, which
    takes model layer ids and does its own ``start_layer`` arithmetic. Routes
    are built on first use, so the router needs no layer count."""

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
