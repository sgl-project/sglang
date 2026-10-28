"""Routes a model layer to the sub-pool view that stores its KV.

Code outside the KV package reaches a layer's storage through a router: which
sub-pool and part hold it (`route`), the view and the layer id that view's
methods take (`layer`, or `read`, which first waits for a layer-wise load-back),
and the pool's sub-pools in plan order (`sub_pools`). Routes and sub-pools carry
names, never a lifetime.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import msgspec

# The part every token KV view holds today; derived parts get their own names.
KV_PART = "kv"


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
    registers after construction, and its ``start_layer``."""

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

    def route(self, layer_id: int) -> PartRoute:
        return self.layer(layer_id).route

    def layer(self, layer_id: int) -> KVLayer:
        """The layer's view, without waiting for a load-back: for writes and
        for what a backend caches at init."""
        return self._layers[layer_id]

    def read(self, layer_id: int) -> KVLayer:
        """`layer`, once a layer-wise host load-back has brought the layer in."""
        counter = self._pool.layer_transfer_counter
        if counter is not None:
            counter.wait_until(layer_id - self._pool.start_layer)
        return self.layer(layer_id)

    def sub_pools(self) -> Tuple[PublicSubPool, ...]:
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

    def layer(self, layer_id: int) -> KVLayer:
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
