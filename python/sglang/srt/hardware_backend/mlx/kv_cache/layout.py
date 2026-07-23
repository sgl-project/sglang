"""Model cache layout helpers for the MLX backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence


@dataclass(frozen=True)
class MlxModelCacheLayout:
    """Map model layers to MLX cache storage components.

    Attention layers store softmax-attention KV in the MLX attention KV pool.
    Auxiliary layers keep native ``mlx-lm`` cache state and are snapshotted by
    the MLX auxiliary-state component.
    """

    layers: tuple[Any, ...]
    attention_attrs: tuple[str | None, ...]
    attention_layer_indices: tuple[int, ...]
    auxiliary_layer_indices: tuple[int, ...]
    attention_pool_index_by_layer: dict[int, int]
    # Native Gemma 4 cache metadata. Logical YOCO-shared layers resolve to an
    # owning logical layer, and only owners receive compact cache-list slots.
    native_cache_owner_by_layer: tuple[int, ...] = ()
    native_cache_index_by_owner: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_attention_discovery(
        cls,
        layers: Sequence[Any],
        attention_attrs: Sequence[str | None],
    ) -> MlxModelCacheLayout:
        if len(layers) != len(attention_attrs):
            raise ValueError(
                "Layer count and attention attribute count differ: "
                f"{len(layers)} != {len(attention_attrs)}"
            )

        attention_layer_indices = tuple(
            idx for idx, attr in enumerate(attention_attrs) if attr is not None
        )
        auxiliary_layer_indices = tuple(
            idx for idx, attr in enumerate(attention_attrs) if attr is None
        )
        attention_pool_index_by_layer = {
            layer_idx: pool_idx
            for pool_idx, layer_idx in enumerate(attention_layer_indices)
        }

        return cls(
            layers=tuple(layers),
            attention_attrs=tuple(attention_attrs),
            attention_layer_indices=attention_layer_indices,
            auxiliary_layer_indices=auxiliary_layer_indices,
            attention_pool_index_by_layer=attention_pool_index_by_layer,
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_attention_layers(self) -> int:
        return len(self.attention_layer_indices)

    @property
    def has_auxiliary_state(self) -> bool:
        return bool(self.auxiliary_layer_indices)

    @property
    def first_attention_layer_index(self) -> int:
        if not self.attention_layer_indices:
            raise RuntimeError("MLX model has no supported attention layers")
        return self.attention_layer_indices[0]

    def attention_pool_index(self, layer_idx: int) -> int:
        try:
            return self.attention_pool_index_by_layer[layer_idx]
        except KeyError as exc:
            raise KeyError(f"Layer {layer_idx} is not an attention layer") from exc

    def attention_attr(self, layer_idx: int) -> str:
        attr = self.attention_attrs[layer_idx]
        if attr is None:
            raise KeyError(f"Layer {layer_idx} is not an attention layer")
        return attr

    def attention_layer_caches(
        self,
        caches_by_request: list[list[Any]],
    ) -> list[list[Any]]:
        """Return layer-major attention caches for batched decode."""
        return [
            [request_cache[layer_idx] for request_cache in caches_by_request]
            for layer_idx in self.attention_layer_indices
        ]

    def with_native_cache_owners(
        self,
        owner_by_layer: Sequence[int],
        cache_index_by_owner: dict[int, int],
    ) -> MlxModelCacheLayout:
        if len(owner_by_layer) != self.num_layers:
            raise ValueError(
                "native cache owner metadata must cover every logical layer"
            )
        return replace(
            self,
            native_cache_owner_by_layer=tuple(int(owner) for owner in owner_by_layer),
            native_cache_index_by_owner=dict(cache_index_by_owner),
        )

    def native_cache_index(self, logical_layer_idx: int) -> int:
        if not self.native_cache_owner_by_layer:
            raise RuntimeError("model layout has no native-cache owner metadata")
        try:
            owner = self.native_cache_owner_by_layer[logical_layer_idx]
            return self.native_cache_index_by_owner[owner]
        except (IndexError, KeyError) as exc:
            raise KeyError(
                f"logical layer {logical_layer_idx} has no compact native cache owner"
            ) from exc
