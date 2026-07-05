"""Model cache layout helpers for the MLX backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class MlxModelCacheLayout:
    """Map model layers to MLX cache storage components.

    Full-attention layers store softmax-attention KV in the shared MLX
    attention KV pool.  Sliding-window attention layers keep per-request
    windowed KV only (no pool storage; SWA prefix hits recompute the
    prefix).  Auxiliary layers keep native ``mlx-lm`` cache state and are
    snapshotted by the MLX auxiliary-state component.

    ``attention_pool_index_by_layer`` is the dense index over *all*
    attention layers, used to address per-request attention cache arrays
    during batched decode.  ``full_kv_pool_index_by_layer`` is the dense
    index over full-attention layers only, used to address the shared KV
    pool's per-layer buffers.
    """

    layers: tuple[Any, ...]
    attention_attrs: tuple[str | None, ...]
    attention_layer_indices: tuple[int, ...]
    auxiliary_layer_indices: tuple[int, ...]
    attention_pool_index_by_layer: dict[int, int]
    # Per-layer sliding window (None or absent = full attention).
    layer_window_sizes: dict[int, int | None] = field(default_factory=dict)
    # Derived from layer_window_sizes; consistent for every construction path.
    full_attention_layer_indices: tuple[int, ...] = field(init=False)
    swa_attention_layer_indices: tuple[int, ...] = field(init=False)
    full_kv_pool_index_by_layer: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        full_indices = tuple(
            idx
            for idx in self.attention_layer_indices
            if self.layer_window_sizes.get(idx) is None
        )
        swa_indices = tuple(
            idx
            for idx in self.attention_layer_indices
            if self.layer_window_sizes.get(idx) is not None
        )
        object.__setattr__(self, "full_attention_layer_indices", full_indices)
        object.__setattr__(self, "swa_attention_layer_indices", swa_indices)
        object.__setattr__(
            self,
            "full_kv_pool_index_by_layer",
            {layer_idx: pool_idx for pool_idx, layer_idx in enumerate(full_indices)},
        )

    @classmethod
    def from_attention_discovery(
        cls,
        layers: Sequence[Any],
        attention_attrs: Sequence[str | None],
        layer_window_sizes: dict[int, int | None] | None = None,
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
            layer_window_sizes=dict(layer_window_sizes or {}),
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_attention_layers(self) -> int:
        return len(self.attention_layer_indices)

    @property
    def num_full_attention_layers(self) -> int:
        return len(self.full_attention_layer_indices)

    @property
    def max_swa_window(self) -> int | None:
        """Largest sliding window across SWA layers, or None without SWA."""
        windows = [
            self.layer_window_sizes[idx] for idx in self.swa_attention_layer_indices
        ]
        return max(windows) if windows else None

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

    def full_kv_pool_index(self, layer_idx: int) -> int:
        try:
            return self.full_kv_pool_index_by_layer[layer_idx]
        except KeyError as exc:
            raise KeyError(f"Layer {layer_idx} is not a full-attention layer") from exc

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
