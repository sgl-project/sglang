"""What each layer of a model stores in the unified KV pool, declared once.

A layer declares parts; each part has a geometry (its byte layout) and a
lifetime (who owns its ids and when they are freed). Parts with the same
lifetime and allocation unit share one sub-pool.
"""

import math
from typing import ClassVar, Optional, Sequence, Tuple, Union

import msgspec
import torch


class FullLifetime(msgspec.Struct, frozen=True, tag="full"):
    """Kept for the whole request; owns the token id space."""

    rank: ClassVar[int] = 2

    def retention(self) -> Tuple[int, float]:
        return (self.rank, math.inf)


class WindowLifetime(msgspec.Struct, frozen=True, tag="window"):
    """Kept for the last ``window`` tokens of a request; ``None`` when the
    model does not state its window."""

    window: Optional[int] = None
    rank: ClassVar[int] = 1

    def retention(self) -> Tuple[int, float]:
        return (self.rank, math.inf if self.window is None else self.window)


class StateLifetime(msgspec.Struct, frozen=True, tag="state"):
    """One fixed-size slot per request, overwritten as the request advances."""

    rank: ClassVar[int] = 0

    def retention(self) -> Tuple[int, float]:
        return (self.rank, 0)


KVLifetime = Union[FullLifetime, WindowLifetime, StateLifetime]


class MHAGeometry(msgspec.Struct, frozen=True, kw_only=True):
    head_num: int  # per rank
    head_dim: int
    v_head_dim: int
    kv_cache_dtype: torch.dtype


class MLAGeometry(msgspec.Struct, frozen=True, kw_only=True):
    kv_lora_rank: int
    qk_rope_head_dim: int
    kv_cache_dtype: torch.dtype


class StateGeometry(msgspec.Struct, frozen=True, kw_only=True):
    conv_state_shapes: Tuple[Tuple[int, ...], ...]
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype
    conv_slice_axis: int = 0

    @classmethod
    def of(cls, cache_params) -> "StateGeometry":
        """The state entry a model's cache params describe."""
        shape = cache_params.shape
        return cls(
            conv_state_shapes=tuple(tuple(int(x) for x in s) for s in shape.conv),
            conv_dtype=cache_params.dtype.conv,
            temporal_state_shape=tuple(int(x) for x in shape.temporal),
            temporal_dtype=cache_params.dtype.temporal,
            # Only some state shapes carry a slice axis; the rest slice axis 0.
            conv_slice_axis=getattr(shape, "conv_slice_axis", 0),
        )


PartGeometry = Union[MHAGeometry, MLAGeometry, StateGeometry]


class KVPartDecl(msgspec.Struct, frozen=True, kw_only=True):
    lifetime: KVLifetime
    geometry: PartGeometry


class KVLayerDecl(msgspec.Struct, frozen=True, kw_only=True):
    layer_id: int
    parts: Tuple[KVPartDecl, ...]


def kv_layer_decls(
    *,
    full_layer_ids: Sequence[int],
    full_geometry: Union[MHAGeometry, MLAGeometry],
    window_layer_ids: Sequence[int] = (),
    window_geometry: Optional[MHAGeometry] = None,
    window: Optional[int] = None,
    state_layer_ids: Sequence[int] = (),
    state_geometry: Optional[StateGeometry] = None,
) -> Tuple[KVLayerDecl, ...]:
    """One declaration per layer that stores anything, in layer order. A layer
    may hold an attention part and a state part at once."""
    assert not set(full_layer_ids) & set(window_layer_ids), (
        "a layer attends either fully or through a window, not both"
    )
    parts = {}
    full = KVPartDecl(lifetime=FullLifetime(), geometry=full_geometry)
    for layer_id in full_layer_ids:
        parts.setdefault(layer_id, []).append(full)
    if window_layer_ids:
        assert window_geometry is not None, "window layers need their geometry"
        part = KVPartDecl(
            lifetime=WindowLifetime(window=window), geometry=window_geometry
        )
        for layer_id in window_layer_ids:
            parts.setdefault(layer_id, []).append(part)
    if state_layer_ids:
        assert state_geometry is not None, "state layers need their geometry"
        part = KVPartDecl(lifetime=StateLifetime(), geometry=state_geometry)
        for layer_id in state_layer_ids:
            parts.setdefault(layer_id, []).append(part)
    return tuple(
        KVLayerDecl(layer_id=layer_id, parts=tuple(parts[layer_id]))
        for layer_id in sorted(parts)
    )
