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
    """Kept for the last ``window`` tokens of a request: the window its layers
    attend over; ``None`` when the model does not state its window."""

    window: Optional[int] = None
    rank: ClassVar[int] = 1

    def retention(self) -> Tuple[int, float]:
        return (self.rank, math.inf if self.window is None else self.window)


class StateLifetime(msgspec.Struct, frozen=True, tag="state"):
    """One fixed-size slot per request, overwritten as the request advances.
    A copy of it is valid only after a multiple of ``checkpoint_interval``
    tokens (the state kernel's chunk); ``None`` when no prefix is cached."""

    checkpoint_interval: Optional[int] = None
    rank: ClassVar[int] = 0

    def retention(self) -> Tuple[int, float]:
        return (self.rank, 0)


KVLifetime = Union[FullLifetime, WindowLifetime, StateLifetime]


def declares_lifetime_beside_full(model_config) -> bool:
    """Whether a model's layers declare a sliding window or recurrent state
    beside full attention. Asks for the window first: argument resolution has
    not yet published what some state configs read to describe their state."""
    from sglang.srt.configs.hybrid_arch import mambaish_config

    return model_config.is_hybrid_swa or mambaish_config(model_config) is not None


def declared_lifetime_types(model_config) -> frozenset:
    """The lifetimes a model's layers declare: full attention always, sliding
    windows and recurrent state when the model has them."""
    from sglang.srt.configs.hybrid_arch import mambaish_config

    types = {FullLifetime}
    if model_config.is_hybrid_swa:
        types.add(WindowLifetime)
    if mambaish_config(model_config) is not None:
        types.add(StateLifetime)
    return frozenset(types)


class MHAGeometry(msgspec.Struct, frozen=True, kw_only=True):
    head_num: int  # per rank
    head_dim: int
    v_head_dim: int
    kv_cache_dtype: torch.dtype


class MLAGeometry(msgspec.Struct, frozen=True, kw_only=True):
    kv_lora_rank: int
    qk_rope_head_dim: int
    kv_cache_dtype: torch.dtype


class StateReplay(msgspec.Struct, frozen=True, kw_only=True):
    decode: bool
    speculative: bool
    is_kda: bool
    cache_len: int
    key_heads: int


class StateGeometry(msgspec.Struct, frozen=True, kw_only=True):
    conv_state_shapes: Tuple[Tuple[int, ...], ...]
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype
    conv_slice_axis: int = 0
    extra_parts: Tuple["TensorPartDecl", ...] = ()
    request_parts: Tuple["TensorPartDecl", ...] = ()
    scratch_parts: Tuple["TensorPartDecl", ...] = ()
    replay: Optional[StateReplay] = None

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
    checkpoint_interval: Optional[int] = None,
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
        part = KVPartDecl(
            lifetime=StateLifetime(checkpoint_interval=checkpoint_interval),
            geometry=state_geometry,
        )
        for layer_id in state_layer_ids:
            parts.setdefault(layer_id, []).append(part)
    return tuple(
        KVLayerDecl(layer_id=layer_id, parts=tuple(parts[layer_id]))
        for layer_id in sorted(parts)
    )


class RequestLifetime(msgspec.Struct, frozen=True, tag="request"):
    """A row owned by a request, reset when the row is acquired again."""


class TensorPartDecl(msgspec.Struct, frozen=True, kw_only=True):
    """One typed payload per id, optionally repeated over model layers.

    ``shape`` excludes the id and layer axes. Layered tensors have shape
    [layer, id, *shape]; request-wide tensors have shape [id, *shape].
    ``slice_axis`` names the TP axis within shape, or None for replicated data.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    layer_ids: Tuple[int, ...] = ()
    reset_value: Union[int, float] = 0
    slice_axis: Optional[int] = None

    def __post_init__(self):
        if not self.name or any(n < 0 for n in self.shape):
            raise ValueError(f"invalid tensor part {self.name!r}: {self.shape}")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError(f"duplicate layers in tensor part {self.name!r}")
        if self.slice_axis is not None and not 0 <= self.slice_axis < len(self.shape):
            raise ValueError(f"invalid TP axis for tensor part {self.name!r}")

    @property
    def row_axis(self) -> int:
        return int(bool(self.layer_ids))

    @property
    def row_bytes(self) -> int:
        return math.prod(self.shape) * max(1, len(self.layer_ids)) * self.dtype.itemsize

    def tensor_shape(self, rows: int) -> Tuple[int, ...]:
        prefix = (len(self.layer_ids),) if self.layer_ids else ()
        return (*prefix, rows, *self.shape)


class ScratchDecl(msgspec.Struct, frozen=True, kw_only=True):
    """Captured workspace with one row per active request, outside the chain."""

    part: TensorPartDecl


class RequestPartDecl(msgspec.Struct, frozen=True, kw_only=True):
    """Persistent request-row payload, including preallocated transfer rows."""

    part: TensorPartDecl
    lifetime: RequestLifetime = RequestLifetime()


class DerivedPartDecl(msgspec.Struct, frozen=True, kw_only=True):
    """Contiguous payload per physical page of a sub-pool.

    Its shape describes one complete page blob; it is not interleaved with
    token entries. Relocation and reset follow the owning physical page.
    """

    sub_pool: str
    part: TensorPartDecl


def with_state_replay(
    geometry: StateGeometry,
    replay: StateReplay,
    *,
    layer_ids: Tuple[int, ...],
    speculative_num_draft_tokens: Optional[int],
) -> StateGeometry:
    """Declare ReplaySSM records in the coordinate space their kernels index."""
    if not (replay.decode or replay.speculative):
        return geometry
    hv, value_dim, key_dim = geometry.temporal_state_shape
    length = replay.cache_len
    if length <= 0:
        raise ValueError("ReplaySSM ring length must be positive")
    if (
        replay.speculative
        and replay.is_kda
        and (
            length & (length - 1)
            or (
                speculative_num_draft_tokens is not None
                and length < 2 * speculative_num_draft_tokens
            )
        )
    ):
        raise ValueError(
            "KDA speculative ring needs a power-of-two length and room for two verify windows"
        )
    dtype = geometry.conv_dtype if replay.speculative else geometry.temporal_dtype
    ring = [
        TensorPartDecl(
            name="replayssm_d",
            shape=(hv, length, value_dim),
            dtype=dtype,
            layer_ids=layer_ids,
            slice_axis=0,
        ),
        TensorPartDecl(
            name="replayssm_k",
            shape=(replay.key_heads, length, key_dim),
            dtype=dtype,
            layer_ids=layer_ids,
            slice_axis=0,
        ),
        TensorPartDecl(
            name="replayssm_g",
            shape=(hv, length, key_dim) if replay.is_kda else (hv, length),
            dtype=torch.float32,
            layer_ids=layer_ids,
            slice_axis=0,
        ),
    ]
    if replay.speculative and (replay.is_kda or dtype != torch.float32):
        ring.extend(
            (
                TensorPartDecl(
                    name="replayssm_rawv",
                    shape=(hv, length, value_dim),
                    dtype=geometry.conv_dtype,
                    layer_ids=layer_ids,
                    slice_axis=0,
                ),
                TensorPartDecl(
                    name="replayssm_rawk",
                    shape=(replay.key_heads, length, key_dim),
                    dtype=geometry.conv_dtype,
                    layer_ids=layer_ids,
                    slice_axis=0,
                ),
            )
        )
    if replay.speculative and replay.is_kda:
        ring.append(
            TensorPartDecl(
                name="replayssm_beta",
                shape=(hv, length),
                dtype=torch.float32,
                layer_ids=layer_ids,
                slice_axis=0,
            )
        )
    extras, requests = list(geometry.extra_parts), list(geometry.request_parts)
    if replay.speculative and not replay.is_kda:
        requests.extend(ring)
        requests.extend(
            TensorPartDecl(name=name, shape=(), dtype=dtype)
            for name, dtype in (
                ("replayssm_spec_write_pos", torch.int32),
                ("replayssm_cache_base", torch.int32),
                ("replayssm_is_flush", torch.int8),
            )
        )
    else:
        extras.extend(ring)
    if replay.decode:
        extras.append(
            TensorPartDecl(name="replayssm_write_pos", shape=(), dtype=torch.int32)
        )
    return msgspec.structs.replace(
        geometry,
        replay=replay,
        extra_parts=tuple(extras),
        request_parts=tuple(requests),
    )


def with_state_ple(
    geometry: StateGeometry,
    *,
    short_conv_layer_ids=(),
    short_conv_state_shape=None,
    ngram_context_len=0,
    ngram_eos_token_id=0,
    speculative_num_draft_tokens=None,
) -> StateGeometry:
    parts = []
    if short_conv_layer_ids and short_conv_state_shape is not None:
        parts.append(
            TensorPartDecl(
                name="short_conv",
                shape=tuple(short_conv_state_shape),
                dtype=geometry.conv_dtype,
                layer_ids=tuple(short_conv_layer_ids),
            )
        )
    if ngram_context_len:
        parts.append(
            TensorPartDecl(
                name="ngram",
                shape=(ngram_context_len,),
                dtype=torch.int64,
                reset_value=ngram_eos_token_id,
            )
        )
    scratch = (
        ()
        if speculative_num_draft_tokens is None
        else tuple(
            msgspec.structs.replace(
                part,
                name=f"intermediate_{part.name}",
                shape=(speculative_num_draft_tokens, *part.shape),
            )
            for part in parts
        )
    )
    return msgspec.structs.replace(
        geometry,
        extra_parts=(*geometry.extra_parts, *parts),
        scratch_parts=(*geometry.scratch_parts, *scratch),
    )
