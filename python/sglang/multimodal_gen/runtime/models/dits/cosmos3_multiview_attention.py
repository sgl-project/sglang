# SPDX-License-Identifier: Apache-2.0
"""Sparse multiview attention for the Cosmos3 Multiview-AV variant.

Inference-only implementation of the Multiview-AV visibility rules on top of
the public PyTorch FlexAttention API, with an optional FlashAttention-4 CuTe
backend (see ``cosmos3_multiview_fa4``). Ported from the vLLM-Omni reference
implementation; the training code is a behavioral oracle, not a source.

Every GEN token carries six fields: sample, frame, view, control-or-RGB,
text-or-vision, timestamp. Under the ``decomposed`` scope this checkpoint
family uses:

* every GEN query sees all real text (UND) keys;
* RGB queries see RGB keys of the same view at any frame and of every view at
  the same frame (or within ``decomposed_temporal_window_seconds``);
* RGB queries see control (WSM) keys of the same view only;
* control queries see control keys of the same view only, plus RGB keys of the
  same view when ``control_attends_sensor`` is set;
* padding is isolated through a ``sample_id == -1`` sentinel.

The mask is never materialized per token pair. Tokens are grouped into
semantic runs, the predicate is evaluated once per run pair, and the result is
projected onto the sparse block grid the attention kernel consumes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, MutableMapping, Sequence
from typing import Any, ClassVar, Literal

import msgspec
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

AttentionScope = Literal["all_views", "same_view", "decomposed"]

SPARSE_Q_BLOCK_SIZE = 64
SPARSE_KV_BLOCK_SIZE = 64

# The default SM100 BF16/FP16 head_dim=128 FlexAttention configuration is
# 128x64 with three stages and eight warps. The multiview mask_mod adds enough
# state that the default first exceeded shared-memory capacity and, after only
# shrinking BLOCK_N, produced an illegal access at launch. Use the smallest
# square tile supported by the forward autotuner, remove software pipelining,
# and keep TMA disabled. Aligning the sparse mask blocks with the compute tile
# also avoids sub-block address arithmetic in the generated kernel.
TRITON_Q_BLOCK_SIZE = 64
TRITON_KV_BLOCK_SIZE = 64
TRITON_NUM_STAGES = 1
TRITON_NUM_WARPS = 4

# FlashAttention-4 runs a fixed 128x128 forward tile on SM100 and stages two Q
# tiles per CTA whenever the query length exceeds one tile, so the sparse block
# map it consumes must be (2 * tile_m, tile_n). These are not tunable: the
# kernel derives the same numbers from its own heuristic and rejects metadata
# that disagrees.
FA4_SPARSE_Q_BLOCK_SIZE = 256
FA4_SPARSE_KV_BLOCK_SIZE = 128

_BACKEND_BLOCK_SIZES: dict[str, tuple[int, int]] = {
    "triton": (SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE),
    "fa4": (FA4_SPARSE_Q_BLOCK_SIZE, FA4_SPARSE_KV_BLOCK_SIZE),
}

# The UND (text) stream is padded to a fixed capacity rather than to the
# nearest block above each prompt's real length. A pad that tracks the prompt
# changes the packed key tensor's sequence dimension, and the flex kernel is
# compiled with dynamic=False, so every distinct prompt-length bucket would be
# a fresh recompile. Past Dynamo's recompile limit the frame falls back to
# eager FlexAttention, which cannot fit the dense score matrix at the released
# 11-view geometry. A fixed capacity gives one compiled kernel per process.
#
# Padding to the capacity is numerically free: the extra keys carry
# ``sample_id == -1`` and are excluded from every real query by the predicate.
#
# The default is the Cosmos3 prompt truncation cap plus the ``eos`` and
# ``vision_start`` framing tokens the tokenizer appends after truncating.
DEFAULT_MAX_UND_TOKENS = 4096 + 2

_VALID_ATTENTION_SCOPES = frozenset({"all_views", "same_view", "decomposed"})

MULTIVIEW_BACKENDS: tuple[str, ...] = tuple(sorted(_BACKEND_BLOCK_SIZES))


def validate_multiview_backend(backend: str) -> str:
    """Reject an unknown backend name (exposed so callers fail at load time)."""
    if backend not in _BACKEND_BLOCK_SIZES:
        raise ValueError(
            "Cosmos3 multiview attention backend must be one of "
            f"{list(MULTIVIEW_BACKENDS)}, got {backend!r}."
        )
    return backend


def _validate_attention_scope(attention_scope: str) -> AttentionScope:
    if attention_scope not in _VALID_ATTENTION_SCOPES:
        raise ValueError(
            "Cosmos3 multiview attention_scope must be one of "
            f"{sorted(_VALID_ATTENTION_SCOPES)}, got {attention_scope!r}."
        )
    return attention_scope  # type: ignore[return-value]


def _validate_positive_finite(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"Cosmos3 multiview {name} must be finite and positive, got {value!r}."
        )


def _validate_temporal_window(value: Any) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            "Cosmos3 multiview decomposed_temporal_window_seconds must be null or a "
            f"finite non-negative number, got {value!r}."
        )


class MaskItem(msgspec.Struct, frozen=True):
    """Semantic description of one packed vision item.

    ``token_shape`` is ``(latent_frames, patch_height, patch_width)``. Frames are
    camera-major: all frames of view zero, then all frames of view one, and so
    on. ``seconds_per_frame`` is the wall-clock duration of one latent frame and
    must agree for items sharing the view grid.
    """

    token_shape: tuple[int, int, int]
    num_views: int
    view_offset: int = 0
    is_control: bool = False
    seconds_per_frame: float = 1.0

    def __post_init__(self) -> None:
        latent_t, patch_h, patch_w = self.token_shape
        if latent_t <= 0 or patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"Cosmos3 multiview token_shape must be positive, got {self.token_shape}."
            )
        if self.num_views <= 0 or latent_t % self.num_views:
            raise ValueError(
                "Cosmos3 multiview latent frames must be divisible by num_views: "
                f"latent_t={latent_t}, num_views={self.num_views}."
            )
        _validate_positive_finite(self.seconds_per_frame, "seconds_per_frame")

    @property
    def num_tokens(self) -> int:
        return math.prod(self.token_shape)


class MultiviewLayout(msgspec.Struct, frozen=True):
    """Request-invariant geometry passed from the pipeline to the transformer."""

    num_views: int
    latent_frames: int
    patch_height: int
    patch_width: int
    attention_scope: AttentionScope = "decomposed"
    decomposed_temporal_window_seconds: float | None = None
    control_attends_sensor: bool = False
    seconds_per_frame: float = 1.0
    backend: str = "triton"
    #: Capacity the UND stream is padded to, independent of any one prompt's
    #: length, so the compiled attention sees a single shape.
    max_und_tokens: int = DEFAULT_MAX_UND_TOKENS

    #: v1 always packs one fully-clean control (WSM) item then one RGB target.
    NUM_ITEMS: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _validate_attention_scope(self.attention_scope)
        validate_multiview_backend(self.backend)
        if self.backend == "fa4":
            # Importing here registers the FA4 custom op while we are still
            # host-side; the module defers every CuTe/CUTLASS import.
            from sglang.multimodal_gen.runtime.models.dits import (  # noqa: F401
                cosmos3_multiview_fa4,
            )
        if self.max_und_tokens <= 0:
            raise ValueError(
                f"Cosmos3 multiview max_und_tokens must be positive, got {self.max_und_tokens}."
            )
        if self.num_views <= 0 or self.latent_frames <= 0:
            raise ValueError(
                "Cosmos3 multiview num_views and latent_frames must be positive."
            )
        if self.latent_frames % self.num_views:
            raise ValueError(
                "Cosmos3 multiview latent_frames must be camera-major and divisible "
                f"by num_views: latent_frames={self.latent_frames}, num_views={self.num_views}."
            )
        if self.patch_height <= 0 or self.patch_width <= 0:
            raise ValueError("Cosmos3 multiview patch dimensions must be positive.")
        _validate_temporal_window(self.decomposed_temporal_window_seconds)
        if not isinstance(self.control_attends_sensor, bool):
            raise TypeError(
                "Cosmos3 multiview control_attends_sensor must be boolean, "
                f"got {type(self.control_attends_sensor).__name__}."
            )
        _validate_positive_finite(self.seconds_per_frame, "seconds_per_frame")

    @property
    def frames_per_view(self) -> int:
        return self.latent_frames // self.num_views

    @property
    def item_tokens(self) -> int:
        return self.latent_frames * self.patch_height * self.patch_width

    @property
    def gen_tokens(self) -> int:
        return self.item_tokens * self.NUM_ITEMS

    @property
    def block_sizes(self) -> tuple[int, int]:
        """The ``(q, kv)`` sparse block granularity this backend demands."""
        return _BACKEND_BLOCK_SIZES[self.backend]

    def mask_items(self) -> tuple[MaskItem, ...]:
        """Build the packed control and target items.

        Attention visibility is independent of target conditioning state, so
        mask items contain only the semantic fields read by the predicate.
        """
        shape = (self.latent_frames, self.patch_height, self.patch_width)
        return (
            MaskItem(
                shape,
                self.num_views,
                is_control=True,
                seconds_per_frame=self.seconds_per_frame,
            ),
            MaskItem(
                shape,
                self.num_views,
                is_control=False,
                seconds_per_frame=self.seconds_per_frame,
            ),
        )

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.num_views,
            self.latent_frames,
            self.patch_height,
            self.patch_width,
            self.attention_scope,
            self.decomposed_temporal_window_seconds,
            self.control_attends_sensor,
            self.seconds_per_frame,
            self.backend,
            self.max_und_tokens,
        )


# eq=False: tensor fields make a generated __eq__ return a tensor.
class MultiviewFlexMetadata(msgspec.Struct, frozen=True, eq=False):
    """Per-token metadata for rectangular GEN-to-[UND|GEN] attention."""

    sample_id: torch.Tensor
    frame_id: torch.Tensor
    view_id: torch.Tensor
    is_control: torch.Tensor
    is_und: torch.Tensor
    timestamp: torch.Tensor
    query_start: int
    attention_scope: AttentionScope
    decomposed_temporal_window_seconds: float | None = None
    control_attends_sensor: bool = False

    @property
    def kv_len(self) -> int:
        return int(self.sample_id.numel())

    @property
    def q_len(self) -> int:
        return self.kv_len - self.query_start

    def query_vectors(self) -> tuple[torch.Tensor, ...]:
        query_slice = slice(self.query_start, None)
        return (
            self.sample_id[query_slice],
            self.frame_id[query_slice],
            self.view_id[query_slice],
            self.is_control[query_slice],
            self.is_und[query_slice],
            self.timestamp[query_slice],
        )

    def key_vectors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.sample_id,
            self.frame_id,
            self.view_id,
            self.is_control,
            self.is_und,
            self.timestamp,
        )

    def query_grouping_vectors(self) -> tuple[torch.Tensor, ...]:
        """Discrete query fields that define semantic runs.

        Timestamp is excluded: under the validated single-rate camera layout it
        is a function of ``(view_id, frame_id)``.
        """
        return self.query_vectors()[:-1]

    def key_grouping_vectors(self) -> tuple[torch.Tensor, ...]:
        return self.key_vectors()[:-1]


class MultiviewAttentionContext(msgspec.Struct, frozen=True, eq=False):
    """Runtime wrapper that keeps the request-local caches on the transformer."""

    layout: MultiviewLayout
    mask_cache: MutableMapping[tuple[Any, ...], BlockMask | MultiviewBlockSparsity]
    buffer_cache: MutableMapping[tuple[Any, ...], torch.Tensor] = msgspec.field(
        default_factory=dict
    )


class PaddedAttentionGeometry(msgspec.Struct, frozen=True):
    real_q_len: int
    padded_q_len: int
    real_und_len: int
    padded_und_len: int


def expand_multiview_condition_frame_indexes(
    indexes: Sequence[int] | int | None,
    num_views: int,
    latent_t: int,
) -> list[int]:
    """Expand per-view-local latent frame indexes into camera-major indexes."""
    if num_views <= 0 or latent_t <= 0 or latent_t % num_views:
        raise ValueError(
            "Cosmos3 multiview expansion requires latent_t divisible by num_views: "
            f"latent_t={latent_t}, num_views={num_views}."
        )
    if indexes is None:
        local_indexes: Sequence[int] = ()
    elif isinstance(indexes, int):
        local_indexes = (indexes,)
    else:
        local_indexes = indexes
    frames_per_view = latent_t // num_views
    filtered = sorted(
        {int(index) for index in local_indexes if 0 <= int(index) < frames_per_view}
    )
    return [
        view * frames_per_view + frame
        for view in range(num_views)
        for frame in filtered
    ]


def build_multiview_flex_metadata(
    seq_len: int,
    full_q_offsets: Sequence[int],
    items_per_sample: Sequence[Sequence[MaskItem]] | Sequence[MaskItem],
    device: torch.device | str,
    num_und: int,
    attention_scope: AttentionScope = "decomposed",
    decomposed_temporal_window_seconds: float | None = None,
    control_attends_sensor: bool = False,
) -> MultiviewFlexMetadata:
    """Build per-token metadata without materializing a dense mask.

    ``full_q_offsets`` contains the start of every packed vision item plus the
    end of the final item. The first offset is also the query start and may be
    larger than ``num_und`` because UND is padded independently.
    """
    attention_scope = _validate_attention_scope(attention_scope)
    _validate_temporal_window(decomposed_temporal_window_seconds)
    if not isinstance(control_attends_sensor, bool):
        raise TypeError("Cosmos3 multiview control_attends_sensor must be boolean.")
    device = torch.device(device)
    if seq_len <= 0 or num_und < 0 or num_und > seq_len:
        raise ValueError(
            f"Invalid Cosmos3 multiview sequence geometry: seq_len={seq_len}, num_und={num_und}."
        )
    if items_per_sample and isinstance(items_per_sample[0], MaskItem):  # type: ignore[index]
        samples: list[list[MaskItem]] = [list(items_per_sample)]  # type: ignore[arg-type]
    else:
        samples = [list(items) for items in items_per_sample]  # type: ignore[arg-type]
    if len(samples) != 1:
        raise ValueError(
            "Cosmos3 multiview v1 supports exactly one sample per request."
        )
    items = samples[0]
    if not items:
        raise ValueError(
            "Cosmos3 multiview metadata requires at least one vision item."
        )
    if len(full_q_offsets) != len(items) + 1:
        raise ValueError(
            "Cosmos3 multiview full_q_offsets must contain one boundary per item plus "
            f"the end: offsets={list(full_q_offsets)}, items={len(items)}."
        )
    offsets = tuple(int(offset) for offset in full_q_offsets)
    if (
        offsets[0] < num_und
        or offsets[-1] > seq_len
        or any(a > b for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError(
            f"Invalid Cosmos3 multiview item offsets: {offsets} for seq_len={seq_len}."
        )

    sample_id = torch.full((seq_len,), -1, dtype=torch.int64, device=device)
    frame_id = torch.full_like(sample_id, -1)
    view_id = torch.full_like(sample_id, -1)
    is_control = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_und = torch.zeros_like(is_control)
    timestamp = torch.full((seq_len,), -1.0, dtype=torch.float32, device=device)
    sample_id[:num_und] = 0
    is_und[:num_und] = True

    view_offsets = {item.view_offset for item in items}
    if (
        attention_scope == "decomposed"
        and len(view_offsets) > 1
        and decomposed_temporal_window_seconds is None
    ):
        raise ValueError(
            "Cosmos3 decomposed attention does not support mixed view offsets "
            "without a temporal window."
        )

    rates_by_view_offset: dict[int, float] = {}
    for item in items:
        expected = rates_by_view_offset.setdefault(
            item.view_offset, item.seconds_per_frame
        )
        if not math.isclose(item.seconds_per_frame, expected):
            raise ValueError(
                "Cosmos3 multiview items sharing a view offset must use the same "
                f"seconds_per_frame: offset={item.view_offset}, got "
                f"{item.seconds_per_frame}, expected {expected}."
            )

    for item_index, (item, start, end) in enumerate(
        zip(items, offsets[:-1], offsets[1:], strict=True)
    ):
        if end - start != item.num_tokens:
            raise ValueError(
                f"Cosmos3 multiview item {item_index} occupies {end - start} tokens, "
                f"expected {item.num_tokens}."
            )
        latent_t, patch_h, patch_w = item.token_shape
        spatial_tokens = patch_h * patch_w
        frames_per_view = latent_t // item.num_views
        item_frames = torch.arange(frames_per_view, dtype=torch.int64, device=device)
        item_frames = item_frames.repeat(item.num_views).repeat_interleave(
            spatial_tokens
        )
        item_views = torch.arange(
            item.view_offset,
            item.view_offset + item.num_views,
            dtype=torch.int64,
            device=device,
        ).repeat_interleave(frames_per_view * spatial_tokens)
        item_timestamps = item_frames.to(torch.float32) * item.seconds_per_frame
        sample_id[start:end] = 0
        frame_id[start:end] = item_frames
        view_id[start:end] = item_views
        is_control[start:end] = item.is_control
        timestamp[start:end] = item_timestamps

    return MultiviewFlexMetadata(
        sample_id=sample_id,
        frame_id=frame_id,
        view_id=view_id,
        is_control=is_control,
        is_und=is_und,
        timestamp=timestamp,
        query_start=offsets[0],
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        control_attends_sensor=control_attends_sensor,
    )


def _make_pair_allowed(
    q_vectors: tuple[torch.Tensor, ...],
    k_vectors: tuple[torch.Tensor, ...],
    attention_scope: AttentionScope,
    decomposed_temporal_window_seconds: float | None,
    control_attends_sensor: bool,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the exact pair predicate with tensor-only traced configuration.

    The returned closure is stored on Triton's ``BlockMask`` and traced by
    Inductor. Scalar options are materialized here, outside that closure, so
    Python strings, booleans, and floats cannot become dynamic captured
    scalars. The FA4 path uses the same closure eagerly to build its packed run
    table.
    """
    device = q_vectors[2].device
    reaches_every_view = torch.tensor(attention_scope == "all_views", device=device)
    is_decomposed = torch.tensor(attention_scope == "decomposed", device=device)
    control_reaches_sensor = torch.tensor(control_attends_sensor, device=device)
    has_temporal_window = torch.tensor(
        decomposed_temporal_window_seconds is not None, device=device
    )
    temporal_window = torch.tensor(
        0.0
        if decomposed_temporal_window_seconds is None
        else decomposed_temporal_window_seconds,
        dtype=torch.float32,
        device=device,
    )
    temporal_window_eps = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def pair_allowed(q_index: torch.Tensor, kv_index: torch.Tensor) -> torch.Tensor:
        q_sample = q_vectors[0][q_index]
        q_frame = q_vectors[1][q_index]
        q_view = q_vectors[2][q_index]
        q_control = q_vectors[3][q_index]
        q_timestamp = q_vectors[5][q_index]
        k_sample = k_vectors[0][kv_index]
        k_frame = k_vectors[1][kv_index]
        k_view = k_vectors[2][kv_index]
        k_control = k_vectors[3][kv_index]
        k_und = k_vectors[4][kv_index]
        k_timestamp = k_vectors[5][kv_index]

        # Sentinel equality isolates padding from real tokens while giving
        # every padded query at least one padded key.
        same_sample = q_sample == k_sample
        same_view = q_view == k_view
        same_frame = q_frame == k_frame
        timestamp_gap = q_timestamp - k_timestamp
        within_temporal_window = (timestamp_gap >= -temporal_window_eps) & (
            timestamp_gap <= temporal_window + temporal_window_eps
        )
        reaches_own_instant = is_decomposed & torch.where(
            has_temporal_window, within_temporal_window, same_frame
        )
        in_scope = reaches_every_view | same_view | reaches_own_instant

        sensor_to_sensor = (~q_control) & (~k_control) & in_scope
        sensor_to_control = (~q_control) & k_control & same_view
        control_to_control = q_control & k_control & same_view
        control_to_sensor = (
            control_reaches_sensor & q_control & (~k_control) & same_view
        )
        return same_sample & (
            k_und
            | sensor_to_sensor
            | sensor_to_control
            | control_to_control
            | control_to_sensor
        )

    return pair_allowed


def multiview_pair_predicate(
    metadata: MultiviewFlexMetadata,
    q_index: torch.Tensor,
    kv_index: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact token visibility predicate (also used as mask_mod)."""
    pair_allowed = _make_pair_allowed(
        metadata.query_vectors(),
        metadata.key_vectors(),
        metadata.attention_scope,
        metadata.decomposed_temporal_window_seconds,
        metadata.control_attends_sensor,
    )
    return pair_allowed(q_index, kv_index)


def _semantic_groups(
    vectors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return run IDs and first-token indexes without converting field dtypes."""
    if not vectors:
        raise ValueError(
            "Cosmos3 multiview semantic grouping requires at least one field."
        )
    seq_len = vectors[0].numel()
    if any(vector.numel() != seq_len for vector in vectors):
        raise ValueError(
            "Cosmos3 multiview semantic grouping fields must have equal lengths."
        )
    changed = torch.zeros(seq_len, dtype=torch.bool, device=vectors[0].device)
    changed[:1] = True
    for vector in vectors:
        changed[1:] |= vector[1:] != vector[:-1]
    group_ids = changed.to(torch.int64).cumsum(0) - 1
    representatives = torch.nonzero(changed, as_tuple=False).flatten()
    return group_ids, representatives


def _block_group_presence(
    group_ids: torch.Tensor, block_size: int, num_groups: int
) -> torch.Tensor:
    if group_ids.numel() % block_size:
        raise ValueError(
            f"Cosmos3 multiview metadata length {group_ids.numel()} is not aligned "
            f"to block size {block_size}."
        )
    blocks = group_ids.view(-1, block_size)
    presence = torch.zeros(
        (blocks.shape[0], num_groups), dtype=torch.bool, device=group_ids.device
    )
    presence.scatter_(1, blocks, True)
    return presence


def _block_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the full KV-block width instead of trimming to the densest row.
    # ``create_block_mask`` always produces full-width contiguous indices and
    # both the Triton template and ``BlockMask`` helpers assume that layout
    # (see pytorch/pytorch#153344). Full width also keeps the mask shapes
    # identical across CFG branches, so both share one compiled kernel.
    counts = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.argsort(mask.to(torch.int8), dim=-1, descending=True, stable=True)
    return counts, indices.to(torch.int32)


def _pack_allowed_bits(group_allowed: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack the ``[q_group, k_group]`` truth table into int32 bit words.

    Bit ``g_k`` of word ``g_q * words_per_row + g_k // 32`` is the answer for
    that group pair. Words hold the unsigned 32-bit pattern stored in ``int32``,
    which is what the CuTe kernel reinterprets.
    """
    num_q_groups, num_k_groups = group_allowed.shape
    words_per_row = (num_k_groups + 31) // 32
    padded = group_allowed.new_zeros((num_q_groups, words_per_row * 32))
    padded[:, :num_k_groups] = group_allowed
    weights = torch.arange(32, device=group_allowed.device, dtype=torch.int64)
    words = (
        padded.view(num_q_groups, words_per_row, 32).to(torch.int64) << weights
    ).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.reshape(-1).to(torch.int32).contiguous(), words_per_row


# eq=False: the fields are tensors, so a generated __eq__ would return a tensor
# rather than a bool. Instances are cache values compared by identity.
class MultiviewBlockSparsity(msgspec.Struct, frozen=True, eq=False):
    """Backend-neutral sparse block map plus the run-compressed mask table.

    ``partial_*``/``full_*`` are the FlexAttention KV-block layout, which
    FlashAttention-4 consumes unchanged. The remaining fields are the exact
    per-element fallback used inside partially masked tiles: a token-to-run id
    for each side plus the packed truth table over run pairs.
    """

    partial_counts: torch.Tensor
    partial_indices: torch.Tensor
    full_counts: torch.Tensor
    full_indices: torch.Tensor
    q_word_base: torch.Tensor
    k_group_ids: torch.Tensor
    allowed_words: torch.Tensor
    group_allowed: torch.Tensor
    words_per_row: int
    q_block_size: int
    kv_block_size: int
    metadata: MultiviewFlexMetadata

    @property
    def q_len(self) -> int:
        return self.metadata.q_len

    @property
    def kv_len(self) -> int:
        return self.metadata.kv_len

    def aux_tensors(self) -> list[torch.Tensor]:
        """The mask_mod auxiliary tensors, in the order the kernel indexes them."""
        return [self.q_word_base, self.k_group_ids, self.allowed_words]

    def to_block_mask(self) -> BlockMask:
        metadata = self.metadata
        pair_allowed = _make_pair_allowed(
            metadata.query_vectors(),
            metadata.key_vectors(),
            metadata.attention_scope,
            metadata.decomposed_temporal_window_seconds,
            metadata.control_attends_sensor,
        )

        def mask_mod(
            batch: torch.Tensor,
            head: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del batch, head
            return pair_allowed(q_idx, kv_idx)

        return BlockMask.from_kv_blocks(
            self.partial_counts[None, None],
            self.partial_indices[None, None],
            self.full_counts[None, None],
            self.full_indices[None, None],
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
            mask_mod=mask_mod,
            seq_lengths=(metadata.q_len, metadata.kv_len),
            compute_q_blocks=False,
        )


def build_multiview_block_sparsity(
    metadata: MultiviewFlexMetadata,
    *,
    q_block_size: int = SPARSE_Q_BLOCK_SIZE,
    kv_block_size: int = SPARSE_KV_BLOCK_SIZE,
) -> MultiviewBlockSparsity:
    """Compress semantic runs into a sparse block map and a mask lookup table.

    The projection works at semantic-run and sparse-block granularity. Its
    largest dense intermediates are block-grid sized (about 10.4M entries for
    the released 11-view geometry at 64x64), never the ~42B-pair dense mask.
    """
    q_vectors = metadata.query_vectors()
    k_vectors = metadata.key_vectors()
    q_group_ids, q_representatives = _semantic_groups(metadata.query_grouping_vectors())
    k_group_ids, k_representatives = _semantic_groups(metadata.key_grouping_vectors())

    pair_allowed = _make_pair_allowed(
        q_vectors,
        k_vectors,
        metadata.attention_scope,
        metadata.decomposed_temporal_window_seconds,
        metadata.control_attends_sensor,
    )
    group_allowed = pair_allowed(q_representatives[:, None], k_representatives[None, :])
    q_presence = _block_group_presence(
        q_group_ids, q_block_size, q_representatives.numel()
    )
    k_presence = _block_group_presence(
        k_group_ids, kv_block_size, k_representatives.numel()
    )

    # Float16 represents these tiny integer overlap counts exactly and gives a
    # fast tensor-core projection on CUDA. CPU uses float32 matmul.
    projection_dtype = (
        torch.float16 if q_presence.device.type == "cuda" else torch.float32
    )
    q_projection = q_presence.to(projection_dtype)
    k_projection = k_presence.to(projection_dtype)
    visible_blocks = (
        q_projection @ group_allowed.to(projection_dtype) @ k_projection.T
    ) > 0
    forbidden_blocks = (
        q_projection @ (~group_allowed).to(projection_dtype) @ k_projection.T
    ) > 0
    full_blocks = visible_blocks & (~forbidden_blocks)
    partial_blocks = visible_blocks & (~full_blocks)

    partial_counts, partial_indices = _block_indices(partial_blocks)
    full_counts, full_indices = _block_indices(full_blocks)

    # Fold the table row stride into the query-side id so the kernel needs no
    # compile-time shape constant and one compiled mask_mod serves every layout.
    allowed_words, words_per_row = _pack_allowed_bits(group_allowed)
    q_word_base = (q_group_ids * words_per_row).to(torch.int32).contiguous()

    return MultiviewBlockSparsity(
        partial_counts=partial_counts,
        partial_indices=partial_indices,
        full_counts=full_counts,
        full_indices=full_indices,
        q_word_base=q_word_base,
        k_group_ids=k_group_ids.to(torch.int32).contiguous(),
        allowed_words=allowed_words,
        group_allowed=group_allowed,
        words_per_row=words_per_row,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        metadata=metadata,
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def get_multiview_attention_plan(
    context: MultiviewAttentionContext,
    *,
    real_und_len: int,
    real_q_len: int,
    device: torch.device,
) -> tuple[BlockMask | MultiviewBlockSparsity, PaddedAttentionGeometry]:
    """Build or retrieve the request-local mask for one CFG text length.

    Returns whichever mask representation the layout's backend consumes: a
    ``BlockMask`` for Triton FlexAttention, or a ``MultiviewBlockSparsity`` for
    FlashAttention-4. Both padded lengths are pure functions of the layout,
    never of this call's ``real_und_len``: prompts of different lengths produce
    different masks but identically shaped tensors.
    """
    layout = context.layout
    if real_q_len != layout.gen_tokens:
        raise ValueError(
            "Cosmos3 multiview packed GEN length does not match the request layout: "
            f"attention={real_q_len}, layout={layout.gen_tokens}."
        )
    if real_und_len > layout.max_und_tokens:
        raise ValueError(
            "Cosmos3 multiview UND stream exceeds the layout capacity the attention "
            f"was sized for: tokens={real_und_len}, max_und_tokens={layout.max_und_tokens}."
        )
    q_block_size, kv_block_size = layout.block_sizes
    padded_q_len = _round_up(real_q_len, q_block_size)
    padded_und_len = _round_up(layout.max_und_tokens, kv_block_size)
    geometry = PaddedAttentionGeometry(
        real_q_len, padded_q_len, real_und_len, padded_und_len
    )
    key = (
        layout.cache_key(),
        real_und_len,
        padded_und_len,
        real_q_len,
        padded_q_len,
        q_block_size,
        kv_block_size,
        device.type,
        device.index,
    )
    cached = context.mask_cache.get(key)
    if cached is not None:
        return cached, geometry

    items = layout.mask_items()
    item_tokens = layout.item_tokens
    item_offsets = tuple(
        padded_und_len + index * item_tokens for index in range(len(items) + 1)
    )
    metadata = build_multiview_flex_metadata(
        seq_len=padded_und_len + padded_q_len,
        full_q_offsets=item_offsets,
        items_per_sample=items,
        device=device,
        num_und=real_und_len,
        attention_scope=layout.attention_scope,
        decomposed_temporal_window_seconds=layout.decomposed_temporal_window_seconds,
        control_attends_sensor=layout.control_attends_sensor,
    )
    sparsity = build_multiview_block_sparsity(
        metadata,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    plan = sparsity if layout.backend == "fa4" else sparsity.to_block_mask()
    context.mask_cache[key] = plan
    return plan, geometry


def get_multiview_block_mask(
    context: MultiviewAttentionContext,
    *,
    real_und_len: int,
    real_q_len: int,
    device: torch.device,
) -> tuple[BlockMask, PaddedAttentionGeometry]:
    """Request-local ``BlockMask`` for the Triton FlexAttention backend."""
    plan, geometry = get_multiview_attention_plan(
        context,
        real_und_len=real_und_len,
        real_q_len=real_q_len,
        device=device,
    )
    if not isinstance(plan, BlockMask):
        raise TypeError(
            "Cosmos3 multiview get_multiview_block_mask requires backend='triton', "
            f"got {context.layout.backend!r}; use get_multiview_attention_plan instead."
        )
    return plan, geometry


def _packing_buffer(
    cache: MutableMapping[tuple[Any, ...], torch.Tensor] | None,
    slot: str,
    reference: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Return a zeroed packing buffer, reused across layers when a cache is given.

    The packed q/k/v layouts are rebuilt in every GEN layer of every forward,
    and for the released 11-view geometry the query buffer alone is ~1.7 GiB,
    so allocating and zeroing one per layer would cost terabytes of memset per
    run. Only real token rows are ever written; padding rows are never read by
    real queries, so a buffer stays reusable for any later call with the same
    slot, shape, dtype, and device.
    """
    if cache is None:
        return reference.new_zeros(shape)
    key = (slot, shape, reference.dtype, reference.device.type, reference.device.index)
    buffer = cache.get(key)
    if buffer is None:
        buffer = reference.new_zeros(shape)
        cache[key] = buffer
    return buffer


def _validate_parts(
    parts: tuple[tuple[torch.Tensor, int], ...],
) -> tuple[torch.Tensor, int, int, int, int]:
    if not parts:
        raise ValueError(
            "Cosmos3 multiview attention requires at least one sequence part."
        )
    reference = parts[0][0]
    batch, _, heads, head_dim = reference.shape
    total_len = sum(target_len for _, target_len in parts)
    for tensor, target_len in parts:
        if (
            tensor.ndim != 4
            or tensor.shape[0] != batch
            or tensor.shape[2:] != (heads, head_dim)
        ):
            raise ValueError(
                "Cosmos3 multiview attention sequence parts must share [B, H, D]: "
                f"reference={tuple(reference.shape)}, part={tuple(tensor.shape)}."
            )
        if tensor.shape[1] > target_len:
            raise ValueError(
                f"Cannot pad sequence length {tensor.shape[1]} down to {target_len}."
            )
    return reference, batch, heads, head_dim, total_len


def _pack_padded_bhsd(
    *parts: tuple[torch.Tensor, int],
    buffer_cache: MutableMapping[tuple[Any, ...], torch.Tensor] | None = None,
    slot: str = "",
) -> torch.Tensor:
    """Pack ``[B, S, H, D]`` parts directly into contiguous ``[B, H, S, D]``."""
    reference, batch, heads, head_dim, total_len = _validate_parts(parts)
    packed = _packing_buffer(
        buffer_cache, f"bhsd:{slot}", reference, (batch, heads, total_len, head_dim)
    )
    offset = 0
    for tensor, target_len in parts:
        packed[:, :, offset : offset + tensor.shape[1]].copy_(tensor.transpose(1, 2))
        offset += target_len
    return packed


def _pack_padded_bshd(
    *parts: tuple[torch.Tensor, int],
    buffer_cache: MutableMapping[tuple[Any, ...], torch.Tensor] | None = None,
    slot: str = "",
) -> torch.Tensor:
    """Concatenate and pad ``[B, S, H, D]`` parts, keeping the native layout.

    FlashAttention-4 consumes ``[B, S, H, D]`` directly, so unlike the Triton
    path this never transposes.
    """
    reference, batch, heads, head_dim, total_len = _validate_parts(parts)
    packed = _packing_buffer(
        buffer_cache, f"bshd:{slot}", reference, (batch, total_len, heads, head_dim)
    )
    offset = 0
    for tensor, target_len in parts:
        packed[:, offset : offset + tensor.shape[1]].copy_(tensor)
        offset += target_len
    return packed


_compiled_flex_attention = None


def flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: BlockMask,
    backend: str,
) -> torch.Tensor:
    """Run pinned Triton FlexAttention on contiguous ``[B, H, S, D]`` tensors."""
    if backend != "triton":
        raise ValueError(
            f"Cosmos3 multiview flex_attention supports only backend='triton', got {backend!r}."
        )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError(
            "Cosmos3 multiview FlexAttention requires contiguous [B, H, S, D] inputs."
        )
    kernel_options = {
        "BACKEND": "TRITON",
        "BLOCK_M": TRITON_Q_BLOCK_SIZE,
        "BLOCK_N": TRITON_KV_BLOCK_SIZE,
        "num_stages": TRITON_NUM_STAGES,
        "num_warps": TRITON_NUM_WARPS,
        "USE_TMA": False,
    }
    if q.device.type == "cuda":
        global _compiled_flex_attention
        if _compiled_flex_attention is None:
            _compiled_flex_attention = torch.compile(
                torch_flex_attention, dynamic=False
            )
        return _compiled_flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=kernel_options,
        )
    # Eager CPU support is for tiny correctness tests; production multiview
    # inference is admitted only on CUDA.
    return torch_flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=True,
        kernel_options=kernel_options,
    )


def padded_multiview_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    """Pad UND and GEN independently, attend once, then trim GEN rows.

    ``q``, ``k``, ``v`` are the GEN projections in ``[B, S_gen, H, D]`` (``k``/``v``
    with the KV head count); ``k_und``/``v_und`` are the cached text keys and
    values in ``[B, S_und, H_kv, D]``. Returns ``[B, S_gen, H, D]``.
    """
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError(
            "Cosmos3 multiview q/k/v sequence geometry must match before GQA: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    if k_und.shape != v_und.shape or k_und.shape[0] != q.shape[0]:
        raise ValueError(
            "Cosmos3 multiview UND key/value geometry mismatch: "
            f"k_und={tuple(k_und.shape)}, v_und={tuple(v_und.shape)}."
        )
    plan, geometry = get_multiview_attention_plan(
        context,
        real_und_len=k_und.shape[1],
        real_q_len=q.shape[1],
        device=q.device,
    )
    buffers = context.buffer_cache
    if context.layout.backend == "fa4":
        from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_fa4 import (
            multiview_fa4_attention,
        )

        q_padded = _pack_padded_bshd(
            (q, geometry.padded_q_len), buffer_cache=buffers, slot="q"
        )
        k_all = _pack_padded_bshd(
            (k_und, geometry.padded_und_len),
            (k, geometry.padded_q_len),
            buffer_cache=buffers,
            slot="k",
        )
        v_all = _pack_padded_bshd(
            (v_und, geometry.padded_und_len),
            (v, geometry.padded_q_len),
            buffer_cache=buffers,
            slot="v",
        )
        output = multiview_fa4_attention(q_padded, k_all, v_all, plan)
        return output[:, : geometry.real_q_len]

    q_padded = _pack_padded_bhsd(
        (q, geometry.padded_q_len), buffer_cache=buffers, slot="q"
    )
    k_all = _pack_padded_bhsd(
        (k_und, geometry.padded_und_len),
        (k, geometry.padded_q_len),
        buffer_cache=buffers,
        slot="k",
    )
    v_all = _pack_padded_bhsd(
        (v_und, geometry.padded_und_len),
        (v, geometry.padded_q_len),
        buffer_cache=buffers,
        slot="v",
    )
    output = flex_attention(
        q_padded, k_all, v_all, block_mask=plan, backend=context.layout.backend
    )
    return output[:, :, : geometry.real_q_len].transpose(1, 2)
