# SPDX-License-Identifier: Apache-2.0
"""Packed-item geometry for Cosmos3 Multiview-AV attention.

A request packs its sensor streams camera-major into one GEN sequence: for
every item (WSM control, RGB target, HD-map control, LiDAR target) all frames
of view zero, then view one, and so on. ``MaskItem`` describes one such item,
``MultiviewLayout`` the whole packed sequence plus the attention contract the
checkpoint was exported with, and ``MultiviewAttentionContext`` carries the
request-local caches from the pipeline to the transformer. The attention
itself lives in ``cosmos3_multiview_attention`` (masked FlexAttention / FA4
block-sparse kernels, exports with ``backend`` ``triton``/``fa4``) and
``cosmos3_multiview_maskless`` (the three unmasked folds of ``maskless`` exports).
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from typing import Any, Literal

import msgspec

AttentionScope = Literal["all_views", "same_view", "decomposed"]

_VALID_ATTENTION_SCOPES = frozenset({"all_views", "same_view", "decomposed"})


def _validate_attention_scope(attention_scope: str) -> AttentionScope:
    if attention_scope not in _VALID_ATTENTION_SCOPES:
        raise ValueError(
            "Cosmos3 multiview attention_scope must be one of "
            f"{sorted(_VALID_ATTENTION_SCOPES)}, got {attention_scope!r}."
        )
    return attention_scope  # type: ignore[return-value]


# ``triton``/``fa4`` run one masked attention (FlexAttention on Triton kernels or
# FlashAttention-4 block-sparse kernels) that counts every permitted key once;
# ``maskless`` runs the three unmasked folds, which double-count the query's own
# (view, frame) cell. A checkpoint is served faithfully only by its own family.
MULTIVIEW_BACKENDS: tuple[str, ...] = ("triton", "fa4", "maskless")
MASKED_BACKENDS: tuple[str, ...] = ("triton", "fa4")

# The masked kernels pad the UND (text) stream to a fixed capacity rather than to
# each prompt's length: the flex kernel is compiled with dynamic=False, so a pad
# that tracked the prompt would recompile per text-length bucket. The default is
# the Cosmos3 caption cap plus the ``eos`` and ``vision_start`` framing tokens.
DEFAULT_MAX_UND_TOKENS = 4096 + 2


def validate_multiview_backend(backend: str) -> str:
    """Reject an unknown backend name (exposed so callers fail at load time)."""
    if backend not in MULTIVIEW_BACKENDS:
        raise ValueError(
            "Cosmos3 multiview attention backend must be one of "
            f"{list(MULTIVIEW_BACKENDS)}, got {backend!r}."
        )
    return backend


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
    """Semantic description of one packed sensor item.

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
    #: LiDAR range-view items share one view id (-2) distinct from every camera
    #: and read every caption of the sample.
    is_lidar: bool = False

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
    #: Attention family the request runs; see ``MULTIVIEW_BACKENDS``.
    backend: str = "maskless"
    #: Capacity the masked kernels pad the UND stream to, independent of any
    #: one prompt's length, so the compiled attention sees a single shape.
    max_und_tokens: int = DEFAULT_MAX_UND_TOKENS
    #: Explicit FA4 sparse block map; None derives it from the device's compute
    #: capability at plan time (``cosmos3_multiview_attention.fa4_sparse_block_sizes``).
    fa4_block_sizes: tuple[int, int] | None = None
    #: Packed GEN items in sequence order. Empty selects the camera-only pair
    #: (one fully-clean WSM control item, then one RGB target); joint
    #: camera/LiDAR requests append the HD-map control and LiDAR target items.
    items: tuple[MaskItem, ...] = ()
    #: Real text tokens per caption, in camera order, when the checkpoint
    #: tokenizes one caption per camera. Empty means one sample-level caption.
    caption_lengths: tuple[int, ...] = ()
    #: Whether LiDAR tokens read the sample's captions; honored by every backend.
    lidar_attends_captions: bool = True

    def __post_init__(self) -> None:
        _validate_attention_scope(self.attention_scope)
        validate_multiview_backend(self.backend)
        if self.max_und_tokens <= 0:
            raise ValueError(
                f"Cosmos3 multiview max_und_tokens must be positive, got {self.max_und_tokens}."
            )
        if self.fa4_block_sizes is not None and (
            len(self.fa4_block_sizes) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.fa4_block_sizes
            )
        ):
            raise ValueError(
                "Cosmos3 multiview fa4_block_sizes must be two positive ints, "
                f"got {self.fa4_block_sizes!r}."
            )
        if not self.items:
            shape = (self.latent_frames, self.patch_height, self.patch_width)
            msgspec.structs.force_setattr(
                self,
                "items",
                (
                    MaskItem(
                        shape,
                        self.num_views,
                        is_control=True,
                        seconds_per_frame=self.seconds_per_frame,
                    ),
                    MaskItem(
                        shape,
                        self.num_views,
                        seconds_per_frame=self.seconds_per_frame,
                    ),
                ),
            )
        if any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for length in self.caption_lengths
        ):
            raise ValueError(
                "Cosmos3 multiview caption_lengths must be positive integers, "
                f"got {self.caption_lengths!r}."
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
        return sum(item.num_tokens for item in self.items)

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
            self.fa4_block_sizes,
            self.items,
            self.caption_lengths,
            self.lidar_attends_captions,
        )


class MultiviewAttentionContext(msgspec.Struct, frozen=True, eq=False):
    """Runtime wrapper that keeps the request-local caches on the transformer.

    ``plan_cache`` holds the maskless folds' gathers and varlen offsets;
    ``mask_cache`` the masked kernels' BlockMask / block sparsity and
    ``buffer_cache`` their zeroed padded q/k/v packing buffers.
    """

    layout: MultiviewLayout
    plan_cache: MutableMapping[tuple[Any, ...], Any]
    mask_cache: MutableMapping[tuple[Any, ...], Any] = msgspec.field(
        default_factory=dict
    )
    buffer_cache: MutableMapping[tuple[Any, ...], Any] = msgspec.field(
        default_factory=dict
    )


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
