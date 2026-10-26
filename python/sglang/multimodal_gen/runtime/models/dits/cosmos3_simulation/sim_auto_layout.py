# SPDX-License-Identifier: Apache-2.0
"""Chunk schedule, packed-item layout and token metadata for Cosmos3-Nano-Sim-Auto.

The student generates one chunk at a time: ``frames_per_chunk`` RGB latents
(0.4 s at 30 fps) plus the LiDAR sweeps of the same wall-clock span, with
latent 0 and sweep 0 alone in chunk 0 and a possibly partial last chunk. Each
chunk packs ``[RGB control | RGB target | LiDAR control | LiDAR target]``,
camera-major (all frames of camera 0, then camera 1, ...), every frame a
row-major patch grid. The per-token metadata built here is what the replay
visibility predicate reads, for the current chunk (queries and keys) and for
chunks retained in the KV ring (keys only).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import msgspec
import torch

ROLE_UND = 0
ROLE_CONTROL = 1
ROLE_TARGET_CONDITION = 2
ROLE_CURRENT_TARGET = 3
ROLE_CLEAN_TARGET = 4

# Camera tokens read their own camera's captions; LiDAR tokens read every caption.
CAPTION_SCOPE_SAME_VIEW = 1
CAPTION_SCOPE_ALL = 2

PASS_NOISY = "noisy"
PASS_CLEAN = "clean"

ITEM_KINDS = ("rgb_control", "rgb_target", "lidar_control", "lidar_target")


def latent_frames_from_pixel_frames(num_frames: int, temporal_compression: int) -> int:
    """Wan ``4k + 1`` pixel frames give ``k + 1`` latent frames."""
    if num_frames < 1 or temporal_compression < 1:
        raise ValueError("num_frames and temporal_compression must be positive.")
    if (num_frames - 1) % temporal_compression:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto requires num_frames = {temporal_compression}k + 1 pixel frames, "
            f"got {num_frames}."
        )
    return 1 + (num_frames - 1) // temporal_compression


def lidar_num_sweeps(num_frames: int, *, camera_fps: float, lidar_fps: float) -> int:
    """Sweeps captured during ``num_frames`` camera frames (never fewer than one)."""
    if num_frames < 1 or camera_fps <= 0 or lidar_fps <= 0:
        raise ValueError("lidar_num_sweeps needs positive frame count and rates.")
    return max(1, round(num_frames * lidar_fps / camera_fps))


class Chunk(msgspec.Struct, frozen=True):
    """One causal step: absolute RGB latent indexes (per camera) and LiDAR sweep indexes."""

    step: int
    rgb_frames: tuple[int, ...]
    lidar_sweeps: tuple[int, ...] = ()

    @property
    def is_first(self) -> bool:
        return self.step == 0


def build_chunk_schedule(
    *,
    rgb_latents: int,
    lidar_sweeps: int | None,
    frames_per_chunk: int,
    rgb_seconds_per_frame: float,
    lidar_seconds_per_frame: float | None = None,
) -> tuple[Chunk, ...]:
    """Assign every RGB latent and LiDAR sweep to a causal step.

    RGB latent ``f`` belongs to step ``ceil(f / frames_per_chunk - 1e-5)`` and
    LiDAR sweep ``s`` to ``ceil(s * lidar_seconds / chunk_seconds - 1e-5)``, the
    training partition: latent 0 and sweep 0 alone in step 0, then
    ``frames_per_chunk`` latents and the sweeps of the same span per step.
    """
    if rgb_latents < 1 or frames_per_chunk < 1:
        raise ValueError(
            "The chunk schedule requires a positive latent count and chunk size."
        )
    if not (math.isfinite(rgb_seconds_per_frame) and rgb_seconds_per_frame > 0):
        raise ValueError(
            f"rgb_seconds_per_frame must be finite and positive, got {rgb_seconds_per_frame}."
        )
    chunk_seconds = frames_per_chunk * rgb_seconds_per_frame
    rgb_steps = [
        math.ceil(frame / frames_per_chunk - 1e-5) for frame in range(rgb_latents)
    ]
    lidar_steps: list[int] = []
    if lidar_sweeps is not None:
        if lidar_sweeps < 1:
            raise ValueError(f"lidar_sweeps must be positive, got {lidar_sweeps}.")
        if lidar_seconds_per_frame is None or not (
            math.isfinite(lidar_seconds_per_frame) and lidar_seconds_per_frame > 0
        ):
            raise ValueError(
                "Joint schedules require a finite positive lidar_seconds_per_frame."
            )
        lidar_steps = [
            math.ceil(sweep * lidar_seconds_per_frame / chunk_seconds - 1e-5)
            for sweep in range(lidar_sweeps)
        ]
    chunks = []
    for step in sorted(set(rgb_steps) | set(lidar_steps)):
        chunk = Chunk(
            step=step,
            rgb_frames=tuple(
                frame for frame, value in enumerate(rgb_steps) if value == step
            ),
            lidar_sweeps=tuple(
                sweep for sweep, value in enumerate(lidar_steps) if value == step
            ),
        )
        if lidar_sweeps is not None and (
            not chunk.rgb_frames or not chunk.lidar_sweeps
        ):
            raise ValueError(
                f"Joint chunk {step} must contain both RGB latents and LiDAR sweeps, got {chunk}."
            )
        chunks.append(chunk)
    return tuple(chunks)


class SensorGrid(msgspec.Struct, frozen=True):
    """Transformer token grid of one latent frame (or sweep) after spatial patching."""

    latent_height: int
    latent_width: int
    patch_height: int
    patch_width: int

    def __post_init__(self) -> None:
        if (
            min(
                self.latent_height,
                self.latent_width,
                self.patch_height,
                self.patch_width,
            )
            < 1
        ):
            raise ValueError("SensorGrid dimensions must be positive.")

    @property
    def grid_height(self) -> int:
        return -(-self.latent_height // self.patch_height)

    @property
    def grid_width(self) -> int:
        return -(-self.latent_width // self.patch_width)

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_height * self.grid_width


def rgb_sensor_grid(
    *, height: int, width: int, vae_spatial_compression: int, patch_size: int
) -> SensorGrid:
    """480x832 -> latent 30x52 -> 15x26 = 390 tokens per camera per latent frame."""
    if height % vae_spatial_compression or width % vae_spatial_compression:
        raise ValueError(
            f"Resolution {height}x{width} must be a multiple of {vae_spatial_compression}."
        )
    return SensorGrid(
        height // vae_spatial_compression,
        width // vae_spatial_compression,
        patch_size,
        patch_size,
    )


def lidar_sensor_grid(
    *, latent_height: int, latent_width: int, patch_hw: tuple[int, int]
) -> SensorGrid:
    """V1.2 LiDAR: 128x1808 -> latent 8x113 -> 8x113 = 904 tokens per sweep with a 1x1 patch."""
    return SensorGrid(latent_height, latent_width, patch_hw[0], patch_hw[1])


class StreamItem(msgspec.Struct, frozen=True):
    """One packed sensor item of a chunk.

    ``view_ids`` are the request's view ids in packing order (LiDAR uses the
    single id one past the cameras); ``frames`` are absolute latent/sweep
    indexes shared by every view of the item. ``condition_frames`` marks the
    frames of a target item whose latents are observed rather than generated.
    """

    kind: str
    view_ids: tuple[int, ...]
    frames: tuple[int, ...]
    grid: SensorGrid
    seconds_per_frame: float
    condition_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ITEM_KINDS:
            raise ValueError(f"Unknown stream item kind {self.kind!r}.")
        if not self.view_ids or len(set(self.view_ids)) != len(self.view_ids):
            raise ValueError(
                f"Stream item view ids must be non-empty and unique, got {self.view_ids}."
            )
        if (
            not self.frames
            or any(f < 0 for f in self.frames)
            or list(self.frames) != sorted(self.frames)
        ):
            raise ValueError(
                f"Stream item frames must be non-empty, non-negative and increasing, got {self.frames}."
            )
        if self.seconds_per_frame <= 0:
            raise ValueError(
                f"seconds_per_frame must be positive, got {self.seconds_per_frame}."
            )
        if self.is_control and self.condition_frames:
            raise ValueError(
                "Control items are fully observed; condition_frames applies to targets only."
            )
        if not set(self.condition_frames) <= set(self.frames):
            raise ValueError(
                f"condition_frames {self.condition_frames} must be frames of the item {self.frames}."
            )

    @property
    def is_control(self) -> bool:
        return self.kind.endswith("_control")

    @property
    def is_lidar(self) -> bool:
        return self.kind.startswith("lidar_")

    @property
    def tokens_per_frame(self) -> int:
        return self.grid.tokens_per_frame

    @property
    def num_tokens(self) -> int:
        return len(self.view_ids) * len(self.frames) * self.tokens_per_frame

    @property
    def caption_scope(self) -> int:
        return CAPTION_SCOPE_ALL if self.is_lidar else CAPTION_SCOPE_SAME_VIEW

    @property
    def generated_frames(self) -> tuple[int, ...]:
        return tuple(
            frame for frame in self.frames if frame not in self.condition_frames
        )


def build_chunk_items(
    chunk: Chunk,
    *,
    num_views: int,
    rgb_grid: SensorGrid,
    rgb_seconds_per_frame: float,
    lidar_grid: SensorGrid | None = None,
    lidar_seconds_per_frame: float | None = None,
    rgb_condition_frames: Iterable[int] = (),
    lidar_condition_sweeps: Iterable[int] = (),
) -> tuple[StreamItem, ...]:
    """Reference item order for one chunk: RGB control, RGB target, LiDAR control, LiDAR target."""
    if num_views < 1:
        raise ValueError(f"num_views must be positive, got {num_views}.")
    cameras = tuple(range(num_views))
    rgb_condition = tuple(sorted(set(rgb_condition_frames) & set(chunk.rgb_frames)))
    items = [
        StreamItem(
            "rgb_control", cameras, chunk.rgb_frames, rgb_grid, rgb_seconds_per_frame
        ),
        StreamItem(
            "rgb_target",
            cameras,
            chunk.rgb_frames,
            rgb_grid,
            rgb_seconds_per_frame,
            rgb_condition,
        ),
    ]
    if chunk.lidar_sweeps:
        if lidar_grid is None or lidar_seconds_per_frame is None:
            raise ValueError(
                "Joint chunks require lidar_grid and lidar_seconds_per_frame."
            )
        lidar_view = (num_views,)
        lidar_condition = tuple(
            sorted(set(lidar_condition_sweeps) & set(chunk.lidar_sweeps))
        )
        items.append(
            StreamItem(
                "lidar_control",
                lidar_view,
                chunk.lidar_sweeps,
                lidar_grid,
                lidar_seconds_per_frame,
            )
        )
        items.append(
            StreamItem(
                "lidar_target",
                lidar_view,
                chunk.lidar_sweeps,
                lidar_grid,
                lidar_seconds_per_frame,
                lidar_condition,
            )
        )
    return tuple(items)


def chunk_token_count(items: Sequence[StreamItem]) -> int:
    return sum(item.num_tokens for item in items)


class TokenMetadata(msgspec.Struct, eq=False):
    """Per-token predicate fields of one key or query stream (CPU tensors, ``[N]``)."""

    view_id: torch.Tensor
    frame_id: torch.Tensor
    timestamp: torch.Tensor
    role: torch.Tensor
    step: torch.Tensor
    caption_scope: torch.Tensor

    def __post_init__(self) -> None:
        length = self.view_id.numel()
        for name in ("frame_id", "timestamp", "role", "step", "caption_scope"):
            if getattr(self, name).numel() != length:
                raise ValueError(
                    f"TokenMetadata.{name} has {getattr(self, name).numel()} entries, expected {length}."
                )

    @property
    def num_tokens(self) -> int:
        return int(self.view_id.numel())

    @staticmethod
    def concat(parts: Sequence[TokenMetadata]) -> TokenMetadata:
        if not parts:
            raise ValueError("TokenMetadata.concat requires at least one part.")
        return TokenMetadata(
            view_id=torch.cat([p.view_id for p in parts]),
            frame_id=torch.cat([p.frame_id for p in parts]),
            timestamp=torch.cat([p.timestamp for p in parts]),
            role=torch.cat([p.role for p in parts]),
            step=torch.cat([p.step for p in parts]),
            caption_scope=torch.cat([p.caption_scope for p in parts]),
        )


def item_token_metadata(
    item: StreamItem, *, step: int, pass_kind: str
) -> TokenMetadata:
    """Camera-major, frame-inner, spatial-innermost token metadata of one item.

    Timestamps are absolute (``frame * seconds_per_frame``) so the cross-view
    window applies identically to current and retained keys; every token of a
    chunk carries the chunk step, the shared clock that decides causality.
    """
    if pass_kind not in (PASS_NOISY, PASS_CLEAN):
        raise ValueError(f"pass_kind must be 'noisy' or 'clean', got {pass_kind!r}.")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")
    spatial = item.tokens_per_frame
    frames = torch.tensor(item.frames, dtype=torch.int64)
    if item.is_control:
        frame_roles = torch.full_like(frames, ROLE_CONTROL)
    else:
        generated_role = (
            ROLE_CURRENT_TARGET if pass_kind == PASS_NOISY else ROLE_CLEAN_TARGET
        )
        is_condition = torch.tensor(
            [f in item.condition_frames for f in item.frames], dtype=torch.bool
        )
        frame_roles = torch.where(
            is_condition, torch.full_like(frames, ROLE_TARGET_CONDITION), generated_role
        )
    num_views = len(item.view_ids)
    frame_id = frames.repeat_interleave(spatial).repeat(num_views)
    role = frame_roles.repeat_interleave(spatial).repeat(num_views)
    view_id = torch.tensor(item.view_ids, dtype=torch.int64).repeat_interleave(
        len(item.frames) * spatial
    )
    return TokenMetadata(
        view_id=view_id,
        frame_id=frame_id,
        timestamp=frame_id.to(torch.float32) * item.seconds_per_frame,
        role=role,
        step=torch.full_like(frame_id, step),
        caption_scope=torch.full_like(frame_id, item.caption_scope),
    )


def chunk_token_metadata(
    items: Sequence[StreamItem], *, step: int, pass_kind: str
) -> TokenMetadata:
    if not items:
        raise ValueError("A chunk layout needs at least one stream item.")
    return TokenMetadata.concat(
        [item_token_metadata(item, step=step, pass_kind=pass_kind) for item in items]
    )


def committed_metadata(metadata: TokenMetadata) -> TokenMetadata:
    """What a ring slot stores after a chunk's clean commit: generated targets become clean targets."""
    role = torch.where(
        metadata.role == ROLE_CURRENT_TARGET,
        torch.full_like(metadata.role, ROLE_CLEAN_TARGET),
        metadata.role,
    )
    return TokenMetadata(
        view_id=metadata.view_id,
        frame_id=metadata.frame_id,
        timestamp=metadata.timestamp,
        role=role,
        step=metadata.step,
        caption_scope=metadata.caption_scope,
    )


class CaptionDocument(msgspec.Struct, frozen=True):
    """One causal text document: a camera's sink prefix or its caption.

    ``start_seconds``/``end_seconds`` bound the query timestamps that may read
    the document (half-open); sink documents are visible at every time.
    """

    view_id: int
    num_tokens: int
    is_sink: bool = False
    start_seconds: float = -math.inf
    end_seconds: float = math.inf

    def __post_init__(self) -> None:
        if self.num_tokens < 1:
            raise ValueError(
                f"Caption documents need at least one token, got {self.num_tokens}."
            )
        if self.is_sink and (
            self.start_seconds != -math.inf or self.end_seconds != math.inf
        ):
            raise ValueError(
                "Sink documents are visible at every time; do not bound them."
            )
        if self.start_seconds >= self.end_seconds:
            raise ValueError(
                f"Caption bounds must be increasing, got [{self.start_seconds}, {self.end_seconds})."
            )


def build_caption_documents(
    *, lengths: Sequence[int], sink_tokens: int, horizon_seconds: float
) -> tuple[CaptionDocument, ...]:
    """Per camera, in packing order: the sink prefix (when any) then the caption, bounded to the clip."""
    if sink_tokens < 0 or horizon_seconds <= 0:
        raise ValueError(
            "Caption documents need a non-negative sink size and a positive horizon."
        )
    documents: list[CaptionDocument] = []
    for view, length in enumerate(lengths):
        if sink_tokens:
            documents.append(CaptionDocument(view, sink_tokens, is_sink=True))
        documents.append(
            CaptionDocument(
                view, int(length), start_seconds=0.0, end_seconds=horizon_seconds
            )
        )
    return tuple(documents)


class CaptionMetadata(msgspec.Struct, eq=False):
    """Per-token caption fields: owning view, sink flag, time bounds and document id (CPU ``[U]``)."""

    view_id: torch.Tensor
    is_sink: torch.Tensor
    start_seconds: torch.Tensor
    end_seconds: torch.Tensor
    doc_id: torch.Tensor
    doc_offsets: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return int(self.view_id.numel())


def caption_metadata(documents: Sequence[CaptionDocument]) -> CaptionMetadata:
    """Expand documents in packing order ``[sink_v0 | caption_v0 | sink_v1 | ...]``."""
    if not documents:
        raise ValueError("Caption metadata requires at least one document.")
    lengths = torch.tensor([doc.num_tokens for doc in documents], dtype=torch.int64)
    offsets = [0]
    for length in lengths.tolist():
        offsets.append(offsets[-1] + length)
    return CaptionMetadata(
        view_id=torch.tensor(
            [doc.view_id for doc in documents], dtype=torch.int64
        ).repeat_interleave(lengths),
        is_sink=torch.tensor(
            [doc.is_sink for doc in documents], dtype=torch.bool
        ).repeat_interleave(lengths),
        start_seconds=torch.tensor(
            [doc.start_seconds for doc in documents], dtype=torch.float32
        ).repeat_interleave(lengths),
        end_seconds=torch.tensor(
            [doc.end_seconds for doc in documents], dtype=torch.float32
        ).repeat_interleave(lengths),
        doc_id=torch.arange(len(documents), dtype=torch.int64).repeat_interleave(
            lengths
        ),
        doc_offsets=tuple(offsets[:-1]),
    )


def ring_slot_for_step(step: int, *, cache_chunks: int, sink_chunks: int) -> int:
    """Reference slot assignment: sinks are pinned, later steps rotate through the rest."""
    if step < 0 or cache_chunks <= sink_chunks or sink_chunks < 0:
        raise ValueError(
            "ring_slot_for_step needs step >= 0 and cache_chunks > sink_chunks >= 0."
        )
    if step < sink_chunks:
        return step
    return sink_chunks + (step - sink_chunks) % (cache_chunks - sink_chunks)
