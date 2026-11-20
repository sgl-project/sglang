# SPDX-License-Identifier: Apache-2.0
"""Camera-frustum K/V retrieval for the Cosmos-Dreams rollout.

Port of the imaginaire4 retrieval baseline. The archive keeps every committed
latent frame; each generated chunk attends to the ``sink_size`` leading frames,
the ``recent_size`` newest frames and up to ``retrieved_size`` older frames whose
view frustums overlap the chunk's the most. Selection is deterministic CPU
float64 geometry, computed once per chunk and shared by every layer, denoising
step and clean refresh of that chunk. Only torch is imported here so the
selector can be checked against the reference implementation in its own
environment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

KVPair = tuple[torch.Tensor, torch.Tensor]

DEFAULT_SINK_SIZE = 8
DEFAULT_RETRIEVED_SIZE = 32
DEFAULT_RECENT_SIZE = 32
# Poses recovered from the unified rows' ego slots; the reference names the
# AgiBot source explicitly and reuses the aligned camera poses for camera
# requests. Here both come from the submitted rows.
POSE_SOURCE_UNIFIED_ROWS = "agibot_unified_v1_chunk16"
POSE_SOURCE_CAMERA_BATCH = "camera_batch"
POSE_SOURCES = (POSE_SOURCE_UNIFIED_ROWS, POSE_SOURCE_CAMERA_BATCH)
EGO_POSE_SLOTS = slice(0, 9)
POSE_CHUNK_FRAMES = 16

_GRID_SIZE = 8
_HISTORY_BATCH_SIZE = 64
_ASINH_ONE = math.asinh(1.0)


class FrustumHistorySelector:
    """Select sinks, recent history and older frames with high approximate IoU.

    Geometry is aligned to latent frames. Poses are OpenCV camera-to-world
    (+x right, +y down, +z forward); intrinsics are in pixels; ``image_size``
    is ``(width, height)``. Each frustum is sampled on a fixed 8^3 midpoint grid
    (u/v uniform in pixels, z^3 uniform between near^3 and far^3, so samples are
    uniform in volume). The intersection estimate is the mean of the two
    directional inside fractions times their source volumes, clamped to the
    smaller volume. Returned IDs are original logical frame indices; the
    selector never stores or evicts K/V.
    """

    def __init__(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        near: float,
        far: float,
        *,
        sink_size: int = DEFAULT_SINK_SIZE,
        retrieved_size: int = DEFAULT_RETRIEVED_SIZE,
        recent_size: int = DEFAULT_RECENT_SIZE,
    ) -> None:
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or c2w.shape[0] == 0:
            raise ValueError("c2w must have shape [T, 4, 4] with T > 0")
        count = c2w.shape[0]
        if intrinsics.shape not in ((3, 3), (count, 3, 3)):
            raise ValueError("intrinsics must have shape [3, 3] or [T, 3, 3]")
        if c2w.is_complex() or intrinsics.is_complex():
            raise ValueError("geometry must be real and finite")
        if len(image_size) != 2 or any(
            type(size) is not int or size <= 0 for size in image_size
        ):
            raise ValueError("image_size must contain positive integer WIDTH, HEIGHT")
        if not math.isfinite(near) or not math.isfinite(far) or not 0 < near < far:
            raise ValueError("depth bounds must be finite and satisfy 0 < near < far")
        if any(
            type(size) is not int or size < 0
            for size in (sink_size, retrieved_size, recent_size)
        ):
            raise ValueError("selection sizes must be nonnegative integers")

        self._poses = c2w.detach().to(device="cpu", dtype=torch.float64).clone()
        self._intrinsics = intrinsics.detach().to(device="cpu", dtype=torch.float64)
        if self._intrinsics.ndim == 2:
            self._intrinsics = self._intrinsics.expand(count, -1, -1)
        self._intrinsics = self._intrinsics.clone()
        if (
            not torch.isfinite(self._poses).all()
            or not torch.isfinite(self._intrinsics).all()
        ):
            raise ValueError("geometry must be real and finite")
        last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(
            count, -1
        )
        if not torch.allclose(self._poses[:, 3], last_row, atol=1e-6, rtol=0):
            raise ValueError("c2w must have homogeneous last row [0, 0, 0, 1]")
        rotation = self._poses[:, :3, :3]
        identity = torch.eye(3, dtype=torch.float64).expand(count, -1, -1)
        if not torch.allclose(
            rotation.transpose(1, 2) @ rotation, identity, atol=1e-4, rtol=0
        ) or not torch.allclose(
            torch.linalg.det(rotation),
            torch.ones(count, dtype=torch.float64),
            atol=1e-4,
            rtol=0,
        ):
            raise ValueError("c2w rotation must be orthonormal with determinant +1")
        k = self._intrinsics
        k_last_row = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(
            count, -1
        )
        if not torch.allclose(k[:, 2], k_last_row, atol=1e-8, rtol=0) or not (
            torch.allclose(
                k[:, 1, 0], torch.zeros(count, dtype=torch.float64), atol=1e-8, rtol=0
            )
        ):
            raise ValueError(
                "intrinsics must be upper triangular with last row [0, 0, 1]"
            )
        if (k[:, 0, 0] <= 0).any() or (k[:, 1, 1] <= 0).any():
            raise ValueError("intrinsics must have positive focal lengths")

        self._width, self._height = image_size
        self._near, self._far = near, far
        self.sink_size = sink_size
        self.retrieved_size = retrieved_size
        self.recent_size = recent_size
        quantiles = (torch.arange(_GRID_SIZE, dtype=torch.float64) + 0.5) / _GRID_SIZE
        near_cubed = torch.tensor(near, dtype=torch.float64).pow(3)
        depth_volume = torch.tensor(far, dtype=torch.float64).pow(3) - near_cubed
        self._volumes = (
            (self._width * self._height / (k[:, 0, 0] * k[:, 1, 1])) * depth_volume / 3
        )
        if not torch.isfinite(self._volumes).all() or (self._volumes <= 0).any():
            raise ValueError("frustum volumes must be finite and positive")
        u, v, z = torch.meshgrid(
            quantiles * self._width,
            quantiles * self._height,
            (near_cubed + quantiles * depth_volume).pow(1 / 3),
            indexing="ij",
        )
        pixels = torch.stack(
            (u.flatten(), v.flatten(), torch.ones(u.numel(), dtype=torch.float64)),
            dim=-1,
        )
        camera_points = (pixels @ torch.linalg.inv(k).transpose(1, 2)) * z.flatten()[
            None, :, None
        ]
        self._world_points = (
            camera_points @ rotation.transpose(1, 2) + self._poses[:, None, :3, 3]
        )
        if not torch.isfinite(self._world_points).all():
            raise ValueError("frustum samples must be finite")

    @classmethod
    def from_rgb_geometry(
        cls,
        geometry: dict[str, Any],
        *,
        num_latent_frames: int,
        temporal_compression_factor: int,
        fps: float,
    ) -> FrustumHistorySelector:
        """Build the selector from RGB-frame poses and explicit calibration.

        Latent frame 0 is RGB frame 0 and latent frame f > 0 ends at RGB frame
        ``f * temporal_compression_factor``; poses are never resampled.
        """
        missing = {"c2w", "image_size", "near", "far", "fps"} - geometry.keys()
        if missing:
            raise ValueError(f"Frustum geometry is missing fields: {sorted(missing)}")
        if float(geometry["fps"]) != float(fps):
            raise ValueError(
                f"Frustum geometry FPS {geometry['fps']} does not match inference FPS {fps}"
            )
        c2w = torch.as_tensor(geometry["c2w"], dtype=torch.float64, device="cpu")
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError("Frustum c2w must have shape [RGB_frames, 4, 4]")
        rgb_indices = torch.arange(num_latent_frames) * temporal_compression_factor
        if c2w.shape[0] <= int(rgb_indices[-1]):
            raise ValueError(
                "Frustum trajectory is shorter than the generated RGB-frame horizon"
            )
        image_size = geometry["image_size"]
        if (
            not isinstance(image_size, (list, tuple))
            or len(image_size) != 2
            or any(type(size) is not int or size <= 0 for size in image_size)
        ):
            raise ValueError(
                "Frustum image_size must be [width, height] in positive integer pixels"
            )
        if "intrinsics" in geometry and "horizontal_fov_degrees" in geometry:
            raise ValueError("Specify intrinsics or horizontal_fov_degrees, not both")
        if "intrinsics" in geometry:
            intrinsics = torch.as_tensor(
                geometry["intrinsics"], dtype=torch.float64, device="cpu"
            )
        elif "horizontal_fov_degrees" in geometry:
            intrinsics = _fov_intrinsics(
                geometry["horizontal_fov_degrees"],
                width=image_size[0],
                height=image_size[1],
            )
        else:
            raise ValueError(
                "Frustum geometry is missing intrinsics or horizontal_fov_degrees"
            )
        if intrinsics.ndim == 3:
            if intrinsics.shape[0] != c2w.shape[0]:
                raise ValueError(
                    "Per-frame frustum intrinsics must match the RGB trajectory length"
                )
            intrinsics = intrinsics[rgb_indices]
        return cls(
            c2w[rgb_indices],
            intrinsics,
            (image_size[0], image_size[1]),
            near=float(geometry["near"]),
            far=float(geometry["far"]),
            sink_size=geometry.get("sink_size", DEFAULT_SINK_SIZE),
            retrieved_size=geometry.get("retrieved_size", DEFAULT_RETRIEVED_SIZE),
            recent_size=geometry.get("recent_size", DEFAULT_RECENT_SIZE),
        )

    @property
    def num_frames(self) -> int:
        return int(self._poses.shape[0])

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.sink_size, self.retrieved_size, self.recent_size

    @property
    def budget(self) -> int:
        """Most past frames a chunk attends to."""
        return self.sink_size + self.retrieved_size + self.recent_size

    def _validate_indices(self, indices: Sequence[int]) -> None:
        if any(
            type(index) is not int or not 0 <= index < self.num_frames
            for index in indices
        ):
            raise ValueError("frame indices must be integers in [0, T)")

    def _inside_fraction(
        self, points: torch.Tensor, target_indices: Sequence[int]
    ) -> torch.Tensor:
        poses = self._poses[target_indices]
        camera = (points - poses[:, None, :3, 3]) @ poses[:, :3, :3]
        pixels = camera @ self._intrinsics[target_indices].transpose(1, 2)
        depth = camera[..., 2]
        # Homogeneous pixel inequalities avoid dividing by depths behind the camera.
        inside = (
            (depth >= self._near)
            & (depth <= self._far)
            & (pixels[..., 0] >= 0)
            & (pixels[..., 0] <= self._width * depth)
            & (pixels[..., 1] >= 0)
            & (pixels[..., 1] <= self._height * depth)
        )
        return inside.to(torch.float64).mean(dim=-1)

    def overlap_scores(
        self, query_indices: Sequence[int], history_indices: Sequence[int]
    ) -> torch.Tensor:
        """Approximate IoU of every history frame against the query frames, averaged
        over the queries; history order is preserved."""
        if not query_indices:
            raise ValueError("query_indices must not be empty")
        self._validate_indices(query_indices)
        self._validate_indices(history_indices)
        scores = torch.zeros(len(history_indices), dtype=torch.float64)
        for start in range(0, len(history_indices), _HISTORY_BATCH_SIZE):
            history = list(history_indices[start : start + _HISTORY_BATCH_SIZE])
            history_volume = self._volumes[history]
            batch_scores = torch.zeros(len(history), dtype=torch.float64)
            for query in query_indices:
                query_volume = self._volumes[query]
                query_inside = self._inside_fraction(
                    self._world_points[query : query + 1], history
                )
                history_inside = self._inside_fraction(
                    self._world_points[history], [query]
                )
                intersection = (
                    query_inside * query_volume + history_inside * history_volume
                ) / 2
                intersection = torch.minimum(
                    intersection, torch.minimum(query_volume, history_volume)
                )
                batch_scores += (
                    intersection / (query_volume + history_volume - intersection)
                ).clamp(0, 1)
            scores[start : start + len(history)] = batch_scores / len(query_indices)
        return scores

    def select(self, frame_idx: int, chunk_end: int | None = None) -> list[int]:
        """Chronological sink + recent + retrieved frame IDs from ``[0, frame_idx)``
        for the chunk ``[frame_idx, chunk_end)``; equal scores prefer newer frames
        and short histories use every available frame."""
        self._validate_indices([frame_idx])
        if chunk_end is None:
            chunk_end = frame_idx + 1
        if type(chunk_end) is not int or not frame_idx < chunk_end <= self.num_frames:
            raise ValueError("chunk_end must satisfy frame_idx < chunk_end <= T")
        sink_end, recent_start = self.partition(frame_idx)
        sinks = list(range(sink_end))
        recent = list(range(recent_start, frame_idx))
        candidates = list(range(sink_end, recent_start))
        if len(candidates) <= self.retrieved_size:
            retrieved = candidates
        elif self.retrieved_size:
            scores = self.overlap_scores(
                list(range(frame_idx, chunk_end)), candidates
            ).tolist()
            ranked = sorted(
                zip(candidates, scores, strict=True),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
            retrieved = [index for index, _ in ranked[: self.retrieved_size]]
        else:
            retrieved = []
        return sorted(sinks + recent + retrieved)

    def partition(self, frame_idx: int) -> tuple[int, int]:
        """``(sink_end, recent_start)``: sinks are ``[0, sink_end)``, recent frames
        ``[recent_start, frame_idx)`` and retrieval candidates lie between."""
        sink_end = min(self.sink_size, frame_idx)
        return sink_end, max(sink_end, frame_idx - self.recent_size)


def _fov_intrinsics(fov: Any, *, width: int, height: int) -> torch.Tensor:
    """Centered pinhole with square pixels from a horizontal field of view."""
    if type(fov) not in (int, float) or not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError(
            "horizontal_fov_degrees must be finite and strictly between 0 and 180"
        )
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def denormalize_unified_slots(
    rows: torch.Tensor, *, offset: Sequence[float], scale: Sequence[float]
) -> torch.Tensor:
    """Invert the global-asinh normalizer, ``y = asinh((x - offset) / scale) / asinh(1)``."""
    values = rows.detach().to(device="cpu", dtype=torch.float64)
    offset_t = torch.tensor(list(offset), dtype=torch.float64)
    scale_t = torch.tensor(list(scale), dtype=torch.float64)
    if values.ndim != 2 or values.shape[1] != offset_t.numel() != scale_t.numel():
        raise ValueError(
            f"Expected rows of width {offset_t.numel()}, got {tuple(values.shape)}."
        )
    return offset_t + scale_t * torch.sinh(values * _ASINH_ONE)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Column-based 6-D rotation to a proper rotation matrix, projected onto SO(3)
    with an SVD like the reference decoder."""
    if rot6d.ndim < 1 or rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must have shape [..., 6], got {tuple(rot6d.shape)}")
    col0 = rot6d[..., :3]
    col1 = rot6d[..., 3:]
    col2 = torch.cross(col0, col1, dim=-1)
    matrices = torch.stack((col0, col1, col2), dim=-1)
    u, _, vt = torch.linalg.svd(matrices)
    rotation = u @ vt
    reflected = torch.linalg.det(rotation) < 0
    if reflected.any():
        u = u.clone()
        u[reflected, :, -1] = -u[reflected, :, -1]
        rotation = u @ vt
    return rotation


def chunk_anchored_poses(
    deltas: torch.Tensor, *, chunk_frames: int = POSE_CHUNK_FRAMES
) -> torch.Tensor:
    """Absolute poses from ``backward_chunk_anchored`` deltas: row i is
    ``T_anchor^-1 @ T_{i+1}`` with the anchor fixed at the chunk start; the first
    pose is the identity, which only sets the world gauge."""
    if deltas.ndim != 3 or deltas.shape[1:] != (4, 4):
        raise ValueError(f"deltas must have shape [T, 4, 4], got {tuple(deltas.shape)}")
    identity = torch.eye(4, dtype=deltas.dtype)
    poses = [identity]
    current = identity
    anchor = identity
    for index in range(deltas.shape[0]):
        if index > 0 and index % chunk_frames == 0:
            anchor = current
        current = anchor @ deltas[index]
        poses.append(current)
    return torch.stack(poses)


def recover_unified_camera_poses(
    rows: torch.Tensor, *, offset: Sequence[float], scale: Sequence[float]
) -> torch.Tensor:
    """OpenCV c2w poses ``[T + 1, 4, 4]`` from ``T`` normalized unified rows.

    Ego slots 0..8 carry ``[translation(3), rot6d(6)]`` for robot and camera rows
    alike; ``offset``/``scale`` are the normalizer statistics of those slots.
    """
    raw = denormalize_unified_slots(
        rows[:, EGO_POSE_SLOTS],
        offset=offset[EGO_POSE_SLOTS],
        scale=scale[EGO_POSE_SLOTS],
    )
    deltas = torch.eye(4, dtype=torch.float64).repeat(raw.shape[0], 1, 1)
    deltas[:, :3, :3] = rot6d_to_matrix(raw[:, 3:9])
    deltas[:, :3, 3] = raw[:, :3]
    return chunk_anchored_poses(deltas)


def resolve_frustum_selector(
    geometry: Any,
    *,
    rows: torch.Tensor | None,
    normalizer_stats: tuple[Sequence[float], Sequence[float]] | None,
    num_latent_frames: int,
    temporal_compression_factor: int,
    fps: float,
) -> FrustumHistorySelector:
    """Selector for one request from its ``frustum_geometry`` object.

    ``pose_source`` recovers the camera poses from the submitted unified rows
    with the checkpoint's normalizer statistics; explicit ``c2w`` is used as is.
    A missing ``fps`` takes the request's; a present one must match it.
    """
    if not isinstance(geometry, dict):
        raise ValueError("frustum_geometry must be a JSON object.")
    geometry = dict(geometry)
    if "pose_source" in geometry:
        if "c2w" in geometry:
            raise ValueError(
                "frustum_geometry takes explicit c2w or pose_source, not both."
            )
        source = geometry.pop("pose_source")
        if source not in POSE_SOURCES:
            raise ValueError(
                f"frustum_geometry pose_source must be one of {POSE_SOURCES}, got {source!r}."
            )
        if rows is None:
            raise ValueError(
                "frustum_geometry pose recovery needs the request's action rows."
            )
        if normalizer_stats is None:
            raise ValueError(
                "frustum_geometry pose recovery needs the unified (schema-5) action contract."
            )
        offset, scale = normalizer_stats
        geometry["c2w"] = recover_unified_camera_poses(rows, offset=offset, scale=scale)
    geometry.setdefault("fps", fps)
    return FrustumHistorySelector.from_rgb_geometry(
        geometry,
        num_latent_frames=num_latent_frames,
        temporal_compression_factor=temporal_compression_factor,
        fps=fps,
    )


class SelectedHistory(Sequence):
    """Read-only per-layer view of the archive restricted to ``frame_ids``.

    Indexable like ``list[KVPair]`` so the transformer gathers one layer at a
    time and frees it after attention instead of materializing every layer's
    selected K/V at once.
    """

    def __init__(
        self,
        archive: Sequence[KVPair],
        *,
        frame_ids: Sequence[int],
        tokens_per_frame: int,
    ) -> None:
        if not archive:
            raise ValueError("SelectedHistory needs a non-empty archive.")
        stored = archive[0][0].shape[1] // tokens_per_frame
        ids = list(frame_ids)
        if ids != sorted(set(ids)) or (ids and (ids[0] < 0 or ids[-1] >= stored)):
            raise ValueError(
                f"Selected frames must be ordered unique committed frames, got {ids} "
                f"with {stored} committed."
            )
        self._archive = archive
        self._frame_ids = tuple(ids)
        self._positions = torch.tensor(
            [
                frame * tokens_per_frame + i
                for frame in ids
                for i in range(tokens_per_frame)
            ],
            dtype=torch.long,
            device=archive[0][0].device,
        )

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return self._frame_ids

    def __len__(self) -> int:
        return len(self._archive)

    def __getitem__(self, layer: int) -> KVPair:  # type: ignore[override]
        key, value = self._archive[layer]
        return key.index_select(1, self._positions), value.index_select(
            1, self._positions
        )


def describe_selection(
    selector: FrustumHistorySelector, frame_idx: int, frame_ids: Sequence[int]
) -> str:
    sink_end, recent_start = selector.partition(frame_idx)
    retrieved = [index for index in frame_ids if sink_end <= index < recent_start]
    return (
        f"{len(frame_ids)} frames: {sink_end} sinks, {frame_idx - recent_start} recent, "
        f"retrieved {retrieved}"
    )
