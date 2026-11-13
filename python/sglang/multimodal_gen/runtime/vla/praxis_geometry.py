# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 chunk geometry, rotations, normalization and the observed context stream."""

from __future__ import annotations

from collections.abc import Callable

import msgspec
import numpy as np
import torch

FPS = 15.0
ACTION_CHUNK = 24
TEMPORAL_RATIO = 4
FRAMES_PER_TOKEN = 4
# Tiles whose aspect differs from the source by more than this get a centre crop first.
ASPECT_TOL = 0.03
GRIP_RANGE = (0.0, 1.0)


def latent_frames(raw_frames: int, temporal_ratio: int = TEMPORAL_RATIO) -> int:
    return (int(raw_frames) - 1) // int(temporal_ratio) + 1


def contiguous_chunk_geometry(
    action_chunk: int, n_chunks: int
) -> list[tuple[int, int]]:
    """``(raw_offset, raw_len)`` per chunk; chunk 0 owns the anchor latent, so it spans 4 * L - 3 raws."""
    lat = (action_chunk - 1) // 4 + 1
    out: list[tuple[int, int]] = []
    offset = 0
    for chunk in range(n_chunks):
        length = 4 * lat - 3 if chunk == 0 else 4 * lat
        out.append((offset, length))
        offset += length
    return out


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` wxyz quaternions to ``(..., 3, 3)``, normalized with a 1e-8 guard (proprio convention)."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return r.reshape(*q.shape[:-1], 3, 3)


def unit_quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """One wxyz quaternion to a float64 matrix with an exact normalization (command convention).

    The scalar operation order is part of the contract: it makes the result bitwise reproducible.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = (float(v) for v in q)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Found zero norm quaternions in `quat`.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.array(
        [
            [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
        ],
        dtype=np.float64,
    )


def rot6d(r: np.ndarray) -> np.ndarray:
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """One rot6d ``[c0(3), c1(3)]`` to a float64 matrix: Gram-Schmidt, third column by cross product."""
    a0 = r6[:3].astype(np.float64)
    a1 = r6[3:6].astype(np.float64)
    c0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1 = a1 - np.dot(c0, a1) * c0
    c1 = a1 / (np.linalg.norm(a1) + 1e-8)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=1)


def pose8_to_10d(pose8: np.ndarray, anchor8: np.ndarray | None = None) -> np.ndarray:
    """``[xyz3, quat_wxyz4, grip1]`` to ``[xyz3, rot6d6, grip1]``, relative to ``anchor8`` when given."""
    xyz = pose8[..., :3]
    r = quat_wxyz_to_matrix(pose8[..., 3:7])
    grip = pose8[..., 7:8]
    if anchor8 is not None:
        xyz = xyz - anchor8[..., :3]
        r_anchor = quat_wxyz_to_matrix(anchor8[..., 3:7])
        r = np.einsum("...ij,...kj->...ik", r, r_anchor)
    return np.concatenate([xyz, rot6d(r), grip], axis=-1).astype(np.float32)


class NormStats(msgspec.Struct, frozen=True, eq=False):
    """Normalization statistics of the actions and the proprio (float32 arrays)."""

    action_q01: np.ndarray
    action_q99: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray

    @classmethod
    def from_config(cls, normalization: dict) -> NormStats:
        return cls(
            action_q01=np.asarray(normalization["action_q01"], dtype=np.float32),
            action_q99=np.asarray(normalization["action_q99"], dtype=np.float32),
            proprio_mean=np.asarray(normalization["proprio_mean"], dtype=np.float32),
            proprio_std=np.asarray(normalization["proprio_std"], dtype=np.float32),
        )


def q99_denormalize(
    normalized: np.ndarray, q01: np.ndarray, q99: np.ndarray, dims: np.ndarray
) -> np.ndarray:
    """``(n + 1) / 2 * (q99 - q01) + q01`` on ``dims``."""
    out = normalized.astype(np.float32).copy()
    n = normalized[..., dims].astype(np.float32)
    lo = q01[dims].astype(np.float32)
    hi = q99[dims].astype(np.float32)
    out[..., dims] = (n + 1.0) / 2.0 * (hi - lo) + lo
    return out.astype(np.float32)


def clip_normalized(rows: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Float32 copy of ``rows`` with ``dims`` clipped to [-1, 1], the range of the training targets."""
    out = np.asarray(rows, dtype=np.float32).copy()
    out[..., dims] = np.clip(out[..., dims], -1.0, 1.0)
    return out


def clip_grip(rel: np.ndarray, dims: np.ndarray) -> np.ndarray:
    out = np.asarray(rel, dtype=np.float32).copy()
    out[..., dims] = np.clip(out[..., dims], *GRIP_RANGE)
    return out


def denormalize_rows(
    rows: np.ndarray, stats: NormStats, dims: np.ndarray
) -> np.ndarray:
    return q99_denormalize(
        clip_normalized(rows, dims), stats.action_q01, stats.action_q99, dims=dims
    )


def proprio_normalize(
    proprio: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """``(p - mean) / max(std, 1e-6)`` in float32."""
    s = np.maximum(std.astype(np.float32), np.float32(1e-6))
    return ((proprio.astype(np.float32) - mean.astype(np.float32)) / s).astype(
        np.float32
    )


def aspect_crop_box(
    src_h: int, src_w: int, tile_h: int, tile_w: int, tol: float = ASPECT_TOL
) -> tuple[int, int, int, int]:
    """Centre crop ``(y0, y1, x0, x1)`` to the tile aspect; the full frame when the aspects agree within ``tol``."""
    src_ar, tile_ar = src_w / src_h, tile_w / tile_h
    if abs(tile_ar / src_ar - 1.0) <= tol:
        return 0, src_h, 0, src_w
    if src_ar > tile_ar:
        new_w = round(src_h * tile_ar)
        x0 = (src_w - new_w) // 2
        return 0, src_h, x0, x0 + new_w
    new_h = round(src_w / tile_ar)
    y0 = (src_h - new_h) // 2
    return y0, y0 + new_h, 0, src_w


def aspect_crop(
    fr: torch.Tensor, tile_h: int, tile_w: int, tol: float = ASPECT_TOL
) -> torch.Tensor:
    y0, y1, x0, x1 = aspect_crop_box(fr.shape[-2], fr.shape[-1], tile_h, tile_w, tol)
    if (y0, y1, x0, x1) == (0, fr.shape[-2], 0, fr.shape[-1]):
        return fr
    return fr[..., y0:y1, x0:x1]


class ContiguousStreamContext:
    """Raw frame and pose streams since the start of the context window.

    The model trains on contiguous windows that the video encoder encodes in one pass: an anchor
    latent, then chunks of ``lat_per_chunk`` latents. Serving appends the executed real frames and
    re-encodes the retained prefix on each call; the clean context chunks are slices of that encode.
    The causal encoder makes this equal to one encode of the whole window. When the window overflows
    ``max_context_chunks`` clean chunks, the oldest ``cont_len`` raws are dropped and the retained
    prefix starts a new window with a new anchor frame.
    """

    def __init__(self, action_chunk: int = ACTION_CHUNK, max_context_chunks: int = 3):
        geometry = contiguous_chunk_geometry(action_chunk, max_context_chunks + 1)
        self.first_len = geometry[0][1]
        self.cont_len = geometry[1][1] if len(geometry) > 1 else 0
        self.lat_per_chunk = (action_chunk - 1) // 4 + 1
        self.max_context_chunks = int(max_context_chunks)
        self.max_raws = self.first_len + self.cont_len * (self.max_context_chunks - 1)
        self._clip: torch.Tensor | None = None
        self._poses: np.ndarray | None = None

    def reset(self) -> None:
        self._clip = None
        self._poses = None

    @property
    def n_raws(self) -> int:
        return 0 if self._clip is None else int(self._clip.shape[2])

    @property
    def depth(self) -> int:
        """Number of complete clean chunks in the stream."""
        raws = self.n_raws
        if raws < self.first_len:
            return 0
        return min(
            1 + (raws - self.first_len) // self.cont_len, self.max_context_chunks
        )

    @property
    def expected_next_rows(self) -> int:
        return self.first_len if self.n_raws == 0 else self.cont_len

    def boundary_offsets(self) -> list[int]:
        offsets = [
            offset
            for offset, _length in contiguous_chunk_geometry(
                self.cont_len, self.max_context_chunks
            )
        ]
        return offsets[: self.depth]

    def append_executed(
        self, clip: torch.Tensor, poses: np.ndarray | None = None
    ) -> None:
        """Append the executed chunk (a ``(1, 3, T, H, W)`` clip) and slide the window on overflow.

        The pose stream stays valid only while every call supplies poses.
        """
        raws = int(clip.shape[2])
        if raws != self.expected_next_rows:
            raise ValueError(
                f"executed chunk spans {raws} raws, expected {self.expected_next_rows} "
                f"(window chunk {'0' if self.n_raws == 0 else 'k>=1'})"
            )
        was_empty = self._clip is None
        self._clip = (
            clip if self._clip is None else torch.cat([self._clip, clip], dim=2)
        )
        aligned = self._poses is not None or was_empty
        if poses is not None and len(poses) == raws and aligned:
            rows = np.asarray(poses, dtype=np.float32)
            self._poses = (
                rows
                if self._poses is None
                else np.concatenate([self._poses, rows], axis=0)
            )
        else:
            self._poses = None
        if self.n_raws > self.max_raws:
            drop = self.n_raws - self.max_raws
            self._clip = self._clip[:, :, drop:]
            if self._poses is not None:
                self._poses = self._poses[drop:]

    def cleans(
        self, encode: Callable[[torch.Tensor], torch.Tensor]
    ) -> list[torch.Tensor]:
        """Encode the retained prefix once and return the clean chunks, oldest first."""
        depth = self.depth
        if depth == 0:
            return []
        n_raws = self.first_len + self.cont_len * (depth - 1)
        latents = encode(self._clip[:, :, :n_raws])
        n = self.lat_per_chunk
        return [latents.narrow(-3, k * n, n).contiguous() for k in range(depth)]

    def anchor_frame(self) -> torch.Tensor | None:
        return None if self._clip is None else self._clip[:, :, :1]

    def boundary_poses(self) -> np.ndarray | None:
        if self._poses is None or self.depth == 0:
            return None
        if len(self._poses) != self.n_raws:
            return None
        offsets = self.boundary_offsets()
        if not offsets or offsets[-1] >= len(self._poses):
            return None
        return np.stack([self._poses[o] for o in offsets], axis=0)


def action_rows(
    pred_action: torch.Tensor, depth: int, frames_per_token: int, action_dim: int
) -> np.ndarray:
    """Unpack predicted ``(1, L, F * D)`` tokens to rows: 21 at depth 0 (token 0 owns the anchor), 24 later."""
    a = pred_action.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 3:
        a = a[0]
    n_latents, packed = a.shape
    per_frame = packed // frames_per_token
    rows: list[np.ndarray] = []
    for k in range(n_latents):
        token = a[k].reshape(frames_per_token, per_frame)
        for f in range(frames_per_token):
            if k == 0 and f > 0 and depth == 0:
                continue
            rows.append(token[f, :action_dim])
    return np.stack(rows, axis=0)


def text_encoder_prompt(prompt: str) -> str:
    """A blank prompt becomes one space, the blank caption of the checkpoint."""
    return prompt if prompt.strip() else " "
