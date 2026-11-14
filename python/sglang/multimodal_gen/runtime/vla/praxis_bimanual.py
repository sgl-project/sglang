# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 bimanual (YAM) rig: three-view canvas, two-arm poses, proprio, command decoding and the session state.

Observation (rig side): ``robot_state.{left,right}.ee_pose`` ``[x, y, z, qw, qx, qy, qz]`` in the world frame of
the rig, ``grip.{left,right}`` (0 = open) and RGB uint8 ``image.exterior`` (the top camera), ``image.wrist_left``
and ``image.wrist_right``; each may carry a leading time axis whose last row is now.
Model space: pose16 ``[L: xyz3 quat_wxyz4 grip1 | R: same]`` and 20-D ``[L: xyz3 rot6d6 grip1 | R: same]``. Each
arm's actions are relative to its own pose at the start of the chunk (the grippers are absolute); xyz and grip of
both arms are q99-normalized.
Commands: one dict per row at 15 Hz with ``robot_command.{left,right}`` (``pose`` is ``[xyz3, R_flat9]``),
``target_grip.{left,right}`` and ``timestamp``. The first chunk of a context window gives 21 rows, every later
chunk 24.
"""

from __future__ import annotations

from collections.abc import Callable

import msgspec
import numpy as np
import torch

from sglang.multimodal_gen.runtime.vla.praxis_geometry import (
    ACTION_CHUNK,
    FPS,
    FRAMES_PER_TOKEN,
    ContiguousStreamContext,
    NormStats,
    action_rows,
    clip_grip,
    denormalize_rows,
    latent_frames,
    pose8_to_10d,
    proprio_normalize,
    rot6d_to_matrix,
    text_encoder_prompt,
    unit_quat_wxyz_to_matrix,
)
from sglang.multimodal_gen.runtime.vla.praxis_rig import TILE_CHUNK_FRAMES

VIEWS: tuple[str, ...] = ("top_left", "wrist_left", "wrist_right")
SIDES: tuple[str, ...] = ("left", "right")
ACTION_DIM = 20
ARM_DIM = 10
# One command row: per arm the target position (3), row-major rotation matrix (9) and target grip (1).
COMMAND_DIM = 26
N_CHUNKS = 4
MAX_CONTEXT_CHUNKS = 3
TILE_SIZE: tuple[int, int] = (256, 416)
# Native YAM view (height, width): the clients bound each view to RESTRICT_SIDE before they send it.
NATIVE_VIEW: tuple[int, int] = (1200, 1920)
RESTRICT_SIDE = 416
# Spatial patch of the model (VAE stride 16 x patch 2): the fit rule trims each view to this grid.
SPATIAL_PATCH = 32

Q99_DIMS = np.array([0, 1, 2, 9, 10, 11, 12, 19])
GRIP_DIMS = np.array([9, 19])

VIEW_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "top_left": ("image.exterior", "image.top", "image.top_left"),
    "wrist_left": ("image.wrist_left", "image.left_wrist"),
    "wrist_right": ("image.wrist_right", "image.right_wrist"),
}


def pose_keys(*, side: str) -> tuple[str, ...]:
    return (f"robot_state.{side}.ee_pose", f"ee_pose.{side}")


def grip_keys(*, side: str) -> tuple[str, ...]:
    return (f"grip.{side}", f"robot_state.{side}.grip", f"gripper.{side}")


def canvas_size(config: dict) -> tuple[int, int]:
    """Canvas ``(height, width)`` of a bimanual config: the tile canvas of :data:`VIEWS`."""
    video = config.get("video", {})
    views, canvas = tuple(video.get("views", VIEWS)), video.get("canvas")
    if views != VIEWS or canvas != "tiles":
        raise ValueError(
            f"The bimanual rig supports the tile canvas of {VIEWS}. The config has views={views} canvas={canvas!r}."
        )
    tile_h = int(video.get("tile_height", TILE_SIZE[0]))
    tile_w = int(video.get("tile_width", TILE_SIZE[1]))
    return tile_h, len(VIEWS) * tile_w


def q99_dims(config: dict) -> np.ndarray:
    return np.asarray(
        config.get("normalization", {}).get("q99_normed_dims", Q99_DIMS),
        dtype=np.int64,
    )


def _pick(obs: dict, candidates: tuple[str, ...]) -> np.ndarray:
    for key in candidates:
        if key in obs:
            return np.asarray(obs[key])
    raise KeyError(f"obs missing any of {candidates}; got keys={sorted(obs)}")


class StackMismatch(ValueError):
    """The pose and grip stacks of an arm have different lengths."""


def _pose8_rows(obs: dict, *, side: str, broadcast: bool = True) -> np.ndarray:
    """``(T, 8)`` ``[xyz, quat_wxyz, grip]`` rows of one arm; a single pose or grip row repeats with ``broadcast``."""
    ee = _pick(obs, pose_keys(side=side)).astype(np.float32)
    grip = _pick(obs, grip_keys(side=side)).astype(np.float32)
    ee = ee.reshape(-1, ee.shape[-1]) if ee.ndim >= 2 else ee.reshape(1, -1)
    if ee.shape[1] < 7:
        raise ValueError(f"{side} ee_pose must be (..., >=7), got {ee.shape}")
    grip = grip.reshape(grip.shape[0], -1) if grip.ndim >= 2 else grip.reshape(-1, 1)
    if grip.shape[0] != ee.shape[0]:
        if not broadcast or 1 not in (grip.shape[0], ee.shape[0]):
            raise StackMismatch(
                f"{side}: ee_pose has {ee.shape[0]} rows but grip has {grip.shape[0]}"
            )
        rows = max(grip.shape[0], ee.shape[0])
        ee = np.broadcast_to(ee[-1:], (rows, ee.shape[1]))
        grip = np.broadcast_to(grip[-1:], (rows, grip.shape[1]))
    rows = np.concatenate([ee[:, :7], grip[:, :1]], axis=1).astype(np.float32)
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{side} ee_pose or grip has values that are not finite")
    norms = np.linalg.norm(rows[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-2):
        raise ValueError(
            f"{side} ee_pose quaternion norms {norms.min()}..{norms.max()} != 1"
        )
    return rows


def extract_pose16(obs: dict) -> np.ndarray:
    """Current pose16 ``(16,)``: the last row of each arm's stacks."""
    return np.concatenate([_pose8_rows(obs, side=s)[-1] for s in SIDES]).astype(
        np.float32
    )


def pose16_history(obs: dict, need: int) -> np.ndarray | None:
    """``(need, 16)`` pose16 rows of the executed frames (stack rows ``[-(need + 1):-1]``), or ``None`` when a stack is missing or too short."""
    try:
        rows = [_pose8_rows(obs, side=s, broadcast=False) for s in SIDES]
    except (KeyError, StackMismatch):
        return None
    if any(r.shape[0] < need + 1 for r in rows):
        return None
    return np.concatenate([r[-(need + 1) : -1] for r in rows], axis=1).astype(
        np.float32
    )


def view_stacks(obs: dict) -> dict[str, np.ndarray]:
    """``{view: (T, H, W, 3) uint8}`` in :data:`VIEWS` order; a float frame would turn black, so it raises."""
    out: dict[str, np.ndarray] = {}
    for view in VIEWS:
        img = _pick(obs, VIEW_KEY_CANDIDATES[view])
        if img.ndim == 3:
            img = img[None]
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"{view}: image must be (T,H,W,3), got {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"{view}: image must be uint8, got {img.dtype}")
        out[view] = img
    return out


def fit_resize_size(
    height: int, width: int, max_side: int = RESTRICT_SIDE
) -> tuple[int, int]:
    """The view scaled down to fit ``max_side`` with its aspect kept (integer floor), never scaled up."""
    if height < 1 or width < 1 or max_side < 1:
        raise ValueError(f"bad fit: {height}x{width} into {max_side}")
    return (
        min(height, max_side, (height * max_side) // width),
        min(width, max_side, (width * max_side) // height),
    )


def fit_row_sizes(
    native: list[tuple[int, int]],
    max_side: int = RESTRICT_SIDE,
    patch: int = SPATIAL_PATCH,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Resize sizes and trimmed sizes of the views of one row: a shared height, then the ``patch`` grid."""
    if not native:
        raise ValueError("the fit rule needs at least one view")
    fitted = [fit_resize_size(h, w, max_side) for h, w in native]
    common_h = min(rh for rh, _ in fitted)
    resize = [
        (common_h, (w * common_h) // h) if rh > common_h else (rh, rw)
        for (h, w), (rh, rw) in zip(native, fitted, strict=True)
    ]
    trimmed = []
    for rh, rw in resize:
        grid_h, grid_w = rh // patch * patch, rw // patch * patch
        if grid_h < patch or grid_w < patch:
            raise ValueError(
                f"a view fits to {rh}x{rw}: less than one {patch}-px patch"
            )
        trimmed.append((grid_h, grid_w))
    return resize, trimmed


def _resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Antialiased bilinear float32 resize of uint8 ``(T, 3, H, W)`` frames, rounded back to uint8."""
    if tuple(frames.shape[-2:]) == (height, width):
        return frames
    return torch.cat(
        [
            torch.nn.functional.interpolate(
                chunk.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            for chunk in frames.split(TILE_CHUNK_FRAMES, dim=0)
        ]
    )


def _trim(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    full_h, full_w = frames.shape[-2:]
    if height > full_h or width > full_w:
        raise ValueError(f"cannot trim {full_h}x{full_w} to {height}x{width}")
    top, left = (full_h - height) // 2, (full_w - width) // 2
    return frames[..., top : top + height, left : left + width]


def canvas_clip(
    views: dict[str, np.ndarray],
    tile_h: int = TILE_SIZE[0],
    tile_w: int = TILE_SIZE[1],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``(1, 3, T, tile_h, 3 * tile_w)`` float32 canvas in [0, 255] of view stacks with the same ``T``.

    Each view goes through the fit rule of training (:func:`fit_row_sizes`); a view that does not fit the
    checkpoint tile raises, because only 16:10 views are covered.
    """
    stacks = [views[v] for v in VIEWS]
    counts = {s.shape[0] for s in stacks}
    if len(counts) != 1:
        raise ValueError(f"the views have different frame counts {sorted(counts)}")
    frames = [
        torch.from_numpy(np.ascontiguousarray(s)).permute(0, 3, 1, 2) for s in stacks
    ]
    native = [(int(f.shape[-2]), int(f.shape[-1])) for f in frames]
    resize, trimmed = fit_row_sizes(native)
    if any(t != (tile_h, tile_w) for t in trimmed):
        raise ValueError(
            f"views of {native} fit to {trimmed}, not the {tile_h}x{tile_w} tiles of the checkpoint. "
            "Send 16:10 views (for example 1920x1200 or 960x600) at their native aspect"
        )
    tiles = [
        _trim(_resize(f, *size), tile_h, tile_w)
        for f, size in zip(frames, resize, strict=True)
    ]
    canvas = torch.cat(tiles, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def pose16_to_20d(pose16: np.ndarray, anchor16: np.ndarray | None = None) -> np.ndarray:
    """pose16 rows ``(..., 16)`` to 20-D rows, one arm at a time (relative to its anchor with ``anchor16``)."""
    arms = []
    for lo in (0, 8):
        anchor = None if anchor16 is None else anchor16[..., lo : lo + 8]
        arms.append(pose8_to_10d(pose16[..., lo : lo + 8], anchor))
    return np.concatenate(arms, axis=-1).astype(np.float32)


def proprio_tensors(
    poses: np.ndarray, stats: NormStats, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Proprio of the ``K`` boundary pose16 rows (current last): the current one ``(1, 1, 20)`` and all ``(1, K, 20)``."""
    p = proprio_normalize(pose16_to_20d(poses), stats.proprio_mean, stats.proprio_std)
    p = np.asarray(p).reshape(-1, p.shape[-1])
    every = torch.from_numpy(np.ascontiguousarray(p)).reshape(1, p.shape[0], -1)
    every = every.to(device, torch.float32)
    return every[:, -1:], every


def rel20_to_wire(
    rel: np.ndarray, pose16_now: np.ndarray, dt: float = 1.0 / FPS
) -> list[dict]:
    """Denormalized ``(T, 20)`` relative actions to command dicts.

    Per arm, ``xyz_abs = xyz_rel + anchor_xyz`` in float64 and ``R_abs = R_rel @ R_anchor`` against that arm's
    current pose; the pose is ``[xyz3, R_flat9]`` float32, row-major. The gripper is absolute.
    """
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] < ACTION_DIM:
        raise ValueError(f"rel actions must be (T, >={ACTION_DIM}), got {rel.shape}")
    pose16_now = np.asarray(pose16_now)
    anchors = []
    for i in range(len(SIDES)):
        arm = pose16_now[8 * i : 8 * i + 8]
        anchors.append((arm[:3].astype(np.float64), unit_quat_wxyz_to_matrix(arm[3:7])))
    out: list[dict] = []
    for t, row in enumerate(rel):
        command: dict[str, object] = {}
        for i, side in enumerate(SIDES):
            lo = ARM_DIM * i
            anchor_xyz, r_anchor = anchors[i]
            xyz_abs = row[lo : lo + 3].astype(np.float64) + anchor_xyz
            r_abs = rot6d_to_matrix(row[lo + 3 : lo + 9]) @ r_anchor
            pose12 = np.concatenate([xyz_abs, r_abs.reshape(-1)]).astype(np.float32)
            command[f"robot_command.{side}"] = {
                "type": "cartesian_pos",
                "pose": pose12,
            }
            command[f"target_grip.{side}"] = float(row[lo + 9])
        command["timestamp"] = float(t * dt)
        out.append(command)
    return out


def decode_actions(
    pred_action: torch.Tensor,
    stats: NormStats,
    depth: int,
    dims: np.ndarray = Q99_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized rows (no clip) and relative actions: clip ``dims`` to [-1, 1], invert the q99 map, clip the grips."""
    rows = action_rows(pred_action, depth, FRAMES_PER_TOKEN, ACTION_DIM)
    return rows, clip_grip(denormalize_rows(rows, stats, dims), GRIP_DIMS)


class BimanualChunkInputs(msgspec.Struct, frozen=True, eq=False):
    """Model inputs of one served chunk: as ``ChunkInputs`` of the single-arm rig, with pose16 rows and 20-D proprio."""

    anchor: torch.Tensor
    cleans: list[torch.Tensor]
    boundary_poses: np.ndarray
    proprio: torch.Tensor
    proprio_all: torch.Tensor
    depth: int
    pose_now: np.ndarray
    chunk_index: int
    prompt: str


class BimanualSession:
    """Observed-context state machine of one bimanual episode (the rules of ``PraxisSession``).

    The poses stay in the world frame of the rig and the rig has one control mode (none), so the commands need no
    frame conversion.
    """

    def __init__(
        self,
        *,
        encode: Callable[[torch.Tensor], torch.Tensor],
        stats: NormStats,
        device: torch.device | str,
        prompt: str = "",
        max_context_chunks: int = MAX_CONTEXT_CHUNKS,
        fps: float = FPS,
        dims: np.ndarray = Q99_DIMS,
        exec_horizon: int | None = None,
        n_chunks: int = N_CHUNKS,
        tile_size: tuple[int, int] = TILE_SIZE,
    ):
        self.tile_size = (int(tile_size[0]), int(tile_size[1]))
        self.encode = encode
        self.stats = stats
        self.device = device
        self.prompt = prompt
        self.fps = float(fps)
        self.dims = np.asarray(dims)
        self.exec_horizon = None if exec_horizon is None else int(exec_horizon)
        self.max_context_chunks = max(
            0, min(int(max_context_chunks), int(n_chunks) - 1)
        )
        self.stateless = self.max_context_chunks == 0 or (
            self.exec_horizon is not None and self.exec_horizon < ACTION_CHUNK
        )
        self.stream = ContiguousStreamContext(
            action_chunk=ACTION_CHUNK,
            max_context_chunks=max(self.max_context_chunks, 1),
        )
        self.chunk_index = 0
        self.restart_count = 0
        self._boundary_latents: list[torch.Tensor] = []
        self._boundary_poses: list[np.ndarray] = []
        self._cap = self.max_context_chunks + 1

    def reset(self) -> None:
        self.chunk_index = 0
        self.restart()

    def restart(self) -> None:
        self.restart_count += 1
        self._boundary_latents = []
        self._boundary_poses = []
        self.stream.reset()

    def _clip(self, views: dict[str, np.ndarray]) -> torch.Tensor:
        return canvas_clip(views, *self.tile_size, self.device)

    def prepare(self, obs: dict) -> BimanualChunkInputs:
        if bool(obs.get("reset_context") or obs.get("__reset_context__")):
            self.restart()
        override = obs.get("task") or obs.get("prompt")
        if override:
            if isinstance(override, bytes):
                override = override.decode("utf-8", "replace")
            new = str(override).strip()
            if new and new != self.prompt:
                self.prompt = new
        pose16 = extract_pose16(obs)
        stacks = view_stacks(obs)
        current = self._clip({v: s[-1:] for v, s in stacks.items()})
        if self.stateless:
            self.restart()
        elif self._boundary_latents:
            self._append_executed(obs, stacks=stacks)
        latent = self.encode(current)
        self._boundary_latents = [*self._boundary_latents, latent.detach()][
            -self._cap :
        ]
        self._boundary_poses = [*self._boundary_poses, pose16][-self._cap :]
        cleans = self.stream.cleans(self.encode)
        depth = len(cleans)
        anchor_frame = self.stream.anchor_frame()
        anchor = (
            self._boundary_latents[-1]
            if anchor_frame is None
            else self.encode(anchor_frame)
        )
        poses = self._window_poses(depth=depth, pose16=pose16)
        proprio, proprio_all = proprio_tensors(poses, self.stats, self.device)
        return BimanualChunkInputs(
            anchor=anchor,
            cleans=cleans,
            boundary_poses=poses,
            proprio=proprio,
            proprio_all=proprio_all,
            depth=depth,
            pose_now=pose16,
            chunk_index=self.chunk_index,
            prompt=text_encoder_prompt(self.prompt),
        )

    def _append_executed(self, obs: dict, *, stacks: dict[str, np.ndarray]) -> None:
        need = self.stream.expected_next_rows
        if any(s.shape[0] < need + 1 for s in stacks.values()):
            self.restart()
            return
        clip = self._clip({v: s[-(need + 1) : -1] for v, s in stacks.items()})
        self.stream.append_executed(clip, pose16_history(obs, need))

    def _window_poses(self, *, depth: int, pose16: np.ndarray) -> np.ndarray:
        if depth == 0:
            return pose16[None].astype(np.float32)
        boundary = self.stream.boundary_poses()
        if boundary is not None:
            return np.concatenate([boundary, pose16[None]], axis=0).astype(np.float32)
        n = min(depth + 1, len(self._boundary_poses))
        return np.stack(self._boundary_poses[-n:], axis=0).astype(np.float32)

    def decode_actions(
        self, inputs: BimanualChunkInputs, pred_action: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        return decode_actions(pred_action, self.stats, inputs.depth, self.dims)

    def finish(
        self, inputs: BimanualChunkInputs, pred_action: torch.Tensor
    ) -> list[dict]:
        """Commands of this chunk (cut to ``exec_horizon``); advances the chunk counter."""
        _rows, rel = self.decode_actions(inputs, pred_action)
        wire = rel20_to_wire(rel, inputs.pose_now, 1.0 / max(self.fps, 1e-6))
        self.chunk_index += 1
        return wire if self.exec_horizon is None else wire[: self.exec_horizon]


def latent_video_shape(anchor: torch.Tensor) -> tuple[int, ...]:
    """Shape of the noised video chunk: the anchor shape with the chunk's latent frames."""
    shape = list(anchor.shape)
    shape[-3] = latent_frames(ACTION_CHUNK)
    return tuple(shape)


def warmup_obs(*, step: int, stack: int) -> dict:
    """A synthetic observation with ``stack`` frames per view, shaped like the client sends 1920x1200 cameras."""
    height, width = fit_resize_size(*NATIVE_VIEW)
    t = step * ACTION_CHUNK + np.arange(stack)

    def frames(phase: int) -> np.ndarray:
        x = np.zeros((stack, height, width, 3), np.uint8)
        x[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, None, :]
        x[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[None, :, None]
        x[..., 2] = ((t * 37 + phase * 61) % 255)[:, None, None]
        return x

    obs: dict = {VIEW_KEY_CANDIDATES[v][0]: frames(i) for i, v in enumerate(VIEWS)}
    for j, side in enumerate(SIDES):
        pose = np.zeros((stack, 7), np.float32)
        pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3] = (
            0.4 + 0.0005 * t,
            0.3 - 0.6 * j,
            0.4,
            1.0,
        )
        obs[pose_keys(side=side)[0]] = pose
        obs[grip_keys(side=side)[0]] = np.full(stack, 0.1 * j, np.float32)
    obs["task"] = "warmup"
    return obs
