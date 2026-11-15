# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 single-arm rig: camera canvas, pose frames, proprio, command decoding and the session state.

Observation (rig side): ``robot_state.ee_pose`` ``[x, y, z, qw, qx, qy, qz]``, ``grip`` (0 = open), and
RGB uint8 ``image.exterior`` / ``image.wrist``; each may carry a leading time axis whose last row is now.
Model space: pose8 ``[xyz3, quat_wxyz4, grip1]`` and 10-D ``[xyz3, rot6d6, grip1]``. Actions are relative
to the pose at the start of the chunk (the gripper is absolute) and q99-normalized on xyz and grip.
Commands: one dict per row at 15 Hz with ``robot_command`` (``pose`` is ``[xyz3, R_flat9]``), ``target_grip``
and ``timestamp``. The first chunk of a context window gives 21 rows, every later chunk 24.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import msgspec
import numpy as np
import torch

from sglang.multimodal_gen.runtime.vla.praxis_geometry import (
    ACTION_CHUNK,
    ASPECT_TOL,
    FPS,
    FRAMES_PER_TOKEN,
    ContiguousStreamContext,
    NormStats,
    action_rows,
    aspect_crop,
    clip_grip,
    denormalize_rows,
    latent_frames,
    pose8_to_10d,
    proprio_normalize,
    rot6d_to_matrix,
    text_encoder_prompt,
    unit_quat_wxyz_to_matrix,
)

VIEWS: tuple[str, ...] = ("exterior", "wrist")
ACTION_DIM = 10
N_CHUNKS = 4
MAX_CONTEXT_CHUNKS = 3
TILE_SIZE: tuple[int, int] = (256, 416)
# Native DROID view (height, width): the antialiased canvas crops to its aspect before resizing.
NATIVE_VIEW: tuple[int, int] = (720, 1280)
# Frames per resize call of the tile canvas; bounds the float32 memory.
TILE_CHUNK_FRAMES = 32

_C, _S = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
# Rz(-45 deg): the turn of the rig frame about the flange z axis.
FLANGE_TURN = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])
NO_TURN = np.eye(3)


class RigFrame(msgspec.Struct, frozen=True):
    """Pose frame of one client relative to the Franka flange.

    ``tool_offset`` is the distance of the pose origin from the flange along the flange z axis (m),
    ``turn`` satisfies ``R_client = R_flange @ turn``, ``base_offset`` is the robot base in client coordinates (m).
    """

    tool_offset: float
    turn: tuple[tuple[float, ...], ...]
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _rows(m: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in m)


TURNED_FLANGE = RigFrame(tool_offset=0.0, turn=_rows(FLANGE_TURN))
FR3_HAND = RigFrame(tool_offset=0.1034, turn=_rows(FLANGE_TURN))
# Keyed by the ``descriptor`` of the observation; the FR3 rig sends none.
RIG_FRAMES: dict[str, RigFrame] = {
    "": FR3_HAND,
    "remote.robolab.droid": FR3_HAND,
    "remote.molmo_spaces.droid": RigFrame(
        tool_offset=0.155, turn=_rows(NO_TURN), base_offset=(0.0, 0.0, 0.58)
    ),
    "praxis.sim": TURNED_FLANGE,
}
# A converted pose must be this close to the flange forward kinematics (m).
FK_TOLERANCE = 0.005
# Franka modified DH parameters (a, d, alpha) of joints 1 to 7; the flange is 0.107 m past joint 7.
FRANKA_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
FRANKA_FLANGE_D = 0.107

# Dims of a 10-D row with the q99 normalization (xyz and grip), and the dim that holds the gripper.
Q99_DIMS = np.array([0, 1, 2, 9])
GRIP_DIMS = np.array([9])

POSE_KEYS: tuple[str, ...] = ("ee_pose", "robot_state.ee_pose")
GRIP_KEYS: tuple[str, ...] = ("grip", "robot_state.grip", "gripper")
VIEW_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "exterior": ("exterior_image", "image.exterior", "observation.images.side"),
    "wrist": ("wrist_image", "image.wrist", "observation.images.left"),
}

# The software impedance law of the DROID demonstrations: joint gains (7), then Cartesian gains (3 + 3).
DROID_IMPEDANCE: dict[str, object] = {
    "type": "impedance",
    "kq": [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
    "kqd": [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
    "kx": [750.0, 750.0, 750.0, 15.0, 15.0, 15.0],
    "kxd": [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
}
# "none" adds no ``mode`` to the commands; the arm keeps its own law.
CONTROL_MODES: dict[str, dict[str, object] | None] = {
    "none": None,
    "droid_impedance": DROID_IMPEDANCE,
}


def control_mode_payload(control_mode: str) -> dict[str, object] | None:
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"Unknown control_mode {control_mode!r}. The known modes are "
            f"{sorted(CONTROL_MODES)}."
        )
    payload = CONTROL_MODES[control_mode]
    return None if payload is None else copy.deepcopy(payload)


def canvas_size(config: dict) -> tuple[int, int]:
    """``(height, 2 * tile_width)`` of the tile canvas declared by a checkpoint config."""
    video = config.get("video", {})
    views = tuple(video.get("views", VIEWS))
    canvas = video.get("canvas")
    if views != VIEWS or canvas != "tiles":
        raise ValueError(
            f"The single-arm rig supports the tile canvas of {VIEWS}. The config "
            f"has views={views} canvas={canvas!r}. This canvas is not "
            "supported; use the single-arm checkpoint."
        )
    tile_h = int(video.get("tile_height", TILE_SIZE[0]))
    tile_w = int(video.get("tile_width", TILE_SIZE[1]))
    return tile_h, len(VIEWS) * tile_w


def q99_dims(config: dict) -> np.ndarray:
    norm = config.get("normalization", {})
    return np.asarray(norm.get("q99_normed_dims", Q99_DIMS), dtype=np.int64)


def _pick(obs: dict, candidates: tuple[str, ...]) -> object:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"obs missing any of {candidates}; got keys={list(obs)}")


def extract_pose8(obs: dict) -> np.ndarray:
    """Current pose8 ``(8,)``: the last row of the pose and grip stacks."""
    ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
    if ee.ndim >= 2:
        ee = ee[-1]
    ee = ee.ravel()
    if ee.shape[0] < 7:
        raise ValueError(f"ee_pose must be (>=7,), got {ee.shape}")
    grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    if grip.ndim >= 2:
        grip = grip[-1]
    elif grip.ndim == 1 and grip.shape[0] > 1:
        grip = grip[-1:]
    grip = grip.ravel()
    return np.concatenate([ee[:7], grip[:1]]).astype(np.float32)


def pose8_history(obs: dict, need: int) -> np.ndarray | None:
    """``(need, 8)`` pose8 rows of the executed frames (stack rows ``[-(need + 1):-1]``), or None."""
    try:
        ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
        grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    except KeyError:
        return None
    if ee.ndim != 2 or ee.shape[0] < need + 1 or ee.shape[1] < 7:
        return None
    g = grip.reshape(grip.shape[0], -1) if grip.ndim >= 1 else None
    if g is None or g.shape[0] < need + 1:
        return None
    rows = np.concatenate([ee[-(need + 1) : -1, :7], g[-(need + 1) : -1, :1]], axis=1)
    return rows.astype(np.float32)


def latest_frame(obs: dict, view: str) -> np.ndarray:
    img = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES[view]))
    if img.ndim == 4:
        img = img[-1]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be (H,W,3), got {img.shape}")
    return img.astype(np.uint8)


def fit_resize_size(tile_w: int) -> tuple[int, int]:
    """Native view scaled down to ``tile_w`` wide, height rounded down: ``(234, 416)`` for 416-wide tiles."""
    native_h, native_w = NATIVE_VIEW
    return (
        min(native_h, tile_w, native_h * tile_w // native_w),
        min(native_w, tile_w),
    )


def _resize_uint8(frames: torch.Tensor, size: tuple[int, int], antialias: bool):
    resized = [
        torch.nn.functional.interpolate(
            chunk.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=antialias,
        )
        for chunk in frames.split(TILE_CHUNK_FRAMES, dim=0)
    ]
    return torch.cat(resized).round().clamp(0, 255).to(torch.uint8)


def _fit_view(fr: torch.Tensor, tile_h: int, tile_w: int, tol: float) -> torch.Tensor:
    """Fit rule of an antialiased checkpoint: crop to the native aspect, resize once, trim to the tile."""
    fit_h, fit_w = fit_resize_size(tile_w)
    if fit_h < tile_h or fit_w < tile_w:
        raise ValueError(
            f"the fit size {fit_h}x{fit_w} is below the tile {tile_h}x{tile_w}"
        )
    cropped = aspect_crop(fr, *NATIVE_VIEW, tol)
    if tuple(cropped.shape[-2:]) != (fit_h, fit_w):
        cropped = _resize_uint8(cropped, (fit_h, fit_w), antialias=True)
    y0, x0 = (fit_h - tile_h) // 2, (fit_w - tile_w) // 2
    return cropped[..., y0 : y0 + tile_h, x0 : x0 + tile_w].contiguous()


def _tile_view(
    frames: np.ndarray, tile_h: int, tile_w: int, tol: float, antialias: bool = False
) -> torch.Tensor:
    """``(T, 3, tile_h, tile_w)`` uint8 tile of one view; all resizes are bilinear float32, rounded to uint8."""
    a = np.asarray(frames, np.uint8)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3), got {a.shape}")
    fr = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
    if antialias:
        return _fit_view(fr, tile_h, tile_w, tol)
    return _resize_uint8(aspect_crop(fr, tile_h, tile_w, tol), (tile_h, tile_w), False)


def tile_clip(
    exterior: np.ndarray,
    wrist: np.ndarray | None,
    tile_h: int = TILE_SIZE[0],
    tile_w: int = TILE_SIZE[1],
    tol: float = ASPECT_TOL,
    device: torch.device | str = "cpu",
    antialias: bool = False,
) -> torch.Tensor:
    """``(1, 3, T, tile_h, 2 * tile_w)`` float32 clip with integer values in [0, 255]; exterior left, wrist right."""
    if wrist is None:
        raise ValueError("The tile canvas needs the exterior and the wrist view.")
    tiles = [_tile_view(v, tile_h, tile_w, tol, antialias) for v in (exterior, wrist)]
    if tiles[0].shape[0] != tiles[1].shape[0]:
        raise ValueError(
            f"exterior has {tiles[0].shape[0]} frames, wrist has {tiles[1].shape[0]}"
        )
    canvas = torch.cat(tiles, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def _tile_args(
    tile_size: tuple[int, int] | None, aspect_tol: float | None
) -> tuple[int, int, float]:
    tile_h, tile_w = TILE_SIZE if tile_size is None else tile_size
    tol = ASPECT_TOL if aspect_tol is None else float(aspect_tol)
    return int(tile_h), int(tile_w), tol


def current_clip(
    obs: dict,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
    antialias: bool = False,
) -> torch.Tensor:
    """Tile clip ``(1, 3, 1, H, W)`` of the current frames."""
    exterior = latest_frame(obs, "exterior")
    try:
        wrist = latest_frame(obs, "wrist")
    except KeyError:
        wrist = None
    tile_h, tile_w, tol = _tile_args(tile_size, aspect_tol)
    return tile_clip(
        exterior[None],
        None if wrist is None else wrist[None],
        tile_h,
        tile_w,
        tol,
        device,
        antialias,
    )


def executed_clip(
    obs: dict,
    need: int,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
    antialias: bool = False,
) -> tuple[torch.Tensor, np.ndarray | None] | None:
    """Clip and pose8 rows of the ``need`` executed raws (stack rows ``[-(need + 1):-1]``), or None.

    None means an image stack is missing or too short. The pose rows are None without a pose stack.
    """
    ext = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES["exterior"]))
    if ext.ndim != 4 or ext.shape[0] < need + 1:
        return None
    ext_chunk = ext[-(need + 1) : -1].astype(np.uint8)
    try:
        wrist = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES["wrist"]))
    except KeyError:
        return None
    if wrist.ndim != 4 or wrist.shape[0] < need + 1:
        return None
    wrist_chunk = wrist[-(need + 1) : -1].astype(np.uint8)
    tile_h, tile_w, tol = _tile_args(tile_size, aspect_tol)
    clip = tile_clip(ext_chunk, wrist_chunk, tile_h, tile_w, tol, device, antialias)
    return clip, pose8_history(obs, need)


def _quat_xyzw_unit(q: np.ndarray) -> np.ndarray:
    """Normalize xyzw quaternions in float64, in the operation order of scipy."""
    q = np.array(q, dtype=np.float64, copy=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return q / np.sqrt(x * x + y * y + z * z + w * w)[..., None]


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Same bits as ``scipy.spatial.transform.Rotation.from_quat(q).as_matrix()``."""
    q = _quat_xyzw_unit(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    m = np.empty((*q.shape[:-1], 3, 3))
    m[..., 0, 0] = x2 - y2 - z2 + w2
    m[..., 1, 0] = 2 * (xy + zw)
    m[..., 2, 0] = 2 * (xz - yw)
    m[..., 0, 1] = 2 * (xy - zw)
    m[..., 1, 1] = -x2 + y2 - z2 + w2
    m[..., 2, 1] = 2 * (yz + xw)
    m[..., 0, 2] = 2 * (xz + yw)
    m[..., 1, 2] = 2 * (yz - xw)
    m[..., 2, 2] = -x2 - y2 + z2 + w2
    return m


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Same bits as ``Rotation.from_matrix(m).as_quat()`` (largest-component branch); the sign is not canonical."""
    m = np.asarray(m, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for n, c in enumerate(flat):
        dec = [c[0, 0], c[1, 1], c[2, 2], c[0, 0] + c[1, 1] + c[2, 2]]
        choice = int(np.argmax(dec))
        q = np.empty(4)
        if choice != 3:
            i = choice
            j = (i + 1) % 3
            k = (j + 1) % 3
            q[i] = 1 - dec[3] + 2 * c[i, i]
            q[j] = c[j, i] + c[i, j]
            q[k] = c[k, i] + c[i, k]
            q[3] = c[k, j] - c[j, k]
        else:
            q[0] = c[2, 1] - c[1, 2]
            q[1] = c[0, 2] - c[2, 0]
            q[2] = c[1, 0] - c[0, 1]
            q[3] = 1 + dec[3]
        out[n] = q
    return _quat_xyzw_unit(out).reshape(*m.shape[:-2], 4)


def pose8_to_model(pose8: np.ndarray, frame: RigFrame = TURNED_FLANGE) -> np.ndarray:
    """Client pose8 rows in ``frame`` to the flange frame of the model (float32, grip clipped to [0, 1]).

    The position becomes ``xyz - base_offset - tool_offset * z_client`` and the rotation
    ``R_client @ turn^T``; the quaternion comes back wxyz with the sign of ``matrix_to_quat_xyzw``.
    """
    out = np.array(pose8, dtype=np.float64, copy=True)
    rot = quat_xyzw_to_matrix(out[..., [4, 5, 6, 3]])
    base = np.asarray(frame.base_offset, dtype=np.float64)
    out[..., :3] = out[..., :3] - base - frame.tool_offset * rot[..., :, 2]
    turn = np.asarray(frame.turn).T @ NO_TURN
    out[..., 3:7] = matrix_to_quat_xyzw(rot @ turn)[..., [3, 0, 1, 2]]
    out[..., 7] = np.clip(out[..., 7], 0.0, 1.0)
    return out.astype(np.float32)


def pose_to_rig(
    xyz: np.ndarray, rot: np.ndarray, frame: RigFrame = TURNED_FLANGE
) -> tuple[np.ndarray, np.ndarray]:
    """Flange-frame target (float64 ``xyz (3,)``, ``rot (3, 3)``) to the client ``frame``."""
    xyz = xyz + frame.tool_offset * rot[:, 2] + np.asarray(frame.base_offset)
    return xyz, rot @ NO_TURN.T @ np.asarray(frame.turn)


def franka_flange_pose(q: np.ndarray) -> np.ndarray:
    """Flange pose ``(4, 4)`` in the robot base frame for the joints ``q (7,)``."""
    m = np.eye(4)
    for (a, d, alpha), theta in zip(
        FRANKA_DH, np.asarray(q, dtype=np.float64), strict=True
    ):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        m = m @ np.array(
            [
                [ct, -st, 0.0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    m[:3, 3] += FRANKA_FLANGE_D * m[:3, 2]
    return m


def rig_frame(obs: dict) -> RigFrame:
    """Pose frame of the client named by the ``descriptor`` of ``obs``."""
    descriptor = obs.get("descriptor") or ""
    if isinstance(descriptor, bytes):
        descriptor = descriptor.decode("utf-8", "replace")
    frame = RIG_FRAMES.get(str(descriptor))
    if frame is None:
        raise ValueError(
            f"The client {descriptor!r} reports poses in an unknown frame. The "
            f"known clients are {sorted(RIG_FRAMES)}."
        )
    return frame


def check_flange(obs: dict, pose8_model: np.ndarray) -> None:
    """With ``robot_state.q`` in ``obs``, require the converted pose to match the flange forward kinematics."""
    if "robot_state.q" not in obs:
        return
    q = np.asarray(obs["robot_state.q"])
    if q.ndim == 2:
        q = q[-1]
    if q.shape != (7,) or q.dtype.kind not in "iuf":
        raise ValueError(f"robot_state.q must be 7 joint angles, got {q.shape}")
    xyz = np.asarray(pose8_model[:3], dtype=np.float64)
    err = float(np.linalg.norm(xyz - franka_flange_pose(q)[:3, 3]))
    if not np.isfinite(err) or err > FK_TOLERANCE:
        raise ValueError(
            f"The converted ee_pose is {err * 1e3:.1f} mm from the flange forward "
            "kinematics of robot_state.q. The client frame is not the frame of "
            "its descriptor in RIG_FRAMES."
        )


def proprio_rows(pose8: np.ndarray, stats: NormStats) -> np.ndarray:
    return proprio_normalize(pose8_to_10d(pose8), stats.proprio_mean, stats.proprio_std)


def proprio_tensors(
    poses: np.ndarray, stats: NormStats, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Proprio of the ``K`` boundary poses (current last): the current one ``(1, 1, 10)`` and all ``(1, K, 10)``."""
    p = proprio_rows(poses, stats)
    p = np.asarray(p).reshape(-1, p.shape[-1])
    last = (
        torch.from_numpy(np.ascontiguousarray(p[-1:]))
        .reshape(1, 1, -1)
        .to(device, torch.float32)
    )
    every = (
        torch.from_numpy(np.ascontiguousarray(p))
        .reshape(1, p.shape[0], -1)
        .to(device, torch.float32)
    )
    return last, every


def rel10_to_wire(
    rel: np.ndarray,
    pose8_now: np.ndarray,
    dt: float = 1.0 / FPS,
    mode: dict[str, object] | None = None,
    frame: RigFrame = TURNED_FLANGE,
) -> list[dict]:
    """Denormalized ``(T, 10)`` relative actions to command dicts in the client ``frame``.

    ``xyz_abs = xyz_rel + anchor_xyz`` in float64 and ``R_abs = R_rel @ R_anchor``, with the anchor the
    current pose in the flange frame; the pose is ``[xyz3, R_flat9]`` float32, row-major.
    """
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] < ACTION_DIM:
        raise ValueError(f"rel actions must be (T, >={ACTION_DIM}), got {rel.shape}")
    anchor = np.asarray(pose8_now)
    anchor_xyz = anchor[:3].astype(np.float64)
    r_anchor = unit_quat_wxyz_to_matrix(anchor[3:7])
    out: list[dict] = []
    for t, r in enumerate(rel):
        xyz_abs = r[:3].astype(np.float64) + anchor_xyz
        r_abs = rot6d_to_matrix(r[3:9]) @ r_anchor
        xyz_abs, r_abs = pose_to_rig(xyz_abs, r_abs, frame)
        pose12 = np.concatenate([xyz_abs, r_abs.reshape(-1)]).astype(np.float32)
        command: dict[str, object] = {"type": "cartesian_pos", "pose": pose12}
        if mode is not None:
            command["mode"] = mode
        out.append(
            {
                "robot_command": command,
                "target_grip": float(r[9]),
                "timestamp": float(t * dt),
            }
        )
    return out


def decode_actions(
    pred_action: torch.Tensor,
    stats: NormStats,
    depth: int,
    dims: np.ndarray = Q99_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized rows (no clip) and relative actions: clip ``dims`` to [-1, 1], invert the q99 map, clip the grip."""
    rows = action_rows(pred_action, depth, FRAMES_PER_TOKEN, ACTION_DIM)
    return rows, clip_grip(denormalize_rows(rows, stats, dims), GRIP_DIMS)


class ChunkInputs(msgspec.Struct, frozen=True, eq=False):
    """Model inputs of one served chunk.

    ``anchor`` is the 1-latent window anchor, ``cleans`` the ``depth`` clean chunk latents (oldest first),
    ``boundary_poses`` the ``depth + 1`` pose8 rows at the chunk starts (current last, flange frame),
    ``proprio`` the normalized current pose ``(1, 1, 10)`` and ``proprio_all`` that of all boundary poses.
    """

    anchor: torch.Tensor
    cleans: list[torch.Tensor]
    boundary_poses: np.ndarray
    proprio: torch.Tensor
    proprio_all: torch.Tensor
    depth: int
    pose8_now: np.ndarray
    chunk_index: int
    prompt: str


class PraxisSession:
    """Observed-context state machine of one single-arm episode.

    ``encode`` maps a tile clip ``(1, 3, T, H, W)`` in [0, 255] to video latents. :meth:`prepare` applies a
    prompt override, appends the executed frames of the previous chunk, encodes the current frame and builds
    the window ``[anchor | d cleans]`` with its proprio; :meth:`finish` turns the sampled actions into
    commands and advances the chunk counter. An image stack that is missing or too short restarts the
    context (a cold start at depth 0); so does ``exec_horizon`` below the chunk length or
    ``max_context_chunks=0``.
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
        control_mode: str = "none",
        exec_horizon: int | None = None,
        n_chunks: int = N_CHUNKS,
        tile_size: tuple[int, int] | None = None,
        aspect_tol: float | None = None,
        antialias: bool = False,
    ):
        self.antialias = bool(antialias)
        self.tile_size = (
            None if tile_size is None else (int(tile_size[0]), int(tile_size[1]))
        )
        self.aspect_tol = None if aspect_tol is None else float(aspect_tol)
        self.encode = encode
        self.stats = stats
        self.device = device
        self.prompt = prompt
        self.fps = float(fps)
        self.dims = np.asarray(dims)
        self.control_mode = str(control_mode)
        self.mode = control_mode_payload(self.control_mode)
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
        self.frame = TURNED_FLANGE

    def reset(self) -> None:
        self.chunk_index = 0
        self.restart()

    def restart(self) -> None:
        self.restart_count += 1
        self._boundary_latents = []
        self._boundary_poses = []
        self.stream.reset()

    def prepare(self, obs: dict) -> ChunkInputs:
        if bool(obs.get("reset_context") or obs.get("__reset_context__")):
            self.restart()
        override = obs.get("task") or obs.get("prompt")
        if override:
            if isinstance(override, bytes):
                override = override.decode("utf-8", "replace")
            new = str(override).strip()
            if new and new != self.prompt:
                self.prompt = new
        frame = rig_frame(obs)
        pose8 = pose8_to_model(extract_pose8(obs), frame)
        check_flange(obs, pose8)
        current = current_clip(
            obs, self.device, self.tile_size, self.aspect_tol, self.antialias
        )
        if self.stateless:
            self.restart()
        elif self._boundary_latents:
            self._append_executed(obs, frame=frame)
        latent = self.encode(current)
        self._boundary_latents = [*self._boundary_latents, latent.detach()][
            -self._cap :
        ]
        pose = np.asarray(pose8, dtype=np.float32)
        self._boundary_poses = [*self._boundary_poses, pose][-self._cap :]
        cleans = self.stream.cleans(self.encode)
        depth = len(cleans)
        anchor_frame = self.stream.anchor_frame()
        anchor = (
            self._boundary_latents[-1]
            if anchor_frame is None
            else self.encode(anchor_frame)
        )
        poses = self._window_poses(depth=depth, pose8=pose8, pose=pose)
        proprio, proprio_all = proprio_tensors(poses, self.stats, self.device)
        # The commands go to the frame of this observation once it is accepted.
        self.frame = frame
        return ChunkInputs(
            anchor=anchor,
            cleans=cleans,
            boundary_poses=poses,
            proprio=proprio,
            proprio_all=proprio_all,
            depth=depth,
            pose8_now=pose8,
            chunk_index=self.chunk_index,
            prompt=text_encoder_prompt(self.prompt),
        )

    def _append_executed(self, obs: dict, *, frame: RigFrame) -> None:
        need = self.stream.expected_next_rows
        executed = executed_clip(
            obs, need, self.device, self.tile_size, self.aspect_tol, self.antialias
        )
        if executed is None:
            self.restart()
            return
        clip, rows = executed
        if rows is not None:
            rows = pose8_to_model(rows, frame)
        self.stream.append_executed(clip, rows)

    def _window_poses(
        self, *, depth: int, pose8: np.ndarray, pose: np.ndarray
    ) -> np.ndarray:
        if depth == 0:
            return pose8[None].astype(np.float32)
        boundary = self.stream.boundary_poses()
        if boundary is not None:
            return np.concatenate([boundary, pose8[None]], axis=0).astype(np.float32)
        n = min(depth + 1, len(self._boundary_poses))
        return np.stack(self._boundary_poses[-n:], axis=0).astype(np.float32)

    def decode_actions(
        self, inputs: ChunkInputs, pred_action: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        return decode_actions(pred_action, self.stats, inputs.depth, self.dims)

    def rel_to_wire(self, rel: np.ndarray, pose8_now: np.ndarray) -> list[dict]:
        dt = 1.0 / max(self.fps, 1e-6)
        return rel10_to_wire(rel, pose8_now, dt, self.mode, self.frame)

    def finish(self, inputs: ChunkInputs, pred_action: torch.Tensor) -> list[dict]:
        """Commands of this chunk (cut to ``exec_horizon``); advances the chunk counter."""
        _rows, rel = self.decode_actions(inputs, pred_action)
        wire = self.rel_to_wire(rel, inputs.pose8_now)
        self.chunk_index += 1
        return wire if self.exec_horizon is None else wire[: self.exec_horizon]


def latent_video_shape(anchor: torch.Tensor) -> tuple[int, ...]:
    """Shape of the noised video chunk: the anchor shape with the chunk's latent frames."""
    shape = list(anchor.shape)
    shape[-3] = latent_frames(ACTION_CHUNK)
    return tuple(shape)
