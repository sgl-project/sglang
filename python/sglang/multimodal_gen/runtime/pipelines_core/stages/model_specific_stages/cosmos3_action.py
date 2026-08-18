# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 action modality helpers: domain mapping, mode constants, the
structured JSON caption, and dataset-derived action (de)normalization.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

ACTION_MODE_POLICY = "policy"
ACTION_MODE_FORWARD_DYNAMICS = "forward_dynamics"
ACTION_MODE_INVERSE_DYNAMICS = "inverse_dynamics"
ACTION_MODES = {
    ACTION_MODE_POLICY,
    ACTION_MODE_FORWARD_DYNAMICS,
    ACTION_MODE_INVERSE_DYNAMICS,
}

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}

# Embodiment -> real (unpadded) action channel count. Channels beyond this are
# zero-padding up to the model's action_dim.
EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "agibotworld": 29,
    "fractal": 10,
}

# Canonical (width, height) targets per resolution tier and aspect ratio, used
# to render the aspect_ratio field of the action caption.
VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "704": {
        "1,1": (960, 960),
        "4,3": (1088, 832),
        "3,4": (832, 1088),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}

VIEWPOINT_TEMPLATES: dict[str, str] = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}

_STAT_KEYS = {"mean", "std", "min", "max", "q01", "q99"}


def get_raw_action_dim(embodiment: str) -> int:
    key = embodiment.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise ValueError(
            f"No raw action dim for Cosmos3 embodiment {embodiment!r}. Expected one "
            f"of {sorted(EMBODIMENT_TO_RAW_ACTION_DIM)}."
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def canonical_aspect_ratio(width: int, height: int) -> str:
    """Canonical ``"W,H"`` aspect string for the action caption."""
    for sizes in VIDEO_RES_SIZE_INFO.values():
        for aspect, (cand_w, cand_h) in sizes.items():
            if width == cand_w and height == cand_h:
                return aspect
    divisor = math.gcd(width, height)
    if divisor == 0:
        raise ValueError(
            f"width and height must be non-zero, got width={width}, height={height}."
        )
    return f"{width // divisor},{height // divisor}"


def build_action_prompt(
    description: str,
    view_point: str,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> str:
    """Render the structured JSON action caption the action checkpoints expect."""
    duration_seconds = num_frames / fps
    minutes, secs = divmod(round(duration_seconds), 60)
    if description and description[-1] not in ".!?":
        description = description + "."
    prompt = {
        "cinematography": {
            "framing": VIEWPOINT_TEMPLATES.get(
                view_point, VIEWPOINT_TEMPLATES["ego_view"]
            )
        },
        "actions": [{"time": f"0:00-{minutes}:{secs:02d}", "description": description}],
        "duration": f"{int(duration_seconds)}s",
        "fps": float(fps),
        "resolution": {"H": int(height), "W": int(width)},
        "aspect_ratio": canonical_aspect_ratio(int(width), int(height)),
    }
    return json.dumps(prompt)


def load_action_stats(
    stats_path: str, stats_key: str = "global"
) -> dict[str, torch.Tensor]:
    """Load per-channel action normalization stats from a JSON file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Action normalization stats not found at {stats_path}."
        )
    raw = json.loads(path.read_text())
    if stats_key in raw:
        raw = raw[stats_key]
    return {
        k: torch.as_tensor(np.array(v, dtype=np.float32))
        for k, v in raw.items()
        if k in _STAT_KEYS
    }


def normalize_action(
    action: torch.Tensor, method: str, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    if method == "quantile":
        q01, q99 = stats["q01"].to(action), stats["q99"].to(action)
        return (2.0 * (action - q01) / (q99 - q01).clamp(min=1e-8) - 1.0).clamp(
            -1.0, 1.0
        )
    if method == "meanstd":
        return (action - stats["mean"].to(action)) / stats["std"].to(action).clamp(
            min=1e-8
        )
    if method == "minmax":
        lo, hi = stats["min"].to(action), stats["max"].to(action)
        return (2.0 * (action - lo) / (hi - lo).clamp(min=1e-8) - 1.0).clamp(-1.0, 1.0)
    raise ValueError(f"Unknown action normalization method {method!r}.")


def denormalize_action(
    action: torch.Tensor, method: str, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    if method == "quantile":
        q01, q99 = stats["q01"].to(action), stats["q99"].to(action)
        return (action + 1.0) / 2.0 * (q99 - q01) + q01
    if method == "meanstd":
        return action * stats["std"].to(action) + stats["mean"].to(action)
    if method == "minmax":
        lo, hi = stats["min"].to(action), stats["max"].to(action)
        return (action + 1.0) / 2.0 * (hi - lo) + lo
    raise ValueError(f"Unknown action normalization method {method!r}.")
