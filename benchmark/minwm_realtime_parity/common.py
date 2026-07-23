#!/usr/bin/env python3
"""Shared, dependency-light helpers for the MinWM parity harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

KEY_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")
TRANS_BITS_TO_LABEL = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 0, 1, 0): 2,
    (0, 1, 0, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 1, 0, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 0, 1, 1): 8,
}
ROT_BITS_TO_LABEL = {
    (0, 0, 0, 0): 0,
    (0, 0, 0, 1): 1,
    (0, 1, 0, 0): 2,
    (1, 0, 0, 0): 3,
    (0, 0, 1, 0): 4,
    (1, 0, 0, 1): 5,
    (0, 0, 1, 1): 6,
    (1, 1, 0, 0): 7,
    (0, 1, 1, 0): 8,
}


def load_cases(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        manifest = json.load(source)
    contract = manifest["contract"]
    cases = manifest["cases"]
    expected_case_count = int(contract.get("case_count", 10))
    if len(cases) != expected_case_count:
        raise ValueError(
            f"parity manifest must contain exactly {expected_case_count} cases, "
            f"got {len(cases)}"
        )
    if contract["generated_latent_frames"] != (
        contract["chunks"] * contract["latent_frames_per_chunk"]
    ):
        raise ValueError("latent frame/chunk contract is inconsistent")
    for case in cases:
        bits = action_bits(case["keys"])
        label = (
            TRANS_BITS_TO_LABEL[tuple(bits[:4])] * 9
            + ROT_BITS_TO_LABEL[tuple(bits[4:])]
        )
        if label != case["action_label"]:
            raise ValueError(
                f"{case['id']}: action_label={case['action_label']} does not match keys={case['keys']}"
            )
        if "action_weights" in case:
            weights = action_weights(case)
            active_keys = [key for key, value in zip(KEY_ORDER, weights) if value > 0]
            if active_keys != case["keys"]:
                raise ValueError(
                    f"{case['id']}: positive action_weights map to {active_keys}, "
                    f"not keys={case['keys']}"
                )
    return manifest


def action_bits(keys: list[str]) -> list[int]:
    bits = [int(key in keys) for key in KEY_ORDER]
    if tuple(bits[:4]) not in TRANS_BITS_TO_LABEL:
        raise ValueError(f"unsupported translation keys: {keys}")
    if tuple(bits[4:]) not in ROT_BITS_TO_LABEL:
        raise ValueError(f"unsupported look keys: {keys}")
    return bits


def action_weights(case: dict) -> list[float]:
    values = case.get("action_weights")
    if values is None:
        return [float(value) for value in action_bits(case["keys"])]
    if not isinstance(values, list) or len(values) != len(KEY_ORDER):
        raise ValueError(f"{case['id']}: action_weights must contain 8 values")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{case['id']}: action weights must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{case['id']}: action weights must be in [0, 1]")
        result.append(value)
    return result


def materialize_first_frame(case: dict, inputs_dir: str | Path) -> Path:
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(case["first_frame"]).suffix or ".png"
    target = inputs_dir / f"{case['id']}{suffix}"
    if target.exists():
        return target
    uri = case["first_frame"]
    if not uri.startswith("s3://"):
        source = Path(uri)
        if not source.is_file():
            raise FileNotFoundError(source)
        target.write_bytes(source.read_bytes())
        return target
    bucket_and_key = uri.removeprefix("s3://").split("/", 1)
    if len(bucket_and_key) == 2:
        # The AWS parity jobs expose the bucket through an S3 CSI mount. Prefer
        # that authenticated, region-local path so the harness does not depend
        # on ambient AWS CLI credentials. MINWM_S3_MOUNT can disable/override it.
        mount_root = os.environ.get("MINWM_S3_MOUNT", "/s3")
        mounted_source = Path(mount_root) / bucket_and_key[1]
        if mount_root and mounted_source.is_file():
            shutil.copyfile(mounted_source, target)
            return target
    subprocess.run(
        ["aws", "s3", "cp", "--only-show-errors", uri, str(target)],
        check=True,
    )
    return target


def build_minwm_message(case: dict, contract: dict, first_frame: Path) -> dict:
    use_weights = contract.get("action_output_format") == "primitive_float"
    action_row = action_weights(case) if use_weights else action_bits(case["keys"])
    pixel_actions = [action_row for _ in range(int(contract["generated_pixel_frames"]))]
    return {
        "schema_version": 2,
        "sample_id": case["id"],
        "messages": [
            {"role": "user", "type": "text", "content": case["prompt"]},
            {
                "role": "target",
                "type": "video",
                "output": {
                    "frames": int(contract["generated_pixel_frames"]),
                    "height": int(contract["height"]),
                    "width": int(contract["width"]),
                },
                "uri": str(first_frame),
                "reference_frame_count": int(contract["reference_pixel_frames"]),
                "controls": [
                    {
                        "type": "keyboard_direction_frame_interval",
                        "actions": pixel_actions,
                    }
                ],
            },
        ],
    }


def save_video(path: str | Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        np.asarray(frames, dtype=np.uint8),
        fps=int(fps),
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
