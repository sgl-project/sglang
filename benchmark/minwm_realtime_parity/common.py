#!/usr/bin/env python3
"""Shared, dependency-light helpers for the MinWM parity harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import urllib.request
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


def is_realtime_trace_event(message: dict) -> bool:
    """Return whether an out-of-band trace message can be skipped by clients."""
    return message.get("type") == "trace_event"


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
    _validate_chunk_contract(contract)
    case_ids = {case["id"] for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("parity manifest case ids must be unique")
    for case in cases:
        resolved_contract = resolve_case_contract(case, contract)
        trajectory = case.get("trajectory")
        if trajectory is not None:
            if case.get("action_schedule") is not None:
                raise ValueError(
                    f"{case['id']}: trajectory and action_schedule are mutually exclusive"
                )
            trajectory_action_labels(
                trajectory,
                expected_frames=int(resolved_contract["generated_latent_frames"]),
            )
            continue
        action_bits(case["keys"])
        label = action_label(case["keys"])
        if label != case["action_label"]:
            raise ValueError(
                f"{case['id']}: action_label={case['action_label']} does not match keys={case['keys']}"
            )
        _validate_action_schedule(case, resolved_contract)
        if "action_weights" in case:
            weights = action_weights(case)
            active_keys = [key for key, value in zip(KEY_ORDER, weights) if value > 0]
            if active_keys != case["keys"]:
                raise ValueError(
                    f"{case['id']}: positive action_weights map to {active_keys}, "
                    f"not keys={case['keys']}"
                )
        switch = case.get("prompt_switch")
        if switch is not None:
            _validate_prompt_switch(case, contract, switch)
            control_case_id = switch.get("control_case_id")
            if control_case_id is not None and control_case_id not in case_ids:
                raise ValueError(
                    f"{case['id']}: unknown prompt-switch control case "
                    f"{control_case_id!r}"
                )
    cases_by_id = {case["id"]: case for case in cases}
    for case in cases:
        switch = case.get("prompt_switch")
        control_case_id = switch.get("control_case_id") if switch else None
        if control_case_id is None:
            continue
        control = cases_by_id[control_case_id]
        if control.get("prompt_switch") is not None:
            raise ValueError(
                f"{case['id']}: prompt-switch control case must not switch prompt"
            )
        comparable_keys = ("prompt", "first_frame", "action_label", "keys")
        mismatches = [
            key for key in comparable_keys if case.get(key) != control.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"{case['id']}: control case {control_case_id!r} differs in "
                f"{mismatches}"
            )
    return manifest


def _validate_chunk_contract(contract: dict) -> None:
    chunk_sizes = contract.get("latent_chunk_sizes")
    if chunk_sizes is None:
        if contract["generated_latent_frames"] != (
            contract["chunks"] * contract["latent_frames_per_chunk"]
        ):
            raise ValueError("latent frame/chunk contract is inconsistent")
        return
    if (
        not isinstance(chunk_sizes, list)
        or len(chunk_sizes) != int(contract["chunks"])
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in chunk_sizes
        )
        or sum(chunk_sizes) != int(contract["generated_latent_frames"])
    ):
        raise ValueError("variable latent frame/chunk contract is inconsistent")


def resolve_case_contract(case: dict, contract: dict) -> dict:
    resolved = dict(contract)
    overrides = case.get("contract", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"{case['id']}: contract override must be an object")
    resolved.update(overrides)
    _validate_chunk_contract(resolved)
    return resolved


def _validate_prompt_switch(case: dict, contract: dict, switch: dict) -> None:
    if not isinstance(switch, dict):
        raise ValueError(f"{case['id']}: prompt_switch must be an object")
    prompt = switch.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{case['id']}: prompt_switch.prompt must be non-empty")
    if prompt == case["prompt"]:
        raise ValueError(
            f"{case['id']}: prompt_switch.prompt must differ from initial prompt"
        )
    target_chunk = switch.get("target_chunk")
    if isinstance(target_chunk, bool) or not isinstance(target_chunk, int):
        raise ValueError(f"{case['id']}: prompt_switch.target_chunk must be an integer")
    # The parity client queues the event immediately after init. MinWM's
    # adapter intentionally refuses to sample prompt events for chunk 0, so the
    # first deterministic cutover is chunk 1. Later wall-clock scheduling would
    # make the test depend on GPU/network timing instead of model semantics.
    if target_chunk != 1:
        raise ValueError(
            f"{case['id']}: deterministic prompt-switch parity currently "
            "requires target_chunk=1"
        )
    if target_chunk >= int(contract["chunks"]):
        raise ValueError(f"{case['id']}: prompt switch is outside the video")
    event_id = switch.get("event_id")
    if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
        raise ValueError(
            f"{case['id']}: prompt_switch.event_id must be a non-negative integer"
        )


def prompt_switch_boundary(case: dict, contract: dict) -> int | None:
    """Return the output pixel-frame index where the new prompt takes effect."""
    switch = case.get("prompt_switch")
    if switch is None:
        return None
    generated_latent_frames = int(contract["generated_latent_frames"])
    generated_pixel_frames = int(contract["generated_pixel_frames"])
    if generated_pixel_frames % generated_latent_frames:
        raise ValueError("pixel/latent frame contract is not integral")
    pixels_per_latent = generated_pixel_frames // generated_latent_frames
    generated_latents_before_switch = int(switch["target_chunk"]) * int(
        contract["latent_frames_per_chunk"]
    )
    return (
        int(contract["reference_pixel_frames"])
        + generated_latents_before_switch * pixels_per_latent
    )


def prompt_switch_event(case: dict) -> dict | None:
    switch = case.get("prompt_switch")
    if switch is None:
        return None
    return {
        "type": "event",
        "kind": "prompt",
        "payload": switch["prompt"],
        "event_id": int(switch["event_id"]),
    }


def action_bits(keys: list[str]) -> list[int]:
    bits = [int(key in keys) for key in KEY_ORDER]
    if tuple(bits[:4]) not in TRANS_BITS_TO_LABEL:
        raise ValueError(f"unsupported translation keys: {keys}")
    if tuple(bits[4:]) not in ROT_BITS_TO_LABEL:
        raise ValueError(f"unsupported look keys: {keys}")
    return bits


def action_label(keys: list[str]) -> int:
    bits = action_bits(keys)
    return TRANS_BITS_TO_LABEL[tuple(bits[:4])] * 9 + ROT_BITS_TO_LABEL[tuple(bits[4:])]


def _validate_action_schedule(case: dict, contract: dict) -> None:
    schedule = case.get("action_schedule")
    if schedule is None:
        return
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(f"{case['id']}: action_schedule must be a non-empty list")
    generated_pixels = int(contract["generated_pixel_frames"])
    generated_latents = int(contract["generated_latent_frames"])
    if generated_pixels % generated_latents:
        raise ValueError("pixel/latent frame contract is not integral")
    pixels_per_latent = generated_pixels // generated_latents
    cursor = 0
    for segment in schedule:
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        keys = segment.get("keys")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ValueError(
                f"{case['id']}: action schedule boundaries must be integers"
            )
        if start != cursor or end <= start or end > generated_pixels:
            raise ValueError(
                f"{case['id']}: action schedule must be contiguous; "
                f"expected start {cursor}, got [{start}, {end})"
            )
        if start % pixels_per_latent or end % pixels_per_latent:
            raise ValueError(
                f"{case['id']}: action boundaries must align to "
                f"{pixels_per_latent}-pixel latent frames"
            )
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError(f"{case['id']}: action schedule keys must be a list")
        action_bits(keys)
        cursor = end
    if cursor != generated_pixels:
        raise ValueError(
            f"{case['id']}: action schedule ends at {cursor}, "
            f"expected {generated_pixels}"
        )


def action_label_sequence(case: dict, contract: dict) -> list[int]:
    contract = resolve_case_contract(case, contract)
    if case.get("trajectory") is not None:
        return trajectory_action_labels(
            case["trajectory"],
            expected_frames=int(contract["generated_latent_frames"]),
        )
    schedule = case.get("action_schedule")
    generated_latents = int(contract["generated_latent_frames"])
    if schedule is None:
        return [int(case["action_label"])] * generated_latents
    pixels_per_latent = int(contract["generated_pixel_frames"]) // generated_latents
    labels = []
    for segment in schedule:
        segment_latents = (
            int(segment["end_frame"]) - int(segment["start_frame"])
        ) // pixels_per_latent
        labels.extend([action_label(segment["keys"])] * segment_latents)
    if len(labels) != generated_latents:
        raise ValueError(
            f"{case['id']}: action schedule produced {len(labels)} latent labels, "
            f"expected {generated_latents}"
        )
    return labels


def trajectory_action_labels(trajectory: str, *, expected_frames: int) -> list[int]:
    """Match minWM ``trajectory_str_to_action_labels`` including the leading idle."""
    if not isinstance(trajectory, str) or not trajectory.strip():
        raise ValueError("trajectory must be a non-empty string")
    labels = [0]
    for raw_segment in trajectory.split(","):
        segment = raw_segment.strip()
        if not segment:
            continue
        if "*" in segment:
            key_expression, count_text = segment.split("*", 1)
        elif "-" in segment:
            key_expression, count_text = segment.split("-", 1)
        else:
            key_expression, count_text = segment, "1"
        keys = [
            key.strip().lower()
            for key in key_expression.strip().lower().split("+")
            if key.strip()
        ]
        normalized_keys = [
            {"up": "i", "left": "j", "down": "k", "right": "l"}.get(key, key)
            for key in keys
        ]
        count = int(float(count_text.strip()))
        if count < 0:
            raise ValueError(f"negative trajectory count in {segment!r}")
        labels.extend([action_label(normalized_keys)] * count)
    if len(labels) != expected_frames:
        raise ValueError(
            f"trajectory produced {len(labels)} latent labels, "
            f"expected {expected_frames}"
        )
    return labels


def pixel_action_bits(case: dict, contract: dict) -> list[list[int]]:
    schedule = case.get("action_schedule")
    generated_pixels = int(contract["generated_pixel_frames"])
    if schedule is None:
        return [action_bits(case["keys"]) for _ in range(generated_pixels)]
    rows = []
    for segment in schedule:
        count = int(segment["end_frame"]) - int(segment["start_frame"])
        rows.extend([action_bits(segment["keys"]) for _ in range(count)])
    if len(rows) != generated_pixels:
        raise ValueError(
            f"{case['id']}: action schedule produced {len(rows)} pixel actions, "
            f"expected {generated_pixels}"
        )
    return rows


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
    uri = case["first_frame"]
    if not target.exists() and uri.startswith(("http://", "https://")):
        request = urllib.request.Request(
            uri,
            headers={"User-Agent": "sglang-minwm-parity/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
    elif not target.exists() and not uri.startswith("s3://"):
        source = Path(uri)
        if not source.is_file():
            raise FileNotFoundError(source)
        target.write_bytes(source.read_bytes())
    elif not target.exists():
        bucket_and_key = uri.removeprefix("s3://").split("/", 1)
        # The AWS parity jobs expose the bucket through an S3 CSI mount. Prefer
        # that authenticated, region-local path so the harness does not depend
        # on ambient AWS CLI credentials. MINWM_S3_MOUNT can disable/override it.
        mount_root = os.environ.get("MINWM_S3_MOUNT", "/s3")
        mounted_source = (
            Path(mount_root) / bucket_and_key[1]
            if mount_root and len(bucket_and_key) == 2
            else None
        )
        if mounted_source is not None and mounted_source.is_file():
            shutil.copyfile(mounted_source, target)
        else:
            subprocess.run(
                ["aws", "s3", "cp", "--only-show-errors", uri, str(target)],
                check=True,
            )
    expected_sha256 = case.get("first_frame_sha256")
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(target)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{case['id']}: first frame SHA-256 {actual_sha256} does not "
                f"match {expected_sha256}"
            )
    return target


def build_minwm_message(case: dict, contract: dict, first_frame: Path) -> dict:
    use_weights = contract.get("action_output_format") == "primitive_float"
    if use_weights and case.get("action_schedule") is not None:
        raise ValueError("scheduled primitive_float actions are not implemented")
    pixel_actions = (
        [action_weights(case) for _ in range(int(contract["generated_pixel_frames"]))]
        if use_weights
        else pixel_action_bits(case, contract)
    )
    controls = [
        {
            "type": "keyboard_direction_frame_interval",
            "actions": pixel_actions,
        }
    ]
    switch_boundary = prompt_switch_boundary(case, contract)
    if switch_boundary is not None:
        total_frames = int(contract["reference_pixel_frames"]) + int(
            contract["generated_pixel_frames"]
        )
        controls.append(
            {
                "type": "text_prompt_interval",
                "segments": [
                    {
                        "start": 0,
                        "end": switch_boundary,
                        "text": case["prompt"],
                    },
                    {
                        "start": switch_boundary,
                        "end": total_frames,
                        "text": case["prompt_switch"]["prompt"],
                    },
                ],
            }
        )
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
                "controls": controls,
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
