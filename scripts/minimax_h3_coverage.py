#!/usr/bin/env python3
"""Coverage run for a live MiniMax-H3 video server.

The loaded fl2va partition serves text-to-video and first/last-frame video.
Reference-image, reference-video, and reference-audio need the ref2va
partition, which this script does not launch.

What it checks, in priority order, inside a wall-clock budget (default 120
minutes):

1. Admission. Out-of-range duration, unknown tasks (including i2va/l2va as
   task names), a middle keyframe, ref2va on this partition, fps, guidance,
   a negative prompt, quality=high, and steps < 2 must be rejected.
2. Judge calibration. Qwen watches the known-good 50-step clip and the
   known-bad 3-step clip. Later visual scores are advisory if that pair
   is not separated.
3. Modes. First frame, last frame, and both, at the same 960x544 canvas as
   the source frames.
4. The published 768 short-edge 16:9 canvas, then the duration ends (4 s and
   15 s), then the other aspect ratios.
5. Settings. Same seed twice, a different seed, 20 steps, an alternate
   flow_shift, and 3:4 if the budget remains.

Qwen scores the picture from the mp4. It does not hear audio. Near-silence
is rejected separately with ffmpeg. Anatomy and the soundtrack are left for
a person.

Usage:
  python3 scripts/minimax_h3_coverage.py
  python3 scripts/minimax_h3_coverage.py --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FPS = 24
MAX_PIXELS = 768 * 1344
CANVAS = 32
# Wall clock of the measured 960x544, 5 s, 50-step clip, divided by 49 evals.
BASE_SEC_PER_EVAL = 11.4
BASE_TOKENS = 18870

H3_URL = os.environ.get("H3_URL", "http://127.0.0.1:30010")
QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:11435")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen38next-nvfp4")
POSITIVE = REPO / "outputs/fde21467-5aa8-423a-9fd6-2d3adbb3e585.mp4"
NEGATIVE = REPO / "outputs/b195e430-c3fe-4bac-84fe-bae1aa733081.mp4"

REPORT_LOCK = threading.RLock()


def align_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    return frame_count + (5 - frame_count) % 17


def video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def nearest_multiple(value: float) -> int:
    return max(CANVAS, int(round(float(value) / CANVAS)) * CANVAS)


def canvas_size(short_edge: int, aspect: str) -> tuple[int, int]:
    w_r, h_r = (int(part) for part in aspect.split(":"))
    ratio = w_r / h_r
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / area)
        width *= scale
        height *= scale
    return nearest_multiple(width), nearest_multiple(height)


def token_count(short_edge: int, aspect: str, duration: float) -> int:
    width, height = canvas_size(short_edge, aspect)
    frames = align_frame_count(int(round(duration * FPS)))
    return (width // CANVAS) * (height // CANVAS) * video_latent_t(frames)


def delivered_seconds(duration: float) -> float:
    frames = align_frame_count(int(round(duration * FPS)))
    return frames / FPS


def t2va_prompt(scene: str, sound: str) -> str:
    return (
        "integrated_multimodal_description: "
        f"[Shot 1] Live-action. {scene}\n\n"
        f"overall_soundscape: {sound}\n\n"
        "non_diegetic_music: None."
    )


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180,
) -> tuple[int, dict[str, Any] | str]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    if not raw:
        return status, {}
    try:
        return status, json.loads(raw.decode())
    except json.JSONDecodeError:
        return status, raw.decode(errors="replace")


def detail_text(body: dict[str, Any] | str) -> str:
    if isinstance(body, str):
        return body
    detail = body.get("detail", body.get("error", body))
    if not isinstance(detail, str):
        detail = json.dumps(detail)
    return detail


def base_request(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": os.environ.get(
            "H3_MODEL", os.path.expanduser("~/models/MiniMax-H3")
        ),
        "prompt": "A red ball rolls across a wooden floor.",
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 544,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        "num_inference_steps": 50,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": 1101,
    }
    body.update(overrides)
    return body


def contract_cases(keyframe: str) -> list[dict[str, Any]]:
    image = Path(keyframe).resolve().as_uri()
    cond = {
        "type": "image",
        "uri": image,
        "role": "keyframe",
        "frame_index": 0,
    }
    return [
        {
            "id": "reject_duration_low",
            "expect": "duration_seconds",
            "body": base_request(
                target={
                    "short_edge": 544,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 3.9,
                }
            ),
        },
        {
            "id": "reject_duration_high",
            "expect": "duration_seconds",
            "body": base_request(
                target={
                    "short_edge": 544,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 15.1,
                }
            ),
        },
        {
            "id": "reject_aspect",
            "expect": "aspect_ratio",
            "body": base_request(
                target={
                    "short_edge": 544,
                    "aspect_ratio": "5:4",
                    "duration_seconds": 5.0,
                }
            ),
        },
        {
            "id": "reject_unknown_task",
            "expect": "task",
            "body": base_request(task="nope"),
        },
        {
            "id": "reject_i2va_task_name",
            "expect": "task",
            "body": base_request(task="i2va", conditions=[cond]),
        },
        {
            "id": "reject_l2va_task_name",
            "expect": "task",
            "body": base_request(
                task="l2va",
                conditions=[{**cond, "frame_index": -1}],
            ),
        },
        {
            "id": "reject_t2va_with_condition",
            "expect": "conditions",
            "body": base_request(conditions=[cond]),
        },
        {
            "id": "reject_fl2va_without_condition",
            "expect": "conditions",
            "body": base_request(task="fl2va", conditions=[]),
        },
        {
            "id": "reject_middle_keyframe",
            "expect": "frame_index",
            "body": base_request(
                task="fl2va",
                conditions=[{**cond, "frame_index": 3}],
            ),
        },
        {
            "id": "reject_ref2va_on_fl2va_partition",
            "expect": "partition",
            "body": base_request(
                task="ref2va",
                conditions=[
                    {
                        "type": "image",
                        "uri": image,
                        "role": "reference",
                    }
                ],
            ),
        },
        {
            "id": "reject_fps",
            "expect": "fps",
            "body": base_request(fps=12),
        },
        {
            "id": "reject_guidance_scale",
            "expect": "guidance_scale",
            "body": base_request(guidance_scale=6.0),
        },
        {
            "id": "reject_negative_prompt",
            "expect": "negative_prompt",
            "body": base_request(negative_prompt="blurry"),
        },
        {
            "id": "reject_quality_high",
            "expect": "quality",
            "body": base_request(quality="high"),
        },
        {
            "id": "reject_steps_below_2",
            "expect": "num_inference_steps",
            "body": base_request(num_inference_steps=1),
        },
        {
            "id": "reject_short_edge_zero",
            "expect": "short_edge",
            "body": base_request(
                target={
                    "short_edge": 0,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                }
            ),
        },
    ]


def generation_jobs(first: Path, last: Path) -> list[dict[str, Any]]:
    first_uri = first.resolve().as_uri()
    last_uri = last.resolve().as_uri()
    end = f"{delivered_seconds(5.0):.2f}"
    jobs: list[dict[str, Any]] = [
        {
            "id": "fl2va_first",
            "priority": "required",
            "subject": "tiger",
            "short_edge": 544,
            "aspect": "16:9",
            "duration": 5.0,
            "steps": 50,
            "seed": 3101,
            "keyframes": [("first", first)],
            "body": base_request(
                seed=3101,
                task="fl2va",
                prompt=(
                    "For the target video, at 0.00 seconds into the target video, "
                    "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                    "integrated_multimodal_description: [Shot 1] Live-action. "
                    "The opening frame is the misty field and the tiger in "
                    "<Picture 1>. The tiger keeps walking slowly to the left "
                    "through the grass while the fog drifts. The field, the "
                    "warm light, and the tiger's stripes stay consistent.\n\n"
                    "overall_soundscape: Distant birds and soft footsteps in grass.\n\n"
                    "non_diegetic_music: None."
                ),
                conditions=[
                    {
                        "type": "image",
                        "uri": first_uri,
                        "role": "keyframe",
                        "frame_index": 0,
                    }
                ],
                target={
                    "short_edge": 544,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                },
            ),
        },
        {
            "id": "fl2va_last",
            "priority": "required",
            "subject": "tiger",
            "short_edge": 544,
            "aspect": "16:9",
            "duration": 5.0,
            "steps": 50,
            "seed": 3102,
            "keyframes": [("last", last)],
            "body": base_request(
                seed=3102,
                task="fl2va",
                prompt=(
                    "How the reference pictures align with the target video — "
                    f"<Picture 1> (from [Shot 1]) aligns with the {end}-second "
                    "mark of the target video.\n\n"
                    "integrated_multimodal_description: [Shot 1] Live-action. "
                    "The clip opens a few steps earlier in the same misty field. "
                    "The tiger walks slowly to the left until its pose and the "
                    f"field match <Picture 1> at {end} seconds.\n\n"
                    "overall_soundscape: Distant birds and soft footsteps in grass.\n\n"
                    "non_diegetic_music: None."
                ),
                conditions=[
                    {
                        "type": "image",
                        "uri": last_uri,
                        "role": "keyframe",
                        "frame_index": -1,
                    }
                ],
                target={
                    "short_edge": 544,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                },
            ),
        },
        {
            "id": "fl2va_both",
            "priority": "required",
            "subject": "tiger",
            "short_edge": 544,
            "aspect": "16:9",
            "duration": 5.0,
            "steps": 50,
            "seed": 3103,
            "keyframes": [("first", first), ("last", last)],
            "body": base_request(
                seed=3103,
                task="fl2va",
                prompt=(
                    "How the reference pictures align with the target video — "
                    "Picture 1 (from Shot 1) aligns with the 0.00-second mark of "
                    "the target video; Picture 2 (from Shot 1) aligns with the "
                    f"{end}-second mark of the target video.\n\n"
                    "integrated_multimodal_description: [Shot 1] Live-action, one "
                    "continuous shot. The tiger starts in the pose of Picture 1, "
                    "under the tree, and walks slowly to the left until it reaches "
                    f"the pose of Picture 2 at {end} seconds. The field and the "
                    "fog stay consistent.\n\n"
                    "overall_soundscape: Distant birds and soft footsteps in grass.\n\n"
                    "non_diegetic_music: None."
                ),
                conditions=[
                    {
                        "type": "image",
                        "uri": first_uri,
                        "role": "keyframe",
                        "frame_index": 0,
                    },
                    {
                        "type": "image",
                        "uri": last_uri,
                        "role": "keyframe",
                        "frame_index": -1,
                    },
                ],
                target={
                    "short_edge": 544,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                },
            ),
        },
    ]

    balloon = t2va_prompt(
        "A single red balloon floats above a green meadow.",
        "Light wind.",
    )
    for job_id, seed in (("seed_repeat_a", 6101), ("seed_repeat_b", 6101)):
        jobs.append(
            {
                "id": job_id,
                "priority": "required",
                "subject": "red balloon",
                "short_edge": 544,
                "aspect": "16:9",
                "duration": 5.0,
                "steps": 8,
                "seed": seed,
                "visual": "informational",
                "pair": "seed_repeat",
                "keyframes": [],
                "body": base_request(
                    seed=seed,
                    num_inference_steps=8,
                    prompt=balloon,
                ),
            }
        )
    jobs.append(
        {
            "id": "seed_different",
            "priority": "required",
            "subject": "red balloon",
            "short_edge": 544,
            "aspect": "16:9",
            "duration": 5.0,
            "steps": 8,
            "seed": 6102,
            "visual": "informational",
            "pair": "seed_different",
            "keyframes": [],
            "body": base_request(
                seed=6102,
                num_inference_steps=8,
                prompt=balloon,
            ),
        }
    )

    scenes = [
        (
            "t2va_768_16x9",
            "required",
            768,
            "16:9",
            5.0,
            50,
            4101,
            "yellow taxi",
            "A bright yellow taxi drives slowly along a wet city street at dusk. "
            "Streetlights reflect on the asphalt and the wheels turn.",
            "Tire noise on wet pavement and a distant engine.",
        ),
        (
            "t2va_15s_16x9",
            "required",
            384,
            "16:9",
            15.0,
            50,
            4102,
            "blue sailboat",
            "A blue sailboat moves slowly from left to right across a calm harbor "
            "over the full length of the clip. The sail stays up and the water ripples.",
            "Water against the hull and a light breeze.",
        ),
        (
            "t2va_4s_16x9",
            "required",
            544,
            "16:9",
            4.0,
            50,
            4103,
            "green frog",
            "A green frog sits on a lily pad, then hops once to the next pad. "
            "The pond and the pads stay visible.",
            "A short splash and quiet water.",
        ),
        (
            "t2va_9x16",
            "required",
            544,
            "9:16",
            5.0,
            50,
            4104,
            "red lighthouse",
            "A tall red lighthouse stands on a rocky cliff above the sea. "
            "Waves hit the rocks and the portrait frame stays on the tower.",
            "Waves and wind.",
        ),
        (
            "t2va_1x1",
            "required",
            544,
            "1:1",
            5.0,
            50,
            4105,
            "white coffee cup",
            "A white coffee cup sits on a wooden table. Steam rises from the cup. "
            "The square frame stays close on the cup.",
            "A quiet room and a faint ceramic tap.",
        ),
        (
            "t2va_21x9",
            "required",
            544,
            "21:9",
            5.0,
            50,
            4106,
            "silver train",
            "A silver passenger train crosses a flat desert from left to right "
            "in a very wide frame. The cars stay connected and the wheels turn.",
            "Rail noise and open wind.",
        ),
        (
            "t2va_4x3",
            "required",
            544,
            "4:3",
            5.0,
            50,
            4107,
            "orange cat",
            "An orange cat sits on a windowsill and turns its head. "
            "Daylight comes through the window.",
            "A quiet room.",
        ),
        (
            "t2va_steps20",
            "optional",
            544,
            "16:9",
            5.0,
            20,
            5101,
            "yellow taxi",
            "A bright yellow taxi drives slowly along a wet city street at dusk.",
            "Tire noise on wet pavement.",
        ),
        (
            "t2va_3x4",
            "optional",
            544,
            "3:4",
            5.0,
            50,
            4108,
            "purple umbrella",
            "A person holds a purple umbrella on a rainy sidewalk. "
            "Rain falls and the umbrella stays open. The face need not be detailed.",
            "Rain on the umbrella.",
        ),
        (
            "t2va_flow_shift_6",
            "optional",
            544,
            "16:9",
            5.0,
            12,
            7101,
            "red balloon",
            "A single red balloon floats above a green meadow.",
            "Light wind.",
        ),
    ]
    for (
        job_id,
        priority,
        short_edge,
        aspect,
        duration,
        steps,
        seed,
        subject,
        scene,
        sound,
    ) in scenes:
        body = base_request(
            seed=seed,
            num_inference_steps=steps,
            prompt=t2va_prompt(scene, sound),
            target={
                "short_edge": short_edge,
                "aspect_ratio": aspect,
                "duration_seconds": duration,
            },
        )
        if job_id == "t2va_flow_shift_6":
            body["flow_shift"] = 6.0
        jobs.append(
            {
                "id": job_id,
                "priority": priority,
                "subject": subject,
                "short_edge": short_edge,
                "aspect": aspect,
                "duration": duration,
                "steps": steps,
                "seed": seed,
                "keyframes": [],
                "body": body,
            }
        )

    return jobs


def estimate_seconds(job: dict[str, Any], sec_per_eval: float) -> float:
    tokens = token_count(job["short_edge"], job["aspect"], job["duration"])
    ratio = max(tokens / BASE_TOKENS, 0.05)
    evals = max(int(job["steps"]) - 1, 1)
    return sec_per_eval * (ratio**1.5) * evals + 30.0


def ffprobe(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,nb_frames,duration,avg_frame_rate,pix_fmt:format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(raw)


def audio_levels(path: Path) -> dict[str, float]:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    text = proc.stderr
    levels: dict[str, float] = {}
    for name in ("mean_volume", "max_volume"):
        match = re.search(rf"{name}:\s*(-?\d+(?:\.\d+)?)\s*dB", text)
        if match:
            levels[name] = float(match.group(1))
    return levels


def extract_frame(video: Path, dest: Path, at_seconds: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{at_seconds:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(dest),
        ]
    )


def frame_psnr(left: Path, right: Path) -> float | None:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(left),
            "-i",
            str(right),
            "-lavfi",
            "psnr",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    match = re.search(r"average:([0-9.]+|inf)", proc.stderr)
    if not match:
        return None
    if match.group(1) == "inf":
        return 99.0
    return float(match.group(1))


def data_url(path: Path, kind: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{kind};base64,{encoded}"


def parse_model_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response was not an object")
    return parsed


def qwen_judge(
    video: Path,
    *,
    subject: str,
    prompt: str,
    keyframes: list[tuple[str, Path]],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for label, path in keyframes:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url(path, "image/png")},
            }
        )
        content.append({"type": "text", "text": f"Condition image for the {label} frame."})
    content.append(
        {
            "type": "video_url",
            "video_url": {"url": data_url(video, "video/mp4")},
        }
    )
    content.append(
        {
            "type": "text",
            "text": (
                "Score this generated video. Reply with one JSON object and no markdown.\n"
                "{\n"
                '  "pass": true or false,\n'
                '  "subject_present": true or false,\n'
                '  "matches_prompt": true or false,\n'
                '  "coherent_scene": true or false,\n'
                '  "has_motion": true or false,\n'
                '  "first_frame_matches": true, false, or null,\n'
                '  "last_frame_matches": true, false, or null,\n'
                '  "defects": ["short notes"],\n'
                '  "description": "one sentence"\n'
                "}\n"
                f"Required subject: {subject}.\n"
                "pass is true only when that subject is clearly drawn, the scene is "
                "coherent rather than fog, noise, or an undefined smear, and the clip "
                "follows the prompt. A vaguely related color wash is a fail.\n"
                "Set first_frame_matches or last_frame_matches only when a condition "
                "image for that end was attached; otherwise null.\n"
                f"Prompt:\n{prompt}"
            ),
        }
    )
    status, body = http_json(
        "POST",
        f"{QWEN_URL}/v1/chat/completions",
        {
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 300,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=240,
    )
    if status != 200 or not isinstance(body, dict):
        return {"error": detail_text(body), "http_status": status}
    message = body["choices"][0]["message"]["content"]
    try:
        parsed = parse_model_json(message)
    except (json.JSONDecodeError, ValueError) as exc:
        return {"error": str(exc), "raw": message}
    parsed["usage"] = body.get("usage")
    return parsed


class CoverageRun:
    def __init__(self, out: Path, minutes: float, dry_run: bool) -> None:
        self.out = out
        self.deadline = time.monotonic() + minutes * 60.0
        self.dry_run = dry_run
        self.sec_per_eval = BASE_SEC_PER_EVAL
        self.report: dict[str, Any] = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "budget_minutes": minutes,
            "h3_url": H3_URL,
            "qwen_url": QWEN_URL,
            "judge_reliable": None,
            "blocked": [
                "ref2va is a separate weight partition and is not on disk, so "
                "reference-image, reference-video, reference-audio, and "
                "video-to-video were not generated.",
                "fps is not a generation option. The server rejects an explicit "
                "fps and always delivers 24.",
                "guidance_scale, negative_prompt, and quality=high are rejected "
                "on this checkpoint and this GPU.",
                "W8A16 with DiT offload is a different server configuration and "
                "was not started.",
            ],
            "needs_human": [
                "Listen to the mp4s. Qwen cannot hear them. The script only "
                "rejects a near-silent audio track.",
                "Check limbs, faces, and flicker. Qwen scores the subject and "
                "whether the clip moves, not anatomy.",
                "R2V (ref2va) is a required follow-up, not an optional extra. "
                "The weights are a separate partition and were not loaded.",
            ],
            "cases": [],
        }
        self.cases_by_id: dict[str, dict[str, Any]] = {}
        self.judge_pool = ThreadPoolExecutor(max_workers=1)
        self.judge_futures: list[Future] = []
        self.videos: dict[str, Path] = {}

    def save(self) -> None:
        with REPORT_LOCK:
            self.out.mkdir(parents=True, exist_ok=True)
            target = self.out / "report.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.report, indent=2))
            tmp.replace(target)

    def add(self, case: dict[str, Any]) -> None:
        self.cases_by_id[case["id"]] = case
        self.report["cases"].append(case)
        state = case.get("status", "recorded")
        print(f"[{case['id']}] {state} {case.get('detail', '')}", flush=True)
        self.save()

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def run_contract(self, cases: list[dict[str, Any]]) -> None:
        for case in cases:
            if self.dry_run:
                self.add({"id": case["id"], "kind": "contract", "status": "dry-run"})
                continue
            started = time.monotonic()
            status, body = http_json(
                "POST", f"{H3_URL}/v1/videos", case["body"], timeout=60
            )
            text = detail_text(body)
            record: dict[str, Any] = {
                "id": case["id"],
                "kind": "contract",
                "priority": "required",
                "http_status": status,
                "elapsed_s": round(time.monotonic() - started, 2),
            }
            if status == 400 and case["expect"].lower() in text.lower():
                record["status"] = "pass"
                record["detail"] = text[:500]
            elif isinstance(body, dict) and body.get("id") and body.get("status") not in {
                "completed",
                "failed",
            }:
                record.update(self._wait_for_rejection(str(body["id"]), case["expect"]))
            else:
                record["status"] = "fail"
                record["detail"] = text[:500]
            self.add(record)

    def _wait_for_rejection(self, video_id: str, expect: str) -> dict[str, Any]:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status, body = http_json("GET", f"{H3_URL}/v1/videos/{video_id}", timeout=30)
            if not isinstance(body, dict):
                return {"status": "fail", "detail": detail_text(body)[:500]}
            state = body.get("status")
            if state in {"completed", "failed"}:
                text = detail_text(body.get("error") or body)
                ok = state == "failed" and expect.lower() in text.lower()
                return {
                    "status": "pass" if ok else "fail",
                    "detail": text[:500],
                    "video_id": video_id,
                }
            time.sleep(2)
        return {
            "status": "fail",
            "detail": f"rejection for {video_id} did not finish within 180s",
            "video_id": video_id,
        }

    def judge_file(
        self,
        case_id: str,
        video: Path,
        *,
        subject: str,
        prompt: str,
        keyframes: list[tuple[str, Path]],
        priority: str,
        kind: str,
        expect_visual_pass: bool,
    ) -> None:
        if self.dry_run:
            self.add({"id": case_id, "kind": kind, "status": "dry-run", "video": str(video)})
            return
        started = time.monotonic()
        try:
            verdict = qwen_judge(
                video, subject=subject, prompt=prompt, keyframes=keyframes
            )
        except Exception as exc:  # noqa: BLE001 - record and keep the suite moving
            verdict = {"error": str(exc)}
        structural = self._structural(video, expected=None)
        visual_ok = (
            verdict.get("pass") is True
            and verdict.get("subject_present") is True
            and verdict.get("coherent_scene") is True
        )
        passed = structural["status"] == "pass" and visual_ok is expect_visual_pass
        self.add(
            {
                "id": case_id,
                "kind": kind,
                "priority": priority,
                "status": "pass" if passed else "fail",
                "detail": verdict.get("description") or verdict.get("error") or "",
                "video": str(video),
                "structural": structural,
                "qwen": verdict,
                "elapsed_s": round(time.monotonic() - started, 2),
            }
        )

    def _structural(self, video: Path, expected: dict[str, Any] | None) -> dict[str, Any]:
        problems: list[str] = []
        try:
            probe = ffprobe(video)
        except subprocess.CalledProcessError as exc:
            return {"status": "fail", "detail": str(exc)}
        streams = probe.get("streams") or []
        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"),
            None,
        )
        if video_stream is None:
            problems.append("no video stream")
        if audio_stream is None or audio_stream.get("codec_name") != "aac":
            problems.append("audio is not aac")
        if video_stream is not None and video_stream.get("codec_name") != "h264":
            problems.append("video is not h264")
        duration = 0.0
        if video_stream is not None:
            duration = float(video_stream.get("duration") or 0.0)
        levels = audio_levels(video)
        mean_volume = levels.get("mean_volume")
        if mean_volume is None or mean_volume < -70:
            problems.append(f"audio near silence ({mean_volume})")
        motion = None
        if duration > 0:
            first = self.out / "frames" / f"{video.stem}_first.png"
            end = self.out / "frames" / f"{video.stem}_last.png"
            extract_frame(video, first, 0.0)
            extract_frame(video, end, max(duration - 0.08, 0.0))
            motion = frame_psnr(first, end)
            if motion is not None and motion >= 45:
                problems.append(f"almost no motion (psnr {motion:.1f})")
        if expected and video_stream is not None:
            width = int(video_stream.get("width") or 0)
            height = int(video_stream.get("height") or 0)
            if (width, height) != (expected["width"], expected["height"]):
                problems.append(
                    f"size {width}x{height}, expected {expected['width']}x{expected['height']}"
                )
            if abs(duration - expected["seconds"]) > 0.08:
                problems.append(
                    f"duration {duration:.3f}s, expected {expected['seconds']:.3f}s"
                )
        return {
            "status": "fail" if problems else "pass",
            "detail": "; ".join(problems),
            "ffprobe": probe,
            "audio": levels,
            "motion_psnr": motion,
        }

    def generate(self, job: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        status, body = http_json("POST", f"{H3_URL}/v1/videos", job["body"], timeout=90)
        if status != 200 or not isinstance(body, dict) or not body.get("id"):
            record = {
                "id": job["id"],
                "kind": "generate",
                "priority": job["priority"],
                "status": "fail",
                "detail": detail_text(body)[:800],
                "http_status": status,
                "elapsed_s": round(time.monotonic() - started, 2),
            }
            self.add(record)
            return record
        video_id = str(body["id"])
        print(f"[{job['id']}] queued {video_id}", flush=True)
        poll_deadline = time.monotonic() + max(
            estimate_seconds(job, self.sec_per_eval) * 3,
            900,
        )
        final: dict[str, Any] | None = None
        while time.monotonic() < poll_deadline:
            _, current = http_json("GET", f"{H3_URL}/v1/videos/{video_id}", timeout=30)
            if isinstance(current, dict) and current.get("status") in {
                "completed",
                "failed",
            }:
                final = current
                break
            elapsed = time.monotonic() - started
            if int(elapsed) % 60 < 5:
                print(
                    f"[{job['id']}] {current.get('status') if isinstance(current, dict) else current} "
                    f"{elapsed:.0f}s",
                    flush=True,
                )
            time.sleep(5)
        elapsed = time.monotonic() - started
        if final is None or final.get("status") != "completed":
            record = {
                "id": job["id"],
                "kind": "generate",
                "priority": job["priority"],
                "status": "fail",
                "detail": "timed out"
                if final is None
                else detail_text(final.get("error") or final)[:800],
                "video_id": video_id,
                "elapsed_s": round(elapsed, 2),
            }
            self.add(record)
            return record
        dest = self.out / "videos" / f"{job['id']}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{H3_URL}/v1/videos/{video_id}/content", dest)
        self.videos[job["id"]] = dest
        evals = max(int(job["steps"]) - 1, 1)
        tokens = token_count(job["short_edge"], job["aspect"], job["duration"])
        ratio = max(tokens / BASE_TOKENS, 0.05)
        implied = (elapsed / evals) / (ratio**1.5)
        self.sec_per_eval = 0.7 * self.sec_per_eval + 0.3 * implied
        width, height = canvas_size(job["short_edge"], job["aspect"])
        structural = self._structural(
            dest,
            {
                "width": width,
                "height": height,
                "seconds": delivered_seconds(job["duration"]),
            },
        )
        record = {
            "id": job["id"],
            "kind": "generate",
            "priority": job["priority"],
            "visual": job.get("visual", "required" if job["priority"] == "required" else "optional"),
            "status": "pending-judge" if structural["status"] == "pass" else "fail",
            "detail": structural["detail"],
            "video_id": video_id,
            "video": str(dest),
            "elapsed_s": round(elapsed, 2),
            "sec_per_eval_baseline": round(self.sec_per_eval, 2),
            "expected_size": f"{width}x{height}",
            "structural": structural,
            "server_job": {
                "size": final.get("size"),
                "seconds": final.get("seconds"),
            },
        }
        self.add(record)
        return record

    def attach_judge(self, job: dict[str, Any], record: dict[str, Any]) -> None:
        if "video" not in record:
            return
        video = Path(record["video"])

        def _judge() -> None:
            try:
                verdict = qwen_judge(
                    video,
                    subject=job["subject"],
                    prompt=job["body"]["prompt"],
                    keyframes=job.get("keyframes") or [],
                )
            except Exception as exc:  # noqa: BLE001
                verdict = {"error": str(exc)}
            visual_required = job.get("visual", "required") != "informational"
            visual_ok = (
                verdict.get("pass") is True
                and verdict.get("subject_present") is True
                and verdict.get("coherent_scene") is True
            )
            if job["id"] == "fl2va_first":
                visual_ok = visual_ok and verdict.get("first_frame_matches") is True
            elif job["id"] == "fl2va_last":
                visual_ok = visual_ok and verdict.get("last_frame_matches") is True
            elif job["id"] == "fl2va_both":
                visual_ok = (
                    visual_ok
                    and verdict.get("first_frame_matches") is True
                    and verdict.get("last_frame_matches") is True
                )
            with REPORT_LOCK:
                record["qwen"] = verdict
                if record["structural"]["status"] != "pass":
                    record["status"] = "fail"
                elif not visual_required:
                    record["status"] = "pass"
                    record["visual_status"] = "pass" if visual_ok else "advisory-fail"
                elif self.report["judge_reliable"] is False:
                    record["status"] = "pass"
                    record["visual_status"] = "advisory"
                else:
                    record["status"] = "pass" if visual_ok else "fail"
                record["detail"] = verdict.get("description") or verdict.get("error") or record["detail"]
            print(f"[{job['id']}] judge {record['status']} {record['detail']}", flush=True)
            self.save()

        self.judge_futures.append(self.judge_pool.submit(_judge))

    def compare_seeds(self) -> None:
        left = self.videos.get("seed_repeat_a")
        right = self.videos.get("seed_repeat_b")
        other = self.videos.get("seed_different")
        if left is None or right is None:
            self.add(
                {
                    "id": "seed_repeat_match",
                    "kind": "compare",
                    "priority": "required",
                    "status": "skip",
                    "detail": "one of the repeat clips was not produced",
                }
            )
            return
        frame_a = self.out / "frames" / "seed_a.png"
        frame_b = self.out / "frames" / "seed_b.png"
        extract_frame(left, frame_a, 1.0)
        extract_frame(right, frame_b, 1.0)
        same = frame_psnr(frame_a, frame_b)
        different = None
        if other is not None:
            frame_c = self.out / "frames" / "seed_c.png"
            extract_frame(other, frame_c, 1.0)
            different = frame_psnr(frame_a, frame_c)
        problems = []
        if same is None or same < 35:
            problems.append(f"same seed psnr {same}")
        if different is not None and different > 40:
            problems.append(f"different seed still matches (psnr {different})")
        self.add(
            {
                "id": "seed_repeat_match",
                "kind": "compare",
                "priority": "required",
                "status": "fail" if problems else "pass",
                "detail": "; ".join(problems),
                "same_seed_psnr": same,
                "different_seed_psnr": different,
            }
        )

    def finish(self) -> int:
        for future in self.judge_futures:
            future.result()
        self.judge_pool.shutdown(wait=True)
        self.compare_seeds()
        required_fail = [
            case["id"]
            for case in self.report["cases"]
            if case.get("priority") == "required" and case.get("status") == "fail"
        ]
        required_skip = [
            case["id"]
            for case in self.report["cases"]
            if case.get("priority") == "required" and case.get("status") == "skip"
        ]
        self.report["required_fail"] = required_fail
        self.report["required_skip"] = required_skip
        self.report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save()
        print(
            f"required failures: {required_fail or 'none'}; "
            f"skipped: {required_skip or 'none'}",
            flush=True,
        )
        if required_fail:
            return 1
        if required_skip:
            return 2
        return 0


def prepare_conditions(out: Path) -> tuple[Path, Path]:
    cond = out / "conditions"
    cond.mkdir(parents=True, exist_ok=True)
    first = cond / "tiger_first.png"
    last = cond / "tiger_last.png"
    extract_frame(POSITIVE, first, 0.0)
    extract_frame(POSITIVE, last, 5.0)
    return first, last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "outputs" / "h3_coverage" / time.strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                print(f"missing {tool}", file=sys.stderr)
                return 1
        if not POSITIVE.is_file() or not NEGATIVE.is_file():
            print("missing the calibration clips", file=sys.stderr)
            return 1
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(args.out, target_is_directory=True)

    run = CoverageRun(args.out, args.minutes, args.dry_run)
    if args.dry_run:
        first = last = Path("/tmp/unused.png")
    else:
        first, last = prepare_conditions(args.out)
    jobs = generation_jobs(first, last)
    print(f"output: {args.out}", flush=True)
    print(f"budget: {args.minutes:.0f} min", flush=True)
    for job in jobs:
        est = estimate_seconds(job, run.sec_per_eval)
        width, height = canvas_size(job["short_edge"], job["aspect"])
        print(
            f"  plan {job['priority']:8} {job['id']:24} "
            f"{width}x{height} {job['duration']}s {job['steps']} steps "
            f"~{est / 60:.1f} min",
            flush=True,
        )

    run.run_contract(contract_cases(str(first)))
    if not args.dry_run:
        run.judge_file(
            "judge_known_good",
            POSITIVE,
            subject="tiger",
            prompt="A tiger walks slowly through morning fog.",
            keyframes=[],
            priority="required",
            kind="calibration",
            expect_visual_pass=True,
        )
        run.judge_file(
            "judge_known_bad",
            NEGATIVE,
            subject="tiger",
            prompt="A tiger walks slowly through morning fog.",
            keyframes=[],
            priority="required",
            kind="calibration",
            expect_visual_pass=False,
        )
        good = run.cases_by_id["judge_known_good"]["status"] == "pass"
        bad = run.cases_by_id["judge_known_bad"]["status"] == "pass"
        run.report["judge_reliable"] = bool(good and bad)
        print(f"judge reliable: {run.report['judge_reliable']}", flush=True)
        run.save()

    for job in jobs:
        est = estimate_seconds(job, run.sec_per_eval)
        if run.remaining() < est:
            run.add(
                {
                    "id": job["id"],
                    "kind": "generate",
                    "priority": job["priority"],
                    "status": "skip",
                    "detail": (
                        f"estimate {est / 60:.1f} min exceeds "
                        f"{run.remaining() / 60:.1f} min left"
                    ),
                }
            )
            continue
        if args.dry_run:
            run.add(
                {
                    "id": job["id"],
                    "kind": "generate",
                    "priority": job["priority"],
                    "status": "dry-run",
                    "estimate_min": round(est / 60, 2),
                }
            )
            continue
        record = run.generate(job)
        run.attach_judge(job, record)
    return run.finish()


if __name__ == "__main__":
    sys.exit(main())
