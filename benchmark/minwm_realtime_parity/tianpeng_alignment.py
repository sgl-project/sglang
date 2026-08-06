#!/usr/bin/env python3
"""Replay Tianpeng's director sample through the SGLang MinWM realtime API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from common import is_realtime_trace_event, save_video, sha256_file, write_json

DEFAULT_ALIGNMENT_URL = (
    "https://leap-world-us-east-2.s3.us-east-2.amazonaws.com/world-model/sft/"
    "prompt_compare/detailmix_director_gap12_20260729_094145/"
    "inference-alignment/"
)
EXPECTED_SAMPLE_ID = "detailmix_director_gap12_seed729001"
EXPECTED_VIDEO_SHA256 = (
    "0295dc25077c550a76ad8cd57a44e4189037443e693127e621bdbbab9d69865c"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "18a48a2709d74b93ce26f0b808f381d191553853aae81dd72d2438430251d379"
)
EXPECTED_MINWM_COMMIT = "4220c8a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-url", default=DEFAULT_ALIGNMENT_URL)
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--ws-url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--model", default="minwm")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate/download inputs and write request.json without calling SGLang.",
    )
    parser.add_argument(
        "--zero-actions",
        action="store_true",
        help=(
            "Replace the published 1088x8 keyboard timeline with all-zero rows "
            "while preserving prompts, per-block seeds, and every other input."
        ),
    )
    return parser.parse_args()


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "sglang-minwm-tianpeng-alignment/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _fetch_json(base_url: str, name: str) -> dict:
    return json.loads(_fetch_bytes(urllib.parse.urljoin(base_url, name)))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _video_boundary_to_latent(boundary: int, temporal_factor: int) -> int:
    if boundary == 0:
        return 0
    if (boundary - 1) % temporal_factor:
        raise ValueError(
            f"prompt boundary {boundary} is not aligned to VAE factor {temporal_factor}"
        )
    return 1 + (boundary - 1) // temporal_factor


def _latent_boundary_to_chunk(boundary: int, block_lengths: list[int]) -> int:
    cursor = 0
    for chunk_index, block_length in enumerate(block_lengths):
        if cursor == boundary:
            return chunk_index
        cursor += block_length
    if cursor == boundary:
        return len(block_lengths)
    raise ValueError(f"latent boundary {boundary} is inside a director block")


def load_contract(base_url: str) -> dict:
    gap_line = _fetch_bytes(urllib.parse.urljoin(base_url, "gap12.jsonl"))
    records = [json.loads(line) for line in gap_line.splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError(f"expected one gap12 sample, got {len(records)}")
    sample = records[0]
    input_manifest = _fetch_json(base_url, "input_manifest.json")
    run_manifest = _fetch_json(base_url, "run_manifest.json")
    if sample["sample_id"] != EXPECTED_SAMPLE_ID:
        raise ValueError(f"unexpected sample id {sample['sample_id']!r}")
    checkpoint = input_manifest["checkpoint"]
    if checkpoint["source_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("alignment checkpoint SHA-256 changed")

    target = next(
        message
        for message in sample["messages"]
        if message.get("role") == "target" and message.get("type") == "video"
    )
    controls = {control["type"]: control for control in target["controls"]}
    actions = controls["keyboard_direction_frame_interval"]["actions"]
    segments = controls["text_prompt_interval"]["segments"]
    output = target["output"]
    inference = input_manifest["inference"]
    reproducibility = sample["reproducibility"]
    block_lengths = [int(value) for value in reproducibility["latent_block_lengths"]]
    seeds = [int(value) for value in reproducibility["seed_schedule"]]
    temporal_factor = 4

    expected = {
        "frames": 1089,
        "height": 480,
        "width": 832,
        "local_attn_size": 32,
        "sink_size": 8,
        "rope_position_mode": "block_relative",
        "rope_max_frame_gap": 12,
        "prompt_first_frame_pin_enabled": True,
    }
    for key, value in expected.items():
        actual = output.get(key, inference.get(key))
        if actual != value:
            raise ValueError(f"alignment {key} changed: {actual!r} != {value!r}")
    if block_lengths != [1] + [4] * 68:
        raise ValueError("unexpected director latent block schedule")
    if seeds != list(range(729001, 729070)):
        raise ValueError("unexpected director seed schedule")
    if len(actions) != output["frames"] - 1:
        raise ValueError("director action timeline must exclude only video frame zero")
    if any(
        not isinstance(row, list)
        or len(row) != 8
        or any(value not in (0, 1) for value in row)
        for row in actions
    ):
        raise ValueError("director action timeline must contain binary 8-key rows")
    if (
        segments[0]["start"] != 0
        or segments[-1]["end"] != output["frames"]
        or any(
            left["end"] != right["start"] for left, right in zip(segments, segments[1:])
        )
    ):
        raise ValueError("director prompt segments are not contiguous")

    prompt_schedule = []
    for segment in segments[1:]:
        latent_boundary = _video_boundary_to_latent(
            int(segment["start"]), temporal_factor
        )
        prompt_schedule.append(
            {
                "target_chunk": _latent_boundary_to_chunk(
                    latent_boundary, block_lengths
                ),
                "prompt": segment["text"],
                "kind": "prompt",
            }
        )
    if [item["target_chunk"] for item in prompt_schedule] != [13, 42, 50]:
        raise ValueError("director prompt switches no longer map to chunks 13/42/50")

    reference_url = urllib.parse.urljoin(
        base_url,
        f"../strategy-ab/{sample['sample_id']}.mp4",
    )
    return {
        "sample": sample,
        "input_manifest": input_manifest,
        "run_manifest": run_manifest,
        "reference_url": reference_url,
        "request": {
            "type": "init",
            "generation_mode": "t2v",
            "model": "minwm",
            "prompt": segments[0]["text"],
            "size": f"{output['width']}x{output['height']}",
            "fps": int(target["metadata"]["fps"]),
            "seed": seeds[0],
            "generator_device": "cuda",
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "max_chunks": len(block_lengths),
            "num_frames": int(output["frames"]),
            "realtime_output_format": "raw",
            "condition_inputs": {
                # T2V latent frame zero is an implicit noop. These 1088 rows
                # map to latent frames 1..272 in groups of four.
                "action_weights": actions,
                "minwm_chunk_seeds": seeds,
                "minwm_prompt_schedule": prompt_schedule,
            },
        },
        "expected": {
            **expected,
            "fps": int(target["metadata"]["fps"]),
            "chunks": len(block_lengths),
            "latent_frames": sum(block_lengths),
            "action_rows": len(actions),
            "prompt_switch_chunks": [13, 42, 50],
            "checkpoint_uri": checkpoint["source_uri"],
            "checkpoint_version_id": checkpoint["source_version_id"],
            "checkpoint_sha256": checkpoint["source_sha256"],
            "reference_video_sha256": EXPECTED_VIDEO_SHA256,
            "minwm_merge_commit": EXPECTED_MINWM_COMMIT,
            "alignment_code_ref": sample["metadata"]["code_git_ref"],
        },
    }


async def run_request(
    *,
    ws_url: str,
    request: dict,
    expected_chunks: int,
    timeout: float,
) -> tuple[np.ndarray, list[dict], dict]:
    import msgspec.msgpack
    import websockets
    from run_sglang_api import decode_frames

    frames = []
    stats = []
    completed = set()
    previous_frame = None
    started = time.perf_counter()
    async with websockets.connect(
        ws_url,
        max_size=None,
        ping_interval=None,
        open_timeout=timeout,
    ) as websocket:
        await websocket.send(msgspec.msgpack.encode(request))
        while len(completed) < expected_chunks or len(stats) < expected_chunks:
            packed = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            header = msgspec.msgpack.decode(packed)
            message_type = header.get("type")
            if is_realtime_trace_event(header):
                continue
            if message_type == "error":
                raise RuntimeError(header.get("content", "unknown realtime error"))
            if message_type == "chunk_stats":
                stats.append(header)
                continue
            if message_type == "frame_batch":
                frame_payload = header.pop("payload")
            elif message_type == "frame_batch_header":
                frame_payload = await asyncio.wait_for(
                    websocket.recv(), timeout=timeout
                )
            else:
                raise ValueError(f"unexpected realtime message: {header}")
            chunk_frames = decode_frames(header, frame_payload, previous_frame)
            frames.extend(chunk_frames)
            if chunk_frames:
                previous_frame = chunk_frames[-1].tobytes()
            if header.get("is_final_frame_batch", True):
                completed.add(int(header["chunk_index"]))
    return np.stack(frames), stats, {"wall_seconds": time.perf_counter() - started}


def _run_ffmpeg_metric(metric: str, baseline: Path, candidate: Path) -> dict:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(baseline),
        "-i",
        str(candidate),
        "-lavfi",
        metric,
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    line = next(
        line for line in reversed(completed.stderr.splitlines()) if metric in line
    )
    values = {
        key: float(value)
        for key, value in re.findall(
            r"([A-Za-z_]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+|inf))", line
        )
        if value != "inf"
    }
    return {"summary": line.strip(), "values": values}


def _write_player(path: Path, sample_id: str, *, zero_actions: bool = False) -> None:
    candidate_label = (
        "SGLang MinWM（全程不按键）" if zero_actions else "SGLang MinWM（原始 action）"
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>MinWM Tianpeng alignment</title>
<style>
body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
button{{font-size:16px;padding:10px 18px;margin:0 8px 18px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
video{{width:100%;background:#000}} h2{{font-size:18px}}
</style>
<h1>{sample_id}</h1>
<button onclick="playBoth()">同步播放</button>
<button onclick="pauseBoth()">暂停</button>
<button onclick="resetBoth()">归零</button>
<div class="grid">
<section><h2>天鹏 baseline</h2><video id="baseline" controls muted loop src="baseline.mp4"></video></section>
<section><h2>{candidate_label}</h2><video id="candidate" controls muted loop src="sglang.mp4"></video></section>
</div>
<script>
const videos=[document.getElementById("baseline"),document.getElementById("candidate")];
function playBoth(){{const t=Math.min(...videos.map(v=>v.currentTime));videos.forEach(v=>v.currentTime=t);void Promise.all(videos.map(v=>v.play()));}}
function pauseBoth(){{videos.forEach(v=>v.pause());}}
function resetBoth(){{pauseBoth();videos.forEach(v=>v.currentTime=0);}}
</script>
</html>"""
    path.write_text(html, encoding="utf-8")


async def async_main(args: argparse.Namespace) -> None:
    results = Path(args.results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    contract = load_contract(args.alignment_url)
    request = contract["request"]
    request["model"] = args.model
    if args.zero_actions:
        action_weights = request["condition_inputs"]["action_weights"]
        request["condition_inputs"]["action_weights"] = [
            [0] * len(row) for row in action_weights
        ]
    action_mode = "all_zero" if args.zero_actions else "published"
    write_json(
        results / "alignment_contract.json",
        {**contract["expected"], "action_mode": action_mode},
    )
    write_json(results / "request.json", request)

    baseline = results / "baseline.mp4"
    if not baseline.exists():
        baseline.write_bytes(_fetch_bytes(contract["reference_url"]))
    if sha256_file(baseline) != EXPECTED_VIDEO_SHA256:
        raise ValueError("downloaded Tianpeng reference video SHA-256 changed")
    if args.prepare_only:
        print(json.dumps({"prepared": str(results)}, ensure_ascii=False))
        return

    frames, stats, timing = await run_request(
        ws_url=args.ws_url,
        request=request,
        expected_chunks=contract["expected"]["chunks"],
        timeout=args.timeout,
    )
    expected_shape = (
        contract["expected"]["frames"],
        contract["expected"]["height"],
        contract["expected"]["width"],
        3,
    )
    if frames.shape != expected_shape:
        raise ValueError(f"SGLang output shape {frames.shape} != {expected_shape}")
    np.save(results / "sglang.npy", frames, allow_pickle=False)
    save_video(results / "sglang.mp4", frames, contract["expected"]["fps"])
    comparison = {
        "sample_id": EXPECTED_SAMPLE_ID,
        "baseline_sha256": sha256_file(baseline),
        "sglang_video_sha256": sha256_file(results / "sglang.mp4"),
        "sglang_frames_sha256": sha256_file(results / "sglang.npy"),
        "psnr": _run_ffmpeg_metric("psnr", baseline, results / "sglang.mp4"),
        "ssim": _run_ffmpeg_metric("ssim", baseline, results / "sglang.mp4"),
        "timing": timing,
        "chunk_stats": stats,
        "action_mode": action_mode,
        "bitwise_claim": (
            "The supplied baseline is an encoded MP4 without latent/cache dumps; "
            "this run can establish decoded-video numerical parity, not latent "
            "bitwise equality."
        ),
    }
    write_json(results / "comparison.json", comparison)
    _write_player(
        results / "index.html",
        EXPECTED_SAMPLE_ID,
        zero_actions=args.zero_actions,
    )
    print(
        json.dumps(
            {
                "results": str(results),
                "player": str(results / "index.html"),
                "frames": int(frames.shape[0]),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
