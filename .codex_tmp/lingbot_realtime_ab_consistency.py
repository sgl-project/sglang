#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

from sglang.multimodal_gen.test.server.realtime_consistency import (
    build_realtime_init_payload,
    collect_realtime_output,
    prepare_realtime_first_frame,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    LINGBOT_WORLD_REALTIME_sampling_params,
)


MODEL_PATH = "robbyant/lingbot-world-fast-diffusers"


def wait_health(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def frame_digest(frames: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(frames.shape).encode())
    h.update(frames.tobytes())
    return h.hexdigest()


def summarize_stats(stats) -> list[dict[str, float | int | str | None]]:
    return [
        {
            "chunk_index": s.chunk_index,
            "content_type": s.content_type,
            "num_frames": s.num_frames,
            "raw_bytes": s.raw_bytes,
            "ws_payload_bytes": s.ws_payload_bytes,
            "request_prepare_ms": s.request_prepare_ms,
            "scheduler_forward_ms": s.scheduler_forward_ms,
            "raw_payload_build_ms": s.raw_payload_build_ms,
            "raw_write_ms": s.raw_write_ms,
            "ws_write_ms": s.ws_write_ms,
            "chunk_total_ms": s.chunk_total_ms,
        }
        for s in stats
    ]


def run_once(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.run_name}_server.log"
    frames_path = out_dir / f"{args.run_name}_frames.npz"
    summary_path = out_dir / f"{args.run_name}_summary.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("HF_HOME", "/scratch/hf_cache")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("SGLANG_TEST_REALTIME_WS_TIMEOUT_SECS", "1800")

    sglang_cli = Path(sys.executable).with_name("sglang")
    cmd = [
        str(sglang_cli),
        "serve",
        "--model-path",
        MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model-type",
        "diffusion",
        "--num-gpus",
        "4",
        "--ulysses-degree",
        "4",
        "--text-encoder-cpu-offload",
        "--strict-ports",
        "--master-port",
        str(args.master_port),
        "--scheduler-port",
        str(args.scheduler_port),
        "--pipeline-class-name",
        "LingBotWorldCausalDMDPipeline",
    ]

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            while True:
                try:
                    wait_health(args.port, 2)
                    break
                except Exception:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"server exited with {proc.returncode}; see {log_path}"
                        )

            params = LINGBOT_WORLD_REALTIME_sampling_params
            first_frame = prepare_realtime_first_frame(params.image_path)
            init_payload = build_realtime_init_payload(
                model_path=MODEL_PATH,
                sampling_params=params,
                output_size=params.output_size,
                first_frame=first_frame,
            )
            if args.output_format:
                init_payload["realtime_output_format"] = args.output_format
            result = asyncio.run(
                collect_realtime_output(
                    ws_url=f"ws://127.0.0.1:{args.port}/v1/realtime_video/generate",
                    init_payload=init_payload,
                    events=list(params.realtime_events),
                    num_chunks=int(params.realtime_num_chunks or 4),
                    require_chunk_stats=True,
                )
            )
            frames = np.stack(result.frames, axis=0)
            np.savez_compressed(frames_path, frames=frames)
            summary = {
                "run_name": args.run_name,
                "frames_shape": list(frames.shape),
                "frames_sha256": frame_digest(frames),
                "chunk_stats": summarize_stats(result.chunk_stats),
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, sort_keys=True))
            return 0
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)


def compare(args: argparse.Namespace) -> int:
    baseline = np.load(args.baseline)["frames"]
    patch = np.load(args.patch)["frames"]
    result: dict[str, object] = {
        "baseline": args.baseline,
        "patch": args.patch,
        "baseline_shape": list(baseline.shape),
        "patch_shape": list(patch.shape),
        "same_shape": baseline.shape == patch.shape,
    }
    if baseline.shape == patch.shape:
        diff = patch.astype(np.int16) - baseline.astype(np.int16)
        abs_diff = np.abs(diff)
        result.update(
            {
                "exact_equal": bool(np.array_equal(baseline, patch)),
                "max_abs_diff": int(abs_diff.max(initial=0)),
                "mean_abs_diff": float(abs_diff.mean()),
                "nonzero_elements": int(np.count_nonzero(abs_diff)),
                "num_elements": int(abs_diff.size),
                "baseline_sha256": frame_digest(baseline),
                "patch_sha256": frame_digest(patch),
            }
        )
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(output)
    print(output, end="")
    return 0 if result.get("same_shape") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-name", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--gpus", default="4,5,6,7")
    run_parser.add_argument("--port", type=int, default=25450)
    run_parser.add_argument("--master-port", type=int, default=25451)
    run_parser.add_argument("--scheduler-port", type=int, default=25452)
    run_parser.add_argument("--output-format")
    run_parser.set_defaults(func=run_once)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--patch", required=True)
    compare_parser.add_argument("--out")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
