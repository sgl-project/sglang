#!/usr/bin/env python3
"""SANA-WM large-scale request manifest builder and runner.

The script assumes the client and diffusion server share a filesystem, so
`input_reference` paths in the JSON payload are visible to the server process.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


DEFAULT_ACTIONS = [
    "w-16",
    "s-16",
    "a-16",
    "d-16",
    "jw-8,w-8",
    "lw-8,w-8",
    "none-16",
]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scan_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _load_prompts(prompt_file: Path | None, fallback: str) -> list[str]:
    if prompt_file and prompt_file.exists():
        prompts = [
            line.strip()
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if prompts:
            return prompts
    return [fallback]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def wait_health(args: argparse.Namespace) -> None:
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{args.base_url}/health", timeout=2)
            if resp.status_code == 200:
                print(json.dumps(resp.json(), ensure_ascii=False))
                return
        except requests.RequestException:
            pass
        time.sleep(args.interval)
    raise SystemExit(f"server did not become ready within {args.timeout}s")


def build_manifest(args: argparse.Namespace) -> None:
    images = _scan_images(Path(args.image_dir))
    if args.image_path:
        images.insert(0, Path(args.image_path))
    images = [p for p in images if p.exists()]
    if not images:
        raise SystemExit("no input images found; pass --image-path or --image-dir")

    prompts = _load_prompts(
        Path(args.prompt_file) if args.prompt_file else None,
        args.fallback_prompt,
    )
    actions = [x.strip() for x in args.actions.split(",") if x.strip()]
    if not actions:
        actions = DEFAULT_ACTIONS

    out_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for i in range(args.num_requests):
        diffusers_kwargs: dict[str, Any] = {
            "action": actions[i % len(actions)],
            "translation_speed": args.translation_speed,
            "rotation_speed_deg": args.rotation_speed_deg,
            "pitch_limit_deg": args.pitch_limit_deg,
        }
        if args.intrinsics_path:
            diffusers_kwargs["intrinsics_path"] = args.intrinsics_path
        if args.camera_to_world_path:
            diffusers_kwargs["camera_to_world_path"] = args.camera_to_world_path
        if args.skip_refiner:
            diffusers_kwargs["skip_refiner"] = True

        row = {
            "model": args.model,
            "prompt": prompts[i % len(prompts)],
            "input_reference": str(images[i % len(images)]),
            "size": f"{args.width}x{args.height}",
            "num_frames": args.num_frames,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed_start + i,
            "output_path": str(out_dir),
            "diffusers_kwargs": diffusers_kwargs,
        }
        if args.output_quality:
            row["output_quality"] = args.output_quality
        rows.append(row)

    _write_jsonl(Path(args.out), rows)
    print(json.dumps({"manifest": args.out, "requests": len(rows)}, indent=2))


def _run_one(
    idx: int,
    payload: dict[str, Any],
    base_url: str,
    submit_timeout: float,
    poll_timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/videos",
        json=payload,
        timeout=submit_timeout,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]
    deadline = time.time() + poll_timeout

    while time.time() < deadline:
        status_resp = requests.get(f"{base_url}/v1/videos/{job_id}", timeout=30)
        status_resp.raise_for_status()
        data = status_resp.json()
        status = data.get("status")
        if status == "completed":
            return {
                "idx": idx,
                "id": job_id,
                "status": status,
                "latency_s": time.perf_counter() - start,
                "file_path": data.get("file_path"),
                "file_paths": data.get("file_paths"),
                "num_outputs": data.get("num_outputs"),
                "peak_memory_mb": data.get("peak_memory_mb"),
                "inference_time_s": data.get("inference_time_s"),
            }
        if status == "failed":
            raise RuntimeError(f"job {job_id} failed: {data.get('error')}")
        time.sleep(poll_interval)

    raise TimeoutError(f"job {job_id} timed out after {poll_timeout}s")


def run_manifest(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.manifest))
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    with Path(args.results).open("w", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    i,
                    row,
                    args.base_url,
                    args.submit_timeout,
                    args.poll_timeout,
                    args.poll_interval,
                ): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    item = future.result()
                    results.append(item)
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out.flush()
                    print(json.dumps(item, ensure_ascii=False))
                except Exception as exc:
                    item = {"idx": idx, "status": "error", "error": str(exc)}
                    errors.append(item)
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out.flush()
                    print(json.dumps(item, ensure_ascii=False))

    latencies = [x["latency_s"] for x in results]
    wall_s = time.perf_counter() - wall_start
    summary = {
        "manifest": str(args.manifest),
        "results": str(args.results),
        "total_requests": len(rows),
        "completed_requests": len(results),
        "failed_requests": len(errors),
        "wall_time_s": wall_s,
        "throughput_req_s": len(results) / wall_s if wall_s > 0 else 0.0,
        "latency_s_min": min(latencies) if latencies else None,
        "latency_s_mean": statistics.mean(latencies) if latencies else None,
        "latency_s_p50": _percentile(latencies, 0.50),
        "latency_s_p90": _percentile(latencies, 0.90),
        "latency_s_p95": _percentile(latencies, 0.95),
        "latency_s_p99": _percentile(latencies, 0.99),
        "latency_s_max": max(latencies) if latencies else None,
        "errors": errors[:20],
    }
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors and args.fail_on_error:
        raise SystemExit(1)


def _paths_from_results(results_path: Path) -> list[Path]:
    paths: list[Path] = []
    for row in _read_jsonl(results_path):
        file_paths = row.get("file_paths")
        if isinstance(file_paths, list):
            paths.extend(Path(p) for p in file_paths if p)
        elif row.get("file_path"):
            paths.append(Path(row["file_path"]))
    return paths


def validate_videos(args: argparse.Namespace) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required for validation") from exc

    paths = []
    if args.results:
        paths.extend(_paths_from_results(Path(args.results)))
    if args.output_dir:
        paths.extend(sorted(Path(args.output_dir).glob("*.mp4")))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no mp4 files found to validate")

    bad = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        reasons = []
        if not path.exists():
            reasons.append("missing")
        if frames <= 0 or width <= 0 or height <= 0:
            reasons.append(f"invalid metadata {width}x{height} frames={frames}")
        if args.expected_width and width != args.expected_width:
            reasons.append(f"width {width} != {args.expected_width}")
        if args.expected_height and height != args.expected_height:
            reasons.append(f"height {height} != {args.expected_height}")
        if args.expected_frames and frames != args.expected_frames:
            reasons.append(f"frames {frames} != {args.expected_frames}")
        if args.expected_fps and abs(fps - args.expected_fps) > args.fps_tolerance:
            reasons.append(f"fps {fps:.3f} != {args.expected_fps}")
        if reasons:
            bad.append({"path": str(path), "reasons": reasons})

    summary = {"videos": len(paths), "bad": bad[:20]}
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if bad:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    health = sub.add_parser("wait-health")
    health.add_argument("--base-url", default="http://127.0.0.1:30000")
    health.add_argument("--timeout", type=float, default=1200)
    health.add_argument("--interval", type=float, default=2)
    health.set_defaults(func=wait_health)

    manifest = sub.add_parser("build-manifest")
    manifest.add_argument("--out", required=True)
    manifest.add_argument(
        "--model", default="Efficient-Large-Model/SANA-WM_bidirectional"
    )
    manifest.add_argument("--num-requests", type=int, default=64)
    manifest.add_argument("--image-dir", default="")
    manifest.add_argument("--image-path", default="")
    manifest.add_argument("--prompt-file", default="")
    manifest.add_argument(
        "--fallback-prompt",
        default="A cinematic camera move through a clean indoor scene.",
    )
    manifest.add_argument("--output-dir", required=True)
    manifest.add_argument("--width", type=int, default=640)
    manifest.add_argument("--height", type=int, default=384)
    manifest.add_argument("--num-frames", type=int, default=17)
    manifest.add_argument("--fps", type=int, default=16)
    manifest.add_argument("--num-inference-steps", type=int, default=12)
    manifest.add_argument("--guidance-scale", type=float, default=4.5)
    manifest.add_argument("--negative-prompt", default="")
    manifest.add_argument("--seed-start", type=int, default=0)
    manifest.add_argument("--actions", default=",".join(DEFAULT_ACTIONS))
    manifest.add_argument("--translation-speed", type=float, default=0.05)
    manifest.add_argument("--rotation-speed-deg", type=float, default=1.2)
    manifest.add_argument("--pitch-limit-deg", type=float, default=85.0)
    manifest.add_argument("--intrinsics-path", default="")
    manifest.add_argument("--camera-to-world-path", default="")
    manifest.add_argument("--skip-refiner", action="store_true")
    manifest.add_argument("--output-quality", default="default")
    manifest.set_defaults(func=build_manifest)

    run = sub.add_parser("run-manifest")
    run.add_argument("--manifest", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:30000")
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--submit-timeout", type=float, default=60)
    run.add_argument("--poll-timeout", type=float, default=1800)
    run.add_argument("--poll-interval", type=float, default=2)
    run.add_argument("--results", required=True)
    run.add_argument("--summary", required=True)
    run.add_argument("--fail-on-error", action="store_true")
    run.set_defaults(func=run_manifest)

    validate = sub.add_parser("validate-videos")
    validate.add_argument("--results", default="")
    validate.add_argument("--output-dir", default="")
    validate.add_argument("--expected-width", type=int, default=0)
    validate.add_argument("--expected-height", type=int, default=0)
    validate.add_argument("--expected-frames", type=int, default=0)
    validate.add_argument("--expected-fps", type=float, default=0.0)
    validate.add_argument("--fps-tolerance", type=float, default=0.5)
    validate.add_argument("--summary", default="")
    validate.set_defaults(func=validate_videos)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
