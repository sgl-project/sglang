"""Offline tuner for --dit-tp-plan: measure candidate plans end-to-end.

Isolated communication/GEMM microbenchmarks cannot rank TP layouts (measured
both over- and under-statements vs the served model — see
layers/tp_shard_planner.py), so this tool measures the only trustworthy
signal: real server end-to-end latency, with candidates ABAB-interleaved and
compared through paired differences. Run it once per (model, machine, TP
degree, workload); point --dit-tp-plan at the emitted JSON afterwards.

Example (compare the built-in plans on this machine):

    python -m sglang.multimodal_gen.tools.tune_dit_tp_plan \\
        --model-path Qwen/Qwen-Image-2512 \\
        --serve-args "--num-gpus 2 --tp-size 2 --performance-mode speed \\
                      --enable-torch-compile" \\
        --workload 1024x1024 --num-inference-steps 50 \\
        --candidate full --candidate auto \\
        --output qwen_image_tp2_h100.json

Each ABAB round boots every candidate server once (including torch.compile
warmup), so a 2-candidate x 3-pair run costs ~6 server startups. Candidates
may be plan modes ("auto", "full", "aggressive") or plan JSON paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

HEALTH_TIMEOUT_S = 2400
REQUEST_TIMEOUT_S = 900


def _launch_server(args, candidate: str, port: int) -> subprocess.Popen:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        args.model_path,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--dit-tp-plan",
        candidate,
    ]
    if args.workload:
        cmd += ["--dit-tp-plan-workload", args.workload]
    cmd += shlex.split(args.serve_args)
    print(f"  launching: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=args.server_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _wait_health(proc: subprocess.Popen, port: int) -> None:
    deadline = time.time() + HEALTH_TIMEOUT_S
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (rc={proc.returncode})")
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"server on port {port} not healthy in {HEALTH_TIMEOUT_S}s")


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()
    time.sleep(4)


def _timed_requests(args, port: int) -> float:
    width, height = (int(v) for v in args.workload.lower().split("x")[:2])
    payload = {
        "model": args.model_path,
        "prompt": args.prompt,
        "size": f"{width}x{height}",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": 42,
    }
    url = f"http://127.0.0.1:{port}/v1/images/generations"
    latencies = []
    for i in range(args.warmup_requests + args.timed_requests):
        start = time.time()
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        latency = time.time() - start
        if i >= args.warmup_requests:
            latencies.append(latency)
    return statistics.median(latencies)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--serve-args",
        required=True,
        help="Extra `sglang serve` args shared by every candidate "
        "(GPUs, TP size, compile flags, ...).",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Plan mode (auto/full/aggressive) or plan JSON path. Repeat per "
        "candidate; the first one is the baseline.",
    )
    parser.add_argument("--workload", required=True, help="WxH, e.g. 1024x1024")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="A futuristic cyberpunk city at night, neon lights on wet streets",
    )
    parser.add_argument("--pairs", type=int, default=3, help="ABAB rounds")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--timed-requests", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=31800)
    parser.add_argument("--output", required=True, help="Winning plan JSON path")
    parser.add_argument(
        "--server-log",
        type=argparse.FileType("w"),
        default=subprocess.DEVNULL,
        help="File for server stdout/stderr (default: discarded)",
    )
    args = parser.parse_args()

    if len(args.candidate) < 2:
        parser.error("need at least two --candidate values to compare")

    results: dict[str, list[float]] = {c: [] for c in args.candidate}
    port = args.base_port
    for round_idx in range(1, args.pairs + 1):
        for candidate in args.candidate:
            print(f"[round {round_idx}] candidate={candidate}")
            proc = _launch_server(args, candidate, port)
            try:
                _wait_health(proc, port)
                latency = _timed_requests(args, port)
            finally:
                _kill(proc)
            results[candidate].append(latency)
            print(f"[round {round_idx}] candidate={candidate} e2e={latency:.3f}s")
            port += 1

    baseline = args.candidate[0]
    summary = {}
    print("\n=== paired summary ===")
    for candidate in args.candidate:
        vals = results[candidate]
        summary[candidate] = {"runs": vals, "mean": statistics.mean(vals)}
        line = f"{candidate:24s} mean={statistics.mean(vals):.3f} runs={vals}"
        if candidate != baseline:
            diffs = [b - c for b, c in zip(results[baseline], vals)]
            summary[candidate]["paired_diff_vs_baseline"] = diffs
            same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
            line += f" paired_diff={['%+.3f' % d for d in diffs]}"
            if not same_sign:
                line += " (mixed sign: treat as noise)"
        print(line)

    winner = min(args.candidate, key=lambda c: statistics.mean(results[c]))
    print(f"\nwinner: {winner}")

    if winner in ("auto", "full", "aggressive"):
        plan: dict = {"mode": winner}
    else:
        with open(winner) as f:
            plan = json.load(f)
    plan["provenance"] = {
        "tool": "tune_dit_tp_plan",
        "model": args.model_path,
        "serve_args": args.serve_args,
        "workload": args.workload,
        "num_inference_steps": args.num_inference_steps,
        "pairs": args.pairs,
        "results": summary,
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.workload and "workload" not in plan:
        plan["workload"] = args.workload
    with open(args.output, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
