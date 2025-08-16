#!/usr/bin/env python3
"""
Simple overlap benchmarking script 

What it does for each model:
- Launch baseline (overlap OFF) and overlap (ON) servers sequentially
- Small memory footprint: --disable-cuda-graph and modest --max-total-tokens
- Waits for /get_model_info to be ready
- Runs sglang.bench_serving with small random prompts
- Writes CSV rows (one per baseline/overlap) to the chosen output 

Usage example:
  python benchmark/overlap/run_overlap_bench_csv.py \
    --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tp 4 --base-port 30000 --csv result.json

Tips:
- For gated models, login first or set HUGGING_FACE_HUB_TOKEN
- To keep cache small: export HF_HOME=/users/5/sharm843/hf-cache TRANSFORMERS_CACHE=/users/5/sharm843/hf-cache
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests


def has_cpp_extension() -> bool:
    try:
        import torch  # type: ignore

        return hasattr(torch.utils, "cpp_extension")
    except Exception:
        return False


def start_server(
    model: str,
    tp: int,
    port: int,
    overlap: bool,
    max_total_tokens: int,
    trust_remote_code: bool,
    extra_launch_args: List[str],
) -> Tuple[subprocess.Popen, str]:
    env = os.environ.copy()
    env["SGLANG_ENABLE_TP_ALLREDUCE_OVERLAP"] = "true" if overlap else "false"
    args = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--device",
        "cuda",
        "--tp-size",
        str(tp),
        "--disable-cuda-graph",
        "--max-total-tokens",
        str(max_total_tokens),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *extra_launch_args,
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    if has_cpp_extension():
        args.append("--enable-symm-mem")
    log_path = f"/tmp/run_overlap_bench_{'ovl' if overlap else 'base'}_{port}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return proc, log_path


def stop_server(p: subprocess.Popen) -> None:
    try:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=10)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def wait_ready(port: int, timeout_s: int = 240) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_bench(port: int, num_prompts: int, in_len: int, out_len: int) -> Dict[str, float]:
    tmp_out = f"/tmp/bench_{port}.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(in_len),
        "--random-output-len",
        str(out_len),
        "--random-range-ratio",
        "0.5",
        "--request-rate",
        "inf",
        "--disable-tqdm",
        "--seed",
        "1",
        "--output-file",
        tmp_out,
    ]
    subprocess.run(cmd, check=True)
    last_line = None
    with open(tmp_out, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                last_line = s
    try:
        os.remove(tmp_out)
    except Exception:
        pass
    import json

    metrics = json.loads(last_line or "{}")
    return {
        "duration": float(metrics.get("duration", 0.0)),
        "request_throughput": float(metrics.get("request_throughput", 0.0)),
        "total_token_throughput": float(metrics.get("total_token_throughput", 0.0)),
        "mean_e2e_ms": float(metrics.get("mean_e2e_latency_ms", 0.0)),
        "median_e2e_ms": float(metrics.get("median_e2e_latency_ms", 0.0)),
        "mean_ttft_ms": float(metrics.get("mean_ttft_ms", 0.0)),
        "median_ttft_ms": float(metrics.get("median_ttft_ms", 0.0)),
        "mean_itl_ms": float(metrics.get("mean_itl_ms", 0.0)),
        "p95_itl_ms": float(metrics.get("p95_itl_ms", 0.0)),
        "p99_itl_ms": float(metrics.get("p99_itl_ms", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]) 
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=30000)
    ap.add_argument("--max-total-tokens", type=int, default=16384)
    ap.add_argument("--num-prompts", type=int, default=60)
    ap.add_argument("--random-input-len", type=int, default=1024)
    ap.add_argument("--random-output-len", type=int, default=64)
    ap.add_argument("--csv", type=str, default="/dev/stdout")
    ap.add_argument(
        "--trust-remote-code-for",
        nargs="*",
        default=["deepseek", "DeepSeek"],
        help="Substring(s) of model names that require --trust-remote-code",
    )
    ap.add_argument(
        "--extra-launch-args",
        nargs="*",
        default=[],
        help="Extra args to pass to sglang.launch_server (e.g., --disable-custom-all-reduce)",
    )
    args = ap.parse_args()

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "tp",
        "overlap",
        "ok",
        "err",
        "duration_s",
        "request_tps",
        "token_tps",
        "mean_e2e_ms",
        "median_e2e_ms",
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
    ]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, model in enumerate(args.models):
            port = args.base_port + i * 10
            trust_rc = any(s in model for s in args.trust_remote_code_for)
            for overlap in (False, True):
                row = {fn: "" for fn in fieldnames}
                row.update({"model": model, "tp": args.tp, "overlap": int(overlap)})
                try:
                    proc, log = start_server(
                        model,
                        args.tp,
                        port if not overlap else port + 1,
                        overlap,
                        args.max_total_tokens,
                        trust_remote_code=trust_rc,
                        extra_launch_args=args.extra_launch_args,
                    )
                    try:
                        if not wait_ready(port if not overlap else port + 1):
                            raise RuntimeError(f"server_not_ready (log {log})")
                        metrics = run_bench(
                            port if not overlap else port + 1,
                            args.num_prompts,
                            args.random_input_len,
                            args.random_output_len,
                        )
                        row.update(
                            {
                                "ok": 1,
                                "duration_s": metrics["duration"],
                                "request_tps": metrics["request_throughput"],
                                "token_tps": metrics["total_token_throughput"],
                                "mean_e2e_ms": metrics["mean_e2e_ms"],
                                "median_e2e_ms": metrics["median_e2e_ms"],
                                "mean_ttft_ms": metrics["mean_ttft_ms"],
                                "median_ttft_ms": metrics["median_ttft_ms"],
                                "mean_itl_ms": metrics["mean_itl_ms"],
                                "p95_itl_ms": metrics["p95_itl_ms"],
                                "p99_itl_ms": metrics["p99_itl_ms"],
                            }
                        )
                    finally:
                        stop_server(proc)
                except Exception as e:
                    row.update({"ok": 0, "err": str(e)})
                w.writerow(row)
                f.flush()

    if args.csv == "/dev/stdout":
        pass
    else:
        print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()


