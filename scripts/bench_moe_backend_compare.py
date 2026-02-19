#!/usr/bin/env python3
"""
Compare FlashInfer MoE backends on small quality + throughput checks.

Example:
python3 scripts/bench_moe_backend_compare.py \
  --model nvidia/DeepSeek-R1-0528-FP4-V2 \
  --backends flashinfer_cutlass flashinfer_trtllm flashinfer_cutedsl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, popen_launch_server


def run_small_mmlu(base_url: str, model: str, num_examples: int, num_threads: int):
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=num_examples,
        num_threads=num_threads,
    )
    return run_eval(args)


def run_small_bench_serving(base_url: str, num_prompts: int, max_concurrency: int):
    with tempfile.NamedTemporaryFile(
        prefix="moe_backend_bench_", suffix=".jsonl", delete=False
    ) as tf:
        output_file = tf.name
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        "128",
        "--random-output-len",
        "64",
        "--random-range-ratio",
        "0.0",
        "--max-concurrency",
        str(max_concurrency),
        "--disable-tqdm",
        "--output-file",
        output_file,
    ]
    try:
        subprocess.run(cmd, check=True)
        with open(output_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise RuntimeError("bench_serving produced no JSON output")
        return json.loads(lines[-1])
    finally:
        try:
            os.remove(output_file)
        except OSError:
            pass


def get_backend_args(
    backend: str,
    tp_size: int,
    ep_size: int,
    mem_fraction_static: float,
    attention_backend: str,
    sampling_backend: str,
):
    args = [
        "--trust-remote-code",
        "--tp-size",
        str(tp_size),
        "--ep-size",
        str(ep_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--moe-runner-backend",
        backend,
        "--quantization",
        "modelopt_fp4",
        "--attention-backend",
        attention_backend,
        "--sampling-backend",
        sampling_backend,
        "--disable-cuda-graph",
    ]
    if backend == "flashinfer_cutedsl":
        args += ["--moe-a2a-backend", "none"]
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark small MoE quality + throughput across backends."
    )
    parser.add_argument("--model", required=True, help="Model path/name.")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:30000", help="Server URL for eval/bench."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[
            "flashinfer_cutlass",
            "flashinfer_trtllm",
            "flashinfer_cutedsl",
        ],
        help="MoE runner backends to compare.",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor-parallel size for server launch."
    )
    parser.add_argument(
        "--ep-size", type=int, default=1, help="Expert-parallel size for server launch."
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.7,
        help="Static memory fraction for server launch.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="triton",
        help="Attention backend for server launch.",
    )
    parser.add_argument(
        "--sampling-backend",
        type=str,
        default="pytorch",
        help="Sampling backend for server launch.",
    )
    parser.add_argument(
        "--launch-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        help="Server launch timeout (seconds).",
    )
    parser.add_argument(
        "--num-examples", type=int, default=32, help="MMLU examples per backend."
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="MMLU eval worker count."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=48, help="bench_serving prompt count."
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=8, help="bench_serving max concurrency."
    )
    args = parser.parse_args()

    results = {}
    for backend in args.backends:
        print(f"\n=== Running backend: {backend} ===")
        process = popen_launch_server(
            model=args.model,
            base_url=args.base_url,
            timeout=args.launch_timeout,
            other_args=get_backend_args(
                backend=backend,
                tp_size=args.tp_size,
                ep_size=args.ep_size,
                mem_fraction_static=args.mem_fraction_static,
                attention_backend=args.attention_backend,
                sampling_backend=args.sampling_backend,
            ),
            env=dict(os.environ),
        )
        try:
            eval_metrics = run_small_mmlu(
                args.base_url, args.model, args.num_examples, args.num_threads
            )
            bench_metrics = run_small_bench_serving(
                args.base_url, args.num_prompts, args.max_concurrency
            )
            results[backend] = {
                "mmlu_score": float(eval_metrics["score"]),
                "output_throughput": float(bench_metrics["output_throughput"]),
                "mean_tpot_ms": float(bench_metrics["mean_tpot_ms"]),
                "mean_ttft_ms": float(bench_metrics["mean_ttft_ms"]),
            }
        finally:
            kill_process_tree(process.pid)

    print("\n=== Summary ===")
    print(
        f"{'backend':24s} {'mmlu':>8s} {'out_thr(tok/s)':>16s} {'mean_tpot(ms)':>14s} {'mean_ttft(ms)':>14s}"
    )
    for backend, metric in results.items():
        print(
            f"{backend:24s} {metric['mmlu_score']:8.4f} {metric['output_throughput']:16.2f} "
            f"{metric['mean_tpot_ms']:14.3f} {metric['mean_ttft_ms']:14.3f}"
        )

    if "flashinfer_cutedsl" in results and "flashinfer_cutlass" in results:
        delta = abs(
            results["flashinfer_cutedsl"]["mmlu_score"]
            - results["flashinfer_cutlass"]["mmlu_score"]
        )
        print(f"\ncutedsl vs cutlass MMLU delta: {delta:.4f}")
    if "flashinfer_cutedsl" in results and "flashinfer_trtllm" in results:
        delta = abs(
            results["flashinfer_cutedsl"]["mmlu_score"]
            - results["flashinfer_trtllm"]["mmlu_score"]
        )
        print(f"cutedsl vs trtllm MMLU delta: {delta:.4f}")

    print("\nJSON results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
