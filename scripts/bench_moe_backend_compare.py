#!/usr/bin/env python3
"""
MoE backend concurrency sweep benchmark.

Compares flashinfer_trtllm, flashinfer_cutlass, and flashinfer_cutedsl across
a range of concurrency levels and EP configurations.  For each (backend, ep)
pair the script launches a server, runs a quick MMLU sanity check, then sweeps
concurrency values with bench_serving and records throughput / latency metrics.

Example (8xGPU, DeepSeek-R1 NVFP4):

    python3 scripts/bench_moe_backend_compare.py \\
        --model nvidia/DeepSeek-R1-0528-NVFP4-v2 \\
        --backends flashinfer_trtllm flashinfer_cutlass flashinfer_cutedsl \\
        --ep-sizes 1 8 \\
        --tp-size 8 \\
        --concurrency 1 2 4 8 16 32 64 128 512 1024 2048 \\
        --output-dir bench_moe_sweep_results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
DEFAULT_BACKENDS = ["flashinfer_trtllm", "flashinfer_cutlass", "flashinfer_cutedsl"]
DEFAULT_EP_SIZES = [1, 8]
DEFAULT_TP_SIZE = 8
DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048]

BENCH_METRIC_KEYS = [
    "output_throughput",
    "request_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "mean_e2e_latency_ms",
    "completed",
    "total_output_tokens",
]

CSV_FIELDS = [
    "backend",
    "ep_size",
    "tp_size",
    "concurrency",
    "num_prompts",
    *BENCH_METRIC_KEYS,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num_prompts_for_concurrency(concurrency: int) -> int:
    """Scale down num-prompts for low concurrency to keep wall-clock reasonable."""
    if concurrency <= 4:
        return 128
    if concurrency <= 32:
        return 512
    return 2048


def _build_server_args(
    backend: str,
    tp_size: int,
    ep_size: int,
    extra_server_args: list[str],
) -> list[str]:
    args = [
        "--trust-remote-code",
        "--tp-size",
        str(tp_size),
        "--ep-size",
        str(ep_size),
        "--attention-backend",
        "trtllm_mla",
        "--quantization",
        "modelopt_fp4",
        "--moe-runner-backend",
        backend,
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
    ]
    args.extend(extra_server_args)
    return args


def run_mmlu_sanity(
    base_url: str,
    model: str,
    num_examples: int = 32,
    num_threads: int = 4,
) -> float:
    """Run a small MMLU eval and return the score."""
    from sglang.test.run_eval import run_eval_once
    from sglang.test.simple_eval_mmlu import MMLUEval

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
    eval_obj = MMLUEval(filename, num_examples, num_threads)
    args = SimpleNamespace(base_url=base_url, model=model)
    base_url_v1 = f"{base_url}/v1"
    result, _latency, _sampler = run_eval_once(args, base_url_v1, eval_obj)
    return float(result.score)


def run_bench_serving(
    base_url: str,
    num_prompts: int,
    max_concurrency: int,
    random_input_len: int = 1024,
    random_output_len: int = 1024,
    random_range_ratio: float = 1.0,
) -> dict[str, Any]:
    """Run bench_serving and return the parsed JSON result."""
    with tempfile.NamedTemporaryFile(
        prefix="moe_sweep_", suffix=".jsonl", delete=False
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
        str(random_input_len),
        "--random-output-len",
        str(random_output_len),
        "--random-range-ratio",
        str(random_range_ratio),
        "--max-concurrency",
        str(max_concurrency),
        "--disable-tqdm",
        "--output-file",
        output_file,
    ]
    try:
        subprocess.run(cmd, check=True)
        with open(output_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise RuntimeError("bench_serving produced no JSON output")
        return json.loads(lines[-1])
    finally:
        try:
            os.remove(output_file)
        except OSError:
            pass


def _print_header(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE backend concurrency sweep benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path/name.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="Server URL.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        help="MoE runner backends to compare.",
    )
    parser.add_argument(
        "--ep-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_EP_SIZES,
        help="Expert-parallel sizes to sweep.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=DEFAULT_TP_SIZE,
        help="Tensor-parallel size (constant across all runs).",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max-concurrency values to sweep.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Input token length for bench_serving.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=1024,
        help="Output token length for bench_serving.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Random range ratio for bench_serving.",
    )
    parser.add_argument(
        "--mmlu-examples",
        type=int,
        default=32,
        help="Number of MMLU examples for sanity check per server launch.",
    )
    parser.add_argument(
        "--mmlu-threads",
        type=int,
        default=4,
        help="Threads for MMLU eval.",
    )
    parser.add_argument(
        "--launch-timeout",
        type=int,
        default=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1200),
        help="Server launch timeout (seconds).",
    )
    parser.add_argument(
        "--output-dir",
        default="bench_moe_sweep_results",
        help="Directory for output CSV / JSONL.",
    )
    parser.add_argument(
        "--extra-server-args",
        nargs="*",
        default=[],
        help="Additional server args passed to every launch (e.g. --disable-cuda-graph).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"sweep_{timestamp}.csv"
    jsonl_path = output_dir / f"sweep_{timestamp}.jsonl"

    concurrency_values = sorted(args.concurrency)

    total_servers = len(args.backends) * len(args.ep_sizes)
    total_bench_runs = total_servers * len(concurrency_values)
    _print_header("MoE Backend Concurrency Sweep")
    print(f"Model            : {args.model}")
    print(f"Backends         : {args.backends}")
    print(f"EP sizes         : {args.ep_sizes}")
    print(f"TP size          : {args.tp_size}")
    print(f"Concurrency      : {concurrency_values}")
    print(f"Input/Output len : {args.random_input_len} / {args.random_output_len}")
    print(f"MMLU examples    : {args.mmlu_examples}")
    print(f"Server launches  : {total_servers}")
    print(f"Bench runs       : {total_bench_runs}")
    print(f"Output           : {csv_path}")

    all_rows: list[dict[str, Any]] = []
    mmlu_results: dict[tuple[str, int], float] = {}
    done = 0

    for ep_size in args.ep_sizes:
        for backend in args.backends:
            label = f"{backend} (EP={ep_size}, TP={args.tp_size})"
            _print_header(f"Launching {label}")

            server_args = _build_server_args(
                backend=backend,
                tp_size=args.tp_size,
                ep_size=ep_size,
                extra_server_args=args.extra_server_args,
            )
            process = popen_launch_server(
                model=args.model,
                base_url=args.base_url,
                timeout=args.launch_timeout,
                other_args=server_args,
                env=dict(os.environ),
            )

            try:
                # -- MMLU sanity check --
                print(f"\n[MMLU] Running {args.mmlu_examples}-example sanity check ...")
                try:
                    score = run_mmlu_sanity(
                        args.base_url,
                        args.model,
                        num_examples=args.mmlu_examples,
                        num_threads=args.mmlu_threads,
                    )
                    mmlu_results[(backend, ep_size)] = score
                    status = "OK" if score >= 0.5 else "WARNING: low accuracy"
                    print(f"[MMLU] {label}: score={score:.4f} [{status}]")
                except Exception as e:
                    print(f"[MMLU] {label}: FAILED ({e})")
                    mmlu_results[(backend, ep_size)] = float("nan")

                # -- Warmup --
                print(f"\n[WARMUP] concurrency=32, 64 prompts (discarded) ...")
                try:
                    run_bench_serving(
                        base_url=args.base_url,
                        num_prompts=64,
                        max_concurrency=32,
                        random_input_len=args.random_input_len,
                        random_output_len=args.random_output_len,
                        random_range_ratio=args.random_range_ratio,
                    )
                    print("[WARMUP] done")
                except Exception as e:
                    print(f"[WARMUP] failed ({e}), continuing anyway")

                # -- Concurrency sweep --
                for conc in concurrency_values:
                    num_prompts = _num_prompts_for_concurrency(conc)
                    done += 1
                    print(
                        f"\n[BENCH {done}/{total_bench_runs}] "
                        f"{label} | concurrency={conc} | num_prompts={num_prompts}"
                    )

                    try:
                        result = run_bench_serving(
                            base_url=args.base_url,
                            num_prompts=num_prompts,
                            max_concurrency=conc,
                            random_input_len=args.random_input_len,
                            random_output_len=args.random_output_len,
                            random_range_ratio=args.random_range_ratio,
                        )
                        row: dict[str, Any] = {
                            "backend": backend,
                            "ep_size": ep_size,
                            "tp_size": args.tp_size,
                            "concurrency": conc,
                            "num_prompts": num_prompts,
                        }
                        for key in BENCH_METRIC_KEYS:
                            row[key] = result.get(key, "")
                        all_rows.append(row)

                        print(
                            f"       out_thr={row['output_throughput']:.2f} tok/s, "
                            f"mean_tpot={row['mean_tpot_ms']:.3f} ms, "
                            f"mean_ttft={row['mean_ttft_ms']:.2f} ms"
                        )
                    except Exception as e:
                        print(f"       FAILED: {e}")
                        all_rows.append(
                            {
                                "backend": backend,
                                "ep_size": ep_size,
                                "tp_size": args.tp_size,
                                "concurrency": conc,
                                "num_prompts": num_prompts,
                                **{k: "" for k in BENCH_METRIC_KEYS},
                            }
                        )

            finally:
                print(f"\n[TEARDOWN] Stopping {label}")
                kill_process_tree(process.pid)

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    with jsonl_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------

    _print_header("MMLU Sanity Check")
    print(f"{'backend':28s} {'ep':>4s} {'mmlu_score':>12s}")
    print("-" * 48)
    for (backend, ep_size), score in sorted(mmlu_results.items()):
        print(f"{backend:28s} {ep_size:4d} {score:12.4f}")

    for ep_size in args.ep_sizes:
        _print_header(f"Throughput Sweep: EP={ep_size}, TP={args.tp_size}")

        # Build pivot: concurrency -> backend -> metrics
        pivot: dict[int, dict[str, dict[str, Any]]] = {}
        for row in all_rows:
            if row["ep_size"] != ep_size:
                continue
            conc = row["concurrency"]
            bk = row["backend"]
            pivot.setdefault(conc, {})[bk] = row

        backends_present = [
            b for b in args.backends if any(b in v for v in pivot.values())
        ]
        if not backends_present:
            print("(no data)")
            continue

        # Header
        hdr_parts = [f"{'conc':>6s}", f"{'n_prompts':>10s}"]
        for bk in backends_present:
            short = bk.replace("flashinfer_", "")
            hdr_parts.append(f"{short + '_thr':>14s}")
            hdr_parts.append(f"{short + '_tpot':>12s}")
        header = " | ".join(hdr_parts)
        print(header)
        print("-" * len(header))

        for conc in concurrency_values:
            if conc not in pivot:
                continue
            parts = [f"{conc:6d}", f"{_num_prompts_for_concurrency(conc):10d}"]
            for bk in backends_present:
                data = pivot[conc].get(bk)
                if data and data.get("output_throughput") != "":
                    parts.append(f"{float(data['output_throughput']):14.2f}")
                    parts.append(f"{float(data['mean_tpot_ms']):12.3f}")
                else:
                    parts.append(f"{'FAIL':>14s}")
                    parts.append(f"{'FAIL':>12s}")
            print(" | ".join(parts))

    print(f"\nResults written to: {csv_path}")
    print(f"Raw JSONL:          {jsonl_path}")


if __name__ == "__main__":
    main()
