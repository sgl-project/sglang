#!/usr/bin/env python3
"""
MoE backend concurrency sweep benchmark.

Compares MoE runner backends (e.g. flashinfer_trtllm, flashinfer_cutedsl) across
a range of concurrency levels and EP configurations.  For each (backend, ep)
pair the script launches a server, runs a warmup pass, then sweeps concurrency
values with bench_serving and records throughput / latency metrics.

Example (8xGPU, DeepSeek-R1 NVFP4):

    python3 scripts/bench_moe_backend_compare.py \\
        --model nvidia/DeepSeek-R1-0528-NVFP4-v2 \\
        --backends flashinfer_trtllm flashinfer_cutedsl \\
        --ep-sizes 1 8 \\
        --tp-size 8 \\
        --concurrency 1 8 64 512 4096 \\
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
    "status",
    *BENCH_METRIC_KEYS,
]

MAX_RETRIES = 2
RETRY_BACKOFF_SECS = [15, 30]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num_prompts_for_concurrency(concurrency: int) -> int:
    """Return num-prompts for a given concurrency level.

    Floor of 32 ensures stable latency percentiles at low concurrency.
    Cap of 256 keeps high-concurrency runs (512+) from dominating runtime.
    """
    return min(max(concurrency, 32), 256)


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


def _server_alive(base_url: str, timeout: float = 5.0) -> bool:
    """Quick health check — True if the server responds to /get_server_info."""
    import requests

    try:
        r = requests.get(f"{base_url}/get_server_info", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _make_fail_row(
    backend: str, ep_size: int, tp_size: int, conc: int, num_prompts: int, status: str
) -> dict[str, Any]:
    return {
        "backend": backend,
        "ep_size": ep_size,
        "tp_size": tp_size,
        "concurrency": conc,
        "num_prompts": num_prompts,
        "status": status,
        **{k: "" for k in BENCH_METRIC_KEYS},
    }


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
    print(f"Server launches  : {total_servers}")
    print(f"Bench runs       : {total_bench_runs}")
    print(f"Output           : {csv_path}")

    all_rows: list[dict[str, Any]] = []
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
                server_dead = False

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
                    print(f"[WARMUP] failed ({e})")
                    if not _server_alive(args.base_url):
                        print("[WARMUP] Server dead after warmup, skipping sweep")
                        server_dead = True

                # -- Concurrency sweep --
                for conc in concurrency_values:
                    num_prompts = _num_prompts_for_concurrency(conc)
                    done += 1

                    if server_dead:
                        print(
                            f"\n[BENCH {done}/{total_bench_runs}] "
                            f"{label} | concurrency={conc} | SKIPPED (server dead)"
                        )
                        all_rows.append(
                            _make_fail_row(
                                backend,
                                ep_size,
                                args.tp_size,
                                conc,
                                num_prompts,
                                "server_dead",
                            )
                        )
                        continue

                    print(
                        f"\n[BENCH {done}/{total_bench_runs}] "
                        f"{label} | concurrency={conc} | num_prompts={num_prompts}"
                    )

                    succeeded = False
                    for attempt in range(1 + MAX_RETRIES):
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
                                "status": "ok",
                            }
                            for key in BENCH_METRIC_KEYS:
                                row[key] = result.get(key, "")
                            all_rows.append(row)
                            print(
                                f"       out_thr={row['output_throughput']:.2f} tok/s, "
                                f"mean_tpot={row['mean_tpot_ms']:.3f} ms, "
                                f"mean_ttft={row['mean_ttft_ms']:.2f} ms"
                            )
                            succeeded = True
                            break
                        except Exception as e:
                            if attempt < MAX_RETRIES and _server_alive(args.base_url):
                                wait = RETRY_BACKOFF_SECS[attempt]
                                print(
                                    f"       FAILED: {e}\n"
                                    f"       Retry {attempt + 1}/{MAX_RETRIES} "
                                    f"in {wait}s ..."
                                )
                                time.sleep(wait)
                            else:
                                print(f"       FAILED: {e}")
                                if not _server_alive(args.base_url):
                                    print("       Server dead, skipping remaining")
                                    server_dead = True
                                break

                    if not succeeded:
                        status = "server_dead" if server_dead else "failed"
                        all_rows.append(
                            _make_fail_row(
                                backend,
                                ep_size,
                                args.tp_size,
                                conc,
                                num_prompts,
                                status,
                            )
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
                if data and data.get("status") == "ok":
                    parts.append(f"{float(data['output_throughput']):14.2f}")
                    parts.append(f"{float(data['mean_tpot_ms']):12.3f}")
                else:
                    tag = data.get("status", "fail") if data else "miss"
                    parts.append(f"{tag:>14s}")
                    parts.append(f"{tag:>12s}")
            print(" | ".join(parts))

    print(f"\nResults written to: {csv_path}")
    print(f"Raw JSONL:          {jsonl_path}")


if __name__ == "__main__":
    main()
