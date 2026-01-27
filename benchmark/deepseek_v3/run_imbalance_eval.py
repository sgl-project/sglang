#!/usr/bin/env python3
"""
Evaluate imbalance score for Waterfill and Baseline under different configurations.

This script runs experiments with:
- Different input_len: 256, 512, 1024, 2048
- EPLB enabled vs disabled
- Waterfill vs Baseline

It collects logs and parses imbalance metrics at stages:
- pre_eplb: before EPLB
- post_eplb: after EPLB
- post_waterfill: after Waterfill (only for Waterfill path)

Usage:
    python run_imbalance_eval.py \
        --model-path /path/to/DeepSeek-V3 \
        --result-root /path/to/results \
        --init-expert-location /path/to/eplb/record.pt \
        --port 31000
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ===================== Configuration =====================

INPUT_LENS = [256, 512, 1024, 2048]
BATCH_SIZE = 16
OUTPUT_LEN = 1

# Server startup timeout (seconds)
SERVER_TIMEOUT = 1800

# ===================== Helper Functions =====================


def kill_server_processes(port: int):
    """Best-effort cleanup of stale sglang server processes using the given port.

    IMPORTANT: do NOT use `lsof -ti:<port>` here.
    `lsof` can return client processes (including this benchmark driver) which can
    lead to self-kill and exit code 137.
    """
    # Kill only launch_server processes that match this port.
    subprocess.run(
        ["pkill", "-9", "-f", rf"sglang\.launch_server.*--port {port}\b"],
        check=False,
    )
    subprocess.run(
        ["pkill", "-9", "-f", rf"sglang\.launch_server.*--port={port}\b"],
        check=False,
    )
    # launch_server can leave behind worker/scheduler processes with custom proctitles
    # like `sglang::scheduler_TP0_EP0` which may not include the port in argv. These
    # can hold onto large GPU allocations and cause OOM on subsequent runs.
    subprocess.run(
        ["pkill", "-9", "-f", r"sglang::scheduler_TP"],
        check=False,
    )
    time.sleep(2)


def wait_for_server(
    port: int,
    timeout: int = SERVER_TIMEOUT,
    proc: Optional[subprocess.Popen] = None,
) -> bool:
    """Wait for server to be ready.

    If `proc` is provided, return early when the process exits to avoid waiting
    the full timeout on startup failures (e.g. OOM).
    """
    import requests

    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def parse_imbalance_logs(log_content: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Parse imbalance logs from server output.

    Returns:
        Dict[stage, Dict[layer_id, List[imbalance_values]]]

    Log format:
        [deepep_eplb_load] mode=<mode> layer=<layer_id> ep_rank=<rank>/<world_size>
        stage=<stage> total=<total> max=<max> avg=<avg> imbal=<imbal>x
    """
    # Pattern to match the log lines
    pattern = re.compile(
        r"\[deepep_eplb_load\].*?"
        r"mode=(\w+).*?"
        r"layer=(\d+).*?"
        r"ep_rank=(\d+)/(\d+).*?"
        r"stage=(\w+).*?"
        r"imbal=([\d.]+)x"
    )

    # Collect imbalance values per stage per layer (only from rank 0)
    result = defaultdict(lambda: defaultdict(list))

    for line in log_content.split("\n"):
        # Some ranks can flush multiple log entries without a newline boundary,
        # so a single physical line may contain multiple `[deepep_eplb_load]` entries.
        for match in pattern.finditer(line):
            mode, layer_id, ep_rank, ep_world, stage, imbal = match.groups()
            # Only collect from rank 0 to avoid duplicates
            if ep_rank == "0":
                result[stage][layer_id].append(float(imbal))

    return result


def _read_last_jsonl(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def compute_average_imbalance(
    stage_data: Dict[str, Dict[str, List[float]]]
) -> Dict[str, float]:
    """
    Compute average imbalance across all layers for each stage.

    Returns:
        Dict[stage, avg_imbalance]
    """
    result = {}
    for stage, layer_data in stage_data.items():
        all_values = []
        for layer_id, values in layer_data.items():
            all_values.extend(values)
        if all_values:
            result[stage] = sum(all_values) / len(all_values)
        else:
            result[stage] = 0.0
    return result


def run_experiment(
    waterfill_sglang_dir: str,
    baseline_sglang_dir: str,
    model_path: str,
    input_len: int,
    batch_size: int,
    output_len: int,
    port: int,
    enable_waterfill: bool,
    enable_eplb: bool,
    init_expert_location: Optional[str],
    log_file: str,
) -> Tuple[Dict[str, float], Optional[dict], Optional[str]]:
    """
    Run a single experiment configuration.

    Returns:
        Dict[stage, avg_imbalance]
    """
    mode = "waterfill" if enable_waterfill else "baseline"
    eplb_str = "eplb" if enable_eplb else "no_eplb"

    print(f"\n{'='*60}")
    print(f"Running: mode={mode}, eplb={eplb_str}, input_len={input_len}")
    print(f"{'='*60}")

    # Kill any existing server
    kill_server_processes(port)

    # Use the appropriate sglang directory
    sglang_dir = waterfill_sglang_dir if enable_waterfill else baseline_sglang_dir
    python_path = os.path.join(sglang_dir, "python")

    # Reinstall the sglang package from the appropriate directory
    print(f"Installing sglang from {sglang_dir}...")
    subprocess.run(
        ["pip", "install", "-e", "python[dev]", "--no-deps", "-q"],
        cwd=sglang_dir,
        check=False,
    )

    # Build server command

    server_cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        "8",
        "--ep-size",
        "8",
        "--port",
        str(port),
        "--trust-remote-code",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--disable-radix-cache",
    ]

    if enable_waterfill:
        server_cmd.append("--enable-deepep-waterfill")

    if enable_eplb and init_expert_location:
        server_cmd.extend(["--init-expert-location", init_expert_location])

    # Environment variables for debug logging
    env = os.environ.copy()
    env["PYTHONPATH"] = python_path + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # Some dev containers mount a source checkout of flashinfer on PYTHONPATH which can
    # mismatch the installed flashinfer-cubin package. Allow bypass so we can run the
    # benchmark without env surgery.
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env["SGLANG_DEBUG_WATERFILL_EPLB"] = "1"
    env["SGLANG_DEBUG_WATERFILL_EPLB_LAYER"] = "all"  # Log all layers
    env["SGLANG_DEBUG_WATERFILL_EPLB_MAX_PRINTS"] = "1"
    # Filter out decode-only steps so we only log prefill.
    env["SGLANG_DEBUG_WATERFILL_EPLB_MIN_TOKENS"] = "64"

    # Start server
    print(f"Starting server: {' '.join(server_cmd)}")
    with open(log_file, "w") as log_f:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    try:
        # Wait for server to be ready
        print("Waiting for server to start...")
        if not wait_for_server(port, proc=server_proc):
            print(f"ERROR: Server failed to start within {SERVER_TIMEOUT}s")
            return {}, None, None

        print("Server is ready. Running benchmark...")

        # Run bench_one_batch_server
        out_dir = os.path.dirname(log_file)
        bench_result_file = os.path.join(
            out_dir,
            f"bench_one_batch_{mode}_{eplb_str}_in{input_len}_bs{batch_size}_o{output_len}.jsonl",
        )
        bench_cmd = [
            sys.executable,
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            "None",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--batch-size",
            str(batch_size),
            "--input-len",
            str(input_len),
            "--output-len",
            str(output_len),
            "--skip-warmup",
            "--result-filename",
            bench_result_file,
            "--no-append-to-github-summary",
        ]

        bench_result = subprocess.run(
            bench_cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        print(f"Benchmark stdout:\n{bench_result.stdout}")
        if bench_result.returncode != 0:
            print(f"Benchmark stderr:\n{bench_result.stderr}")

        # Give time for logs to be flushed
        time.sleep(5)

    finally:
        # Kill server (entire process group).
        try:
            os.killpg(server_proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server_proc.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        kill_server_processes(port)

    # Parse logs
    print(f"Parsing logs from {log_file}...")
    with open(log_file, "r") as f:
        log_content = f.read()

    stage_data = parse_imbalance_logs(log_content)
    avg_imbalance = compute_average_imbalance(stage_data)
    bench_summary = (
        _read_last_jsonl(bench_result_file) if "bench_result_file" in locals() else None
    )

    print(f"Parsed imbalance data:")
    for stage, avg in sorted(avg_imbalance.items()):
        num_layers = len(stage_data.get(stage, {}))
        print(f"  {stage}: avg={avg:.4f}x (from {num_layers} layers)")

    return (
        avg_imbalance,
        bench_summary,
        (bench_result_file if "bench_result_file" in locals() else None),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate imbalance score")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--result-root", type=str, required=True, help="Root directory for results"
    )
    parser.add_argument(
        "--init-expert-location",
        type=str,
        default=None,
        help="Path to EPLB expert location file",
    )
    parser.add_argument("--port", type=int, default=31000, help="Server port")
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        default=INPUT_LENS,
        help="Input lengths to test",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--output-len", type=int, default=OUTPUT_LEN, help="Output length"
    )
    parser.add_argument(
        "--waterfill-sglang-dir",
        type=str,
        default="/home/xutingz/workspace/gitsrc/sglang",
        help="Path to SGLang source directory for Waterfill",
    )
    parser.add_argument(
        "--baseline-sglang-dir",
        type=str,
        default="/home/xutingz/workspace/gitsrc/sglang_baseline_98a107d",
        help="Path to SGLang source directory for Baseline",
    )
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.result_root, f"imbalance_eval_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Results will be saved to: {out_dir}")

    # Store all results
    all_results = []
    results_file = os.path.join(out_dir, "results.json")

    # Test configurations:
    # 1. Waterfill with EPLB
    # 2. Waterfill without EPLB
    # 3. Baseline with EPLB
    # 4. Baseline without EPLB

    configs = [
        ("waterfill", True, True),  # enable_waterfill, enable_eplb
        ("waterfill", True, False),
        ("baseline", False, True),
        ("baseline", False, False),
    ]

    for input_len in args.input_lens:
        for name, enable_waterfill, enable_eplb in configs:
            eplb_str = "eplb" if enable_eplb else "no_eplb"
            log_filename = f"server_{name}_{eplb_str}_in{input_len}.log"
            log_file = os.path.join(out_dir, log_filename)

            # Run experiment
            avg_imbalance, bench_summary, bench_result_file = run_experiment(
                waterfill_sglang_dir=args.waterfill_sglang_dir,
                baseline_sglang_dir=args.baseline_sglang_dir,
                model_path=args.model_path,
                input_len=input_len,
                batch_size=args.batch_size,
                output_len=args.output_len,
                port=args.port,
                enable_waterfill=enable_waterfill,
                enable_eplb=enable_eplb,
                init_expert_location=args.init_expert_location if enable_eplb else None,
                log_file=log_file,
            )

            result = {
                "mode": name,
                "enable_eplb": enable_eplb,
                "input_len": input_len,
                "batch_size": args.batch_size,
                "output_len": args.output_len,
                "avg_imbalance": avg_imbalance,
                "bench": bench_summary,
                "bench_result_file": bench_result_file,
            }
            all_results.append(result)
            # Save partial progress so a long run can be resumed / inspected.
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # Save results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by input_len
    by_input_len = defaultdict(list)
    for r in all_results:
        by_input_len[r["input_len"]].append(r)

    for input_len in sorted(by_input_len.keys()):
        print(f"\n=== input_len={input_len} ===")
        print(
            f"{'Mode':<15} {'EPLB':<8} {'latency(s)':<10} {'overall_tps':<12} {'pre_eplb':<12} {'post_eplb':<12} {'post_waterfill':<15}"
        )
        print("-" * 65)

        for r in by_input_len[input_len]:
            mode = r["mode"]
            eplb = "Yes" if r["enable_eplb"] else "No"
            avg = r["avg_imbalance"]
            bench = r.get("bench") or {}
            lat = bench.get("latency", None)
            tps = bench.get("overall_throughput", None)
            lat_s = f"{float(lat):.3f}" if lat is not None else "N/A"
            tps_s = f"{float(tps):.1f}" if tps is not None else "N/A"
            pre_eplb = (
                f"{avg.get('pre_eplb', 0):.4f}x" if avg.get("pre_eplb") else "N/A"
            )
            post_eplb = (
                f"{avg.get('post_eplb', 0):.4f}x" if avg.get("post_eplb") else "N/A"
            )
            post_wf = (
                f"{avg.get('post_waterfill', 0):.4f}x"
                if avg.get("post_waterfill")
                else "N/A"
            )
            print(
                f"{mode:<15} {eplb:<8} {lat_s:<10} {tps_s:<12} {pre_eplb:<12} {post_eplb:<12} {post_wf:<15}"
            )

    # Calculate improvement metrics
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    for input_len in sorted(by_input_len.keys()):
        print(f"\n=== input_len={input_len} ===")

        results_by_config = {}
        for r in by_input_len[input_len]:
            key = (r["mode"], r["enable_eplb"])
            results_by_config[key] = r["avg_imbalance"]

        # 1. EPLB improvement (comparing pre_eplb vs post_eplb)
        for mode in ["waterfill", "baseline"]:
            with_eplb = results_by_config.get((mode, True), {})
            if with_eplb.get("pre_eplb") and with_eplb.get("post_eplb"):
                pre = with_eplb["pre_eplb"]
                post = with_eplb["post_eplb"]
                improvement = (pre - post) / pre * 100
                print(
                    f"  {mode} EPLB improvement: {pre:.4f}x -> {post:.4f}x ({improvement:+.2f}%)"
                )

        # 2. Waterfill improvement (comparing post_eplb vs post_waterfill)
        wf_with_eplb = results_by_config.get(("waterfill", True), {})
        if wf_with_eplb.get("post_eplb") and wf_with_eplb.get("post_waterfill"):
            post_eplb = wf_with_eplb["post_eplb"]
            post_wf = wf_with_eplb["post_waterfill"]
            improvement = (post_eplb - post_wf) / post_eplb * 100
            print(
                f"  Waterfill improvement over EPLB: {post_eplb:.4f}x -> {post_wf:.4f}x ({improvement:+.2f}%)"
            )

        # 3. Waterfill without EPLB improvement
        wf_no_eplb = results_by_config.get(("waterfill", False), {})
        if wf_no_eplb.get("pre_eplb") and wf_no_eplb.get("post_waterfill"):
            pre = wf_no_eplb["pre_eplb"]
            post_wf = wf_no_eplb["post_waterfill"]
            improvement = (pre - post_wf) / pre * 100
            print(
                f"  Waterfill (no EPLB) improvement: {pre:.4f}x -> {post_wf:.4f}x ({improvement:+.2f}%)"
            )

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
