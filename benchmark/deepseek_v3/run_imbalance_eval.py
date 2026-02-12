#!/usr/bin/env python3
"""
Evaluate imbalance score for Waterfill and Baseline under different configurations.

Supports both EP8 (single-node) and EP16 (multi-node, 2 nodes × 8 GPUs).

This script runs experiments with:
- Different input_len: 256, 512, 1024, 2048
- EPLB enabled vs disabled
- Waterfill vs Baseline

It collects logs and parses imbalance metrics at stages:
- pre_eplb: before EPLB
- post_eplb: after EPLB
- post_waterfill: after Waterfill (only for Waterfill path)

Usage:
    # EP8 (single node, backward compatible):
    python run_imbalance_eval.py --ep 8 \
        --model-path /path/to/DeepSeek-V3 \
        --result-root /path/to/results \
        --init-expert-location /path/to/ep8_logical_count.pt

    # EP16 (multi-node):
    python run_imbalance_eval.py --ep 16 \
        --model-path /path/to/DeepSeek-V3 \
        --result-root /path/to/results \
        --init-expert-location /path/to/ep16_logical_count.pt

    # Run specific configs only:
    python run_imbalance_eval.py --ep 16 \
        --configs waterfill_eplb,baseline_eplb \
        --result-root /path/to/results

    # Show per-layer breakdown:
    python run_imbalance_eval.py --ep 16 --per-layer \
        --result-root /path/to/results
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ===================== Cluster Configuration =====================

NODE_IPS = {
    8: ["10.6.131.5"],
    16: ["10.6.131.5", "10.6.131.6"],
}
EP_CONFIG = {
    8: {"actual_tp": 8, "actual_dp": 8, "nnodes": 1},
    16: {"actual_tp": 16, "actual_dp": 16, "nnodes": 2},
}
DIST_INIT_PORT = 20000
CONTAINER = "sglang_lb"
MODEL_PATH = "/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3"

# ===================== Defaults =====================

INPUT_LENS = [256, 512, 1024, 2048]
BATCH_SIZE = 16  # local batch size (per rank); global = local × dp_size
OUTPUT_LEN = 1
SERVER_TIMEOUT = 1800

# Experiment configurations: (config_name, enable_waterfill, enable_eplb)
ALL_CONFIGS = [
    ("waterfill_eplb", True, True),
    ("waterfill_no_eplb", True, False),
    ("baseline_eplb", False, True),
    ("baseline_no_eplb", False, False),
]

# Debug environment variables for imbalance logging
DEBUG_ENV_VARS = {
    "SGLANG_DEBUG_WATERFILL_EPLB": "1",
    "SGLANG_DEBUG_WATERFILL_EPLB_LAYER": "all",
    "SGLANG_DEBUG_WATERFILL_EPLB_MAX_PRINTS": "1",
    "SGLANG_DEBUG_WATERFILL_EPLB_MIN_TOKENS": "64",
}

# ===================== Multi-node Helpers =====================

# Patterns to kill sglang processes (from bench_waterfill_multinode.py)
KILL_PATTERNS = [
    "sglang.launch_server",
    "sglang::scheduler",
    "sglang::data_pa",
    "sglang::detoken",
    "sglang::nccl",
    "sglang.srt",
]


def kill_servers(node_ips: List[str]) -> None:
    """Kill all sglang server processes on all nodes."""
    for ip in node_ips:
        kill_cmds = "; ".join(
            f"pkill -9 -f '{pat}' 2>/dev/null" for pat in KILL_PATTERNS
        )
        kill_cmds += "; pkill -9 -f bench_one_batch 2>/dev/null"
        kill_cmds += (
            "; rm -f /dev/shm/nccl* 2>/dev/null" "; rm -f /dev/shm/nvshmem* 2>/dev/null"
        )
        if ip == node_ips[0]:
            subprocess.run(
                ["bash", "-c", kill_cmds],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"xutingz@{ip}",
                    f"docker exec {CONTAINER} bash -c '{kill_cmds}'",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    time.sleep(15)


def kill_server_processes_ep8(port: int) -> None:
    """Best-effort cleanup of stale sglang server processes (EP8 only)."""
    subprocess.run(
        ["pkill", "-9", "-f", rf"sglang\.launch_server.*--port {port}\b"],
        check=False,
    )
    subprocess.run(
        ["pkill", "-9", "-f", rf"sglang\.launch_server.*--port={port}\b"],
        check=False,
    )
    subprocess.run(
        ["pkill", "-9", "-f", r"sglang::scheduler_TP"],
        check=False,
    )
    time.sleep(2)


def pip_install_sglang(sglang_dir: str, node_ips: List[str]) -> None:
    """Install sglang from the given directory on all nodes (editable, no-deps)."""
    install_cmd = f"cd {sglang_dir} && pip install -e 'python[dev]' --no-deps -q"
    print(f"  Installing sglang from {sglang_dir} on all nodes...", flush=True)

    # Local node (node 0)
    subprocess.run(
        ["bash", "-c", install_cmd],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Remote nodes
    for ip in node_ips[1:]:
        subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                f"xutingz@{ip}",
                f"docker exec {CONTAINER} bash -c '{install_cmd}'",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"  Install done.\n", flush=True)


def wait_for_server(
    url: str,
    timeout: int = SERVER_TIMEOUT,
    proc: Optional[subprocess.Popen] = None,
) -> bool:
    """Wait for server to be ready at the given health URL."""
    import requests

    start = time.time()
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


# ===================== Log Parsing =====================


def parse_imbalance_logs(log_content: str) -> Dict[str, Dict[str, List[float]]]:
    """Parse ``[deepep_eplb_load]`` lines and return ``{stage: {layer_id: [imbal_values]}}``."""
    pattern = re.compile(
        r"\[deepep_eplb_load\].*?"
        r"mode=(\w+).*?"
        r"layer=(\d+).*?"
        r"ep_rank=(\d+)/(\d+).*?"
        r"stage=(\w+).*?"
        r"imbal=([\d.]+)x"
    )

    result: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for line in log_content.split("\n"):
        for match in pattern.finditer(line):
            mode, layer_id, ep_rank, ep_world, stage, imbal = match.groups()
            # Only collect from rank 0 to avoid duplicates within a node
            if ep_rank == "0":
                result[stage][layer_id].append(float(imbal))

    return dict(result)


def merge_stage_data(
    *stage_datas: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, List[float]]]:
    """Merge imbalance data from multiple nodes."""
    merged: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for sd in stage_datas:
        for stage, layer_data in sd.items():
            for layer_id, values in layer_data.items():
                merged[stage][layer_id].extend(values)
    return dict(merged)


def _read_last_jsonl(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def compute_imbalance_stats(
    stage_data: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict]:
    """
    Compute mean, median, and per-layer imbalance for each stage.

    Returns:
        Dict[stage, {"mean": float, "median": float, "per_layer": Dict[layer_id, float]}]
    """
    result = {}
    for stage, layer_data in stage_data.items():
        per_layer = {}
        all_values = []
        for layer_id, values in sorted(layer_data.items(), key=lambda x: int(x[0])):
            layer_avg = sum(values) / len(values) if values else 0.0
            per_layer[layer_id] = layer_avg
            all_values.append(layer_avg)
        if all_values:
            result[stage] = {
                "mean": sum(all_values) / len(all_values),
                "median": statistics.median(all_values),
                "per_layer": per_layer,
            }
        else:
            result[stage] = {"mean": 0.0, "median": 0.0, "per_layer": {}}
    return result


# ===================== EP8 Experiment Runner =====================


def run_experiment_ep8(
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
) -> Tuple[Dict[str, Dict], Optional[dict], Optional[str]]:
    """
    Run a single EP8 experiment (single node, local processes).

    Returns:
        (imbalance_stats, bench_summary, bench_result_file)
    """
    mode = "waterfill" if enable_waterfill else "baseline"
    eplb_str = "eplb" if enable_eplb else "no_eplb"

    print(f"\n{'='*60}")
    print(f"Running EP8: mode={mode}, eplb={eplb_str}, input_len={input_len}")
    print(f"{'='*60}")

    kill_server_processes_ep8(port)

    sglang_dir = waterfill_sglang_dir if enable_waterfill else baseline_sglang_dir
    python_path = os.path.join(sglang_dir, "python")

    print(f"Installing sglang from {sglang_dir}...")
    subprocess.run(
        ["pip", "install", "-e", "python[dev]", "--no-deps", "-q"],
        cwd=sglang_dir,
        check=False,
    )

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

    env = os.environ.copy()
    env["PYTHONPATH"] = python_path + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.update(DEBUG_ENV_VARS)

    print(f"Starting server: {' '.join(server_cmd)}")
    with open(log_file, "w") as log_f:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    bench_result_file = None
    try:
        print("Waiting for server to start...")
        health_url = f"http://127.0.0.1:{port}/health"
        if not wait_for_server(health_url, proc=server_proc):
            print(f"ERROR: Server failed to start within {SERVER_TIMEOUT}s")
            return {}, None, None

        print("Server is ready. Running benchmark...")

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
            bench_cmd, capture_output=True, text=True, env=env
        )
        print(f"Benchmark stdout:\n{bench_result.stdout}")
        if bench_result.returncode != 0:
            print(f"Benchmark stderr:\n{bench_result.stderr}")

        time.sleep(5)

    finally:
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
        kill_server_processes_ep8(port)

    # Parse logs
    print(f"Parsing logs from {log_file}...")
    with open(log_file, "r") as f:
        log_content = f.read()

    stage_data = parse_imbalance_logs(log_content)
    imbalance_stats = compute_imbalance_stats(stage_data)
    bench_summary = _read_last_jsonl(bench_result_file) if bench_result_file else None

    print(f"Parsed imbalance data:")
    for stage, stats in sorted(imbalance_stats.items()):
        num_layers = len(stats["per_layer"])
        print(
            f"  {stage}: mean={stats['mean']:.4f}x median={stats['median']:.4f}x ({num_layers} layers)"
        )

    return imbalance_stats, bench_summary, bench_result_file


# ===================== EP16 Experiment Runner =====================


def launch_server_ep16(
    *,
    node_ips: List[str],
    enable_waterfill: bool,
    init_expert_location: Optional[str],
    log_dir: str,
    dist_init_port: int = DIST_INIT_PORT,
) -> subprocess.Popen:
    """Launch sglang server across 2 nodes for EP16. Returns the local (node 0) process."""
    cfg = EP_CONFIG[16]
    dist_init_addr = f"{node_ips[0]}:{dist_init_port}"

    def _build_server_cmd(node_rank: int) -> List[str]:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_PATH,
            "--trust-remote-code",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--tp",
            str(cfg["actual_tp"]),
            "--dp-size",
            str(cfg["actual_dp"]),
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "normal",
            "--chunked-prefill-size",
            "-1",
            "--disable-radix-cache",
            "--max-prefill-tokens",
            "8192",
            "--max-running-requests",
            "2048",
            "--load-balance-method",
            "round_robin",
            "--log-level",
            "info",
            "--watchdog-timeout",
            "600",
            "--mem-fraction-static",
            "0.75",
            "--skip-server-warmup",
            "--dist-init-addr",
            dist_init_addr,
            "--nnodes",
            str(cfg["nnodes"]),
            "--node-rank",
            str(node_rank),
            "--enable-dp-attention",
            "--disable-cuda-graph",
        ]
        if enable_waterfill:
            cmd.append("--enable-deepep-waterfill")
        if init_expert_location:
            cmd.extend(["--init-expert-location", init_expert_location])
        return cmd

    # Build env_vars export string for SSH (includes debug vars)
    env_exports = (
        "export SGLANG_LOG_MS=1; "
        "export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0; "
        "export NCCL_DEBUG=WARN; "
    )
    for k, v in DEBUG_ENV_VARS.items():
        env_exports += f"export {k}={v}; "

    os.makedirs(log_dir, exist_ok=True)

    # Launch worker nodes (rank 1+) via SSH
    for rank in range(1, cfg["nnodes"]):
        ip = node_ips[rank]
        worker_cmd = _build_server_cmd(rank)
        log_file = os.path.join(log_dir, f"server_node{rank}.log")
        docker_cmd = env_exports + " ".join(worker_cmd)
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"xutingz@{ip}",
            f"docker exec -d {CONTAINER} bash -c '"
            f"mkdir -p {log_dir} && "
            f"{docker_cmd} > {log_file} 2>&1'",
        ]
        subprocess.Popen(ssh_cmd)
        time.sleep(2)

    # Launch node 0 locally
    if cfg["nnodes"] > 1:
        time.sleep(3)
    local_cmd = _build_server_cmd(0)
    log_file_path = os.path.join(log_dir, "server_node0.log")
    log_f = open(log_file_path, "w")
    env = os.environ.copy()
    env["SGLANG_LOG_MS"] = "1"
    env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    env["NCCL_DEBUG"] = "WARN"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.update(DEBUG_ENV_VARS)

    proc = subprocess.Popen(
        local_cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_f = log_f  # type: ignore[attr-defined]
    return proc


def collect_logs_ep16(node_ips: List[str], log_dir: str) -> str:
    """Collect and concatenate logs from all EP16 nodes."""
    all_logs = []

    # Node 0: local
    node0_log = os.path.join(log_dir, "server_node0.log")
    if os.path.exists(node0_log):
        with open(node0_log, "r") as f:
            all_logs.append(f.read())

    # Remote nodes: fetch via SSH
    for rank in range(1, len(node_ips)):
        ip = node_ips[rank]
        remote_log = os.path.join(log_dir, f"server_node{rank}.log")
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"xutingz@{ip}",
                    f"docker exec {CONTAINER} cat {remote_log}",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                all_logs.append(result.stdout)
            else:
                print(
                    f"  Warning: failed to collect log from node {rank} ({ip}): {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            print(f"  Warning: timeout collecting log from node {rank} ({ip})")

    return "\n".join(all_logs)


def run_experiment_ep16(
    waterfill_sglang_dir: str,
    baseline_sglang_dir: str,
    input_len: int,
    batch_size: int,
    output_len: int,
    enable_waterfill: bool,
    enable_eplb: bool,
    init_expert_location: Optional[str],
    log_dir: str,
    node_ips: List[str],
) -> Tuple[Dict[str, Dict], Optional[dict], Optional[str]]:
    """
    Run a single EP16 experiment (multi-node).

    batch_size is LOCAL (per rank). Global = local × dp_size.

    Returns:
        (imbalance_stats, bench_summary, bench_result_file)
    """
    cfg = EP_CONFIG[16]
    dp_size = cfg["actual_dp"]
    global_batch_size = batch_size * dp_size
    mode = "waterfill" if enable_waterfill else "baseline"
    eplb_str = "eplb" if enable_eplb else "no_eplb"

    print(f"\n{'='*60}")
    print(f"Running EP16: mode={mode}, eplb={eplb_str}, input_len={input_len}")
    print(f"  local_bs={batch_size}, global_bs={global_batch_size}")
    print(f"{'='*60}")

    kill_servers(node_ips)

    # Install correct sglang on all nodes
    sglang_dir = waterfill_sglang_dir if enable_waterfill else baseline_sglang_dir
    pip_install_sglang(sglang_dir, node_ips)

    os.makedirs(log_dir, exist_ok=True)

    print(f"Launching EP16 server (dist port {DIST_INIT_PORT})...", flush=True)
    proc = launch_server_ep16(
        node_ips=node_ips,
        enable_waterfill=enable_waterfill,
        init_expert_location=init_expert_location,
        log_dir=log_dir,
        dist_init_port=DIST_INIT_PORT,
    )

    bench_result_file = None
    try:
        base_url = f"http://{node_ips[0]}:30000"
        health_url = f"{base_url}/health"
        print(f"Waiting for server at {base_url}...", flush=True)
        if not wait_for_server(health_url, proc=proc):
            print(f"ERROR: Server failed to start within {SERVER_TIMEOUT}s")
            return {}, None, None

        print("Server is ready. Running benchmark...", flush=True)

        # Switch local node to optimized repo for bench client
        optimized_dir = waterfill_sglang_dir
        if sglang_dir != optimized_dir:
            print("  Switching local node to optimized repo for bench client...")
            subprocess.run(
                [
                    "bash",
                    "-c",
                    f"cd {optimized_dir} && pip install -e 'python[dev]' --no-deps -q",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        bench_result_file = os.path.join(
            log_dir,
            f"bench_one_batch_{mode}_{eplb_str}_in{input_len}_bs{global_batch_size}_o{output_len}.jsonl",
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "99"  # client on CPU
        bench_cmd = [
            sys.executable,
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            "None",
            "--base-url",
            base_url,
            "--batch-size",
            str(global_batch_size),
            "--input-len",
            str(input_len),
            "--output-len",
            str(output_len),
            "--dataset-name",
            "random",
            "--result-filename",
            bench_result_file,
            "--no-append-to-github-summary",
        ]

        bench_result = subprocess.run(
            bench_cmd, capture_output=True, text=True, env=env
        )
        print(f"Benchmark stdout:\n{bench_result.stdout}")
        if bench_result.returncode != 0:
            print(f"Benchmark stderr:\n{bench_result.stderr}")

        time.sleep(5)

    finally:
        print("Stopping server...", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        try:
            proc._log_f.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        kill_servers(node_ips)

    # Collect and parse logs from all nodes
    print(f"Collecting logs from all nodes...", flush=True)
    combined_logs = collect_logs_ep16(node_ips, log_dir)

    stage_data = parse_imbalance_logs(combined_logs)
    imbalance_stats = compute_imbalance_stats(stage_data)
    bench_summary = _read_last_jsonl(bench_result_file) if bench_result_file else None

    print(f"Parsed imbalance data:")
    for stage, stats in sorted(imbalance_stats.items()):
        num_layers = len(stats["per_layer"])
        print(
            f"  {stage}: mean={stats['mean']:.4f}x median={stats['median']:.4f}x ({num_layers} layers)"
        )

    return imbalance_stats, bench_summary, bench_result_file


# ===================== Main =====================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate imbalance score for EP8/EP16"
    )
    parser.add_argument(
        "--ep",
        type=int,
        choices=[8, 16],
        default=8,
        help="EP size: 8 (single node) or 16 (2 nodes). Default: 8",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to model (used for EP8; EP16 uses MODEL_PATH constant)",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        required=True,
        help="Root directory for results",
    )
    parser.add_argument(
        "--init-expert-location",
        type=str,
        default=None,
        help="Path to EPLB expert location .pt file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=31000,
        help="Server port (EP8 only; EP16 always uses 30000)",
    )
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        default=INPUT_LENS,
        help="Input lengths to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Local batch size (per rank). For EP16, global = local × dp_size",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=OUTPUT_LEN,
        help="Output length",
    )
    parser.add_argument(
        "--waterfill-sglang-dir",
        type=str,
        default="/lustre/raplab/client/xutingz/workspace/gitsrc/sglang",
        help="Path to SGLang source directory for Waterfill",
    )
    parser.add_argument(
        "--baseline-sglang-dir",
        type=str,
        default="/lustre/raplab/client/xutingz/workspace/gitsrc/sglang_baseline_98a107d",
        help="Path to SGLang source directory for Baseline",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated config names to run. "
        "Available: waterfill_eplb,waterfill_no_eplb,baseline_eplb,baseline_no_eplb. "
        "Default: all 4",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Print per-layer imbalance breakdown in summary",
    )
    args = parser.parse_args()

    ep = args.ep
    node_ips = NODE_IPS[ep]

    # Filter configs
    if args.configs:
        selected = {c.strip() for c in args.configs.split(",")}
        configs = [c for c in ALL_CONFIGS if c[0] in selected]
        unknown = selected - {c[0] for c in ALL_CONFIGS}
        if unknown:
            print(f"WARNING: Unknown configs ignored: {unknown}")
        if not configs:
            print(
                f"ERROR: No valid configs selected. Available: {[c[0] for c in ALL_CONFIGS]}"
            )
            sys.exit(1)
    else:
        configs = list(ALL_CONFIGS)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.result_root, f"imbalance_eval_ep{ep}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nImbalance Evaluation Config:", flush=True)
    print(f"  EP: {ep}", flush=True)
    print(f"  Nodes: {node_ips}", flush=True)
    print(f"  Configs: {[c[0] for c in configs]}", flush=True)
    print(f"  Input lens: {args.input_lens}", flush=True)
    print(f"  Batch size (local): {args.batch_size}", flush=True)
    if ep == 16:
        dp_size = EP_CONFIG[16]["actual_dp"]
        print(f"  Batch size (global): {args.batch_size * dp_size}", flush=True)
    print(f"  Output dir: {out_dir}\n", flush=True)

    all_results = []
    results_file = os.path.join(out_dir, "results.json")

    for input_len in args.input_lens:
        for config_name, enable_waterfill, enable_eplb in configs:
            eplb_str = "eplb" if enable_eplb else "no_eplb"
            mode = "waterfill" if enable_waterfill else "baseline"

            # Skip EPLB configs if no expert location file
            if enable_eplb and not args.init_expert_location:
                print(
                    f"SKIP {config_name}: --init-expert-location required for EPLB configs"
                )
                continue

            if ep == 8:
                log_filename = f"server_{mode}_{eplb_str}_in{input_len}.log"
                log_file = os.path.join(out_dir, log_filename)

                imbalance_stats, bench_summary, bench_result_file = run_experiment_ep8(
                    waterfill_sglang_dir=args.waterfill_sglang_dir,
                    baseline_sglang_dir=args.baseline_sglang_dir,
                    model_path=args.model_path,
                    input_len=input_len,
                    batch_size=args.batch_size,
                    output_len=args.output_len,
                    port=args.port,
                    enable_waterfill=enable_waterfill,
                    enable_eplb=enable_eplb,
                    init_expert_location=(
                        args.init_expert_location if enable_eplb else None
                    ),
                    log_file=log_file,
                )
            else:
                log_subdir = os.path.join(
                    out_dir, f"logs_{mode}_{eplb_str}_in{input_len}"
                )

                imbalance_stats, bench_summary, bench_result_file = run_experiment_ep16(
                    waterfill_sglang_dir=args.waterfill_sglang_dir,
                    baseline_sglang_dir=args.baseline_sglang_dir,
                    input_len=input_len,
                    batch_size=args.batch_size,
                    output_len=args.output_len,
                    enable_waterfill=enable_waterfill,
                    enable_eplb=enable_eplb,
                    init_expert_location=(
                        args.init_expert_location if enable_eplb else None
                    ),
                    log_dir=log_subdir,
                    node_ips=node_ips,
                )

            # Flatten stats for backward compat: store both avg_imbalance (mean only) and full stats
            avg_imbalance = {
                stage: stats["mean"] for stage, stats in imbalance_stats.items()
            }
            result = {
                "config": config_name,
                "mode": mode,
                "enable_eplb": enable_eplb,
                "ep": ep,
                "input_len": input_len,
                "batch_size": args.batch_size,
                "output_len": args.output_len,
                "avg_imbalance": avg_imbalance,
                "imbalance_stats": {
                    stage: {
                        "mean": stats["mean"],
                        "median": stats["median"],
                        "per_layer": stats["per_layer"],
                    }
                    for stage, stats in imbalance_stats.items()
                },
                "bench": bench_summary,
                "bench_result_file": bench_result_file,
            }
            all_results.append(result)
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # Save final results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print summary table ──
    print("\n" + "=" * 100)
    print(f"SUMMARY (EP{ep})")
    print("=" * 100)

    by_input_len = defaultdict(list)
    for r in all_results:
        by_input_len[r["input_len"]].append(r)

    for input_len in sorted(by_input_len.keys()):
        print(f"\n=== input_len={input_len} ===")
        print(
            f"{'Config':<22} {'latency(s)':<10} {'overall_tps':<12} "
            f"{'pre_eplb(mean)':<15} {'pre_eplb(med)':<14} "
            f"{'post_eplb(mean)':<16} {'post_eplb(med)':<15} "
            f"{'post_wf(mean)':<14} {'post_wf(med)':<13}"
        )
        print("-" * 131)

        for r in by_input_len[input_len]:
            config = r["config"]
            stats = r.get("imbalance_stats", {})
            bench = r.get("bench") or {}
            lat = bench.get("latency", None)
            tps = bench.get("overall_throughput", None)
            lat_s = f"{float(lat):.3f}" if lat is not None else "N/A"
            tps_s = f"{float(tps):.1f}" if tps is not None else "N/A"

            def _fmt(stage_name: str) -> Tuple[str, str]:
                s = stats.get(stage_name, {})
                if s and s.get("mean"):
                    return f"{s['mean']:.4f}x", f"{s['median']:.4f}x"
                return "N/A", "N/A"

            pre_mean, pre_med = _fmt("pre_eplb")
            post_mean, post_med = _fmt("post_eplb")
            wf_mean, wf_med = _fmt("post_waterfill")

            print(
                f"{config:<22} {lat_s:<10} {tps_s:<12} "
                f"{pre_mean:<15} {pre_med:<14} "
                f"{post_mean:<16} {post_med:<15} "
                f"{wf_mean:<14} {wf_med:<13}"
            )

    # ── Per-layer breakdown ──
    if args.per_layer:
        print("\n" + "=" * 100)
        print("PER-LAYER IMBALANCE BREAKDOWN")
        print("=" * 100)

        for r in all_results:
            config = r["config"]
            input_len = r["input_len"]
            stats = r.get("imbalance_stats", {})

            print(f"\n--- {config} | input_len={input_len} ---")
            for stage, stage_stats in sorted(stats.items()):
                per_layer = stage_stats.get("per_layer", {})
                if not per_layer:
                    continue
                print(f"  {stage}:")
                for layer_id, val in sorted(per_layer.items(), key=lambda x: int(x[0])):
                    print(f"    layer {layer_id:>3s}: {val:.4f}x")

    # ── Improvement analysis ──
    print("\n" + "=" * 100)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 100)

    for input_len in sorted(by_input_len.keys()):
        print(f"\n=== input_len={input_len} ===")

        results_by_config = {}
        for r in by_input_len[input_len]:
            results_by_config[r["config"]] = r.get("avg_imbalance", {})

        # EPLB improvement
        for cfg_name in ["waterfill_eplb", "baseline_eplb"]:
            avg = results_by_config.get(cfg_name, {})
            if avg.get("pre_eplb") and avg.get("post_eplb"):
                pre = avg["pre_eplb"]
                post = avg["post_eplb"]
                improvement = (pre - post) / pre * 100
                print(
                    f"  {cfg_name} EPLB reduction: {pre:.4f}x -> {post:.4f}x ({improvement:+.2f}%)"
                )

        # Waterfill improvement over EPLB
        wf_eplb = results_by_config.get("waterfill_eplb", {})
        if wf_eplb.get("post_eplb") and wf_eplb.get("post_waterfill"):
            post_eplb = wf_eplb["post_eplb"]
            post_wf = wf_eplb["post_waterfill"]
            improvement = (post_eplb - post_wf) / post_eplb * 100
            print(
                f"  Waterfill improvement over EPLB: {post_eplb:.4f}x -> {post_wf:.4f}x ({improvement:+.2f}%)"
            )

        # Waterfill without EPLB
        wf_no_eplb = results_by_config.get("waterfill_no_eplb", {})
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
