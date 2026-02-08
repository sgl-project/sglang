#!/usr/bin/env python3
"""
Multi-node benchmark for DeepEP Waterfill on EP16/EP32.

Measures decode throughput with bench_one_batch_server across
baseline (no waterfill) and waterfill modes.

Usage (run from node 0 inside sglang_eplb container):
  # EP16 (2 nodes)
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 16

  # EP32 (4 nodes)
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 32

  # EP16 waterfill only
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 16 --modes waterfill

  # EP16 with EPLB + waterfill
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 16 --modes baseline,waterfill,eplb_waterfill \
      --init-expert-location /root/xutingz/output/eplb/ep16_logical_count.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Cluster config
NODE_IPS = {
    16: ["10.6.131.20", "10.6.131.21"],
    32: ["10.6.131.20", "10.6.131.21", "10.6.131.22", "10.6.131.23"],
}
DIST_INIT_PORT = 20000
MODEL_PATH = "/raid/model/DeepSeek-R1"
CONTAINER = "sglang_eplb"

# EP config: label -> (actual_tp, actual_dp, nnodes)
EP_CONFIG = {
    16: {"tp": 16, "dp": 16, "actual_tp": 16, "actual_dp": 16, "nnodes": 2},
    32: {"tp": 32, "dp": 32, "actual_tp": 16, "actual_dp": 2, "nnodes": 4},
}


@dataclass(frozen=True)
class BenchCase:
    name: str
    batch_size: int  # global batch size
    input_len: int
    output_len: int


# Benchmark cases: decode-heavy (output_len=1 to measure pure decode throughput)
BENCH_CASES = [
    BenchCase("bs128_il512", 128, 512, 1),
    BenchCase("bs128_il1024", 128, 1024, 1),
    BenchCase("bs256_il1024", 256, 1024, 1),
    BenchCase("bs256_il2048", 256, 2048, 1),
    BenchCase("bs512_il1024", 512, 1024, 1),
]


def wait_server(base_url: str, timeout_s: int = 1800) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"Server not ready after {timeout_s}s")


def kill_servers(node_ips: List[str]) -> None:
    """Kill all sglang server processes on all nodes.

    Uses specific patterns to avoid killing the benchmark script itself.
    """
    # Patterns that match server/worker processes but NOT the benchmark script
    kill_patterns = [
        "sglang.launch_server",
        "sglang::scheduler",
        "sglang::data_pa",
        "sglang::detoken",
        "sglang::nccl",
        "sglang.srt",
    ]
    for ip in node_ips:
        kill_cmds = "; ".join(
            f"pkill -9 -f '{pat}' 2>/dev/null" for pat in kill_patterns
        )
        kill_cmds += "; pkill -9 -f bench_one_batch 2>/dev/null"
        # Also clean up stale NCCL/NVSHMEM shared memory
        kill_cmds += (
            "; rm -f /dev/shm/nccl* 2>/dev/null"
            "; rm -f /dev/shm/nvshmem* 2>/dev/null"
        )
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"root@{ip}",
             f"docker exec {CONTAINER} bash -c '{kill_cmds}'"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    # Wait long enough for NCCL resources to fully release
    time.sleep(15)


def launch_server(
    *,
    ep: int,
    node_ips: List[str],
    enable_waterfill: bool = False,
    init_expert_location: Optional[str] = None,
    disable_cuda_graph: bool = False,
    log_dir: Path,
    dist_init_port: int = DIST_INIT_PORT,
) -> subprocess.Popen:
    """Launch sglang server across multiple nodes. Returns the local server process."""
    cfg = EP_CONFIG[ep]
    dist_init_addr = f"{node_ips[0]}:{dist_init_port}"

    def _build_server_cmd(node_rank: int) -> List[str]:
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", MODEL_PATH,
            "--trust-remote-code",
            "--host", "0.0.0.0", "--port", "30000",
            "--tp", str(cfg["actual_tp"]),
            "--dp-size", str(cfg["actual_dp"]),
            "--moe-a2a-backend", "deepep",
            "--deepep-mode", "auto",
            "--chunked-prefill-size", "-1",
            "--disable-radix-cache",
            "--max-prefill-tokens", "8192",
            "--max-running-requests", "2048",
            "--load-balance-method", "round_robin",
            "--log-level", "info",
            "--watchdog-timeout", "600",
            "--mem-fraction-static", "0.75",
            "--skip-server-warmup",
            "--dist-init-addr", dist_init_addr,
            "--nnodes", str(cfg["nnodes"]),
            "--node-rank", str(node_rank),
        ]
        if cfg["actual_dp"] > 1:
            cmd.append("--enable-dp-attention")
        if not disable_cuda_graph:
            cmd.extend(["--cuda-graph-max-bs", "128"])
        else:
            cmd.append("--disable-cuda-graph")
        if enable_waterfill:
            cmd.append("--enable-deepep-waterfill")
        if init_expert_location:
            cmd.extend(["--init-expert-location", init_expert_location])
        return cmd

    env_vars = (
        "export SGLANG_LOG_MS=1; "
        "export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0; "
        "export NVSHMEM_IB_GID_INDEX=3; "
        'export NVSHMEM_HCA_LIST="mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,'
        'mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1"; '
    )

    # Launch worker nodes (rank 1+) via SSH
    for rank in range(1, cfg["nnodes"]):
        ip = node_ips[rank]
        worker_cmd = _build_server_cmd(rank)
        log_file = log_dir / f"server_node{rank}.log"
        docker_cmd = env_vars + " ".join(worker_cmd)
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", f"root@{ip}",
            f"mkdir -p {log_dir} && "
            f"nohup docker exec {CONTAINER} bash -c '{docker_cmd}' "
            f"> {log_file} 2>&1 &"
        ]
        subprocess.Popen(ssh_cmd)
        time.sleep(2)

    # Launch node 0 locally
    time.sleep(3)
    local_cmd = _build_server_cmd(0)
    log_file = log_dir / "server_node0.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_file.open("w")
    env = os.environ.copy()
    env["SGLANG_LOG_MS"] = "1"
    env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    env["NVSHMEM_IB_GID_INDEX"] = "3"
    env["NVSHMEM_HCA_LIST"] = (
        "mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,"
        "mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1"
    )
    proc = subprocess.Popen(
        local_cmd, env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_f = log_f  # type: ignore
    return proc


def run_bench(
    *,
    base_url: str,
    case: BenchCase,
    result_file: Path,
    dataset_path: Optional[str] = None,
) -> Optional[dict]:
    """Run bench_one_batch_server and return parsed result."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "99"  # client on CPU

    cmd = [
        sys.executable, "-m", "sglang.bench_one_batch_server",
        "--model", "None",
        "--base-url", base_url,
        "--batch-size", str(case.batch_size),
        "--input-len", str(case.input_len),
        "--output-len", str(case.output_len),
        "--dataset-name", "random",
        "--result-filename", str(result_file),
        "--no-append-to-github-summary",
    ]
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])

    result_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, env=env, check=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  FAILED: {e}", flush=True)
        return None

    # Parse result
    if result_file.exists():
        lines = result_file.read_text().strip().split("\n")
        if lines:
            return json.loads(lines[-1])
    return None


def parse_decode_throughput(log_path: Path) -> Optional[float]:
    """Parse gen throughput from server log (last decode batch line)."""
    pattern = re.compile(r"gen throughput.*?:\s*([0-9.]+)")
    last_tp = None
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            m = pattern.search(line)
            if m:
                last_tp = float(m.group(1))
    return last_tp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-node waterfill benchmark for EP16/EP32"
    )
    parser.add_argument("--ep", type=int, required=True, choices=[16, 32])
    parser.add_argument(
        "--modes", type=str, default="baseline,waterfill",
        help="Comma-separated modes: baseline,waterfill,eplb_waterfill"
    )
    parser.add_argument("--init-expert-location", type=str, default=None,
                       help="EPLB .pt file for eplb_waterfill mode")
    parser.add_argument("--out-dir", type=str,
                       default="/root/xutingz/output/waterfill_bench")
    parser.add_argument("--dataset-path", type=str,
                       default="/root/xutingz/data/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                       help="Disable CUDA graph (needed for EP32 4-node)")
    parser.add_argument("--cases", type=str, default=None,
                       help="Override bench cases: 'bs:il' comma-separated, e.g. '128:1024,256:2048'")
    args = parser.parse_args()

    ep = args.ep
    node_ips = NODE_IPS[ep]
    modes = [m.strip() for m in args.modes.split(",")]
    out_dir = Path(args.out_dir) / f"ep{ep}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse custom cases if provided
    cases = BENCH_CASES
    if args.cases:
        cases = []
        for item in args.cases.split(","):
            bs, il = item.strip().split(":")
            cases.append(BenchCase(f"bs{bs}_il{il}", int(bs), int(il), 1))

    # Always disable CUDA graph for fair comparison.
    # Waterfill mode cannot use CUDA graph (DeepEP Buffer.sync() fails during
    # graph capture), so we disable it for all modes to keep the comparison fair.
    disable_cuda_graph = True

    all_results: Dict[str, Dict[str, dict]] = {}

    for mode_idx, mode in enumerate(modes):
        enable_waterfill = mode in ("waterfill", "eplb_waterfill")
        init_expert_loc = (
            args.init_expert_location
            if mode in ("eplb", "eplb_waterfill")
            else None
        )

        if mode in ("eplb", "eplb_waterfill") and not args.init_expert_location:
            print(f"SKIP {mode}: --init-expert-location required", flush=True)
            continue

        print(f"\n{'='*70}", flush=True)
        print(f" MODE: {mode} | EP{ep} | waterfill={enable_waterfill}", flush=True)
        if init_expert_loc:
            print(f" EPLB: {init_expert_loc}", flush=True)
        print(f"{'='*70}\n", flush=True)

        mode_dir = out_dir / mode
        log_dir = mode_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Kill any stale servers
        kill_servers(node_ips)

        # Launch server
        # Use a different dist-init port per mode to avoid port conflicts
        # from lingering rendezvous stores after kill
        mode_port = DIST_INIT_PORT + mode_idx

        print(f"[{mode}] Launching server (dist port {mode_port})...", flush=True)
        proc = launch_server(
            ep=ep,
            node_ips=node_ips,
            enable_waterfill=enable_waterfill,
            init_expert_location=init_expert_loc,
            disable_cuda_graph=disable_cuda_graph,
            log_dir=log_dir,
            dist_init_port=mode_port,
        )

        try:
            base_url = f"http://{node_ips[0]}:30000"
            print(f"[{mode}] Waiting for server at {base_url}...", flush=True)
            wait_server(base_url, timeout_s=1800)
            print(f"[{mode}] Server ready!\n", flush=True)

            mode_results = {}
            for case in cases:
                print(f"[{mode}] Running {case.name} (bs={case.batch_size}, "
                      f"il={case.input_len}, ol={case.output_len})...", flush=True)
                result_file = mode_dir / f"result_{case.name}.jsonl"
                result = run_bench(
                    base_url=base_url,
                    case=case,
                    result_file=result_file,
                    dataset_path=args.dataset_path,
                )
                if result:
                    mode_results[case.name] = result
                    it = result.get("input_throughput", 0)
                    ot = result.get("output_throughput", 0)
                    lat = result.get("latency", 0)
                    print(f"  -> input_tp={it:.1f} tok/s, "
                          f"output_tp={ot:.1f} tok/s, lat={lat:.2f}s", flush=True)
                else:
                    print(f"  -> SKIPPED or FAILED", flush=True)

            all_results[mode] = mode_results

        finally:
            print(f"\n[{mode}] Stopping server...", flush=True)
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
                proc._log_f.close()  # type: ignore
            except Exception:
                pass
            kill_servers(node_ips)
            print(f"[{mode}] Done.\n", flush=True)

    # Print comparison table
    print(f"\n{'='*80}", flush=True)
    print(f" RESULTS: EP{ep} Waterfill Benchmark", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Determine base and optimized modes for gain calculation
    active_modes = [m for m in modes if m in all_results]
    base_mode = active_modes[0] if active_modes else None
    opt_mode = active_modes[-1] if len(active_modes) > 1 else None

    # Header
    header = f"{'Case':<20}"
    for mode in modes:
        if mode in all_results:
            header += f"| {mode:>20} "
    if base_mode and opt_mode:
        header += f"| {'gain':>10} "
    print(header, flush=True)
    print("-" * len(header), flush=True)

    # Rows: output throughput
    print("\n  Output Throughput (tok/s):", flush=True)
    all_case_names = set()
    for mr in all_results.values():
        all_case_names.update(mr.keys())

    for case_name in sorted(all_case_names):
        row = f"  {case_name:<18}"
        vals = {}
        for mode in modes:
            if mode in all_results and case_name in all_results[mode]:
                val = all_results[mode][case_name].get("output_throughput", 0)
                row += f"| {val:>18.1f}  "
                vals[mode] = val
            else:
                row += f"| {'N/A':>18}  "
        if base_mode in vals and opt_mode in vals and vals[base_mode] > 0:
            gain = (vals[opt_mode] - vals[base_mode]) / vals[base_mode] * 100
            row += f"| {gain:>+8.1f}%  "
        print(row, flush=True)

    # Rows: input throughput
    print("\n  Input Throughput (tok/s):", flush=True)
    for case_name in sorted(all_case_names):
        row = f"  {case_name:<18}"
        vals = {}
        for mode in modes:
            if mode in all_results and case_name in all_results[mode]:
                val = all_results[mode][case_name].get("input_throughput", 0)
                row += f"| {val:>18.1f}  "
                vals[mode] = val
            else:
                row += f"| {'N/A':>18}  "
        if base_mode in vals and opt_mode in vals and vals[base_mode] > 0:
            gain = (vals[opt_mode] - vals[base_mode]) / vals[base_mode] * 100
            row += f"| {gain:>+8.1f}%  "
        print(row, flush=True)

    # Save summary
    summary = {
        "ep": ep,
        "modes": modes,
        "results": all_results,
    }
    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
