#!/usr/bin/env python3
"""
Benchmark for DeepEP Waterfill on EP8/EP16/EP32.

Measures throughput with bench_one_batch_server across
baseline, waterfill, eplb, and eplb_waterfill modes.

For baseline mode, uses a separate sglang installation (--baseline-sglang-dir)
to get a true A/B comparison between codebases.

Usage (run from node 0 inside sglang_lb container):
  # EP16 - baseline vs waterfill (two repos)
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 16 \
      --modes baseline,waterfill \
      --baseline-sglang-dir /lustre/.../sglang_baseline_98a107d

  # EP16 - all 4 modes
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 16 \
      --modes baseline,waterfill,eplb,eplb_waterfill \
      --baseline-sglang-dir /lustre/.../sglang_baseline_98a107d \
      --init-expert-location /lustre/.../ep16_logical_count.pt

  # EP16 - repeat 3 times for variance measurement
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 16 \
      --modes baseline,waterfill --repeat 3 \
      --baseline-sglang-dir /lustre/.../sglang_baseline_98a107d

  # EP8 - accuracy only (MMLU), all 4 modes
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 8 \
      --modes baseline,waterfill,eplb,eplb_waterfill \
      --accuracy-only \
      --baseline-sglang-dir /lustre/.../sglang_baseline_98a107d \
      --init-expert-location /lustre/.../ep8_logical_count.pt

  # EP16 - perf + accuracy together
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 16 \
      --modes baseline,waterfill --run-accuracy \
      --baseline-sglang-dir /lustre/.../sglang_baseline_98a107d
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Cluster config
NODE_IPS = {
    8: ["10.6.131.5"],
    16: ["10.6.131.5", "10.6.131.6"],
}
DIST_INIT_PORT = 20000
MODEL_PATH = "/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3"
CONTAINER = "sglang_lb"

# Wrapper script that sets ulimit -l unlimited before exec python3.
# Required for multi-node NVSHMEM IBGDA transport (memlock limit fix).
LAUNCH_WRAPPER = (
    "/lustre/raplab/client/xutingz/workspace/bench/waterfill/launch_sglang.sh"
)

# EP config: actual_tp/actual_dp are what sglang --tp/--dp-size receive.
# For EP8:  single node, 8 GPUs, tp=8, dp=8 (dp_attention)
# For EP16: 2 nodes, tp=16, dp=16 (dp_attention)
# For EP32: 4 nodes, tp=16, dp=32 (dp_attention), moe_dense_tp_size=1
EP_CONFIG = {
    8: {"actual_tp": 8, "actual_dp": 8, "nnodes": 1},
    16: {"actual_tp": 16, "actual_dp": 16, "nnodes": 2},
    32: {"actual_tp": 16, "actual_dp": 32, "nnodes": 4, "moe_dense_tp_size": 1},
}


@dataclass(frozen=True)
class BenchCase:
    name: str
    local_batch_size: int  # per-rank batch size
    input_len: int
    output_len: int


# Benchmark cases: output_len=1, local_bs is per DP rank.
# Global batch size = local_bs * dp_size (computed at runtime).
# deepep_mode = normal for all cases.
BENCH_CASES = [
    BenchCase("bs128_il512", 128, 512, 1),
    BenchCase("bs64_il1024", 64, 1024, 1),
    BenchCase("bs32_il2048", 32, 2048, 1),
    BenchCase("bs16_il4096", 16, 4096, 1),
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
        kill_cmds += (
            "; rm -f /dev/shm/nccl* 2>/dev/null" "; rm -f /dev/shm/nvshmem* 2>/dev/null"
        )
        if ip == node_ips[0]:
            # Local node: run directly (we are inside the container)
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


def pip_install_sglang(sglang_dir: str, node_ips: List[str]) -> None:
    """Install sglang from the given directory on all nodes (editable, no-deps)."""
    install_cmd = f"cd {sglang_dir} && pip install -e 'python[dev]' --no-deps -q"
    print(f"  Installing sglang from {sglang_dir} on all nodes...", flush=True)

    # Local node (node 0) — we are inside the container
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


def pip_install_sglang_local(sglang_dir: str) -> None:
    """Install sglang from the given directory on local node only (for bench client)."""
    install_cmd = f"cd {sglang_dir} && pip install -e 'python[dev]' --no-deps -q"
    subprocess.run(
        ["bash", "-c", install_cmd],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_server(
    *,
    ep: int,
    node_ips: List[str],
    enable_waterfill: bool = False,
    init_expert_location: Optional[str] = None,
    disable_cuda_graph: bool = False,
    log_dir: Path,
    dist_init_port: int = DIST_INIT_PORT,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """Launch sglang server across nodes. Returns the local (node 0) server process."""
    cfg = EP_CONFIG[ep]
    dist_init_addr = f"{node_ips[0]}:{dist_init_port}"
    use_wrapper = cfg["nnodes"] > 1 and os.path.isfile(LAUNCH_WRAPPER)
    if cfg["nnodes"] > 1 and not use_wrapper:
        print(
            f"  WARNING: Multi-node but wrapper not found at {LAUNCH_WRAPPER}. "
            f"NVSHMEM may fail without ulimit -l unlimited.",
            flush=True,
        )

    def _build_server_cmd(node_rank: int) -> List[str]:
        if use_wrapper:
            cmd = [LAUNCH_WRAPPER, "-m", "sglang.launch_server"]
        else:
            cmd = [sys.executable, "-m", "sglang.launch_server"]
        cmd += [
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
        if cfg.get("moe_dense_tp_size") is not None:
            cmd.extend(["--moe-dense-tp-size", str(cfg["moe_dense_tp_size"])])
        return cmd

    env_vars = (
        "export SGLANG_LOG_MS=1; "
        "export NCCL_DEBUG=WARN; "
        "export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0; "
    )
    if extra_env:
        for k, v in extra_env.items():
            env_vars += f"export {k}={v}; "

    # Launch worker nodes (rank 1+) via SSH
    for rank in range(1, cfg["nnodes"]):
        ip = node_ips[rank]
        worker_cmd = _build_server_cmd(rank)
        log_file = log_dir / f"server_node{rank}.log"
        docker_cmd = env_vars + " ".join(worker_cmd)
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
        time.sleep(5)

    # Launch node 0 locally (inside the container)
    if cfg["nnodes"] > 1:
        time.sleep(3)
    local_cmd = _build_server_cmd(0)
    log_file = log_dir / "server_node0.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_file.open("w")
    env = os.environ.copy()
    env["SGLANG_LOG_MS"] = "1"
    env["NCCL_DEBUG"] = "WARN"
    env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        local_cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_f = log_f  # type: ignore
    return proc


def run_mmlu_eval(
    *,
    base_url: str,
    num_examples: Optional[int] = None,
    num_threads: int = 512,
) -> Optional[dict]:
    """Run MMLU evaluation and return metrics dict with 'score' key."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "99"

    cmd = [
        sys.executable,
        "-m",
        "sglang.test.run_eval",
        "--base-url",
        base_url,
        "--eval-name",
        "mmlu",
        "--num-threads",
        str(num_threads),
    ]
    if num_examples is not None:
        cmd.extend(["--num-examples", str(num_examples)])

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            timeout=3600,
            capture_output=True,
            text=True,
        )
        # Parse score from stdout: "Score: 0.xxx"
        for line in result.stdout.split("\n"):
            if line.startswith("Score:"):
                score = float(line.split(":")[1].strip())
                return {"score": score, "stdout": result.stdout}
        # Fallback: try to find the JSON results file
        for line in result.stdout.split("\n"):
            if "Writing results to" in line:
                json_path = line.split("Writing results to")[-1].strip()
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        return json.load(f)
        print(f"  MMLU: could not parse score from output", flush=True)
        print(f"  stdout: {result.stdout[-500:]}", flush=True)
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  MMLU FAILED: {e}", flush=True)
        if hasattr(e, "stdout") and e.stdout:
            print(f"  stdout: {e.stdout[-500:]}", flush=True)
        if hasattr(e, "stderr") and e.stderr:
            print(f"  stderr: {e.stderr[-500:]}", flush=True)
        return None


def run_bench(
    *,
    base_url: str,
    case: BenchCase,
    result_file: Path,
    dp_size: int = 1,
    dataset_path: Optional[str] = None,
) -> Optional[dict]:
    """Run bench_one_batch_server and return parsed result."""
    global_batch_size = case.local_batch_size * dp_size
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "99"  # client on CPU

    cmd = [
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
        str(case.input_len),
        "--output-len",
        str(case.output_len),
        "--dataset-name",
        "random",
        "--result-filename",
        str(result_file),
        "--no-append-to-github-summary",
    ]
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])

    result_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, env=env, check=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  FAILED: {e}", flush=True)
        return None

    # Parse result
    if result_file.exists():
        lines = result_file.read_text().strip().split("\n")
        if lines:
            return json.loads(lines[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Waterfill benchmark for EP8/EP16/EP32"
    )
    parser.add_argument("--ep", type=int, required=True, choices=[8, 16, 32])
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,waterfill",
        help="Comma-separated modes: baseline,waterfill,eplb,eplb_waterfill",
    )
    parser.add_argument(
        "--init-expert-location",
        type=str,
        default=None,
        help="EPLB .pt file for eplb/eplb_waterfill modes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/lustre/raplab/client/xutingz/workspace/bench/waterfill/waterfill_bench",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/lustre/raplab/client/xutingz/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable CUDA graph"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Override bench cases: 'local_bs:il' comma-separated, "
        "e.g. '128:512,64:1024'",
    )
    parser.add_argument(
        "--baseline-sglang-dir",
        type=str,
        default=None,
        help="Path to baseline sglang repo (for baseline mode). "
        "If not set, baseline uses the same code as waterfill.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each mode (for variance measurement)",
    )
    parser.add_argument(
        "--run-accuracy",
        action="store_true",
        help="Run MMLU accuracy eval for each mode",
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Skip performance benchmark, only run accuracy eval",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=2000,
        help="Number of MMLU examples (default: 2000; seed=0 for reproducibility)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=512, help="Number of threads for MMLU eval"
    )
    parser.add_argument(
        "--skip-jit-warmup",
        action="store_true",
        help="Skip JIT cache pre-warm (use when caches are already populated)",
    )
    args = parser.parse_args()

    if args.accuracy_only:
        args.run_accuracy = True

    ep = args.ep
    cfg = EP_CONFIG[ep]
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

    dp_size = cfg["actual_dp"]

    # Determine sglang directories for each mode.
    # The script itself lives in the optimized repo; use its parent as default.
    optimized_sglang_dir = str(Path(__file__).resolve().parents[2])
    baseline_sglang_dir = args.baseline_sglang_dir or optimized_sglang_dir

    def _sglang_dir_for_mode(mode: str) -> str:
        """Return the sglang repo path to use for a given mode."""
        if mode == "baseline":
            return baseline_sglang_dir
        return optimized_sglang_dir

    print(f"\nEP{ep} Benchmark Config:", flush=True)
    print(f"  Nodes: {node_ips}", flush=True)
    print(f"  TP={cfg['actual_tp']}, DP={dp_size}, nnodes={cfg['nnodes']}", flush=True)
    print(f"  Modes: {modes}", flush=True)
    print(f"  Repeat: {args.repeat}", flush=True)
    print(f"  Cases: {[c.name for c in cases]}", flush=True)
    print(f"  CUDA graph: disabled", flush=True)
    print(f"  DeepEP mode: normal", flush=True)
    print(f"  Baseline sglang: {baseline_sglang_dir}", flush=True)
    print(f"  Optimized sglang: {optimized_sglang_dir}", flush=True)
    print(
        f"  Accuracy: {'yes' if args.run_accuracy else 'no'}"
        f"{' (accuracy-only)' if args.accuracy_only else ''}",
        flush=True,
    )
    if args.run_accuracy:
        print(f"  MMLU examples: {args.num_examples or 'all'} (seed=0)", flush=True)
    print(f"  Output dir: {out_dir}\n", flush=True)

    # ── JIT Cache Pre-Warm ──────────────────────────────────────────────
    # DeepGEMM JIT-compiles ~103 GEMM kernels on the first server run and
    # caches them at /root/.cache/deep_gemm/cache/.  If we skip this step,
    # the first benchmark mode bears all compilation overhead and looks ~2x
    # slower than the second mode (which reuses the disk cache).
    # Pre-warming ensures every mode starts with a fully-populated cache.
    #
    # We install the optimized repo for warmup (DeepGEMM kernels are the same
    # regardless of the waterfill flag or baseline vs optimized code).
    if args.skip_jit_warmup:
        print(f"\n{'='*70}", flush=True)
        print(f" JIT CACHE PRE-WARM SKIPPED (--skip-jit-warmup)", flush=True)
        print(f"{'='*70}\n", flush=True)
        kill_servers(node_ips)
    else:
        print(f"\n{'='*70}", flush=True)
        print(f" JIT CACHE PRE-WARM (server + one warmup request)", flush=True)
        print(f"{'='*70}\n", flush=True)

        kill_servers(node_ips)
        pip_install_sglang(optimized_sglang_dir, node_ips)
        warmup_log_dir = out_dir / "_jit_warmup" / "logs"
        warmup_log_dir.mkdir(parents=True, exist_ok=True)
        warmup_proc = launch_server(
            ep=ep,
            node_ips=node_ips,
            enable_waterfill=False,
            init_expert_location=None,
            disable_cuda_graph=disable_cuda_graph,
            log_dir=warmup_log_dir,
            dist_init_port=DIST_INIT_PORT + 99,  # avoid collision with real runs
        )
        try:
            warmup_url = f"http://{node_ips[0]}:30000"
            print("[warmup] Waiting for server...", flush=True)
            wait_server(warmup_url, timeout_s=1800)
            print(
                "[warmup] Server ready. JIT cache pre-warm complete (server-only).\n",
                flush=True,
            )
        finally:
            try:
                os.killpg(warmup_proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                warmup_proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(warmup_proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            try:
                warmup_proc._log_f.close()  # type: ignore
            except Exception:
                pass
            kill_servers(node_ips)

    all_results: Dict[str, Dict[str, dict]] = {}
    # For repeat > 1, collect all runs: {mode: {case: [result1, result2, ...]}}
    all_runs: Dict[str, Dict[str, List[dict]]] = {}
    accuracy_results: Dict[str, dict] = {}  # mode -> {score, ...}

    for mode_idx, mode in enumerate(modes):
        enable_waterfill = mode in (
            "waterfill",
            "eplb_waterfill",
        )  # V2 uses env var only, no --enable-deepep-waterfill
        init_expert_loc = (
            args.init_expert_location
            if mode in ("eplb", "eplb_waterfill", "eplb_waterfill_v2")
            else None
        )

        if (
            mode in ("eplb", "eplb_waterfill", "eplb_waterfill_v2")
            and not args.init_expert_location
        ):
            print(f"SKIP {mode}: --init-expert-location required", flush=True)
            continue

        mode_extra_env: Optional[Dict[str, str]] = None
        if mode == "eplb_waterfill_v2":
            mode_extra_env = {"SGLANG_WATERFILL_V2": "1"}

        sglang_dir = _sglang_dir_for_mode(mode)
        mode_runs: Dict[str, List[dict]] = {}

        for run_i in range(args.repeat):
            run_label = (
                f"{mode}"
                if args.repeat == 1
                else f"{mode} (run {run_i+1}/{args.repeat})"
            )

            print(f"\n{'='*70}", flush=True)
            print(
                f" MODE: {run_label} | EP{ep} | waterfill={enable_waterfill}",
                flush=True,
            )
            print(f" sglang: {sglang_dir}", flush=True)
            if init_expert_loc:
                print(f" EPLB: {init_expert_loc}", flush=True)
            print(f"{'='*70}\n", flush=True)

            mode_dir = out_dir / mode / (f"run{run_i}" if args.repeat > 1 else "")
            log_dir = mode_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Kill any stale servers
            kill_servers(node_ips)

            # Install the correct sglang version on all nodes
            pip_install_sglang(sglang_dir, node_ips)

            # Use a different dist-init port per mode to avoid port conflicts
            mode_port = DIST_INIT_PORT + mode_idx

            print(
                f"[{run_label}] Launching server (dist port {mode_port})...", flush=True
            )
            proc = launch_server(
                ep=ep,
                node_ips=node_ips,
                enable_waterfill=enable_waterfill,
                init_expert_location=init_expert_loc,
                disable_cuda_graph=disable_cuda_graph,
                log_dir=log_dir,
                dist_init_port=mode_port,
                extra_env=mode_extra_env,
            )

            try:
                base_url = f"http://{node_ips[0]}:30000"
                print(f"[{run_label}] Waiting for server at {base_url}...", flush=True)
                wait_server(base_url, timeout_s=1800)
                print(f"[{run_label}] Server ready!\n", flush=True)

                # Always use the optimized repo's bench_one_batch_server as the
                # bench client.  The baseline repo's client has a bug where
                # skip_token_capacity_threshold is not multiplied by dp_size,
                # causing it to skip valid benchmark cases.  The server process
                # has already loaded all modules into memory, so reinstalling
                # on node 0 only affects the bench client subprocess.
                if sglang_dir != optimized_sglang_dir:
                    print(
                        f"[{run_label}] Switching local node to optimized repo for bench client...",
                        flush=True,
                    )
                    pip_install_sglang_local(optimized_sglang_dir)

                # ── Performance benchmark ──
                if not args.accuracy_only:
                    for case in cases:
                        global_bs = case.local_batch_size * dp_size
                        print(
                            f"[{run_label}] Running {case.name} (local_bs={case.local_batch_size}, "
                            f"global_bs={global_bs}, il={case.input_len}, ol={case.output_len})...",
                            flush=True,
                        )
                        result_file = mode_dir / f"result_{case.name}.jsonl"
                        result = run_bench(
                            base_url=base_url,
                            case=case,
                            result_file=result_file,
                            dp_size=dp_size,
                            dataset_path=args.dataset_path,
                        )
                        if result:
                            mode_runs.setdefault(case.name, []).append(result)
                            in_tp = result.get("input_throughput", 0)
                            out_tp = result.get("output_throughput", 0)
                            lat = result.get("latency", 0)
                            print(
                                f"  -> input_tp={in_tp:.1f} tok/s, "
                                f"output_tp={out_tp:.1f} tok/s, lat={lat:.2f}s",
                                flush=True,
                            )
                        else:
                            print(f"  -> SKIPPED or FAILED", flush=True)

                # ── Accuracy evaluation (MMLU) ──
                if args.run_accuracy and run_i == 0:
                    # Only run accuracy once per mode (not per repeat)
                    print(f"\n[{run_label}] Running MMLU accuracy eval...", flush=True)
                    mmlu_result = run_mmlu_eval(
                        base_url=base_url,
                        num_examples=args.num_examples,
                        num_threads=args.num_threads,
                    )
                    if mmlu_result:
                        score = mmlu_result.get("score", -1)
                        accuracy_results[mode] = mmlu_result
                        print(f"  -> MMLU score: {score:.4f}", flush=True)
                    else:
                        print(f"  -> MMLU FAILED", flush=True)

            finally:
                print(f"\n[{run_label}] Stopping server...", flush=True)
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
                print(f"[{run_label}] Done.\n", flush=True)

        # Aggregate: use last run for all_results (backward compat), keep all runs
        all_runs[mode] = mode_runs
        if mode_runs:
            all_results[mode] = {
                case_name: runs[-1] for case_name, runs in mode_runs.items()
            }

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

    # Rows: latency
    print("\n  Latency (s):", flush=True)
    for case_name in sorted(all_case_names):
        row = f"  {case_name:<18}"
        vals = {}
        for mode in modes:
            if mode in all_results and case_name in all_results[mode]:
                val = all_results[mode][case_name].get("latency", 0)
                row += f"| {val:>18.3f}  "
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
        "repeat": args.repeat,
        "baseline_sglang_dir": baseline_sglang_dir,
        "optimized_sglang_dir": optimized_sglang_dir,
        "results": all_results,
        "accuracy": (
            {
                mode: {"score": r.get("score", -1)}
                for mode, r in accuracy_results.items()
            }
            if accuracy_results
            else {}
        ),
    }
    # Include per-run data when repeat > 1
    if args.repeat > 1:
        summary["all_runs"] = {
            mode: {
                case_name: [r for r in runs]
                for case_name, runs in mode_runs_data.items()
            }
            for mode, mode_runs_data in all_runs.items()
        }
        # Print per-run variance
        print(f"\n  Per-Run Details (input_throughput tok/s):", flush=True)
        for mode in modes:
            if mode not in all_runs:
                continue
            for case_name in sorted(all_runs[mode].keys()):
                runs = all_runs[mode][case_name]
                vals = [r.get("input_throughput", 0) for r in runs]
                if len(vals) > 1:
                    avg = sum(vals) / len(vals)
                    mn, mx = min(vals), max(vals)
                    spread = (mx - mn) / avg * 100 if avg > 0 else 0
                    vals_str = ", ".join(f"{v:.1f}" for v in vals)
                    print(
                        f"  {mode}/{case_name}: [{vals_str}]  "
                        f"avg={avg:.1f}  spread={spread:.1f}%",
                        flush=True,
                    )

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_file}", flush=True)

    # Print accuracy results
    if accuracy_results:
        print(f"\n{'='*80}", flush=True)
        print(f" ACCURACY: EP{ep} MMLU Scores", flush=True)
        print(f"{'='*80}\n", flush=True)
        for mode in modes:
            if mode in accuracy_results:
                score = accuracy_results[mode].get("score", -1)
                print(f"  {mode:<20} {score:.4f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
