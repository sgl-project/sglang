"""
Benchmark SGLang cache-eviction policies across multiple memory configurations.

This runner:
- launches a fresh SGLang server for each experiment
- waits for readiness on /v1/models
- runs `python -m sglang.bench_serving`
- saves benchmark stdout and server logs under ./output
- uses one GPU + one port per worker slot, so two experiments can run in parallel

Example:
    python3 benchmark_eviction_mem_sweep.py --request-rate 16
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


DEFAULT_POLICIES = ["lru", "marconi", "seglen"]
DEFAULT_MEM_FRACTIONS = [0.3, 0.5, 0.7, 1.0]
DEFAULT_GPUS = ["0", "1"]
DEFAULT_PORTS = [30000, 30001]


@dataclass(frozen=True)
class Experiment:
    policy: str
    mem_fraction: Optional[float]

    @property
    def slug(self) -> str:
        mem_str = "auto" if self.mem_fraction is None else str(self.mem_fraction).replace(".", "p")
        return f"{self.policy}_mem{mem_str}"

    @property
    def mem_label(self) -> str:
        return "auto" if self.mem_fraction is None else f"{self.mem_fraction:.2f}"


@dataclass(frozen=True)
class WorkerSlot:
    gpu_id: str
    port: int


@dataclass
class ExperimentResult:
    policy: str
    mem_fraction: Optional[float]
    gpu_id: str
    port: int
    status: str
    start_time: float
    end_time: float
    benchmark_returncode: Optional[int]
    server_returncode: Optional[int]
    benchmark_log: str
    server_log: str
    metadata_json: str
    error: Optional[str] = None


def parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_mem_fractions(raw: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "auto":
            out.append(None)
        else:
            out.append(float(item))
    return out


def parse_csv_strs(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def wait_for_server_ready(host: str, port: int, process: subprocess.Popen, timeout_s: int) -> None:
    url = f"http://{host}:{port}/v1/models"
    headers = {"Authorization": "Bearer None"}
    start = time.time()
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    time.sleep(5)
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Server at {url} did not become ready within {timeout_s}s")
        time.sleep(1)


def terminate_process_tree(process: subprocess.Popen, timeout_s: int = 30) -> Optional[int]:
    if process.poll() is not None:
        return process.returncode

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()

    start = time.time()
    while time.time() - start < timeout_s:
        rc = process.poll()
        if rc is not None:
            return rc
        time.sleep(1)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait(timeout=10)


def build_server_cmd(exp: Experiment, slot: WorkerSlot, args: argparse.Namespace) -> List[str]:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        args.model_path,
        "--trust-remote-code",
        "--mamba-scheduler-strategy",
        args.mamba_scheduler_strategy,
        "--host",
        args.host,
        "--port",
        str(slot.port),
    ]

    if exp.mem_fraction is not None:
        cmd += [
            "--mem-fraction-static",
            f"{exp.mem_fraction:.2f}",
        ]

    if exp.policy in ("marconi", "seglen"):
        cmd += [
            "--radix-eviction-policy",
            exp.policy,
            "--marconi-eff-weight",
            str(args.marconi_eff_weight),
        ]

    return cmd


def build_benchmark_cmd(slot: WorkerSlot, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(slot.port),
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        args.dataset_path,
        "--num-prompts",
        str(args.num_prompts),
        "--sharegpt-output-len",
        str(args.sharegpt_output_len),
        "--request-rate",
        str(args.request_rate),
        "--seed",
        str(args.seed),
    ]


def run_experiment(exp: Experiment, slot: WorkerSlot, args: argparse.Namespace, output_dir: Path) -> ExperimentResult:
    start_time = time.time()
    benchmark_log = output_dir / f"bench_{exp.slug}.log"
    server_log = output_dir / f"server_{exp.slug}.log"
    metadata_json = output_dir / f"run_{exp.slug}.json"

    server_cmd = build_server_cmd(exp, slot, args)
    bench_cmd = build_benchmark_cmd(slot, args)

    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = slot.gpu_id

    result = ExperimentResult(
        policy=exp.policy,
        mem_fraction=exp.mem_fraction,
        gpu_id=slot.gpu_id,
        port=slot.port,
        status="running",
        start_time=start_time,
        end_time=start_time,
        benchmark_returncode=None,
        server_returncode=None,
        benchmark_log=str(benchmark_log),
        server_log=str(server_log),
        metadata_json=str(metadata_json),
    )

    server_proc: Optional[subprocess.Popen] = None
    server_log_f = None
    try:
        server_log_f = open(server_log, "w")
        server_log_f.write("# Server command:\n")
        server_log_f.write(" ".join(server_cmd) + "\n\n")
        server_log_f.flush()

        server_proc = subprocess.Popen(
            server_cmd,
            stdout=server_log_f,
            stderr=subprocess.STDOUT,
            env=server_env,
            cwd=os.getcwd(),
            start_new_session=True,
            text=True,
        )

        wait_for_server_ready(args.host, slot.port, server_proc, args.server_ready_timeout)

        with open(benchmark_log, "w") as bench_log_f:
            bench_log_f.write("# Benchmark command:\n")
            bench_log_f.write(" ".join(bench_cmd) + "\n\n")
            bench_log_f.flush()

            bench_completed = subprocess.run(
                bench_cmd,
                stdout=bench_log_f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                text=True,
            )
            result.benchmark_returncode = bench_completed.returncode
            if bench_completed.returncode != 0:
                raise RuntimeError(
                    f"Benchmark failed with code {bench_completed.returncode}"
                )

        result.status = "completed"
        return result
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
        return result
    finally:
        if server_proc is not None:
            result.server_returncode = terminate_process_tree(server_proc)
        if server_log_f is not None:
            server_log_f.flush()
            server_log_f.close()
        result.end_time = time.time()
        metadata_json.write_text(json.dumps(asdict(result), indent=2) + "\n")


def worker_loop(
    slot: WorkerSlot,
    work_q: "queue.Queue[Experiment]",
    args: argparse.Namespace,
    output_dir: Path,
    results: List[ExperimentResult],
    results_lock: threading.Lock,
) -> None:
    while True:
        try:
            exp = work_q.get_nowait()
        except queue.Empty:
            return

        print(
            f"[worker gpu={slot.gpu_id} port={slot.port}] "
            f"starting {exp.policy} mem_fraction={exp.mem_label}",
            flush=True,
        )
        result = run_experiment(exp, slot, args, output_dir)
        with results_lock:
            results.append(result)
        print(
            f"[worker gpu={slot.gpu_id} port={slot.port}] "
            f"{result.status} {exp.policy} mem_fraction={exp.mem_label}",
            flush=True,
        )
        work_q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang eviction policies across memory configurations."
    )
    parser.add_argument("--request-rate", type=float, required=True, help="Benchmark request rate.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="multi_group_shared_prefix_dataset_long.json",
        help="Dataset path for bench_serving.",
    )
    parser.add_argument("--num-prompts", type=int, default=2300)
    parser.add_argument("--sharegpt-output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--mem-fractions",
        type=str,
        default="0.3,0.5,0.7,1.0",
        help="Comma-separated mem-fraction-static values. Use 'auto' to omit the flag.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="lru,marconi,seglen",
        help="Comma-separated policies to test.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated CUDA_VISIBLE_DEVICES values for worker slots.",
    )
    parser.add_argument(
        "--ports",
        type=str,
        default="30000,30001",
        help="Comma-separated ports, one per worker slot.",
    )
    parser.add_argument("--marconi-eff-weight", type=float, default=0.85)
    parser.add_argument("--mamba-scheduler-strategy", type=str, default="extra_buffer")
    parser.add_argument(
        "--server-ready-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for each server to become ready.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where benchmark output and server logs will be written.",
    )
    args = parser.parse_args()

    policies = parse_csv_strs(args.policies)
    mem_fractions = parse_csv_mem_fractions(args.mem_fractions)
    gpus = parse_csv_strs(args.gpus)
    ports = parse_csv_ints(args.ports)

    if len(gpus) != len(ports):
        raise ValueError("--gpus and --ports must have the same number of entries.")
    if not policies:
        raise ValueError("No policies specified.")
    if not mem_fractions:
        raise ValueError("No mem fractions specified.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        Experiment(policy=policy, mem_fraction=mem_fraction)
        for mem_fraction in mem_fractions
        for policy in policies
    ]

    manifest = {
        "request_rate": args.request_rate,
        "dataset_path": args.dataset_path,
        "num_prompts": args.num_prompts,
        "sharegpt_output_len": args.sharegpt_output_len,
        "seed": args.seed,
        "model_path": args.model_path,
        "host": args.host,
        "policies": policies,
        "mem_fractions": mem_fractions,
        "mem_fraction_labels": [
            "auto" if mf is None else f"{mf:.2f}" for mf in mem_fractions
        ],
        "gpus": gpus,
        "ports": ports,
        "experiments": [asdict(x) for x in experiments],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    work_q: "queue.Queue[Experiment]" = queue.Queue()
    for exp in experiments:
        work_q.put(exp)

    results: List[ExperimentResult] = []
    results_lock = threading.Lock()
    threads: List[threading.Thread] = []
    slots = [WorkerSlot(gpu_id=g, port=p) for g, p in zip(gpus, ports)]

    for slot in slots:
        t = threading.Thread(
            target=worker_loop,
            args=(slot, work_q, args, output_dir, results, results_lock),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    results_sorted = sorted(
        results,
        key=lambda x: (
            float("inf") if x.mem_fraction is None else x.mem_fraction,
            x.policy,
        ),
    )
    (output_dir / "results.json").write_text(
        json.dumps([asdict(x) for x in results_sorted], indent=2) + "\n"
    )

    completed = sum(1 for r in results_sorted if r.status == "completed")
    failed = sum(1 for r in results_sorted if r.status != "completed")
    print(
        f"Finished {len(results_sorted)} experiments: {completed} completed, {failed} failed. "
        f"Logs written to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
