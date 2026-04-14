"""
Sequential driver to benchmark SGLang cache-eviction policies on a single GPU.

For each policy, this script:
- launches a fresh SGLang server
- waits for readiness on /v1/models
- runs `python -m sglang.bench_serving`
- saves the benchmark stdout and server log in a fresh output directory
- stops the server before moving to the next policy

Example:
    python3 benchmark_eviction_policies.py
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DEFAULT_POLICIES = ["lru", "seglen", "marconi", "marconi-fixed"]


@dataclass(frozen=True)
class Experiment:
    policy: str

    @property
    def slug(self) -> str:
        return self.policy.replace("-", "_")


@dataclass
class ExperimentResult:
    policy: str
    status: str
    start_time: float
    end_time: float
    request_rate: float
    benchmark_returncode: Optional[int]
    server_returncode: Optional[int]
    benchmark_log: str
    server_log: str
    metadata_json: str
    server_command: List[str]
    benchmark_command: List[str]
    error: Optional[str] = None


def parse_csv_strs(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def format_cmd(cmd: List[str]) -> str:
    return shlex.join(cmd)


def wait_for_server_ready(
    host: str, port: int, process: subprocess.Popen, timeout_s: int
) -> None:
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
                    time.sleep(3)
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Server at {url} did not become ready within {timeout_s}s"
            )
        time.sleep(1)


def terminate_process_tree(
    process: subprocess.Popen, timeout_s: int = 30
) -> Optional[int]:
    if process.poll() is not None:
        return process.returncode

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rc = process.poll()
        if rc is not None:
            return rc
        time.sleep(1)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait(timeout=10)


def build_server_cmd(exp: Experiment, args: argparse.Namespace) -> List[str]:
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
        str(args.port),
    ]

    if exp.policy != "lru":
        cmd += [
            "--radix-eviction-policy",
            exp.policy,
            "--marconi-eff-weight",
            str(args.marconi_eff_weight),
        ]

    return cmd


def build_benchmark_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dataset-name",
        args.dataset_name,
        "--dataset-path",
        args.dataset_path,
        "--sharegpt-output-len",
        str(args.sharegpt_output_len),
        "--request-rate",
        str(args.request_rate),
        "--seed",
        str(args.seed),
    ]

    if args.num_prompts is not None:
        cmd += ["--num-prompts", str(args.num_prompts)]

    return cmd


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def run_experiment(
    exp: Experiment,
    args: argparse.Namespace,
    output_dir: Path,
    index: int,
    total: int,
) -> ExperimentResult:
    run_dir = output_dir / f"{index:02d}_{exp.slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_log = run_dir / "benchmark.log"
    server_log = run_dir / "server.log"
    metadata_json = run_dir / "result.json"
    start_time = time.time()

    server_cmd = build_server_cmd(exp, args)
    benchmark_cmd = build_benchmark_cmd(args)

    result = ExperimentResult(
        policy=exp.policy,
        status="running",
        start_time=start_time,
        end_time=start_time,
        request_rate=args.request_rate,
        benchmark_returncode=None,
        server_returncode=None,
        benchmark_log=str(benchmark_log),
        server_log=str(server_log),
        metadata_json=str(metadata_json),
        server_command=server_cmd,
        benchmark_command=benchmark_cmd,
    )

    server_proc: Optional[subprocess.Popen] = None
    server_log_f = None

    try:
        print(f"[{index}/{total}] Starting server for policy={exp.policy}", flush=True)
        print(f"  server: {format_cmd(server_cmd)}", flush=True)

        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = args.gpu

        server_log_f = open(server_log, "w")
        server_log_f.write("# Server command\n")
        server_log_f.write(format_cmd(server_cmd) + "\n\n")
        server_log_f.flush()

        server_proc = subprocess.Popen(
            server_cmd,
            stdout=server_log_f,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=server_env,
            text=True,
            start_new_session=True,
        )

        print(f"[{index}/{total}] Waiting for server readiness", flush=True)
        wait_for_server_ready(
            args.host, args.port, server_proc, args.server_ready_timeout
        )

        print(
            f"[{index}/{total}] Running benchmark for policy={exp.policy}", flush=True
        )
        print(f"  bench:  {format_cmd(benchmark_cmd)}", flush=True)

        with open(benchmark_log, "w") as bench_log_f:
            bench_log_f.write("# Benchmark command\n")
            bench_log_f.write(format_cmd(benchmark_cmd) + "\n\n")
            bench_log_f.flush()

            bench_completed = subprocess.run(
                benchmark_cmd,
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
        print(f"[{index}/{total}] Completed policy={exp.policy}", flush=True)
        return result
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
        print(f"[{index}/{total}] Failed policy={exp.policy}: {exc}", flush=True)
        return result
    finally:
        if server_proc is not None:
            print(
                f"[{index}/{total}] Stopping server for policy={exp.policy}", flush=True
            )
            result.server_returncode = terminate_process_tree(server_proc)

        if server_log_f is not None:
            server_log_f.flush()
            server_log_f.close()

        result.end_time = time.time()
        write_json(metadata_json, asdict(result))


def build_output_dir(base_output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"eviction_policy_bench_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential benchmark driver for SGLang radix eviction policies."
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="lru,seglen,marconi,marconi-fixed",
        help="Comma-separated eviction policies to benchmark.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="token-trace",
        help="Dataset name passed to sglang.bench_serving, e.g. sharegpt or token-trace.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="swebench_sps=10_art=10_nums=100.jsonl",
        help="Dataset path passed to sglang.bench_serving.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=8.0,
        help="Request rate passed to sglang.bench_serving.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=128,
        help="Output length passed to sglang.bench_serving.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1200,
        help="Optional num-prompts override for sglang.bench_serving.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--marconi-eff-weight", type=float, default=0.85)
    parser.add_argument(
        "--mamba-scheduler-strategy",
        type=str,
        default="extra_buffer",
    )
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
        help="Base directory where a fresh timestamped run directory will be created.",
    )
    args = parser.parse_args()

    policies = parse_csv_strs(args.policies)
    if not policies:
        raise ValueError("No policies specified.")

    output_dir = build_output_dir(Path(args.output_dir))

    manifest = {
        "policies": policies,
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "request_rate": args.request_rate,
        "sharegpt_output_len": args.sharegpt_output_len,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "model_path": args.model_path,
        "host": args.host,
        "port": args.port,
        "gpu": args.gpu,
        "marconi_eff_weight": args.marconi_eff_weight,
        "mamba_scheduler_strategy": args.mamba_scheduler_strategy,
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Output directory: {output_dir}", flush=True)
    print(f"Policies: {', '.join(policies)}", flush=True)
    print(f"Dataset: {args.dataset_name} ({args.dataset_path})", flush=True)
    print(f"Request rate: {args.request_rate}", flush=True)

    results = []
    experiments = [Experiment(policy=policy) for policy in policies]
    total = len(experiments)

    for index, exp in enumerate(experiments, start=1):
        results.append(run_experiment(exp, args, output_dir, index, total))

    write_json(output_dir / "summary.json", [asdict(r) for r in results])

    completed = sum(r.status == "completed" for r in results)
    failed = total - completed
    print(
        f"Finished {total} policy runs: {completed} completed, {failed} failed.",
        flush=True,
    )
    print(f"Summary written to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
