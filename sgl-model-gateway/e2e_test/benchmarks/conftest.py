"""Benchmark-specific fixtures."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from infra import GPUMonitor, should_monitor_gpu, terminate_process

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Parsed benchmark metrics from genai-bench output."""

    ttft_mean: float
    e2e_latency_mean: float
    input_throughput_mean: float
    output_throughput_mean: float
    file_name: str

    @classmethod
    def from_json(cls, path: Path) -> "BenchmarkResult":
        """Parse benchmark results from JSON file."""
        with path.open() as f:
            data = json.load(f)
        stats = data.get("aggregated_metrics", {}).get("stats", {})
        return cls(
            ttft_mean=float(stats.get("ttft", {}).get("mean", float("inf"))),
            e2e_latency_mean=float(
                stats.get("e2e_latency", {}).get("mean", float("inf"))
            ),
            input_throughput_mean=float(
                stats.get("input_throughput", {}).get("mean", 0.0)
            ),
            output_throughput_mean=float(
                stats.get("output_throughput", {}).get("mean", 0.0)
            ),
            file_name=path.name,
        )

    def log(self, experiment: str) -> None:
        """Log benchmark results."""
        logger.info(
            "genai-bench[%s] %s ttft=%.3fs e2e=%.3fs input=%.1f tok/s output=%.1f tok/s",
            experiment,
            self.file_name,
            self.ttft_mean,
            self.e2e_latency_mean,
            self.input_throughput_mean,
            self.output_throughput_mean,
        )

    def validate(self, thresholds: dict) -> None:
        """Validate metrics against thresholds."""
        checks = [
            ("ttft_mean_max", self.ttft_mean, "<=", "TTFT"),
            ("e2e_latency_mean_max", self.e2e_latency_mean, "<=", "E2E latency"),
            (
                "input_throughput_mean_min",
                self.input_throughput_mean,
                ">=",
                "Input throughput",
            ),
            (
                "output_throughput_mean_min",
                self.output_throughput_mean,
                ">=",
                "Output throughput",
            ),
        ]
        for key, value, op, name in checks:
            if key not in thresholds:
                continue
            threshold = thresholds[key]
            if op == "<=" and value > threshold:
                raise AssertionError(f"{name}: {value:.2f} > {threshold}")
            if op == ">=" and value < threshold:
                raise AssertionError(f"{name}: {value:.2f} < {threshold}")


def _build_command(
    cli: str,
    router_url: str,
    model_path: str,
    experiment_folder: str,
    num_concurrency: int,
    traffic_scenario: str,
    max_requests: int,
) -> list[str]:
    """Build genai-bench command."""
    return [
        cli,
        "benchmark",
        "--api-backend",
        "openai",
        "--api-base",
        router_url,
        "--api-key",
        "dummy-token",
        "--api-model-name",
        model_path,
        "--model-tokenizer",
        model_path,
        "--task",
        "text-to-text",
        "--num-concurrency",
        str(num_concurrency),
        "--traffic-scenario",
        traffic_scenario,
        "--max-requests-per-run",
        str(max_requests),
        "--max-time-per-run",
        "3",
        "--experiment-folder-name",
        experiment_folder,
        "--experiment-base-dir",
        str(Path.cwd()),
    ]


def _find_results(experiment_folder: str, timeout: int = 10) -> list[Path]:
    """Find benchmark result JSON files."""
    base = Path.cwd()
    folder = base / experiment_folder

    if not folder.is_dir():
        # Search for folder
        for p in base.rglob(experiment_folder):
            if p.is_dir() and p.name == experiment_folder:
                folder = p
                break

    if not folder.is_dir():
        raise AssertionError(f"Experiment folder not found: {experiment_folder}")

    # Wait for JSON results
    for _ in range(timeout):
        files = [
            p
            for p in folder.rglob("*.json")
            if "experiment_metadata" not in p.name and "gpu_utilization" not in p.name
        ]
        if files:
            return files
        time.sleep(1)

    raise AssertionError(f"No JSON results found in {folder}")


def _cleanup_procs(procs: list, drain_delay: int) -> None:
    """Terminate processes gracefully."""
    if not procs:
        return
    if drain_delay > 0:
        time.sleep(drain_delay)
    for p in procs:
        try:
            proc = getattr(p, "proc", p) if hasattr(p, "proc") else p
            if isinstance(proc, subprocess.Popen):
                terminate_process(proc)
        except Exception:
            pass
    time.sleep(2)


@pytest.fixture(scope="session")
def genai_bench_runner():
    """Run genai-bench and validate metrics.

    Usage:
        def test_perf(setup_backend, genai_bench_runner):
            backend, model_path, client, gateway = setup_backend
            genai_bench_runner(
                router_url=gateway.base_url,
                model_path=model_path,
                experiment_folder="benchmark_results",
                thresholds={"ttft_mean_max": 5, "gpu_util_p50_min": 99},
            )
    """

    def _run(
        *,
        router_url: str,
        model_path: str,
        experiment_folder: str,
        thresholds: dict | None = None,
        timeout_sec: int | None = None,
        num_concurrency: int = 32,
        traffic_scenario: str = "D(4000,100)",
        max_requests_per_run: int | None = None,
        kill_procs: list | None = None,
        drain_delay_sec: int = 6,
    ) -> None:
        cli = shutil.which("genai-bench")
        if not cli:
            pytest.fail("genai-bench CLI not found")

        # Clean previous results
        exp_dir = Path.cwd() / experiment_folder
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)

        # Build and run command
        max_requests = max_requests_per_run or num_concurrency * 5
        cmd = _build_command(
            cli,
            router_url,
            model_path,
            experiment_folder,
            num_concurrency,
            traffic_scenario,
            max_requests,
        )
        timeout = timeout_sec or int(os.environ.get("GENAI_BENCH_TEST_TIMEOUT", "120"))

        proc = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Start GPU monitor if needed
        gpu_monitor: GPUMonitor | None = None
        if should_monitor_gpu(thresholds):
            interval = float(os.environ.get("GPU_UTIL_SAMPLE_INTERVAL", "2.0"))
            gpu_monitor = GPUMonitor(output_dir=exp_dir, interval=interval)
            gpu_monitor.start(target_pid=proc.pid)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        try:
            # Parse and validate results
            for path in _find_results(experiment_folder):
                result = BenchmarkResult.from_json(path)
                result.log(experiment_folder)
                if thresholds:
                    result.validate(thresholds)

            # Validate GPU utilization
            if gpu_monitor:
                gpu_monitor.stop()
                gpu_monitor.log_summary()
                gpu_monitor.assert_thresholds(thresholds)

        finally:
            _cleanup_procs(kill_procs, drain_delay_sec)
            if gpu_monitor:
                gpu_monitor.stop(timeout=2)

    return _run
