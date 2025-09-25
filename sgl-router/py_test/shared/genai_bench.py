from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pytest

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkStats:
    filename: str
    ttft_mean: float
    e2e_latency_mean: float
    input_throughput_mean: float
    output_throughput_mean: float


@dataclass
class BenchmarkResult:
    experiment_folder: Path
    stats: list[BenchmarkStats]
    stdout: str
    stderr: str
    returncode: Optional[int]
    gpu_utilization: Optional[dict]


def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("shutil.which(%r) failed: %s", cmd, exc)
        return None


def _graceful_stop_popen(proc: subprocess.Popen) -> None:
    if proc is None:  # pragma: no cover - defensive
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            for _ in range(5):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if proc.poll() is None:
                proc.kill()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to stop process %s: %s", proc, exc)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _graceful_stop_pid(pid: int) -> None:
    try:
        if not _pid_alive(pid):
            return
        try:
            os.kill(pid, signal.SIGTERM)  # type: ignore[name-defined]
        except Exception:
            pass
        for _ in range(5):
            if not _pid_alive(pid):
                break
            time.sleep(1)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    except Exception:  # pragma: no cover - defensive
        pass


def _graceful_stop_any(obj: Any) -> None:
    try:
        if isinstance(obj, subprocess.Popen):
            _graceful_stop_popen(obj)
            return
        if isinstance(obj, int):
            _graceful_stop_pid(obj)
            return
        proc_obj = getattr(obj, "proc", None)
        if isinstance(proc_obj, subprocess.Popen):
            _graceful_stop_popen(proc_obj)
    except Exception:  # pragma: no cover - defensive
        pass


def _gpu_monitor_should_run(thresholds: Optional[dict]) -> bool:
    try:
        mean_th = None if thresholds is None else thresholds.get("gpu_util_mean_min")
        p50_th = None if thresholds is None else thresholds.get("gpu_util_p50_min")
        want = bool(mean_th is not None or p50_th is not None)
    except Exception:
        want = False
    if not want:
        env_flag = os.environ.get("GPU_UTIL_LOG", "").lower() in {"1", "true", "yes"}
        want = want or env_flag
    return want


def _gpu_monitor_path(experiment_folder: str) -> Path:
    return Path.cwd() / experiment_folder / "gpu_utilization.json"


def _launch_gpu_monitor(bench_pid: int, experiment_folder: str, interval: float):
    try:
        from multiprocessing import Process

        out_path = _gpu_monitor_path(experiment_folder)
        proc = Process(
            target=_gpu_monitor_proc_entry,
            args=(bench_pid, str(out_path), interval),
            daemon=True,
        )
        proc.start()
        return proc, out_path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to launch GPU monitor: %s", exc)
        return None, None


def _read_gpu_monitor_result(path: Optional[Path]) -> Optional[dict]:
    if not path:
        return None
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read GPU monitor result from %s: %s", path, exc)
    return None


def _log_and_assert_gpu_thresholds(
    result: Optional[dict], thresholds: Optional[dict]
) -> None:
    if not result or not isinstance(result, dict) or result.get("count", 0) <= 0:
        logger.warning("GPU utilization monitor produced no samples.")
        return
    overall = result.get("overall", {}) if isinstance(result, dict) else {}
    count = int(result.get("count", 0))
    mean_th = None if thresholds is None else thresholds.get("gpu_util_mean_min")
    p50_th = None if thresholds is None else thresholds.get("gpu_util_p50_min")

    mean_v = float(overall.get("mean", 0.0))
    p50_v = overall.get("p50")

    logger.info(
        "GPU utilization overall: mean=%.2f%% p50=%s (samples=%d)",
        mean_v,
        (f"{float(p50_v):.2f}%" if p50_v is not None else "n/a"),
        count,
    )

    if mean_th is not None:
        assert mean_v >= float(
            mean_th
        ), f"GPU utilization mean below threshold: {mean_v:.2f}% < {mean_th}%"
    if p50_th is not None and p50_v is not None:
        p50_f = float(p50_v)
        assert p50_f >= float(
            p50_th
        ), f"GPU utilization p50 below threshold: {p50_f:.2f}% < {p50_th}%"


def _gpu_monitor_proc_entry(
    bench_pid: int, out_file: str, interval: float
) -> None:  # pragma: no cover - multiprocessing
    total = 0.0
    n = 0
    try:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
        except Exception:
            _write_empty_gpu_payload(out_file)
            return
        try:
            import pynvml  # type: ignore

            count = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception:
            _write_empty_gpu_payload(out_file)
            return

        per_gpu_samples: dict[str, list[float]] = {}
        overall_samples: list[float] = []

        while True:
            if not os.path.exists(f"/proc/{bench_pid}"):
                break
            try:
                import pynvml  # type: ignore

                vals: list[float] = []
                for idx, handle in enumerate(handles):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        vals.append(float(util))
                        key = str(idx)
                        per_gpu_samples.setdefault(key, []).append(float(util))
                    except Exception:
                        continue
                if vals:
                    avg = sum(vals) / len(vals)
                    overall_samples.append(avg)
                    total += avg
                    n += 1
            except Exception:
                pass
            time.sleep(interval)
    finally:
        try:
            _write_gpu_payload(
                out_file,
                bench_pid,
                interval,
                total,
                n,
                per_gpu_samples,
                overall_samples,
            )
        except Exception:
            pass
        try:
            import pynvml  # type: ignore

            pynvml.nvmlShutdown()
        except Exception:
            pass


def _write_empty_gpu_payload(out_file: str) -> None:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(
        json.dumps(
            {
                "count": 0,
                "overall": {"mean": 0.0},
                "per_gpu": {},
                "raw": {},
            }
        )
    )


def _write_gpu_payload(
    out_file: str,
    bench_pid: int,
    interval: float,
    total: float,
    count: int,
    per_gpu_samples: dict[str, list[float]],
    overall_samples: list[float],
) -> None:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    def pct_from(samples: list[float], percentile: float) -> float:
        if not samples:
            return 0.0
        srt = sorted(samples)
        idx = max(
            0, min(len(srt) - 1, int(round((percentile / 100.0) * (len(srt) - 1))))
        )
        return float(srt[idx])

    overall_mean = (total / count) if count > 0 else 0.0
    per_gpu_summary: dict[str, dict] = {}
    for key, arr in per_gpu_samples.items():
        per_gpu_summary[key] = {
            "mean": float(sum(arr) / len(arr)) if arr else 0.0,
            "p5": pct_from(arr, 5),
            "p10": pct_from(arr, 10),
            "p25": pct_from(arr, 25),
            "p50": pct_from(arr, 50),
            "p75": pct_from(arr, 75),
            "p90": pct_from(arr, 90),
            "p95": pct_from(arr, 95),
            "min": float(min(arr)) if arr else 0.0,
            "max": float(max(arr)) if arr else 0.0,
            "count": len(arr),
        }

    payload = {
        "bench_pid": bench_pid,
        "interval_sec": interval,
        "count": count,
        "overall": {
            "mean": float(overall_mean),
            "p5": pct_from(overall_samples, 5),
            "p10": pct_from(overall_samples, 10),
            "p25": pct_from(overall_samples, 25),
            "p50": pct_from(overall_samples, 50),
            "p75": pct_from(overall_samples, 75),
            "p90": pct_from(overall_samples, 90),
            "p95": pct_from(overall_samples, 95),
            "min": float(min(overall_samples)) if overall_samples else 0.0,
            "max": float(max(overall_samples)) if overall_samples else 0.0,
        },
        "per_gpu": per_gpu_summary,
        "raw": {
            "overall": overall_samples,
            "per_gpu": per_gpu_samples,
        },
    }
    Path(out_file).write_text(json.dumps(payload))


def run_genai_bench(
    *,
    router_url: str,
    model_path: str,
    experiment_folder: str,
    timeout_sec: Optional[int] = None,
    thresholds: Optional[dict] = None,
    extra_env: Optional[dict] = None,
    num_concurrency: int = 32,
    traffic_scenario: str = "D(4000,100)",
    max_requests_per_run: Optional[int] = None,
    clean_experiment: bool = True,
    kill_procs: Optional[Iterable[Any]] = None,
    drain_delay_sec: int = 6,
) -> BenchmarkResult:
    cli = _which("genai-bench")
    if not cli:
        pytest.fail("genai-bench CLI not found; please install it to run benchmarks")

    if clean_experiment:
        exp_dir = Path.cwd() / experiment_folder
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)

    requests_per_run = (
        max_requests_per_run
        if max_requests_per_run is not None
        else num_concurrency * 5
    )

    cmd = [
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
        str(requests_per_run),
        "--max-time-per-run",
        "3",
        "--experiment-folder-name",
        experiment_folder,
        "--experiment-base-dir",
        str(Path.cwd()),
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    timeout = timeout_sec or int(os.environ.get("GENAI_BENCH_TEST_TIMEOUT", "120"))
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    want_gpu_monitor = _gpu_monitor_should_run(thresholds)
    monitor_proc = None
    monitor_path: Optional[Path] = None
    gpu_util_result: Optional[dict] = None
    if want_gpu_monitor:
        interval = float(os.environ.get("GPU_UTIL_SAMPLE_INTERVAL", "2.0"))
        monitor_proc, monitor_path = _launch_gpu_monitor(
            bench_pid=proc.pid,
            experiment_folder=experiment_folder,
            interval=interval,
        )

    stdout = stderr = ""
    rc: Optional[int] = None
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = proc.communicate()
        rc = proc.returncode

        experiment_dir = _locate_experiment_dir(experiment_folder)
        json_files = _collect_json_results(experiment_dir)
        stats = _parse_benchmark_stats(json_files, experiment_folder, thresholds)

        if want_gpu_monitor:
            try:
                if monitor_proc is not None:
                    monitor_proc.join(timeout=5)
            except Exception:
                pass
            gpu_util_result = _read_gpu_monitor_result(monitor_path)
            _log_and_assert_gpu_thresholds(gpu_util_result, thresholds)

        return BenchmarkResult(
            experiment_folder=experiment_dir,
            stats=stats,
            stdout=stdout,
            stderr=stderr,
            returncode=rc,
            gpu_utilization=gpu_util_result,
        )
    finally:
        if kill_procs:
            if drain_delay_sec > 0:
                try:
                    time.sleep(drain_delay_sec)
                except Exception:
                    pass
            for handle in kill_procs:
                _graceful_stop_any(handle)
            try:
                time.sleep(2)
            except Exception:
                pass
        if monitor_proc is not None and monitor_proc.is_alive():
            try:
                monitor_proc.terminate()
            except Exception:
                pass


def _locate_experiment_dir(experiment_folder: str) -> Path:
    base = Path.cwd()
    direct = base / experiment_folder
    if direct.is_dir():
        return direct
    for path in base.rglob(experiment_folder):
        if path.is_dir() and path.name == experiment_folder:
            return path
    raise AssertionError(
        f"Benchmark failed: experiment folder not found: {experiment_folder}"
    )


def _collect_json_results(experiment_dir: Path) -> list[Path]:
    json_files: list[Path] = []
    for _ in range(10):
        json_files = [
            path
            for path in experiment_dir.rglob("*.json")
            if "experiment_metadata" not in path.name
        ]
        if json_files:
            break
        time.sleep(1)
    if not json_files:
        raise AssertionError("Benchmark failed: no JSON results found")
    return json_files


def _parse_benchmark_stats(
    json_files: Iterable[Path],
    experiment_folder: str,
    thresholds: Optional[dict],
) -> list[BenchmarkStats]:
    stats: list[BenchmarkStats] = []
    for json_file in json_files:
        data = json.loads(json_file.read_text())
        agg = data.get("aggregated_metrics", {}).get("stats", {})
        ttft_mean = float(agg.get("ttft", {}).get("mean", float("inf")))
        e2e_latency_mean = float(agg.get("e2e_latency", {}).get("mean", float("inf")))
        input_tp_mean = float(agg.get("input_throughput", {}).get("mean", 0.0))
        output_tp_mean = float(agg.get("output_throughput", {}).get("mean", 0.0))

        logger.info(
            "genai-bench[%s] %s ttft_mean=%.3fs e2e_latency_mean=%.3fs input_tp_mean=%.1f tok/s output_tp_mean=%.1f tok/s",
            experiment_folder,
            json_file.name,
            ttft_mean,
            e2e_latency_mean,
            input_tp_mean,
            output_tp_mean,
        )

        if thresholds is not None:
            assert (
                ttft_mean <= thresholds["ttft_mean_max"]
            ), f"TTFT validation failed: {ttft_mean} > {thresholds['ttft_mean_max']} (file={json_file.name})"
            assert e2e_latency_mean <= thresholds["e2e_latency_mean_max"], (
                "E2E latency validation failed: "
                f"{e2e_latency_mean} > {thresholds['e2e_latency_mean_max']} (file={json_file.name})"
            )
            assert input_tp_mean >= thresholds["input_throughput_mean_min"], (
                "Input throughput validation failed: "
                f"{input_tp_mean} < {thresholds['input_throughput_mean_min']} (file={json_file.name})"
            )
            assert output_tp_mean >= thresholds["output_throughput_mean_min"], (
                "Output throughput validation failed: "
                f"{output_tp_mean} < {thresholds['output_throughput_mean_min']} (file={json_file.name})"
            )

        stats.append(
            BenchmarkStats(
                filename=json_file.name,
                ttft_mean=ttft_mean,
                e2e_latency_mean=e2e_latency_mean,
                input_throughput_mean=input_tp_mean,
                output_throughput_mean=output_tp_mean,
            )
        )
    return stats


@pytest.fixture(scope="session")
def genai_bench_runner() -> Callable[..., BenchmarkResult]:
    def _run(**kwargs: Any) -> BenchmarkResult:
        return run_genai_bench(**kwargs)

    return _run
