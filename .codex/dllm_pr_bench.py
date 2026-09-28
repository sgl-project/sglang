#!/usr/bin/env python3
"""Run the upstream-style dLLM compact-vocab serving benchmark matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/results/upstream-sync-fixed-length")
PORT = 31000
BASE_URL = f"http://127.0.0.1:{PORT}"


@dataclass(frozen=True)
class ModelCase:
    key: str
    served_name: str
    path: str
    concurrencies: tuple[int, ...]
    repeats: int
    disable_cuda_graph: bool


MODELS = (
    ModelCase(
        key="llada2-flash",
        served_name="inclusionAI/LLaDA2.0-flash",
        path="/models/inclusionAI/LLaDA2.0-flash",
        concurrencies=(4, 8, 16, 32),
        repeats=1,
        disable_cuda_graph=True,
    ),
    ModelCase(
        key="sdar-8b",
        served_name="JetLM/SDAR-8B-Chat",
        path="/models/JetLM/SDAR-8B-Chat",
        concurrencies=(4, 8, 16, 32),
        repeats=1,
        disable_cuda_graph=True,
    ),
    ModelCase(
        key="sdar-30b",
        served_name="JetLM/SDAR-30B-A3B-Chat-b32",
        path="/models/JetLM/SDAR-30B-A3B-Chat-b32",
        concurrencies=(16, 32),
        repeats=3,
        disable_cuda_graph=True,
    ),
)

MODES = {
    "baseline": {
        "SGLANG_DLLM_TP_LOCAL_VOCAB": "0",
        "SGLANG_DLLM_TP_LOCAL_VOCAB_PACKED_GATHER": "1",
    },
    "optimized": {
        "SGLANG_DLLM_TP_LOCAL_VOCAB": "1",
        "SGLANG_DLLM_TP_LOCAL_VOCAB_PACKED_GATHER": "1",
    },
}


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=90)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass


def wait_ready(process: subprocess.Popen, timeout_s: int = 2400) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not contacted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        for endpoint in ("/health_generate", "/health"):
            try:
                with urllib.request.urlopen(BASE_URL + endpoint, timeout=5) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = repr(exc)
        time.sleep(3)
    raise TimeoutError(f"server did not become ready: {last_error}")


def gpu_memory_mib() -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


class GpuMemoryMonitor:
    def __init__(self) -> None:
        self.before = gpu_memory_mib()
        self.peak = list(self.before)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self.stop_event.is_set():
            try:
                values = gpu_memory_mib()
                self.peak = [max(a, b) for a, b in zip(self.peak, values)]
            except Exception:
                pass
            self.stop_event.wait(0.2)

    def __enter__(self) -> "GpuMemoryMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)


def launch_server(case: ModelCase, mode: str) -> tuple[subprocess.Popen, object, list[str]]:
    run_dir = ROOT / case.key / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log = (run_dir / "server.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(MODES[mode])
    env["PYTHONPATH"] = "/workspace/sglang/python"
    env.pop("SGLANG_CONSUMER_STATE_TRACE_JSONL", None)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        case.path,
        "--served-model-name",
        case.served_name,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--tp-size",
        "4",
        "--dllm-algorithm",
        "LowConfidence",
        "--attention-backend",
        "flashinfer",
        "--sampling-backend",
        "flashinfer",
        "--max-running-requests",
        "32",
    ]
    if case.disable_cuda_graph:
        command.append("--cuda-graph-backend-decode=disabled")
    print(f"\n=== START SERVER {case.key} {mode} ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, server_log, command


def result_path(case: ModelCase, mode: str, concurrency: int, repeat: int) -> Path:
    return ROOT / case.key / mode / f"c{concurrency}-r{repeat}.json"


def valid_existing_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return result.get("completed") == 512 and result.get("status") == "ok"


def run_benchmark(case: ModelCase, mode: str, concurrency: int, repeat: int) -> dict:
    final_path = result_path(case, mode, concurrency, repeat)
    if valid_existing_result(final_path):
        print(f"SKIP completed {final_path}", flush=True)
        return json.loads(final_path.read_text(encoding="utf-8"))

    run_dir = final_path.parent
    raw_path = run_dir / f"c{concurrency}-r{repeat}.raw.jsonl"
    bench_log_path = run_dir / f"c{concurrency}-r{repeat}.bench.log"
    raw_path.unlink(missing_ok=True)
    seed = 100000 + concurrency + repeat * 1000000
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang-oai",
        "--base-url",
        BASE_URL,
        "--model",
        case.served_name,
        "--served-model-name",
        case.served_name,
        "--tokenizer",
        case.path,
        "--dataset-name",
        "random",
        "--random-input-len",
        "30",
        "--random-output-len",
        "32",
        "--random-range-ratio",
        "1",
        "--num-prompts",
        "512",
        "--warmup-requests",
        "256",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--temperature",
        "0",
        "--seed",
        str(seed),
        "--output-file",
        str(raw_path),
        "--output-details",
    ]
    print(f"\n--- RUN {case.key} {mode} c{concurrency} r{repeat} ---", flush=True)
    before = gpu_memory_mib()
    with bench_log_path.open("w", encoding="utf-8") as bench_log:
        with GpuMemoryMonitor() as monitor:
            completed = run(command, stdout=bench_log, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not raw_path.exists():
        raise RuntimeError(f"benchmark failed: {bench_log_path}")
    raw_lines = [line for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    if not raw_lines:
        raise RuntimeError(f"benchmark produced no JSON: {raw_path}")
    result = json.loads(raw_lines[-1])
    result.update(
        {
            "status": "ok",
            "model_key": case.key,
            "served_name": case.served_name,
            "mode": mode,
            "repeat": repeat,
            "seed": seed,
            "gpu_memory_before_mib": before,
            "gpu_memory_peak_mib": monitor.peak,
            "gpu_memory_peak_increment_mib": [
                peak - base for peak, base in zip(monitor.peak, before)
            ],
            "command": command,
        }
    )
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"DONE {case.key} {mode} c{concurrency} r{repeat}: "
        f"{result['output_throughput']:.2f} output tok/s, "
        f"completed={result['completed']}",
        flush=True,
    )
    return result


def all_results_for(case: ModelCase, mode: str) -> list[dict]:
    values = []
    for concurrency in case.concurrencies:
        for repeat in range(1, case.repeats + 1):
            path = result_path(case, mode, concurrency, repeat)
            if valid_existing_result(path):
                values.append(json.loads(path.read_text(encoding="utf-8")))
    return values


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize() -> dict:
    summary: dict = {"models": {}}
    for case in MODELS:
        rows = {}
        mismatches = 0
        for concurrency in case.concurrencies:
            ratios = []
            baseline_tps = []
            optimized_tps = []
            for repeat in range(1, case.repeats + 1):
                baseline = json.loads(
                    result_path(case, "baseline", concurrency, repeat).read_text()
                )
                optimized = json.loads(
                    result_path(case, "optimized", concurrency, repeat).read_text()
                )
                baseline_tps.append(float(baseline["output_throughput"]))
                optimized_tps.append(float(optimized["output_throughput"]))
                ratios.append(optimized_tps[-1] / baseline_tps[-1])
                before_texts = baseline.get("generated_texts", [])
                after_texts = optimized.get("generated_texts", [])
                mismatches += sum(
                    hashlib.sha256(a.encode()).digest()
                    != hashlib.sha256(b.encode()).digest()
                    for a, b in zip(before_texts, after_texts)
                ) + abs(len(before_texts) - len(after_texts))
            rows[str(concurrency)] = {
                "baseline_output_tps": geometric_mean(baseline_tps),
                "optimized_output_tps": geometric_mean(optimized_tps),
                "ratio": geometric_mean(ratios),
                "repeats": case.repeats,
            }
        claim_concurrencies = [c for c in case.concurrencies if c >= 8]
        claim_ratio = geometric_mean([rows[str(c)]["ratio"] for c in claim_concurrencies])
        summary["models"][case.key] = {
            "served_name": case.served_name,
            "rows": rows,
            "claim_ratio": claim_ratio,
            "output_hash_mismatches": mismatches,
        }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n# Upstream-style throughput ratios\n", flush=True)
    print("| model | c4 | c8 | c16 | c32 | claim geomean | hash mismatches |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
    for case in MODELS:
        model = summary["models"][case.key]
        cells = []
        for concurrency in (4, 8, 16, 32):
            row = model["rows"].get(str(concurrency))
            cells.append("-" if row is None else f"{row['ratio']:.3f}x")
        print(
            f"| {case.served_name} | {' | '.join(cells)} | "
            f"{model['claim_ratio']:.3f}x | {model['output_hash_mismatches']} |",
            flush=True,
        )
    return summary


def main() -> int:
    # The GB300 image ships 0.6.7.post2 cubins while current main requires the
    # 0.6.18 Python API. This matches the already-validated matrix container.
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    os.environ["PYTHONPATH"] = "/workspace/sglang/python"
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": [case.__dict__ for case in MODELS],
        "modes": MODES,
        "client": "python -m sglang.benchmark.serving --backend sglang-oai",
        "tp_size": 4,
        "num_prompts": 512,
        "warmup_requests": 256,
        "random_input_len": 30,
        "random_output_len": 32,
        "random_range_ratio": 1,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failures = []
    for case in MODELS:
        for mode in MODES:
            needed = [
                (c, r)
                for c in case.concurrencies
                for r in range(1, case.repeats + 1)
                if not valid_existing_result(result_path(case, mode, c, r))
            ]
            if not needed:
                print(f"SKIP server: all runs complete for {case.key} {mode}", flush=True)
                continue
            process = None
            log_file = None
            try:
                process, log_file, server_command = launch_server(case, mode)
                wait_ready(process)
                print(f"READY {case.key} {mode}", flush=True)
                for concurrency, repeat in needed:
                    run_benchmark(case, mode, concurrency, repeat)
            except Exception as exc:
                message = f"FAILED {case.key} {mode}: {exc!r}"
                print(message, flush=True)
                failures.append(message)
                (ROOT / "failures.log").open("a", encoding="utf-8").write(message + "\n")
            finally:
                stop_process(process)
                if log_file is not None:
                    log_file.close()
                time.sleep(10)
    if failures:
        print("Failures occurred; rerun the script to resume.", flush=True)
        return 1
    summarize()
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
