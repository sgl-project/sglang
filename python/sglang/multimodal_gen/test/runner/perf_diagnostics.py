"""Opt-in CI evidence; never changes performance assertions or retry decisions."""

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil
import pynvml

_ROOT_ENV = "SGLANG_DIFFUSION_DIAGNOSTICS_DIR"
_ATTEMPT_ENV = "SGLANG_DIFFUSION_DIAGNOSTICS_ATTEMPT_DIR"
_BOUNDARIES = (
    ("case_begin", re.compile(r"BEGIN diffusion testcase: ([\w.-]+)")),
    ("case_end", re.compile(r"END diffusion testcase: ([\w.-]+)")),
    ("modules_begin", re.compile(r"Loading pipeline modules from config")),
    ("modules_end", re.compile(r"Module load summary")),
    ("workers_ready", re.compile(r"All workers are ready")),
    ("warmup_progress", re.compile(r"Warmup.*?(\d+/\d+)")),
    ("stage_begin", re.compile(r"\[([\w.]+Stage)\] started")),
    ("stage_end", re.compile(r"\[([\w.]+Stage)\] finished")),
)


def _write_event(stream, event, **fields):
    stream.write(
        json.dumps(
            {
                "event": event,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                **fields,
            }
        )
        + "\n"
    )
    stream.flush()


def record_request(result: dict) -> None:
    """Flush each request before validation, even if pytest later times out."""
    directory = os.environ.get(_ATTEMPT_ENV)
    if directory:
        try:
            with (Path(directory) / "requests.jsonl").open("a") as stream:
                _write_event(stream, "request", result=result)
        except OSError as exc:
            print(f"[diagnostics] request write failed: {type(exc).__name__}")


def _process_sample(process):
    with process.oneshot():
        # comm can contain spaces/parentheses; fields here start at stat field 3
        stat = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": process.pid,
            "created": process.create_time(),
            "name": process.name(),
            "status": process.status(),
            "cpu_times": process.cpu_times()._asdict(),
            "rss_bytes": process.memory_info().rss,
            "io": process.io_counters()._asdict(),
            "minor_faults": int(stat[7]),
            "major_faults": int(stat[9]),
            "context_switches": process.num_ctx_switches()._asdict(),
            "kernel": _kernel_sample(process.pid),
            "scheduler": _scheduler_sample(process.pid, int(stat[36])),
        }


def _kernel_sample(pid):
    # sample the main thread without ptrace; never retain syscall arguments
    result = {}
    for name in ("wchan", "syscall"):
        try:
            value = Path(f"/proc/{pid}/{name}").read_text().strip()
            result[name] = value.split()[0] if value else None
        except OSError as exc:
            # restricted procfs must not discard the other process counters
            result[name] = {"error": type(exc).__name__}
    return result


def _scheduler_sample(pid, cpu):
    # main-thread counters; frequency is an instantaneous sample, not a guarantee
    result = {"cpu": cpu}
    try:
        values = Path(f"/proc/{pid}/schedstat").read_text().split()
        result["schedstat"] = dict(
            zip(
                ("runtime_ns", "runqueue_wait_ns", "timeslices"),
                map(int, values),
                strict=True,
            )
        )
    except (OSError, ValueError) as exc:
        result["schedstat"] = {"error": type(exc).__name__}
    try:
        result["frequency_khz"] = int(
            Path(
                f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            ).read_text()
        )
    except (OSError, ValueError) as exc:
        result["frequency_khz"] = {"error": type(exc).__name__}
    return result


def _nvml_value(call, *args):
    try:
        return call(*args)
    except pynvml.NVMLError as exc:
        return {"error": type(exc).__name__}


def _gpu_sample(index, handle):
    util = _nvml_value(pynvml.nvmlDeviceGetUtilizationRates, handle)
    memory = _nvml_value(pynvml.nvmlDeviceGetMemoryInfo, handle)
    return {
        "nvml_index": index,
        "name": pynvml.nvmlDeviceGetName(handle),
        "utilization": util
        if isinstance(util, dict)
        else {"gpu": util.gpu, "memory": util.memory},
        "memory_used_bytes": memory if isinstance(memory, dict) else memory.used,
        "sm_clock_mhz": _nvml_value(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM
        ),
        "memory_clock_mhz": _nvml_value(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM
        ),
        "power_mw": _nvml_value(pynvml.nvmlDeviceGetPowerUsage, handle),
        "power_limit_mw": _nvml_value(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle),
        "temperature_c": _nvml_value(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
        ),
        "throttle_reasons": _nvml_value(
            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle
        ),
    }


class AttemptDiagnostics:
    """Own one attempt's sampler and append-only artifacts, outside GPU workers."""

    def __init__(self, attempt: int):
        self.directory = None
        self.events = None
        self.thread = None
        self.stop = threading.Event()
        self.pending_line = ""
        root = os.environ.get(_ROOT_ENV)
        if not root:
            return
        try:
            Path(root).mkdir(parents=True, exist_ok=True)
            self.directory = Path(
                tempfile.mkdtemp(prefix=f"attempt-{attempt}-", dir=root)
            )
            self.events = (self.directory / "events.jsonl").open("a")
            packages = {}
            for package in (
                "sglang",
                "torch",
                "triton",
                "flashinfer-python",
                "diffusers",
                "transformers",
                "cuda-python",
                "cuda-bindings",
                "cuda-core",
            ):
                try:
                    packages[package] = version(package)
                except PackageNotFoundError:
                    packages[package] = None
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            _write_event(
                self.events,
                "attempt_begin",
                attempt=attempt,
                commit=commit.stdout.strip() if commit.returncode == 0 else None,
                packages=packages,
                machine=os.uname().machine,
                run_id=os.environ.get("GITHUB_RUN_ID"),
                run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
                partition=os.environ.get("DIFFUSION_PARTITION_ID"),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[diagnostics] initialization failed: {type(exc).__name__}")
            if self.events:
                self.events.close()
            self.events = None
            self.directory = None

    def environment(self):
        env = os.environ.copy()
        # nested runners must not append to their parent's request file
        env.pop(_ATTEMPT_ENV, None)
        if self.directory:
            env[_ATTEMPT_ENV] = str(self.directory.resolve())
        return env

    def start(self, pid: int):
        if self.directory:
            self.thread = threading.Thread(
                target=self._sample, args=(pid,), daemon=True
            )
            self.thread.start()

    def observe(self, chunk: bytes):
        if not self.events:
            return
        # retain only allowlisted boundaries, never arbitrary logs, prompts or env
        lines = (self.pending_line + chunk.decode("utf-8", errors="replace")).split(
            "\n"
        )
        self.pending_line = lines.pop()[-65536:]
        try:
            for line in lines:
                for boundary, pattern in _BOUNDARIES:
                    match = pattern.search(line)
                    if match:
                        _write_event(
                            self.events,
                            "observed_boundary",
                            boundary=boundary,
                            labels=match.groups(),
                        )
        except OSError as exc:
            print(f"[diagnostics] boundary write failed: {type(exc).__name__}")
            self.events.close()
            self.events = None

    def _sample(self, pid):
        initialized = False
        try:
            with (self.directory / "resources.jsonl").open("a") as stream:
                parent = psutil.Process(pid)
                handles = []
                try:
                    pynvml.nvmlInit()
                    initialized = True
                    handles = [
                        (i, pynvml.nvmlDeviceGetHandleByIndex(i))
                        for i in range(pynvml.nvmlDeviceGetCount())
                    ]
                    _write_event(
                        stream,
                        "gpu_metadata",
                        driver=pynvml.nvmlSystemGetDriverVersion(),
                        software_power_cap_mask=pynvml.nvmlClocksThrottleReasonSwPowerCap,
                    )
                except pynvml.NVMLError as exc:
                    _write_event(stream, "gpu_unavailable", error=type(exc).__name__)
                while not self.stop.is_set():
                    started = time.monotonic()
                    started_wall_time_ns = time.time_ns()
                    try:
                        processes = [parent, *parent.children(recursive=True)]
                    except psutil.NoSuchProcess:
                        break
                    rows = []
                    for process in processes:
                        try:
                            rows.append(_process_sample(process))
                        except (psutil.Error, OSError) as exc:
                            rows.append(
                                {"pid": process.pid, "error": type(exc).__name__}
                            )
                    pids = {p.pid for p in processes}
                    process_sample_seconds = time.monotonic() - started
                    gpus = []
                    gpu_query_timings = []
                    for index, handle in handles:
                        query_started = time.monotonic()
                        query_timing = {"nvml_index": index}
                        try:
                            owners = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                            query_timing["ownership_seconds"] = (
                                time.monotonic() - query_started
                            )
                            # NVML indices need not match CUDA_VISIBLE_DEVICES: attribute
                            # devices by this attempt's live descendants, not ordinal
                            if any(owner.pid in pids for owner in owners):
                                metrics_started = time.monotonic()
                                gpus.append(_gpu_sample(index, handle))
                                query_timing["metrics_seconds"] = (
                                    time.monotonic() - metrics_started
                                )
                        except pynvml.NVMLError as exc:
                            gpus.append(
                                {"nvml_index": index, "error": type(exc).__name__}
                            )
                        finally:
                            query_timing["total_seconds"] = (
                                time.monotonic() - query_started
                            )
                            gpu_query_timings.append(query_timing)
                    _write_event(
                        stream,
                        "resources",
                        processes=rows,
                        gpus=gpus,
                        sample_started_wall_time_ns=started_wall_time_ns,
                        process_sample_seconds=process_sample_seconds,
                        gpu_query_timings=gpu_query_timings,
                        sample_seconds=time.monotonic() - started,
                    )
                    self.stop.wait(1)
        except (OSError, psutil.Error) as exc:
            print(f"[diagnostics] sampling stopped: {type(exc).__name__}")
        finally:
            if initialized:
                pynvml.nvmlShutdown()

    def finish(self, returncode):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.events:
            try:
                _write_event(
                    self.events,
                    "attempt_end",
                    returncode=returncode,
                    sampler_stopped=self.thread is None or not self.thread.is_alive(),
                )
            except OSError as exc:
                print(f"[diagnostics] final write failed: {type(exc).__name__}")
            finally:
                self.events.close()
