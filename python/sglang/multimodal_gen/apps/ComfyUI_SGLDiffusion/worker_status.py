"""
Diagnostics for the SGLD worker: latency stats, GPU memory, status report.

Everything here must degrade to "unavailable" instead of raising; the status
node is used while debugging broken setups.
"""

import platform
import statistics
import time
from collections import deque

_SECRET_MARKERS = ("token", "secret", "password", "api_key", "apikey", "auth", "cred")
_PARALLEL_FIELDS = (
    "num_gpus",
    "tp_size",
    "sp_degree",
    "ulysses_degree",
    "ring_degree",
    "dp_size",
    "enable_cfg_parallel",
    "attention_backend",
    "dit_cpu_offload",
    "dit_layerwise_offload",
)
_UNAVAILABLE = "unavailable"


class StepLatency:
    """Wall-clock per sampler step, in seconds.

    Round-trips with the same timestep key are summed into one step, so CFG
    (several DiT calls per step) matches ComfyUI's per-step progress rate. A
    new sampler run always starts a new step. The first step ever seen is the
    warmup (compile, lazy init) and is kept apart from the stats; the rest go
    into a rolling window that persists across runs. A new worker means a new
    executor, which starts empty.
    """

    def __init__(self, window: int = 512):
        self._window = deque(maxlen=window)
        self._first = None
        self._first_calls = 0
        self._open = None
        self._open_calls = 0
        self._open_key = None
        self._steps_seen = 0

    def begin_run(self) -> None:
        self._close_open()
        self._open_key = None

    def record(self, seconds: float, key=None) -> None:
        if self._open is not None and key is not None and key == self._open_key:
            self._open += seconds
            self._open_calls += 1
        else:
            self._close_open()
            self._open, self._open_calls, self._open_key = seconds, 1, key

    def _close_open(self) -> None:
        if self._open is None:
            return
        if self._first is None:
            self._first, self._first_calls = self._open, self._open_calls
        else:
            self._window.append((self._open, self._open_calls))
        self._open = None
        self._steps_seen += 1

    def stats(self) -> dict:
        """Stats over finished steps plus the open one; does not mutate."""
        first, first_calls = self._first, self._first_calls
        pairs = list(self._window)
        if self._open is not None:
            if first is None:
                first, first_calls = self._open, self._open_calls
            else:
                pairs.append((self._open, self._open_calls))
        samples = [sec for sec, _ in pairs]
        calls = sum(n for _, n in pairs)
        steps = self._steps_seen + (1 if self._open is not None else 0)
        out = {"count": steps, "first": first, "first_calls": first_calls}
        if not samples:
            out.update(mean=None, p50=None, p95=None, last=None, calls_per_step=None)
            return out
        ordered = sorted(samples)
        out["mean"] = statistics.fmean(samples)
        out["p50"] = _percentile(ordered, 50)
        out["p95"] = _percentile(ordered, 95)
        out["last"] = samples[-1]
        out["calls_per_step"] = calls / len(samples)
        return out


def _percentile(sorted_values, pct):
    """Nearest-rank percentile of an ascending list."""
    rank = max(1, -(-len(sorted_values) * pct // 100))
    return sorted_values[int(rank) - 1]


def timed_call(tracker: StepLatency, key, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    tracker.record(time.perf_counter() - start, key)
    return result


def redact(value):
    """Copy of value with secret-looking dict entries masked."""
    if isinstance(value, dict):
        return {
            k: (
                "***"
                if any(m in str(k).lower() for m in _SECRET_MARKERS)
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def query_gpus():
    """List of dicts (name, total_mb, used_mb); [] when no GPU info is found."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            gpus = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append(
                    {
                        "name": name,
                        "total_mb": mem.total // 2**20,
                        "used_mb": mem.used // 2**20,
                    }
                )
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_mb": total // 2**20,
                    "used_mb": (total - free) // 2**20,
                }
            )
        return gpus
    except Exception:
        return []


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("sglang", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = _UNAVAILABLE
    return out


def collect_status(owner, tracker=None, gpu_query=query_gpus) -> dict:
    """Plain-dict snapshot of the worker; never raises."""
    status = {"versions": _versions(), "model_path": owner.model_path}
    patcher = owner._patcher
    status["model_type"] = patcher.model_type if patcher is not None else None
    server_args = owner.generator.server_args if owner.generator else None
    status["pipeline_class"] = server_args.pipeline_class_name if server_args else None
    status["parallel"] = (
        {f: getattr(server_args, f) for f in _PARALLEL_FIELDS} if server_args else {}
    )
    status["quantization"] = (
        server_args.component_quantizations if server_args else None
    )
    try:
        status["workers"] = [
            (p.pid, bool(p.is_alive())) for p in owner._worker_processes()
        ]
    except Exception:
        status["workers"] = []
    try:
        status["live"] = bool(owner._is_live())
    except Exception:
        status["live"] = False
    try:
        status["gpus"] = gpu_query()
    except Exception:
        status["gpus"] = []
    tracker = (
        tracker
        if tracker is not None
        else (owner.executor.step_latency if owner.executor is not None else None)
    )
    status["latency"] = tracker.stats() if tracker is not None else None
    status["options"] = redact(owner.last_options) if owner.last_options else None
    return status


def _ms(seconds):
    return "n/a" if seconds is None else f"{seconds * 1000:.1f}ms"


def format_status(status: dict, native_seconds_per_step: float = 0.0) -> str:
    v = status["versions"]
    lines = ["SGLD Worker Status"]
    if status["model_path"] is None:
        lines.append("worker: not started (run the workflow once to load a model)")
    else:
        lines.append(
            f"model: {status['model_path']} type={status['model_type']} "
            f"pipeline={status['pipeline_class']}"
        )
        alive = [pid for pid, ok in status["workers"] if ok]
        dead = [pid for pid, ok in status["workers"] if not ok]
        state = "live" if status["live"] else "NOT live"
        lines.append(f"worker: {state}, pids alive={alive} dead={dead}")
    par = status["parallel"]
    if par:
        lines.append("parallel: " + " ".join(f"{k}={val}" for k, val in par.items()))
    if status["quantization"]:
        lines.append(f"quantization: {status['quantization']}")
    gpus = status["gpus"]
    if gpus:
        for i, g in enumerate(gpus):
            lines.append(f"gpu{i}: {g['name']} {g['used_mb']}/{g['total_mb']} MiB used")
    else:
        lines.append(f"gpus: VRAM {_UNAVAILABLE} (no NVML or CUDA)")
    lat = status["latency"]
    if not lat or lat["count"] == 0:
        lines.append("step latency: no steps recorded yet")
    else:
        lines.append(
            f"sampler step latency: n={lat['count']} mean={_ms(lat['mean'])} "
            f"p50={_ms(lat['p50'])} p95={_ms(lat['p95'])} "
            f"last={_ms(lat['last'])} first(warmup)={_ms(lat['first'])} "
            f"calls_per_step={lat['calls_per_step'] and round(lat['calls_per_step'], 2)}"
        )
        if native_seconds_per_step and native_seconds_per_step > 0:
            if lat["mean"]:
                speedup = native_seconds_per_step / lat["mean"]
                lines.append(
                    f"speedup vs native: {speedup:.2f}x "
                    f"(native {native_seconds_per_step:.3f}s/step)"
                )
            else:
                lines.append("speedup vs native: need more than one step")
    if status["options"]:
        lines.append(f"options: {status['options']}")
    lines.append(
        "issue: "
        f"sglang={v['sglang']} torch={v['torch']} python={v['python']} "
        f"model={status['model_type']} gpus={len(gpus)} "
        f"{'x'.join(sorted({g['name'] for g in gpus})) if gpus else ''}".rstrip()
    )
    return "\n".join(lines)
