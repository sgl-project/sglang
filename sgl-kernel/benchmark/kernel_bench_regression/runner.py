"""Run kernel benchmarks and capture their metrics into a results dict.

The runner imports each benchmark module by file path, grabs its
``triton.testing.perf_report`` object, and calls the underlying benchmark function
directly for the kernel-under-test provider on each config. Driving ``mark.fn``
directly (instead of relying on the printed table or ``return_df``) is robust across
triton versions and skips the slow reference (torch) provider entirely.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .registry import BenchCase

# sgl-kernel/benchmark/ -- the directory that holds the bench_*.py files.
BENCH_DIR = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = 1


def _load_mark(bench_file: str, mark_attr: str):
    """Import ``bench_file`` from the benchmark dir and return its perf_report mark.

    Each bench file must already see ``SGLANG_IS_IN_CI`` in the environment (it reads
    it at import time), so callers set that before invoking the runner.
    """
    path = BENCH_DIR / bench_file
    if not path.exists():
        raise FileNotFoundError(f"benchmark file not found: {path}")
    module_name = f"_dsbench_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Make the benchmark dir importable in case the file uses sibling imports.
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    mark = getattr(module, mark_attr)
    return mark


def _benchmark_obj(mark):
    """Return the single ``triton.testing.Benchmark`` backing a perf_report mark."""
    benchmarks = getattr(mark, "benchmarks", None)
    if benchmarks is None:
        raise AttributeError("perf_report object has no .benchmarks attribute")
    if isinstance(benchmarks, (list, tuple)):
        if len(benchmarks) != 1:
            raise ValueError(
                "regression harness only supports single-Benchmark perf_report objects"
            )
        return benchmarks[0]
    return benchmarks


def _config_key(x_names: List[str], x_val) -> str:
    """Stable string key for one config, e.g. ``num_tokens=1,num_heads=8``."""
    values = x_val if isinstance(x_val, (list, tuple)) else (x_val,)
    return ",".join(f"{name}={value}" for name, value in zip(x_names, values))


def _median_metric(result) -> float:
    """Benchmark fns return either a scalar metric or a (median, max, min) tuple."""
    if isinstance(result, (list, tuple)):
        return float(result[0])
    return float(result)


def _device_capability() -> Optional[Tuple[int, int]]:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _device_name() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def run_case(case: BenchCase, repeat: int = 3) -> Dict[str, Optional[float]]:
    """Run every config of one case and return ``{config_key: best_metric}``.

    ``best_metric`` is the best-of-``repeat`` value (max for throughput, min for
    latency) to suppress noise from non-isolated CI GPUs. A config that raises is
    recorded as ``None`` so a single bad shape never aborts the whole run.
    """
    import torch  # noqa: F401  (ensures CUDA context errors surface here)

    mark = _load_mark(case.bench_file, case.mark_attr)
    bench = _benchmark_obj(mark)
    x_names = bench.x_names
    line_arg = bench.line_arg
    const_args = dict(getattr(bench, "args", {}) or {})
    const_args.update(case.extra_args)

    x_vals = (
        case.configs_override if case.configs_override is not None else bench.x_vals
    )

    measurements: Dict[str, Optional[float]] = {}
    for x_val in x_vals:
        key = _config_key(x_names, x_val)
        values = x_val if isinstance(x_val, (list, tuple)) else (x_val,)
        kwargs = dict(zip(x_names, values))
        kwargs.update(const_args)
        kwargs[line_arg] = case.provider

        best: Optional[float] = None
        for _ in range(max(1, repeat)):
            try:
                metric = _median_metric(mark.fn(**kwargs))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {case.case_id}::{key} raised: {exc}")
                best = None
                break
            if best is None:
                best = metric
            elif case.higher_is_better:
                best = max(best, metric)
            else:
                best = min(best, metric)
        measurements[key] = best
        if best is not None:
            print(f"  {case.case_id}::{key} = {best:.4g} {case.metric}")
    return measurements


def generate(
    cases: List[BenchCase],
    repeat: int = 3,
    tolerance: float = 0.05,
    commit: str = "",
) -> dict:
    """Run all ``cases`` (skipping ones the current GPU can't run) and build a
    results dict ready to serialize as ground truth or to compare against one."""
    capability = _device_capability()
    cap_tuple = capability if capability is not None else (0, 0)

    results: Dict[str, dict] = {}
    skipped: List[str] = []
    for case in cases:
        if cap_tuple < case.min_compute_capability:
            need = ".".join(map(str, case.min_compute_capability))
            have = ".".join(map(str, cap_tuple))
            print(
                f"[skip] {case.case_id}: needs compute capability >= {need}, have {have}"
            )
            skipped.append(case.case_id)
            continue
        tag_str = ",".join(case.tags)
        print(f"[run] {case.case_id} ({tag_str}: {case.component})")
        measurements = run_case(case, repeat=repeat)
        results[case.case_id] = {
            "metric": case.metric,
            "higher_is_better": case.higher_is_better,
            "tags": list(case.tags),
            "component": case.component,
            "measurements": measurements,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "device_name": _device_name(),
            "compute_capability": ".".join(map(str, cap_tuple)),
            "commit": commit or os.environ.get("GITHUB_SHA", ""),
            "generated_at_unix": int(time.time()),
            "repeat": repeat,
            "skipped_cases": skipped,
        },
        "tolerance": tolerance,
        "cases": results,
    }
