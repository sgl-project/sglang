"""Structured metrics from CI tests.

A test (or the kit it uses) calls ``record(name, value, **labels)`` for each
number it measures. In CI, each call appends one JSON line to a file under
``$SGLANG_CI_METRICS_DIR``; the job uploads that directory, and the
workflow's ``ci-metrics`` job prints every record it collected, so all of a
run's metrics are in one log, one JSON object per line. Outside CI (no
``SGLANG_CI_METRICS_DIR``) ``record`` does nothing.

Every name must be in ``METRICS``: that is what keeps one quantity under one
name and one unit across tests. A name that isn't there, a label that isn't
in ``LABELS``, or a value that isn't a finite number is skipped with a
warning; ``record`` never fails a test.

Adding a metric: add a line to ``METRICS``. Adding a label: add it to
``LABELS``.
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
METRICS_DIR_ENV = "SGLANG_CI_METRICS_DIR"
# Set by the CI test runner (ci_utils.run_unittest_files) for each attempt of a file.
ATTEMPT_ENV = "SGLANG_CI_TEST_ATTEMPT"

_REPO_ROOT = Path(__file__).resolve().parents[4]


class Kind(str, Enum):
    PERF = "perf"
    ACCURACY = "accuracy"
    SPEC = "spec"
    RESOURCE = "resource"


@dataclass(frozen=True)
class Metric:
    kind: Kind
    unit: str
    # Which way is better: "higher" or "lower".
    better: str
    description: str


def _m(kind, unit, better, description):
    return Metric(kind, unit, better, description)


METRICS = {
    # Serving benchmarks (bench_serving, through check_perf).
    "output_throughput": _m(
        Kind.PERF, "token/s", "higher", "Generated tokens per second"
    ),
    "input_throughput": _m(Kind.PERF, "token/s", "higher", "Prompt tokens per second"),
    "throughput": _m(Kind.PERF, "req/s", "higher", "Requests per second"),
    "median_e2e_latency_ms": _m(
        Kind.PERF, "ms", "lower", "Median request latency, end to end"
    ),
    "median_ttft_ms": _m(Kind.PERF, "ms", "lower", "Median time to first token"),
    "median_itl_ms": _m(Kind.PERF, "ms", "lower", "Median inter-token latency"),
    "avg_latency_ms": _m(Kind.PERF, "ms", "lower", "Mean request latency"),
    "p95_latency_ms": _m(Kind.PERF, "ms", "lower", "95th percentile request latency"),
    # CUDA graph persistence.
    "load_cuda_graph": _m(
        Kind.PERF, "s", "lower", "CUDA graph load time, from a saved graph"
    ),
    "load_scheduler_e2e": _m(
        Kind.PERF, "s", "lower", "Scheduler start to ready, loading saved graphs"
    ),
    "native_cuda_graph": _m(
        Kind.PERF, "s", "lower", "CUDA graph capture time, captured natively"
    ),
    "native_scheduler_e2e": _m(
        Kind.PERF, "s", "lower", "Scheduler start to ready, capturing graphs"
    ),
    # Nightly one-batch benchmarks (NightlyBenchmarkRunner), one per batch size and length.
    "bench_latency": _m(Kind.PERF, "s", "lower", "One batch, prompt to last token"),
    "bench_input_throughput": _m(
        Kind.PERF, "token/s", "higher", "One batch, prompt tokens per second"
    ),
    "bench_output_throughput": _m(
        Kind.PERF, "token/s", "higher", "One batch, generated tokens per second"
    ),
    "bench_overall_throughput": _m(
        Kind.PERF, "token/s", "higher", "One batch, all tokens per second"
    ),
    "bench_last_ttft": _m(Kind.PERF, "s", "lower", "One batch, time to first token"),
    "bench_last_gen_throughput": _m(
        Kind.PERF, "token/s", "higher", "One batch, decode tokens per second"
    ),
    # Accuracy evals; the `eval` label names the benchmark.
    "eval_score": _m(
        Kind.ACCURACY, "", "higher", "An eval's score or accuracy, 0 to 1"
    ),
    "eval_invalid_rate": _m(
        Kind.ACCURACY, "", "lower", "Share of answers that couldn't be parsed"
    ),
    "eval_latency": _m(Kind.PERF, "s", "lower", "Time to run the eval"),
    "eval_output_throughput": _m(
        Kind.PERF, "token/s", "higher", "Generated tokens per second during the eval"
    ),
    # Speculative decoding.
    "accept_length": _m(
        Kind.SPEC, "", "higher", "Mean tokens accepted per verification step"
    ),
    "bs_1_speed": _m(Kind.PERF, "token/s", "higher", "Decode speed at batch size 1"),
}

# The labels a record may carry: what tells two measurements of one metric
# apart within a test.
LABELS = (
    "label",  # the check_perf block: test method plus suffix
    "model",
    "backend",
    "eval",
    "dataset",
    "variant",
    "num_shots",
    "batch_size",
    "input_len",
    "output_len",
    "tp",
    "dp",
)

_case: Optional[str] = None


def set_case(case: Optional[str]) -> None:
    """The test method now running, as Class.method (CustomTestCase.setUp)."""
    global _case
    _case = case


def _test_file() -> Optional[str]:
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if not main:
        return None
    path = Path(main).resolve()
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _attempt() -> Optional[int]:
    raw = os.environ.get(ATTEMPT_ENV)
    return int(raw) if raw and raw.isdigit() else None


def _number(value):
    if hasattr(value, "item"):  # numpy and torch scalars
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _scalar_label(value):
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return None


def record(name: str, value, *, bound=None, **labels) -> Optional[dict]:
    """Records one measurement; returns the record, or None if skipped.

    ``bound`` is the threshold the test holds the value to, if it has one.
    Never raises.
    """
    try:
        return _record(name, value, bound, labels)
    except Exception as e:  # never fail a test over its metrics
        logger.warning(f"ci metrics: could not record {name!r}: {e}")
        return None


def _record(name, value, bound, labels):
    metric = METRICS.get(name)
    if metric is None:
        logger.warning(f"ci metrics: {name!r} is not in METRICS; not recorded")
        return None
    number = _number(value)
    if number is None:
        logger.warning(
            f"ci metrics: {name}={value!r} is not a finite number; not recorded"
        )
        return None
    clean = {}
    for key, raw in labels.items():
        label = _scalar_label(raw)
        if key not in LABELS:
            logger.warning(
                f"ci metrics: label {key!r} is not in LABELS; dropped from {name}"
            )
        elif label is not None:
            clean[key] = label
    rec = {
        "v": SCHEMA_VERSION,
        "name": name,
        "value": number,
        "unit": metric.unit,
        "kind": metric.kind.value,
        "better": metric.better,
        "bound": _number(bound),
        "test": _test_file(),
        "case": _case,
        "attempt": _attempt(),
        "labels": clean,
        "ts": round(time.time(), 3),
    }
    directory = os.environ.get(METRICS_DIR_ENV)
    if directory:
        os.makedirs(directory, exist_ok=True)
        # One file per process: test files run as separate processes.
        with open(
            os.path.join(directory, f"{os.getpid()}.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
