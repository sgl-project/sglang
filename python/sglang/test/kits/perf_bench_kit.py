"""Report and bound the numbers a serving benchmark produces.

The bounds are tuned per CI runner, so they are only enforced under
``is_in_ci()``; a local run still prints every number, which is the reason to
run one of these by hand.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from sglang.test.test_utils import is_in_amd_ci, is_in_ci, write_github_step_summary


@dataclass
class Metric:
    name: str
    value: float
    unit: str
    bound: Optional[float] = None
    amd_bound: Optional[float] = None
    # unittest method that enforces `bound`, e.g. "assertLessEqual".
    assertion: Optional[str] = None

    def line(self) -> str:
        return f"{self.name}: {self.value:.2f}" + (f" {self.unit}" if self.unit else "")

    def check(self, test_case) -> None:
        if self.bound is None:
            return
        limit = self.bound
        if is_in_amd_ci() and self.amd_bound is not None:
            limit = self.amd_bound
        getattr(test_case, self.assertion)(self.value, limit)


def at_least(name, value, bound, *, amd=None, unit="") -> Metric:
    """A throughput-like number: the run passes when it reaches `bound`."""
    return Metric(name, value, unit, bound, amd, "assertGreaterEqual")


def at_most(name, value, bound, *, amd=None, unit="") -> Metric:
    """A latency-like number: the run passes when it stays under `bound`."""
    return Metric(name, value, unit, bound, amd, "assertLessEqual")


def reported(name, value, *, unit="") -> Metric:
    """A number worth printing that no threshold is attached to."""
    return Metric(name, value, unit)


def check_perf(test_case, label: str, *metrics: Metric) -> None:
    report = f"### {label}\n" + "".join(m.line() + "\n" for m in metrics)
    print(report, end="")
    if not is_in_ci():
        return
    write_github_step_summary(report)
    for m in metrics:
        m.check(test_case)


def check_batch_scaling(
    test_case,
    label: str,
    run_one: Callable[[int], dict],
    bounds: Sequence[tuple],
) -> None:
    """Run `run_one` once per batch size and bound its latency at each one.

    Each `bounds` entry is `(batch_size, avg_ms, p95_ms, amd_avg_ms, amd_p95_ms)`.
    """
    for batch_size, avg_ms, p95_ms, amd_avg_ms, amd_p95_ms in bounds:
        res = run_one(batch_size)
        test_case.assertEqual(res["successful_requests"], res["total_requests"])
        check_perf(
            test_case,
            f"{label}_size_{batch_size}",
            at_most("avg_latency_ms", res["avg_latency_ms"], avg_ms, amd=amd_avg_ms),
            at_most("p95_latency_ms", res["p95_latency_ms"], p95_ms, amd=amd_p95_ms),
            reported("throughput", res["throughput"], unit="req/s"),
        )
