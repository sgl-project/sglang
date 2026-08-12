"""Recorded throughput baselines for nightly performance benchmarks.

A perf test that only prints its numbers cannot fail when those numbers get
worse: the regression sits in a step summary until somebody reads it. These
helpers turn a recorded baseline into an assertion, so a drop fails the job
that measured it.

A baseline is the output throughput in tok/s per batch size for one benchmark
configuration, recorded from that job's own nightly history; `tolerance` is the
relative drop still accepted. Only output throughput is gated. It is the stable
half of `bench_one_batch_server` at a fixed batch size and input/output length
-- night-to-night spread on the MI35x DeepSeek-V4 jobs stays under 7%, against
~24% for the prefill-side input throughput, which reflects TTFT scheduling
jitter more than kernel speed.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

# 15% covers roughly twice the worst night-to-night spread measured on the
# DeepSeek-V4 MI35x jobs (7.0%), while still catching the kind of drop a
# toolchain regression produces (tens of percent).
DEFAULT_TOLERANCE = 0.15


@dataclass(frozen=True)
class ThroughputBaseline:
    """Expected output throughput in tok/s, keyed by batch size."""

    output_throughput: Mapping[int, float]
    tolerance: float = DEFAULT_TOLERANCE
    # Provenance of the numbers, e.g. "median of 11 nightly runs, 2026-07-30..2026-08-11".
    recorded_from: str = ""

    def floor(self, batch_size: int) -> Optional[float]:
        """Lowest throughput accepted at `batch_size`, or None if unrecorded."""
        expected = self.output_throughput.get(batch_size)
        return None if expected is None else expected * (1.0 - self.tolerance)


@dataclass(frozen=True)
class BaselineCheck:
    """Outcome of comparing one benchmark run against its baseline."""

    label: str
    markdown: str
    regressions: Sequence[str]

    @property
    def ok(self) -> bool:
        return not self.regressions

    def failure_message(self) -> str:
        lines = [f"{self.label}: output throughput below recorded baseline"]
        lines.extend(f"  {r}" for r in self.regressions)
        lines.append(
            "Update the baseline in the test file if the drop is an accepted "
            "trade-off; otherwise this is a performance regression."
        )
        return "\n".join(lines)


def _field(result: Any, name: str) -> Any:
    """Read `name` from a BenchmarkResult object or a raw result dict."""
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def check_output_throughput(
    results: Iterable[Any],
    baseline: Optional[ThroughputBaseline],
    label: str,
) -> BaselineCheck:
    """Compare measured output throughput against `baseline`.

    `results` holds one entry per batch size, either `BenchmarkResult` objects
    or the raw dicts `bench_one_batch_server` writes. Never raises: the caller
    writes `markdown` to the step summary first, then fails on `regressions`,
    so a failing run still publishes its numbers. Every batch size is reported,
    so one run surfaces all regressions rather than only the first.
    """
    measured = {
        int(_field(r, "batch_size")): float(_field(r, "output_throughput") or 0.0)
        for r in results
    }

    header = f"#### perf gate: {label}\n"
    if baseline is None:
        header += "No baseline recorded yet, reporting only.\n\n"
    else:
        provenance = f" ({baseline.recorded_from})" if baseline.recorded_from else ""
        header += (
            f"Baseline{provenance}, "
            f"tolerance {baseline.tolerance * 100:.0f}%.\n\n"
        )

    rows = [
        "| batch size | output throughput (tok/s) | baseline | min allowed | status |",
        "| ---------- | ------------------------- | -------- | ----------- | ------ |",
    ]
    regressions = []

    batch_sizes = sorted(
        set(measured) | (set(baseline.output_throughput) if baseline else set())
    )
    for batch_size in batch_sizes:
        expected = baseline.output_throughput.get(batch_size) if baseline else None
        floor = baseline.floor(batch_size) if baseline else None
        actual = measured.get(batch_size)

        if actual is None:
            regressions.append(f"bs={batch_size}: no measurement reported")
            rows.append(f"| {batch_size} | n/a | {expected:.1f} | {floor:.1f} | MISSING |")
            continue

        if floor is None:
            rows.append(f"| {batch_size} | {actual:.2f} | n/a | n/a | no baseline |")
            continue

        if actual < floor:
            delta = (actual - expected) / expected * 100
            regressions.append(
                f"bs={batch_size}: {actual:.2f} tok/s < {floor:.2f} tok/s floor "
                f"(baseline {expected:.2f} tok/s, {delta:+.1f}%)"
            )
            status = f"FAIL ({delta:+.1f}%)"
        else:
            status = f"pass ({(actual - expected) / expected * 100:+.1f}%)"
        rows.append(
            f"| {batch_size} | {actual:.2f} | {expected:.1f} | {floor:.1f} | {status} |"
        )

    return BaselineCheck(
        label=label, markdown=header + "\n".join(rows) + "\n", regressions=regressions
    )
