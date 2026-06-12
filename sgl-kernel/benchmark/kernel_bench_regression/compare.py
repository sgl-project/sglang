"""Pure ground-truth comparison logic (no torch / CUDA dependency).

Kept import-light on purpose so the regression math can be unit-tested on any
machine (see ``test_compare.py``) without a GPU.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Delta:
    case_id: str
    config: str
    metric: str
    higher_is_better: bool
    gt_value: float
    measured_value: float
    # Signed regression ratio: > 0 means measured is WORSE than ground truth.
    regression_ratio: float


@dataclass
class CompareReport:
    regressions: List[Delta]
    improvements: List[Delta]
    within_tolerance: List[Delta]
    missing: List[str]  # "case_id::config" present in GT but not measured
    new_measurements: List[str]  # present in measured but not in GT (informational)

    @property
    def passed(self) -> bool:
        return not self.regressions


def _regression_ratio(
    gt_value: float, measured_value: float, higher_is_better: bool
) -> float:
    """Fraction by which ``measured`` is worse than ``gt``. Positive == worse.

    For throughput (higher_is_better) a drop is worse: (gt - measured) / gt.
    For latency a rise is worse: (measured - gt) / gt.
    """
    if gt_value == 0:
        # Degenerate ground truth -- treat any nonzero measurement as matching to
        # avoid a divide-by-zero false alarm.
        return 0.0
    if higher_is_better:
        return (gt_value - measured_value) / gt_value
    return (measured_value - gt_value) / gt_value


def compare_results(
    ground_truth: dict,
    measured: dict,
    tolerance: float,
) -> CompareReport:
    """Compare a fresh ``measured`` run against ``ground_truth`` with ``tolerance``.

    A config regresses when its kernel is more than ``tolerance`` (e.g. 0.05 == 5%)
    worse than the ground truth in the metric's "good" direction. Configs that are
    missing on either side are reported but never fail the run -- config sets evolve,
    and the goal is to catch *regressions on shared configs*, not enforce identical
    coverage.
    """
    regressions: List[Delta] = []
    improvements: List[Delta] = []
    within: List[Delta] = []
    missing: List[str] = []
    new_measurements: List[str] = []

    gt_cases: Dict[str, dict] = ground_truth.get("cases", {})
    m_cases: Dict[str, dict] = measured.get("cases", {})

    for case_id, gt_case in gt_cases.items():
        higher_is_better = gt_case["higher_is_better"]
        metric = gt_case["metric"]
        gt_meas: Dict[str, Optional[float]] = gt_case.get("measurements", {})
        m_case = m_cases.get(case_id)
        m_meas: Dict[str, Optional[float]] = (m_case or {}).get("measurements", {})

        for config, gt_value in gt_meas.items():
            if gt_value is None:
                continue
            measured_value = m_meas.get(config)
            if measured_value is None:
                missing.append(f"{case_id}::{config}")
                continue
            ratio = _regression_ratio(gt_value, measured_value, higher_is_better)
            delta = Delta(
                case_id=case_id,
                config=config,
                metric=metric,
                higher_is_better=higher_is_better,
                gt_value=gt_value,
                measured_value=measured_value,
                regression_ratio=ratio,
            )
            if ratio > tolerance:
                regressions.append(delta)
            elif ratio < -tolerance:
                improvements.append(delta)
            else:
                within.append(delta)

    # Informational: configs measured this run that the ground truth doesn't know.
    for case_id, m_case in m_cases.items():
        gt_meas = gt_cases.get(case_id, {}).get("measurements", {})
        for config, value in m_case.get("measurements", {}).items():
            if value is not None and config not in gt_meas:
                new_measurements.append(f"{case_id}::{config}")

    return CompareReport(
        regressions=regressions,
        improvements=improvements,
        within_tolerance=within,
        missing=missing,
        new_measurements=new_measurements,
    )


def format_report(report: CompareReport, tolerance: float) -> str:
    """Render a human-readable summary for CI logs."""
    lines: List[str] = []
    pct = f"{tolerance * 100:.1f}%"
    lines.append(f"Kernel benchmark regression report (tolerance ±{pct})")
    lines.append("=" * 72)

    def _row(d: Delta) -> str:
        arrow = "↑good" if d.higher_is_better else "↓good"
        return (
            f"  {d.case_id}::{d.config}\n"
            f"      gt={d.gt_value:.4g} {d.metric}  measured={d.measured_value:.4g} {d.metric} "
            f"({arrow})  delta={d.regression_ratio * 100:+.2f}% worse"
        )

    if report.regressions:
        lines.append(f"\nREGRESSIONS ({len(report.regressions)}):")
        for d in report.regressions:
            lines.append(_row(d))
    if report.improvements:
        lines.append(f"\nImprovements ({len(report.improvements)}):")
        for d in report.improvements:
            lines.append(_row(d))
    lines.append(
        f"\nWithin tolerance: {len(report.within_tolerance)}  |  "
        f"Regressions: {len(report.regressions)}  |  "
        f"Improvements: {len(report.improvements)}"
    )
    if report.missing:
        lines.append(
            f"\nMissing this run ({len(report.missing)}) -- not counted as regressions:"
        )
        for key in report.missing:
            lines.append(f"  {key}")
    if report.new_measurements:
        lines.append(
            f"\nNew configs not in ground truth ({len(report.new_measurements)}):"
        )
        for key in report.new_measurements:
            lines.append(f"  {key}")

    verdict = "PASS" if report.passed else "FAIL"
    lines.append("\n" + "=" * 72)
    lines.append(f"VERDICT: {verdict}")
    return "\n".join(lines)
