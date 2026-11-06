"""Handoff helpers for DeepSeek-V4 two-pass (FIA unset vs FIA=0) comparisons.

The pass-1 case records its metrics to a JSON file under the per-run output
dir (METRICS_DATA_FILE); the pass-2 case loads that snapshot and asserts the
second run does not regress: performance within PERF_TOLERANCE (2%) and
accuracy within ACC_TOLERANCE (1%).
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

PERF_TOLERANCE = 0.02
ACC_TOLERANCE = 0.01

PASS1_SUFFIX = "_fia_unset"
_PASS1_METRICS_FILENAME = "_pass1_metrics.json"


def _handoff_path(case_id: str) -> str:
    # Both passes share the per-run output root; pass-1's snapshot lives in the
    # pass-1 tc_name dir (its own METRICS_DATA_FILE), which pass-2 reaches via
    # the sibling directory under the same parent.
    base = os.environ.get("METRICS_DATA_FILE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dual_pass_output"
    )
    return os.path.join(
        os.path.dirname(base),
        f"{case_id}{PASS1_SUFFIX}",
        f"{case_id}{_PASS1_METRICS_FILENAME}",
    )


_PASS1_IN_PROCESS: dict = {}


def record_pass1(case_id: str, metrics: dict) -> None:
    """Persist the pass-1 metrics snapshot for the pass-2 comparison."""
    _PASS1_IN_PROCESS[case_id] = metrics
    path = _handoff_path(case_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        logger.info("record_pass1: saved %s to %s", sorted(metrics), path)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("record_pass1: failed to persist %s: %s", path, e)


def load_pass1(case_id: str) -> dict:
    # Same-process passes (single-node cases run both classes in one worker)
    # share the in-memory snapshot; multi-node passes run in separate pods and
    # hand off via the JSON file under the shared per-run output dir.
    if case_id in _PASS1_IN_PROCESS:
        return _PASS1_IN_PROCESS[case_id]
    path = _handoff_path(case_id)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def assert_no_regression(testcase, case_id: str, metrics: dict) -> None:
    """Assert pass-2 metrics do not regress against the pass-1 snapshot.

    Perf: mean_tpot must stay within 2% (higher is worse) and total_tps must
    stay within 2% (lower is worse). Accuracy: within 1% (lower is worse).
    """
    pass1 = load_pass1(case_id)

    if "mean_tpot" in pass1 and metrics.get("mean_tpot"):
        limit = pass1["mean_tpot"] * (1 + PERF_TOLERANCE)
        testcase.assertLessEqual(
            float(metrics["mean_tpot"]),
            limit,
            f"TPOT regression: pass2 {metrics['mean_tpot']} > "
            f"pass1 {pass1['mean_tpot']} * {1 + PERF_TOLERANCE}",
        )
    if "total_tps" in pass1 and metrics.get("total_tps"):
        limit = pass1["total_tps"] * (1 - PERF_TOLERANCE)
        testcase.assertGreaterEqual(
            float(metrics["total_tps"]),
            limit,
            f"Throughput regression: pass2 {metrics['total_tps']} < "
            f"pass1 {pass1['total_tps']} * {1 - PERF_TOLERANCE}",
        )
    if "accuracy" in pass1 and metrics.get("accuracy"):
        limit = pass1["accuracy"] * (1 - ACC_TOLERANCE)
        testcase.assertGreaterEqual(
            float(metrics["accuracy"]),
            limit,
            f"Accuracy regression: pass2 {metrics['accuracy']} < "
            f"pass1 {pass1['accuracy']} * {1 - ACC_TOLERANCE}",
        )
    logger.info("assert_no_regression: pass2 metrics within tolerance vs pass1")
