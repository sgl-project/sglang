"""CPU-only unit tests for the regression comparison math.

These intentionally have no torch/CUDA dependency so they run anywhere:

    python -m pytest sgl-kernel/benchmark/kernel_bench_regression/test_compare.py
"""

import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kernel_bench_regression import compare as compare_mod
else:
    from . import compare as compare_mod


def _gt():
    return {
        "tolerance": 0.05,
        "cases": {
            "latency_kernel": {
                "metric": "us",
                "higher_is_better": False,
                "measurements": {"n=1": 100.0, "n=16": 200.0},
            },
            "throughput_kernel": {
                "metric": "TFLOPs",
                "higher_is_better": True,
                "measurements": {"n=1": 50.0, "n=16": 80.0},
            },
        },
    }


def _measured(cases):
    return {"cases": cases}


class CompareTest(unittest.TestCase):
    def test_within_tolerance_passes(self):
        measured = _measured(
            {
                "latency_kernel": {
                    "metric": "us",
                    "higher_is_better": False,
                    "measurements": {"n=1": 103.0, "n=16": 196.0},
                },
                "throughput_kernel": {
                    "metric": "TFLOPs",
                    "higher_is_better": True,
                    "measurements": {"n=1": 49.0, "n=16": 82.0},
                },
            }
        )
        report = compare_mod.compare_results(_gt(), measured, tolerance=0.05)
        self.assertTrue(report.passed)
        self.assertEqual(report.regressions, [])

    def test_latency_regression_fails(self):
        # 200 -> 220 is +10% latency == worse than 5% tolerance.
        measured = _measured(
            {
                "latency_kernel": {
                    "metric": "us",
                    "higher_is_better": False,
                    "measurements": {"n=1": 100.0, "n=16": 220.0},
                },
            }
        )
        report = compare_mod.compare_results(_gt(), measured, tolerance=0.05)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.regressions), 1)
        d = report.regressions[0]
        self.assertEqual(d.case_id, "latency_kernel")
        self.assertEqual(d.config, "n=16")
        self.assertAlmostEqual(d.regression_ratio, 0.10, places=6)

    def test_throughput_regression_fails(self):
        # 80 -> 70 is a 12.5% throughput drop == regression.
        measured = _measured(
            {
                "throughput_kernel": {
                    "metric": "TFLOPs",
                    "higher_is_better": True,
                    "measurements": {"n=1": 50.0, "n=16": 70.0},
                },
            }
        )
        report = compare_mod.compare_results(_gt(), measured, tolerance=0.05)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.regressions), 1)
        self.assertAlmostEqual(report.regressions[0].regression_ratio, 0.125, places=6)

    def test_improvement_is_not_a_regression(self):
        measured = _measured(
            {
                "latency_kernel": {
                    "metric": "us",
                    "higher_is_better": False,
                    "measurements": {"n=1": 80.0, "n=16": 150.0},
                },
            }
        )
        report = compare_mod.compare_results(_gt(), measured, tolerance=0.05)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.improvements), 2)

    def test_missing_config_is_reported_not_failed(self):
        measured = _measured(
            {
                "latency_kernel": {
                    "metric": "us",
                    "higher_is_better": False,
                    "measurements": {"n=1": 100.0},  # n=16 missing
                },
            }
        )
        report = compare_mod.compare_results(_gt(), measured, tolerance=0.05)
        self.assertTrue(report.passed)
        self.assertIn("latency_kernel::n=16", report.missing)

    def test_zero_ground_truth_does_not_divide_by_zero(self):
        gt = {
            "tolerance": 0.05,
            "cases": {
                "k": {
                    "metric": "us",
                    "higher_is_better": False,
                    "measurements": {"n=1": 0.0},
                }
            },
        }
        measured = _measured(
            {
                "k": {
                    "metric": "us",
                    "higher_is_better": False,
                    "measurements": {"n=1": 5.0},
                }
            }
        )
        report = compare_mod.compare_results(gt, measured, tolerance=0.05)
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
