import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.perf_baseline import (
    DEFAULT_TOLERANCE,
    ThroughputBaseline,
    check_output_throughput,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BASELINE = ThroughputBaseline(
    {1: 100.0, 8: 800.0},
    tolerance=0.15,
    recorded_from="unit test",
)


def _result(batch_size, output_throughput):
    return {"batch_size": batch_size, "output_throughput": output_throughput}


class TestCheckOutputThroughput(unittest.TestCase):
    def test_within_tolerance_passes(self):
        # 86 is a 14% drop at bs=1: under the baseline but above the floor.
        check = check_output_throughput(
            [_result(1, 86.0), _result(8, 820.0)], BASELINE, "case"
        )
        self.assertTrue(check.ok)
        self.assertIn("| 1 | 86.00 | 100.0 | 85.0 |", check.markdown)

    def test_drop_past_tolerance_fails(self):
        check = check_output_throughput(
            [_result(1, 84.0), _result(8, 800.0)], BASELINE, "case"
        )
        self.assertFalse(check.ok)
        self.assertEqual(len(check.regressions), 1)
        self.assertIn("bs=1", check.regressions[0])
        self.assertIn("-16.0%", check.regressions[0])
        self.assertIn("case", check.failure_message())

    def test_every_regression_is_reported(self):
        check = check_output_throughput(
            [_result(1, 10.0), _result(8, 80.0)], BASELINE, "case"
        )
        self.assertEqual(len(check.regressions), 2)

    def test_missing_measurement_fails(self):
        check = check_output_throughput([_result(1, 100.0)], BASELINE, "case")
        self.assertFalse(check.ok)
        self.assertIn("bs=8: no measurement", check.regressions[0])

    def test_batch_size_without_baseline_is_reported_not_gated(self):
        check = check_output_throughput(
            [_result(1, 100.0), _result(8, 800.0), _result(64, 1.0)], BASELINE, "case"
        )
        self.assertTrue(check.ok)
        self.assertIn("no baseline", check.markdown)

    def test_no_baseline_reports_only(self):
        check = check_output_throughput([_result(1, 1.0)], None, "case")
        self.assertTrue(check.ok)
        self.assertIn("No baseline recorded yet", check.markdown)

    def test_reads_objects_as_well_as_dicts(self):
        class Result:
            def __init__(self, batch_size, output_throughput):
                self.batch_size = batch_size
                self.output_throughput = output_throughput

        check = check_output_throughput(
            [Result(1, 100.0), Result(8, 800.0)], BASELINE, "case"
        )
        self.assertTrue(check.ok)

    def test_floor_follows_tolerance(self):
        baseline = ThroughputBaseline({4: 200.0})
        self.assertAlmostEqual(baseline.floor(4), 200.0 * (1 - DEFAULT_TOLERANCE))
        self.assertIsNone(baseline.floor(16))


if __name__ == "__main__":
    unittest.main()
