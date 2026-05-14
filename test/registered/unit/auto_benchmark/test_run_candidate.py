import sys
import time
import unittest
from pathlib import Path
from unittest import mock

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from auto_benchmark import AutoBenchmarkTestCase

from sglang.auto_benchmark_lib import SearchDeadlineExceeded, run_candidate
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=6, suite="stage-b-test-1-gpu-small-amd")


class TestAutoBenchmarkRunCandidate(AutoBenchmarkTestCase):
    def test_run_candidate_binary_search_avoids_rounding_loop(self):
        benchmark_cfg = {
            "qps": {"lower": 1.0, "upper": 1.00000001, "tolerance": 1e-12},
            "max_concurrency": [None],
        }
        calls = []

        with mock.patch(
            "sglang.auto_benchmark_lib.run_trial",
            side_effect=self._make_run_trial_side_effect(calls),
        ):
            records = run_candidate(**self._run_candidate_kwargs(benchmark_cfg))

        self.assertLess(len(calls), 40)
        self.assertEqual(len(records), len(calls))

    def test_run_candidate_binary_search_respects_max_rounds(self):
        benchmark_cfg = {
            "qps": {"lower": 1.0, "upper": 32.0, "tolerance": 1e-12, "max_rounds": 2},
            "max_concurrency": [None],
        }
        calls = []

        with mock.patch(
            "sglang.auto_benchmark_lib.run_trial",
            side_effect=self._make_run_trial_side_effect(calls),
        ):
            records = run_candidate(**self._run_candidate_kwargs(benchmark_cfg))

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(records), 2)

    def test_run_candidate_stops_when_search_budget_is_exhausted(self):
        benchmark_cfg = {
            "qps": {"lower": 1.0, "upper": 2.0, "tolerance": 0.1},
            "max_concurrency": [None],
        }

        with self.assertRaises(SearchDeadlineExceeded):
            run_candidate(
                **self._run_candidate_kwargs(
                    benchmark_cfg,
                    search_deadline=time.time() - 1.0,
                    search_budget_hours=0.1,
                )
            )

    def test_run_candidate_resume_skips_existing_fixed_trials(self):
        benchmark_cfg = {
            "qps": [1.0, 2.0],
            "max_concurrency": [None],
        }
        existing_records = [self._trial_record(1.0)]
        calls = []

        with mock.patch(
            "sglang.auto_benchmark_lib.run_trial",
            side_effect=self._make_run_trial_side_effect(
                calls,
                output_throughput=2.0,
                mean_ttft_ms=2.0,
                mean_tpot_ms=2.0,
            ),
        ):
            records = run_candidate(
                **self._run_candidate_kwargs(
                    benchmark_cfg,
                    existing_records=existing_records,
                )
            )

        self.assertEqual(calls, [2.0])
        self.assertEqual([record["requested_qps"] for record in records], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
