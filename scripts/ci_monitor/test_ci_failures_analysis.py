"""AMD job-name cutover tests.

Run with:
    python -m unittest discover -s scripts/ci_monitor -p 'test_ci_failures_analysis.py'
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_failures_analysis import _filter_legacy_amd_job_rows  # noqa: E402


class TestFilterLegacyAmdJobRows(unittest.TestCase):
    def test_drops_legacy_names_and_nested_utilities(self):
        rows = {
            "stage-b-test-1-gpu-small-amd-rocm720 (linux-mi300-1gpu-sglang, 0)": {},
            "nightly-accuracy-2-gpu-rocm720 (rocm724)": {},
            "nightly-accuracy-2-gpu-rocm724": {},
            "nightly-test-1-gpu-unit (rocm724)": {},
            "call-pr-test-amd-rocm720 / call-pr-test-amd-extra-rocm720 / extra-a-test-1-gpu-small-amd (linux-mi300-1gpu-sglang)": {},
            "wait-for-stage-a-amd": {},
            "call-pr-test-amd-extra / pr-test-amd-extra-finish": {},
            "call-pr-test-amd-extra / call-gate / pr-gate": {},
        }

        self.assertEqual(_filter_legacy_amd_job_rows(rows), {})

    def test_keeps_current_flavors_and_nested_callers_separate(self):
        new_success = {"current_streak": 0}
        rows = {
            "stage-b-test-1-gpu-small-amd (rocm724, linux-mi300-1gpu-sglang, 0)": new_success,
            "nightly-accuracy-2-gpu (rocm720, linux-mi300-2gpu-sglang)": {
                "current_streak": 1
            },
            "call-pr-test-amd-rocm720 / stage-c-test-4-gpu-amd (rocm724, linux-mi300-4gpu-sglang, 0)": {
                "current_streak": 0
            },
            "call-pr-test-amd-rocm720 / call-pr-test-amd-extra / extra-a-test-1-gpu-small-amd (rocm724, linux-mi300-1gpu-sglang)": {
                "current_streak": 0
            },
        }

        filtered = _filter_legacy_amd_job_rows(rows)

        self.assertEqual(set(filtered), set(rows))
        self.assertIs(
            filtered[
                "stage-b-test-1-gpu-small-amd (rocm724, linux-mi300-1gpu-sglang, 0)"
            ],
            new_success,
        )


if __name__ == "__main__":
    unittest.main()
