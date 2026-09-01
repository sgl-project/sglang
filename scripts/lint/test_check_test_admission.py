import datetime as dt
import pathlib
import tempfile
import unittest

from check_test_admission import check_file


class TestCheckTestAdmission(unittest.TestCase):
    def check_source(self, source: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "test_example.py"
            path.write_text(source, encoding="utf-8")
            return check_file(path, today=dt.date(2026, 8, 31))

    def test_rejects_unowned_disabled_registration(self):
        errors = self.check_source(
            'register_cuda_ci(est_time=10, stage="base-b", '
            'runner_config="1-gpu-small", disabled="temporarily disabled")\n'
        )
        self.assertTrue(any("reference an issue" in error for error in errors))
        self.assertTrue(any("until YYYY-MM-DD" in error for error in errors))

    def test_accepts_owned_unexpired_registration(self):
        errors = self.check_source(
            'register_cuda_ci(est_time=10, stage="base-b", '
            'runner_config="1-gpu-small", disabled="see #123; until 2026-09-30")\n'
        )
        self.assertEqual(errors, [])

    def test_rejects_expired_registration(self):
        errors = self.check_source(
            'register_cuda_ci(est_time=10, stage="base-b", '
            'runner_config="1-gpu-small", disabled="see #123; until 2026-08-01")\n'
        )
        self.assertTrue(any("expired" in error for error in errors))

    def test_requires_mixed_backend_rationale(self):
        errors = self.check_source(
            'register_cuda_ci(est_time=10, stage="base-b", '
            'runner_config="1-gpu-small")\n'
            'register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")\n'
        )
        self.assertTrue(any("backend-specific:" in error for error in errors))

    def test_accepts_mixed_backend_rationale(self):
        errors = self.check_source(
            'register_cuda_ci(est_time=10, stage="base-b", '
            'runner_config="1-gpu-small")\n'
            "# Backend-specific: exercises the ROCm-only AITER path.\n"
            'register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")\n'
        )
        self.assertEqual(errors, [])

    def test_rejects_oversized_default_pr_registration(self):
        errors = self.check_source(
            'register_cuda_ci(est_time=400, stage="base-c", '
            'runner_config="4-gpu-h100")\n'
        )
        self.assertTrue(
            any("1600 weighted accelerator-seconds" in error for error in errors)
        )

    def test_legacy_amd_suite_uses_the_same_budget(self):
        errors = self.check_source(
            'register_amd_ci(est_time=400, suite="stage-c-test-4-gpu-amd")\n'
        )
        self.assertTrue(
            any("1600 weighted accelerator-seconds" in error for error in errors)
        )

    def test_npu_suite_uses_the_same_budget(self):
        errors = self.check_source(
            'register_npu_ci(est_time=400, suite="base-c-test-4-npu-a3")\n'
        )
        self.assertTrue(
            any("1600 weighted accelerator-seconds" in error for error in errors)
        )

    def test_accepts_cost_override(self):
        errors = self.check_source(
            "# ci-cost-override: one 4-GPU integration smoke is required.\n"
            'register_cuda_ci(est_time=400, stage="base-c", '
            'runner_config="4-gpu-h100")\n'
        )
        self.assertEqual(errors, [])

    def test_unconditional_skip_also_has_a_ttl(self):
        errors = self.check_source(
            'import unittest\n@unittest.skip("see #55; until 2026-09-15")\n'
            "class TestThing: pass\n"
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
