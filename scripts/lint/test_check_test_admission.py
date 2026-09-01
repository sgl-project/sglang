import datetime as dt
import pathlib
import tempfile
import unittest

from check_test_admission import _added_lines_by_file, check_file


class TestCheckTestAdmission(unittest.TestCase):
    def check_source(
        self, source: str, *, changed_lines: set[int] | None = None
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "test_example.py"
            path.write_text(source, encoding="utf-8")
            return check_file(
                path, today=dt.date(2026, 8, 31), changed_lines=changed_lines
            )

    def test_zero_context_diff_tracks_only_added_new_file_lines(self):
        changed = _added_lines_by_file(
            """diff --git a/test/registered/e2e/x/test_a.py b/test/registered/e2e/x/test_a.py
--- a/test/registered/e2e/x/test_a.py
+++ b/test/registered/e2e/x/test_a.py
@@ -1 +1,2 @@
 unchanged
+added
diff --git a/test/registered/e2e/x/test_old.py b/test/registered/e2e/x/test_new.py
similarity index 100%
rename from test/registered/e2e/x/test_old.py
rename to test/registered/e2e/x/test_new.py
"""
        )
        self.assertEqual(
            changed,
            {
                pathlib.Path("test/registered/e2e/x/test_a.py"): {2},
            },
        )

    def test_ignores_untouched_legacy_registration(self):
        errors = self.check_source(
            'register_cuda_ci(est_time=400, stage="base-c", '
            'runner_config="4-gpu-h100")\n'
            "value = 1\n",
            changed_lines={2},
        )
        self.assertEqual(errors, [])

    def test_checks_registration_when_any_call_line_changes(self):
        errors = self.check_source(
            "register_cuda_ci(\n"
            "    est_time=400,\n"
            '    stage="base-c", runner_config="4-gpu-h100"\n'
            ")\n",
            changed_lines={2},
        )
        self.assertTrue(
            any("weighted accelerator-seconds" in error for error in errors)
        )

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
