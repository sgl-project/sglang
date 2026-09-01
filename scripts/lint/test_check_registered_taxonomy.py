import ast
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from check_registered_tests import _changed_registered_files, taxonomy_errors


def registry(backend="CUDA", suite="base-b-test-1-gpu-small", est_time=10):
    return SimpleNamespace(
        backend=SimpleNamespace(name=backend),
        effective_suite=suite,
        est_time=est_time,
    )


class TestRegisteredTaxonomy(unittest.TestCase):
    def check(self, path, registries, source="pass"):
        return taxonomy_errors(path, registries, ast.parse(source))

    def test_requires_kind_and_subsystem(self):
        errors = self.check("test/registered/models/test_model.py", [registry()])
        self.assertTrue(any("<kind>/<subsystem>" in error for error in errors))

    def test_changed_files_uses_rename_destination(self):
        with patch(
            "check_registered_tests._git_lines",
            return_value=[
                "R100\ttest/registered/models/test_old.py\t"
                "test/registered/e2e/models/test_new.py",
                "A\ttest/registered/unit/cache/test_cache.py",
                "A\tdocs/test_notes.py",
            ],
        ):
            self.assertEqual(
                _changed_registered_files(),
                {
                    "test/registered/e2e/models/test_new.py",
                    "test/registered/unit/cache/test_cache.py",
                },
            )

    def test_accepts_e2e_registration(self):
        errors = self.check("test/registered/e2e/models/test_model.py", [registry()])
        self.assertEqual(errors, [])

    def test_unit_is_cpu_only_and_bounded(self):
        path = "test/registered/unit/scheduler/test_queue.py"
        self.assertEqual(self.check(path, [registry("CPU", "base-a-test-cpu", 60)]), [])
        errors = self.check(path, [registry(est_time=61)])
        self.assertTrue(any("only CPU" in error for error in errors))
        self.assertTrue(any("<= 60" in error for error in errors))

    def test_unit_cannot_launch_server(self):
        errors = self.check(
            "test/registered/unit/server/test_launch.py",
            [registry("CPU", "base-a-test-cpu")],
            "popen_launch_server('model')",
        )
        self.assertTrue(any("launch a server" in error for error in errors))

    def test_kernel_requires_kernel_suite(self):
        path = "test/registered/kernel/attention/test_attention.py"
        self.assertEqual(
            self.check(path, [registry(suite="base-b-kernel-unit-test-1-gpu-large")]),
            [],
        )
        self.assertTrue(self.check(path, [registry()]))

    def test_accuracy_and_perf_require_scheduled_cadence(self):
        for kind in ("accuracy", "perf"):
            path = f"test/registered/{kind}/models/test_model.py"
            self.assertEqual(
                self.check(path, [registry(suite="nightly-test-1-gpu-large")]), []
            )
            self.assertTrue(self.check(path, [registry()]))

    def test_stress_requires_stress_or_weekly(self):
        path = "test/registered/stress/models/test_model.py"
        self.assertEqual(self.check(path, [registry(suite="stress")]), [])
        self.assertEqual(
            self.check(path, [registry(suite="weekly-test-1-gpu-large")]), []
        )
        self.assertTrue(self.check(path, [registry()]))


if __name__ == "__main__":
    unittest.main()
