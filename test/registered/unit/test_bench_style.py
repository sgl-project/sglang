import importlib.util
import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-b-test-cpu")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_checker():
    path = os.path.join(_REPO_ROOT, "scripts", "ci", "check_bench_style.py")
    spec = importlib.util.spec_from_file_location("check_bench_style", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBenchStyleConformance(CustomTestCase):
    """All JIT kernel benchmarks conform to the unified marker style.

    Mirrors the `check-bench-style` pre-commit hook so the gate also runs in the
    CPU CI suite. Fails on style violations and on stale LEGACY_ALLOWLIST entries
    (files that now conform must be removed from the allowlist — the ratchet only
    shrinks).
    """

    def test_all_benchmarks_conform(self):
        checker = _load_checker()
        cwd = os.getcwd()
        os.chdir(_REPO_ROOT)
        try:
            rc = checker.main()
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0, "benchmark style check failed; see output above")


if __name__ == "__main__":
    unittest.main()
