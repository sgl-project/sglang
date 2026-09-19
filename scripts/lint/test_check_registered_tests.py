import ast
import unittest
from types import SimpleNamespace

from scripts.lint.check_registered_tests import taxonomy_errors


def _registry(suite: str):
    return SimpleNamespace(effective_suite=suite, est_time=1)


class TestRegisteredTestTaxonomy(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse("")
        self.kernel_registry = [_registry("base-b-kernel-unit-test-1-gpu-large")]

    def test_plural_kernel_ops_layout_is_accepted(self):
        errors = taxonomy_errors(
            "test/registered/kernels/ops/attention/test_example.py",
            self.kernel_registry,
            self.tree,
        )
        self.assertEqual(errors, [])

    def test_plural_kernel_benchmark_layout_is_accepted(self):
        errors = taxonomy_errors(
            "test/registered/kernels/benchmark/attention/bench_example.py",
            [_registry("base-b-kernel-benchmark-test-1-gpu-large")],
            self.tree,
        )
        self.assertEqual(errors, [])

    def test_singular_kernel_root_is_rejected(self):
        errors = taxonomy_errors(
            "test/registered/kernel/attention/test_example.py",
            self.kernel_registry,
            self.tree,
        )
        self.assertTrue(errors)

    def test_kernel_group_is_required(self):
        errors = taxonomy_errors(
            "test/registered/kernels/ops/test_example.py",
            self.kernel_registry,
            self.tree,
        )
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
