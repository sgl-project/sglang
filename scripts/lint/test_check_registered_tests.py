import ast
import unittest

from scripts.lint.check_registered_tests import taxonomy_errors


class TestRegisteredTestTaxonomy(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse("")

    def test_plural_kernel_ops_layout_is_accepted(self):
        errors = taxonomy_errors(
            "test/registered/kernels/ops/attention/test_example.py",
            self.tree,
        )
        self.assertEqual(errors, [])

    def test_plural_kernel_benchmark_layout_is_accepted(self):
        errors = taxonomy_errors(
            "test/registered/kernels/benchmark/attention/bench_example.py",
            self.tree,
        )
        self.assertEqual(errors, [])

    def test_singular_kernel_root_is_rejected(self):
        errors = taxonomy_errors(
            "test/registered/kernel/attention/test_example.py",
            self.tree,
        )
        self.assertTrue(errors)

    def test_kernel_group_is_required(self):
        errors = taxonomy_errors(
            "test/registered/kernels/ops/test_example.py",
            self.tree,
        )
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
