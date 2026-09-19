import ast
import unittest
from types import SimpleNamespace

from scripts.lint.check_registered_tests import taxonomy_errors


def _registry(suite: str, est_time: int = 1, backend: str = "CUDA"):
    return SimpleNamespace(
        effective_suite=suite,
        est_time=est_time,
        backend=SimpleNamespace(name=backend),
    )


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

    def test_topic_directories_are_free_form(self):
        for path in (
            "test/registered/lora/test_example.py",
            "test/registered/perf/test_example.py",
            "test/registered/e2e/models/test_example.py",
            "test/registered/amd/test_example.py",
        ):
            with self.subTest(path=path):
                errors = taxonomy_errors(
                    path, [_registry("base-b-test-1-gpu-large")], self.tree
                )
                self.assertEqual(errors, [])

    def test_unit_tests_mirror_the_srt_tree(self):
        registry = [_registry("base-a-test-cpu", backend="CPU")]
        self.assertEqual(
            taxonomy_errors(
                "test/registered/unit/mem_cache/test_example.py", registry, self.tree
            ),
            [],
        )
        self.assertTrue(
            taxonomy_errors("test/registered/unit/test_example.py", registry, self.tree)
        )

    def test_unit_cost_and_suite_limits_still_apply(self):
        unit_path = "test/registered/unit/layers/test_example.py"
        unit_registry = [_registry("base-b-unit-test-1-gpu-large")]

        # est_time over the unit budget
        self.assertTrue(
            taxonomy_errors(
                unit_path,
                [_registry("base-a-test-cpu", est_time=300, backend="CPU")],
                self.tree,
            )
        )
        # a GPU suite that is not a unit suite
        self.assertTrue(
            taxonomy_errors(
                unit_path, [_registry("base-b-test-1-gpu-large")], self.tree
            )
        )
        # launching a server
        self.assertTrue(
            taxonomy_errors(
                unit_path, unit_registry, ast.parse("popen_launch_server()")
            )
        )
        self.assertEqual(taxonomy_errors(unit_path, unit_registry, self.tree), [])


if __name__ == "__main__":
    unittest.main()
