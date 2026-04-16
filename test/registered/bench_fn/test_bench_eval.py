import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestBenchEvalImports(unittest.TestCase):
    def test_subpackage_importable(self):
        from sglang.benchmark import eval_harness  # noqa: F401

    def test_public_symbols(self):
        from sglang.benchmark.eval_harness import (
            BenchServingLM,
            merge_report,
            write_report,
        )
        self.assertTrue(callable(BenchServingLM))
        self.assertTrue(callable(merge_report))
        self.assertTrue(callable(write_report))


if __name__ == "__main__":
    unittest.main()
