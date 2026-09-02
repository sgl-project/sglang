"""Unit tests for CompilationCounter in srt/compilation/compilation_counter.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.compilation.compilation_counter import (
    CompilationCounter,
    compilation_counter,
)
from sglang.test.test_utils import CustomTestCase


class TestCompilationCounterDefaults(CustomTestCase):
    def test_all_fields_start_at_zero(self):
        c = CompilationCounter()
        for field in c.__dataclass_fields__:
            self.assertEqual(getattr(c, field), 0, msg=f"{field} should default to 0")

    def test_all_expected_fields_exist(self):
        c = CompilationCounter()
        expected = [
            "num_models_seen",
            "num_graphs_seen",
            "num_piecewise_graphs_seen",
            "num_piecewise_capturable_graphs_seen",
            "num_backend_compilations",
            "num_gpu_runner_capture_triggers",
            "num_cudagraph_captured",
            "num_inductor_compiles",
            "num_eager_compiles",
            "num_cache_entries_updated",
            "num_compiled_artifacts_saved",
            "dynamo_as_is_count",
        ]
        for name in expected:
            self.assertTrue(hasattr(c, name), msg=f"missing field {name}")


class TestCompilationCounterClone(CustomTestCase):
    def test_clone_produces_independent_copy(self):
        c = CompilationCounter()
        c.num_graphs_seen = 5
        cloned = c.clone()
        cloned.num_graphs_seen = 99
        self.assertEqual(c.num_graphs_seen, 5)

    def test_clone_copies_all_values(self):
        c = CompilationCounter()
        c.num_models_seen = 3
        c.num_backend_compilations = 7
        cloned = c.clone()
        self.assertEqual(cloned.num_models_seen, 3)
        self.assertEqual(cloned.num_backend_compilations, 7)

    def test_clone_of_fresh_counter_is_all_zeros(self):
        c = CompilationCounter()
        cloned = c.clone()
        for field in cloned.__dataclass_fields__:
            self.assertEqual(getattr(cloned, field), 0)

    def test_original_unaffected_after_clone_mutated(self):
        c = CompilationCounter()
        c.num_inductor_compiles = 10
        cloned = c.clone()
        cloned.num_inductor_compiles = 0
        self.assertEqual(c.num_inductor_compiles, 10)


class TestCompilationCounterExpect(CustomTestCase):
    def test_expect_passes_when_diff_matches_exactly(self):
        c = CompilationCounter()
        with c.expect(num_graphs_seen=1):
            c.num_graphs_seen += 1

    def test_expect_passes_for_zero_diff(self):
        c = CompilationCounter()
        with c.expect(num_models_seen=0):
            pass  # no mutation

    def test_expect_passes_for_multiple_fields(self):
        c = CompilationCounter()
        with c.expect(num_graphs_seen=1, num_models_seen=2):
            c.num_graphs_seen += 1
            c.num_models_seen += 2

    def test_expect_raises_assertion_error_on_wrong_diff(self):
        c = CompilationCounter()
        with self.assertRaises(AssertionError) as ctx:
            with c.expect(num_graphs_seen=2):
                c.num_graphs_seen += 1  # diff is 1, not 2
        msg = str(ctx.exception)
        self.assertIn("num_graphs_seen", msg)
        self.assertIn("expected diff is 2", msg)

    def test_expect_error_message_contains_before_after_values(self):
        c = CompilationCounter()
        c.num_backend_compilations = 4
        with self.assertRaises(AssertionError) as ctx:
            with c.expect(num_backend_compilations=3):
                c.num_backend_compilations += 10
        msg = str(ctx.exception)
        self.assertIn("before it is 4", msg)
        self.assertIn("after it is 14", msg)

    def test_expect_raises_when_one_field_has_wrong_diff(self):
        c = CompilationCounter()
        with self.assertRaises(AssertionError):
            with c.expect(num_graphs_seen=1, num_models_seen=0):
                c.num_graphs_seen += 5  # diff is 5, not 1


class TestCompilationCounterSingleton(CustomTestCase):
    def test_module_level_singleton_is_consistent_across_imports(self):
        from sglang.srt.compilation.compilation_counter import (
            compilation_counter as cc2,
        )

        self.assertIs(compilation_counter, cc2)

    def test_singleton_is_a_compilation_counter(self):
        self.assertIsInstance(compilation_counter, CompilationCounter)


if __name__ == "__main__":
    unittest.main()
