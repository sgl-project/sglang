"""Unit tests for srt/compilation/compilation_counter.py — no server, no model loading."""

import unittest

from sglang.srt.compilation.compilation_counter import (
    CompilationCounter,
    compilation_counter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestCompilationCounterDefaults(CustomTestCase):
    def setUp(self):
        self.counter = CompilationCounter()

    def test_all_fields_default_to_zero(self):
        self.assertEqual(self.counter.num_models_seen, 0)
        self.assertEqual(self.counter.num_graphs_seen, 0)
        self.assertEqual(self.counter.num_piecewise_graphs_seen, 0)
        self.assertEqual(self.counter.num_piecewise_capturable_graphs_seen, 0)
        self.assertEqual(self.counter.num_backend_compilations, 0)
        self.assertEqual(self.counter.num_gpu_runner_capture_triggers, 0)
        self.assertEqual(self.counter.num_cudagraph_captured, 0)
        self.assertEqual(self.counter.num_inductor_compiles, 0)
        self.assertEqual(self.counter.num_eager_compiles, 0)
        self.assertEqual(self.counter.num_cache_entries_updated, 0)
        self.assertEqual(self.counter.num_compiled_artifacts_saved, 0)
        self.assertEqual(self.counter.dynamo_as_is_count, 0)

    def test_fields_are_mutable(self):
        self.counter.num_models_seen = 5
        self.assertEqual(self.counter.num_models_seen, 5)


class TestCompilationCounterClone(CustomTestCase):
    def setUp(self):
        self.counter = CompilationCounter()

    def test_clone_returns_equal_values(self):
        self.counter.num_models_seen = 3
        self.counter.num_graphs_seen = 7
        cloned = self.counter.clone()
        self.assertEqual(cloned.num_models_seen, 3)
        self.assertEqual(cloned.num_graphs_seen, 7)

    def test_clone_is_independent(self):
        """Mutating the clone must not affect the original."""
        self.counter.num_models_seen = 2
        cloned = self.counter.clone()
        cloned.num_models_seen = 99
        self.assertEqual(self.counter.num_models_seen, 2)

    def test_original_independent_of_clone(self):
        """Mutating the original after clone must not affect the clone."""
        self.counter.num_inductor_compiles = 4
        cloned = self.counter.clone()
        self.counter.num_inductor_compiles = 100
        self.assertEqual(cloned.num_inductor_compiles, 4)

    def test_clone_all_fields(self):
        self.counter.num_models_seen = 1
        self.counter.num_graphs_seen = 2
        self.counter.num_piecewise_graphs_seen = 3
        self.counter.num_piecewise_capturable_graphs_seen = 4
        self.counter.num_backend_compilations = 5
        self.counter.num_gpu_runner_capture_triggers = 6
        self.counter.num_cudagraph_captured = 7
        self.counter.num_inductor_compiles = 8
        self.counter.num_eager_compiles = 9
        self.counter.num_cache_entries_updated = 10
        self.counter.num_compiled_artifacts_saved = 11
        self.counter.dynamo_as_is_count = 12
        cloned = self.counter.clone()
        for field in [
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
        ]:
            self.assertEqual(getattr(cloned, field), getattr(self.counter, field))


class TestCompilationCounterExpect(CustomTestCase):
    def setUp(self):
        self.counter = CompilationCounter()

    def test_expect_single_field_passes(self):
        with self.counter.expect(num_models_seen=1):
            self.counter.num_models_seen += 1

    def test_expect_multiple_fields_passes(self):
        with self.counter.expect(num_models_seen=2, num_graphs_seen=3):
            self.counter.num_models_seen += 2
            self.counter.num_graphs_seen += 3

    def test_expect_zero_increment_passes(self):
        with self.counter.expect(num_models_seen=0):
            pass  # no change

    def test_expect_fails_on_wrong_increment(self):
        with self.assertRaises(AssertionError):
            with self.counter.expect(num_models_seen=1):
                self.counter.num_models_seen += 2  # expected 1, got 2

    def test_expect_fails_when_no_increment(self):
        with self.assertRaises(AssertionError):
            with self.counter.expect(num_models_seen=1):
                pass  # expected +1, but no change

    def test_expect_captures_baseline_before_block(self):
        """Baseline is captured at block entry, not at object creation."""
        self.counter.num_graphs_seen = 10
        with self.counter.expect(num_graphs_seen=1):
            self.counter.num_graphs_seen += 1
        self.assertEqual(self.counter.num_graphs_seen, 11)

    def test_expect_nested_does_not_interfere(self):
        with self.counter.expect(num_models_seen=1):
            self.counter.num_models_seen += 1
            # inner block on a different field
            with self.counter.expect(num_graphs_seen=2):
                self.counter.num_graphs_seen += 2

    def test_expect_error_message_contains_field_name(self):
        try:
            with self.counter.expect(num_eager_compiles=5):
                self.counter.num_eager_compiles += 3
            self.fail("Expected AssertionError")
        except AssertionError as exc:
            self.assertIn("num_eager_compiles", str(exc))

    def test_expect_negative_increment(self):
        """Counter can be decremented; expect should track the diff."""
        self.counter.num_models_seen = 5
        with self.counter.expect(num_models_seen=-2):
            self.counter.num_models_seen -= 2

    def test_expect_does_not_reset_counter(self):
        """State accumulated before the block is preserved after it."""
        self.counter.num_models_seen = 7
        with self.counter.expect(num_models_seen=1):
            self.counter.num_models_seen += 1
        self.assertEqual(self.counter.num_models_seen, 8)


class TestGlobalCompilationCounter(CustomTestCase):
    def test_global_instance_is_compilation_counter(self):
        self.assertIsInstance(compilation_counter, CompilationCounter)

    def test_global_instance_is_singleton(self):
        from sglang.srt.compilation.compilation_counter import (
            compilation_counter as cc2,
        )

        self.assertIs(compilation_counter, cc2)


if __name__ == "__main__":
    unittest.main()
