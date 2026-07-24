"""Unit tests for compilation/compilation_counter.py — no server, no model loading.

Covers ``CompilationCounter``: the dataclass that tracks torch.compile /
CUDA-graph capture counts, its ``clone()`` deep copy, and the ``expect()``
context manager that asserts exact per-field deltas across a code block.
"""

import dataclasses
import unittest

from sglang.srt.compilation.compilation_counter import (
    CompilationCounter,
    compilation_counter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDefaultsAndSingleton(CustomTestCase):
    def test_all_counters_start_at_zero(self):
        counter = CompilationCounter()
        for field in dataclasses.fields(counter):
            self.assertEqual(getattr(counter, field.name), 0, field.name)

    def test_module_singleton_is_a_counter(self):
        self.assertIsInstance(compilation_counter, CompilationCounter)


class TestClone(CustomTestCase):
    def test_clone_copies_values(self):
        counter = CompilationCounter()
        counter.num_graphs_seen = 3
        self.assertEqual(counter.clone().num_graphs_seen, 3)

    def test_clone_returns_new_instance(self):
        counter = CompilationCounter()
        self.assertIsNot(counter.clone(), counter)

    def test_clone_is_independent(self):
        counter = CompilationCounter()
        counter.num_graphs_seen = 3
        clone = counter.clone()
        clone.num_graphs_seen = 99
        # Mutating the clone must not touch the original.
        self.assertEqual(counter.num_graphs_seen, 3)


class TestExpect(CustomTestCase):
    def test_passes_on_exact_delta(self):
        counter = CompilationCounter()
        with counter.expect(num_graphs_seen=2):
            counter.num_graphs_seen += 1
            counter.num_graphs_seen += 1  # total in-block delta == 2

    def test_raises_on_too_small_delta(self):
        counter = CompilationCounter()
        with self.assertRaises(AssertionError):
            with counter.expect(num_graphs_seen=2):
                counter.num_graphs_seen += 1  # only +1, expected +2

    def test_raises_when_expecting_no_change_but_changed(self):
        counter = CompilationCounter()
        with self.assertRaises(AssertionError):
            with counter.expect(num_graphs_seen=0):
                counter.num_graphs_seen += 1

    def test_validates_multiple_fields(self):
        counter = CompilationCounter()
        with counter.expect(num_graphs_seen=1, num_backend_compilations=2):
            counter.num_graphs_seen += 1
            counter.num_backend_compilations += 2

    def test_no_kwargs_is_noop(self):
        counter = CompilationCounter()
        with counter.expect():
            counter.num_graphs_seen += 5  # unchecked, must not raise

    def test_measures_in_block_delta_not_absolute(self):
        counter = CompilationCounter()
        counter.num_graphs_seen += 10  # happens before the block, must not count
        with counter.expect(num_graphs_seen=1):
            counter.num_graphs_seen += 1

    def test_assertion_message_names_the_field(self):
        counter = CompilationCounter()
        with self.assertRaises(AssertionError) as ctx:
            with counter.expect(num_graphs_seen=5):
                counter.num_graphs_seen += 1
        self.assertIn("num_graphs_seen", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
