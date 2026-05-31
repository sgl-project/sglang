"""Unit tests for scripted_runtime/scheduler_hook._advance_generator — no engine.

``_advance_generator`` is the single step that drives a script generator one yield
forward and decides whether the run is done and whether it failed. Every GPU
integration test's pass/fail verdict flows through this triple
``(running / finished-clean / finished-with-traceback)``, so the three branches
are pinned here on CPU with plain fake generators: a yield means "keep going",
StopIteration means "done, no error", and any other exception means "done, and
here is the captured traceback".
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime import scheduler_hook
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _yielding_gen():
    """A generator with a pending yield: advancing it does not finish the run."""
    yield
    yield


def _empty_gen():
    """A generator that raises StopIteration on the first advance (clean finish)."""
    return
    yield  # pragma: no cover — makes this a generator function


def _raising_gen():
    """A generator that raises on the first advance (finish with traceback)."""
    raise ValueError("scripted-boom")
    yield  # pragma: no cover — makes this a generator function


class TestAdvanceGenerator(CustomTestCase):
    """_advance_generator returns (done, traceback) for the three step outcomes."""

    def test_not_done_when_generator_yields(self):
        """A pending yield reports not-done with no traceback."""
        done, exc_tb = scheduler_hook._advance_generator(_yielding_gen())

        self.assertEqual((done, exc_tb), (False, None))

    def test_done_without_traceback_on_stop_iteration(self):
        """An exhausted generator reports done with no traceback."""
        done, exc_tb = scheduler_hook._advance_generator(_empty_gen())

        self.assertEqual((done, exc_tb), (True, None))

    def test_done_with_traceback_on_exception(self):
        """A raising generator reports done and captures the traceback (logged once)."""
        original = scheduler_hook.logger.exception
        scheduler_hook.logger.exception = MagicMock()
        try:
            done, exc_tb = scheduler_hook._advance_generator(_raising_gen())
            scheduler_hook.logger.exception.assert_called_once()
        finally:
            scheduler_hook.logger.exception = original

        self.assertTrue(done)
        self.assertIsNotNone(exc_tb)
        self.assertIn("ValueError", exc_tb)
        self.assertIn("scripted-boom", exc_tb)


if __name__ == "__main__":
    unittest.main()
