from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime import scheduler_hook
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _yielding_gen():
    yield
    yield


def _empty_gen():
    return
    yield  # pragma: no cover — makes this a generator function


def _raising_gen():
    raise ValueError("scripted-boom")
    yield  # pragma: no cover — makes this a generator function


class TestAdvanceGenerator(CustomTestCase):

    def test_not_done_when_generator_yields(self):
        done, exc_tb = scheduler_hook._advance_generator(_yielding_gen())

        self.assertEqual((done, exc_tb), (False, None))

    def test_done_without_traceback_on_stop_iteration(self):
        done, exc_tb = scheduler_hook._advance_generator(_empty_gen())

        self.assertEqual((done, exc_tb), (True, None))

    def test_done_with_traceback_on_exception(self):
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
