"""The idle sleeper must not hold back a request waiting for EPD embeddings."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from parameterized import parameterized

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestIdleSleepEpdWait(CustomTestCase):
    def test_no_sleep_while_embedding_pending(self):
        """With --sleep-on-idle and --language-only, the encoder's embeddings arrive
        on a socket the idle sleeper does not poll, so sleeping delayed every image
        request by the full poll timeout (1 s)."""
        sleeper = Mock()
        scheduler = SimpleNamespace(
            idle_sleeper=sleeper,
            mm_receiver=SimpleNamespace(waiting_list=[object()]),
        )
        Scheduler.maybe_sleep_on_idle(scheduler)
        sleeper.maybe_sleep.assert_not_called()

    @parameterized.expand(
        [
            ("no_epd", None),
            ("epd_nothing_pending", SimpleNamespace(waiting_list=[])),
        ]
    )
    def test_sleeps_when_idle(self, _name, mm_receiver):
        sleeper = Mock()
        scheduler = SimpleNamespace(idle_sleeper=sleeper, mm_receiver=mm_receiver)
        Scheduler.maybe_sleep_on_idle(scheduler)
        sleeper.maybe_sleep.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
