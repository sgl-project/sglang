"""run_with_deadline: a hanging transfer-engine init must fail startup with a
named error instead of waiting for the scheduler watchdog, and an exception
raised on the worker thread must reach the caller rather than vanish."""

import threading
import time
import unittest

from sglang.srt.utils.common import run_with_deadline
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRunWithDeadline(CustomTestCase):
    def test_hang_raises_a_named_error_at_the_deadline(self):
        release = threading.Event()
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "NIXL create_backend.*0.2s"):
            run_with_deadline(release.wait, timeout_s=0.2, what="NIXL create_backend")
        self.assertLess(time.monotonic() - started, 2.0)
        release.set()

    def test_worker_exception_reaches_the_caller(self):
        def boom():
            raise ValueError("engine says no")

        with self.assertRaisesRegex(ValueError, "engine says no"):
            run_with_deadline(boom, timeout_s=1.0, what="x")


if __name__ == "__main__":
    unittest.main()
