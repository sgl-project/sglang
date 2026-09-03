"""run_with_deadline bounds a blocking call: a wedged transfer-engine init must
fail startup with a named error instead of hanging until the scheduler watchdog."""

import threading
import time
import unittest

from sglang.srt.utils.common import run_with_deadline
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRunWithDeadline(CustomTestCase):
    def test_returns_the_call_result(self):
        self.assertEqual(run_with_deadline(lambda: 42, timeout_s=1.0, what="x"), 42)

    def test_propagates_the_call_exception(self):
        def boom():
            raise ValueError("engine says no")

        with self.assertRaisesRegex(ValueError, "engine says no"):
            run_with_deadline(boom, timeout_s=1.0, what="x")

    def test_raises_with_the_call_name_when_the_call_hangs(self):
        release = threading.Event()
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "NIXL create_backend.*0.2s"):
            run_with_deadline(release.wait, timeout_s=0.2, what="NIXL create_backend")
        self.assertLess(time.monotonic() - started, 2.0)
        release.set()


if __name__ == "__main__":
    unittest.main()
