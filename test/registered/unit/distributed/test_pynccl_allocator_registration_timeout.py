"""Bounding the symmetric-memory window registration.

NCCL's window registration blocks in a Unix-domain-socket fd-exchange
rendezvous that has no timeout of its own. When that rendezvous wedges, every
rank in the group sits in recvmsg with no error and no log line, `/health`
keeps answering 200, and the engine silently serves nothing
(sgl-project/sglang#36943). ``_run_with_timeout`` is what turns that into a
reported failure, and ``describe_registration_in_progress`` is what lets the
scheduler watchdog name the stall.
"""

import time
import unittest

from sglang.srt.distributed.device_communicators import pynccl_allocator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    _run_with_timeout,
    describe_registration_in_progress,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRunWithTimeout(CustomTestCase):
    def test_returns_result(self):
        self.assertEqual(_run_with_timeout(lambda: 7, timeout_s=5.0), 7)

    def test_propagates_exception(self):
        def boom():
            raise RuntimeError("registration failed")

        with self.assertRaises(RuntimeError):
            _run_with_timeout(boom, timeout_s=5.0)

    def test_times_out_instead_of_hanging(self):
        def never_returns():
            time.sleep(30)

        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            _run_with_timeout(never_returns, timeout_s=0.2)
        # The bound decided when to give up, not the call.
        self.assertLess(time.monotonic() - started, 5.0)


class TestDescribeRegistrationInProgress(CustomTestCase):
    def test_idle_by_default(self):
        self.assertIsNone(describe_registration_in_progress())

    def test_reports_what_is_in_flight(self):
        pynccl_allocator._registration_in_progress = "comm=1 group=tp"
        try:
            self.assertEqual(
                describe_registration_in_progress(), "comm=1 group=tp"
            )
        finally:
            pynccl_allocator._registration_in_progress = None


if __name__ == "__main__":
    unittest.main()
