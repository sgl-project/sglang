import unittest
from types import SimpleNamespace

from sglang.srt.distributed.device_communicators import pynccl_allocator
from sglang.srt.managers.scheduler_components.invariant_checker import (
    create_scheduler_watchdog,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSymmMemRegistrationWatchdog(CustomTestCase):
    def test_dump_names_registration_during_init(self):
        # A stall in NCCL's window registration has no error of its own; the
        # watchdog dump must name it, also while CUDA graphs are captured.
        scheduler = SimpleNamespace(is_initializing=True, forward_ct=0)
        watchdog = create_scheduler_watchdog(scheduler, watchdog_timeout=3600)
        self.assertEqual(watchdog.dump_info(), "")

        pynccl_allocator._registration_in_progress = SimpleNamespace(
            _comm_ptr=1, group_coordinator=SimpleNamespace(unique_name="tp:0")
        )
        try:
            self.assertEqual(
                watchdog.dump_info(),
                "symmetric-memory registration in progress: comm=1 group=tp:0",
            )
        finally:
            pynccl_allocator._registration_in_progress = None


if __name__ == "__main__":
    unittest.main()
