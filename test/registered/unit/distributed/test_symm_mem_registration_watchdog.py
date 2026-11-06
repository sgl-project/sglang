import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.distributed.device_communicators import pynccl_allocator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    SymmetricMemoryContext,
    describe_registration_in_progress,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    create_scheduler_watchdog,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FAKE_CTX = SimpleNamespace(
    _comm_ptr=1, group_coordinator=SimpleNamespace(unique_name="tp:0")
)
FAKE_DESCRIPTION = "comm=1 group=tp:0"


class TestSymmMemRegistrationWatchdog(CustomTestCase):
    def test_registration_is_visible_only_while_in_flight(self):
        seen = []

        def register(comm_ptr):
            seen.append(describe_registration_in_progress())
            return 0

        with mock.patch.object(pynccl_allocator, "_register_func", register):
            SymmetricMemoryContext._register_segments_for_comm(FAKE_CTX)
        self.assertEqual(seen, [FAKE_DESCRIPTION])
        self.assertIsNone(describe_registration_in_progress())

        with mock.patch.object(
            pynccl_allocator, "_register_func", side_effect=RuntimeError
        ):
            with self.assertRaises(RuntimeError):
                SymmetricMemoryContext._register_segments_for_comm(FAKE_CTX)
        self.assertIsNone(describe_registration_in_progress())

    def test_dump_names_registration_during_init(self):
        # A stall in NCCL's window registration has no error of its own; the
        # watchdog dump must name it, also while CUDA graphs are captured.
        scheduler = SimpleNamespace(is_initializing=True, forward_ct=0)
        watchdog = create_scheduler_watchdog(scheduler, watchdog_timeout=3600)
        self.assertEqual(watchdog.dump_info(), "")

        with mock.patch.object(pynccl_allocator, "_registering_context", FAKE_CTX):
            self.assertEqual(
                watchdog.dump_info(),
                f"symmetric-memory registration in progress: {FAKE_DESCRIPTION}",
            )


if __name__ == "__main__":
    unittest.main()
