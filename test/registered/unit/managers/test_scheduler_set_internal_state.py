"""Scheduler.set_internal_state runtime-mutable allowlist tests.

Focused on the `max_running_requests` runtime override path: it is the only
allowlisted knob that mirrors onto a scheduler instance attribute (the schedule
policy reads `self.max_running_requests` fresh each iteration, so the mirror is
what actually gates the next admission decision).

Fragility: bypasses `Scheduler.__init__` via `__new__` and injects only the
attrs `set_internal_state` reads (`_max_running_requests_hard_cap`,
`max_running_requests`, `spec_algorithm`, `metrics_reporter`). Update the
fixture if `set_internal_state` starts reading another attr for the non-DSpark
path.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_scheduler(hard_cap: int, current: int) -> Scheduler:
    sch = Scheduler.__new__(Scheduler)
    sch._max_running_requests_hard_cap = hard_cap
    sch.max_running_requests = current
    sch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    sch.metrics_reporter = SimpleNamespace(
        spec_total_num_forward_ct=0,
        spec_total_num_accept_tokens=0,
    )
    return sch


class TestSetInternalStateMaxRunningRequests(CustomTestCase):
    def test_accepted_update_mirrors_onto_instance_attribute(self):
        # Guards: without the mirror, `schedule_policy` keeps reading the init
        # cap and the runtime override has no admission-side effect.
        sch = _make_scheduler(hard_cap=128, current=128)
        req = SetInternalStateReq(server_args={"max_running_requests": 32})
        with patch(
            "sglang.srt.managers.scheduler.get_server_args"
        ) as mock_get_server_args:
            mock_get_server_args.return_value.override = lambda source, **kw: None
            resp = sch.set_internal_state(req)
        self.assertTrue(resp.updated)
        self.assertEqual(sch.max_running_requests, 32)

    def test_value_above_hard_cap_is_rejected_and_leaves_state_unchanged(self):
        # Guards the upper clamp: raising above the init cap would violate
        # `buffer_size = max_running_requests * 2` (sized at init) and the
        # pp_max_micro_batch_size default.
        sch = _make_scheduler(hard_cap=128, current=64)
        req = SetInternalStateReq(server_args={"max_running_requests": 256})
        resp = sch.set_internal_state(req)
        self.assertFalse(resp.updated)
        self.assertEqual(sch.max_running_requests, 64)

    def test_value_below_one_is_rejected(self):
        # Guards the lower clamp: 0 would stall admission entirely and any
        # negative value is nonsensical.
        sch = _make_scheduler(hard_cap=128, current=64)
        for bad in (0, -1):
            with self.subTest(bad=bad):
                req = SetInternalStateReq(server_args={"max_running_requests": bad})
                resp = sch.set_internal_state(req)
                self.assertFalse(resp.updated)
                self.assertEqual(sch.max_running_requests, 64)

    def test_max_running_requests_is_in_allowlist(self):
        # Bookkeeping guard: an "unknown key" rejection would silently regress
        # the runtime tunable. This case fails red if someone removes the
        # allowlist entry without also removing the branch.
        sch = _make_scheduler(hard_cap=128, current=128)
        req = SetInternalStateReq(server_args={"max_running_requests": 64})
        with patch(
            "sglang.srt.managers.scheduler.get_server_args"
        ) as mock_get_server_args:
            mock_get_server_args.return_value.override = lambda source, **kw: None
            resp = sch.set_internal_state(req)
        self.assertTrue(resp.updated)


if __name__ == "__main__":
    unittest.main()
