"""Unit tests for the prefix_match load balance method.

The prefix_match scheduler routes requests by a stable hash of the leading
input tokens, so requests sharing a prefix land on the same DP rank. These
tests assert determinism (same prefix -> same rank), even spread across
distinct prefixes, and the empty-input fallback path -- all without booting
a real server.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    LoadBalanceMethod,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_controller(dp_size: int = 8) -> DataParallelController:
    """Construct a controller with just enough state to exercise the scheduler.

    The real ``__init__`` spawns workers and binds ZMQ sockets, which we do
    not need for a routing-only unit test. We bypass it and stitch in the
    minimum attributes the scheduler reads.
    """
    ctrl = DataParallelController.__new__(DataParallelController)
    ctrl.server_args = SimpleNamespace(dp_size=dp_size)
    ctrl.workers = [MagicMock(name=f"worker-{i}") for i in range(dp_size)]
    ctrl.round_robin_counter = 0
    return ctrl


def _make_req(input_ids, routed_dp_rank=None):
    """Build a minimal req object recognised by the scheduler."""
    return SimpleNamespace(input_ids=input_ids, routed_dp_rank=routed_dp_rank)


class TestPrefixMatchScheduler(CustomTestCase):
    def test_enum_registered(self):
        self.assertIs(
            LoadBalanceMethod.from_str("prefix_match"),
            LoadBalanceMethod.PREFIX_MATCH,
        )

    def test_same_prefix_routes_to_same_rank(self):
        ctrl = _make_controller(dp_size=16)
        prefix = list(range(1, 4001))
        req_a = _make_req(prefix + [9001, 9002])
        req_b = _make_req(prefix + [7777])

        ctrl.prefix_match_scheduler(req_a)
        ctrl.prefix_match_scheduler(req_b)

        sent_a = [i for i, w in enumerate(ctrl.workers) if w.send_pyobj.called]
        called_on_b = [
            i for i, w in enumerate(ctrl.workers) if w.send_pyobj.call_count >= 2
        ]
        self.assertEqual(len(sent_a), 1)
        self.assertEqual(sent_a, called_on_b)

    def test_distinct_prefixes_spread_across_ranks(self):
        dp_size = 16
        ctrl = _make_controller(dp_size=dp_size)
        # 256 distinct prefixes -> we expect coverage of most ranks.
        for seed in range(256):
            req = _make_req([seed] * 4096)
            ctrl.prefix_match_scheduler(req)

        used = sum(1 for w in ctrl.workers if w.send_pyobj.called)
        # With 256 prefixes uniformly hashed into 16 buckets the empty-bucket
        # probability is vanishingly small; require at least 12/16 to allow
        # generous slack while still catching a degenerate hash.
        self.assertGreaterEqual(used, 12)

    def test_empty_input_falls_back_to_round_robin(self):
        dp_size = 4
        ctrl = _make_controller(dp_size=dp_size)
        for _ in range(dp_size):
            ctrl.prefix_match_scheduler(_make_req([]))
        for w in ctrl.workers:
            self.assertEqual(w.send_pyobj.call_count, 1)

    def test_external_routed_dp_rank_takes_precedence(self):
        ctrl = _make_controller(dp_size=8)
        req = _make_req([1, 2, 3, 4], routed_dp_rank=5)
        ctrl.prefix_match_scheduler(req)
        for i, w in enumerate(ctrl.workers):
            if i == 5:
                w.send_pyobj.assert_called_once_with(req)
            else:
                w.send_pyobj.assert_not_called()


if __name__ == "__main__":
    unittest.main()
