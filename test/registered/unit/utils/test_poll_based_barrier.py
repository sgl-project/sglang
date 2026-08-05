"""Unit tests for PollBasedBarrier without a live distributed process group."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.utils.poll_based_barrier as barrier_module
from sglang.srt.utils.poll_based_barrier import PollBasedBarrier
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TwoRankAllReduce:
    """Deterministic two-rank MIN collective used by the state-machine tests."""

    def __init__(self, peer_arrivals):
        self._peer_arrivals = iter(peer_arrivals)
        self.calls = []

    def __call__(self, tensor, op, group):
        local_arrived = bool(tensor.item())
        peer_arrived = next(self._peer_arrivals)
        self.calls.append((local_arrived, op, group))
        tensor.fill_(local_arrived and peer_arrived)


class TestPollBasedBarrier(CustomTestCase):
    def setUp(self):
        self.cpu_group = object()
        world_group = SimpleNamespace(cpu_group=self.cpu_group)
        self._world_group_patch = patch.object(
            barrier_module, "get_world_group", return_value=world_group
        )
        self._world_group_patch.start()
        self.addCleanup(self._world_group_patch.stop)

    def test_waits_for_peer_then_resets_for_next_round(self):
        collective = _TwoRankAllReduce([False, True, True])
        barrier = PollBasedBarrier()

        barrier.local_arrive()
        with patch.object(torch.distributed, "all_reduce", collective):
            self.assertFalse(barrier.poll_global_arrived())

            # A rank cannot enter the next round while this round is pending.
            with self.assertRaises(AssertionError):
                barrier.local_arrive()

            self.assertTrue(barrier.poll_global_arrived())

            # Completion resets local state, so the barrier is reusable.
            barrier.local_arrive()
            self.assertTrue(barrier.poll_global_arrived())

        self.assertEqual([call[0] for call in collective.calls], [True, True, True])

    def test_local_arrival_is_required_for_completion(self):
        collective = _TwoRankAllReduce([True, True])
        barrier = PollBasedBarrier()

        with patch.object(torch.distributed, "all_reduce", collective):
            self.assertFalse(barrier.poll_global_arrived())
            barrier.local_arrive()
            self.assertTrue(barrier.poll_global_arrived())

        self.assertEqual([call[0] for call in collective.calls], [False, True])

    def test_uses_min_reduction_on_cpu_world_group(self):
        collective = _TwoRankAllReduce([True])
        barrier = PollBasedBarrier()
        barrier.local_arrive()

        with patch.object(torch.distributed, "all_reduce", collective):
            self.assertTrue(barrier.poll_global_arrived())

        _, op, group = collective.calls[0]
        self.assertIs(op, torch.distributed.ReduceOp.MIN)
        self.assertIs(group, self.cpu_group)

    def test_noop_participates_without_signaling_completion(self):
        collective = _TwoRankAllReduce([True])
        barrier = PollBasedBarrier(noop=True)

        with patch.object(torch.distributed, "all_reduce", collective):
            self.assertFalse(barrier.poll_global_arrived())

        # Noop ranks contribute an arrived value to avoid blocking active ranks,
        # but do not report a local completion to their caller.
        self.assertTrue(collective.calls[0][0])


if __name__ == "__main__":
    unittest.main()
