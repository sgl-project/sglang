"""A transport-marked peer is retired once its heartbeat stops moving, including one
that died before its first beat and so reads as tick 0 forever."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.layers.moe.token_dispatcher.nixl as nixl
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.test.test_utils import CustomTestCase

WIDTH = 4
MARKED = 3


class _MarkingBuffer:
    def query_mask_buffer(self, out):
        out.zero_()
        out[MARKED] = 1


class TestFaultHeartbeat(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.now = 100.0
        self.ticks = {}
        self.active = torch.ones(WIDTH, dtype=torch.int32)
        self.state = SimpleNamespace(
            buffer=_MarkingBuffer(),
            connected_ep_size=WIDTH,
            mask_buffer=torch.zeros(WIDTH, dtype=torch.int32),
            mask_width=(WIDTH, WIDTH),
            mask_settled_at=0.0,
            mask_cleared=True,
            last_fault_poll=0.0,
            heartbeats={},
            held_marks=set(),
        )
        inst = SimpleNamespace(active_ranks=self.active, active_ranks_cpu=self.active)
        for patch in (
            mock.patch.object(
                nixl.NixlEPBuffer, "_state", classmethod(lambda cls: self.state)
            ),
            mock.patch.object(
                ElasticEPStateManager, "instance", classmethod(lambda cls: inst)
            ),
            mock.patch.object(
                ElasticEPStateManager,
                "is_scale_pending",
                classmethod(lambda cls: False),
            ),
            mock.patch.object(
                ElasticEPStateManager,
                "get_data_plane_ep_size",
                classmethod(lambda cls: WIDTH),
            ),
            mock.patch.object(nixl, "is_device_stream_capturing", lambda device: False),
            mock.patch.object(nixl, "beat_heartbeat", lambda: None),
            mock.patch.object(
                nixl,
                "read_heartbeats",
                lambda ranks: {rank: self.ticks.get(rank, 0) for rank in ranks},
            ),
            mock.patch.object(
                nixl, "time", SimpleNamespace(monotonic=lambda: self.now)
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _poll_at(self, now):
        self.now = now
        nixl.NixlEPBuffer.poll_rank_faults()
        return bool(self.active[MARKED])

    def test_a_never_beaten_peer_is_retired_once_its_zero_goes_stale(self):
        stale = nixl._FAULT_HEARTBEAT_STALE_S
        self.assertTrue(self._poll_at(100.0), "first sighting starts the window")
        self.assertTrue(self._poll_at(100.0 + stale / 2))
        self.assertFalse(self._poll_at(100.0 + stale + 1.0))

    def test_a_beating_peer_stays_in_the_mask(self):
        stale = nixl._FAULT_HEARTBEAT_STALE_S
        for step in range(4):
            self.ticks[MARKED] = step + 1
            self.assertTrue(self._poll_at(100.0 + step * stale * 0.6))


if __name__ == "__main__":
    unittest.main()
