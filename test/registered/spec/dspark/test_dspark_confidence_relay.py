"""ConfidenceRelay: every TP rank resolves the same relayed confidence.

The relayed confidence sets the verify budget, and the budget picks the verify
graph tier, so ranks that resolve different values run different graph shapes
and mismatched collectives.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_DEPTH,
    CONFIDENCE_RELAY_RING_LAG,
    ConfidenceRelay,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

GAMMA = 4
POOL = 4


class _PendingEvent:
    """A copy event that has not completed yet when first asked."""

    def __init__(self):
        self.synchronized = False

    def query(self):
        return self.synchronized

    def synchronize(self):
        self.synchronized = True


class TestConfidenceRelayResolve(CustomTestCase):
    def test_unfinished_copy_is_waited_for(self):
        # On one rank the copy has finished, on another it has not; both must
        # resolve the same confidence, never None on just one of them.
        relay = ConfidenceRelay(
            device=torch.device("cpu"),
            req_pool_size=POOL,
            pool=SimpleNamespace(req_generation=torch.zeros(POOL, dtype=torch.int64)),
        )
        relay.initialized = True
        relay.conf_ring = torch.rand(CONFIDENCE_RELAY_RING_DEPTH, POOL, GAMMA)
        relay.gen_ring = torch.arange(CONFIDENCE_RELAY_RING_DEPTH * POOL).view(
            CONFIDENCE_RELAY_RING_DEPTH, POOL
        )
        relay.copy_done = [_PendingEvent() for _ in range(CONFIDENCE_RELAY_RING_DEPTH)]
        relay.ring_pos = CONFIDENCE_RELAY_RING_LAG + 1
        slot = 1
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=torch.tensor([2, 0])),
            req_pool_indices_cpu=torch.tensor([2, 0]),
        )

        resolved = relay.resolve(batch, stream=object(), publish_ready=object())

        self.assertIsNotNone(resolved)
        self.assertTrue(relay.copy_done[slot].synchronized)
        torch.testing.assert_close(resolved.confidence, relay.conf_ring[slot][[2, 0]])
        torch.testing.assert_close(resolved.generation, relay.gen_ring[slot][[2, 0]])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestConfidenceRelayRingCopy(CustomTestCase):
    def test_next_scatter_waits_for_the_pending_copy(self):
        device = torch.device("cuda")
        relay = ConfidenceRelay(
            device=device,
            req_pool_size=POOL,
            pool=SimpleNamespace(req_generation=torch.zeros(POOL, dtype=torch.int64)),
        )
        indices = torch.arange(POOL, device=device)
        first = torch.full((POOL, GAMMA), 0.25, device=device)
        second = torch.full((POOL, GAMMA), 0.75, device=device)
        copy_stream = torch.cuda.Stream()

        publish_ready = torch.cuda.Event()

        def publish(confidence):
            relay.scatter(indices, confidence)
            publish_ready.record()
            relay.issue_ring_copy(stream=copy_stream, publish_ready=publish_ready)

        publish(first)  # warm up: allocate the ring, issue one copy
        torch.cuda.synchronize()

        relay.scatter(indices, first)
        publish_ready.record()
        # Hold the copy back so the next step's scatter is issued before it runs.
        with torch.cuda.stream(copy_stream):
            torch.cuda._sleep(1_000_000_000)
        relay.issue_ring_copy(stream=copy_stream, publish_ready=publish_ready)
        relay.scatter(indices, second)
        self.assertFalse(
            relay.copy_done[1].query(), "precondition: the copy is still pending"
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(relay.conf_ring[1], first.cpu())
        torch.testing.assert_close(
            relay.confidence_buf, second, msg="the scatter itself must still land"
        )


if __name__ == "__main__":
    unittest.main()
