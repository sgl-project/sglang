"""Unit tests for ReusableEventRing (uniform per-step Event reuse).

Follow-up to #31468: the overlap scheduler and the cuda-graph runners recorded
a freshly constructed cuda Event every decode step (WAR ``read_done`` twice
per DFlash step, scheduler ``copy_done`` once per iteration). The ring reuses
a fixed set of events instead; these tests pin the ring semantics and the
CUDA record/wait-across-reuse behavior.
"""

import unittest

import torch

from sglang.srt.utils.cuda_event_ring import ReusableEventRing
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()


class _CountingFactory:
    def __init__(self):
        self.created = 0

    def __call__(self):
        self.created += 1
        return object()


class TestReusableEventRing(unittest.TestCase):
    def test_lazy_fixed_allocation_and_round_robin(self):
        factory = _CountingFactory()
        ring = ReusableEventRing(factory, depth=3)
        self.assertEqual(factory.created, 0)

        seen = [ring.next() for _ in range(10)]
        self.assertEqual(factory.created, 3)
        self.assertEqual(len(set(map(id, seen[:3]))), 3)
        for i, ev in enumerate(seen):
            self.assertIs(ev, seen[i % 3])

    def test_consecutive_next_distinct_within_depth(self):
        # The in-flight window must never receive the same object twice:
        # with depth d, any d consecutive next() results are distinct.
        ring = ReusableEventRing(_CountingFactory(), depth=2)
        window = [ring.next(), ring.next()]
        self.assertIsNot(window[0], window[1])

    def test_depth_validation(self):
        with self.assertRaises(ValueError):
            ReusableEventRing(_CountingFactory(), depth=0)

    @unittest.skipUnless(_HAS_CUDA, "requires CUDA")
    def test_cuda_record_wait_across_reuse(self):
        # Re-record + wait cycles across ring wrap-around: mimics the WAR
        # read_done (record -> wait_event -> re-record next step) and the
        # copy_done (record -> synchronize) lifecycles.
        ring = ReusableEventRing(torch.cuda.Event, depth=2)
        stream = torch.cuda.Stream()
        x = torch.zeros(1 << 20, device="cuda")
        for step in range(8):
            with torch.cuda.stream(stream):
                x.add_(1.0)
            ev = ring.next()
            ev.record(stream)
            torch.cuda.current_stream().wait_event(ev)
            ev2 = ring.next()
            ev2.record()
            ev2.synchronize()
            self.assertEqual(float(x[0].item()), float(step + 1))


if __name__ == "__main__":
    unittest.main()
