"""Unit tests for ReusableEventRing and the WAR read-done publishers.

The ring must allocate exactly ``depth`` events however many times it is drawn,
and every publisher must take its event from the runner's ring rather than
construct one per step.
"""

import pathlib
import unittest

import torch

import sglang.srt as srt
from sglang.srt.utils.cuda_event_ring import ReusableEventRing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_HAS_CUDA = torch.cuda.is_available()

_MAILBOX_WRITE = "shared_read_done_event = "
_RING_DRAW = "shared_read_done_events.next()"
_PER_PUBLISH_ALLOC = (
    "read_done = device_module.Event()",
    "read_done = self.device_module.Event()",
)


def _publisher_sources():
    """Sources of every srt module that writes a non-None event into the mailbox.

    Scans rather than imports: the publishers pull in the whole server stack,
    which keeps the check runnable on a CPU runner. Discovering them beats a
    hard-coded list — a publisher added later has to satisfy the ring too.
    """
    for path in sorted(pathlib.Path(srt.__path__[0]).rglob("*.py")):
        blob = path.read_bytes()
        if _MAILBOX_WRITE.encode() not in blob:
            continue
        src = blob.decode()
        if any(
            _MAILBOX_WRITE in line and "None" not in line for line in src.splitlines()
        ):
            yield path, src


class _CountingFactory:
    def __init__(self):
        self.created = 0

    def __call__(self):
        self.created += 1
        return object()


class TestWarMailboxRingWiring(CustomTestCase):
    def test_publishers_use_the_mailbox_ring(self):
        found = list(_publisher_sources())
        # Guards the scan itself: renaming the mailbox field would otherwise
        # leave every assertion below vacuously true.
        self.assertGreaterEqual(len(found), 3, "no shared-read publisher found")
        for path, src in found:
            # assertTrue, not assertIn: the container here is a whole module.
            self.assertTrue(_RING_DRAW in src, f"{path} publishes without the ring")
            for pattern in _PER_PUBLISH_ALLOC:
                self.assertTrue(
                    pattern not in src,
                    f"{path} still allocates a per-publish Event: {pattern}",
                )


class TestReusableEventRing(CustomTestCase):
    def test_lazy_fixed_allocation_and_round_robin(self):
        factory = _CountingFactory()
        ring = ReusableEventRing(factory, depth=3)
        self.assertEqual(factory.created, 0)

        seen = [ring.next() for _ in range(10)]
        self.assertEqual(factory.created, 3)
        self.assertEqual(len(set(map(id, seen[:3]))), 3)
        for i, ev in enumerate(seen):
            self.assertIs(ev, seen[i % 3])

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
