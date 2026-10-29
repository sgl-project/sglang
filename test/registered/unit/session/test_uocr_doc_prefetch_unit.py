"""CPU tests for the split-document next-window prefetch (park, release, budget)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.session import uocr_doc_prefetch as p
from sglang.srt.session.uocr_doc_reuse import UocrReqState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid, *, after=None, budget=None, keep=False, max_new=1000):
    """A minimal request stand-in for one document window."""
    cp = {"uocr_doc_keep": keep}
    if after is not None:
        cp[p.KEY_AFTER] = after
        cp[p.KEY_BUDGET] = budget
    return SimpleNamespace(
        rid=rid,
        sampling_params=SimpleNamespace(custom_params=cp, max_new_tokens=max_new),
        multimodal_inputs=None,
        output_ids=[],
        finished_len=None,
        finished=lambda: False,
        uocr_doc=None,
    )


def _prefetch():
    return p.DocPrefetch(
        early=None, device=torch.device("cpu"), forward_stream=None, tp_size=1
    )


class _Sched:
    """Scheduler stand-in: a waiting queue that parks follow-ups first."""

    def __init__(self, prefetch):
        self.prefetch = prefetch
        self.queue = []

    def add(self, req):
        if not self.prefetch.park(req):
            self.queue.append(req)

    def ingest(self):
        """The scheduler queues released windows at the top of its next step."""
        for req in self.prefetch.take_released():
            self.add(req)


def _finish(req, n_out, forced=0):
    """Mark a window finished with ``n_out`` outputs, ``forced`` of them spliced."""
    req.output_ids = list(range(n_out))
    req.uocr_doc = UocrReqState(forced_len=forced)
    req.finished = lambda: True


class TestNextWindowPrefetch(CustomTestCase):
    """A prefetched window waits for its predecessor and inherits its budget."""

    def test_a_follow_up_waits_for_its_window_and_takes_the_rest_of_the_budget(self):
        """Sent while window 0 decodes, window 1 must not start before window 0
        has retained its KV, and its budget is what window 0 left over. The
        release happens while window 0's KV is freed, so the window is only
        recorded there and the scheduler queues it on its next step."""
        prefetch = _prefetch()
        sched = _Sched(prefetch)
        w0 = _req("w0", keep=True)
        w1 = _req("w1", after="w0", budget=1000)
        sched.add(w1)
        self.assertEqual(sched.queue, [])
        _finish(w0, 300, forced=0)
        prefetch.on_window_finished(w0)
        self.assertEqual(sched.queue, [])
        sched.ingest()
        self.assertEqual(sched.queue, [w1])
        self.assertEqual(w1.sampling_params.max_new_tokens, 700)
        self.assertEqual(prefetch.take_released(), [])

    def test_a_follow_up_arriving_late_is_not_parked(self):
        """If the window already finished, the follow-up must go straight in,
        or it would wait forever for a release that already happened."""
        prefetch = _prefetch()
        sched = _Sched(prefetch)
        w0 = _req("w0", keep=True)
        _finish(w0, 150, forced=129)
        prefetch.on_window_finished(w0)
        w1 = _req("w1", after="w0", budget=500)
        sched.add(w1)
        self.assertEqual(sched.queue, [w1])
        # Spliced text is not generation; it does not spend the budget.
        self.assertEqual(w1.sampling_params.max_new_tokens, 479)

    def test_a_retracted_window_releases_nothing(self):
        """Retraction passes the same cache hook unfinished; releasing there
        would start the follow-up before the window's KV is retained."""
        prefetch = _prefetch()
        sched = _Sched(prefetch)
        w0 = _req("w0", keep=True)
        w1 = _req("w1", after="w0", budget=1000)
        sched.add(w1)
        prefetch.on_window_finished(w0)
        sched.ingest()
        self.assertEqual(sched.queue, [])

    def test_aborting_the_document_drops_its_held_windows(self):
        """An abort must reach a window still parked off the queue, and one
        released but not yet queued by the scheduler."""
        prefetch = _prefetch()
        sched = _Sched(prefetch)
        w1 = _req("doc-w1-ab", after="doc-w0-cd", budget=1000)
        sched.add(w1)
        self.assertEqual(prefetch.drop("doc-w1"), [w1])
        self.assertEqual(prefetch.held_count(), 0)

        w0 = _req("doc-w0-ef", keep=True)
        w2 = _req("doc-w2-gh", after="doc-w0-ef", budget=1000)
        sched.add(w2)
        _finish(w0, 10)
        prefetch.on_window_finished(w0)
        self.assertEqual(prefetch.drop("doc-w2"), [w2])
        sched.ingest()
        self.assertEqual(sched.queue, [])

    def test_reset_hands_back_every_held_window(self):
        """A cache flush aborts held windows; any it forgot would never be
        answered."""
        prefetch = _prefetch()
        sched = _Sched(prefetch)
        w0 = _req("w0", keep=True)
        parked = _req("w2", after="w1", budget=10)
        released = _req("w1b", after="w0", budget=10)
        sched.add(parked)
        sched.add(released)
        _finish(w0, 1)
        prefetch.on_window_finished(w0)
        self.assertCountEqual(prefetch.reset(), [parked, released])
        self.assertEqual(prefetch.held_count(), 0)
        self.assertEqual(prefetch.take_released(), [])


class TestEarlyEmbeddings(CustomTestCase):
    def test_early_embedding_must_match_the_item_length(self):
        """A stale embedding of another length must not replace the vision pass."""
        early = p.EarlyEmbeddings(encode=lambda items: None)
        early._by_hash[42] = torch.zeros(5, 4)
        self.assertTrue(early.has_prefetched())
        self.assertIsNotNone(early.prefetched(42, 5))
        self.assertIsNone(early.prefetched(42, 6))
        self.assertIsNone(early.prefetched(None, 5))


if __name__ == "__main__":
    unittest.main()
