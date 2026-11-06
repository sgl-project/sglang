"""CPU tests for the split-document next-window prefetch (park, release, budget)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.unlimited_ocr_v2_support import document_prefetch as p
from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import UocrReqState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid, *, after=None, budget=None, keep=False, max_new=1000):
    """A minimal request stand-in for one document window."""
    cp = {"uocr_doc_keep": keep}
    if after is not None:
        cp[p.KEY_AFTER] = after
        cp[p.KEY_BUDGET] = budget
    return FakeReq(
        rid=rid,
        sampling_params=SimpleNamespace(custom_params=cp, max_new_tokens=max_new),
        multimodal_inputs=None,
        output_ids=[],
        finished_len=None,
        finished=lambda: False,
        uocr_doc=None,
    )


def _prefetch(aborted=None):
    return p.DocPrefetch(
        early=None,
        device=torch.device("cpu"),
        forward_stream=None,
        tp_size=1,
        abort_windows=lambda reqs, finished_reason: aborted.extend(reqs),
    )


def _abort(prefetch, rid):
    for windows in prefetch.request_lists().values():
        windows.abort(AbortReq(rid=rid))


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

    def test_every_follow_up_of_a_window_is_released(self):
        """Two follow-ups waiting on one window are both queued when it finishes."""
        prefetch = _prefetch()
        sched = _Sched(prefetch)
        w0 = _req("w0", keep=True)
        w1a = _req("w1a", after="w0", budget=1000)
        w1b = _req("w1b", after="w0", budget=1000)
        sched.add(w1a)
        sched.add(w1b)
        _finish(w0, 300)
        prefetch.on_window_finished(w0)
        sched.ingest()
        self.assertEqual(sched.queue, [w1a, w1b])

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
        aborted = []
        prefetch = _prefetch(aborted)
        sched = _Sched(prefetch)
        w1 = _req("doc-w1-ab", after="doc-w0-cd", budget=1000)
        sched.add(w1)
        _abort(prefetch, "doc-w1")
        self.assertEqual(aborted, [w1])
        self.assertEqual(prefetch.held_count(), 0)

        w0 = _req("doc-w0-ef", keep=True)
        w2 = _req("doc-w2-gh", after="doc-w0-ef", budget=1000)
        sched.add(w2)
        _finish(w0, 10)
        prefetch.on_window_finished(w0)
        _abort(prefetch, "doc-w2")
        self.assertEqual(aborted, [w1, w2])
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


class TestEarlyStart(CustomTestCase):
    """On MPS a parked window is queued before its predecessor ends, once that
    one decodes and the window's pages are encoded."""

    @staticmethod
    def _decode(prefetch, *reqs):
        prefetch.note_running(
            SimpleNamespace(forward_mode=ForwardMode.DECODE, reqs=reqs)
        )

    def _decoding_pair(self, prefetch):
        sched = _Sched(prefetch)
        w0 = _req("w0", keep=True)
        w0.kv = SimpleNamespace(holds_kv=True)
        w1 = _req("w1", after="w0", budget=1000)
        sched.add(w1)
        return sched, w0, w1

    def test_a_follow_up_starts_once_its_predecessor_decodes(self):
        prefetch = _prefetch()
        _, w0, w1 = self._decoding_pair(prefetch)
        self.assertEqual(prefetch.early_candidates(), [])
        self._decode(prefetch, w0)
        self.assertEqual(prefetch.early_candidates(), [(w1, w0)])
        w0.kv.holds_kv = False  # retracted: nothing to copy from
        self.assertEqual(prefetch.early_candidates(), [])

    def test_a_follow_up_waits_for_its_pages(self):
        early = p.EarlyEmbeddings(encode=lambda items: None)
        prefetch = p.DocPrefetch(
            early=early,
            device=torch.device("cpu"),
            forward_stream=None,
            tp_size=1,
            abort_windows=lambda reqs, finished_reason: None,
        )
        _, w0, w1 = self._decoding_pair(prefetch)
        self._decode(prefetch, w0)
        early._stepped.append(("w1", [], iter(())))
        self.assertEqual(prefetch.early_candidates(), [])
        early._stepped.clear()
        self.assertEqual(prefetch.early_candidates(), [(w1, w0)])

    def test_an_early_follow_up_gets_its_budget_when_the_predecessor_ends(self):
        prefetch = _prefetch()
        sched, w0, w1 = self._decoding_pair(prefetch)
        self._decode(prefetch, w0)
        prefetch.start_early(w1)
        self.assertEqual(prefetch.held_count(), 0)
        _finish(w0, 300)
        self.assertEqual(prefetch.on_window_finished(w0), [w1])
        self.assertEqual(w1.sampling_params.max_new_tokens, 700)
        # Already queued: nothing to release, and no late-arrival record left.
        sched.ingest()
        self.assertEqual(sched.queue, [])
        self.assertEqual(prefetch._finished, {})
        self.assertEqual(prefetch.early_candidates(), [])

    def test_a_retracted_early_follow_up_parks_again(self):
        prefetch = _prefetch()
        _, w0, w1 = self._decoding_pair(prefetch)
        self._decode(prefetch, w0)
        prefetch.start_early(w1)
        self.assertTrue(prefetch.repark_early(w1))
        self.assertEqual(prefetch.held_count(), 1)
        self.assertFalse(prefetch.repark_early(w1))
        _finish(w0, 300)
        self.assertEqual(prefetch.on_window_finished(w0), [])
        self.assertEqual(prefetch.take_released(), [w1])


class TestEarlyEmbeddings(CustomTestCase):
    def test_early_embedding_must_match_the_item_length(self):
        """A stale embedding of another length must not replace the vision pass."""
        early = p.EarlyEmbeddings(encode=lambda items: None)
        early._by_key[b"a" * 32] = torch.zeros(5, 4)
        self.assertTrue(early.has_prefetched())
        self.assertIsNotNone(early.prefetched(b"a" * 32, 5))
        self.assertIsNone(early.prefetched(b"a" * 32, 6))
        self.assertIsNone(early.prefetched(None, 5))

    def test_mps_encodes_parked_pages_in_slices_between_forwards(self):
        """MPS has no side stream: a parked window's encode runs one slice after
        every second forward, and a release drops what is left of it."""
        events = []

        def steps(items):
            events.append("slice 1")
            yield
            events.append("slice 2")
            yield
            return torch.full((sum(_tokens(it) for it in items), 4), 3.0)

        early = p.EarlyEmbeddings(encode=lambda items: None, encode_steps=steps)
        a, b = _item(b"a", 5), _item(b"b", 3)
        with _no_device_move():
            _encode_on_mps(early, "w1", [a, b])
        self.assertTrue(early.has_prefetched())
        self.assertTrue(early.encoding("w1"))
        early.on_forward_step()
        self.assertEqual(events, [])
        early.on_forward_step()
        self.assertEqual(events, ["slice 1"])
        for _ in range(4):
            early.on_forward_step()
        self.assertFalse(early.encoding("w1"))
        self.assertEqual(early.prefetched(a.cache_key, 5).shape, (5, 4))
        self.assertEqual(early.prefetched(b.cache_key, 3).shape, (3, 4))

        events.clear()
        with _no_device_move():
            _encode_on_mps(early, "w2", [_item(b"c", 4)])
        early.release("w2")
        for _ in range(6):
            early.on_forward_step()
        self.assertEqual(events, [])
        self.assertIsNone(early.prefetched(b"c" * 32, 4))
        early.release("w1")
        self.assertEqual(early._bytes, 0)

    def test_a_prefill_miss_finishes_only_the_encode_holding_that_page(self):
        """A prefill that reaches a page still being encoded finishes that
        window's encode; another document's pages or encodes are not waited for."""
        started = []

        def steps(items):
            started.append(items[0].cache_key[:1])
            yield
            yield
            return torch.zeros(sum(_tokens(it) for it in items), 4)

        early = p.EarlyEmbeddings(encode=lambda items: None, encode_steps=steps)
        a, b = _item(b"a", 2), _item(b"b", 2)
        with _no_device_move():
            _encode_on_mps(early, "w1", [a])
            _encode_on_mps(early, "w2", [b])
        self.assertIsNone(early.prefetched(b"x" * 32, 2))
        self.assertEqual(started, [])
        self.assertIsNotNone(early.prefetched(b.cache_key, 2))
        self.assertEqual(started, [b"b"])
        self.assertTrue(early.encoding("w1"))
        self.assertFalse(early.encoding("w2"))

    def test_mps_does_not_encode_a_page_twice(self):
        """A window's lookahead page is the next window's first page. If the
        previous window encoded it early, the next one shares that embedding
        and keeps it past the previous window's release: its prefill still
        reads it. If a prefill embedded it, the embedding cache has it."""
        encoded = []

        def steps(items):
            encoded.extend(it.cache_key[:1] for it in items)
            yield
            return torch.zeros(sum(_tokens(it) for it in items), 4)

        early = p.EarlyEmbeddings(encode=lambda items: None, encode_steps=steps)
        shared, cached, new = _item(b"s", 2), _item(b"c", 2), _item(b"n", 2)
        cache = SimpleNamespace(
            get_single=lambda key: (
                SimpleNamespace(embedding=torch.zeros(1, 2, 4))
                if key == cached.cache_key
                else None
            )
        )
        with (
            _no_device_move(),
            mock.patch.object(p.mm_schedule, "embedding_cache", cache),
        ):
            _encode_on_mps(early, "w1", [shared])
            early.prefetched(shared.cache_key, 2)
            _encode_on_mps(early, "w2", [cached, shared, new])
            early.prefetched(new.cache_key, 2)
        self.assertEqual(encoded, [b"s", b"n"])
        early.release("w1")
        self.assertIsNotNone(early.prefetched(shared.cache_key, 2))
        early.release("w2")
        self.assertIsNone(early.prefetched(shared.cache_key, 2))
        self.assertEqual(early._bytes, 0)


def _tokens(item):
    return sum(end - start + 1 for start, end in item.offsets)


def _item(tag, tokens):
    return SimpleNamespace(
        hash=1,
        cache_key=tag * 32,
        feature=torch.zeros(tokens, 4),
        offsets=[(0, tokens - 1)],
    )


def _no_device_move():
    """Let CPU-only CI run the MPS branch: the features stay where they are."""
    return mock.patch.object(p, "_move_items_to_device", lambda items, dev, fn: None)


def _encode_on_mps(early, rid, items):
    early.encode_ahead(
        rid=rid,
        items=items,
        device=torch.device("mps"),
        forward_stream=None,
        max_bytes=1 << 20,
    )


if __name__ == "__main__":
    unittest.main()
