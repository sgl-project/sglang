"""--pp-min-chunked-prefill-size uses the small PP prefill chunk unless a
pending request, waiting or being chunked, has a full chunk of uncached tokens.
Waiting requests are probed about once while they wait: later steps read their
uncached hint."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MOD = "sglang.srt.managers.scheduler"


def _req(input_len, matched, prefix_len=0, hint=None):
    return SimpleNamespace(
        seqlen=input_len,
        num_matched_prefix_tokens=matched,
        prefix_len=prefix_len,
        pp_uncached_hint=hint,
    )


def _scheduler(
    waiting, min_chunk=4096, chunked_req=None, policy_matched=True, overhead=None
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.pp_min_chunked_prefill_size = min_chunk
    scheduler.pp_chunk_overhead_tokens = overhead
    scheduler._pp_pending_tokens = 0
    scheduler._pp_held_chunk = 0
    scheduler._pp_dip_steps = 0
    scheduler.page_size = 64
    scheduler.chunked_req = chunked_req
    scheduler.waiting_queue = waiting
    scheduler.tree_cache = None
    scheduler.policy = SimpleNamespace(
        waiting_queue_prefix_matched=lambda q: policy_matched
    )
    return scheduler


def _chunk(waiting, chunk=16384, pp_size=8, **kwargs):
    scheduler = _scheduler(waiting, **kwargs)
    parallel = SimpleNamespace(pp_size=pp_size)
    with patch(f"{_MOD}.get_parallel", return_value=parallel):
        return scheduler._pp_chunk_size_for_pending(chunk)


class _FakeTree:
    """Stands in for peek_uncached_tokens: answers from a per-request table of
    cached tokens and counts the probes. It never touches the request's match
    fields, as the real probe never changes cache or request state."""

    def __init__(self, cached):
        self.cached = cached
        self.calls = 0

    def peek(self, req, tree_cache):
        self.calls += 1
        cached = self.cached[id(req)]
        if cached is None:
            return None
        return max(1, req.seqlen - cached)


def _chunk_unmatched(waiting, tree, chunk=16384, **kwargs):
    scheduler = _scheduler(waiting, policy_matched=False, **kwargs)
    parallel = SimpleNamespace(pp_size=8)
    with (
        patch(f"{_MOD}.get_parallel", return_value=parallel),
        patch(f"{_MOD}.peek_uncached_tokens", side_effect=tree.peek),
    ):
        return scheduler._pp_chunk_size_for_pending(chunk)


class TestPPChunkByPending(CustomTestCase):
    def test_high_hit_batch_uses_small_chunk(self):
        # bs4 x 128K at 95% hit: 6554 uncached tokens each.
        waiting = [_req(131072, 124518) for _ in range(4)]
        self.assertEqual(_chunk(waiting), 4096)
        self.assertEqual(_chunk(waiting, min_chunk=1024), 1024)
        # Many short requests still use the small chunk, whatever their total.
        waiting = [_req(32768, 26214) for _ in range(64)]
        self.assertEqual(_chunk(waiting), 4096)

    def test_large_prefill_keeps_full_chunk(self):
        self.assertEqual(_chunk([_req(1040000, 0)]), 16384)
        # Cold 32K requests have more uncached tokens than a chunk.
        self.assertEqual(_chunk([_req(32768, 0) for _ in range(3)]), 16384)
        # One cold request among high-hit ones keeps the full chunk too.
        waiting = [_req(32768, 31130), _req(32768, 0), _req(32768, 31130)]
        self.assertEqual(_chunk(waiting), 16384)

    def test_long_chunked_req_keeps_full_chunk(self):
        long_req = _req(262144, 0, prefix_len=262144 - 40000)
        self.assertEqual(_chunk([], chunked_req=long_req), 16384)
        tail = _req(262144, 0, prefix_len=262144 - 10000)
        self.assertEqual(_chunk([], chunked_req=tail), 4096)

    def test_new_requests_are_probed_once(self):
        waiting = [_req(32768, 0) for _ in range(64)]
        tree = _FakeTree({id(r): 31130 for r in waiting})
        self.assertEqual(_chunk_unmatched(waiting, tree), 4096)
        self.assertEqual(tree.calls, 64)
        self.assertEqual(waiting[0].pp_uncached_hint, 1638)
        # The next step reads the hints and probes nothing.
        self.assertEqual(_chunk_unmatched(waiting[8:], tree), 4096)
        self.assertEqual(tree.calls, 64)

    def test_long_hint_is_probed_again(self):
        # Matched cold, then its prefix was cached: the long hint is rechecked.
        cached = _req(32768, 0, hint=32768)
        short = _req(32768, 0, hint=1638)
        tree = _FakeTree({id(cached): 31130})
        self.assertEqual(_chunk_unmatched([cached, short], tree), 4096)
        self.assertEqual((tree.calls, cached.pp_uncached_hint), (1, 1638))
        # A request that is still cold keeps the full chunk after one probe.
        cold = _req(32768, 0, hint=32768)
        tree = _FakeTree({id(cold): 0})
        self.assertEqual(_chunk_unmatched([short, cold, short], tree), 16384)
        self.assertEqual(tree.calls, 1)

    def test_unsupported_probe_keeps_full_chunk(self):
        waiting = [_req(32768, 0) for _ in range(4)]
        tree = _FakeTree({id(r): None for r in waiting})
        self.assertEqual(_chunk_unmatched(waiting, tree), 16384)
        self.assertEqual(tree.calls, 1)
        self.assertIsNone(waiting[0].pp_uncached_hint)

    def test_admission_restores_full_chunk_for_stale_hint(self):
        scheduler = _scheduler([])
        # The hint said 1638 tokens, but the admission match found none cached.
        req = _req(32768, 0, hint=1638)
        adder = SimpleNamespace(rem_chunk_tokens=4096 - 1638)
        chunk = scheduler._pp_restore_full_chunk(req, adder, 4096, 16384)
        self.assertEqual((chunk, adder.rem_chunk_tokens), (16384, 16384 - 1638))
        self.assertEqual(req.pp_uncached_hint, 32768)
        # A short request, or a batch that already has the full chunk, is kept.
        adder = SimpleNamespace(rem_chunk_tokens=4096)
        hit = _req(32768, 31130)
        self.assertEqual(
            scheduler._pp_restore_full_chunk(hit, adder, 4096, 16384), 4096
        )
        self.assertEqual(adder.rem_chunk_tokens, 4096)
        cold = _req(32768, 0)
        self.assertEqual(
            scheduler._pp_restore_full_chunk(cold, adder, 16384, 16384), 16384
        )
        self.assertEqual(adder.rem_chunk_tokens, 4096)


class TestPPChunkByLoad(CustomTestCase):
    """--pp-chunk-overhead-tokens sizes the chunk from the total pending
    uncached tokens U: 1500 tokens of overhead at pp4 moves from 4K to 8K at
    U = 2 * 3 * 4096^2 / 1500 (67109) and from 8K to 16K at 268436."""

    def _load(self, waiting, **kwargs):
        return _chunk(waiting, pp_size=4, overhead=1500, **kwargs)

    def test_chunk_follows_total_load(self):
        self.assertEqual(self._load([_req(8192, 0)]), 4096)
        self.assertEqual(self._load([_req(67108, 0)]), 4096)
        self.assertEqual(self._load([_req(67109, 0)]), 8192)
        self.assertEqual(self._load([_req(131000, 0)]), 8192)
        self.assertEqual(self._load([_req(131000, 0), _req(131000, 0)]), 8192)
        self.assertEqual(self._load([_req(524000, 0)]), 16384)
        # Many mostly cached requests add up: 32 x 13100 uncached tokens.
        waiting = [_req(131000, 117900) for _ in range(32)]
        self.assertEqual(self._load(waiting), 16384)
        self.assertEqual(self._load(waiting[:8]), 8192)
        self.assertEqual(self._load(waiting[:1]), 4096)

    def test_chunked_req_keeps_its_size(self):
        # Admitted with 131000 uncached tokens: 8K to the end of its prefill.
        req = _req(131000, 0, prefix_len=131000 - 20000, hint=131000)
        self.assertEqual(self._load([], chunked_req=req), 8192)
        # Without a hint the rest is all that is known.
        req = _req(1040000, 0, prefix_len=1040000 - 300000)
        self.assertEqual(self._load([], chunked_req=req), 16384)
        req.prefix_len = 1040000 - 20000
        self.assertEqual(self._load([], chunked_req=req), 4096)
        # Waiting requests add to the chunked request's load.
        self.assertEqual(self._load([_req(65536, 0)], chunked_req=req), 8192)

    def test_sum_stops_at_the_full_chunk(self):
        waiting = [_req(524000, 0)] + [_req(32768, 0) for _ in range(8)]
        tree = _FakeTree({id(r): 0 for r in waiting})
        scheduler = _scheduler(waiting, policy_matched=False, overhead=1500)
        with (
            patch(f"{_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=4)),
            patch(f"{_MOD}.peek_uncached_tokens", side_effect=tree.peek),
        ):
            self.assertEqual(scheduler._pp_chunk_size_for_pending(16384), 16384)
        self.assertEqual(tree.calls, 1)

    def test_dip_keeps_the_larger_chunk(self):
        # Two 131K requests at a time: between one prefill's end and the next
        # request only one is pending, which alone would take 8K chunks
        # (2000 tokens of overhead: 8K up to 201327 pending, 16K above).
        scheduler = _scheduler([], overhead=2000)
        two = [_req(131000, 0), _req(131000, 0)]
        one = two[:1]

        def size(waiting):
            scheduler.waiting_queue = waiting
            with patch(f"{_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=4)):
                return scheduler._pp_chunk_size_for_pending(16384)

        self.assertEqual(size(two), 16384)
        self.assertEqual([size(one) for _ in range(16)], [16384] * 16)
        self.assertEqual(size(two), 16384)
        # A lasting drop shrinks the chunk after 4 steps per stage.
        self.assertEqual([size(one) for _ in range(17)], [16384] * 16 + [8192])
        self.assertEqual(size(one), 8192)

    def test_admission_adds_undercounted_tokens(self):
        scheduler = _scheduler([], overhead=1500)
        scheduler._pp_pending_tokens = 13100
        # The hint said 13100 uncached tokens; the admission match found 131000.
        req = _req(131000, 0, hint=13100)
        adder = SimpleNamespace(rem_chunk_tokens=4096)
        with patch(f"{_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=4)):
            chunk = scheduler._pp_restore_full_chunk(req, adder, 4096, 16384)
        self.assertEqual((chunk, adder.rem_chunk_tokens), (8192, 8192))
        self.assertEqual(scheduler._pp_pending_tokens, 131000)


if __name__ == "__main__":
    unittest.main()
