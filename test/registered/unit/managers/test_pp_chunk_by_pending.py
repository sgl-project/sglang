import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler_components.pp_chunk_sizer import PPChunkSizer

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_MOD = "sglang.srt.managers.scheduler_components.pp_chunk_sizer"


def _req(input_len, matched, prefix_len=0, hint=None):
    return SimpleNamespace(
        seqlen=input_len,
        num_matched_prefix_tokens=matched,
        prefix_len=prefix_len,
        pp_uncached_hint=hint,
        pp_uncached_hint_version=None,
    )


def _probing_cache(cached=0, version=None):
    return SimpleNamespace(
        peek_prefix_len=lambda key: cached, prefix_cache_version=version
    )


def _sizer(
    min_chunk=4096, pp_size=8, policy_matched=True, overhead=None, tree_cache=None
):
    return PPChunkSizer(
        min_chunk=min_chunk,
        overhead_tokens=overhead,
        pp_size=pp_size,
        tree_cache=tree_cache or _probing_cache(),
        policy=SimpleNamespace(waiting_queue_prefix_matched=lambda q: policy_matched),
    )


def _adder(rem_chunk_tokens):
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.rem_chunk_tokens = rem_chunk_tokens
    return adder


def _chunk(waiting, chunk=16384, chunked_req=None, **kwargs):
    return _sizer(**kwargs).chunk_size(chunk, chunked_req, waiting)


class _FakeTree:
    def __init__(self, cached):
        self.cached = cached
        self.calls = 0

    def peek(self, req, tree_cache):
        self.calls += 1
        return max(1, req.seqlen - self.cached[id(req)])


class TestPPChunkByPending(CustomTestCase):
    def test_high_hit_batch_uses_small_chunk(self):
        # 6554 uncached tokens each: bs4 x 128K at 95% hit.
        self.assertEqual(_chunk([_req(131072, 124518) for _ in range(4)]), 4096)
        # The binary rule does not sum: many short requests keep the small chunk.
        self.assertEqual(_chunk([_req(32768, 26214) for _ in range(64)]), 4096)

    def test_long_pending_request_keeps_full_chunk(self):
        waiting = [_req(32768, 31130), _req(32768, 0), _req(32768, 31130)]
        self.assertEqual(_chunk(waiting), 16384)
        long_req = _req(262144, 0, prefix_len=262144 - 40000)
        self.assertEqual(_chunk([], chunked_req=long_req), 16384)
        tail = _req(262144, 0, prefix_len=262144 - 10000)
        self.assertEqual(_chunk([], chunked_req=tail), 4096)

    def test_small_chunk_never_exceeds_the_dynamic_chunk(self):
        for overhead in (None, 1500):
            with self.subTest(overhead=overhead):
                sizer = _sizer(pp_size=4, overhead=overhead)
                req = _req(32768, 31130)
                self.assertEqual(sizer.chunk_size(2048, None, [req]), 2048)
                adder = _adder(1000)
                chunk = sizer.resize_after_match(req, adder, 2048, 2048)
                self.assertEqual((chunk, adder.rem_chunk_tokens), (2048, 1000))

    def test_probe_waits_for_a_cache_change(self):
        # Binary rule: an insert shrinks a full-chunk hint.
        # Load rule: an eviction grows a small-chunk hint, which it sums.
        for overhead, seqlen, cached, cached_after, hint_after in (
            (None, 32768, 0, 31130, 1638),
            (1500, 8192, 4096, 0, 8192),
        ):
            with self.subTest(overhead=overhead):
                cache = _probing_cache(version=0)
                req = _req(seqlen, 0)
                tree = _FakeTree({id(req): cached})
                sizer = _sizer(
                    pp_size=4, policy_matched=False, overhead=overhead, tree_cache=cache
                )
                with patch(f"{_MOD}.peek_uncached_tokens", side_effect=tree.peek):
                    for _ in range(3):
                        sizer.chunk_size(16384, None, [req])
                    self.assertEqual(tree.calls, 1)
                    tree.cached[id(req)] = cached_after
                    cache.prefix_cache_version += 1
                    sizer.chunk_size(16384, None, [req])
                self.assertEqual((tree.calls, req.pp_uncached_hint), (2, hint_after))

    def test_admission_corrects_a_stale_hint(self):
        # Each hint said few uncached tokens; the admission match found them all.
        adder = _adder(4096 - 1638)
        req = _req(32768, 0, hint=1638)
        self.assertEqual(_sizer().resize_after_match(req, adder, 4096, 16384), 16384)
        self.assertEqual(adder.rem_chunk_tokens, 16384 - 1638)
        load = _sizer(pp_size=4, overhead=1500)
        load.pending_tokens = 13100
        adder = _adder(4096)
        req = _req(131000, 0, hint=13100)
        self.assertEqual(load.resize_after_match(req, adder, 4096, 16384), 8192)
        self.assertEqual((adder.rem_chunk_tokens, load.pending_tokens), (8192, 131000))
        adder = _adder(4096)
        hit = _req(32768, 31130)
        self.assertEqual(_sizer().resize_after_match(hit, adder, 4096, 16384), 4096)
        self.assertEqual(adder.rem_chunk_tokens, 4096)


# 1500 tokens of overhead at pp4: 4K grows to 8K at U = 2 * 3 * 4096^2 / 1500
# (67109) and 8K to 16K at 268436.
class TestPPChunkByLoad(CustomTestCase):
    def _load(self, waiting, **kwargs):
        return _chunk(waiting, pp_size=4, overhead=1500, **kwargs)

    def test_chunk_follows_total_load(self):
        self.assertEqual(self._load([_req(67108, 0)]), 4096)
        self.assertEqual(self._load([_req(67109, 0)]), 8192)
        self.assertEqual(self._load([_req(524000, 0)]), 16384)
        # 32 x 13100 uncached tokens add up.
        self.assertEqual(self._load([_req(131000, 117900) for _ in range(32)]), 16384)
        # 3000 doubles to 12000; the last step goes to 16384 from
        # U = 3 * 12000 * 16384 / 1500 (393216), summed over three requests.
        waiting = [_req(131072, 0) for _ in range(3)]
        self.assertEqual(self._load(waiting, min_chunk=3000), 16384)
        waiting[0] = _req(131071, 0)
        self.assertEqual(self._load(waiting, min_chunk=3000), 12000)

    def test_chunked_req_keeps_its_size(self):
        # Admitted with 131000 uncached tokens, 20000 left: still 8K.
        req = _req(131000, 0, prefix_len=131000 - 20000, hint=131000)
        self.assertEqual(self._load([], chunked_req=req), 8192)
        req = _req(1040000, 0, prefix_len=1040000 - 20000)
        self.assertEqual(self._load([_req(65536, 0)], chunked_req=req), 8192)

    def test_dip_keeps_the_larger_chunk(self):
        # 2000 tokens of overhead: one 131K request alone takes 8K chunks, two 16K.
        sizer = _sizer(pp_size=4, overhead=2000)
        two = [_req(131000, 0), _req(131000, 0)]
        one = two[:1]

        def size(waiting):
            return sizer.chunk_size(16384, None, waiting)

        self.assertEqual(size(two), 16384)
        self.assertEqual([size(one) for _ in range(16)], [16384] * 16)
        self.assertEqual(size(two), 16384)
        self.assertEqual([size(one) for _ in range(17)], [16384] * 16 + [8192])
        self.assertEqual(size(one), 8192)


if __name__ == "__main__":
    unittest.main()
