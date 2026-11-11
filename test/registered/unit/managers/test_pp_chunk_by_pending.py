import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pp_chunk_sizer import PPChunkSizer
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.runtime_context import get_context

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
    # Stands in for peek_uncached_tokens; None in the table means "cannot probe".

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
    with patch(f"{_MOD}.peek_uncached_tokens", side_effect=tree.peek):
        return _chunk(waiting, chunk, policy_matched=False, **kwargs)


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

    def test_new_requests_are_probed_once(self):
        waiting = [_req(32768, 0) for _ in range(64)]
        tree = _FakeTree({id(r): 31130 for r in waiting})
        self.assertEqual(_chunk_unmatched(waiting, tree), 4096)
        self.assertEqual(tree.calls, 64)
        self.assertEqual(waiting[0].pp_uncached_hint, 1638)
        self.assertEqual(_chunk_unmatched(waiting[8:], tree), 4096)
        self.assertEqual(tree.calls, 64)

    def test_long_hint_is_probed_again_without_a_version(self):
        cached = _req(32768, 0, hint=32768)
        short = _req(32768, 0, hint=1638)
        tree = _FakeTree({id(cached): 31130})
        self.assertEqual(_chunk_unmatched([cached, short], tree), 4096)
        self.assertEqual((tree.calls, cached.pp_uncached_hint), (1, 1638))

    def test_unchanged_cache_skips_the_recheck(self):
        cache = _probing_cache(version=0)
        cold = _req(32768, 0)
        tree = _FakeTree({id(cold): 0})
        for _ in range(5):
            self.assertEqual(_chunk_unmatched([cold], tree, tree_cache=cache), 16384)
        self.assertEqual(tree.calls, 1)
        cache.prefix_cache_version += 1
        tree.cached[id(cold)] = 31130
        self.assertEqual(_chunk_unmatched([cold], tree, tree_cache=cache), 4096)
        self.assertEqual((tree.calls, cold.pp_uncached_hint), (2, 1638))

    def test_eviction_rechecks_load_hints(self):
        # Eviction only grows a hint, but the load rule sums every hint of at
        # least the small chunk, so it probes them again.
        cache = _probing_cache(version=0)
        req = _req(8192, 0)
        tree = _FakeTree({id(req): 4096})
        kwargs = dict(pp_size=4, overhead=1500, tree_cache=cache)
        sizer = _sizer(policy_matched=False, **kwargs)
        with patch(f"{_MOD}.peek_uncached_tokens", side_effect=tree.peek):
            sizer.chunk_size(16384, None, [req])
            sizer.chunk_size(16384, None, [req])
            self.assertEqual((tree.calls, sizer.pending_tokens), (1, 4096))
            tree.cached[id(req)] = 0
            cache.prefix_cache_version += 1
            sizer.chunk_size(16384, None, [req])
        self.assertEqual((tree.calls, sizer.pending_tokens), (2, 8192))

    def test_match_hint_is_probed_again(self):
        cache = _probing_cache(version=0)
        req = _req(32768, 0, hint=1638)
        _sizer(tree_cache=cache).resize_after_match(req, _adder(0), 16384, 16384)
        self.assertIsNone(req.pp_uncached_hint_version)
        tree = _FakeTree({id(req): 0})
        self.assertEqual(_chunk_unmatched([req], tree, tree_cache=cache), 16384)
        self.assertEqual((tree.calls, req.pp_uncached_hint_version), (1, 0))

    def test_unsupported_probe_keeps_full_chunk(self):
        waiting = [_req(32768, 0) for _ in range(4)]
        tree = _FakeTree({id(r): None for r in waiting})
        self.assertEqual(_chunk_unmatched(waiting, tree), 16384)
        self.assertEqual(tree.calls, 1)

    def test_admission_restores_full_chunk_for_stale_hint(self):
        sizer = _sizer()
        # The hint said 1638 uncached tokens; the admission match found 32768.
        req = _req(32768, 0, hint=1638)
        adder = _adder(4096 - 1638)
        chunk = sizer.resize_after_match(req, adder, 4096, 16384)
        self.assertEqual((chunk, adder.rem_chunk_tokens), (16384, 16384 - 1638))
        self.assertEqual(req.pp_uncached_hint, 32768)
        adder = _adder(4096)
        hit = _req(32768, 31130)
        self.assertEqual(sizer.resize_after_match(hit, adder, 4096, 16384), 4096)
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

    def test_last_step_reaches_the_full_chunk(self):
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

    def test_admission_adds_undercounted_tokens(self):
        sizer = _sizer(pp_size=4, overhead=1500)
        sizer.pending_tokens = 13100
        # The hint said 13100 uncached tokens; the admission match found 131000.
        req = _req(131000, 0, hint=13100)
        adder = _adder(4096)
        chunk = sizer.resize_after_match(req, adder, 4096, 16384)
        self.assertEqual((chunk, adder.rem_chunk_tokens), (8192, 8192))
        self.assertEqual(sizer.pending_tokens, 131000)


class _BatchBuilt(Exception):
    pass


class _WaitingReq:
    def __init__(self, events):
        self.events = events
        self.kv = SimpleNamespace(holds_kv=False)
        self.beam_group = None
        self.token_indices_to_pool = None
        self.owner_list = None

    def init_next_round_input(self, tree_cache=None, **_):
        self.events.append(("match", self))


_ADDER_PARAMS = inspect.signature(PrefillAdder.__init__)


def _wiring_scheduler(waiting, pp_chunk_sizer, events):
    class _Adder(PrefillAdder):
        def __init__(self, *args, **kwargs):
            bound = _ADDER_PARAMS.bind(self, *args, **kwargs).arguments
            self.rem_chunk_tokens = bound["rem_chunk_tokens"]
            events.append(("adder", self.rem_chunk_tokens))
            self.can_run_list = []
            self.preempt_list = []
            self.new_chunked_req = None

        def budget_state(self):
            return AddReqResult.CONTINUE

        def add_one_req(self, req, has_chunked_req, truncation_align_size):
            events.append(("add", req))
            self.can_run_list.append(req)
            return AddReqResult.CONTINUE

    s = Scheduler.__new__(Scheduler)
    s.pp_chunk_sizer = pp_chunk_sizer
    s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.waiting_queue = IndexedReqList(ListKind.WAITING)
    s.waiting_queue.join(waiting)
    s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = lambda *args, **kwargs: len(waiting)
    s.invariant_checker = Mock()
    s.req_to_token_pool = None
    s.policy = Mock(calc_priority=lambda waiting, *args, **kwargs: list(waiting))
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = 8192
    s.dynamic_chunk_sizer = None
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
    )
    s.get_prefill_adder_class = lambda: _Adder
    s.request_retirer = None
    s.page_size = 1
    s.tree_cache = SimpleNamespace(buffer_pipeline=None, storage_prefetch_retries=None)
    s.token_to_kv_pool_allocator = SimpleNamespace(
        state_slots=None, pools_share_memory=False
    )
    s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
    s.max_prefill_tokens = 1 << 20
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.max_running_requests = len(waiting)
    s.dllm_config = None
    s.enable_lora = False
    s.disaggregation_mode = DisaggregationMode.NULL
    s.disagg_prefill_bootstrap_queue = None
    s.disagg_prefill_transfer_queue = []
    s.enable_hicache_storage = False
    s.enable_lmcache = False
    s.enable_unified_cache_external_linker = False
    s.truncation_align_size = None
    s.model_config = None
    s.enable_overlap = False
    s.spec_algorithm = None
    s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
    s.running_reqs = KeyedReqList(ListKind.DECODING)
    s.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
    return s


class TestSchedulerWiring(CustomTestCase):
    def _prefill(self, make_sizer):
        events = []
        waiting = [_WaitingReq(events), _WaitingReq(events)]
        s = _wiring_scheduler(waiting, make_sizer(events), events)
        with (
            get_context().override_server_args(),
            patch.object(ScheduleBatch, "init_new", side_effect=_BatchBuilt),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt):
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None,
                    running_batch=ScheduleBatch(reqs=[]),
                )
        return events, waiting

    def test_without_sizer_the_batch_keeps_the_configured_chunk(self):
        events, (a, b) = self._prefill(lambda events: None)
        self.assertEqual(
            events,
            [("adder", 8192), ("match", a), ("add", a), ("match", b), ("add", b)],
        )

    def test_sizer_sizes_the_batch_and_restores_after_each_match(self):
        def make_sizer(events):
            def chunk_size(chunk, chunked_req, waiting_queue):
                events.append(("size", chunk, chunked_req, list(waiting_queue)))
                return 2048

            def resize_after_match(req, adder, chunk, full_chunk):
                events.append(("restore", req, chunk, full_chunk))
                return chunk * 2

            return Mock(chunk_size=chunk_size, resize_after_match=resize_after_match)

        events, (a, b) = self._prefill(make_sizer)
        self.assertEqual(
            events,
            [
                ("size", 8192, None, [a, b]),
                ("adder", 2048),
                ("match", a),
                ("restore", a, 2048, 8192),
                ("add", a),
                ("match", b),
                ("restore", b, 4096, 8192),
                ("add", b),
            ],
        )

    def _init(self, min_chunk, pp_size=4, chunked_prefill_size=8192, overhead=None):
        s = Scheduler.__new__(Scheduler)
        s.chunked_prefill_size = chunked_prefill_size
        s.tree_cache = _probing_cache()
        s.policy = SimpleNamespace()
        with get_context().override_server_args(
            pp_min_chunked_prefill_size=min_chunk,
            pp_chunk_overhead_tokens=overhead,
            pp_size=pp_size,
        ):
            s.maybe_init_pp_chunk_sizer()
        return s.pp_chunk_sizer

    def test_init_gate(self):
        self.assertIsNone(self._init(None))
        sizer = self._init(4096, overhead=1500)
        self.assertEqual(
            (sizer.min_chunk, sizer.overhead_tokens, sizer.pp_size), (4096, 1500, 4)
        )
        self.assertEqual(self._init(16384).min_chunk, 8192)


if __name__ == "__main__":
    unittest.main()
