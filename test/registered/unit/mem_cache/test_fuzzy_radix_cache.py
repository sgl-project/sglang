"""Unit tests for the ``fuzzy_match`` radix-cache backend (FuzzyRadixCache).

Exercises the backend against a scripted FuzzyMatchProvider and a recording
allocator: provider gating, fuzzy-result merge/validation fallbacks, donor
pinning across the request lifecycle, node-registry maintenance, and the
registry/factory wiring for ``--radix-cache-backend fuzzy_match``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from typing import List, Optional
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.mem_cache.fuzzy_match as fuzzy_match_pkg
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
)
from sglang.srt.mem_cache.fuzzy_match.fuzzy_radix_cache import FuzzyRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.registry import (
    _RADIX_CACHE_REGISTRY,
    TreeCacheBuildContext,
    create_tree_cache,
    get_radix_cache_factory,
)
from sglang.test.test_utils import CustomTestCase


class _RecordingAllocator:
    """Token-pool allocator stand-in: hands out increasing int64 slots and
    records every alloc/free so tests can assert slot accounting."""

    def __init__(self):
        self.device = "cpu"
        self.alloc_calls: List[int] = []
        self.free_calls: List[torch.Tensor] = []
        self.fail_alloc = False
        self._next_slot = 1000
        self._outstanding = 0

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        self.alloc_calls.append(need_size)
        if self.fail_alloc:
            return None
        start = self._next_slot
        self._next_slot += need_size
        self._outstanding += need_size
        return torch.arange(start, start + need_size, dtype=torch.int64)

    def free(self, free_index: torch.Tensor) -> None:
        self.free_calls.append(free_index)
        self._outstanding -= int(free_index.numel())

    @property
    def outstanding_slots(self) -> int:
        return self._outstanding


class _StubReqToTokenPool:
    """req_to_token table backed by a plain tensor."""

    def __init__(self, *, max_batch: int = 4, max_seq_len: int = 32):
        self.req_to_token = torch.zeros(max_batch, max_seq_len, dtype=torch.int64)


class _StubReq:
    """Carries exactly the request fields FuzzyRadixCache reads and writes."""

    def __init__(
        self,
        *,
        rid: str,
        origin_input_ids: Optional[List[int]] = None,
        output_ids: Optional[List[int]] = None,
        req_pool_idx: int = 0,
        committed_kv_len: int = 0,
    ):
        self.rid = rid
        self.extra_key = None
        self.priority = 0
        self.cache_protected_len = 0
        self.cache_fuzzy_matched_len = 0
        self.fuzzy_match_result = None
        self.fuzzy_donor_node = None
        self.fuzzy_realized_locs = None
        self.origin_input_ids = list(origin_input_ids or [])
        self.output_ids = list(output_ids or [])
        self.req_pool_idx = req_pool_idx
        self.last_node = None
        self._committed_kv_len = committed_kv_len

    def pop_committed_kv_cache(self) -> int:
        return self._committed_kv_len


class _ScriptedProvider(FuzzyMatchProvider):
    """Answers every prefix miss with ``self.result``; records all calls."""

    def __init__(self, *, config: FuzzyMatchConfig, result=None):
        super().__init__(config)
        self.result = result
        self.match_calls = []
        self.finished_rids = []
        self.reset_count = 0

    def cache_on_request_finished(
        self,
        request,
        token_ids,
        kv_cache,
        cache_start_pos,
        cache_end_pos,
        radix_tree=None,
    ) -> bool:
        self.finished_rids.append(request.rid)
        return True

    def match_on_prefix_miss(
        self, prompt_token_ids, already_matched_len, request=None, extra_key=None
    ):
        self.match_calls.append(
            {
                "prompt_token_ids": list(prompt_token_ids),
                "already_matched_len": already_matched_len,
                "extra_key": extra_key,
            }
        )
        return self.result

    def on_cache_reset(self) -> None:
        self.reset_count += 1


def _key(token_ids: List[int]) -> RadixKey:
    return RadixKey(token_ids=array("q", token_ids), extra_key=None)


def _make_cache(
    *,
    scripted_result: Optional[FuzzyMatchResult] = None,
    min_match_length: int = 1,
    req_to_token_pool: Optional[_StubReqToTokenPool] = None,
):
    allocator = _RecordingAllocator()
    cache = FuzzyRadixCache(
        params=CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    config = FuzzyMatchConfig(
        enable_fuzzy_match=True,
        fuzzy_min_match_length=min_match_length,
        cache_fuzzy_results=False,
        # Tests use short keys; the suffix gate is exercised explicitly in
        # its own case.
        fuzzy_min_suffix_tokens=0,
    )
    provider = _ScriptedProvider(config=config, result=scripted_result)
    cache.init_fuzzy_match(config=config, provider=provider)
    return cache, provider, allocator


def _seed_exact(cache, *, token_ids: List[int], values: List[int]):
    return cache.insert(
        InsertParams(
            key=_key(token_ids),
            value=torch.tensor(values, dtype=torch.int64),
        )
    )


def _scripted_fuzzy_result(
    *,
    cached_token_count: int,
    kv_indices: List[int],
    cached_start_pos: int,
    donor_last_node_id: Optional[int] = None,
) -> FuzzyMatchResult:
    return FuzzyMatchResult(
        cached_token_count=cached_token_count,
        cached_token_ids=list(range(cached_token_count)),
        prompt_token_count=cached_token_count,
        kv_cache_indices=torch.tensor(kv_indices, dtype=torch.int64),
        position_offset=0,
        cached_start_pos=cached_start_pos,
        donor_last_node_id=donor_last_node_id,
    )


def _iter_nodes(node):
    yield node
    for child in node.children.values():
        yield from _iter_nodes(child)


class TestFuzzyMatchGating(CustomTestCase):
    def test_internal_rematch_without_request_never_queries_provider(self):
        """``cache_unfinished_req``'s internal re-match passes req=None and
        has no request to attach donor/realization state to; if the gate is
        dropped, the provider is consulted and ``_apply_fuzzy_result``
        crashes (or corrupts state) on the missing request."""
        scripted = _scripted_fuzzy_result(
            cached_token_count=2, kv_indices=[50, 51], cached_start_pos=0
        )
        cache, provider, _allocator = _make_cache(scripted_result=scripted)
        _seed_exact(cache, token_ids=[1, 2, 3], values=[10, 11, 12])

        result = cache.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9])))

        self.assertEqual(provider.match_calls, [])
        self.assertIsNone(result.fuzzy_matched_len)
        self.assertEqual(result.device_indices.tolist(), [10, 11, 12])

    def test_short_suffix_skips_provider_lookup(self):
        """The suffix gate must skip the provider (and its embedding cost)
        when the exact-miss suffix is below fuzzy_min_suffix_tokens; if it
        degrades, every short-suffix miss pays a full semantic lookup —
        the ~10%% no-hit throughput overhead this gate exists to bound."""
        scripted = _scripted_fuzzy_result(
            cached_token_count=2, kv_indices=[50, 51], cached_start_pos=0
        )
        cache, provider, _allocator = _make_cache(scripted_result=scripted)
        cache.fuzzy_config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_min_match_length=1,
            cache_fuzzy_results=False,
            fuzzy_min_suffix_tokens=8,
        )
        _seed_exact(cache, token_ids=[1, 2, 3], values=[10, 11, 12])
        req = _StubReq(rid="short-suffix")

        # Suffix of 3 (< 8): gated, provider never consulted.
        result = cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9]), req=req)
        )
        self.assertEqual(provider.match_calls, [])
        self.assertIsNone(result.fuzzy_matched_len)

        # Suffix of 8 (>= 8): gate passes, provider consulted.
        cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9, 20, 21, 22, 23, 24]), req=req)
        )
        self.assertEqual(len(provider.match_calls), 1)

    def test_short_exact_anchor_skips_provider_but_zero_anchor_queries(self):
        """The min-length gate is ``0 < exact < min``: partial exact anchors
        below the floor must not trigger fuzzy lookups, while a zero exact
        prefix stays eligible. A '<=' rewrite silently disables fuzzy
        matching for every cold prompt."""
        cache, provider, _allocator = _make_cache(min_match_length=8)
        _seed_exact(cache, token_ids=[1, 2, 3], values=[10, 11, 12])

        anchored = _StubReq(rid="anchored")
        result = cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9]), req=anchored)
        )
        self.assertEqual(provider.match_calls, [])
        self.assertIsNone(result.fuzzy_matched_len)

        cold = _StubReq(rid="cold")
        cache.match_prefix(MatchPrefixParams(key=_key([40, 41, 42]), req=cold))
        self.assertEqual(len(provider.match_calls), 1)
        self.assertEqual(provider.match_calls[0]["already_matched_len"], 0)


class TestFuzzyResultApplication(CustomTestCase):
    def test_valid_fuzzy_result_merges_indices_and_marks_request(self):
        """A validated fuzzy hit must extend ``device_indices`` with the
        donor slots, report ``fuzzy_matched_len`` and
        ``cache_protected_len = exact + fuzzy``, stash the result on the
        request, and pre-allocate realization slots when the donor span is
        not position-aligned; losing any of these either recomputes the
        donor tokens or feeds position-wrong KV to the forward pass."""
        scripted = _scripted_fuzzy_result(
            cached_token_count=2, kv_indices=[50, 51], cached_start_pos=5
        )
        cache, provider, allocator = _make_cache(scripted_result=scripted)
        _seed_exact(cache, token_ids=[1, 2, 3], values=[10, 11, 12])
        req = _StubReq(rid="recipient")

        result = cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9]), req=req)
        )

        self.assertEqual(result.device_indices.tolist(), [10, 11, 12, 50, 51])
        self.assertEqual(result.fuzzy_matched_len, 2)
        self.assertEqual(result.cache_protected_len, 5)
        self.assertIs(req.fuzzy_match_result, scripted)
        self.assertIsNotNone(req.fuzzy_realized_locs)
        self.assertEqual(int(req.fuzzy_realized_locs.numel()), 2)
        self.assertEqual(allocator.alloc_calls, [2])
        call = provider.match_calls[0]
        self.assertEqual(call["prompt_token_ids"], [1, 2, 3, 7, 8, 9])
        self.assertEqual(call["already_matched_len"], 3)

    def test_position_aligned_result_falls_back_to_exact_only(self):
        """A donor span already aligned to the recipient position
        (``cached_start_pos == exact`` and no segments) must be DROPPED:
        reusing it would put donor-owned slots into the recipient's branch
        at finish-time insert — two tree nodes owning the same pool slots,
        which double-frees on eviction. Aligned token+position content is
        the exact radix tree's job."""
        scripted = _scripted_fuzzy_result(
            cached_token_count=2, kv_indices=[50, 51], cached_start_pos=3
        )
        cache, _provider, allocator = _make_cache(scripted_result=scripted)
        _seed_exact(cache, token_ids=[1, 2, 3], values=[10, 11, 12])
        req = _StubReq(rid="aligned")

        result = cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9]), req=req)
        )

        self.assertEqual(result.device_indices.tolist(), [10, 11, 12])
        self.assertIsNone(result.fuzzy_matched_len)
        self.assertEqual(allocator.alloc_calls, [])
        self.assertIsNone(req.fuzzy_realized_locs)
        self.assertIsNone(req.fuzzy_match_result)

    def test_stale_donor_node_id_falls_back_to_exact_only(self):
        """A donor whose tree node is gone (flush/eviction race) cannot be
        reused: the match must degrade to exact-only with request state
        untouched, or the forward pass reads freed KV slots."""
        scripted = _scripted_fuzzy_result(
            cached_token_count=2,
            kv_indices=[50, 51],
            cached_start_pos=5,
            donor_last_node_id=10**12,
        )
        cache, _provider, allocator = _make_cache(scripted_result=scripted)
        _seed_exact(cache, token_ids=[1, 2, 3], values=[10, 11, 12])
        req = _StubReq(rid="stale-donor")

        result = cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9]), req=req)
        )

        self.assertEqual(result.device_indices.tolist(), [10, 11, 12])
        self.assertIsNone(result.fuzzy_matched_len)
        self.assertIsNone(result.cache_protected_len)
        self.assertIsNone(req.fuzzy_match_result)
        self.assertIsNone(req.fuzzy_donor_node)
        self.assertIsNone(req.fuzzy_realized_locs)
        self.assertEqual(allocator.alloc_calls, [])

    def test_kv_indices_length_mismatch_falls_back_without_leaking(self):
        """A provider result whose kv_cache_indices length disagrees with
        ``cached_token_count`` violates the prefix-shaped device_indices
        contract and must be rejected without any net pool allocation;
        accepting it misaligns every token after the exact prefix."""
        scripted = _scripted_fuzzy_result(
            cached_token_count=3, kv_indices=[50, 51], cached_start_pos=5
        )
        cache, _provider, allocator = _make_cache(scripted_result=scripted)
        _seed_exact(cache, token_ids=[1, 2, 3], values=[10, 11, 12])
        req = _StubReq(rid="mismatched")

        result = cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 7, 8, 9]), req=req)
        )

        self.assertEqual(result.device_indices.tolist(), [10, 11, 12])
        self.assertIsNone(result.fuzzy_matched_len)
        self.assertIsNone(req.fuzzy_match_result)
        self.assertIsNone(req.fuzzy_realized_locs)
        self.assertEqual(allocator.outstanding_slots, 0)

    def test_alloc_failure_falls_back_without_pinning_donor(self):
        """When the pool cannot supply realization slots, the match must
        degrade to exact-only with no donor pin and no request state; a
        pin taken before the (failed) alloc would never be released and
        permanently protects the donor from eviction."""
        cache, provider, allocator = _make_cache()
        donor = _seed_exact(
            cache, token_ids=[1, 2, 3, 4], values=[10, 11, 12, 13]
        ).last_device_node
        provider.result = _scripted_fuzzy_result(
            cached_token_count=2,
            kv_indices=[10, 11],
            cached_start_pos=2,
            donor_last_node_id=donor.id,
        )
        allocator.fail_alloc = True
        req = _StubReq(rid="starved")

        result = cache.match_prefix(MatchPrefixParams(key=_key([90, 91, 92]), req=req))

        self.assertIsNone(result.fuzzy_matched_len)
        self.assertEqual(int(result.device_indices.numel()), 0)
        self.assertEqual(donor.lock_ref, 0)
        self.assertIsNone(req.fuzzy_match_result)
        self.assertIsNone(req.fuzzy_donor_node)
        self.assertIsNone(req.fuzzy_realized_locs)


class TestDonorLifecycle(CustomTestCase):
    def test_finished_request_unpins_donor_and_frees_realized_locs(self):
        """A fuzzy hit pins the donor node (lock_ref +1) for the recipient's
        lifetime; ``cache_finished_req`` must release the pin and return
        unconsumed realization slots to the allocator. Skipping the release
        leaks lock_ref (donor unevictable forever); skipping the free leaks
        pool slots on every fuzzy request."""
        pool = _StubReqToTokenPool()
        pool.req_to_token[0, :3] = torch.tensor([70, 71, 72], dtype=torch.int64)
        cache, provider, allocator = _make_cache(req_to_token_pool=pool)
        donor = _seed_exact(
            cache, token_ids=[1, 2, 3, 4], values=[10, 11, 12, 13]
        ).last_device_node
        provider.result = _scripted_fuzzy_result(
            cached_token_count=2,
            kv_indices=[10, 11],
            cached_start_pos=2,
            donor_last_node_id=donor.id,
        )
        req = _StubReq(
            rid="recipient",
            origin_input_ids=[90, 91],
            output_ids=[92],
            req_pool_idx=0,
            committed_kv_len=3,
        )

        cache.match_prefix(MatchPrefixParams(key=_key([90, 91, 92]), req=req))

        self.assertEqual(donor.lock_ref, 1)
        self.assertIs(req.fuzzy_donor_node, donor)
        realized = req.fuzzy_realized_locs
        self.assertIsNotNone(realized)

        cache.cache_finished_req(req)

        self.assertEqual(donor.lock_ref, 0)
        self.assertIsNone(req.fuzzy_donor_node)
        self.assertIsNone(req.fuzzy_realized_locs)
        self.assertTrue(any(t is realized for t in allocator.free_calls))
        self.assertEqual(allocator.outstanding_slots, 0)

    def test_same_donor_refire_keeps_single_pin(self):
        """(bug regression) A queued request can run match_prefix on several
        scheduling rounds and hit the SAME donor each time. Each re-fire
        must not stack another pin: with inc-per-fire but one dec at
        finish, the donor branch ends with lock_ref > 0 forever
        (protected-size leak, donor unevictable)."""
        pool = _StubReqToTokenPool()
        pool.req_to_token[0, :3] = torch.tensor([70, 71, 72], dtype=torch.int64)
        cache, provider, allocator = _make_cache(req_to_token_pool=pool)
        donor = _seed_exact(
            cache, token_ids=[1, 2, 3, 4], values=[10, 11, 12, 13]
        ).last_device_node
        provider.result = _scripted_fuzzy_result(
            cached_token_count=2,
            kv_indices=[10, 11],
            cached_start_pos=2,
            donor_last_node_id=donor.id,
        )
        req = _StubReq(
            rid="refire",
            origin_input_ids=[90, 91],
            output_ids=[92],
            req_pool_idx=0,
            committed_kv_len=3,
        )

        cache.match_prefix(MatchPrefixParams(key=_key([90, 91, 92]), req=req))
        cache.match_prefix(MatchPrefixParams(key=_key([90, 91, 92]), req=req))

        self.assertEqual(donor.lock_ref, 1)

        cache.cache_finished_req(req)

        self.assertEqual(donor.lock_ref, 0)
        self.assertEqual(allocator.outstanding_slots, 0)

    def test_realization_narrows_protected_prefix_to_exact(self):
        """(bug regression) After the realizer repoints the fuzzy span to
        request-owned slots, ``req.cache_protected_len`` must shrink from
        exact+fuzzy to exact. If it keeps claiming the span is tree-owned
        and a concurrent request extends the shared tree path past
        ``exact`` before this request finishes, the finish-time
        duplicate-free window [cache_protected_len : insert_prefix_len] is
        empty and the realized slots are orphaned — neither tree-adopted
        nor freed (observed as a 2-slot pool-consistency failure on GSM8K
        near-duplicates)."""
        from sglang.srt.mem_cache.fuzzy_match.realizer import FuzzyKVRealizer

        pool = _StubReqToTokenPool()
        allocator = _RecordingAllocator()
        realizer = FuzzyKVRealizer.__new__(FuzzyKVRealizer)
        realizer.req_to_token_pool = pool
        realizer.token_to_kv_pool_allocator = allocator
        realizer.pool = MagicMock(k_buffer=[torch.zeros(1)])
        realizer.pool_supported = True
        realizer.rotary_emb = MagicMock()

        req = _StubReq(rid="narrow", req_pool_idx=0)
        req.prefix_indices = torch.arange(5)  # exact 3 + fuzzy 2
        req.cache_protected_len = 5
        req.cache_fuzzy_matched_len = 2
        req.fuzzy_realized_locs = torch.tensor([500, 501], dtype=torch.int64)
        req.fuzzy_match_result = _scripted_fuzzy_result(
            cached_token_count=2, kv_indices=[10, 11], cached_start_pos=0
        )

        with patch(
            "sglang.srt.mem_cache.fuzzy_match.realizer." "copy_kv_with_rope_correction"
        ):
            realizer.realize(fuzzy_reqs=[req])

        self.assertEqual(req.cache_protected_len, 3)
        self.assertEqual(pool.req_to_token[0, 3:5].tolist(), [500, 501])
        self.assertIsNone(req.fuzzy_realized_locs)
        self.assertEqual(req.cache_fuzzy_matched_len, 0)

    def test_finished_insert_registers_donor_before_duplicate_free(self):
        """With ``cache_fuzzy_results`` on, a finishing request must be
        offered to the provider while its KV slots are still live (between
        insert and duplicate-free), and the provider must receive the
        TreeNode id under which the donor is addressable in the node
        registry. Moving the hook after the frees hands the provider
        recycled slots; dropping ``on_donor_inserted`` leaves donors
        unaddressable and every future fuzzy hit falls back."""
        pool = _StubReqToTokenPool()
        pool.req_to_token[0, :3] = torch.tensor([70, 71, 72], dtype=torch.int64)
        allocator = _RecordingAllocator()
        cache = FuzzyRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
        config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_min_match_length=1,
            cache_fuzzy_results=True,
        )
        captured = {}

        class _CaptureProvider(FuzzyMatchProvider):
            def cache_on_request_finished(
                self,
                request,
                token_ids,
                kv_cache,
                cache_start_pos,
                cache_end_pos,
                radix_tree=None,
            ) -> bool:
                captured["rid"] = request.rid
                captured["token_ids"] = list(token_ids)
                captured["kv_cache"] = kv_cache.tolist()
                captured["cache_end_pos"] = cache_end_pos
                captured["frees_at_hook"] = len(allocator.free_calls)
                return True

            def match_on_prefix_miss(
                self,
                prompt_token_ids,
                already_matched_len,
                request=None,
                extra_key=None,
            ):
                return None

            def on_donor_inserted(self, request, donor_last_node_id: int) -> None:
                captured["donor_last_node_id"] = donor_last_node_id

        cache.init_fuzzy_match(config=config, provider=_CaptureProvider(config))
        req = _StubReq(
            rid="donor-req",
            origin_input_ids=[90, 91],
            output_ids=[92],
            req_pool_idx=0,
            committed_kv_len=3,
        )

        cache.cache_finished_req(req)

        self.assertEqual(captured["rid"], "donor-req")
        self.assertEqual(captured["token_ids"], [90, 91, 92])
        self.assertEqual(captured["kv_cache"], [70, 71, 72])
        self.assertEqual(captured["cache_end_pos"], 3)
        self.assertEqual(
            captured["frees_at_hook"],
            0,
            "donor must be handed to the provider before any pool frees",
        )
        donor_node = cache._node_registry[captured["donor_last_node_id"]]
        self.assertEqual(list(donor_node.key.token_ids), [90, 91, 92])


class TestNodeRegistry(CustomTestCase):
    def test_registry_tracks_insert_split_and_delete(self):
        """Donors are addressed by TreeNode.id, so every reachable node must
        be resolvable through ``_node_registry`` after inserts and splits,
        and deleted leaves must drop out. A missed split registration makes
        valid donors unaddressable; a missed delete leaves stale ids that
        resolve to evicted nodes."""
        cache, _provider, _allocator = _make_cache()
        _seed_exact(cache, token_ids=[1, 2, 3, 4], values=[10, 11, 12, 13])
        # Forces a split at length 2: [1,2] -> {[3,4], [9]}
        _seed_exact(cache, token_ids=[1, 2, 9], values=[10, 11, 99])

        reachable = list(_iter_nodes(cache.root_node))
        self.assertEqual(len(reachable), 4)  # root, [1,2], [3,4], [9]
        for node in reachable:
            self.assertIs(cache._node_registry.get(node.id), node)

        cache.evict(EvictParams(num_tokens=16))

        self.assertEqual(set(cache._node_registry), {cache.root_node.id})

    def test_reset_clears_registry_and_notifies_provider(self):
        """A cache flush must clear the node registry (so stale donor ids
        from before the flush are detected as gone) and tell the provider to
        drop its donor index; keeping either alive lets a post-flush match
        resolve a donor whose KV slots were recycled."""
        cache, provider, _allocator = _make_cache()
        donor = _seed_exact(
            cache, token_ids=[1, 2, 3, 4], values=[10, 11, 12, 13]
        ).last_device_node

        cache.reset()

        self.assertEqual(provider.reset_count, 1)
        self.assertNotIn(donor.id, cache._node_registry)
        self.assertIs(cache._node_registry[cache.root_node.id], cache.root_node)


class _RegistryIsolationMixin:
    """Restore the global backend registry around each test (same pattern as
    test_registry.py) so registrations don't leak between tests."""

    def setUp(self):
        super().setUp()
        self._registry_snapshot = dict(_RADIX_CACHE_REGISTRY)

    def tearDown(self):
        _RADIX_CACHE_REGISTRY.clear()
        _RADIX_CACHE_REGISTRY.update(self._registry_snapshot)
        super().tearDown()


def _make_ctx(*, backend, disable_radix_cache=False, is_eagle=False):
    server_args = MagicMock()
    server_args.radix_cache_backend = backend
    server_args.enable_streaming_session = False
    server_args.enable_lmcache = False
    server_args.enable_flexkv = False
    params = MagicMock()
    params.is_eagle = is_eagle
    return TreeCacheBuildContext(
        server_args=server_args,
        params=params,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        enable_hierarchical_cache=False,
        disable_radix_cache=disable_radix_cache,
        effective_chunked_prefill_size=None,
        tp_worker=MagicMock(),
        model_config=MagicMock(),
        tp_size=1,
        tp_rank=0,
        tp_group=MagicMock(),
        full_tokens_per_layer=None,
    )


class TestBackendRegistration(_RegistryIsolationMixin, CustomTestCase):
    def test_import_registers_fuzzy_match_backend(self):
        """Importing the package must register the ``fuzzy_match`` factory;
        without it ``--radix-cache-backend fuzzy_match`` is rejected as an
        unknown backend at startup."""
        self.assertIs(
            get_radix_cache_factory("fuzzy_match"),
            fuzzy_match_pkg.fuzzy_match_backend_factory,
        )

    def test_create_tree_cache_routes_and_rejects_disable_radix_cache(self):
        """``create_tree_cache`` must dispatch ``fuzzy_match`` to the real
        factory, which refuses ``--disable-radix-cache`` (the backend needs
        the exact tree as donor ground truth). The message check separates
        this from the generic unknown-backend ValueError, so a broken
        registration cannot masquerade as a pass."""
        with self.assertRaises(ValueError) as caught:
            create_tree_cache(
                _make_ctx(backend="fuzzy_match", disable_radix_cache=True)
            )
        self.assertIn("disable-radix-cache", str(caught.exception))

    def test_create_tree_cache_rejects_eagle(self):
        """The factory must refuse EAGLE speculative decoding, which the
        backend does not support yet; silently constructing the cache would
        run donor KV reuse against bigram keys it was never validated on."""
        with self.assertRaises(ValueError) as caught:
            create_tree_cache(_make_ctx(backend="fuzzy_match", is_eagle=True))
        self.assertIn("EAGLE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
