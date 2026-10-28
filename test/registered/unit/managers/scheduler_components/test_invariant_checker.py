import os
import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.scheduler_request_lists import make_request_registry
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.request_lists import CoreLists, RequestList
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components import invariant_checker
from sglang.srt.managers.scheduler_components.engine_state import LaunchCounters
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
    kv_private_swa_tokens,
)
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.session.streaming_session import SessionSlot

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestCheckTreeCacheGate(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    @contextmanager
    def _without_explicit_sanity_check_setting(self):
        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.clear()
            yield

    def _make_checker(self, prefix_sharing=True, hybrid_swa=True, hybrid_ssm=False):
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = prefix_sharing
        tree_cache.supports_swa.return_value = hybrid_swa
        tree_cache.supports_mamba.return_value = hybrid_ssm
        return SchedulerInvariantChecker(
            capacity=SimpleNamespace(
                is_hybrid_swa=hybrid_swa,
                is_hybrid_ssm=hybrid_ssm,
                full_tokens_per_layer=None,
                swa_tokens_per_layer=None,
                widened_max_total_num_tokens=1024,
            ),
            kv=SimpleNamespace(
                tree_cache=tree_cache,
                token_to_kv_pool_allocator=MagicMock(),
                req_to_token_pool=MagicMock(),
            ),
            pool_stats_observer=MagicMock(),
            counters=LaunchCounters(),
            request_registry=make_request_registry(),
            pending=PendingSteps(),
            scheduler_stage_metrics=None,
        )

    def test_disabled_by_default(self):
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker._check_tree_cache()

            checker.kv.tree_cache.sanity_check.assert_not_called()

    def test_enabled_by_default_in_ci(self):
        with (
            envs.SGLANG_IS_IN_CI.override(True),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker._check_tree_cache()

            checker.kv.tree_cache.sanity_check.assert_called_once()

    def test_explicitly_disabled_in_ci(self):
        with envs.SGLANG_IS_IN_CI.override(True):
            checker = self._make_checker()

            with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(False):
                checker._check_tree_cache()

            checker.kv.tree_cache.sanity_check.assert_not_called()

    def test_runs_when_enabled(self):
        with envs.SGLANG_IS_IN_CI.override(False):
            checker = self._make_checker()

            with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(True):
                checker._check_tree_cache()

            checker.kv.tree_cache.sanity_check.assert_called_once()

    def test_skipped_without_prefix_sharing(self):
        with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(True):
            checker = self._make_checker(
                prefix_sharing=False, hybrid_swa=False, hybrid_ssm=True
            )

            checker._check_tree_cache()

            checker.kv.tree_cache.sanity_check.assert_not_called()


class TestPrivateSwaTokens(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_floor_shielded_prefix_stays_private(self):
        # A prefill-aware SWA request with radix off: the window cursor jumps to
        # the floor (278) without freeing it, then frees [278, 300).
        kv = ReqKvInfo(
            req_pool_idx=0,
            kv_allocated_len=280,
            swa_evict_floor=278,
            component_evicted_seqlens={ComponentType.SWA: 278},
        )
        self.assertEqual(kv_private_swa_tokens(kv, page_size=1), 280)
        kv.kv_allocated_len = 450
        kv.set_evicted_seqlen(ComponentType.SWA, 300)
        self.assertEqual(kv_private_swa_tokens(kv, page_size=1), 428)


class TestShardedFullPoolInvariant(CustomTestCase):
    PAGE_SIZE = 16
    SHARD_SIZE = 4

    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(page_size=self.PAGE_SIZE))
        parallel = patch.object(
            invariant_checker,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=False),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def _make_checker(self):
        self.allocator = PageInterleavePoolAllocator(
            size=4 * self.PAGE_SIZE,
            physical_page_size=self.PAGE_SIZE,
            shard_size=self.SHARD_SIZE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        req_pool = ReqToTokenPool(
            size=4,
            max_context_len=128,
            device="cpu",
            enable_memory_saver=False,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            self.cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=req_pool,
                    token_to_kv_pool_allocator=self.allocator,
                    page_size=self.PAGE_SIZE,
                    tree_components=(ComponentType.FULL,),
                )
            )
        self.counters = LaunchCounters(cur_batch_for_debug=SimpleNamespace(reqs=[]))
        self.core_lists = CoreLists()
        self.mode_owners = ModeOwners()
        observer = SchedulerPoolStatsObserver(
            kv=SimpleNamespace(
                tree_cache=self.cache,
                token_to_kv_pool_allocator=self.allocator,
                req_to_token_pool=req_pool,
            ),
            session_controller=None,
            hisparse_coordinator=None,
            capacity=SimpleNamespace(
                is_hybrid_swa=False,
                is_hybrid_ssm=False,
                full_tokens_per_layer=None,
                swa_tokens_per_layer=None,
                widened_max_total_num_tokens=self.allocator.size,
            ),
        )
        return SchedulerInvariantChecker(
            capacity=SimpleNamespace(
                is_hybrid_swa=False,
                is_hybrid_ssm=False,
                full_tokens_per_layer=None,
                swa_tokens_per_layer=None,
                widened_max_total_num_tokens=self.allocator.size,
            ),
            kv=SimpleNamespace(
                tree_cache=self.cache,
                token_to_kv_pool_allocator=self.allocator,
                req_to_token_pool=req_pool,
            ),
            pool_stats_observer=observer,
            counters=self.counters,
            request_registry=make_request_registry(
                core_lists=self.core_lists, mode_owners=self.mode_owners
            ),
            pending=PendingSteps(),
            scheduler_stage_metrics=None,
        )

    def _allocate(self, tokens, *, prefix=0, last_loc=-1, rotation_base=None):
        prefix_lens = torch.tensor([prefix], dtype=torch.int64)
        seq_lens = torch.tensor([prefix + tokens], dtype=torch.int64)
        bases = [rotation_base]
        locs = self.allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            last_loc=torch.tensor([last_loc], dtype=torch.int64),
            extend_num_tokens=tokens,
            rotation_bases=bases,
        )
        self.assertIsNotNone(locs)
        return locs, bases[0]

    def _cache_prefix(self, tokens, *, protected=False):
        locs, base = self._allocate(tokens)
        key = RadixKey(array("q", range(tokens)))
        self.cache.insert(InsertParams(key=key, value=locs, rotation_base=base))
        if protected:
            matched = self.cache.match_prefix(MatchPrefixParams(key=key))
            self.cache.inc_lock_ref(matched.last_device_node)
        return locs, base

    def _active_partial_page(self, checker):
        prefix_locs, base = self._cache_prefix(self.PAGE_SIZE, protected=True)
        tail_locs, _ = self._allocate(
            3,
            prefix=self.PAGE_SIZE,
            last_loc=prefix_locs[-1].item(),
            rotation_base=base,
        )
        locs = torch.cat((prefix_locs, tail_locs))
        checker.kv.req_to_token_pool.req_to_token[0, : len(locs)] = locs
        req = SimpleNamespace(
            kv=ReqKvInfo(
                req_pool_idx=0,
                kv_committed_len=len(locs),
                kv_allocated_len=len(locs),
                cache_protected_len=self.PAGE_SIZE,
            ),
            beam_group=None,
        )
        self.counters.cur_batch_for_debug = SimpleNamespace(reqs=[req])
        self.core_lists.decoding = [req]
        return req

    def test_balanced_and_skewed_cache_ownership_conserve_full_pool(self):
        for cached_pages in (1, self.SHARD_SIZE):
            for protected in (False, True):
                with self.subTest(cached_pages=cached_pages, protected=protected):
                    checker = self._make_checker()
                    cached_tokens = cached_pages * self.PAGE_SIZE
                    self._cache_prefix(cached_tokens, protected=protected)
                    self.assertEqual(
                        self.cache.protected_size() + self.cache.evictable_size(),
                        cached_tokens,
                    )
                    stats = checker.pool_stats_observer.get_pool_stats()
                    if cached_pages == 1:
                        self.assertLess(
                            stats.full_available_size,
                            self.allocator.aggregate_free_size(),
                        )
                    else:
                        self.assertEqual(
                            stats.full_available_size,
                            self.allocator.aggregate_free_size(),
                        )
                    leak, msg = checker._check_full_pool(stats)
                    self.assertFalse(leak, msg)
                    self.assertIn("class_free_pages=", msg)

    def test_idle_check_detects_one_unowned_page(self):
        checker = self._make_checker()
        self._cache_prefix(self.PAGE_SIZE)
        self._allocate(self.PAGE_SIZE)  # Deliberately no request or cache owner.
        leak, messages = checker._check_all_pools(
            checker.pool_stats_observer.get_pool_stats()
        )
        self.assertTrue(leak, messages)

    def test_busy_check_detects_one_unowned_page(self):
        checker = self._make_checker()
        self._active_partial_page(checker)
        self._allocate(self.PAGE_SIZE)  # Leak in addition to the active tail.
        self.assertEqual(checker._get_total_uncached_sizes(), (self.PAGE_SIZE, 0))
        with self.assertRaisesRegex(AssertionError, "Full Pool Mem Leak Detected"):
            checker.self_check_during_busy()

    def test_partial_active_page_is_accounted_without_slack_exemption(self):
        checker = self._make_checker()
        req = self._active_partial_page(checker)
        # Count the same request only once, including when it is parked
        # between prefill chunks instead of decoding.
        for parked in (False, True):
            with self.subTest(parked=parked):
                self.core_lists.chunked_req = req
                if parked:
                    self.core_lists.decoding = []
                uncached, swa_uncached = checker._get_total_uncached_sizes()
                self.assertEqual((uncached, swa_uncached), (self.PAGE_SIZE, 0))
                stats = checker.pool_stats_observer.get_pool_stats()
                leak, msg = checker._check_full_pool(stats, uncached=uncached)
                self.assertFalse(leak, msg)
                self.assertNotIn("slack_allowed", msg)
                with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False):
                    checker.self_check_during_busy()

    def test_mode_lists_holding_kv_own_their_rows(self):
        """A request whose KV row a PD or HiSparse list holds is counted as
        uncached by the busy check, not reported as a leak."""

        def hisparse(req):
            coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
            coordinator.staging = [req]
            return dict(hisparse_coordinator=coordinator)

        cases = {
            "pd_decode_transfer": lambda req: dict(
                disagg_decode=decode_owner_shell(
                    transfer_queue=SimpleNamespace(queue=[SimpleNamespace(req=req)])
                )
            ),
            "pd_decode_ready": lambda req: dict(
                disagg_decode=decode_owner_shell(ready=[req])
            ),
            "pd_decode_releasing": lambda req: dict(
                disagg_decode=decode_owner_shell(releasing=[req])
            ),
            "pd_prefill_prefilling": lambda req: dict(
                disagg_prefill=prefill_owner_shell(prefilling=[req])
            ),
            "pd_prefill_transfer": lambda req: dict(
                disagg_prefill=prefill_owner_shell(transfer=[req])
            ),
            "hisparse_staging": hisparse,
        }
        for name, owners in cases.items():
            with self.subTest(name):
                checker = self._make_checker()
                req = self._active_partial_page(checker)
                self.core_lists.decoding = []
                for field, owner in owners(req).items():
                    setattr(self.mode_owners, field, owner)
                self.assertEqual(
                    checker._get_total_uncached_sizes(), (self.PAGE_SIZE, 0)
                )
                with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False):
                    checker.self_check_during_busy()

    def test_session_record_is_counted_once_by_its_owner(self):
        checker = self._make_checker()
        req = self._active_partial_page(checker)
        # The session owns the row: session-held whether the turn is in a batch
        # or parked between prefill chunks, never also uncached.
        self.cache.session.slots["s"] = SessionSlot(kv=req.kv)
        for parked in (False, True):
            with self.subTest(parked=parked):
                self.core_lists.chunked_req = req
                if parked:
                    self.core_lists.decoding = []
                self.assertEqual(checker._get_total_uncached_sizes(), (0, 0))
                self.assertEqual(
                    checker.pool_stats_observer.session_held_tokens(), self.PAGE_SIZE
                )
                with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False):
                    checker.self_check_during_busy()

    def test_misaligned_evictable_counter_is_not_rounded_up(self):
        checker = self._make_checker()
        self._cache_prefix(self.PAGE_SIZE)
        self.cache.tree_core.component_evictable_size_[ComponentType.FULL] -= 1
        leak, msg = checker._check_full_pool(
            checker.pool_stats_observer.get_pool_stats()
        )
        self.assertTrue(leak, msg)
        self.assertIn(f"evictable={self.PAGE_SIZE - 1}", msg)

    def test_compensating_misaligned_counters_still_report_alignment_error(self):
        checker = self._make_checker()
        self._cache_prefix(self.PAGE_SIZE)
        core = self.cache.tree_core
        core.component_evictable_size_[ComponentType.FULL] -= 1
        core.component_protected_size_[ComponentType.FULL] += 1
        # Their sum still matches the allocated page; neither individual
        # counter can legitimately contain a fraction of a physical page.
        self.assertEqual(
            self.cache.evictable_size() + self.cache.protected_size(), self.PAGE_SIZE
        )
        leak, msg = checker._check_full_pool(
            checker.pool_stats_observer.get_pool_stats()
        )
        self.assertTrue(leak, msg)
        self.assertIn("align", msg.lower())


class TestNoFinishedRequestInLists(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_only_a_releasing_list_holds_finished_requests(self):
        live = SimpleNamespace(rid="live", finished=lambda: False)
        done = SimpleNamespace(rid="done", finished=lambda: True)
        check = SchedulerInvariantChecker.check_no_finished_reqs

        check({"releasing": RequestList(name="releasing", reqs=[done], releasing=True)})
        check({"decoding": RequestList(name="decoding", reqs=[live])})
        with self.assertRaisesRegex(AssertionError, "'decoding' list: \\['done'\\]"):
            check({"decoding": RequestList(name="decoding", reqs=[live, done])})


if __name__ == "__main__":
    unittest.main()
