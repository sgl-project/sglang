import os
import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components import invariant_checker
from sglang.srt.managers.scheduler_components.invariant_checker import (
    CheckPoint,
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
    kv_private_swa_tokens,
)
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage, StaticPoolShape
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.session.streaming_session import SessionSlot
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestCheckTreeCacheGate(CustomTestCase):
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
        pool_stats_observer = MagicMock()
        pool_stats_observer.get_pool_stats.return_value = PoolStats(
            full_num_used=0,
            full_token_usage=0.0,
            full_available_size=1024,
            full_evictable_size=0,
            is_hybrid_swa=hybrid_swa,
            is_hybrid_ssm=hybrid_ssm,
        )
        return SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=MagicMock(),
            req_to_token_pool=MagicMock(),
            pool_stats_observer=pool_stats_observer,
            get_last_batch=lambda: None,
            get_skip_idle_pool_checks=lambda: True,
            scheduler_stage_metrics=None,
        )

    def test_disabled_by_default(self):
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker.run(CheckPoint.IDLE)

            checker.tree_cache.sanity_check.assert_not_called()

    def test_enabled_by_default_in_ci(self):
        with (
            envs.SGLANG_IS_IN_CI.override(True),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker.run(CheckPoint.IDLE)

            checker.tree_cache.sanity_check.assert_called_once()

    def test_explicitly_disabled_in_ci(self):
        with envs.SGLANG_IS_IN_CI.override(True):
            checker = self._make_checker()

            with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(False):
                checker.run(CheckPoint.IDLE)

            checker.tree_cache.sanity_check.assert_not_called()

    def test_runs_when_enabled(self):
        with envs.SGLANG_IS_IN_CI.override(False):
            checker = self._make_checker()

            with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(True):
                checker.run(CheckPoint.IDLE)

            checker.tree_cache.sanity_check.assert_called_once()

    def test_skipped_without_prefix_sharing(self):
        with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(True):
            checker = self._make_checker(
                prefix_sharing=False, hybrid_swa=False, hybrid_ssm=True
            )

            checker.run(CheckPoint.IDLE)

            checker.tree_cache.sanity_check.assert_not_called()


class TestStateCheckLevels(CustomTestCase):
    OFF, WARN, STRICT = (
        InvariantCheckLevel.OFF,
        InvariantCheckLevel.WARN,
        InvariantCheckLevel.STRICT,
    )

    def _levels(self, in_ci, **flags):
        with patch.dict(os.environ, {}, clear=False):
            for env in (
                envs.SGLANG_INVARIANT_CHECK,
                envs.SGLANG_ENABLE_ASYNC_ASSERT,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE,
                envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK,
                envs.SGLANG_ENABLE_RUNNING_REQS_CHECK,
                envs.SGLANG_CHECK_KV_PAGE_INVARIANTS,
            ):
                env.clear()
            envs.SGLANG_IS_IN_CI.set(in_ci)
            for name, value in flags.items():
                getattr(envs, name).set(value)
            checker = SchedulerInvariantChecker(
                disaggregation_mode=DisaggregationMode.NULL,
                page_size=1,
                tree_cache=None,
                token_to_kv_pool_allocator=None,
                req_to_token_pool=None,
                pool_stats_observer=None,
                get_last_batch=lambda: None,
                scheduler_stage_metrics=None,
            )
            return {
                check.name: check.level()
                for checks in checker.checks.values()
                for check in checks
            }

    def test_legacy_flags_decide_the_level(self):
        """With SGLANG_INVARIANT_CHECK unset, every registered check keeps the
        level its own flags gave it, in prod and in CI."""
        OFF, WARN, STRICT = self.OFF, self.WARN, self.STRICT
        prod = dict(
            request_lists=OFF,
            batch_consistency=OFF,
            busy_pools=OFF,
            kv_pages=OFF,
            idle_pools=STRICT,
            req_pool=STRICT,
            pool_bytes=STRICT,
            tree_cache=OFF,
        )
        ci = {
            **prod,
            "request_lists": STRICT,
            "batch_consistency": STRICT,
            "tree_cache": STRICT,
        }
        cases = [
            (False, {}, prod),
            (True, {}, ci),
            (
                True,
                {
                    "SGLANG_ENABLE_RUNNING_REQS_CHECK": False,
                    "SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK": False,
                },
                prod,
            ),
            (False, {"SGLANG_CHECK_KV_PAGE_INVARIANTS": True}, prod),
            (
                False,
                {"SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": 2},
                {**prod, "busy_pools": STRICT},
            ),
            (
                False,
                {
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": 1,
                    "SGLANG_CHECK_KV_PAGE_INVARIANTS": True,
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": False,
                },
                {
                    **prod,
                    "busy_pools": STRICT,
                    "kv_pages": WARN,
                    "idle_pools": WARN,
                    "req_pool": WARN,
                    "pool_bytes": WARN,
                },
            ),
        ]
        for in_ci, flags, expected in cases:
            with self.subTest(in_ci=in_ci, **flags):
                self.assertEqual(self._levels(in_ci, **flags), expected)

    def test_explicit_level_applies_unless_the_check_flag_is_set(self):
        OFF, WARN, STRICT = self.OFF, self.WARN, self.STRICT
        names = self._levels(False).keys()
        for level in (OFF, WARN, STRICT):
            with self.subTest(level=level):
                self.assertEqual(
                    self._levels(False, SGLANG_INVARIANT_CHECK=int(level)),
                    dict.fromkeys(names, level),
                )
        # An explicit check flag beats the explicit level, for its checks only.
        self.assertEqual(
            self._levels(
                True,
                SGLANG_INVARIANT_CHECK=int(OFF),
                SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=False,
                SGLANG_ENABLE_RUNNING_REQS_CHECK=True,
            ),
            dict(
                request_lists=STRICT,
                batch_consistency=STRICT,
                busy_pools=OFF,
                kv_pages=OFF,
                idle_pools=WARN,
                req_pool=WARN,
                pool_bytes=WARN,
                tree_cache=OFF,
            ),
        )
        # The SGLANG_ENABLE_ASYNC_ASSERT bridge is not an explicit level.
        self.assertEqual(
            self._levels(False, SGLANG_ENABLE_ASYNC_ASSERT=True),
            self._levels(False),
        )


class TestPrivateSwaTokens(CustomTestCase):
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
                    tree_slots=tree_slots_of((ComponentType.FULL,)),
                )
            )
        self.last_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
        self.running_batch = self.last_batch
        self.chunked_req = None
        observer = SchedulerPoolStatsObserver(
            tree_cache=self.cache,
            storage=StaticKVStorage(
                req_pool,
                self.allocator,
                shape=StaticPoolShape(
                    has_window=False,
                    has_state=False,
                    token_capacity=self.allocator.size,
                ),
            ),
            token_to_kv_pool_allocator=self.allocator,
            session_controller=None,
            hisparse_coordinator=None,
            enable_hisparse=False,
        )
        return SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.PREFILL,
            page_size=self.PAGE_SIZE,
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=req_pool,
            pool_stats_observer=observer,
            get_last_batch=lambda: self.last_batch,
            get_request_lists=lambda: {
                ListKind.CHUNKED: (
                    [] if self.chunked_req is None else [self.chunked_req]
                ),
                ListKind.DECODING: self.running_batch.reqs,
            },
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
        checker.req_to_token_pool.req_to_token[0, : len(locs)] = locs
        req = SimpleNamespace(
            kv=SimpleNamespace(
                holds_kv=True,
                req_pool_idx=0,
                kv_allocated_len=len(locs),
                cache_protected_len=self.PAGE_SIZE,
            ),
            beam_group=None,
        )
        self.last_batch = SimpleNamespace(reqs=[req], is_empty=lambda: False)
        self.running_batch = self.last_batch
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
        self.assertEqual(
            checker._get_total_uncached_sizes(has_window=False), (self.PAGE_SIZE, 0)
        )
        with (
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
            self.assertRaisesRegex(AssertionError, "Full Pool Mem Leak Detected"),
        ):
            checker.run(CheckPoint.BUSY)

    def test_partial_active_page_is_accounted_without_slack_exemption(self):
        checker = self._make_checker()
        req = self._active_partial_page(checker)
        # Count the same request only once, including when it is parked
        # between prefill chunks instead of appearing in either batch.
        for parked in (False, True):
            with self.subTest(parked=parked):
                self.chunked_req = req
                if parked:
                    self.last_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
                    self.running_batch = self.last_batch
                uncached, swa_uncached = checker._get_total_uncached_sizes(
                    has_window=False
                )
                self.assertEqual((uncached, swa_uncached), (self.PAGE_SIZE, 0))
                stats = checker.pool_stats_observer.get_pool_stats()
                leak, msg = checker._check_full_pool(stats, uncached=uncached)
                self.assertFalse(leak, msg)
                self.assertNotIn("slack_allowed", msg)
                with (
                    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
                    envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False),
                ):
                    checker.run(CheckPoint.BUSY)

    def test_session_record_is_counted_once_by_its_owner(self):
        checker = self._make_checker()
        req = self._active_partial_page(checker)
        # The session owns the row: session-held whether the turn is in a batch
        # or parked between prefill chunks, never also uncached.
        self.cache.session.slots["s"] = SessionSlot(kv=req.kv)
        for parked in (False, True):
            with self.subTest(parked=parked):
                self.chunked_req = req
                if parked:
                    self.last_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
                    self.running_batch = self.last_batch
                self.assertEqual(
                    checker._get_total_uncached_sizes(has_window=False), (0, 0)
                )
                self.assertEqual(
                    checker.pool_stats_observer.session_held_tokens(), self.PAGE_SIZE
                )
                with (
                    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
                    envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False),
                ):
                    checker.run(CheckPoint.BUSY)

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


if __name__ == "__main__":
    unittest.main()
