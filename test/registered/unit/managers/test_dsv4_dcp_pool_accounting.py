import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator.swa import (
    DeepSeekV4DCPTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV4DCPPoolAccounting(CustomTestCase):
    def setUp(self):
        super().setUp()
        pool = MagicMock(spec=BaseSWAKVPool)
        pool.supports_dsv4_dcp = True
        pool.full_kv_pool = None
        pool.swa_kv_pool = None
        self.allocator = DeepSeekV4DCPTokenToKVPoolAllocator(
            physical_size_full=262144,
            physical_size_swa=39168,
            physical_page_size=256,
            dcp_size=8,
            dcp_rank=0,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        self.tree_cache = MagicMock()
        self.tree_cache.full_evictable_size.return_value = 0
        self.tree_cache.swa_evictable_size.return_value = 0
        self.tree_cache.full_protected_size.return_value = 0
        self.tree_cache.swa_protected_size.return_value = 0
        self.tree_cache.session_held_full_tokens.return_value = 0
        self.tree_cache.session_held_swa_tokens.return_value = 0
        self.observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=MagicMock(),
            session_controller=SimpleNamespace(sessions={}),
            hisparse_coordinator=None,
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=self.allocator.size_full,
            swa_tokens_per_layer=self.allocator.size_swa,
            max_total_num_tokens=self.allocator.size_full,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )

    def test_empty_pool_reports_zero_usage(self):
        stats = self.observer.get_pool_stats()
        self.assertEqual(stats.full_num_used, 0)
        self.assertEqual(stats.swa_num_used, 0)
        self.assertEqual(stats.full_token_usage, 0.0)
        self.assertEqual(stats.swa_token_usage, 0.0)

    def test_partially_used_pool_reports_logical_usage(self):
        self.allocator.full_attn_allocator.alloc(2 * self.allocator.page_size)
        self.allocator.swa_attn_allocator.alloc(self.allocator.page_size)

        stats = self.observer.get_pool_stats()
        self.assertEqual(stats.full_num_used, 2 * self.allocator.page_size)
        self.assertEqual(stats.swa_num_used, self.allocator.page_size)
        self.assertGreater(stats.get_max_pool_usage(), 0.0)

    def test_empty_pool_passes_idle_invariants(self):
        checker = self._make_checker()

        has_leak, messages = checker._check_all_pools(self.observer.get_pool_stats())
        self.assertFalse(has_leak, messages)

    def _make_checker(self):
        return SchedulerInvariantChecker(
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=self.allocator.page_size,
            full_tokens_per_layer=self.allocator.size_full,
            swa_tokens_per_layer=self.allocator.size_swa,
            max_total_num_tokens=self.allocator.size_full,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=MagicMock(),
            pool_stats_observer=self.observer,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )

    def test_dcp_full_invariant_allows_only_subpage_slack(self):
        checker = self._make_checker()
        stats = self.observer.get_pool_stats()

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler_components.invariant_checker.get_parallel",
            return_value=SimpleNamespace(dcp_enabled=True),
        ):
            subpage_stats = dataclasses.replace(
                stats,
                full_available_size=stats.full_available_size - 1,
            )
            subpage_leak, _ = checker._check_full_pool(subpage_stats)
            self.assertFalse(subpage_leak)

            whole_page_stats = dataclasses.replace(
                stats,
                full_available_size=(
                    stats.full_available_size - self.allocator.page_size
                ),
            )
            whole_page_leak, _ = checker._check_full_pool(whole_page_stats)
            self.assertTrue(whole_page_leak)

    def test_scheduler_uses_allocator_logical_geometry(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = self.allocator
        scheduler.is_hybrid_swa = True
        scheduler.is_hybrid_ssm = False
        scheduler.enable_hisparse = False
        scheduler.full_tokens_per_layer = 262144
        scheduler.swa_tokens_per_layer = 39168
        scheduler.max_total_num_tokens = 262144
        scheduler.tree_cache = self.tree_cache
        scheduler.req_to_token_pool = MagicMock()
        scheduler.session_controller = SimpleNamespace(sessions={})
        scheduler.hisparse_coordinator = None
        scheduler.last_batch = None
        scheduler.running_batch = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL

        capacities = scheduler.get_token_pool_capacities()
        scheduler.init_pool_stats_observer()
        scheduler.init_invariant_checker()

        self.assertEqual(
            capacities,
            (self.allocator.size_full, self.allocator.size_swa),
        )
        self.assertEqual(
            scheduler.pool_stats_observer.full_tokens_per_layer,
            self.allocator.size_full,
        )
        self.assertEqual(
            scheduler.pool_stats_observer.swa_tokens_per_layer,
            self.allocator.size_swa,
        )
        self.assertEqual(
            scheduler.invariant_checker.page_size,
            self.allocator.page_size,
        )

    def test_metrics_report_logical_capacity(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = self.allocator
        scheduler.is_hybrid_swa = True
        scheduler.max_total_num_tokens = 262144
        scheduler.full_tokens_per_layer = 262144
        scheduler.swa_tokens_per_layer = 39168
        scheduler.metrics_collector = MagicMock()
        scheduler.tp_worker = MagicMock()
        scheduler.tp_worker.model_runner.weight_load_mem_usage = 1.0
        scheduler.tp_worker.graph_memory_usage = {}
        scheduler.draft_worker = None
        scheduler.model_config = SimpleNamespace(context_len=1048576)
        scheduler.startup_available_gpu_memory_gb = 100.0
        scheduler.page_size = 256
        self.allocator.get_kvcache().mem_usage = 2.0

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.get_observability"
        ) as get_observability:
            get_observability.return_value.enable_metrics = True
            scheduler.emit_metrics_constants()

        kwargs = scheduler.metrics_collector.emit_constants.call_args.kwargs
        self.assertEqual(kwargs["max_total_num_tokens"], self.allocator.size_full)
        self.assertEqual(kwargs["max_total_num_tokens_swa"], self.allocator.size_swa)
        self.assertEqual(
            kwargs["num_pages"],
            self.allocator.size_full // self.allocator.page_size,
        )
        self.assertEqual(kwargs["page_size"], self.allocator.page_size)

    def test_pure_swa_preserves_zero_full_capacity(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.is_hybrid_swa = True
        scheduler.full_tokens_per_layer = 0
        scheduler.swa_tokens_per_layer = 4096
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            size_full=4096,
            size_swa=4096,
        )

        self.assertEqual(scheduler.get_token_pool_capacities(), (0, 4096))

    def test_unified_swa_observer_uses_conservation_available(self):
        allocator = MagicMock()
        allocator._conserve_full_available_size.return_value = 80
        allocator._conserve_swa_available_size.return_value = 60
        allocator.full_available_size.return_value = 50
        allocator.swa_available_size.return_value = 40
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = 20
        tree_cache.swa_evictable_size.return_value = 40
        observer = SchedulerPoolStatsObserver(
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=MagicMock(),
            session_controller=SimpleNamespace(sessions={}),
            hisparse_coordinator=None,
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=100,
            swa_tokens_per_layer=100,
            max_total_num_tokens=100,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )

        stats = observer.get_pool_stats()
        self.assertEqual(stats.full_num_used, 0)
        self.assertEqual(stats.swa_num_used, 0)


if __name__ == "__main__":
    unittest.main()
