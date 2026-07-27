import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.p2p_allocator_observer import P2PAllocatorObserver
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _SizedAllocator:
    def __init__(self, size, available):
        self.size = size
        self._available = available

    def available_size(self):
        return self._available


class TestP2PAllocatorObserver(unittest.TestCase):
    def _observer(self, full_protected=25):
        clock = _Clock()
        logger = MagicMock()
        tree_cache = SimpleNamespace(
            full_protected_size=lambda: full_protected,
            mamba_protected_size=lambda: 3,
        )
        token_allocator = SimpleNamespace(size=100)
        mamba_allocator = _SizedAllocator(size=16, available=7)
        pool_stats = PoolStats(
            full_num_used=60,
            full_token_usage=0.6,
            full_available_size=40,
            full_evictable_size=20,
            is_hybrid_ssm=True,
            mamba_num_used=9,
            mamba_usage=9 / 16,
            mamba_available_size=7,
            mamba_evictable_size=2,
        )
        pool_stats_observer = SimpleNamespace(
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=token_allocator,
            get_pool_stats=lambda: pool_stats,
            session_held_full_tokens=lambda: 5,
            session_held_tokens=lambda: 5,
            session_held_mamba_slots=lambda: 1,
        )
        invariant_checker = SimpleNamespace(_get_total_uncached_sizes=lambda: (10, 0))
        req_to_token_pool = SimpleNamespace(mamba_allocator=mamba_allocator)
        ps = SimpleNamespace(attn_tp_rank=1, pp_rank=2, dp_rank=3)
        observer = P2PAllocatorObserver(
            pool_stats_observer=pool_stats_observer,
            invariant_checker=invariant_checker,
            req_to_token_pool=req_to_token_pool,
            ps=ps,
            interval_s=30.0,
            clock=clock,
            logger=logger,
        )
        return observer, clock, logger

    def test_snapshot_contains_full_and_mamba_accounting(self):
        observer, _, logger = self._observer()

        snapshot = observer.maybe_log()

        self.assertTrue(snapshot["accounting_ok"])
        self.assertEqual(snapshot["full_total"], 100)
        self.assertEqual(snapshot["full_available"], 40)
        self.assertEqual(snapshot["full_evictable"], 20)
        self.assertEqual(snapshot["full_protected"], 25)
        self.assertEqual(snapshot["full_session_held"], 5)
        self.assertEqual(snapshot["full_uncached"], 10)
        self.assertEqual(snapshot["full_token_usage"], 0.6)
        self.assertTrue(snapshot["accounting_complete"])
        self.assertEqual(snapshot["full_unobserved_allocated"], 0)
        self.assertEqual(snapshot["full_overaccounted"], 0)
        self.assertEqual(snapshot["mamba_total"], 16)
        self.assertEqual(snapshot["mamba_available"], 7)
        self.assertEqual(snapshot["mamba_used"], 9)
        self.assertEqual(snapshot["mamba_evictable"], 2)
        self.assertEqual(snapshot["mamba_protected"], 3)
        self.assertEqual(snapshot["mamba_session_held"], 1)
        self.assertEqual(snapshot["tp_rank"], 1)
        self.assertEqual(snapshot["pp_rank"], 2)
        self.assertEqual(snapshot["dp_rank"], 3)
        args, _ = logger.info.call_args
        self.assertEqual(args[0], "p2p_allocator_snapshot %s")
        self.assertEqual(json.loads(args[1]), snapshot)

    def test_interval_and_over_accounting(self):
        observer, clock, logger = self._observer(full_protected=30)

        first = observer.maybe_log()
        clock.now = 29.9
        skipped = observer.maybe_log()
        clock.now = 30.0
        second = observer.maybe_log()

        self.assertFalse(first["accounting_ok"])
        self.assertFalse(first["accounting_complete"])
        self.assertEqual(first["full_accounting_delta"], -5)
        self.assertEqual(first["full_overaccounted"], 5)
        self.assertIsNone(skipped)
        self.assertIsNotNone(second)
        self.assertEqual(logger.info.call_count, 2)

    def test_unobserved_allocations_are_incomplete_not_corrupt(self):
        observer, _, _ = self._observer(full_protected=20)

        snapshot = observer.maybe_log()

        self.assertTrue(snapshot["accounting_ok"])
        self.assertFalse(snapshot["accounting_complete"])
        self.assertEqual(snapshot["full_accounting_delta"], 5)
        self.assertEqual(snapshot["full_unobserved_allocated"], 5)
        self.assertEqual(snapshot["full_overaccounted"], 0)

    def test_idle_without_a_last_batch_reports_zero_uncached_without_warning(self):
        observer, _, logger = self._observer(full_protected=35)
        observer.invariant_checker.get_last_batch = MagicMock(return_value=None)
        observer.invariant_checker._get_total_uncached_sizes = MagicMock(
            side_effect=AssertionError("must not inspect an idle batch")
        )

        snapshot = observer.maybe_log()

        self.assertEqual(snapshot["full_uncached"], 0)
        self.assertTrue(snapshot["uncached_observation_ok"])
        self.assertTrue(snapshot["accounting_ok"])
        observer.invariant_checker._get_total_uncached_sizes.assert_not_called()
        logger.warning.assert_not_called()

    def test_uncached_observation_failure_is_non_fatal(self):
        observer, _, logger = self._observer(full_protected=35)
        observer.invariant_checker._get_total_uncached_sizes = MagicMock(
            side_effect=AttributeError("idle batch is None")
        )

        snapshot = observer.maybe_log()

        self.assertEqual(snapshot["full_uncached"], 0)
        self.assertFalse(snapshot["uncached_observation_ok"])
        self.assertTrue(snapshot["accounting_ok"])
        logger.warning.assert_called_once()

    def test_arbitrary_observation_failure_never_escapes(self):
        observer, _, logger = self._observer()
        observer.pool_stats_observer.get_pool_stats = MagicMock(
            side_effect=RuntimeError("observer failure")
        )

        snapshot = observer.maybe_log()

        self.assertIsNone(snapshot)
        logger.exception.assert_called_once()

    def test_scheduler_calls_observer_from_active_and_idle_paths(self):
        active_source = inspect.getsource(Scheduler.process_batch_result)
        idle_source = inspect.getsource(Scheduler.on_idle)

        hook = "self.p2p_allocator_observer.maybe_log()"
        self.assertIn(hook, active_source)
        self.assertIn(hook, idle_source)


if __name__ == "__main__":
    unittest.main()
