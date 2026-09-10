"""Unit tests for scheduler pool statistics - no server or model loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1.0, suite="base-a-test-cpu")


def _batch(*pool_indices):
    return SimpleNamespace(
        is_empty=lambda: not pool_indices,
        reqs=[SimpleNamespace(req_pool_idx=index) for index in pool_indices],
    )


def _make_observer(**overrides):
    defaults = {
        "tree_cache": MagicMock(),
        "token_to_kv_pool_allocator": MagicMock(),
        "req_to_token_pool": MagicMock(),
        "session_controller": SimpleNamespace(sessions={}),
        "hisparse_coordinator": None,
        "is_hybrid_swa": False,
        "is_hybrid_ssm": False,
        "enable_hisparse": False,
        "full_tokens_per_layer": 100,
        "swa_tokens_per_layer": 80,
        "max_total_num_tokens": 100,
        "get_last_batch": lambda: None,
        "get_running_batch": lambda: None,
    }
    defaults.update(overrides)
    return SchedulerPoolStatsObserver(**defaults)


class TestPoolStats(CustomTestCase):
    def test_standard_pool_accessors_and_messages(self):
        stats = PoolStats(
            full_num_used=30,
            full_token_usage=0.3,
            full_available_size=60,
            full_evictable_size=10,
        )

        self.assertEqual(stats.get_kv_token_stats(), (30, 0.3))
        self.assertEqual(stats.get_max_pool_usage(), 0.3)
        self.assertEqual(stats.get_prefill_usage_msg_parts(), ["token usage: 0.30"])
        self.assertEqual(
            stats.get_decode_usage_msg_parts(),
            ["#token: 30, token usage: 0.30"],
        )

    def test_hybrid_swa_uses_larger_pool_usage(self):
        stats = PoolStats(
            full_num_used=30,
            full_token_usage=0.3,
            full_available_size=60,
            full_evictable_size=10,
            is_hybrid_swa=True,
            swa_num_used=40,
            swa_token_usage=0.5,
            swa_available_size=30,
            swa_evictable_size=10,
        )

        self.assertEqual(stats.get_kv_token_stats(), (40, 0.5))
        self.assertEqual(stats.get_max_pool_usage(), 0.5)
        self.assertEqual(
            stats.get_prefill_usage_msg_parts(),
            ["full token usage: 0.30", "swa token usage: 0.50"],
        )
        self.assertEqual(
            stats.get_decode_usage_msg_parts(),
            [
                "#full token: 30",
                "full token usage: 0.30",
                "#swa token: 40",
                "swa token usage: 0.50",
            ],
        )

    def test_hybrid_ssm_messages_and_max_usage(self):
        stats = PoolStats(
            full_num_used=30,
            full_token_usage=0.3,
            full_available_size=60,
            full_evictable_size=10,
            is_hybrid_ssm=True,
            mamba_num_used=8,
            mamba_usage=0.8,
            mamba_available_size=2,
            mamba_evictable_size=0,
        )

        self.assertEqual(stats.get_kv_token_stats(), (30, 0.3))
        self.assertEqual(stats.get_max_pool_usage(), 0.8)
        self.assertEqual(
            stats.get_prefill_usage_msg_parts(),
            ["full token usage: 0.30", "mamba usage: 0.80"],
        )
        self.assertEqual(
            stats.get_decode_usage_msg_parts(),
            [
                "#full token: 30",
                "full token usage: 0.30",
                "mamba num: 8",
                "mamba usage: 0.80",
            ],
        )

    def test_combined_swa_ssm_messages_do_not_repeat_full_pool(self):
        stats = PoolStats(
            full_num_used=30,
            full_token_usage=0.3,
            full_available_size=60,
            full_evictable_size=10,
            is_hybrid_swa=True,
            swa_num_used=40,
            swa_token_usage=0.5,
            swa_available_size=30,
            swa_evictable_size=10,
            is_hybrid_ssm=True,
            mamba_num_used=8,
            mamba_usage=0.8,
            mamba_available_size=2,
            mamba_evictable_size=0,
        )

        self.assertEqual(
            stats.get_prefill_usage_msg_parts(),
            [
                "full token usage: 0.30",
                "swa token usage: 0.50",
                "mamba usage: 0.80",
            ],
        )
        self.assertEqual(stats.get_decode_usage_msg_parts().count("#full token: 30"), 1)

    def test_hisparse_decode_message(self):
        stats = PoolStats(
            full_num_used=30,
            full_token_usage=0.3,
            full_available_size=60,
            full_evictable_size=10,
            is_hisparse=True,
            hisparse_device_tokens=20,
            hisparse_device_token_usage=0.2,
            hisparse_host_tokens=15,
            hisparse_host_token_usage=0.15,
        )

        self.assertEqual(
            stats.get_decode_usage_msg_parts(),
            [
                "#gpu token: 20",
                "gpu token usage: 0.20",
                "#cpu token: 15",
                "cpu token usage: 0.15",
            ],
        )

    def test_update_scheduler_stats(self):
        pool_stats = PoolStats(
            full_num_used=30,
            full_token_usage=0.333,
            full_available_size=60,
            full_evictable_size=10,
            is_hybrid_swa=True,
            swa_num_used=40,
            swa_token_usage=0.555,
            swa_available_size=30,
            swa_evictable_size=10,
            is_hybrid_ssm=True,
            mamba_num_used=8,
            mamba_usage=0.777,
            mamba_available_size=2,
            mamba_evictable_size=0,
        )
        scheduler_stats = SimpleNamespace()

        pool_stats.update_scheduler_stats(scheduler_stats)

        self.assertEqual(scheduler_stats.num_used_tokens, 40)
        self.assertEqual(scheduler_stats.token_usage, 0.78)
        self.assertEqual(scheduler_stats.full_token_usage, 0.333)
        self.assertEqual(scheduler_stats.swa_used_tokens, 40)
        self.assertEqual(scheduler_stats.mamba_used_tokens, 8)
        self.assertEqual(scheduler_stats.kv_available_tokens, 60)
        self.assertEqual(scheduler_stats.kv_evictable_tokens, 10)
        self.assertEqual(scheduler_stats.kv_used_tokens, 30)

    def test_negative_usage_rejected(self):
        stats = PoolStats(
            full_num_used=-1,
            full_token_usage=-0.1,
            full_available_size=101,
            full_evictable_size=0,
        )
        with self.assertRaisesRegex(AssertionError, "not valid"):
            stats.get_max_pool_usage()


class TestSchedulerPoolStatsObserver(CustomTestCase):
    def test_session_and_active_pool_counts(self):
        sessions = {
            "streaming": SimpleNamespace(streaming=True),
            "idle": SimpleNamespace(streaming=False),
        }
        observer = _make_observer(
            session_controller=SimpleNamespace(sessions=sessions),
            get_last_batch=lambda: _batch(1, None, 2),
            get_running_batch=lambda: _batch(2, 3),
        )

        self.assertEqual(observer.streaming_session_count(), 1)
        self.assertEqual(observer.active_pool_idxs(), {1, 2, 3})

    def test_active_pool_idxs_skips_empty_batches(self):
        observer = _make_observer(
            get_last_batch=lambda: None,
            get_running_batch=lambda: _batch(),
        )
        self.assertEqual(observer.active_pool_idxs(), set())

    def test_session_held_accessors_delegate_to_tree_cache(self):
        tree_cache = MagicMock()
        tree_cache.session_held_tokens.return_value = 11
        tree_cache.session_held_full_tokens.return_value = 12
        tree_cache.session_held_swa_tokens.return_value = 13
        tree_cache.session_held_req_count.return_value = 2
        tree_cache.session_held_mamba_slots.return_value = 3
        observer = _make_observer(
            tree_cache=tree_cache,
            get_last_batch=lambda: _batch(4),
        )

        self.assertEqual(observer.session_held_tokens(), 11)
        self.assertEqual(observer.session_held_full_tokens(), 12)
        self.assertEqual(observer.session_held_swa_tokens(), 13)
        self.assertEqual(observer.session_held_req_count(), 2)
        self.assertEqual(observer.session_held_mamba_slots(), 3)
        tree_cache.session_held_tokens.assert_called_once_with({4})
        tree_cache.session_held_full_tokens.assert_called_once_with({4})
        tree_cache.session_held_swa_tokens.assert_called_once_with({4})
        tree_cache.session_held_req_count.assert_called_once_with()
        tree_cache.session_held_mamba_slots.assert_called_once_with({4})

    def test_standard_pool_stats(self):
        allocator = MagicMock()
        allocator.available_size.return_value = 60
        tree_cache = MagicMock()
        tree_cache.evictable_size.return_value = 10
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            max_total_num_tokens=100,
        )

        stats = observer.get_pool_stats()

        self.assertEqual(stats.full_num_used, 30)
        self.assertEqual(stats.full_token_usage, 0.3)
        self.assertEqual(stats.full_available_size, 60)
        self.assertEqual(stats.full_evictable_size, 10)

    def test_hisparse_stats_overlay(self):
        allocator = MagicMock()
        allocator.available_size.return_value = 60
        tree_cache = MagicMock()
        tree_cache.evictable_size.return_value = 10
        coordinator = MagicMock()
        coordinator.get_token_stats.return_value = SimpleNamespace(
            device_tokens=20,
            device_token_usage=0.2,
            host_tokens=15,
            host_token_usage=0.15,
        )
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            enable_hisparse=True,
            hisparse_coordinator=coordinator,
        )

        stats = observer.get_pool_stats()

        self.assertTrue(stats.is_hisparse)
        self.assertEqual(stats.hisparse_device_tokens, 20)
        self.assertEqual(stats.hisparse_host_tokens, 15)

    def test_hisparse_without_coordinator_keeps_base_stats(self):
        allocator = MagicMock()
        allocator.available_size.return_value = 60
        tree_cache = MagicMock()
        tree_cache.evictable_size.return_value = 10
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            enable_hisparse=True,
            hisparse_coordinator=None,
        )

        stats = observer.get_pool_stats()

        self.assertFalse(stats.is_hisparse)

    def test_swa_pool_stats(self):
        allocator = MagicMock()
        allocator.full_available_size.return_value = 50
        allocator.swa_available_size.return_value = 30
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = 10
        tree_cache.swa_evictable_size.return_value = 10
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            is_hybrid_swa=True,
            full_tokens_per_layer=100,
            swa_tokens_per_layer=80,
        )

        stats = observer.get_pool_stats()

        self.assertTrue(stats.is_hybrid_swa)
        self.assertEqual(stats.full_num_used, 40)
        self.assertEqual(stats.full_token_usage, 0.4)
        self.assertEqual(stats.swa_num_used, 40)
        self.assertEqual(stats.swa_token_usage, 0.5)

    def test_swa_hisparse_clamps_negative_usage(self):
        allocator = MagicMock()
        allocator.full_available_size.return_value = 120
        allocator.swa_available_size.return_value = 90
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = 10
        tree_cache.swa_evictable_size.return_value = 10
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            is_hybrid_swa=True,
            enable_hisparse=True,
            full_tokens_per_layer=100,
            swa_tokens_per_layer=80,
        )

        stats = observer.get_pool_stats()

        self.assertEqual(stats.full_num_used, 0)
        self.assertEqual(stats.full_token_usage, 0.0)
        self.assertEqual(stats.swa_num_used, 0)
        self.assertEqual(stats.swa_token_usage, 0.0)

    def test_swa_with_no_full_pool(self):
        allocator = MagicMock()
        allocator.full_available_size.return_value = 0
        allocator.swa_available_size.return_value = 40
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = 0
        tree_cache.swa_evictable_size.return_value = 0
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            is_hybrid_swa=True,
            full_tokens_per_layer=0,
            swa_tokens_per_layer=80,
        )

        stats = observer.get_pool_stats()

        self.assertEqual(stats.full_num_used, 0)
        self.assertEqual(stats.full_available_size, 0)
        self.assertEqual(stats.full_token_usage, 0.0)
        self.assertEqual(stats.swa_token_usage, 0.5)

    def test_mamba_pool_stats_with_radix_cache(self):
        allocator = MagicMock(size=100)
        allocator.available_size.return_value = 60
        tree_cache = MagicMock()
        tree_cache.supports_mamba.return_value = True
        tree_cache.is_tree_cache.return_value = True
        tree_cache.full_evictable_size.return_value = 10
        tree_cache.mamba_evictable_size.return_value = 2
        mamba_allocator = MagicMock()
        mamba_allocator.available_size.return_value = 3
        req_pool = SimpleNamespace(
            mamba_allocator=mamba_allocator,
            mamba_pool=SimpleNamespace(size=10),
            mamba_ckpt_pool=None,
        )
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            req_to_token_pool=req_pool,
            is_hybrid_ssm=True,
        )

        stats = observer.get_pool_stats()

        self.assertEqual(stats.full_num_used, 30)
        self.assertEqual(stats.full_token_usage, 0.3)
        self.assertEqual(stats.mamba_num_used, 5)
        self.assertEqual(stats.mamba_usage, 0.5)
        self.assertEqual(stats.mamba_evictable_size, 2)

    def test_mamba_int8_checkpoint_excludes_cached_slots(self):
        allocator = MagicMock(size=100)
        allocator.available_size.return_value = 60
        tree_cache = MagicMock()
        tree_cache.supports_mamba.return_value = True
        tree_cache.is_tree_cache.return_value = True
        tree_cache.full_evictable_size.return_value = 10
        tree_cache.mamba_evictable_size.return_value = 4
        mamba_allocator = MagicMock()
        mamba_allocator.available_size.return_value = 3
        req_pool = SimpleNamespace(
            mamba_allocator=mamba_allocator,
            mamba_pool=SimpleNamespace(size=10),
            mamba_ckpt_pool=object(),
        )
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            req_to_token_pool=req_pool,
            is_hybrid_ssm=True,
        )

        stats = observer.get_pool_stats()

        self.assertEqual(stats.mamba_num_used, 7)
        self.assertEqual(stats.mamba_evictable_size, 0)
        tree_cache.mamba_evictable_size.assert_not_called()

    def test_combined_swa_and_ssm_pool_stats(self):
        allocator = MagicMock(size=100)
        allocator.full_available_size.return_value = 50
        allocator.swa_available_size.return_value = 30
        allocator.available_size.return_value = 60
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = 10
        tree_cache.swa_evictable_size.return_value = 10
        tree_cache.supports_mamba.return_value = True
        tree_cache.is_tree_cache.return_value = True
        tree_cache.mamba_evictable_size.return_value = 2
        mamba_allocator = MagicMock()
        mamba_allocator.available_size.return_value = 3
        req_pool = SimpleNamespace(
            mamba_allocator=mamba_allocator,
            mamba_pool=SimpleNamespace(size=10),
            mamba_ckpt_pool=None,
        )
        observer = _make_observer(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
            req_to_token_pool=req_pool,
            is_hybrid_swa=True,
            is_hybrid_ssm=True,
        )

        stats = observer.get_pool_stats()

        self.assertTrue(stats.is_hybrid_swa)
        self.assertTrue(stats.is_hybrid_ssm)
        self.assertEqual(stats.swa_num_used, 40)
        self.assertEqual(stats.mamba_num_used, 5)


if __name__ == "__main__":
    unittest.main()
