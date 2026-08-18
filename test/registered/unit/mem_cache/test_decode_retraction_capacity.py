"""Capacity gating for the inferred PD decode retraction backend.

``host_pool`` mirrors the decode device KV pool in host RAM, so the inferred
default has to check that the mirror fits before committing to it. These cases
cover the sizing arithmetic and the invariant that deciding host memory is
insufficient never enlarges the resulting host pool.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.kv_cache_builder import (
    _host_pool_retraction_fits,
    _local_scheduler_count,
)
from sglang.srt.mem_cache.pool_host.base import HICACHE_HOST_MEMORY_RESERVE_BYTES
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1024**3


def _fake_pool(total_bytes: int) -> SimpleNamespace:
    """A stand-in exposing only what the estimate reads."""
    return SimpleNamespace(
        get_kv_size_bytes=lambda: (total_bytes // 2, total_bytes // 2)
    )


class TestLocalSchedulerCount(CustomTestCase):
    def _count(self, **fields) -> int:
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)
        return _local_scheduler_count()

    def test_single_node_tp_only(self):
        self.assertEqual(self._count(tp_size=4, nnodes=1), 4)

    def test_tp_split_across_nodes(self):
        self.assertEqual(self._count(tp_size=8, nnodes=2), 4)

    def test_pipeline_parallel_is_not_divided_out_of_tp(self):
        # tp_size // nnodes would give 2; the launcher puts 4 schedulers per
        # node here (pp_size_per_node=1, tp_size_per_node=4).
        self.assertEqual(self._count(tp_size=4, pp_size=2, nnodes=2), 4)

    def test_classic_data_parallel_multiplies_replicas(self):
        # Each replica owns its own KV pool, hence its own host mirror.
        self.assertEqual(
            self._count(tp_size=2, dp_size=3, nnodes=1, enable_dp_attention=False), 6
        )

    def test_attention_dp_does_not_multiply(self):
        # Attention DP folds dp_size into the TP group instead of replicating.
        self.assertEqual(
            self._count(tp_size=4, dp_size=4, nnodes=1, enable_dp_attention=True), 4
        )


class TestHostPoolRetractionFits(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(tp_size=1, nnodes=1)
        override.install()
        self.addCleanup(override.restore)

    def _fits(self, pool_bytes, available_bytes, draft_pools=()):
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.psutil.virtual_memory",
            return_value=SimpleNamespace(available=available_bytes),
        ):
            return _host_pool_retraction_fits(_fake_pool(pool_bytes), draft_pools)

    def test_small_pool_fits(self):
        fits, required, available = self._fits(8 * GB, 512 * GB)
        self.assertTrue(fits)
        self.assertEqual(required, 8 * GB)
        self.assertEqual(available, 512 * GB - HICACHE_HOST_MEMORY_RESERVE_BYTES)

    def test_oversized_pool_does_not_fit(self):
        fits, required, available = self._fits(600 * GB, 300 * GB)
        self.assertFalse(fits)
        self.assertGreater(required, available)

    def test_reserve_is_withheld_from_the_budget(self):
        # Exactly the available figure, so only the reserve can tip it over.
        fits, _, _ = self._fits(200 * GB, 200 * GB)
        self.assertFalse(fits)

    def test_draft_pools_count_toward_the_requirement(self):
        without = self._fits(100 * GB, 1024 * GB)[1]
        with_draft = self._fits(
            100 * GB, 1024 * GB, draft_pools=(_fake_pool(20 * GB),)
        )[1]
        self.assertEqual(with_draft - without, 20 * GB)

    def test_requirement_scales_with_local_schedulers(self):
        override = get_context().override_server_args(tp_size=4, nnodes=1)
        override.install()
        self.addCleanup(override.restore)
        _, required, _ = self._fits(50 * GB, 1024 * GB)
        self.assertEqual(required, 200 * GB)


class TestFallbackKeepsHostPoolRatio(CustomTestCase):
    """The capacity gate must not enlarge the host pool it just judged too big.

    ``hicache_ratio`` keys off the pre-fallback verdict, so a decode server that
    degrades to ``cpu_tensor`` keeps the 1.0 the host-pool path resolved rather
    than inheriting the 2.0 meant for real hierarchical caching.
    """

    def _resolve(self, *, available_bytes: int, pool_bytes: int = 100 * GB) -> str:
        class _Pool:
            def get_kv_size_bytes(self):
                return (pool_bytes // 2, pool_bytes // 2)

        override = get_context().override_server_args(
            disaggregation_mode="decode", tp_size=1, nnodes=1
        )
        override.install()
        self.addCleanup(override.restore)
        # Reset the ratio the dummy-boundary ServerArgs baked in: a real decode
        # server reaches resolution with it unset.
        get_context().override("test.reset_ratio", hicache_ratio=None)

        pool = _Pool()
        tp_worker = SimpleNamespace(
            get_memory_pool=lambda: (None, SimpleNamespace(get_kvcache=lambda: pool)),
            is_hybrid_swa=False,
            model_runner=SimpleNamespace(mtp_draft_device_pools=()),
        )
        with patch.object(kv_cache_builder, "MHATokenToKVPool", _Pool), patch(
            "sglang.srt.mem_cache.kv_cache_builder.psutil.virtual_memory",
            return_value=SimpleNamespace(available=available_bytes),
        ):
            return kv_cache_builder.resolve_decode_retraction_backup(
                tp_worker=tp_worker
            )

    def test_host_pool_kept_when_it_fits(self):
        backend = self._resolve(available_bytes=1024 * GB)
        self.assertEqual(backend, "host_pool")
        self.assertEqual(get_memory().hicache_ratio, 1.0)

    def test_fallback_keeps_ratio_at_one(self):
        backend = self._resolve(available_bytes=8 * GB)
        self.assertEqual(backend, "cpu_tensor")
        # The bug this pins: keying the ratio off the post-fallback backend
        # would publish 2.0 here, doubling the pool on a host that is short.
        self.assertEqual(get_memory().hicache_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
