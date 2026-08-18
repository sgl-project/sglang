"""Capacity gating for the inferred PD decode retraction backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.kv_cache_builder import (
    _host_pool_retraction_fits,
    _kv_pool_bytes,
    _local_scheduler_count,
)
from sglang.srt.mem_cache.pool_host.base import HICACHE_HOST_MEMORY_RESERVE_BYTES
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1024**3


def _fake_pool(total_bytes: int) -> SimpleNamespace:
    """MHA/SWA shape: ``get_kv_size_bytes`` returns a ``(k, v)`` pair."""
    return SimpleNamespace(
        get_kv_size_bytes=lambda: (total_bytes // 2, total_bytes // 2)
    )


def _scalar_pool(total_bytes: int) -> SimpleNamespace:
    """MLA/DSA/Mamba shape: ``get_kv_size_bytes`` returns a scalar."""
    return SimpleNamespace(get_kv_size_bytes=lambda: total_bytes)


class TestLocalSchedulerCount(CustomTestCase):
    def test_topologies(self):
        # (fields, expected schedulers on this node)
        cases = [
            ({"tp_size": 4, "nnodes": 1}, 4),
            ({"tp_size": 8, "nnodes": 2}, 4),
            # tp_size // nnodes would say 2; the launcher puts 4 here.
            ({"tp_size": 4, "pp_size": 2, "nnodes": 2}, 4),
            # Classic DP replicates whole pools; attention DP does not.
            ({"tp_size": 2, "dp_size": 3, "enable_dp_attention": False}, 6),
            ({"tp_size": 4, "dp_size": 4, "enable_dp_attention": True}, 4),
        ]
        for fields, expected in cases:
            with self.subTest(**fields):
                override = get_context().override_server_args(**{"nnodes": 1, **fields})
                override.install()
                self.addCleanup(override.restore)
                self.assertEqual(_local_scheduler_count(), expected)


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
        # Exactly the available figure; only the reserve tips it over.
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
    """A decode server that degrades keeps the host-pool ratio of 1.0, not the
    2.0 meant for real hierarchical caching."""

    def _resolve(self, *, available_bytes: int, pool_bytes: int = 100 * GB) -> str:
        class _Pool:
            def get_kv_size_bytes(self):
                return (pool_bytes // 2, pool_bytes // 2)

        override = get_context().override_server_args(
            disaggregation_mode="decode", tp_size=1, nnodes=1
        )
        override.install()
        self.addCleanup(override.restore)
        # The dummy-boundary ServerArgs bakes in a ratio; a real decode server
        # reaches resolution with it unset.
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
        # Keying off the post-fallback backend would publish 2.0 here.
        self.assertEqual(get_memory().hicache_ratio, 1.0)


class TestCapacityReductionIsUnconditional(CustomTestCase):
    """The all-reduce must not sit behind ``supports_host_pool``, which is
    per-rank: a rank that skipped it would block its peers."""

    def _resolve_with_pool(self, pool):
        override = get_context().override_server_args(
            disaggregation_mode="decode",
            tp_size=1,
            nnodes=1,
            disaggregation_decode_retraction_backup=None,
        )
        override.install()
        self.addCleanup(override.restore)
        get_context().override("test.reset_ratio", hicache_ratio=None)

        tp_worker = SimpleNamespace(
            get_memory_pool=lambda: (None, SimpleNamespace(get_kvcache=lambda: pool)),
            is_hybrid_swa=False,
            model_runner=SimpleNamespace(mtp_draft_device_pools=()),
        )
        with patch.object(
            kv_cache_builder, "_agree_across_ranks", side_effect=lambda v: v
        ) as agree, patch(
            "sglang.srt.mem_cache.kv_cache_builder.psutil.virtual_memory",
            return_value=SimpleNamespace(available=1024 * GB),
        ):
            backend = kv_cache_builder.resolve_decode_retraction_backup(
                tp_worker=tp_worker
            )
        return backend, agree

    def test_reduced_even_when_the_pool_cannot_use_host_pool(self):
        # Config-eligible but unsupported pool: resolves to cpu_tensor, yet the
        # reduction still runs so peers choosing host_pool are not stranded.
        backend, agree = self._resolve_with_pool(_scalar_pool(8 * GB))
        self.assertEqual(backend, "cpu_tensor")
        agree.assert_called_once()

    def test_estimate_itself_performs_no_collective(self):
        # Purity keeps the reduction visible at the call site.
        override = get_context().override_server_args(tp_size=1, nnodes=1)
        override.install()
        self.addCleanup(override.restore)
        with patch.object(kv_cache_builder, "_agree_across_ranks") as agree, patch(
            "sglang.srt.mem_cache.kv_cache_builder.psutil.virtual_memory",
            return_value=SimpleNamespace(available=1024 * GB),
        ):
            _host_pool_retraction_fits(_fake_pool(8 * GB), ())
        agree.assert_not_called()


class TestKvPoolBytesShapes(CustomTestCase):
    """``get_kv_size_bytes`` returns a pair for MHA/SWA and a scalar elsewhere."""

    def test_tuple_shape_is_summed(self):
        self.assertEqual(_kv_pool_bytes(_fake_pool(64 * GB)), 64 * GB)

    def test_scalar_shape_is_accepted(self):
        # Assuming the pair shape here raises "not iterable".
        self.assertEqual(_kv_pool_bytes(_scalar_pool(64 * GB)), 64 * GB)

    def test_estimate_accepts_a_scalar_pool(self):
        override = get_context().override_server_args(tp_size=1, nnodes=1)
        override.install()
        self.addCleanup(override.restore)
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.psutil.virtual_memory",
            return_value=SimpleNamespace(available=1024 * GB),
        ):
            fits, required, _ = _host_pool_retraction_fits(
                _scalar_pool(16 * GB), (_scalar_pool(4 * GB),)
            )
        self.assertTrue(fits)
        self.assertEqual(required, 20 * GB)


class TestNonDecodeServersSkipTheEstimate(CustomTestCase):
    """Resolution runs on every scheduler, so non-decode servers must not read
    host memory, reduce over the world group, or reach the estimate."""

    def _resolve(self, pool, **fields):
        # Pin what the gate reads: the backup must be unset for the inference
        # branch to run, and the mode decides eligibility.
        fields.setdefault("disaggregation_mode", "null")
        override = get_context().override_server_args(
            tp_size=1,
            nnodes=1,
            disaggregation_decode_retraction_backup=None,
            **fields,
        )
        override.install()
        self.addCleanup(override.restore)
        get_context().override("test.reset_ratio", hicache_ratio=None)

        tp_worker = SimpleNamespace(
            get_memory_pool=lambda: (None, SimpleNamespace(get_kvcache=lambda: pool)),
            is_hybrid_swa=False,
            model_runner=SimpleNamespace(mtp_draft_device_pools=()),
        )
        with patch.object(
            kv_cache_builder, "_agree_across_ranks", side_effect=lambda v: v
        ) as agree, patch.object(
            kv_cache_builder,
            "_host_pool_retraction_fits",
            return_value=(True, 0, 0),
        ) as fits:
            backend = kv_cache_builder.resolve_decode_retraction_backup(
                tp_worker=tp_worker
            )
        return backend, agree, fits

    def test_non_pd_server_never_estimates_or_reduces(self):
        backend, agree, fits = self._resolve(_scalar_pool(8 * GB))
        self.assertEqual(backend, "cpu_tensor")
        fits.assert_not_called()
        agree.assert_not_called()

    def test_prefill_server_never_estimates_or_reduces(self):
        backend, agree, fits = self._resolve(
            _fake_pool(8 * GB), disaggregation_mode="prefill"
        )
        self.assertEqual(backend, "cpu_tensor")
        fits.assert_not_called()
        agree.assert_not_called()

    def test_decode_server_does_estimate(self):
        _, agree, fits = self._resolve(_fake_pool(8 * GB), disaggregation_mode="decode")
        fits.assert_called_once()
        agree.assert_called_once()


if __name__ == "__main__":
    unittest.main()
