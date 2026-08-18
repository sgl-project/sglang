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
    """An MHA/SWA-shaped stand-in: ``get_kv_size_bytes`` returns a ``(k, v)`` pair."""
    return SimpleNamespace(
        get_kv_size_bytes=lambda: (total_bytes // 2, total_bytes // 2)
    )


def _scalar_pool(total_bytes: int) -> SimpleNamespace:
    """An MLA/DSA/Mamba-shaped stand-in: ``get_kv_size_bytes`` returns a scalar.

    The accessor is not a uniform contract, and a tuple-only fixture hides the
    shape that reaches the estimate on every DeepSeek-class and Mamba-hybrid
    server.
    """
    return SimpleNamespace(get_kv_size_bytes=lambda: total_bytes)


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


class TestCapacityReductionIsUnconditional(CustomTestCase):
    """The all-reduce must not sit behind a per-rank predicate.

    ``supports_host_pool`` is per-rank state (pool class, full-token capacity).
    If the reduction were reached only by ranks that chose ``host_pool``, a rank
    that decided otherwise would leave its peers blocked in the all-reduce. Only
    ``backend is None`` -- pure config, hence rank-uniform -- may gate it.
    """

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
        # A scalar-shaped pool (MLA/DSA/Mamba) is config-eligible but cannot host
        # the mirror: it resolves to cpu_tensor, and the reduction still runs so
        # the rank cannot strand peers that did choose host_pool. Uses the scalar
        # fixture because a tuple-only stand-in hides the shape that actually
        # reaches the estimate here.
        backend, agree = self._resolve_with_pool(_scalar_pool(8 * GB))
        self.assertEqual(backend, "cpu_tensor")
        agree.assert_called_once()

    def test_estimate_itself_performs_no_collective(self):
        # Purity keeps the helper unit-testable and keeps the reduction visible
        # at the call site rather than buried in the estimate.
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
        # MLA, DSA, and Mamba pools return a scalar; assuming the pair shape
        # raises TypeError: 'int' object is not iterable.
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
    """Only a config-eligible decode server should pay for the capacity check.

    ``resolve_decode_retraction_backup`` runs from ``init_memory_pools`` on every
    scheduler, so a prefill, embedding, or non-PD server reaches this code too.
    It must not read host memory, reduce over the world group, or hand its pool
    to the estimate.
    """

    def _resolve(self, pool, **fields):
        # Pin the inputs the gate reads rather than inheriting whatever a
        # previous case left published: the retraction backup must be unset for
        # the inference branch to run at all, and the mode decides eligibility.
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
