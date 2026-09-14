"""Unit tests for hybrid HiCache pool assembly."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _log_mamba_host_coverage,
    _mamba_host_bytes_per_slot,
    _resolve_hicache_mamba_split,
    _split_hicache_size,
    build_full_draft_pools,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes):
        self._kv_bytes = kv_bytes

    def get_kv_size_bytes(self):
        return self._kv_bytes


class TestDeviceAllocEviction(CustomTestCase):
    def test_swa_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 8

        _evict_swa_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(swa_num_tokens=2))
        cache.evict.assert_not_called()

    def test_mamba_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        allocator = cache.req_to_token_pool.mamba_allocator
        allocator.schedulable_available_size.return_value = 8

        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(mamba_num=2))
        cache.evict.assert_not_called()

    def test_sufficient_capacity_skips_eviction(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 10
        cache.req_to_token_pool.mamba_allocator.schedulable_available_size.return_value = 10

        _evict_swa_for_device_alloc(cache, required_size=10)
        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_not_called()
        cache.evict.assert_not_called()


class TestSplitHicacheSize(CustomTestCase):
    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


class _DevicePool(_Pool):
    def __init__(self, kv_bytes, size):
        super().__init__(kv_bytes)
        self.size = size


def _fake_mamba_pool(*, slots, layers, conv_shapes, temporal_shape, itemsize):
    """A device MambaPool as MambaPoolHost.get_size_per_token reads it."""
    dtype = SimpleNamespace(itemsize=itemsize)
    cache = SimpleNamespace(
        conv=[
            SimpleNamespace(shape=(layers, slots) + shape, dtype=dtype)
            for shape in conv_shapes
        ],
        temporal=SimpleNamespace(shape=(layers, slots) + temporal_shape, dtype=dtype),
    )
    per_slot = (
        sum(math.prod(conv.shape[2:]) * conv.dtype.itemsize for conv in cache.conv)
        + math.prod(cache.temporal.shape[2:]) * cache.temporal.dtype.itemsize
    ) * layers
    pool = _DevicePool(slots * per_slot, slots)
    pool.mamba_cache = cache
    pool.num_mamba_layers = layers
    return pool


def _glm_mamba_pool(slots=160):
    # GLM-5.3-Flash KDA checkpoint: 35 layers x (36,864 B conv + 524,288 B ssm).
    return _fake_mamba_pool(
        slots=slots,
        layers=35,
        conv_shapes=((4, 4608),),
        temporal_shape=(16, 128, 128),
        itemsize=2,
    )


class TestHicacheMambaSizeKnob(CustomTestCase):
    """--hicache-mamba-size-gb resolved in the assembler against fake device pools."""

    def _publish(self, **overrides):
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy", **overrides), role="scheduler")
        self.addCleanup(reset_context)

    def test_bytes_per_slot_mirrors_mamba_pool_host(self):
        self.assertEqual(_mamba_host_bytes_per_slot(_glm_mamba_pool()), 19_640_320)

    def test_explicit_gigabytes_take_the_remainder_from_kv(self):
        self._publish(hicache_size=32)
        kv_pool = _DevicePool(7_920 * 1_114_112, 1_114_112)  # 7,920 B per token
        params = SimpleNamespace(
            chunked_prefill_size=4096, req_to_token_pool=SimpleNamespace(size=8)
        )

        kv_shares, mamba_gb, split = _resolve_hicache_mamba_split(
            knob=14.0, kv_pools=(kv_pool,), mamba_pool=_glm_mamba_pool(), params=params
        )

        self.assertEqual(kv_shares, (18.0,))
        self.assertEqual(mamba_gb, 14.0)
        self.assertEqual(split.mode, "explicit")
        self.assertEqual(split.slots, int(14e9 // 19_640_320))
        self.assertGreaterEqual(split.coverage, 1.0)

    def test_auto_keeps_kv_pools_proportional_among_themselves(self):
        self._publish(hicache_size=32)
        full_kv_pool = _DevicePool(3 * 10**9, 1_000_000)
        swa_kv_pool = _DevicePool(1 * 10**9, 1_000_000)
        params = SimpleNamespace(
            chunked_prefill_size=4096, req_to_token_pool=SimpleNamespace(size=4)
        )

        kv_shares, mamba_gb, split = _resolve_hicache_mamba_split(
            knob="auto",
            kv_pools=(full_kv_pool, swa_kv_pool),
            mamba_pool=_glm_mamba_pool(),
            params=params,
        )

        self.assertEqual(split.mode, "auto")
        self.assertAlmostEqual(kv_shares[0], 3 * kv_shares[1])
        self.assertAlmostEqual(sum(kv_shares) + mamba_gb, 32.0)
        self.assertGreaterEqual(split.coverage, 1.0)
        self.assertGreaterEqual(split.slots, math.ceil(split.kv_tokens / 4096) + 4 * 4)


class TestMambaHostCoverageLine(CustomTestCase):
    """The boot line is informational: mocks skip it, real sizes log it."""

    _LOGGER = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler"

    def _publish(self, **overrides):
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy", **overrides), role="scheduler")
        self.addCleanup(reset_context)

    def test_mock_params_and_pools_skip_the_line(self):
        # Assembly under MagicMock params and a patched MambaPoolHost: no
        # TypeError, no line.
        with self.assertNoLogs(self._LOGGER, level="INFO"):
            _log_mamba_host_coverage(
                kv_host_pool=MagicMock(),
                mamba_host_pool=MagicMock(),
                params=MagicMock(),
                split=None,
            )

    def test_real_sizes_log_the_production_line(self):
        self._publish(hicache_size=32)
        params = SimpleNamespace(
            chunked_prefill_size=4096, req_to_token_pool=SimpleNamespace(size=8)
        )
        with self.assertLogs(self._LOGGER, level="WARNING") as logs:
            _log_mamba_host_coverage(
                kv_host_pool=SimpleNamespace(size=2_680_896),
                mamba_host_pool=SimpleNamespace(size=565),
                params=params,
                split=None,
            )
        # The GLM-5.3-Flash TP4 boot line at the default split.
        self.assertEqual(len(logs.records), 1)
        self.assertIn(
            "HiCache host split (proportional): host mamba slots 565 cover "
            "2183168 tokens at one checkpoint per 4096-token chunk; KV host "
            "tier 2680896 tokens (coverage 82%)",
            logs.output[0],
        )
        self.assertIn("--hicache-mamba-size-gb", logs.output[0])


class TestDraftSidecarPoolDispatch(CustomTestCase):
    def test_full_builder_unwraps_empty_hybrid_linear_pool(self):
        draft_kv_pool = object.__new__(HybridLinearKVPool)
        draft_kv_pool.full_kv_pool = SimpleNamespace(layer_num=0)

        specs, entries = build_full_draft_pools(
            draft_kv_pool=draft_kv_pool,
            tree_cache=None,
        )

        self.assertEqual(specs, [])
        self.assertEqual(entries, [])

    def test_full_builder_sizes_sidecar_for_anchor_logical_space(self):
        draft_kv_pool = SimpleNamespace(layer_num=1, size=800)
        draft_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=100, logical_size=800),
                page_size=512,
            )
        )
        # The layout comes from the published configuration.
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(model_path="dummy", hicache_mem_layout="page_first")
        publish(server_args, role="scheduler")
        self.addCleanup(reset_context)

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_build_mha_mla_host_pool",
                return_value=draft_host_pool,
            ) as build_host_pool,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_get_allocator_type",
                return_value="default",
            ),
        ):
            specs, entries = build_full_draft_pools(
                draft_kv_pool=draft_kv_pool,
                tree_cache=tree_cache,
            )

        self.assertEqual(build_host_pool.call_args.kwargs["host_to_device_ratio"], 1.0)
        self.assertEqual(len(specs), 1)
        self.assertIs(entries[0].host_pool, draft_host_pool)


if __name__ == "__main__":
    unittest.main()
