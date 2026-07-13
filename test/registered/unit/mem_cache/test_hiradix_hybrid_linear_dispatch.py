import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _hybrid_pool(full_kv_pool):
    pool = object.__new__(HybridLinearKVPool)
    pool.full_kv_pool = full_kv_pool
    return pool


class TestHiRadixHybridLinearDispatch(unittest.TestCase):
    def _build_cache(self, kv_cache):
        params = MagicMock()
        params.token_to_kv_pool_allocator.get_kvcache.return_value = kv_cache
        params.tp_cache_group = object()
        params.attn_cp_cache_group = None
        params.attn_tp_cache_group = None
        params.pp_cache_group = None
        params.pp_rank = 0
        params.pp_size = 1
        params.enable_metrics = False
        params.page_size = 1

        server_args = MagicMock()
        server_args.hicache_storage_backend = None
        server_args.hicache_storage_backend_extra_config = None

        parse_result = ({}, 256, MagicMock(), False)
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch.object(
                HiRadixCache,
                "_parse_storage_backend_extra_config",
                return_value=parse_result,
            ),
            patch.object(HiRadixCache, "_apply_storage_runtime_config"),
            patch.object(RadixCache, "__init__", return_value=None),
            patch("atexit.register"),
        ):
            return HiRadixCache(params, server_args), params, server_args

    def test_mha_inner_pool_uses_standard_hicache_controller(self):
        full_kv_pool = object.__new__(MHATokenToKVPool)
        kv_cache = _hybrid_pool(full_kv_pool)
        host_pool_cls = MagicMock()

        with (
            patch(
                "sglang.srt.mem_cache.hiradix_cache.get_mha_host_pool_cls",
                return_value=host_pool_cls,
            ),
            patch(
                "sglang.srt.mem_cache.hiradix_cache.HiCacheController"
            ) as controller_cls,
            patch(
                "sglang.srt.mem_cache.hiradix_cache.attach_hybrid_dsa_pool_to_hiradix_cache"
            ) as attach_dsa,
        ):
            cache, params, server_args = self._build_cache(kv_cache)

        host_pool_cls.assert_called_once_with(
            full_kv_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            1,
            server_args.hicache_mem_layout,
            allocator_type=None,
        )
        controller_cls.assert_called_once()
        self.assertIs(
            controller_cls.call_args.args[0], params.token_to_kv_pool_allocator
        )
        self.assertIs(controller_cls.call_args.args[1], cache.token_to_kv_pool_host)
        attach_dsa.assert_not_called()

    def test_dsa_inner_pool_uses_dsa_sidecar_stack(self):
        full_kv_pool = object.__new__(DSATokenToKVPool)
        kv_cache = _hybrid_pool(full_kv_pool)

        with (
            patch(
                "sglang.srt.mem_cache.hiradix_cache.HiCacheController"
            ) as controller_cls,
            patch(
                "sglang.srt.mem_cache.hiradix_cache.attach_hybrid_dsa_pool_to_hiradix_cache"
            ) as attach_dsa,
        ):
            cache, params, server_args = self._build_cache(kv_cache)

        attach_dsa.assert_called_once()
        self.assertIs(attach_dsa.call_args.args[0], cache)
        self.assertIs(attach_dsa.call_args.args[1], params)
        self.assertIs(attach_dsa.call_args.args[2], server_args)
        controller_cls.assert_not_called()

    def test_mla_inner_pool_uses_standard_hicache_controller(self):
        full_kv_pool = object.__new__(MLATokenToKVPool)
        kv_cache = _hybrid_pool(full_kv_pool)

        with (
            patch(
                "sglang.srt.mem_cache.hiradix_cache.MLATokenToKVPoolHost"
            ) as host_pool_cls,
            patch(
                "sglang.srt.mem_cache.hiradix_cache.HiCacheController"
            ) as controller_cls,
            patch(
                "sglang.srt.mem_cache.hiradix_cache.attach_hybrid_dsa_pool_to_hiradix_cache"
            ) as attach_dsa,
        ):
            cache, params, server_args = self._build_cache(kv_cache)

        host_pool_cls.assert_called_once_with(
            full_kv_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            1,
            server_args.hicache_mem_layout,
            allocator_type=None,
        )
        controller_cls.assert_called_once()
        self.assertIs(
            controller_cls.call_args.args[0], params.token_to_kv_pool_allocator
        )
        self.assertIs(controller_cls.call_args.args[1], cache.token_to_kv_pool_host)
        attach_dsa.assert_not_called()

    def test_mha_inner_pool_does_not_declare_indexer_sidecar(self):
        cache = object.__new__(HiRadixCache)
        cache.cache_controller = object.__new__(HybridCacheController)
        cache.kv_cache = _hybrid_pool(object.__new__(MHATokenToKVPool))

        self.assertEqual(cache._get_extra_pools(), {})

    def test_dsa_inner_pool_declares_indexer_sidecar(self):
        cache = object.__new__(HiRadixCache)
        cache.cache_controller = object.__new__(HybridCacheController)
        cache.kv_cache = _hybrid_pool(object.__new__(DSATokenToKVPool))

        extra_pools = cache._get_extra_pools()["extra_pools"]

        self.assertEqual(len(extra_pools), 1)
        self.assertEqual(extra_pools[0].name, hybrid_pool_assembler.PoolName.INDEXER)

    def test_dsa_attach_uses_unwrapped_pool_type(self):
        full_kv_pool = object.__new__(DSATokenToKVPool)
        full_kv_pool.layer_num = 2
        full_kv_pool.kv_cache_dim = 128
        kv_cache = _hybrid_pool(full_kv_pool)
        kv_cache.use_mla = False

        radix_cache = MagicMock()
        radix_cache.kv_cache = kv_cache
        radix_cache.page_size = 1
        host_pool_group = MagicMock()
        cache_controller = MagicMock()
        server_args = MagicMock()

        with patch.object(
            hybrid_pool_assembler,
            "build_anchor_sidecar_stack",
            return_value=(host_pool_group, cache_controller),
        ) as build_stack:
            hybrid_pool_assembler.attach_hybrid_dsa_pool_to_hiradix_cache(
                radix_cache,
                MagicMock(),
                server_args,
                extra_config={},
                prefetch_threshold=256,
                enable_storage_metrics=False,
                load_cache_event=object(),
            )

        self.assertIs(build_stack.call_args.kwargs["kv_pool"], full_kv_pool)
        self.assertTrue(build_stack.call_args.kwargs["use_mla"])


if __name__ == "__main__":
    unittest.main()
