"""CPU-only regression tests for external-cache committed KV boundaries."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib
import sys
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache


class _Metadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _optional_dependency_stubs():
    lmcache_modules = {
        "lmcache": types.ModuleType("lmcache"),
        "lmcache.integration": types.ModuleType("lmcache.integration"),
        "lmcache.integration.sglang": types.ModuleType("lmcache.integration.sglang"),
        "lmcache.integration.sglang.multi_process_adapter": types.ModuleType(
            "lmcache.integration.sglang.multi_process_adapter"
        ),
        "lmcache.integration.sglang.sglang_adapter": types.ModuleType(
            "lmcache.integration.sglang.sglang_adapter"
        ),
        "lmcache.integration.sglang.utils": types.ModuleType(
            "lmcache.integration.sglang.utils"
        ),
    }
    lmcache_modules[
        "lmcache.integration.sglang.multi_process_adapter"
    ].LMCacheMPConnector = object
    adapter = lmcache_modules["lmcache.integration.sglang.sglang_adapter"]
    adapter.LMCacheLayerwiseConnector = object
    adapter.LoadMetadata = _Metadata
    adapter.StoreMetadata = _Metadata
    lmcache_modules["lmcache.integration.sglang.utils"].lmcache_get_config = MagicMock()

    flexkv_connector = types.ModuleType(
        "sglang.srt.mem_cache.storage.flexkv.flexkv_connector"
    )
    flexkv_connector.FlexKVConnector = object
    return {
        **lmcache_modules,
        "sglang.srt.mem_cache.storage.flexkv.flexkv_connector": flexkv_connector,
    }


class TestExternalCacheEffectiveKVLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._module_names = (
            "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache",
            "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache",
        )
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._module_names}
        for name in cls._module_names:
            sys.modules.pop(name, None)
        with patch.dict(sys.modules, _optional_dependency_stubs()):
            cls.lmc = importlib.import_module(cls._module_names[0])
            cls.flexkv = importlib.import_module(cls._module_names[1])

    @classmethod
    def tearDownClass(cls):
        for name, module in cls._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _make_cache_and_req(self, cache_cls, *, is_lmcache):
        cache = cache_cls.__new__(cache_cls)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int64).reshape(1, 32)
        )
        cache.inc_lock_ref = MagicMock()
        cache.dec_lock_ref = MagicMock()
        cache._node_lock = nullcontext()
        cache.store_stream = object()

        connector = MagicMock()
        if is_lmcache:
            cache._mode = self.lmc.LMCacheMode.MP
            cache._mp_load_back_markers = {}
        else:
            cache._inflight_store_nodes = {}
            connector.store_kv.return_value = -1
        cache.lmcache_connector = connector
        cache.flexkv_connector = connector

        req = SimpleNamespace(
            rid="req",
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[5, 6, 7, 8, 9],
            kv_committed_len=9,
            req_pool_idx=0,
            extra_key=None,
        )
        return cache, connector, req

    def test_external_stores_honor_effective_committed_length(self):
        cases = (
            (self.lmc.LMCRadixCache, True),
            (self.flexkv.FlexKVRadixCache, False),
        )
        node = object()
        match = SimpleNamespace(last_device_node=node)

        for cache_cls, is_lmcache in cases:
            with self.subTest(backend=cache_cls.__name__):
                cache, connector, req = self._make_cache_and_req(
                    cache_cls, is_lmcache=is_lmcache
                )
                with (
                    patch.object(RadixCache, "cache_finished_req"),
                    patch.object(RadixCache, "match_prefix", return_value=match),
                    patch.object(
                        self.lmc if is_lmcache else self.flexkv,
                        "get_spec",
                        return_value=SimpleNamespace(speculative_eagle_topk=4),
                    ),
                    patch.object(torch.cuda, "stream", return_value=nullcontext()),
                ):
                    cache.cache_finished_req(req, kv_len_to_handle=4)

                if is_lmcache:
                    metadata = connector.store_kv.call_args.args[0]
                    self.assertEqual(metadata.token_ids, [1, 2, 3, 4])
                    self.assertEqual(metadata.kv_indices.tolist(), [0, 1, 2, 3])
                else:
                    kwargs = connector.store_kv.call_args.kwargs
                    self.assertEqual(kwargs["token_ids"], [1, 2, 3, 4])
                    self.assertEqual(kwargs["kv_indices"].tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
