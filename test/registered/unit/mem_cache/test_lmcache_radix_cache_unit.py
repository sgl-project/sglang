import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _FakeStream:
    def synchronize(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeKVCache:
    def __init__(self):
        self.layer_transfer_counter = None

    def register_layer_transfer_counter(self, counter):
        self.layer_transfer_counter = counter


class _FakeAllocator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.kvcache = _FakeKVCache()

    def get_kvcache(self):
        return self.kvcache


def _build_fake_lmcache_modules():
    lmcache_mod = types.ModuleType("lmcache")
    lmcache_mod.__path__ = []

    integration_mod = types.ModuleType("lmcache.integration")
    integration_mod.__path__ = []

    sglang_mod = types.ModuleType("lmcache.integration.sglang")
    sglang_mod.__path__ = []

    mp_mod = types.ModuleType("lmcache.integration.sglang.multi_process_adapter")
    adapter_mod = types.ModuleType("lmcache.integration.sglang.sglang_adapter")

    class FakeConnector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class FakeMPConnector(FakeConnector):
        pass

    class FakeLoadMetadata:
        pass

    class FakeStoreMetadata:
        pass

    mp_mod.LMCacheMPLayerwiseConnector = FakeMPConnector
    adapter_mod.LMCacheLayerwiseConnector = FakeConnector
    adapter_mod.LoadMetadata = FakeLoadMetadata
    adapter_mod.StoreMetadata = FakeStoreMetadata

    return (
        {
            "lmcache": lmcache_mod,
            "lmcache.integration": integration_mod,
            "lmcache.integration.sglang": sglang_mod,
            "lmcache.integration.sglang.multi_process_adapter": mp_mod,
            "lmcache.integration.sglang.sglang_adapter": adapter_mod,
        },
        FakeConnector,
        FakeMPConnector,
    )


class TestLMCRadixCacheConnectorSelection(unittest.TestCase):
    def _import_module(self):
        target = "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
        sys.modules.pop(target, None)
        self.addCleanup(sys.modules.pop, target, None)
        fake_modules, fake_connector_cls, fake_mp_connector_cls = (
            _build_fake_lmcache_modules()
        )
        with patch.dict(sys.modules, fake_modules):
            module = importlib.import_module(target)
        return module, fake_connector_cls, fake_mp_connector_cls

    def _make_params(self):
        return CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=_FakeAllocator(),
            page_size=16,
        )

    def test_uses_in_process_connector_when_mp_host_is_unset(self):
        module, fake_connector_cls, fake_mp_connector_cls = self._import_module()
        params = self._make_params()
        model_config = SimpleNamespace(num_hidden_layers=2)
        server_args = SimpleNamespace(lmcache_mp_host=None, lmcache_mp_port=5555)

        with patch(
            "sglang.srt.server_args.get_global_server_args", return_value=server_args
        ):
            with patch.object(module.torch.cuda, "Stream", side_effect=_FakeStream):
                cache = module.LMCRadixCache(params, model_config=model_config)

        self.assertIsInstance(cache.lmcache_connector, fake_connector_cls)
        self.assertNotIsInstance(cache.lmcache_connector, fake_mp_connector_cls)
        self.assertNotIn("page_size", cache.lmcache_connector.kwargs)
        self.assertIs(
            params.token_to_kv_pool_allocator.kvcache.layer_transfer_counter.lmc_connector,
            cache.lmcache_connector,
        )

        cache.reset()
        self.assertEqual(cache.lmcache_connector.reset_calls, 0)

    def test_uses_mp_connector_and_resets_it_when_mp_host_is_set(self):
        module, _, fake_mp_connector_cls = self._import_module()
        params = self._make_params()
        model_config = SimpleNamespace(num_hidden_layers=2)
        server_args = SimpleNamespace(lmcache_mp_host="127.0.0.1", lmcache_mp_port=6000)

        with patch(
            "sglang.srt.server_args.get_global_server_args", return_value=server_args
        ):
            with patch.object(module.torch.cuda, "Stream", side_effect=_FakeStream):
                cache = module.LMCRadixCache(params, model_config=model_config)

        self.assertIsInstance(cache.lmcache_connector, fake_mp_connector_cls)
        self.assertEqual(cache.lmcache_connector.kwargs["page_size"], 16)
        self.assertEqual(cache.lmcache_connector.kwargs["host"], "127.0.0.1")
        self.assertEqual(cache.lmcache_connector.kwargs["port"], 6000)

        cache.reset()
        self.assertEqual(cache.lmcache_connector.reset_calls, 1)
