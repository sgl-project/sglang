import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.base_prefix_cache import CacheFinishedReqResult
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


class TestCacheFinishedReqStorageHandoff(unittest.TestCase):
    def test_flexkv_returns_the_base_cache_handoff(self) -> None:
        """FlexKV forwards the base ownership boundary on a non-store path."""
        external_modules: dict[str, ModuleType] = {
            "flexkv": _make_module("flexkv"),
            "flexkv.common": _make_module("flexkv.common"),
            "flexkv.common.request": _make_module(
                "flexkv.common.request", KVResponseStatus=object
            ),
            "flexkv.common.storage": _make_module(
                "flexkv.common.storage",
                KVCacheLayout=object,
                KVCacheLayoutType=object,
            ),
            "flexkv.integration": _make_module("flexkv.integration"),
            "flexkv.integration.config": _make_module(
                "flexkv.integration.config", FlexKVConfig=object
            ),
            "flexkv.kvmanager": _make_module(
                "flexkv.kvmanager", KVManager=object
            ),
            "flexkv.server": _make_module("flexkv.server"),
            "flexkv.server.client": _make_module(
                "flexkv.server.client", KVTPClient=object
            ),
            "flexkv.transfer": _make_module("flexkv.transfer"),
            "flexkv.transfer.layerwise": _make_module(
                "flexkv.transfer.layerwise",
                build_layerwise_eventfd_socket_path=object,
            ),
            "flexkv.transfer_manager": _make_module(
                "flexkv.transfer_manager", TransferManagerOnRemote=object
            ),
        }
        expected = CacheFinishedReqResult(unhandled_kv_start=8)

        with (
            mock.patch.dict(sys.modules, external_modules),
            mock.patch.object(
                RadixCache,
                "cache_finished_req",
                return_value=expected,
            ),
        ):
            module = importlib.import_module(
                "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"
            )
            cache = module.FlexKVRadixCache.__new__(module.FlexKVRadixCache)
            cache._load_markers = {}
            result = cache.cache_finished_req(
                SimpleNamespace(rid="request"),
                is_insert=False,
                kv_len_to_handle=10,
            )

        self.assertIs(result, expected)

    def test_lmcache_returns_the_base_cache_handoff(self) -> None:
        """LMCache forwards the base ownership boundary on a non-store path."""
        external_modules: dict[str, ModuleType] = {
            "lmcache": _make_module("lmcache"),
            "lmcache.integration": _make_module("lmcache.integration"),
            "lmcache.integration.sglang": _make_module(
                "lmcache.integration.sglang"
            ),
            "lmcache.integration.sglang.multi_process_adapter": _make_module(
                "lmcache.integration.sglang.multi_process_adapter",
                LMCacheMPConnector=object,
            ),
            "lmcache.integration.sglang.sglang_adapter": _make_module(
                "lmcache.integration.sglang.sglang_adapter",
                LMCacheLayerwiseConnector=object,
                LoadMetadata=object,
                StoreMetadata=object,
            ),
            "lmcache.integration.sglang.utils": _make_module(
                "lmcache.integration.sglang.utils", lmcache_get_config=object
            ),
        }
        expected = CacheFinishedReqResult(unhandled_kv_start=8)

        with (
            mock.patch.dict(sys.modules, external_modules),
            mock.patch.object(
                RadixCache,
                "cache_finished_req",
                return_value=expected,
            ),
        ):
            module = importlib.import_module(
                "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
            )
            cache = module.LMCRadixCache.__new__(module.LMCRadixCache)
            cache._mode = module.LMCacheMode.IP
            result = cache.cache_finished_req(
                SimpleNamespace(rid="request"),
                is_insert=False,
                kv_len_to_handle=10,
            )

        self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
