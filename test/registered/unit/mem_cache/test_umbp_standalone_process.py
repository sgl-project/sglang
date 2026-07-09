import sys
import types

import pytest

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

fake_memory_pool_host = types.ModuleType("sglang.srt.mem_cache.memory_pool_host")
fake_memory_pool_host.HostKVCache = object
sys.modules.setdefault("sglang.srt.mem_cache.memory_pool_host", fake_memory_pool_host)

from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore


def _storage_config(extra_config):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="unit-test",
        extra_config=extra_config,
    )


def test_standalone_address_extra_config_requires_env(monkeypatch):
    monkeypatch.delenv("UMBP_STANDALONE_ADDRESS", raising=False)
    with pytest.raises(ValueError, match="UMBP_STANDALONE_ADDRESS"):
        UMBPStore(
            _storage_config({"standalone_address": "unix:///tmp/umbp.grpc.sock"}),
            mem_pool_host=None,
        )


class _FakeTensor:
    def data_ptr(self):
        return 0x100000

    def numel(self):
        return 4096

    def element_size(self):
        return 1


class _FakeHostPool:
    layout = "page_first"
    kv_buffer = _FakeTensor()
    allocator = types.SimpleNamespace(mapped_size=4096)


class _FakeStandaloneClient:
    def __init__(self, mode, register_result=True, register_exception=None):
        self._mode = mode
        self._register_result = register_result
        self._register_exception = register_exception

    def get_deployment_mode(self):
        return self._mode

    def is_distributed(self):
        return False

    def register_memory(self, ptr, size):
        if self._register_exception is not None:
            raise self._register_exception
        return self._register_result


def _store_for_register(mode, *, disable=False, result=True, exc=None):
    store = UMBPStore.__new__(UMBPStore)
    store.client = _FakeStandaloneClient(
        mode, register_result=result, register_exception=exc
    )
    store._umbp_deployment_mode_enum = types.SimpleNamespace(StandaloneProcess=mode)
    store._standalone_process_expected = True
    store._disable_zero_copy_register = disable
    return store


def test_standalone_disable_zero_copy_register_is_fatal():
    store = _store_for_register("standalone", disable=True)
    with pytest.raises(RuntimeError, match="disable_zero_copy_register"):
        store.register_mem_pool_host(_FakeHostPool())


def test_standalone_register_memory_exception_is_fatal():
    store = _store_for_register("standalone", exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="cannot fall back"):
        store.register_mem_pool_host(_FakeHostPool())


def test_standalone_register_memory_false_is_fatal():
    store = _store_for_register("standalone", result=False)
    with pytest.raises(RuntimeError, match="returned false"):
        store.register_mem_pool_host(_FakeHostPool())
