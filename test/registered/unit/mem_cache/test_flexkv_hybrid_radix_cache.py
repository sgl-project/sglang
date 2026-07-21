import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_hybrid_cache_class():
    """Load the wrapper without requiring the optional FlexKV package."""
    module_name = "_flexkv_hybrid_radix_cache_under_test"
    connector_name = "sglang.srt.mem_cache.storage.flexkv.flexkv_connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object

    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_hybrid_radix_cache.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch.dict(sys.modules, {connector_name: connector_stub}):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.FlexKVHybridRadixCache


FlexKVHybridRadixCache = _load_hybrid_cache_class()


def test_pool_accounting_delegates_to_inner_cache():
    inner = MagicMock()
    inner.evictable_size.return_value = 1280
    inner.full_evictable_size.return_value = 1024
    inner.swa_evictable_size.return_value = 256
    inner.protected_size.return_value = 640
    inner.full_protected_size.return_value = 512
    inner.swa_protected_size.return_value = 128

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner

    assert cache.evictable_size() == 1280
    assert cache.full_evictable_size() == 1024
    assert cache.swa_evictable_size() == 256
    assert cache.protected_size() == 640
    assert cache.full_protected_size() == 512
    assert cache.swa_protected_size() == 128
