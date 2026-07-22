import importlib.util
import sys
import threading
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

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


def test_restored_swa_tail_marks_older_prefix_as_evicted_before_cache_insert():
    inner = MagicMock()
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    req = SimpleNamespace(
        kv=SimpleNamespace(swa_evicted_seqlen=0),
        _flexkv_swa_evicted_seqlen=10240,
    )

    cache.cache_unfinished_req(req, chunked=True)

    assert req.kv.swa_evicted_seqlen == 10240
    assert not hasattr(req, "_flexkv_swa_evicted_seqlen")
    inner.cache_unfinished_req.assert_called_once_with(req, chunked=True)


def test_prefill_boundary_is_stored_with_an_independent_tracking_key():
    inner = MagicMock()
    inner.is_eagle = False
    inner.root_node = object()
    node = object()
    indices = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    inner.match_prefix.return_value = SimpleNamespace(
        last_device_node=node,
        device_indices=indices,
    )
    dec_params = object()
    inner.inc_lock_ref.return_value = SimpleNamespace(to_dec_params=lambda: dec_params)

    connector = MagicMock()
    connector.store_kv.side_effect = [17, 18]
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache.flexkv_connector = connector
    cache.page_size = 4
    cache._node_lock = threading.Lock()
    cache._store_generation = 0
    cache._inflight_store_nodes = {}

    req = SimpleNamespace(
        rid="request",
        extra_key=None,
        kv=SimpleNamespace(swa_evicted_seqlen=0),
        get_fill_ids=lambda: array("q", [1, 2, 3, 4]),
    )

    cache.cache_unfinished_req(req)
    cache._store_prefix(req, [1, 2, 3, 4])

    inner.cache_unfinished_req.assert_called_once_with(req)
    first_store, second_store = connector.store_kv.call_args_list
    assert first_store.args[:2] == ("request:flexkv-store:0", [1, 2, 3, 4])
    assert first_store.args[2] is indices
    assert second_store.args[:2] == ("request:flexkv-store:1", [1, 2, 3, 4])
    assert second_store.args[2] is indices
    assert cache._inflight_store_nodes == {
        "request:flexkv-store:0": (node, dec_params),
        "request:flexkv-store:1": (node, dec_params),
    }


def test_reset_drains_flexkv_before_releasing_inner_slots():
    calls = MagicMock()
    connector = MagicMock()
    inner = MagicMock()
    calls.attach_mock(connector, "connector")
    calls.attach_mock(inner, "inner")

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.flexkv_connector = connector
    cache._inner_cache = inner
    cache._load_markers = {}
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}

    cache.reset()

    assert calls.mock_calls[:2] == [
        call.connector.reset(),
        call.inner.reset(),
    ]


def test_page_size_one_restore_requests_swa_for_the_full_hit():
    allocator = MagicMock()
    restored_slots = torch.arange(8, dtype=torch.int64)
    allocator.alloc.side_effect = [None, restored_slots]

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.page_size = 1
    cache.token_to_kv_pool_allocator = allocator
    cache.supports_swa = MagicMock(return_value=True)
    req = SimpleNamespace()

    with patch(
        "sglang.srt.mem_cache.common.evict_from_tree_cache"
    ) as evict_from_tree_cache:
        result = cache._alloc_restore_slots(req, host_hit_length=8)

    assert result is restored_slots
    evict_from_tree_cache.assert_called_once_with(cache, 8, swa_num_tokens=8)
