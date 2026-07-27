import importlib.util
import sys
import threading
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
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
    cache._restore_leases = {}
    req = SimpleNamespace(
        rid="request",
        kv=SimpleNamespace(swa_evicted_seqlen=0),
        _flexkv_swa_evicted_seqlen=10240,
    )

    cache.cache_unfinished_req(req, chunked=True)

    assert req.kv.swa_evicted_seqlen == 10240
    assert not hasattr(req, "_flexkv_swa_evicted_seqlen")
    inner.cache_unfinished_req.assert_called_once_with(req, chunked=True)


def test_restore_lease_blocks_duplicate_lookup_until_cache_commit():
    node = object()
    inner_match = MatchResult(
        device_indices=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        last_device_node=node,
        last_host_node=node,
        best_match_node=node,
    )
    inner = MagicMock()
    inner.match_prefix.return_value = inner_match

    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    allocator = MagicMock()
    connector = MagicMock()
    connector.enable_layerwise = False
    connector.lookup_kv.return_value = (7, 4)
    connector.retrieve_kv.return_value = 4

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache.token_to_kv_pool_allocator = allocator
    cache.flexkv_connector = connector
    cache.disable = False
    cache.page_size = 4
    cache.device = torch.device("cpu")
    cache._load_markers = {}
    cache._restore_leases = {}
    cache._restore_generation = 0
    cache._alloc_restore_slots = MagicMock(return_value=restored)
    cache.supports_swa = MagicMock(return_value=False)

    req = SimpleNamespace(
        rid="request",
        prefix_indices=inner_match.device_indices,
        last_node=node,
        cache_protected_len=4,
        kv=None,
        pending_restore_generation=None,
        pending_restore_slots=None,
    )
    key = RadixKey(array("q", range(8)), extra_key=None)
    params = MatchPrefixParams(key=key, req=req)

    first_match = cache.match_prefix(params)
    loaded_indices, _ = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=first_match.best_match_node,
            host_hit_length=first_match.host_hit_length,
            req=req,
        )
    )

    assert loaded_indices is restored
    assert cache.has_uncommitted_restore(req)
    assert req.pending_restore_generation == 0
    assert req.pending_restore_slots is restored

    # Direct callers are guarded too, even though the scheduler normally
    # defers the request before reaching a second prefix match.
    duplicate_match = cache.match_prefix(params)
    assert duplicate_match is inner_match
    connector.lookup_kv.assert_called_once()

    duplicate_indices, _ = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=first_match.best_match_node,
            host_hit_length=first_match.host_hit_length,
            req=req,
        )
    )
    assert duplicate_indices.numel() == 0
    cache._alloc_restore_slots.assert_called_once()
    connector.retrieve_kv.assert_called_once()

    cache.cache_unfinished_req(req, chunked=True)

    assert not cache.has_uncommitted_restore(req)
    assert req._flexkv_uncached_restore is False
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None


def test_partial_restore_is_freed_in_full():
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    allocator = MagicMock()
    connector = MagicMock()
    connector.enable_layerwise = False
    connector.retrieve_kv.return_value = 2

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.token_to_kv_pool_allocator = allocator
    cache.flexkv_connector = connector
    cache.device = torch.device("cpu")
    cache._load_markers = {
        "request": SimpleNamespace(device_length=0),
    }
    cache._restore_leases = {}
    cache._restore_generation = 0
    cache._alloc_restore_slots = MagicMock(return_value=restored)

    req = SimpleNamespace(rid="request", last_node=object())
    loaded_indices, _ = cache.init_load_back(
        InitLoadBackParams(best_match_node=req.last_node, host_hit_length=4, req=req)
    )

    assert loaded_indices.numel() == 0
    allocator.free.assert_called_once_with(restored)
    assert not cache.has_uncommitted_restore(req)
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None


def test_restore_launch_exception_is_retained_until_safe_reset():
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    allocator = MagicMock()
    connector = MagicMock()
    connector.enable_layerwise = True
    connector.start_load_kv_layerwise.side_effect = RuntimeError("launch failed")

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.token_to_kv_pool_allocator = allocator
    cache.flexkv_connector = connector
    cache.device = torch.device("cpu")
    cache._load_markers = {
        "request": SimpleNamespace(device_length=0),
    }
    cache._restore_leases = {}
    cache._restore_generation = 0
    cache._alloc_restore_slots = MagicMock(return_value=restored)

    req = SimpleNamespace(rid="request", last_node=object())
    with pytest.raises(RuntimeError, match="launch failed"):
        cache.init_load_back(
            InitLoadBackParams(
                best_match_node=req.last_node,
                host_hit_length=4,
                req=req,
            )
        )

    allocator.free.assert_not_called()
    assert cache.has_uncommitted_restore(req)
    assert req.pending_restore_slots is restored


def test_finished_release_commits_restore_lease_after_inner_cache():
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    req = SimpleNamespace(
        rid="request",
        pending_restore_generation=3,
        pending_restore_slots=restored,
        _flexkv_uncached_restore=True,
        kv_committed_len=4,
        origin_input_ids=array("q", range(4)),
        output_ids=array("q"),
    )
    inner = MagicMock()
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache._restore_leases = {
        "request": SimpleNamespace(
            generation=3,
            req=req,
            device_indices=restored,
        )
    }

    cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=4)

    inner.cache_finished_req.assert_called_once_with(
        req, is_insert=False, kv_len_to_handle=4
    )
    assert not cache.has_uncommitted_restore(req)
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None
    assert req._flexkv_uncached_restore is False


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
    cache._restore_leases = {}
    cache._restore_generation = 0
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}

    cache.reset()

    assert calls.mock_calls[:2] == [
        call.connector.reset(),
        call.inner.reset(),
    ]


def test_reset_frees_uncommitted_restore_after_connector_drain():
    calls = MagicMock()
    connector = MagicMock()
    allocator = MagicMock()
    inner = MagicMock()
    calls.attach_mock(connector, "connector")
    calls.attach_mock(allocator, "allocator")
    calls.attach_mock(inner, "inner")
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    req = SimpleNamespace(
        rid="request",
        pending_restore_generation=0,
        pending_restore_slots=restored,
        _flexkv_uncached_restore=True,
    )

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.flexkv_connector = connector
    cache.token_to_kv_pool_allocator = allocator
    cache._inner_cache = inner
    cache._load_markers = {}
    cache._restore_leases = {
        "request": SimpleNamespace(
            generation=0,
            req=req,
            device_indices=restored,
        ),
    }
    cache._restore_generation = 1
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}

    cache.reset()

    assert calls.mock_calls[:3] == [
        call.connector.reset(),
        call.allocator.free(restored),
        call.inner.reset(),
    ]
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None
    assert req._flexkv_uncached_restore is False


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
