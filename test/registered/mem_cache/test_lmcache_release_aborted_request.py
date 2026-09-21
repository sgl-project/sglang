"""Two-phase abort cleanup for LMCache / FlexKV (issue #40360).

``release_aborted_request`` conflated cancel and session-close. FlexKV must
cancel lookup/prefetch *before* ``cache_finished_req`` (an async STORE keeps
its source-node lock). LMCache must ``end_session`` *after* that optional
STORE, or immediately when no STORE runs.

Connectors are faked so the tests stay CPU-only and do not need ``lmcache`` or
``flexkv``.

    python -m pytest test/registered/mem_cache/test_lmcache_release_aborted_request.py -v
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
    MatchResult,
)
from sglang.srt.mem_cache.common import abort_prefix_cache_request, release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_LMC_MODULE = "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
_FKV_MODULE = "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"


def _install_lmcache_stubs():
    """Satisfy the hard ``import lmcache`` at lmc_radix_cache module scope."""
    mods = {}
    for name in (
        "lmcache",
        "lmcache.integration",
        "lmcache.integration.sglang",
        "lmcache.integration.sglang.multi_process_adapter",
        "lmcache.integration.sglang.sglang_adapter",
        "lmcache.integration.sglang.utils",
    ):
        mods[name] = types.ModuleType(name)
    mods["lmcache.integration.sglang.multi_process_adapter"].LMCacheMPConnector = object
    adapter = mods["lmcache.integration.sglang.sglang_adapter"]
    adapter.LMCacheLayerwiseConnector = object
    adapter.LoadMetadata = object
    adapter.StoreMetadata = object
    mods["lmcache.integration.sglang.utils"].lmcache_get_config = lambda _: None
    return mock.patch.dict(sys.modules, mods)


def _import_under_stubs(module_name, stub_cm):
    """Import against stubbed extras without poisoning a later real import."""
    already_imported = module_name in sys.modules
    with stub_cm():
        module = importlib.import_module(module_name)
    if not already_imported:
        sys.modules.pop(module_name, None)
    return module


LMCacheMode, LMCRadixCache = (lambda m: (m.LMCacheMode, m.LMCRadixCache))(
    _import_under_stubs(_LMC_MODULE, _install_lmcache_stubs)
)


class FakeMPConnector:
    """Models pending-lookup / read-lock bookkeeping of the real connector.

    Mirrors ``LMCacheMPConnector``: ``lookup_kv`` opens a session and takes
    read locks, ``release_pending`` drops only the locks, and ``end_session``
    is the final cleanup that drops both.
    """

    def __init__(self, lookup_result: int):
        self.lookup_result = lookup_result
        self.pending_lookups = {}
        self.locks_held = {}
        self.end_session_calls = []
        self.ops = []

    def lookup_kv(self, token_ids, request_id):
        self.ops.append(("lookup", request_id))
        self.pending_lookups[request_id] = list(token_ids)
        self.locks_held[request_id] = True
        return self.lookup_result

    def release_pending(self, request_id):
        self.ops.append(("release_pending", request_id))
        self.locks_held.pop(request_id, None)

    def store_kv(self, store_md):
        request_id = getattr(store_md, "request_id", store_md)
        self.ops.append(("store", request_id))

    def end_session(self, request_id):
        self.ops.append(("end_session", request_id))
        self.end_session_calls.append(request_id)
        self.pending_lookups.pop(request_id, None)
        self.locks_held.pop(request_id, None)


def _make_lmcache(lookup_result: int, mode=None):
    """Build only the state the methods under test read.

    ``__init__`` needs CUDA, a model config and a live LMCache daemon, none of
    which these methods touch.
    """
    if mode is None:
        mode = LMCacheMode.MP
    cache = object.__new__(LMCRadixCache)
    cache._mode = mode
    cache._mp_load_back_markers = {}
    cache.lmcache_connector = FakeMPConnector(lookup_result)
    return cache


def _make_req(rid: str):
    return types.SimpleNamespace(rid=rid)


def _match(cache, req, token_ids, radix_hit: int):
    """Drive _mp_match_prefix the way match_prefix does."""
    key = RadixKey(list(token_ids), None)
    value = torch.arange(radix_hit, dtype=torch.int64)
    base_res = MatchResult(
        device_indices=value,
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
    )
    return cache._mp_match_prefix(key, base_res, value, None, req)


class TestLMCacheTwoPhaseAbort(unittest.TestCase):
    def test_abort_after_match_prefix_without_store(self):
        cache = _make_lmcache(lookup_result=16)
        req = _make_req("rid-abort")
        handle = CacheRequestHandle(rid=req.rid, attempt_id=0)

        res = _match(cache, req, list(range(16)), radix_hit=4)

        self.assertEqual(res.host_hit_length, 12)
        self.assertIn(req.rid, cache._mp_load_back_markers)
        self.assertIn(req.rid, cache.lmcache_connector.pending_lookups)
        self.assertTrue(cache.lmcache_connector.locks_held[req.rid])

        cache.cancel_aborted_request_work(handle)
        self.assertNotIn(req.rid, cache._mp_load_back_markers)
        self.assertEqual(cache.lmcache_connector.end_session_calls, [])
        self.assertEqual(cache.lmcache_connector.locks_held, {})

        cache.finish_request_session(handle)
        self.assertEqual(cache.lmcache_connector.pending_lookups, {})
        self.assertEqual(cache.lmcache_connector.end_session_calls, [req.rid])

    def test_cancel_does_not_end_session(self):
        cache = _make_lmcache(lookup_result=16)
        req = _make_req("rid-cancel")
        handle = CacheRequestHandle(rid=req.rid, attempt_id=0)
        _match(cache, req, list(range(16)), radix_hit=4)

        cache.cancel_aborted_request_work(handle)

        self.assertEqual(cache.lmcache_connector.end_session_calls, [])
        self.assertNotIn("end_session", [op[0] for op in cache.lmcache_connector.ops])

    def test_abort_then_store_ends_session_after_store(self):
        cache = _make_lmcache(lookup_result=16)
        req = _make_req("rid-store")
        handle = CacheRequestHandle(rid=req.rid, attempt_id=0)
        _match(cache, req, list(range(16)), radix_hit=4)

        cache.cancel_aborted_request_work(handle)
        self.assertEqual(cache.lmcache_connector.end_session_calls, [])

        cache.lmcache_connector.store_kv(types.SimpleNamespace(request_id=req.rid))
        cache.finish_request_session(handle)

        self.assertEqual(
            [op[0] for op in cache.lmcache_connector.ops],
            ["lookup", "release_pending", "store", "end_session"],
        )

    def test_abort_before_match_prefix_is_a_noop(self):
        cache = _make_lmcache(lookup_result=0)
        handle = CacheRequestHandle(rid="rid-never-matched", attempt_id=0)

        cache.cancel_aborted_request_work(handle)
        cache.finish_request_session(handle)
        cache.cancel_aborted_request_work(handle)
        cache.finish_request_session(handle)

        self.assertEqual(cache._mp_load_back_markers, {})
        self.assertEqual(cache.lmcache_connector.pending_lookups, {})
        self.assertEqual(
            cache.lmcache_connector.end_session_calls,
            ["rid-never-matched", "rid-never-matched"],
        )

    def test_abort_after_lookup_miss_still_ends_session(self):
        cache = _make_lmcache(lookup_result=4)
        req = _make_req("rid-miss")
        handle = CacheRequestHandle(rid=req.rid, attempt_id=0)

        res = _match(cache, req, list(range(16)), radix_hit=4)

        self.assertEqual(res.host_hit_length, 0)
        self.assertNotIn(req.rid, cache._mp_load_back_markers)
        self.assertEqual(cache.lmcache_connector.locks_held, {})
        self.assertIn(req.rid, cache.lmcache_connector.pending_lookups)

        cache.cancel_aborted_request_work(handle)
        cache.finish_request_session(handle)

        self.assertEqual(cache.lmcache_connector.pending_lookups, {})
        self.assertEqual(cache.lmcache_connector.end_session_calls, [req.rid])

    def test_ip_mode_is_untouched(self):
        cache = _make_lmcache(lookup_result=16, mode=LMCacheMode.IP)
        handle = CacheRequestHandle(rid="rid-ip", attempt_id=0)

        cache.cancel_aborted_request_work(handle)
        cache.finish_request_session(handle)

        self.assertEqual(cache.lmcache_connector.end_session_calls, [])
        self.assertEqual(cache.lmcache_connector.ops, [])

    def test_finish_abort_does_not_end_session(self):
        cache = _make_lmcache(lookup_result=16)
        req = _make_req("rid-finish")
        handle = CacheRequestHandle(rid=req.rid, attempt_id=0)
        _match(cache, req, list(range(16)), radix_hit=4)

        cache.finish(handle, CacheRequestOutcome.ABORT)

        self.assertNotIn(req.rid, cache._mp_load_back_markers)
        self.assertEqual(cache.lmcache_connector.end_session_calls, [])


class _FakeKv:
    def __init__(self, holds_kv=False, holds_mamba=False):
        self.holds_kv = holds_kv
        self.holds_mamba = holds_mamba
        self.is_kv_released = not holds_kv


class _RecordingCache:
    """Records abort-phase ordering without a real radix tree."""

    def __init__(self):
        self.ops = []

    def finish(self, handle, outcome):
        assert outcome == CacheRequestOutcome.ABORT
        self.ops.append("cancel")

    def finish_request_session(self, handle):
        self.ops.append("end_session")

    def cache_finished_req(self, req, is_insert=True, **kwargs):
        self.ops.append("store" if is_insert else "no_insert")
        req.kv.holds_kv = False
        req.kv.is_kv_released = True

    def supports_mamba(self):
        return False


def _req_for_helper(rid="rid-helper", holds_kv=False, holds_mamba=False):
    return types.SimpleNamespace(
        rid=rid,
        cache_request_handle=CacheRequestHandle(rid=rid, attempt_id=0),
        kv=_FakeKv(holds_kv=holds_kv, holds_mamba=holds_mamba),
        owned_kv_len=lambda: 0,
        skip_radix_cache_insert=False,
    )


class TestAbortPrefixCacheRequest(unittest.TestCase):
    def test_waiting_queue_abort_closes_session_without_store(self):
        cache = _RecordingCache()
        abort_prefix_cache_request(_req_for_helper(), cache)
        self.assertEqual(cache.ops, ["cancel", "end_session"])

    def test_abort_then_store_never_ends_session_before_store(self):
        cache = _RecordingCache()
        abort_prefix_cache_request(
            _req_for_helper(holds_kv=True), cache, release_kv=True
        )
        self.assertEqual(cache.ops, ["cancel", "store", "end_session"])
        self.assertLess(cache.ops.index("store"), cache.ops.index("end_session"))

    def test_abort_with_is_insert_false_still_cancels_before_kv_finalize(self):
        cache = _RecordingCache()
        abort_prefix_cache_request(
            _req_for_helper(holds_kv=True),
            cache,
            release_kv=True,
            is_insert=False,
        )
        self.assertEqual(cache.ops, ["cancel", "no_insert", "end_session"])

    def test_release_kv_cache_closes_session_after_store(self):
        cache = _RecordingCache()
        req = _req_for_helper(holds_kv=True)
        release_kv_cache(req, cache)
        self.assertEqual(cache.ops, ["store", "end_session"])


def _install_flexkv_stubs():
    mods = {}
    for name in (
        "flexkv",
        "flexkv.common",
        "flexkv.common.request",
        "flexkv.common.storage",
        "flexkv.integration",
        "flexkv.integration.config",
        "flexkv.kvmanager",
        "flexkv.server",
        "flexkv.server.client",
        "flexkv.transfer",
        "flexkv.transfer.layerwise",
        "flexkv.transfer_manager",
    ):
        mods[name] = types.ModuleType(name)
    mods["flexkv.common.request"].KVResponseStatus = object
    mods["flexkv.common.storage"].KVCacheLayout = object
    mods["flexkv.common.storage"].KVCacheLayoutType = object
    mods["flexkv.integration.config"].FlexKVConfig = object
    mods["flexkv.kvmanager"].KVManager = object
    mods["flexkv.server.client"].KVTPClient = object
    mods["flexkv.transfer.layerwise"].build_layerwise_eventfd_socket_path = (
        lambda *a, **k: ""
    )
    mods["flexkv.transfer_manager"].TransferManagerOnRemote = object
    return mock.patch.dict(sys.modules, mods)


class FakeFlexKVConnector:
    def __init__(self):
        self.released = []
        self.cancelled_prefetch = []

    def release_pending(self, handle):
        self.released.append(handle)

    def cancel_prefetch(self, handle):
        self.cancelled_prefetch.append(handle)


class TestFlexKVCancelBeforeStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = _import_under_stubs(_FKV_MODULE, _install_flexkv_stubs)
        cls.FlexKVRadixCache = module.FlexKVRadixCache

    def _make_cache(self):
        cache = object.__new__(self.FlexKVRadixCache)
        cache._load_markers = {}
        cache._inflight_store_nodes = {}
        cache._node_lock = threading.Lock()
        cache._lock_refs = {}
        cache.inc_lock_ref = lambda node: cache._lock_refs.__setitem__(
            id(node), cache._lock_refs.get(id(node), 0) + 1
        )
        cache.dec_lock_ref = lambda node, params=None: cache._lock_refs.__setitem__(
            id(node), cache._lock_refs.get(id(node), 0) - 1
        )
        cache.flexkv_connector = FakeFlexKVConnector()
        return cache

    def test_cancel_before_store_keeps_inflight_lock(self):
        cache = self._make_cache()
        handle = CacheRequestHandle(rid="rid-fkv", attempt_id=0)
        node = object()

        cache.cancel_aborted_request_work(handle)
        cache.inc_lock_ref(node)
        with cache._node_lock:
            cache._inflight_store_nodes[handle] = node

        self.assertEqual(cache._lock_refs[id(node)], 1)
        self.assertIs(cache._inflight_store_nodes[handle], node)
        self.assertEqual(cache.flexkv_connector.released, [handle])
        self.assertEqual(cache.flexkv_connector.cancelled_prefetch, [handle])

    def test_cancel_after_store_would_drop_inflight_lock(self):
        """Documents why FlexKV cancel must run before cache_finished_req."""
        cache = self._make_cache()
        handle = CacheRequestHandle(rid="rid-fkv-after", attempt_id=0)
        node = object()

        cache.inc_lock_ref(node)
        with cache._node_lock:
            cache._inflight_store_nodes[handle] = node
        cache.cancel_aborted_request_work(handle)

        self.assertEqual(cache._lock_refs[id(node)], 0)
        self.assertEqual(cache._inflight_store_nodes, {})


if __name__ == "__main__":
    unittest.main()
