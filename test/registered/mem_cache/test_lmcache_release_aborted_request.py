"""Two-phase abort cleanup for LMCache / FlexKV (issue #40360).

``release_aborted_request`` conflated cancel and session-close. FlexKV must
cancel lookup/prefetch *before* KV finalization / STORE (an async STORE keeps
its source-node lock). LMCache must ``finish_request`` *after* that optional
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
from unittest import mock

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.mem_cache.common import abort_prefix_cache_request, release_kv_cache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_LMC_MODULE = "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache"
_FKV_MODULE = "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"


def _install_lmcache_stubs():
    """Satisfy the hard ``import lmcache`` at lmcache_unified_radix_cache import."""
    mods = {}
    for name in (
        "lmcache",
        "lmcache.integration",
        "lmcache.integration.sglang",
        "lmcache.integration.sglang.lmcache_mp_metadata",
        "lmcache.integration.sglang.unified_lmcache_mp_connector",
    ):
        mods[name] = types.ModuleType(name)

    class _LMCacheExternalFlow:
        def __init__(self, key=None, lookup=None, **kwargs):
            self.key = key
            self.lookup = lookup
            self.total_hit = kwargs.get("total_hit")
            self.local_hit_tokens = kwargs.get("local_hit_tokens")
            self.load = kwargs.get("load")
            self.cancelled = False
            self.mamba_value = None
            self.load_req = None
            self.retire_requested = False
            self.load_completed = False
            self.prefix_published = False
            self.free_mamba_after_load = False

    mods[
        "lmcache.integration.sglang.lmcache_mp_metadata"
    ].LMCacheExternalFlow = _LMCacheExternalFlow
    mods["lmcache.integration.sglang.lmcache_mp_metadata"].LMCachePendingStore = object
    mods[
        "lmcache.integration.sglang.unified_lmcache_mp_connector"
    ].UnifiedLMCacheMPConnector = object
    return mock.patch.dict(sys.modules, mods)


def _import_under_stubs(module_name, stub_cm):
    """Import against stubbed extras without poisoning a later real import."""
    already_imported = module_name in sys.modules
    with stub_cm():
        module = importlib.import_module(module_name)
    if not already_imported:
        sys.modules.pop(module_name, None)
    return module


LMCacheUnifiedRadixCache = _import_under_stubs(
    _LMC_MODULE, _install_lmcache_stubs
).LMCacheUnifiedRadixCache


class FakeMPConnector:
    """Models session finish bookkeeping of the unified LMCache connector."""

    def __init__(self):
        self.finish_request_calls = []
        self.ops = []
        self.freed_locks = []

    def finish_request(self, request_id):
        self.ops.append(("finish_request", request_id))
        self.finish_request_calls.append(request_id)

    def free_lookup_locks(self, *args, **kwargs):
        self.ops.append(("free_lookup_locks", args, kwargs))
        self.freed_locks.append((args, kwargs))


class FakeFlow:
    def __init__(self, *, total_hit=None, load=None):
        self.key = None
        self.lookup = types.SimpleNamespace(request_id="unused", lock_start=0)
        self.total_hit = total_hit
        self.local_hit_tokens = None
        self.load = load
        self.cancelled = False
        self.mamba_value = None
        self.load_req = None
        self.retire_requested = False
        self.load_completed = False
        self.prefix_published = False
        self.free_mamba_after_load = False


def _make_lmcache():
    """Build only the state the methods under test read.

    ``__init__`` needs CUDA, a model config and a live LMCache daemon, none of
    which these methods touch.
    """
    cache = object.__new__(LMCacheUnifiedRadixCache)
    cache._external_flows = {}
    cache._pending_stores = []
    cache._pending_store_counts = {}
    cache._session_finish_requested = set()
    cache._open_sessions = set()
    cache._sessions_finished = set()
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache.lmcache_connector = FakeMPConnector()
    cache._retire_loaded_flow = mock.Mock()
    cache._finish_failed_load = mock.Mock()
    return cache


def _open_lookup(cache, rid: str, *, total_hit=None):
    cache._open_sessions.add(rid)
    cache._external_flows[rid] = FakeFlow(total_hit=total_hit)


class TestLMCacheTwoPhaseAbort(CustomTestCase):
    def test_abort_after_lookup_without_store(self):
        cache = _make_lmcache()
        rid = "rid-abort"
        handle = CacheRequestHandle(rid=rid, attempt_id=0)
        _open_lookup(cache, rid)

        cache.cancel_aborted_request_work(handle)
        self.assertNotIn(rid, cache._external_flows)
        self.assertEqual(cache.lmcache_connector.finish_request_calls, [])
        self.assertIn(rid, cache._open_sessions)

        cache.finish_request_session(handle)
        self.assertEqual(cache.lmcache_connector.finish_request_calls, [rid])
        self.assertNotIn(rid, cache._open_sessions)

    def test_cancel_does_not_end_session(self):
        cache = _make_lmcache()
        rid = "rid-cancel"
        handle = CacheRequestHandle(rid=rid, attempt_id=0)
        _open_lookup(cache, rid, total_hit=12)

        cache.cancel_aborted_request_work(handle)

        self.assertEqual(cache.lmcache_connector.finish_request_calls, [])
        self.assertNotIn(
            "finish_request", [op[0] for op in cache.lmcache_connector.ops]
        )
        cache._retire_loaded_flow.assert_called_once_with(rid)

    def test_finish_request_session_is_idempotent(self):
        """Bootstrap abort can finalize twice; finish_request must fire once."""
        cache = _make_lmcache()
        rid = "rid-idem"
        handle = CacheRequestHandle(rid=rid, attempt_id=0)
        _open_lookup(cache, rid)

        cache.cancel_aborted_request_work(handle)
        cache.finish_request_session(handle)
        cache.finish_request_session(handle)

        self.assertEqual(cache.lmcache_connector.finish_request_calls, [rid])
        self.assertEqual(
            [op[0] for op in cache.lmcache_connector.ops].count("finish_request"), 1
        )

    def test_production_insert_req_stores_before_finish_request(self):
        """Drive real LMCacheUnifiedRadixCache.insert_req (CPU stubs, no GPU)."""
        cache = _make_lmcache()
        rid = "rid-prod-store"
        handle = CacheRequestHandle(rid=rid, attempt_id=0)
        _open_lookup(cache, rid, total_hit=4)

        cache.cancel_aborted_request_work(handle)
        self.assertEqual(cache.lmcache_connector.finish_request_calls, [])

        req = types.SimpleNamespace(
            rid=rid,
            origin_input_ids=list(range(4)),
            output_ids=[],
            cache_request_handle=handle,
        )
        cache._publish_external_loaded_prefix = mock.Mock()
        cache._submit_store = mock.Mock(
            side_effect=lambda *_a, **_k: cache.lmcache_connector.ops.append(
                ("store", rid)
            )
        )

        with mock.patch.object(UnifiedRadixCache, "insert_req", return_value=None):
            cache.insert_req(req, up_to=4)

        op_names = [op[0] for op in cache.lmcache_connector.ops]
        self.assertIn("store", op_names)
        self.assertIn("finish_request", op_names)
        self.assertLess(op_names.index("store"), op_names.index("finish_request"))
        self.assertEqual(cache.lmcache_connector.finish_request_calls, [rid])

    def test_abort_before_lookup_is_a_noop(self):
        cache = _make_lmcache()
        handle = CacheRequestHandle(rid="rid-never-matched", attempt_id=0)

        cache.cancel_aborted_request_work(handle)
        cache.finish_request_session(handle)
        cache.cancel_aborted_request_work(handle)
        cache.finish_request_session(handle)

        self.assertEqual(cache._external_flows, {})
        self.assertEqual(cache.lmcache_connector.finish_request_calls, [])

    def test_finish_abort_does_not_end_session(self):
        cache = _make_lmcache()
        rid = "rid-finish"
        handle = CacheRequestHandle(rid=rid, attempt_id=0)
        _open_lookup(cache, rid)

        cache.finish(handle, CacheRequestOutcome.ABORT)

        self.assertNotIn(rid, cache._external_flows)
        self.assertEqual(cache.lmcache_connector.finish_request_calls, [])
        self.assertIn(rid, cache._open_sessions)


class _FakeKv:
    def __init__(self, holds_kv=False, holds_mamba=False):
        self.holds_kv = holds_kv
        self.holds_mamba = holds_mamba
        self.is_kv_released = not holds_kv
        self.cache_protected_len = 0
        self.kv_allocated_len = 4 if holds_kv else 0
        self.kv_committed_len = 4 if holds_kv else 0

    def mark_kv_released(self):
        self.holds_kv = False
        self.is_kv_released = True


class _RecordingCache:
    """Records abort-phase ordering without a real radix tree."""

    def __init__(self):
        self.ops = []
        self.req_to_token_pool = types.SimpleNamespace(free=lambda req: None)
        self.token_to_kv_pool_allocator = types.SimpleNamespace(page_size=1)

    def finish(self, handle, outcome):
        assert outcome == CacheRequestOutcome.ABORT
        self.ops.append("cancel")

    def finish_request_session(self, handle):
        self.ops.append("end_session")

    def claim_kv_row(self, req):
        return False

    def insert_req(self, req, *, up_to, **kwargs):
        self.ops.append("store")

    def free_kv_row(self, kv, ranges):
        pass

    def unpin(self, req):
        pass

    def on_release(self, req, *, inserted):
        if not inserted:
            self.ops.append("no_insert")

    def supports_mamba(self):
        return False


def _req_for_helper(rid="rid-helper", holds_kv=False, holds_mamba=False):
    return types.SimpleNamespace(
        rid=rid,
        cache_request_handle=CacheRequestHandle(rid=rid, attempt_id=0),
        kv=_FakeKv(holds_kv=holds_kv, holds_mamba=holds_mamba),
        owned_kv_len=lambda: 4 if holds_kv else 0,
        skip_radix_cache_insert=False,
    )


class TestAbortPrefixCacheRequest(CustomTestCase):
    def setUp(self):
        spec = types.SimpleNamespace(speculative_algorithm=None)
        serving = types.SimpleNamespace(strip_thinking_cache=False)
        self._ctx = mock.patch.multiple(
            "sglang.srt.mem_cache.common",
            get_spec=lambda: spec,
            get_serving=lambda: serving,
        )
        self._ctx.start()

    def tearDown(self):
        self._ctx.stop()

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


class TestFlexKVCancelBeforeStore(CustomTestCase):
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
        """Documents why FlexKV cancel must run before KV finalization / STORE."""
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
    import unittest

    unittest.main()
