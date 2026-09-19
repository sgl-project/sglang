"""Regression test for LMCRadixCache.release_aborted_request.

A request aborted between ``match_prefix`` (which runs the LMCache LOOKUP and
records a load-back marker) and ``init_load_back`` (which runs RETRIEVE) never
reaches ``cache_finished_req``, so before the abort hook was implemented the
marker, the connector-side pending lookup and the daemon read locks all leaked.

The LMCache connector is faked here so the test stays CPU-only and does not
need the ``lmcache`` package; the two methods under test touch nothing but the
connector and the marker dict.

    python -m pytest test/registered/mem_cache/test_lmcache_release_aborted_request.py -v
"""

import importlib
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_LMC_MODULE = "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"


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


def _import_under_stubs():
    """Import the module against stubbed lmcache without poisoning sys.modules.

    The stub packages only live for the duration of the patch, but the module
    they initialize would outlive it with its lmcache symbols permanently bound
    to the stubs, making a later real-lmcache test in the same process fail
    depending on import order. So drop our own import from the module cache
    afterwards -- unless the module was already imported (against the real
    package, which the stub patch leaves untouched), in which case it is not
    ours to evict.
    """
    already_imported = _LMC_MODULE in sys.modules
    with _install_lmcache_stubs():
        module = importlib.import_module(_LMC_MODULE)
    if not already_imported:
        sys.modules.pop(_LMC_MODULE, None)
    return module.LMCacheMode, module.LMCRadixCache


LMCacheMode, LMCRadixCache = _import_under_stubs()


class FakeMPConnector:
    """Models the pending-lookup / read-lock bookkeeping of the real connector.

    Mirrors ``LMCacheMPConnector``: ``lookup_kv`` opens a session and takes read
    locks, ``release_pending`` drops only the locks, and ``end_session`` is the
    idempotent final cleanup that drops both.
    """

    def __init__(self, lookup_result: int):
        self.lookup_result = lookup_result
        self.pending_lookups = {}
        self.locks_held = {}
        self.end_session_calls = []

    def lookup_kv(self, token_ids, request_id):
        self.pending_lookups[request_id] = list(token_ids)
        self.locks_held[request_id] = True
        return self.lookup_result

    def release_pending(self, request_id):
        self.locks_held.pop(request_id, None)

    def end_session(self, request_id):
        self.end_session_calls.append(request_id)
        # Idempotent: the real adapter pops the pending lookup and returns
        # early when there is none.
        self.pending_lookups.pop(request_id, None)
        self.locks_held.pop(request_id, None)


def _make_cache(lookup_result: int, mode=LMCacheMode.MP):
    """Build only the state the two methods under test read.

    ``__init__`` needs CUDA, a model config and a live LMCache daemon, none of
    which these methods touch.
    """
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


class TestLMCacheReleaseAbortedRequest(CustomTestCase):
    def test_abort_after_match_prefix_releases_marker_and_session(self):
        cache = _make_cache(lookup_result=16)
        req = _make_req("rid-abort")

        res = _match(cache, req, list(range(16)), radix_hit=4)

        # LMCache had more tokens than radix: load-back is pending.
        self.assertEqual(res.host_hit_length, 12)
        self.assertIn(req.rid, cache._mp_load_back_markers)
        self.assertIn(req.rid, cache.lmcache_connector.pending_lookups)
        self.assertTrue(cache.lmcache_connector.locks_held[req.rid])

        # Abort before init_load_back ever runs.
        cache.release_aborted_request(CacheRequestHandle(rid=req.rid, attempt_id=0))

        self.assertNotIn(req.rid, cache._mp_load_back_markers)
        self.assertEqual(cache.lmcache_connector.pending_lookups, {})
        self.assertEqual(cache.lmcache_connector.locks_held, {})
        self.assertEqual(cache.lmcache_connector.end_session_calls, [req.rid])

    def test_abort_before_match_prefix_is_a_noop(self):
        cache = _make_cache(lookup_result=0)
        handle = CacheRequestHandle(rid="rid-never-matched", attempt_id=0)

        cache.release_aborted_request(handle)
        # Repeated cleanup must not raise either.
        cache.release_aborted_request(handle)

        self.assertEqual(cache._mp_load_back_markers, {})
        self.assertEqual(cache.lmcache_connector.pending_lookups, {})
        self.assertEqual(
            cache.lmcache_connector.end_session_calls,
            ["rid-never-matched", "rid-never-matched"],
        )

    def test_abort_after_lookup_miss_still_ends_session(self):
        # LMCache had nothing beyond radix: _mp_match_prefix already dropped the
        # locks but deliberately kept the session for end_session.
        cache = _make_cache(lookup_result=4)
        req = _make_req("rid-miss")

        res = _match(cache, req, list(range(16)), radix_hit=4)

        self.assertEqual(res.host_hit_length, 0)
        self.assertNotIn(req.rid, cache._mp_load_back_markers)
        self.assertEqual(cache.lmcache_connector.locks_held, {})
        self.assertIn(req.rid, cache.lmcache_connector.pending_lookups)

        cache.release_aborted_request(CacheRequestHandle(rid=req.rid, attempt_id=0))

        self.assertEqual(cache.lmcache_connector.pending_lookups, {})
        self.assertEqual(cache.lmcache_connector.end_session_calls, [req.rid])

    def test_ip_mode_is_untouched(self):
        # IP mode keeps no per-request markers and has no session concept.
        cache = _make_cache(lookup_result=16, mode=LMCacheMode.IP)

        cache.release_aborted_request(CacheRequestHandle(rid="rid-ip", attempt_id=0))

        self.assertEqual(cache.lmcache_connector.end_session_calls, [])


if __name__ == "__main__":
    unittest.main()
