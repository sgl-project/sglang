"""Unit tests for MooncakeBaseStore request-context propagation and MooncakeStore
read/write-path wrapping introduced by augusto.yjh since dae126d5.

These cover the request_context mechanism on MooncakeBaseStore (set/clear ordering,
finally-on-raise, store-is-None / legacy-wheel no-op), the `_request_context_from_extra_info`
extraction helper, and the asymmetry that the *read/query* store RPCs
(`batch_exists`, `batch_exists_v2`, `batch_get_v1`, `batch_get_v2`) wrap RPC calls in
`request_context(...)` while the *write* paths do not.

Usage:
    python3 -m pytest test/registered/unit/mem_cache/test_mooncake_store_request_context_unit.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeBaseStore,
    MooncakeStore,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --------------------------------------------------------------------------- #
# Fake MooncakeDistributedStore bindings.
#
# Important: production checks `hasattr(type(self.store), "set_request_context")`.
# A bare MagicMock reports False here (auto-created attrs live on the *instance*),
# so the supported branch must use a real shim class that *defines* the methods,
# and the legacy branch must use a class that deliberately omits them.
# --------------------------------------------------------------------------- #
class _SupportedBinding:
    """Fake MooncakeDistributedStore WITH request-context support."""

    def __init__(self, exist=1):
        # Ordered event log: ("event", payload). Lets tests assert call ORDER.
        self.event_log = []
        self._exist = exist  # 1 -> exists / read hit, 0 -> missing

    def set_request_context(
        self, request_id=None, trace_id=None, span_id=None, parent_span_id=None
    ):
        self.event_log.append(
            (
                "set_request_context",
                {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                },
            )
        )

    def clear_request_context(self):
        self.event_log.append(("clear_request_context", None))

    def batch_is_exist(self, keys, *a, **k):
        self.event_log.append(("batch_is_exist", list(keys)))
        return [self._exist] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes, *a, **k):
        self.event_log.append(("batch_get_into", list(keys)))
        return [self._exist] * len(keys)

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes, *a, **k):
        self.event_log.append(("batch_get_into_multi_buffers", list(keys)))
        return [self._exist] * len(keys)

    def batch_put_from(self, keys, ptrs, sizes, *a, **k):
        self.event_log.append(("batch_put_from", list(keys)))
        return [0] * len(keys)  # 0 -> success on the put path

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *a, **k):
        self.event_log.append(("batch_put_from_multi_buffers", list(keys)))
        return [0] * len(keys)

    def remove_all(self, *a, **k):
        self.event_log.append(("remove_all", None))


class _LegacyBinding:
    """Fake MooncakeDistributedStore WITHOUT request-context support (old wheel)."""

    def __init__(self, exist=1):
        self.event_log = []
        self._exist = exist

    # NOTE: deliberately no set_request_context / clear_request_context.

    def batch_is_exist(self, keys, *a, **k):
        self.event_log.append(("batch_is_exist", list(keys)))
        return [self._exist] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes, *a, **k):
        self.event_log.append(("batch_get_into", list(keys)))
        return [self._exist] * len(keys)

    def batch_put_from(self, keys, ptrs, sizes, *a, **k):
        self.event_log.append(("batch_put_from", list(keys)))
        return [0] * len(keys)


def _names(event_log):
    return [e[0] for e in event_log]


def _make_base_store(binding):
    st = MooncakeBaseStore.__new__(MooncakeBaseStore)
    st.store = binding
    return st


def _make_mooncake_store(binding):
    st = MooncakeStore.__new__(MooncakeStore)
    st.store = binding
    # Minimal config read by batch_exists / _tag_keys / _batch_preprocess etc.
    st.config_prefix = None
    st.is_mla_backend = False
    st.should_split_heads = False
    st.mha_suffix = "r0"
    st.split_factor = 0
    st._use_group_semantics = False
    st.enable_storage_metrics = False
    return st


class TestRequestContextMechanism(unittest.TestCase):
    def test_request_context_sets_then_clears(self):
        binding = _SupportedBinding()
        store = _make_base_store(binding)

        with store.request_context(request_id="r-1", trace_id="t-1"):
            pass

        self.assertEqual(
            _names(binding.event_log),
            ["set_request_context", "clear_request_context"],
        )
        self.assertEqual(
            binding.event_log[0][1],
            {
                "request_id": "r-1",
                "trace_id": "t-1",
                "span_id": None,
                "parent_span_id": None,
            },
        )

    def test_request_context_clears_even_when_body_raises(self):
        binding = _SupportedBinding()
        store = _make_base_store(binding)

        with self.assertRaises(RuntimeError):
            with store.request_context(request_id="r-err"):
                raise RuntimeError("boom")

        # finally semantics: clear must run even on the failing path
        self.assertEqual(
            _names(binding.event_log), ["set_request_context", "clear_request_context"]
        )

    def test_request_context_noop_when_store_is_none(self):
        store = _make_base_store(None)

        # Must not raise and must be a true no-op.
        with store.request_context(request_id="r"):
            pass
        store.set_request_context(request_id="r")
        store.clear_request_context()

    def test_request_context_noop_on_legacy_wheel(self):
        binding = _LegacyBinding()
        store = _make_base_store(binding)

        # Legacy wheel has no set/clear_request_context → guarded no-op, no raise.
        with store.request_context(request_id="r"):
            pass
        store.set_request_context(request_id="r")
        store.clear_request_context()
        # No request-context events were emitted by the binding (it has none).
        self.assertEqual(binding.event_log, [])

    def test_has_request_context_support_reflects_binding(self):
        supported = _make_base_store(_SupportedBinding())
        legacy = _make_base_store(_LegacyBinding())

        self.assertTrue(supported._has_request_context_support)
        self.assertFalse(legacy._has_request_context_support)
        # cached_property caches its result on the instance __dict__.
        self.assertTrue(supported.__dict__.get("_has_request_context_support"))
        self.assertIs(
            supported._has_request_context_support,
            supported.__dict__["_has_request_context_support"],
        )
        self.assertIs(
            legacy._has_request_context_support,
            legacy.__dict__["_has_request_context_support"],
        )


class TestRequestContextFromExtraInfo(unittest.TestCase):
    def test_request_context_from_extra_info_extraction(self):
        extract = MooncakeStore._request_context_from_extra_info

        # None extra_info / non-dict extra_info -> empty kwargs
        self.assertEqual(extract(None), {})
        self.assertEqual(extract(HiCacheStorageExtraInfo(extra_info=None)), {})
        self.assertEqual(extract(HiCacheStorageExtraInfo(extra_info="oops")), {})
        self.assertEqual(extract(HiCacheStorageExtraInfo(prefix_keys=["p"])), {})

        # Only non-None fields are carried through.
        self.assertEqual(
            extract(HiCacheStorageExtraInfo(extra_info={"request_id": "r-1"})),
            {"request_id": "r-1"},
        )
        self.assertEqual(
            extract(
                HiCacheStorageExtraInfo(
                    extra_info={
                        "request_id": "r-1",
                        "trace_id": None,
                        "span_id": None,
                        "parent_span_id": None,
                    }
                )
            ),
            {"request_id": "r-1"},
        )
        self.assertEqual(
            extract(
                HiCacheStorageExtraInfo(
                    extra_info={
                        "request_id": "r-1",
                        "trace_id": "t-1",
                        "span_id": "s-1",
                        "parent_span_id": "ps-1",
                    }
                )
            ),
            {
                "request_id": "r-1",
                "trace_id": "t-1",
                "span_id": "s-1",
                "parent_span_id": "ps-1",
            },
        )


class TestMooncakeStoreReadPathWrapping(unittest.TestCase):
    def test_batch_exists_sets_request_context_from_extra_info(self):
        binding = _SupportedBinding(exist=1)
        store = _make_mooncake_store(binding)

        ret = store.batch_exists(
            ["page0"], HiCacheStorageExtraInfo(extra_info={"request_id": "r-1"})
        )

        # query_keys == ["page0_r0_k", "page0_r0_v"], both exist -> 1 page.
        self.assertEqual(ret, 1)
        self.assertEqual(
            _names(binding.event_log),
            ["set_request_context", "batch_is_exist", "clear_request_context"],
        )
        self.assertEqual(binding.event_log[0][1]["request_id"], "r-1")
        self.assertEqual(binding.event_log[1][1], ["page0_r0_k", "page0_r0_v"])

    def test_batch_get_v1_wraps_get_with_request_context(self):
        binding = _SupportedBinding(exist=1)
        store = _make_mooncake_store(binding)
        # Bypass the buffer-meta plumbing so the test stays focused on wrapping.
        store.mem_pool_host = SimpleNamespace(kv_buffer=object(), page_size=1)
        store._batch_preprocess = lambda keys, hi: (
            list(keys),
            [0] * len(keys),
            [4096] * len(keys),
        )
        store._batch_postprocess = (
            lambda results, is_set_operate=False, key_multiplier=None: results
        )

        store.batch_get_v1(
            ["p0", "p1"],
            host_indices=object(),
            extra_info=HiCacheStorageExtraInfo(extra_info={"request_id": "g-7"}),
        )

        self.assertEqual(
            _names(binding.event_log),
            ["set_request_context", "batch_get_into", "clear_request_context"],
        )
        self.assertEqual(binding.event_log[0][1]["request_id"], "g-7")

    def test_batch_get_v2_wraps_get_but_batch_set_v2_does_not(self):
        transfer = PoolTransfer(
            name=PoolName.KV, host_indices=[0, 1], keys=["k0", "k1"]
        )

        # Read (get) path wraps.
        get_binding = _SupportedBinding(exist=1)
        get_store = _make_mooncake_store(get_binding)
        host_pool = SimpleNamespace(
            page_size=1,
            get_page_buffer_meta=lambda hi: ([0x1, 0x2], [4096, 4096]),
        )
        get_store.registered_pools = {PoolName.KV: host_pool}
        get_store._get_hybrid_page_component_keys = lambda keys, t: (list(keys), 1)

        get_store.batch_get_v2(
            [transfer], HiCacheStorageExtraInfo(extra_info={"request_id": "g-9"})
        )

        self.assertEqual(
            _names(get_binding.event_log),
            ["set_request_context", "batch_get_into", "clear_request_context"],
        )
        self.assertEqual(get_binding.event_log[0][1]["request_id"], "g-9")

        # Write (set) path with the same plumbing must NOT touch request_context,
        # even when it actually issues a put RPC (exist=0 -> missing -> put).
        set_binding = _SupportedBinding(exist=0)
        set_store = _make_mooncake_store(set_binding)
        set_store.registered_pools = {PoolName.KV: host_pool}
        set_store._get_hybrid_page_component_keys = lambda keys, t: (list(keys), 1)

        set_store.batch_set_v2(
            [transfer], HiCacheStorageExtraInfo(extra_info={"request_id": "s-9"})
        )

        # No set/clear anywhere; put RPC was issued.
        self.assertNotIn("set_request_context", _names(set_binding.event_log))
        self.assertNotIn("clear_request_context", _names(set_binding.event_log))
        self.assertIn("batch_put_from", _names(set_binding.event_log))

    def test_batch_exists_v2_wraps_sidecar_exist(self):
        binding = _SupportedBinding(exist=1)
        store = _make_mooncake_store(binding)
        # Logical KV anchor: kv_buffer is None -> kv_pages == len(keys), no KV RPC.
        store.mem_pool_host = SimpleNamespace(kv_buffer=None)
        store._get_hybrid_page_component_keys = lambda keys, t: ([f"{keys[0]}_c"], 1)

        result = store.batch_exists_v2(
            ["page0"],
            pool_transfers=[PoolTransfer(name=PoolName.SWA, keys=["page0"])],
            extra_info=HiCacheStorageExtraInfo(extra_info={"request_id": "e-10"}),
        )

        # Per sidecar: set -> batch_is_exist -> clear.
        self.assertEqual(
            _names(binding.event_log),
            ["set_request_context", "batch_is_exist", "clear_request_context"],
        )
        self.assertEqual(binding.event_log[0][1]["request_id"], "e-10")
        self.assertEqual(result.kv_hit_pages, 1)
        # kv_pages was truthy, so the KV key is included alongside the sidecar.
        self.assertEqual(result.extra_pool_hit_pages, {PoolName.KV: 1, PoolName.SWA: 1})


class TestMooncakeStoreWritePathNoContext(unittest.TestCase):
    def test_write_paths_never_touch_request_context(self):
        # `set(...)` and `batch_set(...)` drive `_put_batch_zero_copy_impl`; neither
        # read method's `with request_context(...)` is present on the write paths.
        binding = _SupportedBinding(exist=0)  # missing -> put RPC actually runs

        # set(): single key (scalar target_location so the simple
        # batch_put_from path is taken, not the multi-buffer path).
        store_set = _make_mooncake_store(binding)
        store_set.set("k", target_location=0x10, target_sizes=4096)
        self.assertNotIn("set_request_context", _names(binding.event_log))
        self.assertNotIn("clear_request_context", _names(binding.event_log))
        self.assertIn("batch_put_from", _names(binding.event_log))

        # batch_set(): multiple keys.
        binding.event_log.clear()
        store_batch = _make_mooncake_store(binding)
        store_batch.batch_set(
            ["a", "b"], target_locations=[0x20, 0x30], target_sizes=[4096, 4096]
        )
        self.assertNotIn("set_request_context", _names(binding.event_log))
        self.assertNotIn("clear_request_context", _names(binding.event_log))
        self.assertIn("batch_put_from", _names(binding.event_log))

        # batch_set_v1(): piggybacks on the same non-wrapping machinery.
        binding.event_log.clear()
        store_v1 = _make_mooncake_store(binding)
        store_v1.mem_pool_host = SimpleNamespace(kv_buffer=object(), page_size=1)
        store_v1._batch_preprocess = lambda keys, hi: (
            list(keys),
            [0] * len(keys),
            [4096] * len(keys),
        )
        store_v1._batch_exist = lambda keys: [0] * len(keys)
        store_v1._put_batch_zero_copy_impl = lambda keys, ptrs, sizes, group_ids=None: (
            [0] * len(keys)
        )
        store_v1._batch_postprocess = (
            lambda results, is_set_operate=False, key_multiplier=None: results
        )
        store_v1.batch_set_v1(
            ["k0", "k1"],
            host_indices=object(),
            extra_info=HiCacheStorageExtraInfo(extra_info={"request_id": "ignored"}),
        )
        self.assertNotIn("set_request_context", _names(binding.event_log))
        self.assertNotIn("clear_request_context", _names(binding.event_log))

    def test_batch_exists_with_none_extra_info_still_clears_context(self):
        binding = _SupportedBinding(exist=1)
        store = _make_mooncake_store(binding)

        store.batch_exists(["page0"], None)

        # None extra_info -> set_request_context called with all-None kwargs,
        # then cleared symmetrically (does not leave stale per-thread context).
        self.assertEqual(
            _names(binding.event_log),
            ["set_request_context", "batch_is_exist", "clear_request_context"],
        )
        self.assertEqual(
            binding.event_log[0][1],
            {
                "request_id": None,
                "trace_id": None,
                "span_id": None,
                "parent_span_id": None,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
