import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAMBA_SUFFIXES = ("_temporal", "_conv_0")
PAGES = ("p0", "p1", "p2", "p3")


class _LegacyBinding:
    def __init__(self, missing=(), evicted_after_probe=()):
        self.missing = set(missing)
        self.evicted_after_probe = set(evicted_after_probe)
        self.calls = []

    def _exists(self, keys):
        return [0 if key in self.missing else 1 for key in keys]

    def batch_is_exist(self, keys):
        self.calls.append(("batch_is_exist", list(keys)))
        self.missing |= self.evicted_after_probe
        return self._exists(keys)


class _ProbeBinding(_LegacyBinding):
    def batch_probe_key(self, keys):
        self.calls.append(("batch_probe_key", list(keys)))
        return self._exists(keys)


def _component_keys(page_keys, transfer):
    if transfer.name == PoolName.MAMBA:
        return [p + s for p in page_keys for s in MAMBA_SUFFIXES], len(MAMBA_SUFFIXES)
    return [f"{p}_{PoolName(transfer.name).value}" for p in page_keys], 1


def _make_store(binding, kv_pages):
    store = MooncakeStore.__new__(MooncakeStore)
    store.store = binding
    store.config_prefix = None
    store.enable_storage_metrics = False
    store.mem_pool_host = SimpleNamespace(kv_buffer=object())
    store.batch_exists = lambda keys, extra_info=None: kv_pages
    store._get_hybrid_page_component_keys = _component_keys
    return store


def _transfers():
    trailing = PoolHitPolicy.TRAILING_PAGES
    return [
        PoolTransfer(name=PoolName.SWA, keys=["w0", "w1"], hit_policy=trailing),
        PoolTransfer(name=PoolName.MAMBA, keys=["m0"], hit_policy=trailing),
        PoolTransfer(name=PoolName.INDEXER, keys=["i0"]),
    ]


def _mamba(*pages):
    return [p + s for p in pages for s in MAMBA_SUFFIXES]


class TestMooncakeTrailingWindowLease(CustomTestCase):
    def test_only_the_restore_windows_are_leased(self):
        binding = _ProbeBinding(missing={"p3_conv_0"})
        store = _make_store(binding, kv_pages=4)

        result = store.batch_exists_v2(list(PAGES) + ["p4"], _transfers())

        self.assertEqual(
            binding.calls,
            [
                ("batch_probe_key", [f"{p}_swa" for p in PAGES]),
                ("batch_probe_key", _mamba(*PAGES)),
                ("batch_is_exist", [f"{p}_indexer" for p in PAGES + ("p4",)]),
                ("batch_is_exist", ["p1_swa", "p2_swa", *_mamba("p2")]),
            ],
        )
        self.assertEqual(result.restorable_prefix_pages, [1, 2, 3])
        self.assertEqual(result.kv_hit_pages, 3)

    def test_window_evicted_after_probe_is_looked_up_again(self):
        binding = _ProbeBinding(
            missing={"p3_conv_0"}, evicted_after_probe={"p2_temporal"}
        )
        store = _make_store(binding, kv_pages=4)

        result = store.batch_exists_v2(list(PAGES), _transfers())

        self.assertEqual(binding.calls[-1][0], "batch_is_exist")
        self.assertEqual(len(binding.calls[-1][1]), 4 * 4)
        self.assertEqual(result.restorable_prefix_pages, [1, 2])

    def test_legacy_binding_leases_every_page(self):
        binding = _LegacyBinding(missing={"p3_conv_0"})
        store = _make_store(binding, kv_pages=4)

        result = store.batch_exists_v2(list(PAGES), _transfers())

        self.assertEqual([call[0] for call in binding.calls], ["batch_is_exist"])
        self.assertEqual(result.restorable_prefix_pages, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
