import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAMBA_SUFFIXES = ("_temporal", "_conv_0")


class _LegacyBinding:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def _exists(self, keys):
        return [0 if key in self.missing else 1 for key in keys]

    def batch_is_exist(self, keys):
        self.calls.append(("batch_is_exist", list(keys)))
        return self._exists(keys)


class _ProbeBinding(_LegacyBinding):
    def batch_probe_key(self, keys, policy="none", candidate_size=1):
        self.calls.append(("batch_probe_key", list(keys), policy, candidate_size))
        return self._exists(keys)


def _component_keys(page_keys, transfer):
    if transfer.name == PoolName.MAMBA:
        return [p + s for p in page_keys for s in MAMBA_SUFFIXES], len(MAMBA_SUFFIXES)
    return [f"{p}_swa" for p in page_keys], 1


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
    return [
        PoolTransfer(name=PoolName.SWA, keys=["p2"]),
        PoolTransfer(name=PoolName.MAMBA, keys=["p2"]),
    ]


class TestMooncakeMambaLookupLease(CustomTestCase):
    def test_mamba_pages_are_probed_with_last_hit_lease(self):
        binding = _ProbeBinding(missing={"p2_conv_0"})
        store = _make_store(binding, kv_pages=3)

        result = store.batch_exists_v2(["p0", "p1", "p2", "p3"], _transfers())

        self.assertEqual(
            binding.calls,
            [
                (
                    "batch_probe_key",
                    [p + s for p in ("p0", "p1", "p2") for s in MAMBA_SUFFIXES],
                    "LastHitOnly",
                    2,
                ),
                ("batch_is_exist", ["p0_swa", "p1_swa", "p2_swa", "p3_swa"]),
            ],
        )
        self.assertEqual(result.restorable_prefix_pages, [1, 2])
        self.assertEqual(result.kv_hit_pages, 2)

    def test_legacy_binding_leases_every_page(self):
        binding = _LegacyBinding(missing={"p2_conv_0"})
        store = _make_store(binding, kv_pages=3)

        result = store.batch_exists_v2(["p0", "p1", "p2", "p3"], _transfers())

        self.assertEqual([call[0] for call in binding.calls], ["batch_is_exist"])
        self.assertEqual(result.restorable_prefix_pages, [1, 2])
        self.assertEqual(result.kv_hit_pages, 2)


if __name__ == "__main__":
    unittest.main()
