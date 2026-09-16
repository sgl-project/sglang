"""Unit tests for per-pool HiCache host-tier stats (torch-free, loaded by path)."""

import importlib.util
import pathlib
import sys
import unittest
from enum import Enum
from types import SimpleNamespace

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except Exception:  # pragma: no cover - checkout without torch

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "observability"
    / "hicache_pool_stats.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("hicache_pool_stats", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stats = _load_module()


class _PoolName(str, Enum):
    """Stand-in for the PoolName str-enum."""

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    INDEXER = "indexer"
    C4 = "deepseek_v4_c4"
    C4_STATE = "deepseek_v4_c4_state"


class _HostPool:
    def __init__(self, size, available, logical_size=None):
        self.size = size
        self._available = available
        self.logical_size = size if logical_size is None else logical_size

    def available_size(self):
        return self._available


class _NoAllocatorHostPool(_HostPool):
    """Stand-in for DeepSeekV4StateHostPool: sized, but no allocator."""

    def __init__(self, size):
        super().__init__(size=size, available=None)

    def available_size(self):
        raise NotImplementedError(
            "deepseek_v4_c4_state reuses SWA transfer indices and has no allocator"
        )


class _BrokenHostPool(_HostPool):
    def available_size(self):
        raise RuntimeError("free list corrupted")


def _group(*entries):
    return SimpleNamespace(entries=list(entries))


def _entry(name, host_pool):
    return SimpleNamespace(name=name, host_pool=host_pool)


class TestCollectHostPoolStats(unittest.TestCase):
    def test_kv_and_mamba_entries(self):
        group = _group(
            _entry(_PoolName("kv"), _HostPool(size=1000, available=250)),
            _entry(_PoolName("mamba"), _HostPool(size=500, available=500)),
        )
        used, total = stats.collect_host_pool_stats(group)
        self.assertEqual(used, {"kv": 750, "mamba": 0})
        self.assertEqual(total, {"kv": 1000, "mamba": 500})

    def test_logical_size_wins_when_present(self):
        # Under DCP the anchor pool exposes dcp_size x size logical slots and
        # available_size() counts in that same space; match the anchor gauge.
        group = _group(
            _entry(
                _PoolName("kv"), _HostPool(size=1000, available=1200, logical_size=2000)
            ),
        )
        used, total = stats.collect_host_pool_stats(group)
        self.assertEqual(total, {"kv": 2000})
        self.assertEqual(used, {"kv": 800})

    def test_release_slots_count_as_available(self):
        # MambaPoolHost.available_size() = free + release slots; used never
        # goes negative even if a pool over-reports.
        group = _group(_entry(_PoolName("mamba"), _HostPool(size=10, available=12)))
        used, total = stats.collect_host_pool_stats(group)
        self.assertEqual(used, {"mamba": 0})
        self.assertEqual(total, {"mamba": 10})

    def test_plain_string_names(self):
        group = _group(_entry("indexer", _HostPool(size=8, available=3)))
        used, total = stats.collect_host_pool_stats(group)
        self.assertEqual((used, total), ({"indexer": 5}, {"indexer": 8}))

    def test_pool_without_allocator_is_skipped(self):
        # DeepSeekV4StateHostPool.available_size() raises NotImplementedError
        # by design; the DeepSeek V4 stack registers it as a group entry, and
        # the metrics tick must not take the scheduler down over it.
        group = _group(
            _entry(_PoolName("kv"), _HostPool(size=1000, available=250)),
            _entry(_PoolName("deepseek_v4_c4_state"), _NoAllocatorHostPool(size=64)),
            _entry(_PoolName("swa"), _HostPool(size=64, available=60)),
        )
        used, total = stats.collect_host_pool_stats(group)
        self.assertEqual(used, {"kv": 750, "swa": 4})
        self.assertEqual(total, {"kv": 1000, "swa": 64})

    def test_derived_pools_are_skipped(self):
        # A sidecar pool that follows KV indices owns a free list that never
        # moves: available == size while it holds data. Naming it in
        # derived_pools drops it instead of publishing used=0.
        group = _group(
            _entry(_PoolName("kv"), _HostPool(size=1000, available=250)),
            _entry(_PoolName("deepseek_v4_c4"), _HostPool(size=1000, available=1000)),
            _entry(_PoolName("deepseek_v4_c4_state"), _NoAllocatorHostPool(size=64)),
        )
        used, total = stats.collect_host_pool_stats(
            group, derived_pools=[_PoolName.C4, _PoolName.C4_STATE]
        )
        self.assertEqual(used, {"kv": 750})
        self.assertEqual(total, {"kv": 1000})

    def test_derived_pools_accept_plain_strings(self):
        group = _group(
            _entry(_PoolName("kv"), _HostPool(size=10, available=1)),
            _entry(_PoolName("indexer"), _HostPool(size=10, available=10)),
        )
        used, total = stats.collect_host_pool_stats(group, derived_pools=["indexer"])
        self.assertEqual((used, total), ({"kv": 9}, {"kv": 10}))

    def test_other_errors_propagate(self):
        # Only NotImplementedError means "no allocator"; anything else is a bug
        # and must surface.
        group = _group(_entry(_PoolName("kv"), _BrokenHostPool(size=10, available=0)))
        with self.assertRaises(RuntimeError):
            stats.collect_host_pool_stats(group)


class _Component:
    """Stand-in for ComponentType: str() is the lower-cased name."""

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return str(self) == str(other)


class TestHostPoolEvictionCounts(unittest.TestCase):
    def test_component_frees_map_to_pool_labels(self):
        host_frees = {
            _Component("full"): [[1, 2, 3], [4]],
            _Component("mamba"): [[7], [8]],
            _Component("swa"): [],
        }
        self.assertEqual(
            stats.host_pool_eviction_counts(host_frees), {"kv": 4, "mamba": 2}
        )

    def test_c128_component_joins_the_occupancy_label(self):
        # NPU C128 frees into PoolName.DEEPSEEK_V4_C128; the counter must carry
        # the same {pool} label as the occupancy gauges.
        host_frees = {_Component("c128"): [[1, 2]]}
        self.assertEqual(
            stats.host_pool_eviction_counts(host_frees), {"deepseek_v4_c128": 2}
        )

    def test_unmapped_component_keeps_its_name(self):
        host_frees = {_Component("linear"): [[1, 2]]}
        self.assertEqual(stats.host_pool_eviction_counts(host_frees), {"linear": 2})

    def test_empty(self):
        self.assertEqual(stats.host_pool_eviction_counts({}), {})

    def test_custom_labels(self):
        host_frees = {_Component("full"): [[1]]}
        self.assertEqual(
            stats.host_pool_eviction_counts(host_frees, labels={"full": "attention"}),
            {"attention": 1},
        )


if __name__ == "__main__":
    unittest.main()
