"""CPU-only unit tests for the mamba pool ratio vs the prefill->decode peak.

Pins the sizing invariant behind MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO:
at the first cache_unfinished_req, a request still holds its admission-locked
matched-prefix mamba (protected) plus its own COW slot, and then allocates a
donated slot. With N distinct-prefix requests that peak is N own + N locked +
1 donated. An effective ratio of 2 (pool = 2N) leaves no evictable victim and
the donated alloc asserts; ratio 3 (pool = 3N) has headroom. Once decode's
skip_mamba leaves the matched prefix evictable, even ratio 2 recovers via
eviction -- which is why the peak, not the decode steady state, sets the floor.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, IncLockRefResult
from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

N = 4  # concurrent distinct-prefix requests


class _BoundedMambaAllocator:
    """Fixed-capacity slot allocator; alloc returns None once exhausted."""

    def __init__(self, size: int):
        self.free_ids = list(range(size))

    def alloc(self, n: int):
        if len(self.free_ids) < n:
            return None
        return torch.tensor([self.free_ids.pop() for _ in range(n)], dtype=torch.int64)

    def free(self, value: torch.Tensor):
        self.free_ids.extend(int(v) for v in value.tolist())


class _RatioCache:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self, pool_size: int):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.allocator = _BoundedMambaAllocator(pool_size)
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=self.allocator)
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.component_protected_size_ = {ComponentType.MAMBA: 0}
        self.prefix_nodes = []

    def evict(self, params: EvictParams):
        # Reclaim up to mamba_num evictable (unlocked) prefix snapshots, mirroring
        # what the real tree eviction can hand back under mamba pressure.
        need = params.mamba_num
        for node in list(self.prefix_nodes):
            if need <= 0:
                break
            cd = node.component_data[ComponentType.MAMBA]
            if cd.lock_ref == 0 and cd.value is not None:
                self.allocator.free(cd.value)
                self.component_evictable_size_[ComponentType.MAMBA] -= len(cd.value)
                cd.value = None
                self.prefix_nodes.remove(node)
                need -= 1


def _build_peak(pool_size: int, lock_prefixes: bool):
    """N own slots + N matched-prefix snapshots, then return the component ready
    to allocate one donated slot. Prefix snapshots are locked (protected,
    prefill peak) or left evictable (decode steady state after skip_mamba)."""
    cache = _RatioCache(pool_size)
    component = object.__new__(MambaComponent)
    component.cache = cache
    component.component_type = ComponentType.MAMBA

    owned = [cache.allocator.alloc(1) for _ in range(N)]
    assert all(s is not None for s in owned)

    for _ in range(N):
        node = UnifiedTreeNode(cache.tree_components)
        slot = cache.allocator.alloc(1)
        assert slot is not None
        node.component_data[ComponentType.MAMBA].value = slot
        cache.component_evictable_size_[ComponentType.MAMBA] += len(slot)
        cache.prefix_nodes.append(node)
        if lock_prefixes:
            component.acquire_component_lock(node, IncLockRefResult())

    return component, cache, owned


class TestMambaDonatedAllocRatio(unittest.TestCase):
    def test_prefill_peak_ratio2_exhausts_pool(self):
        # pool = 2N, all N prefixes admission-locked: no evictable victim.
        component, _, _ = _build_peak(pool_size=2 * N, lock_prefixes=True)
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            component._alloc_mamba_slot()

    def test_prefill_peak_ratio3_has_headroom(self):
        # pool = 3N: N free slots remain after own + locked prefix.
        component, cache, _ = _build_peak(pool_size=3 * N, lock_prefixes=True)
        slot = component._alloc_mamba_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(cache.component_protected_size_[ComponentType.MAMBA], N)

    def test_decode_steady_evictable_prefix_ratio2_ok(self):
        # pool = 2N but the matched prefixes are evictable (skip_mamba on decode):
        # eviction reclaims a victim, so even ratio 2 serves the donated alloc.
        component, cache, _ = _build_peak(pool_size=2 * N, lock_prefixes=False)
        slot = component._alloc_mamba_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(len(cache.prefix_nodes), N - 1)


if __name__ == "__main__":
    unittest.main()
