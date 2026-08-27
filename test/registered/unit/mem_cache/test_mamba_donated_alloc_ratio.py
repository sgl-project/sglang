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

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.unified_cache.components.mamba_component import MambaComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
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
    # The TreeCore owns the tree member-var state the component reads through.
    component.tree_core = cache
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


class TestMambaRatioEnvGate(unittest.TestCase):
    """SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK gates the pool ratio: off restores the
    original base 3 (overlap 5, lazy 4, no_buffer 3), on drops the base to 2
    (overlap 4, lazy 3) while no_buffer stays 3. Guards the flag wiring so the
    ratio can never drift out of sync with whether the decode lock is skipped."""

    @staticmethod
    def _ratio(*, extra_buffer, lazy, disable_overlap, skip):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        server_args = SimpleNamespace(
            disable_radix_cache=False,
            disable_overlap_schedule=disable_overlap,
            enable_mamba_extra_buffer=lambda: extra_buffer,
            enable_mamba_extra_buffer_lazy=lambda: lazy,
        )
        fake = SimpleNamespace(server_args=server_args)
        # The bag reads (disable_radix_cache / disable_overlap_schedule) come
        # from the published context; the derived-method calls stay on the
        # injected stand-in.
        from sglang.srt import runtime_context as rc

        with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(skip):
            with rc.get_context().override_server_args(
                disable_radix_cache=False,
                disable_overlap_schedule=disable_overlap,
            ):
                return KVCacheConfigurator._calculate_mamba_ratio(fake)

    def test_flag_off_restores_original_ratios(self):
        r = lambda **kw: self._ratio(skip=False, **kw)
        self.assertEqual(
            r(extra_buffer=False, lazy=False, disable_overlap=True), 3
        )  # no_buffer
        self.assertEqual(
            r(extra_buffer=True, lazy=True, disable_overlap=False), 4
        )  # lazy
        self.assertEqual(
            r(extra_buffer=True, lazy=False, disable_overlap=False), 5
        )  # overlap

    def test_flag_on_drops_base_but_keeps_no_buffer(self):
        r = lambda **kw: self._ratio(skip=True, **kw)
        self.assertEqual(
            r(extra_buffer=False, lazy=False, disable_overlap=True), 3
        )  # no_buffer
        self.assertEqual(
            r(extra_buffer=True, lazy=True, disable_overlap=False), 3
        )  # lazy
        self.assertEqual(
            r(extra_buffer=True, lazy=False, disable_overlap=False), 4
        )  # overlap


class _RecordingComp:
    """Fake tree component: records the dec params it is asked to release with."""

    def __init__(self, component_type, priority):
        self.component_type = component_type
        self._priority = priority
        self.released = []

    def eviction_priority(self, is_leaf):
        return self._priority

    def release_component_lock(self, node, params):
        self.released.append(params)

    def release_window_lock(  # SWA only
        self, node, swa_uuid_for_lock, device_frees, host_frees
    ):
        pass


class TestDecSwaLockSkip(unittest.TestCase):
    """dec_swa_lock_only early-releases SWA plus co-located lower-tier (Mamba)
    locks. On a full-only-locked node (decode skip) it must thread the skip set
    into that lower-tier release, else it drops a mamba lock it never took --
    another request's, on a shared FULL+SWA+MAMBA node (Inkling). Guards the
    contract without booting a 3-component model."""

    def test_threads_skip_ids_into_lower_tier_release(self):
        # internal-node priority: full=2 > swa=1 > mamba=0
        full = _RecordingComp(ComponentType.FULL, 2)
        swa = _RecordingComp(ComponentType.SWA, 1)
        mamba = _RecordingComp(ComponentType.MAMBA, 0)
        node = SimpleNamespace(id=7)
        tree_core = SimpleNamespace(
            components=(full, swa, mamba),
            components_by_type={ComponentType.SWA: swa},
            node_by_id=lambda node_id: node,
        )

        UnifiedTreeCore.dec_swa_lock_only(
            tree_core,
            node.id,
            swa_uuid_for_lock=None,
            skip_lock_node_ids={ComponentType.MAMBA: {7}},
        )

        # mamba (below swa) is released, honoring the skip set
        self.assertEqual(len(mamba.released), 1)
        self.assertEqual(
            mamba.released[0].skip_lock_node_ids.get(ComponentType.MAMBA), {7}
        )
        # full (above swa) is never touched
        self.assertEqual(full.released, [])


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
