"""Acceptance test for the commit-time c4/indexer STATE coupling guard (Fix2)
in SWAComponent._commit_prefetch -- defense-in-depth for non-file backends /
partial per-pool get. CPU-only -- no GPU / model.

The sidecar state pools ride the SWA window key family, so under co-lifetime a
loaded SWA window has one coupled state page per state pool. If a registered
state pool loaded fewer pages than SWA (e.g. flexkv or a partially failing
per-pool get), attaching would restore a desynced (dirty) state, so the whole
window must be dropped (recompute) instead.
"""

import types
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent

SWA = ComponentType.SWA


class TestSwaStateCommitCouplingGuard(unittest.TestCase):
    PAGE_SIZE = 256
    RING = 512  # one window == 2 Full pages

    def _comp(self):
        comp = types.SimpleNamespace()
        comp._strict_bit_exact = True
        comp.component_type = SWA
        comp._swa_kv_pool_host = types.SimpleNamespace(slot_page_size=self.RING)
        attached = []
        comp._attach_swa_host_value = lambda n, s: attached.append(n)
        comp._release_swa_host = lambda s: None
        comp.cache = types.SimpleNamespace(
            page_size=self.PAGE_SIZE, _split_node=lambda *a, **k: None
        )
        comp._attached = attached
        return comp

    def _call(self, comp, extra, kv_hit):
        carrier = types.SimpleNamespace(
            component_data={SWA: types.SimpleNamespace(host_value=None, value=None)},
            parent=None,
        )
        carrier.key = list(range(self.RING))
        transfers = [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.arange(self.RING, dtype=torch.int64),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        insert_result = types.SimpleNamespace(
            inserted_host_node=carrier, total_len=self.RING
        )
        psr = types.SimpleNamespace(extra_pool_hit_pages=extra, kv_hit_pages=kv_hit)
        SWAComponent._commit_prefetch(
            comp,
            anchor=None,
            transfers=transfers,
            insert_result=insert_result,
            pool_storage_result=psr,
        )
        return carrier

    def test_drops_window_when_state_pool_underloaded(self):
        # SWA window loaded (1 ring page) but its coupled c4 STATE pool loaded 0
        # -> attaching would restore desynced state; the guard drops the window.
        comp = self._comp()
        self._call(
            comp,
            {PoolName.SWA: 1, PoolName.DEEPSEEK_V4_C4_STATE: 0},
            kv_hit=2,
        )
        self.assertEqual(comp._attached, [])

    def test_attaches_when_state_pool_coupled(self):
        # state pool loaded == SWA -> co-lifetime satisfied, window attaches.
        comp = self._comp()
        carrier = self._call(
            comp,
            {PoolName.SWA: 1, PoolName.DEEPSEEK_V4_C4_STATE: 1},
            kv_hit=2,
        )
        self.assertTrue(any(n is carrier for n in comp._attached))


if __name__ == "__main__":
    unittest.main()
