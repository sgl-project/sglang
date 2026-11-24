"""The HiCache L3 victim tier (SGLANG_OPT_HICACHE_L3_VICTIM_TIER) on the real
cache: the Python tree core, the write_back host tier, the file storage backend
(real bytes) and the real load-back path, with FULL-only fixtures and one
FULL + MAMBA fixture.

* victim tier: a device eviction lands in host and is NOT put; the victim pass
  puts it when it leaves host; a fresh tree restores it with the KV bytes the
  producer wrote; the producer itself restores it and records the promoted copy;
* FULL + MAMBA: the victim put carries the node's state slot, and a restore
  brings back its KV and its state;
* switch unset: the landing is put (inclusive), unchanged, and a storage attach
  reads nothing of the cache's layout;
* the ranks that move in lockstep agree on the tier's parts at attach;
* pin: a span restored from storage survives host pressure until released;
* detach: held victim pages are freed and the tier turns off;
* a PD decode: the tier declines.
"""

import itertools
import unittest
from array import array
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_device_value,
)
from sglang.test.kits.unified_radix_cache.load_back import (
    UnifiedRadixCacheLoadBackFixtures,
)
from sglang.test.kits.unified_radix_cache.storage import (
    UnifiedRadixCacheStorageFixtures,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestL3VictimTierReal(UnifiedRadixCacheStorageFixtures, CustomTestCase):
    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL,),
        kv_size=256,
        max_context_len=256,
    )

    def _cache(self, storage_dir, **switches):
        """A write_back cache on the file backend; a switch set to None is unset."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in switches.items():
            field = getattr(envs, name)
            stack.enter_context(field.override(value))
            if value is None:
                field.clear()
        cache, alloc, rtp = build_fixture(self.cfg, tree_core_backend="python")
        self._init_hicache(
            cache,
            write_policy="write_back",
            storage_backend="file",
            storage_dir=storage_dir,
            prefetch_threshold=1,
        )
        return cache, alloc, rtp

    def _stored_pages(self, cache, keys):
        return cache.cache_controller.storage_backend.batch_exists(keys)

    def _path_keys(self, cache, node):
        keys = []
        for n in self._path_chain(cache, node):
            keys.extend(cache.tree_core.get_hash_values(n))
        return keys

    def _victim_pass_all(self, cache):
        """Run the served victim pass with the margin raised to the whole host
        pool (evicts every evictable host node), then drain the puts' acks."""
        vt = cache._l3vt
        saved = vt.kv_target
        vt.kv_target = cache.cache_controller.mem_pool_host.size
        try:
            cache.check_hicache_events()
        finally:
            vt.kv_target = saved
        self._pump_hicache_until(cache, lambda: not vt.held, "victim puts not acked")

    def _prefetch(self, cache, req_id, anchor, tokens, last_hash=None):
        cache.prefetch_from_storage(req_id, anchor, array("q", tokens), last_hash, None)
        self._run_prefetch_to_completion(cache, req_id)

    def test_victim_tier_puts_on_host_eviction_and_restores_bytes(self):
        d = self._make_storage_dir()
        prod, palloc, prtp = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=True)
        self.assertIsNotNone(prod._l3vt)
        seq = self._make_seq(1, 4)
        self._insert(prod, palloc, prtp, seq)
        leaf = match_prefix(prod, seq).last_device_node
        indices = prod.path_device_indices(leaf)
        self._fill_full_kv(palloc, indices, marker=7)
        expected = self._snapshot_full_kv(palloc, indices)
        keys = self._path_keys(prod, leaf)
        self.assertEqual(len(keys), 4)

        # Device eviction under write_back: lands in host, recorded, not put.
        prod.evict(EvictParams(num_tokens=len(seq)))
        prod.check_hicache_events()
        m = match_prefix(prod, seq)
        self.assertEqual(m.host_hit_length, len(seq))
        self.assertEqual(prod._l3vt.stats["put_ops"], 0)
        self.assertEqual(prod._l3vt.stats["recorded_pages"], 4)
        self.assertEqual(self._stored_pages(prod, keys), 0)

        # Leaving host: put, then gone from host.
        self._victim_pass_all(prod)
        self.assertEqual(prod._l3vt.stats["put_kv_pages"], 4)
        self.assertEqual(prod._l3vt.stats["acked_kv_pages_ok"], 4)
        self.assertEqual(self._stored_pages(prod, keys), 4)
        self.assertEqual(match_prefix(prod, seq).host_hit_length, 0)
        self.assertEqual(prod._l3vt.keymap.kv_entries, 0)

        # A fresh (stock) tree restores the producer's bytes. The producer's
        # switch override is still active: switch it off for this tree.
        cons, calloc, _ = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=False)
        self.assertIsNone(cons._l3vt)
        rid = CacheRequestHandle("l3vt-cons", 0)
        self._prefetch(cons, rid, cons.root_node_handle(), seq)
        mc = match_prefix(cons, seq)
        self.assertEqual(mc.host_hit_length, len(seq))
        self._load_back_node(cons, mc.last_host_node)
        got = self._snapshot_full_kv(
            calloc, cons.path_device_indices(match_prefix(cons, seq).last_device_node)
        )
        self.assertTrue(torch.equal(got[0], expected[0]))
        self.assertTrue(torch.equal(got[1], expected[1]))
        cons.sanity_check()

        # The producer restores it too, and records the promoted copy.
        recorded = prod._l3vt.stats["recorded_nodes"]
        rid = CacheRequestHandle("l3vt-prod", 0)
        self._prefetch(prod, rid, prod.root_node_handle(), seq)
        mp = match_prefix(prod, seq)
        self.assertEqual(mp.host_hit_length, len(seq))
        self.assertGreater(prod._l3vt.stats["recorded_nodes"], recorded)
        self._load_back_node(prod, mp.last_host_node)
        got = self._snapshot_full_kv(
            palloc, prod.path_device_indices(match_prefix(prod, seq).last_device_node)
        )
        self.assertTrue(torch.equal(got[0], expected[0]))
        self.assertTrue(torch.equal(got[1], expected[1]))
        prod.sanity_check()

    def test_stock_path_unchanged_when_off(self):
        d = self._make_storage_dir()
        prod, palloc, prtp = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=None)
        self.assertIsNone(prod._l3vt)
        self.assertFalse(prod.l3vt_pin_active)
        seq = self._make_seq(1, 3)
        self._insert(prod, palloc, prtp, seq)
        leaf = match_prefix(prod, seq).last_device_node
        keys = self._path_keys(prod, leaf)
        prod.evict(EvictParams(num_tokens=len(seq)))
        self._flush_l3_backups(prod)
        # Inclusive: the landing is put while the host still holds it.
        self.assertEqual(match_prefix(prod, seq).host_hit_length, len(seq))
        self.assertEqual(self._stored_pages(prod, keys), len(keys))

    def test_attach_reads_no_layout_with_the_switch_unset(self):
        """A DeepSeek V4 cache's KV anchor (a LogicalHostPool) has no DCP width.
        With the switch unset, whatever the runtime master says, a storage attach
        must not read it: the tier stays off with no decline."""
        d = self._make_storage_dir()
        cache, _, _ = self._cache(
            d,
            SGLANG_OPT_HICACHE_L3_VICTIM_TIER=None,
            SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=True,
        )
        anchor = LogicalHostPool(16 * self.cfg.page_size, self.cfg.page_size)
        entry = cache.cache_controller.mem_pool_host.anchor_entry
        with (
            mock.patch.object(entry, "host_pool", anchor),
            mock.patch.object(
                cache, "_l3vt_unsupported", wraps=cache._l3vt_unsupported
            ) as layout_check,
        ):
            cache.init_l3_victim_tier()
        self.assertEqual(layout_check.call_count, 0)
        self.assertFalse(cache._l3vt_cfg.any_on)
        self.assertIsNone(cache._l3vt_cfg.decline)
        self.assertIsNone(cache._l3vt)
        self.assertFalse(cache.l3vt_pin_active)

    def test_a_part_another_rank_leaves_off_is_off_here(self):
        """Each part changes which storage writes and host evictions a rank makes,
        so the ranks that move in lockstep take the MIN of their parts at attach."""
        d = self._make_storage_dir()
        cache, _, _ = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=True)
        engaged = cache._l3vt_cfg
        self.assertTrue(engaged.victim and engaged.pin)

        def attach_beside(other):
            def reduce(tensor, op):
                other_parts = torch.tensor(other.parts(), dtype=tensor.dtype)
                tensor.copy_(torch.minimum(tensor, other_parts))

            with mock.patch.object(cache, "_all_reduce_attn_groups", reduce):
                cache.init_l3_victim_tier()

        attach_beside(msgspec.structs.replace(engaged, pin=False))
        self.assertIsNotNone(cache._l3vt)
        self.assertIsNone(cache._l3pin)
        self.assertFalse(cache.l3vt_pin_active)
        self.assertEqual(cache._l3vt_cfg.ignored.get("pin"), "off on another rank")

        attach_beside(msgspec.structs.replace(engaged, victim=False))
        self.assertIsNone(cache._l3vt)
        self.assertFalse(cache.l3vt_pin_active)
        self.assertEqual(cache._l3vt_cfg.decline, "declined on another rank")

    def test_pin_survives_host_pressure_until_released(self):
        d = self._make_storage_dir()
        prod, palloc, prtp = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=True)
        self.assertTrue(prod.l3vt_pin_active)
        seq = self._make_seq(1, 4)
        self._insert(prod, palloc, prtp, seq)
        leaf = match_prefix(prod, seq).last_device_node
        indices = prod.path_device_indices(leaf)
        self._fill_full_kv(palloc, indices, marker=11)
        expected = self._snapshot_full_kv(palloc, indices)

        # Device -> host -> storage.
        prod.evict(EvictParams(num_tokens=len(seq)))
        prod.check_hicache_events()
        self._victim_pass_all(prod)
        self.assertEqual(match_prefix(prod, seq).host_hit_length, 0)

        rid = CacheRequestHandle("l3vt-pin", 0)
        self._prefetch(prod, rid, prod.root_node_handle(), seq)
        self.assertEqual(match_prefix(prod, seq).host_hit_length, len(seq))
        self.assertIn(rid, prod._l3pin.pins)

        # Pinned: survives a victim pass that takes everything evictable.
        self._victim_pass_all(prod)
        m = match_prefix(prod, seq)
        self.assertEqual(m.host_hit_length, len(seq))

        self._load_back_node(prod, m.last_host_node)
        got = self._snapshot_full_kv(
            palloc, prod.path_device_indices(match_prefix(prod, seq).last_device_node)
        )
        self.assertTrue(torch.equal(got[0], expected[0]))
        self.assertTrue(torch.equal(got[1], expected[1]))

        # Released at admission: the pin ledger empties and the lock is gone.
        prod.release_l3_pins([type("R", (), {"cache_request_handle": rid})()])
        self.assertEqual(prod._l3pin.pinned_tokens, 0)
        self.assertEqual(prod._l3pin.stats["released_admit"], 1)
        prod.sanity_check()

    def test_detach_frees_held_victims(self):
        d = self._make_storage_dir()
        prod, palloc, prtp = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=True)
        host = prod.cache_controller.mem_pool_host
        free0 = host.available_size()
        seq = self._make_seq(1, 4)
        self._insert(prod, palloc, prtp, seq)
        prod.evict(EvictParams(num_tokens=len(seq)))
        prod.check_hicache_events()
        self.assertLess(host.available_size(), free0)
        # Victim puts that never reach the backup thread: no ack before detach.
        vt = prod._l3vt
        vt.kv_target = host.size
        op_ids = itertools.count(1 << 40)
        with mock.patch.object(
            prod.cache_controller,
            "write_storage",
            side_effect=lambda *a, **k: next(op_ids),
        ):
            prod.check_hicache_events()
        self.assertTrue(vt.held)
        # What keeps the scheduler from calling a flush or detach idle.
        self.assertEqual(prod.l3vt_puts_in_flight(), len(vt.held))
        ok, msg = prod.detach_storage_backend()
        self.assertTrue(ok, msg)
        self.assertIsNone(prod._l3vt)
        self.assertFalse(prod.l3vt_pin_active)
        self.assertEqual(prod.l3vt_puts_in_flight(), 0)
        self.assertEqual(host.available_size(), free0)

    def test_pd_decode_declines(self):
        """A PD decode restores its storage hits from the transfer queue, which
        never releases a pin: the tier, and with it the pins, stay off there."""
        d = self._make_storage_dir()
        decode = SimpleNamespace(
            disaggregation_mode="decode", enable_pd_role_switch=False
        )
        with mock.patch(
            "sglang.srt.mem_cache.unified_radix_cache.get_disagg", return_value=decode
        ):
            cache, _, _ = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=True)
        self.assertIsNone(cache._l3vt)
        self.assertIsNone(cache._l3pin)
        self.assertFalse(cache.l3vt_pin_active)
        self.assertIn("PD decode", cache._l3vt_cfg.decline)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestL3VictimTierMambaReal(
    UnifiedRadixCacheStorageFixtures, UnifiedRadixCacheLoadBackFixtures, CustomTestCase
):
    """FULL + MAMBA, the layout of the GDN models the tier serves."""

    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
    )
    _cache = TestL3VictimTierReal._cache
    _prefetch = TestL3VictimTierReal._prefetch

    def test_victim_put_carries_the_state_and_restores_kv_and_state(self):
        d = self._make_storage_dir()
        prod, palloc, prtp = self._cache(d, SGLANG_OPT_HICACHE_L3_VICTIM_TIER=True)
        vt = prod._l3vt
        seq = self._make_seq(1, 4)
        self._insert(prod, palloc, prtp, seq)
        leaf = match_prefix(prod, seq).last_device_node
        indices = prod.path_device_indices(leaf).clone()
        slot = node_device_value(prod, leaf, ComponentType.MAMBA).clone()
        self._fill_full_kv(palloc, indices, marker=7)
        self._fill_mamba_state(prtp, slot, marker=11)
        want_kv = self._snapshot_full_kv(palloc, indices)
        want_st = self._snapshot_mamba_state(prtp, slot)

        prod.evict(EvictParams(num_tokens=len(seq), mamba_num=1))
        prod.check_hicache_events()
        self.assertEqual(vt.stats["recorded_states"], 1)
        # Raise both margins to the whole host pools: the node leaves host.
        saved = vt.kv_target, vt.state_target
        vt.kv_target = prod.cache_controller.mem_pool_host.size
        vt.state_target = prod.host_pool_group.get_pool(PoolName.MAMBA).size
        try:
            prod.check_hicache_events()
        finally:
            vt.kv_target, vt.state_target = saved
        self._pump_hicache_until(prod, lambda: not vt.held, "victim puts not acked")
        self.assertEqual(vt.stats["put_states"], 1)
        self.assertEqual(vt.stats["acked_states_ok"], 1)
        self.assertEqual(match_prefix(prod, seq).host_hit_length, 0)

        # Clobber the old device copies, then restore: storage -> host -> device.
        self._fill_full_kv(palloc, indices, marker=9)
        self._fill_mamba_state(prtp, slot, marker=21)
        self._prefetch(
            prod, CacheRequestHandle("l3vt-st", 0), prod.root_node_handle(), seq
        )
        m = match_prefix(prod, seq)
        self.assertEqual(m.host_hit_length, len(seq))
        self._load_back_node(prod, m.last_host_node)
        leaf = match_prefix(prod, seq).last_device_node
        got_kv = self._snapshot_full_kv(palloc, prod.path_device_indices(leaf))
        got_st = self._snapshot_mamba_state(
            prtp, node_device_value(prod, leaf, ComponentType.MAMBA)
        )
        for got, want in zip(
            (*got_kv, got_st[0], *got_st[1]), (*want_kv, want_st[0], *want_st[1])
        ):
            self.assertTrue(torch.equal(got, want))
        prod.sanity_check()


if __name__ == "__main__":
    unittest.main()
