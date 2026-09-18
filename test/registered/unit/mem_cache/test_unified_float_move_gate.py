"""FLOAT movement gates protect transfer addresses while allowing hole reuse."""

import itertools
import unittest
from unittest.mock import patch

import test_unified_tri_pool as tri_fixture
import torch

from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def build_geometry(ps, ratio, lazy, pattern, gate):
    fs, ss, ms = tri_fixture._tri_specs(
        full_layer_num=ratio[0],
        swa_layer_num=ratio[1],
        state_layer_num=1,
        head_num=1,
        head_dim=8,
    )
    total = (fs.entry_bytes() + ss.entry_bytes()) * ps * 32 + ms.entry_bytes() * 9
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[fs, ss, ms],
        device="cpu",
        enable_memory_saver=False,
        page_size=ps,
    )
    kv = tri_fixture._FakeUnifiedSWAKVPool(pool)
    mk = tri_fixture._FakeKVCache(pool.max_slots("mamba"))
    a = UnifiedMambaSWATokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=kv,
        mamba_kvcache=mk,
        device="cpu",
        full_max_total_num_tokens=ps * 1000,
        swa_max_total_num_tokens=ps * 1000,
        page_size=ps,
        lazy_compaction=lazy,
    )
    states = a.mamba_allocator.alloc(8)
    assert states is not None
    slots = a.alloc(26 * ps)
    assert slots is not None
    for member, buf in [
        (a.full_attn_allocator, kv.full_kv_pool.buf),
        (a.swa_attn_allocator, kv.swa_kv_pool.buf),
    ]:
        ids = member.translate_kv_loc(slots)
        buf[ids] = slots
    mk.buf[a.mamba_allocator.virtual_to_physical[states]] = states + 1000
    victims = {
        "none": [],
        "paired": [0, 4, 25],
        "swa_edges": [0, 25],
        "swa_internal": [2, 4, 8],
        "empty": list(range(26)),
        "mixed": [0, 2, 25],
    }[pattern]
    ix = (
        torch.cat([slots[i * ps : (i + 1) * ps] for i in victims])
        if victims
        else slots[:0]
    )
    if pattern == "paired":
        a.free(ix)
    else:
        a.free_swa(ix)
    if pattern == "mixed":
        a.free_full(slots[2 * ps : 3 * ps])
        a.mamba_allocator.free(states[1:3])
    if pattern == "swa_edges":
        a.mamba_allocator.free(states[1:4])
    if gate:
        for x in a._flush_targets():
            x.disagg_move_gate = lambda: False
    return (a, kv, mk)


def snapshot(a):
    return [
        (
            x._byte_low_frontier(),
            x._byte_high_frontier(),
            x._free_phys_pages.tolist(),
            x.virtual_to_physical.tolist(),
            x.physical_to_virtual.tolist(),
            x.live_page_count,
            [(id(e), v[0]) for e, v in x._pending_reuse.items()],
        )
        for x in a._flush_targets()
    ]


def payload(a, kv, mk):
    result = []
    for member, buf, offset in [
        (a.full_attn_allocator, kv.full_kv_pool.buf, 0),
        (a.swa_attn_allocator, kv.swa_kv_pool.buf, 0),
        (a.mamba_allocator, mk.buf, 1000),
    ]:
        ids = torch.where(member.virtual_to_physical >= member.min_page_index)[0]
        ids = ids[ids >= member.min_page_index]
        tokens = (
            ids[:, None] * member.page_size + torch.arange(member.page_size)[None, :]
        ).flatten()
        result.append(
            (member, buf, tokens.clone(), buf[member.translate_kv_loc(tokens)].clone())
        )
    return result


class TestFloatMovementGate(CustomTestCase):
    def check_saved(self, a, saved):
        for member, buf, ids, values in saved:
            self.assertTrue(
                torch.all(
                    member.virtual_to_physical[ids // member.page_size]
                    >= member.min_page_index
                )
            )
            torch.testing.assert_close(buf[member.translate_kv_loc(ids)], values)
        self.assertFalse(a.verify_byte_accounting())

    def test_float_movement_owners_respect_gate(self):
        for side, method, gates in itertools.product(
            ("low", "high"),
            ("make_room", "compact_holes"),
            ((False, True), (True, False), (False, False)),
        ):
            with self.subTest(side=side, method=method, gates=gates):
                a, kv, mk = build_geometry(1, (1, 1), True, "swa_internal", False)
                sa = a.swa_attn_allocator
                saved = payload(a, kv, mk)
                sa.disagg_move_gate = lambda: gates[0]
                sa.host_transfer_move_gate = lambda: gates[1]
                before = snapshot(a)
                low, high = sa._gap_pages()
                gap = (low if side == "low" else high) * sa.entry_bytes_per_page
                with (
                    patch.object(
                        sa,
                        "_settle_inflight_forward",
                        wraps=sa._settle_inflight_forward,
                    ) as settle,
                    patch.object(
                        kv.swa_kv_pool,
                        "move_kv_cache",
                        wraps=kv.swa_kv_pool.move_kv_cache,
                    ) as copies,
                ):
                    if method == "make_room":
                        self.assertEqual(
                            sa.make_room(
                                side=side, min_bytes=gap + sa.entry_bytes_per_page
                            ),
                            gap,
                        )
                    else:
                        self.assertEqual(sa.compact_holes(retreat_side=side), 0)
                self.assertEqual(settle.call_count, 0)
                self.assertEqual(copies.call_count, 0)
                self.assertEqual(snapshot(a), before)
                sa.disagg_move_gate = lambda: True
                sa.host_transfer_move_gate = lambda: True
                with patch.object(
                    kv.swa_kv_pool, "move_kv_cache", wraps=kv.swa_kv_pool.move_kv_cache
                ) as copies:
                    if method == "make_room":
                        self.assertGreaterEqual(
                            sa.make_room(
                                side=side, min_bytes=gap + sa.entry_bytes_per_page
                            ),
                            gap + sa.entry_bytes_per_page,
                        )
                    else:
                        self.assertGreater(sa.compact_holes(retreat_side=side), 0)
                self.assertGreater(copies.call_count, 0)
                self.check_saved(a, saved)

    def test_gated_token_recovery_and_hole_reuse(self):
        for ratio, pattern, demand in (
            ((1, 4), "swa_edges", 5),
            ((4, 1), "paired", 7),
            ((1, 1), "swa_internal", 7),
        ):
            with self.subTest(ratio=ratio, pattern=pattern):
                a, kv, mk = build_geometry(1, ratio, True, pattern, True)
                saved = payload(a, kv, mk)
                with patch.object(
                    kv.swa_kv_pool, "move_kv_cache", wraps=kv.swa_kv_pool.move_kv_cache
                ) as copies:
                    a.ensure_capacity(demand, demand)
                    got = a.alloc(demand)
                self.assertEqual(copies.call_count, 0)
                self.check_saved(a, saved)
                if got is None:
                    for member in a._flush_targets():
                        member.disagg_move_gate = None
                    a.ensure_capacity(demand, demand)
                    self.assertIsNotNone(a.alloc(demand))
                    self.check_saved(a, saved)
        a, kv, mk = build_geometry(1, (1, 1), True, "paired", True)
        saved = payload(a, kv, mk)
        with patch.object(
            kv.swa_kv_pool, "move_kv_cache", wraps=kv.swa_kv_pool.move_kv_cache
        ) as copies:
            self.assertIsNotNone(a.alloc(2))
        self.assertEqual(copies.call_count, 0)
        self.check_saved(a, saved)

    def test_state_recovery_obeys_float_gate(self):
        a, kv, mk = build_geometry(1, (4, 1), True, "none", False)
        ma = a.mamba_allocator
        sa = a.swa_attn_allocator
        saved = payload(a, kv, mk)
        demand = ma.available_size() + 1
        sa.disagg_move_gate = lambda: False
        with patch.object(
            kv.swa_kv_pool, "move_kv_cache", wraps=kv.swa_kv_pool.move_kv_cache
        ) as copies:
            self.assertIsNone(ma.alloc(demand))
        self.assertEqual(copies.call_count, 0)
        self.check_saved(a, saved)
        sa.disagg_move_gate = lambda: True
        with patch.object(
            kv.swa_kv_pool, "move_kv_cache", wraps=kv.swa_kv_pool.move_kv_cache
        ) as copies:
            self.assertIsNotNone(ma.alloc(demand))
        self.assertGreater(copies.call_count, 0)
        self.check_saved(a, saved)


if __name__ == "__main__":
    unittest.main()
