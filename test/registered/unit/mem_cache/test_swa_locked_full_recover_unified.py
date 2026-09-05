# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Locked-full SWA tombstone-recovery under the unified pool (action handler).

`RecoverSWAWithLockedFull` recovers a tombstoned SWA node whose full value is
LOCKED: the node cannot adopt the incoming request's ids wholesale, so the
static-pool recipe hands the node the INCOMING ids' swa pages, frees only their
FULL pages, and re-points the locked ids through `full_to_swa_index_mapping`.

The unified composite has no mapping tensor — the swa sub-pool's v2p IS the
mapping — and its `set_full_to_swa_mapping` is an explicit no-op stub. The
pre-fix handler therefore raised AttributeError on `full_to_swa_index_mapping`
(and, had that line been removed, would have silently skipped the rebind while
line 1 freed swa pages the kept ids still referenced). The fix expresses the
same move as a page-ownership REBIND: bind the node's virtual pages to the
incoming pages' physical pages, tombstone the incoming ones, then free the
incoming ids through the composite — whose `swa_v2p_pages > 0` filter skips the
tombstoned swa side, releasing ONLY the full side.

Why the recovery must succeed rather than decline (the v1 lesson, still true on
this branch): the TreeCore insert walk counts the node in `prefix_len`
regardless of component consumption, while the SWA match validator rejects a
`value is None` node — a declined recovery makes `insert` report a prefix the
follow-up `match_prefix` cannot honor, tripping
`new_prefix_len <= len(new_indices)` in `cache_unfinished_req`.

    python -m pytest test/registered/unit/mem_cache/test_swa_locked_full_recover_unified.py -v
"""

import unittest

import torch
from test_multi_ended_allocator import _FakeUnifiedSWAKVPool  # sibling fixture

from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_cache.cache_action import RecoverSWAWithLockedFull
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DEV = "cpu"
_SWA = ComponentType.SWA


def _build_swa_composite(n_full=64, n_swa=64):
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=4,
        head_num=2,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction="up",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=2,
        head_num=2,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction="down",
    )
    total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full_spec, swa_spec],
        device=_DEV,
        enable_memory_saver=False,
    )
    kvcache = _FakeUnifiedSWAKVPool(pool)
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=kvcache,
        device=_DEV,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        need_sort=False,
        forward_stream=None,
    )
    return allocator


class _StubTreeCore:
    """Just what the handler touches: page_size + the device-value setter."""

    def __init__(self, page_size=1):
        self.page_size = page_size
        self.set_calls = []

    def set_component_device_value(self, node_id, component_type, value):
        self.set_calls.append((node_id, component_type, value))


class _Cache:
    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator


class _Probe(SWAComponent):
    """SWAComponent wired to the real allocator and stub tree core."""

    def __init__(self, allocator):
        self.cache = _Cache(allocator)
        self.tree_core = _StubTreeCore()


class _StaticAllocRecorder:
    """Stands in for the STATIC SWATokenToKVPoolAllocator: has the mapping
    tensor and a real set_full_to_swa_mapping. The handler must keep routing
    static pools through the original recipe."""

    def __init__(self, n=16):
        self.full_to_swa_index_mapping = torch.arange(n, dtype=torch.int64)
        self.mapping_calls = []
        self.clear_calls = []
        self.freed_full = []
        self.freed_via_inner = []
        self.full_attn_allocator = self

    def set_full_to_swa_mapping(self, full, swa):
        # Honour the write like the real static allocator: the handler routes
        # every mapping write THROUGH the API (never by indexing the tensor),
        # so the fake must apply it for the mapping asserts to observe it.
        self.mapping_calls.append((full, swa))
        self.full_to_swa_index_mapping[full.to(torch.int64)] = swa.to(torch.int64)

    def clear_full_to_swa_mapping(self, full):
        self.clear_calls.append(full)
        self.full_to_swa_index_mapping[full.to(torch.int64)] = 0

    def free_full_segment(self, indices, *, start_pos):
        self.freed_full.append(indices)

    def free(self, indices):
        # The handler must not reach the inner allocator: that skips the
        # free-group defer.
        self.freed_via_inner.append(indices)

    def translate_loc_from_full_to_swa(self, full_indices):
        return self.full_to_swa_index_mapping[full_indices.to(torch.int64)]


class _RecoverTestBase(unittest.TestCase):
    def _probe(self):
        allocator = _build_swa_composite()
        self.assertIsInstance(allocator, UnifiedSWATokenToKVPoolAllocator)
        return _Probe(allocator), allocator

    def _two_ranges(self, allocator, n=4):
        kept = allocator.alloc(n)
        incoming = allocator.alloc(n)
        self.assertIsNotNone(kept)
        self.assertIsNotNone(incoming)
        return kept, incoming


class TestPagePairing(_RecoverTestBase):
    def test_pairs_positionally_not_by_sorted_id(self):
        """Allocation hands out virtual ids in no particular order; deduping
        with `torch.unique` (which sorts) would bind the node's page k to an
        unrelated incoming page — silent wrong-KV."""
        probe, _ = self._probe()
        kept = torch.tensor([9, 7, 5], dtype=torch.int64)  # descending
        incoming = torch.tensor([2, 4, 6], dtype=torch.int64)  # ascending
        kept_pages, incoming_pages = probe._page_pairs(kept, incoming)
        self.assertEqual(kept_pages.tolist(), [9, 7, 5])
        self.assertEqual(incoming_pages.tolist(), [2, 4, 6])

    def test_length_mismatch_is_rejected(self):
        probe, _ = self._probe()
        with self.assertRaises(AssertionError):
            probe._page_pairs(
                torch.tensor([1, 2, 3], dtype=torch.int64),
                torch.tensor([4, 5], dtype=torch.int64),
            )


class TestOwnershipTransfer(_RecoverTestBase):
    def test_node_ids_end_up_owning_the_incoming_physical_pages(self):
        probe, allocator = self._probe()
        swa = allocator.swa_attn_allocator
        kept, incoming = self._two_ranges(allocator)
        donated = swa.virtual_to_physical[incoming.to(torch.int64)].clone()

        probe._transfer_swa_pages(allocator, kept, incoming)

        self.assertEqual(
            swa.virtual_to_physical[kept.to(torch.int64)].tolist(),
            donated.tolist(),
            "the node's ids must now resolve to the donated physical pages",
        )
        self.assertTrue(
            bool((swa.virtual_to_physical[incoming.to(torch.int64)] == -1).all()),
            "the incoming ids' swa side must be tombstoned",
        )
        self.assertEqual(
            swa.physical_to_virtual[donated].tolist(),
            kept.to(torch.int64).tolist(),
            "the inverse map must follow, or a later free credits the wrong id",
        )

    def test_sink_or_dead_donor_fails_loud(self):
        """Handing the node the padding sink would serve zeros; refuse."""
        probe, allocator = self._probe()
        kept, incoming = self._two_ranges(allocator)
        allocator.free_swa(incoming)  # donor no longer owns anything
        with self.assertRaises(AssertionError):
            probe._transfer_swa_pages(allocator, kept, incoming)


class TestRecoverActionHandler(_RecoverTestBase):
    def test_recovery_sets_a_live_device_value_and_frees_only_the_full_side(self):
        """End-to-end through apply_component_action — the pre-fix handler
        raises AttributeError (`full_to_swa_index_mapping`) on this exact
        call. Post-fix: the node gets a LIVE swa value, the HANDLER neither
        allocates nor frees any swa page (ownership only moves), and the
        incoming ids' FULL side returns to the pool."""
        probe, allocator = self._probe()
        swa = allocator.swa_attn_allocator
        kept, incoming = self._two_ranges(allocator)
        allocator.free_swa(kept)  # what eviction does when it tombstones
        # Snapshot AFTER the setup traffic: the invariant under test is that
        # the recovery handler itself moves ownership without moving capacity.
        swa_live = swa.allocated_count()
        full_avail = allocator.full_attn_allocator.available_size()

        probe.apply_component_action(
            RecoverSWAWithLockedFull(node_id=7, kept_full=kept, incoming_full=incoming)
        )

        ((node_id, ct, value),) = probe.tree_core.set_calls
        self.assertEqual((node_id, ct), (7, _SWA))
        self.assertEqual(len(value), len(kept))
        self.assertTrue(
            bool((value > 0).all()),
            "recovered value must address live swa pages, not the sink",
        )
        self.assertEqual(
            swa.allocated_count(),
            swa_live,
            "no swa page may be released or gained — ownership only moved",
        )
        self.assertEqual(
            allocator.full_attn_allocator.available_size(),
            full_avail + len(incoming),
            "the incoming ids' FULL side must come back",
        )

    def test_recovered_ids_translate_to_live_pages_not_the_sink(self):
        """The tombstoned range translates to the clamped sink before the
        recovery and to real pages after — recovering from the node's OWN
        already-freed ids (instead of the donated ones) reintroduces the sink."""
        probe, allocator = self._probe()
        kept, incoming = self._two_ranges(allocator)
        allocator.free_swa(kept)
        self.assertTrue(
            bool((allocator.translate_loc_from_full_to_swa(kept) == 0).all()),
            "precondition: a tombstoned range translates to the sink",
        )
        probe.apply_component_action(
            RecoverSWAWithLockedFull(node_id=1, kept_full=kept, incoming_full=incoming)
        )
        self.assertTrue(
            bool((allocator.translate_loc_from_full_to_swa(kept) > 0).all()),
            "after recovery the node's ids must address live swa pages",
        )


class TestStaticPoolPathUnchanged(unittest.TestCase):
    def test_static_allocator_keeps_the_mapping_recipe(self):
        """A static SWA allocator (has the mapping tensor) must keep the
        original recipe — the unified branch must not hijack it."""
        static = _StaticAllocRecorder()
        probe = _Probe.__new__(_Probe)
        probe.cache = _Cache(static)
        probe.tree_core = _StubTreeCore()

        kept = torch.tensor([1, 2], dtype=torch.int64)
        incoming = torch.tensor([5, 6], dtype=torch.int64)
        probe.apply_component_action(
            RecoverSWAWithLockedFull(node_id=3, kept_full=kept, incoming_full=incoming)
        )

        # Both mapping writes go through the allocator API -- the kept remap
        # via set_full_to_swa_mapping, the incoming tombstone via
        # clear_full_to_swa_mapping -- never by indexing
        # `full_to_swa_index_mapping` (the tensor is absent on the unified
        # composite by design).
        self.assertEqual(len(static.mapping_calls), 1, "static recipe must run")
        self.assertEqual(len(static.clear_calls), 1, "incoming must be tombstoned")
        self.assertTrue(
            bool(
                (static.full_to_swa_index_mapping[incoming.to(torch.int64)] == 0).all()
            ),
            "incoming ids' mapping entries must be zeroed (static recipe)",
        )
        # Through free_full_segment, not the inner allocator: the latter skips
        # the free-group defer.
        self.assertEqual(len(static.freed_full), 1)
        self.assertEqual(static.freed_via_inner, [])
        ((node_id, ct, _),) = probe.tree_core.set_calls
        self.assertEqual((node_id, ct), (3, _SWA))


if __name__ == "__main__":
    unittest.main()
