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
"""Locked-full SWA tombstone recovery under the unified pool (action handler).

`RecoverSWAWithLockedFull` hands a tombstoned SWA node the INCOMING request's
swa pages and frees only their FULL pages. The static recipe re-points the
locked ids through `full_to_swa_index_mapping`; the unified composite has no
such tensor -- the swa sub-pool's v2p IS the mapping -- so it expresses the
same move as a page-ownership rebind and then frees the incoming ids through
the composite, whose `swa_v2p_pages > 0` filter skips the tombstoned swa side.

The recovery must succeed rather than decline: the TreeCore insert walk counts
the node in `prefix_len` regardless of component consumption, while the SWA
match validator rejects a `value is None` node, so a declined recovery reports
a prefix `match_prefix` cannot honor and trips
`new_prefix_len <= len(new_indices)` in `checkpoint`.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.storage import _page_pairs
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.unified_cache.cache_action import (
    RecoverEvictedSWA,
    RecoverSWAWithLockedFull,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"
_SWA = ComponentType.SWA


class _StubTreeCore:
    """Just what the handler touches: page_size + the device-value setter."""

    def __init__(self, page_size=1):
        self.page_size = page_size
        self.set_calls = []

    def set_component_device_value(self, node_id, component_type, value):
        self.set_calls.append((node_id, component_type, value))


class _Cache:
    def __init__(self, allocator, storage=None):
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token_pool = None
        self._storage = storage

    def kv_storage(self):
        if self._storage is not None:
            return self._storage
        return kv_storage_of(None, self.token_to_kv_pool_allocator)


class _Probe(SWAComponent):
    """SWAComponent wired to the real allocator, its storage and a stub tree core."""

    def __init__(self, allocator):
        self.cache = _Cache(allocator, allocator.storage)
        self.tree_core = _StubTreeCore()


class _StaticAllocRecorder:
    """Stands in for the STATIC SWATokenToKVPoolAllocator: has the mapping
    tensor and a real `set_full_to_swa_mapping`, so the handler must keep
    routing static pools through the original recipe."""

    hands_out_virtual_ids = False
    storage = None

    def __init__(self, n=16):
        self.full_to_swa_index_mapping = torch.arange(n, dtype=torch.int64)
        self.mapping_calls = []
        self.clear_calls = []
        self.freed_full = []
        self.freed_via_inner = []
        self.full_attn_allocator = self

    def set_full_to_swa_mapping(self, full, swa):
        # The handler routes every mapping write THROUGH the API, so the fake
        # must apply it for the mapping asserts to observe anything.
        self.mapping_calls.append((full, swa))
        self.full_to_swa_index_mapping[full.to(torch.int64)] = swa.to(torch.int64)

    def translate_swa_indices_for_transfer(self, full):
        return self.translate_loc_from_full_to_swa(full)

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
        # The full + window chain as the builder makes it, with its storage.
        setup_allocator_context()
        allocator = build_unified_bundle(
            UnifiedGeometry(
                chain="swa",
                full_layer_ids=(0, 1),
                swa_layer_ids=(2, 3),
                page_size=1,
                full_tokens=64,
                swa_tokens=64,
            )
        ).token_to_kv_pool_allocator
        self.assertIsInstance(allocator, ChainAllocator)
        return _Probe(allocator), allocator

    def _two_ranges(self, allocator, n=4):
        kept = allocator.alloc(n)
        incoming = allocator.alloc(n)
        self.assertIsNotNone(kept)
        self.assertIsNotNone(incoming)
        return kept, incoming


class TestPagePairing(_RecoverTestBase):
    def test_pairs_positionally_not_by_sorted_id(self):
        """Allocation hands out virtual ids in no particular order, so a
        `torch.unique` dedup (which sorts) would bind the node's page k to an
        unrelated incoming page."""
        probe, _ = self._probe()
        kept = torch.tensor([9, 7, 5], dtype=torch.int64)  # descending
        incoming = torch.tensor([2, 4, 6], dtype=torch.int64)  # ascending
        kept_pages, incoming_pages = _page_pairs(kept, incoming, 1)
        self.assertEqual(kept_pages.tolist(), [9, 7, 5])
        self.assertEqual(incoming_pages.tolist(), [2, 4, 6])

    def test_length_mismatch_is_rejected(self):
        probe, _ = self._probe()
        with self.assertRaises(AssertionError):
            _page_pairs(
                torch.tensor([1, 2, 3], dtype=torch.int64),
                torch.tensor([4, 5], dtype=torch.int64),
                1,
            )


class TestOwnershipTransfer(_RecoverTestBase):
    def test_node_ids_end_up_owning_the_incoming_physical_pages(self):
        probe, allocator = self._probe()
        swa = allocator.swa_attn_allocator
        kept, incoming = self._two_ranges(allocator)
        donated = swa.ids.virtual_to_physical[incoming.to(torch.int64)].clone()

        allocator.storage.rehome_window("swa", kept, incoming)

        self.assertEqual(
            swa.ids.virtual_to_physical[kept.to(torch.int64)].tolist(),
            donated.tolist(),
            "the node's ids must now resolve to the donated physical pages",
        )
        self.assertTrue(
            bool((swa.ids.virtual_to_physical[incoming.to(torch.int64)] == -1).all()),
            "the incoming ids' swa side must be tombstoned",
        )
        self.assertEqual(
            swa.ids.physical_to_virtual[donated].tolist(),
            kept.to(torch.int64).tolist(),
            "the inverse map must follow, or a later free credits the wrong id",
        )

    def test_sink_or_dead_donor_fails_loud(self):
        """Handing the node the padding sink would serve zeros; refuse."""
        probe, allocator = self._probe()
        kept, incoming = self._two_ranges(allocator)
        allocator.free_swa(incoming)  # donor no longer owns anything
        with self.assertRaises(AssertionError):
            allocator.storage.rehome_window("swa", kept, incoming)


class TestRecoverActionHandler(_RecoverTestBase):
    def test_recovery_sets_a_live_device_value_and_frees_only_the_full_side(self):
        """Bug regression: recovery must give the node a LIVE swa value and
        return only the incoming ids' FULL side, allocating and freeing no swa
        page of its own."""
        probe, allocator = self._probe()
        swa = allocator.swa_attn_allocator
        kept, incoming = self._two_ranges(allocator)
        allocator.free_swa(kept)  # what eviction does when it tombstones
        self.assertTrue(
            bool(
                (allocator.id_spaces()[IdSpaceKind.SLIDING_WINDOW].ids(kept) == 0).all()
            ),
            "precondition: a tombstoned range translates to the sink",
        )
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
        self.assertTrue(
            bool(
                (allocator.id_spaces()[IdSpaceKind.SLIDING_WINDOW].ids(kept) > 0).all()
            ),
            "after recovery the node's ids must address live swa pages",
        )


class TestStaticPoolPathUnchanged(unittest.TestCase):
    def test_static_allocator_keeps_the_mapping_recipe(self):
        """A static SWA allocator must keep the mapping recipe; the unified
        branch must not hijack it."""
        static = _StaticAllocRecorder()
        probe = _Probe.__new__(_Probe)
        probe.cache = _Cache(static)
        probe.tree_core = _StubTreeCore()

        kept = torch.tensor([1, 2], dtype=torch.int64)
        incoming = torch.tensor([5, 6], dtype=torch.int64)
        probe.apply_component_action(
            RecoverSWAWithLockedFull(node_id=3, kept_full=kept, incoming_full=incoming)
        )

        # Both mapping writes go through the allocator API, never by indexing
        # `full_to_swa_index_mapping` (absent on the unified composite).
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


class TestRecoverEvictedSWAHandler(_RecoverTestBase):
    """`RecoverEvictedSWA` is the mapping-only step of locked-full recovery:
    the node's ids adopt the incoming swa pages, while the incoming FULL ids
    and the node's device value stay with the caller. Folding either into the
    shared step would double-free under a tree core that still owns them."""

    def test_unified_pool_moves_ownership_and_frees_nothing(self):
        probe, allocator = self._probe()
        swa = allocator.swa_attn_allocator
        kept, incoming = self._two_ranges(allocator)
        allocator.free_swa(kept)  # what eviction does when it tombstones
        swa_live = swa.allocated_count()
        full_avail = allocator.full_attn_allocator.available_size()

        probe.apply_component_action(
            RecoverEvictedSWA(
                component_type=_SWA, kept_full=kept, incoming_full=incoming
            )
        )

        self.assertTrue(
            bool(
                (allocator.id_spaces()[IdSpaceKind.SLIDING_WINDOW].ids(kept) > 0).all()
            ),
            "the node's ids must address live swa pages after recovery",
        )
        self.assertTrue(
            bool((swa.ids.virtual_to_physical[incoming.to(torch.int64)] == -1).all()),
            "the incoming ids' swa side must be tombstoned",
        )
        self.assertEqual(swa.allocated_count(), swa_live, "ownership only moved")
        self.assertEqual(
            allocator.full_attn_allocator.available_size(),
            full_avail,
            "the incoming FULL side belongs to the caller",
        )
        self.assertEqual(probe.tree_core.set_calls, [], "publication is the caller's")

    def test_static_allocator_moves_the_mapping_and_frees_nothing(self):
        static = _StaticAllocRecorder()
        probe = _Probe.__new__(_Probe)
        probe.cache = _Cache(static)
        probe.tree_core = _StubTreeCore()

        kept = torch.tensor([1, 2], dtype=torch.int64)
        incoming = torch.tensor([5, 6], dtype=torch.int64)
        donated = static.full_to_swa_index_mapping[incoming].clone()
        probe.apply_component_action(
            RecoverEvictedSWA(
                component_type=_SWA, kept_full=kept, incoming_full=incoming
            )
        )

        self.assertEqual(
            static.full_to_swa_index_mapping[kept].tolist(), donated.tolist()
        )
        self.assertEqual(len(static.clear_calls), 1, "incoming must be tombstoned")
        self.assertEqual(static.freed_full, [])
        self.assertEqual(static.freed_via_inner, [])
        self.assertEqual(probe.tree_core.set_calls, [])


if __name__ == "__main__":
    unittest.main()
