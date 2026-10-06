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
"""Unified-memory HiCache transfers on the real SWA allocator stack.

Allocation takes only free pages, so it does not wait on HiCache copies. The
pages a copy uses are kept out of the free list by their owner instead: a
physical reservation until its load is queued, the host-transfer move gate
until the ack, a tree lock until a write-through ack. These tests pin those
lifecycles and check the rows' contents, not only the bookkeeping.

The pools are real CPU tensors. CUDA stream waits are recorded by a stand-in
for `torch.cuda.current_stream()` rather than executed.

    python -m pytest test/registered/unit/mem_cache/test_unified_hicache_transfer_lifecycle.py -v
"""

import atexit
import contextlib
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import cache_controller
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import _StagedPrefetch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _swa_allocation_callbacks,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pool_host import common as host_memory
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.tree_core_registry import _TREE_CORE_REGISTRY
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import build_tri_pool

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 4
D2H = ("controller", "write")
H2D = ("controller", "load")
FULL, SWA = ComponentType.FULL, ComponentType.SWA


def _python_inspector(params, components):
    from unified_tree_core_inspector import UnifiedTreeCoreInspector

    return UnifiedTreeCoreInspector(params, components)


def _rust_inspector(params, _components):
    from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

    return RustUnifiedTreeCoreInspector(params)


def _unified_swa_bundle(*, lazy: bool = True, tokens: int = 64):
    return init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=PAGE,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        full_max_total_num_tokens=tokens,
        swa_max_total_num_tokens=tokens,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=lazy,
    )


def _unified_swa(*, lazy: bool = True, tokens: int = 64):
    bundle = _unified_swa_bundle(lazy=lazy, tokens=tokens)
    return bundle.token_to_kv_pool_allocator, _Rows(bundle)


class _Rows:
    """One marker per token row, read and written through the allocator's own
    virtual-to-physical translation, so a relocated row is found where it went."""

    def __init__(self, bundle):
        self.allocator = bundle.token_to_kv_pool_allocator
        self.full = bundle.token_to_kv_pool.full_kv_pool.get_key_buffer(0)
        self.swa = bundle.token_to_kv_pool.swa_kv_pool.get_key_buffer(0)

    def full_rows(self, ids: torch.Tensor) -> torch.Tensor:
        return self.allocator.full_attn_allocator.translate_kv_loc(ids.to(torch.int64))

    def swa_rows(self, ids: torch.Tensor) -> torch.Tensor:
        return self.allocator.translate_loc_from_full_to_swa(ids)

    def markers(self, base: int, count: int) -> torch.Tensor:
        # One value per token row; float16 holds these integers exactly.
        values = torch.arange(base, base + count, dtype=torch.float16)
        return values[:, None, None].expand(count, *self.swa.shape[1:]).clone()

    def seed(self, ids: torch.Tensor, base: int):
        self.full[self.full_rows(ids)] = self.markers(base, len(ids))
        self.swa[self.swa_rows(ids)] = self.markers(base + 1000, len(ids))
        return self.read(ids)

    def read(self, ids: torch.Tensor):
        return (
            self.full[self.full_rows(ids)].clone(),
            self.swa[self.swa_rows(ids)].clone(),
        )


class _Stream:
    """Stands in for `torch.cuda.current_stream()` and records what it waits on."""

    def __init__(self, log: list):
        self.log = log

    def wait_event(self, event) -> None:
        self.log.append(("wait", event))

    def wait_stream(self, stream) -> None:
        self.log.append(("wait_stream", stream))


@contextlib.contextmanager
def _recorded_stream():
    log = []
    with mock.patch.object(
        torch.cuda, "current_stream", lambda *args, **kwargs: _Stream(log)
    ):
        yield log


def _record_moves(allocator, log: list) -> None:
    """Log every compaction copy on either side into `log`, then perform it."""
    for side in (allocator.full_attn_allocator, allocator.swa_attn_allocator):
        kvcache = side._kvcache
        move = kvcache.move_kv_cache

        def recorded(dst, src, *, move=move, name=side.sub_pool_name):
            log.append(("move", name))
            return move(dst, src)

        kvcache.move_kv_cache = recorded


def _events(allocator) -> dict:
    return {
        side.sub_pool_name: dict(side._hicache_transfer_done_events)
        for side in (allocator.full_attn_allocator, allocator.swa_attn_allocator)
    }


def _state_bytes(allocator, slots: torch.Tensor) -> list:
    """The shared-buffer bytes behind each Mamba state slot (views, not copies)."""
    end = allocator.mamba_allocator
    raw, size = end.unified_buffer._raw, end.entry_bytes_per_page
    pages = end.virtual_to_physical[slots].tolist()
    return [raw[page * size : (page + 1) * size] for page in pages]


class _ConfigCase(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")


class TestPhysicalReservation(_ConfigCase):
    """`alloc_physical` / bind / cancel / `free_physical` on the SWA end."""

    def test_reservation_is_counted_unbound_and_released_by_cancel(self):
        allocator, rows = _unified_swa()
        swa = allocator.swa_attn_allocator
        live = allocator.alloc(2 * PAGE)
        allocated = swa.allocated_count()
        available = swa.available_size()

        reserved = swa.alloc_physical(2 * PAGE)
        pages = reserved[::PAGE] // PAGE
        self.assertEqual(reserved.numel(), 2 * PAGE)
        self.assertEqual(swa._pending_hicache_load_pages, 2)
        self.assertTrue(swa.moves_blocked())
        self.assertEqual(swa.allocated_count(), allocated + 2 * PAGE)
        self.assertEqual(swa.available_size(), available - 2 * PAGE)
        # Owned by the reservation alone: no virtual page maps to these pages.
        self.assertTrue(bool((swa.physical_to_virtual[pages] == -1).all()))
        self.assertFalse(bool(torch.isin(swa.virtual_to_physical, pages).any()))
        # Nor does a later allocation hand them out.
        other = allocator.alloc(2 * PAGE)
        self.assertFalse(bool(torch.isin(rows.swa_rows(other), reserved).any()))
        allocator.free(other)

        swa.cancel_physical_reservation(reserved)

        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertFalse(swa.moves_blocked())
        self.assertEqual(swa.allocated_count(), allocated)
        self.assertTrue(bool(torch.isin(pages, swa._free_phys_pages).all()))
        self.assertEqual(allocator.verify_byte_accounting(), [])
        allocator.free(live)

    def test_failed_reservation_leaves_no_reservation(self):
        allocator, _ = _unified_swa()
        swa = allocator.swa_attn_allocator
        live = allocator.alloc(2 * PAGE)
        allocated = swa.allocated_count()

        self.assertIsNone(swa.alloc_physical(swa.available_size() + PAGE))

        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertFalse(swa.moves_blocked())
        self.assertEqual(swa.allocated_count(), allocated)
        self.assertEqual(allocator.verify_byte_accounting(), [])
        allocator.free(live)

    def test_pending_reservation_blocks_moves_until_its_load_is_queued(self):
        allocator, rows = _unified_swa()
        swa = allocator.swa_attn_allocator
        head = allocator.alloc(2 * PAGE)
        gap = allocator.alloc(2 * PAGE)
        tail = allocator.alloc(2 * PAGE)
        expected_head = rows.seed(head, 100)
        expected_tail = rows.seed(tail, 200)
        # The node being loaded: its FULL rows are resident, SWA is tombstoned.
        target = allocator.alloc(2 * PAGE)
        allocator.free_swa(target)
        allocator.free(gap)

        reserved = swa.alloc_physical(2 * PAGE)
        # Stand in for the H2D copy into the reserved rows.
        rows.swa[reserved] = rows.markers(300, 2 * PAGE)
        allocator.set_full_to_swa_mapping(target, reserved)
        self.assertTrue(torch.equal(rows.swa_rows(target), reserved))
        live = torch.cat([head, tail, target])
        placed = rows.swa_rows(live)

        moves = []
        _record_moves(allocator, moves)
        with _recorded_stream() as waits:
            # Not queued yet: nothing on the SWA end may move.
            self.assertEqual(swa._flush(urgent=True), 0)
            self.assertEqual(swa.flush_opportunistic(), 0)
            self.assertEqual(moves, [])
            self.assertEqual(waits, [])
            self.assertTrue(torch.equal(rows.swa_rows(live), placed))

            load_done = object()
            allocator.set_hicache_transfer_done_event(H2D, load_done)
            self.assertEqual(swa._pending_hicache_load_pages, 0)
            self.assertFalse(swa.moves_blocked())

            # Queued: compaction waits on the copy, then relocates rows.
            self.assertGreater(swa._flush(urgent=True), 0)
        self.assertEqual(waits[0], ("wait", load_done))
        self.assertIn(("move", "swa"), moves)
        self.assertFalse(torch.equal(rows.swa_rows(live), placed))
        _, target_swa = rows.read(target)
        self.assertTrue(torch.equal(target_swa, rows.markers(300, 2 * PAGE)))
        self.assertTrue(torch.equal(rows.read(head)[1], expected_head[1]))
        self.assertTrue(torch.equal(rows.read(tail)[1], expected_tail[1]))
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_cancel_after_partial_bind_releases_every_reserved_page(self):
        # Rollback releases the whole reservation, bound or not.
        allocator, rows = _unified_swa()
        swa = allocator.swa_attn_allocator
        target = allocator.alloc(2 * PAGE)
        allocator.free_swa(target)
        allocated = swa.allocated_count()

        reserved = swa.alloc_physical(2 * PAGE)
        allocator.set_full_to_swa_mapping(target[:PAGE], reserved[:PAGE])
        swa.cancel_physical_reservation(reserved)

        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertEqual(swa.allocated_count(), allocated)
        # The tombstone is back: the loaded row is no longer reachable.
        self.assertTrue(bool((rows.swa_rows(target) == 0).all()))
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_redundant_destination_is_freed_after_its_ack(self):
        # Buffer-only staged loads reserve the whole SWA window and keep the
        # rows of resident pages until the H2D ack, then free them.
        allocator, rows = _unified_swa()
        swa = allocator.swa_attn_allocator
        resident = allocator.alloc(PAGE)
        missing = allocator.alloc(PAGE)
        allocator.free_swa(missing)
        expected_resident = rows.seed(resident, 400)
        resident_rows = rows.swa_rows(resident)
        allocated = swa.allocated_count()

        reserved = swa.alloc_physical(2 * PAGE)
        rows.swa[reserved] = rows.markers(500, 2 * PAGE)
        redundant, loaded = reserved[:PAGE], reserved[PAGE:]
        allocator.set_full_to_swa_mapping(missing, loaded)
        in_flight = {"copies": 1}
        allocator.set_host_transfer_move_gate(lambda: in_flight["copies"] == 0)
        with _recorded_stream() as waits:
            allocator.set_hicache_transfer_done_event(H2D, object())
            # Until the ack the redundant rows stay allocated and nothing moves.
            self.assertTrue(swa.moves_blocked())
            self.assertEqual(swa._flush(urgent=True), 0)
            again = allocator.alloc(PAGE)
            self.assertFalse(bool(torch.isin(rows.swa_rows(again), reserved).any()))
            allocator.free(again)
            self.assertEqual(waits, [])

            in_flight["copies"] = 0
            swa.free_physical(redundant)
        self.assertEqual(len(waits), 1)
        self.assertEqual(swa.allocated_count(), allocated + PAGE)
        self.assertTrue(torch.equal(rows.swa_rows(resident), resident_rows))
        self.assertTrue(torch.equal(rows.read(resident)[1], expected_resident[1]))
        self.assertTrue(
            torch.equal(rows.read(missing)[1], rows.markers(504, PAGE)),
        )
        self.assertEqual(allocator.verify_byte_accounting(), [])


class TestFloatSwaReservation(_ConfigCase):
    """The tri-pool's SWA middle floats: it relocates its pages to make room for
    a neighbour. A load's SWA reservation pins it until the load is queued or
    cancelled, as on an end pool; the transfer gate then holds until the ack."""

    def setUp(self):
        super().setUp()
        bundle, self.allocator = build_tri_pool(page_size=PAGE)
        self.rows = _Rows(bundle)
        self.swa = self.allocator.swa_attn_allocator
        self.states = bundle.req_to_token_pool.mamba_allocator
        mamba = self.allocator.mamba_allocator
        resident = self.allocator.alloc(4 * PAGE)
        # The Mamba end is full: its next slot needs the float to move.
        self.held = mamba.alloc(mamba.available_size())
        for row in _state_bytes(self.allocator, self.held):
            row.fill_(90)
        # A window slides past its oldest page: a hole at the float's low edge.
        self.allocator.free_swa(resident[:PAGE])
        self.resident = resident[PAGE:]
        self.expected = self.rows.seed(self.resident, 100)
        self.attempts = []
        self.moves = []
        _record_moves(self.allocator, self.moves)

    def _controller(self, evict_state=None) -> HybridCacheController:
        """What `HybridCacheController.load` reads, with the production SWA and
        Mamba callbacks of the tri-pool."""
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_device_allocator = self.allocator
        controller.device = "cpu"
        controller.load_queue, controller.ack_load_queue = [], []

        def alloc_state(need_size):
            # What guards the float while the state slot is allocated.
            self.attempts.append(
                (self.swa._pending_hicache_load_pages, len(controller.load_queue))
            )
            return self.states.alloc(need_size)

        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.SWA: SimpleNamespace(
                    device_evict_fn=None, **_swa_allocation_callbacks(self.swa)
                ),
                PoolName.MAMBA: SimpleNamespace(
                    device_alloc_fn=alloc_state,
                    device_free_fn=self.states.free,
                    device_evict_fn=evict_state,
                ),
            }
        )
        self.allocator.set_host_transfer_move_gate(
            lambda: not (controller.load_queue or controller.ack_load_queue)
        )
        return controller

    def _load(self, controller):
        # The production order: FULL, then SWA, then the Mamba checkpoint.
        self.transfers = [
            PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(PAGE)),
            PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([1])),
        ]
        return controller.load(
            host_indices=torch.arange(PAGE), extra_pools=self.transfers
        )

    def _assert_kept(self, held: torch.Tensor):
        got = self.rows.read(self.resident)[1]
        self.assertTrue(torch.equal(got, self.expected[1]))
        for row in _state_bytes(self.allocator, held):
            self.assertTrue(bool((row == 90).all()))

    def test_state_allocation_cannot_move_a_reservation_before_its_load_is_queued(
        self,
    ):
        full, swa = self.allocator.full_attn_allocator, self.swa
        controller = self._controller()
        allocated = (full.allocated_count(), swa.allocated_count())
        span = (swa.low_wm_page, swa.high_wm_page)

        # The state slot needs the float to move, which the SWA reservation
        # taken just before forbids: the load rolls back instead.
        self.assertIsNone(self._load(controller))

        self.assertEqual(self.attempts, [(1, 0)])
        self.assertEqual(self.moves, [])
        self.assertEqual((swa.low_wm_page, swa.high_wm_page), span)
        self.assertIsNone(self.transfers[0].device_indices)
        self.assertEqual(controller.load_queue, [])
        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertEqual((full.allocated_count(), swa.allocated_count()), allocated)
        # No move wrote through the unbound page's -1 owner.
        self.assertEqual(swa.virtual_to_physical[-1].item(), -1)
        self._assert_kept(self.held)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])

        # Cancelled, so the float makes room again.
        self.assertIsNotNone(self.states.alloc(1))
        self.assertIn(("move", "swa"), self.moves)
        self._assert_kept(self.held)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])

    def test_reservation_stays_put_until_its_load_is_acked(self):
        swa, rows = self.swa, self.rows
        evicted, held = self.held[-1:], self.held[:-1]
        controller = self._controller(
            evict_state=lambda _num_slots: self.states.free(evicted)
        )

        loaded = self._load(controller)

        self.assertIsNotNone(loaded)
        reserved, state = (transfer.device_indices for transfer in self.transfers)
        for row in _state_bytes(self.allocator, state):
            row.fill_(91)
        # The H2D will write the reserved rows, which no state slot may overlap.
        rows.swa[reserved] = rows.markers(300, PAGE)
        for row in _state_bytes(self.allocator, state):
            self.assertTrue(bool((row == 91).all()))
        self._assert_kept(held)
        # The first state attempt could not move the float; the retry after the
        # tree evicted a checkpoint took that slot. Both ran before the load was
        # queued, so the reservation alone kept the float in place.
        self.assertEqual(self.attempts, [(1, 0), (1, 0)])
        self.assertEqual(self.moves, [])

        # Queued and bound into the tree, not yet submitted.
        self.allocator.set_full_to_swa_mapping(loaded, reserved)
        self.assertEqual(swa._pending_hicache_load_pages, 1)
        self.assertIsNone(self.states.alloc(1))
        self.assertEqual(self.moves, [])

        # Submitted: the reservation count hands over to the transfer gate.
        controller.ack_load_queue.append(controller.load_queue.pop())
        self.allocator.set_hicache_transfer_done_event(H2D, object())
        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertIsNone(self.states.alloc(1))
        self.assertEqual(self.moves, [])

        # Acked: the float moves again, and the loaded rows go with it.
        controller.ack_load_queue.clear()
        self.assertIsNotNone(self.states.alloc(1))
        self.assertIn(("move", "swa"), self.moves)
        self.assertFalse(torch.equal(rows.swa_rows(loaded), reserved))
        self.assertTrue(torch.equal(rows.read(loaded)[1], rows.markers(300, PAGE)))
        for row in _state_bytes(self.allocator, state):
            self.assertTrue(bool((row == 91).all()))
        self._assert_kept(held)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])


class TestAllocationAndTransferOrder(_ConfigCase):
    """Allocation reuses free pages without waiting; moves and frees still wait."""

    def test_hole_reuse_does_not_wait_for_unrelated_copies(self):
        allocator, rows = _unified_swa()
        swa = allocator.swa_attn_allocator
        source = allocator.alloc(2 * PAGE)  # a write-through D2H source (locked)
        gap = allocator.alloc(3 * PAGE)
        target = allocator.alloc(2 * PAGE)  # an H2D target owned by its load
        expected_source = rows.seed(source, 100)
        expected_target = rows.seed(target, 200)
        allocator.free(gap)
        holes = {
            side.sub_pool_name: set(side._free_phys_pages.tolist())
            for side in (allocator.full_attn_allocator, swa)
        }
        backup_done, load_done = object(), object()
        allocator.set_hicache_transfer_done_event(D2H, backup_done)
        allocator.set_hicache_transfer_done_event(H2D, load_done)
        registered = _events(allocator)

        with _recorded_stream() as waits:
            fresh = allocator.alloc(2 * PAGE)
            staged = swa.alloc_physical(PAGE)

        self.assertEqual(waits, [])
        self.assertEqual(_events(allocator), registered)
        # Both came out of the holes, not out of rows a copy is using.
        full_pages = set((rows.full_rows(fresh)[::PAGE] // PAGE).tolist())
        swa_pages = set((rows.swa_rows(fresh)[::PAGE] // PAGE).tolist())
        swa_pages |= set((staged[::PAGE] // PAGE).tolist())
        self.assertLessEqual(full_pages, holes["full"])
        self.assertLessEqual(swa_pages, holes["swa"])
        rows.seed(fresh, 600)
        rows.swa[staged] = rows.markers(700, PAGE)
        for ids, expected in ((source, expected_source), (target, expected_target)):
            got = rows.read(ids)
            self.assertTrue(torch.equal(got[0], expected[0]))
            self.assertTrue(torch.equal(got[1], expected[1]))

    def test_moves_and_frees_wait_for_every_copy_first(self):
        allocator, rows = _unified_swa()
        swa = allocator.swa_attn_allocator
        head = allocator.alloc(2 * PAGE)
        gap = allocator.alloc(2 * PAGE)
        tail = allocator.alloc(2 * PAGE)
        expected_tail = rows.seed(tail, 100)
        allocator.free(gap)
        log = []
        _record_moves(allocator, log)
        backup_done, load_done = object(), object()
        allocator.set_hicache_transfer_done_event(D2H, backup_done)
        allocator.set_hicache_transfer_done_event(H2D, load_done)

        with mock.patch.object(
            torch.cuda, "current_stream", lambda *args, **kwargs: _Stream(log)
        ):
            self.assertGreater(swa._flush(urgent=True), 0)
        self.assertEqual(
            log[:3], [("wait", backup_done), ("wait", load_done), ("move", "swa")]
        )
        self.assertEqual(swa._hicache_transfer_done_events, {})
        self.assertTrue(torch.equal(rows.read(tail)[1], expected_tail[1]))

        # Releasing a reservation also orders after outstanding copies.
        reserved = swa.alloc_physical(PAGE)
        allocator.set_hicache_transfer_done_event(H2D, load_done)
        with _recorded_stream() as waits:
            swa.free_physical(reserved)
        self.assertEqual(waits, [("wait", load_done)])
        allocator.free(head)

    def test_eager_free_waits_for_copies_before_compacting(self):
        allocator, rows = _unified_swa(lazy=False)
        full = allocator.full_attn_allocator
        head = allocator.alloc(2 * PAGE)
        tail = allocator.alloc(2 * PAGE)
        expected_tail = rows.seed(tail, 100)
        log = []
        _record_moves(allocator, log)
        backup_done = object()
        allocator.set_hicache_transfer_done_event(D2H, backup_done)

        with mock.patch.object(
            torch.cuda, "current_stream", lambda *args, **kwargs: _Stream(log)
        ):
            full.free(head)
        self.assertEqual(log[0], ("wait", backup_done))
        self.assertIn(("move", "full"), log)
        self.assertTrue(torch.equal(rows.read(tail)[0], expected_tail[0]))

    def test_closed_transfer_gate_keeps_rows_in_place(self):
        allocator, rows = _unified_swa()
        in_flight = {"copies": 1}
        allocator.set_host_transfer_move_gate(lambda: in_flight["copies"] == 0)
        head = allocator.alloc(2 * PAGE)
        gap = allocator.alloc(2 * PAGE)
        tail = allocator.alloc(2 * PAGE)
        expected_tail = rows.seed(tail, 100)
        allocator.free(gap)
        full_rows, swa_rows = rows.full_rows(tail), rows.swa_rows(tail)
        moves = []
        _record_moves(allocator, moves)

        with _recorded_stream() as waits:
            for side in (allocator.full_attn_allocator, allocator.swa_attn_allocator):
                self.assertEqual(side._flush(urgent=True), 0)
            # Free pages are still handed out while the copy runs.
            reused = allocator.alloc(PAGE)
        self.assertIsNotNone(reused)
        self.assertEqual(moves, [])
        self.assertEqual(waits, [])
        self.assertTrue(torch.equal(rows.full_rows(tail), full_rows))
        self.assertTrue(torch.equal(rows.swa_rows(tail), swa_rows))

        in_flight["copies"] = 0
        with _recorded_stream():
            self.assertGreater(allocator.swa_attn_allocator._flush(urgent=True), 0)
        got = rows.read(tail)
        self.assertTrue(torch.equal(got[0], expected_tail[0]))
        self.assertTrue(torch.equal(got[1], expected_tail[1]))
        allocator.free(head)

    def test_allocation_that_needs_compaction_stays_behind_the_gate(self):
        # The SWA end holds only fragmented holes; FULL can grow into the shared
        # gap only after SWA compacts, which the closed gate forbids.
        allocator, rows = _unified_swa(tokens=32)
        full, swa = allocator.full_attn_allocator, allocator.swa_attn_allocator
        in_flight = {"copies": 1}
        allocator.set_host_transfer_move_gate(lambda: in_flight["copies"] == 0)
        chunks = []
        while (ids := allocator.alloc(PAGE)) is not None:
            chunks.append(ids)
        for ids in chunks[::2]:
            allocator.free_swa(ids)
        survivors = chunks[1::2]
        expected = [rows.read(ids) for ids in survivors]
        need = full.available_size() + PAGE
        moves = []
        _record_moves(allocator, moves)

        with _recorded_stream():
            self.assertIsNone(full.alloc(need))
        self.assertEqual(moves, [])
        for ids, before in zip(survivors, expected):
            self.assertTrue(torch.equal(rows.read(ids)[1], before[1]))
        self.assertGreater(len(swa._free_phys_pages), 0)
        self.assertEqual(allocator.verify_byte_accounting(), [])


class _UnifiedHiCacheCase(_ConfigCase):
    """A FULL+SWA `UnifiedRadixCache` with HiCache over the unified pool.

    With no storage backend the host side is the shared page-envelope pool,
    whose copies run on CPU; the controller, tree and allocator are real. The
    TreeCore is whichever backend the environment selects (Rust when its
    extension loads, else Python), wrapped by the suite's read-only inspector.
    """

    window = 2 * PAGE

    def _cache(
        self,
        *,
        write_policy: str = "write_through",
        tokens: int = 128,
        host_memory_mode: str = "cache",
    ):
        server_args = ServerArgs(
            model_path="dummy",
            page_size=PAGE,
            enable_unified_memory=True,
            hicache_write_policy=write_policy,
            hicache_mem_layout="layer_first",
            hicache_host_memory_mode=host_memory_mode,
        )
        set_global_server_args_for_scheduler(server_args)
        bundle = _unified_swa_bundle(tokens=tokens)
        self.allocator = bundle.token_to_kv_pool_allocator
        self.kv_pool = bundle.token_to_kv_pool
        self.rows = _Rows(bundle)
        self.req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=tokens, device="cpu", enable_memory_saver=False
        )
        params = CacheInitParams(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=PAGE,
            disable=False,
            sliding_window_size=self.window,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
        with mock.patch.dict(
            _TREE_CORE_REGISTRY, {"python": _python_inspector, "rust": _rust_inspector}
        ):
            cache = UnifiedRadixCache(params=params)
        # CPU host tensors need no page-locking.
        with mock.patch.object(host_memory, "_cuda_host_register"):
            cache.init_hicache(server_args, params)
        atexit.unregister(cache.shutdown)
        self.addCleanup(cache.release_host_resources)
        controller = cache.cache_controller
        # The scheduler's gate: rows stay put until host copies are acknowledged.
        self.allocator.set_host_transfer_move_gate(
            lambda: not controller.has_inflight_device_transfers()
        )
        self.cache = cache
        self.controller = controller
        # Compaction and frees wait through this stand-in stream.
        stream = _recorded_stream()
        self.waits = stream.__enter__()
        self.addCleanup(stream.__exit__, None, None, None)
        return cache

    def _insert(self, tokens, base):
        ids = self.allocator.alloc(len(tokens))
        expected = self.rows.seed(ids, base)
        self.cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=ids))
        return ids, expected

    def _match(self, tokens):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )

    def _assert_rows(self, ids, expected, *, swa_tail: int):
        full, swa = self.rows.read(ids)
        self.assertTrue(torch.equal(full, expected[0]))
        self.assertTrue(torch.equal(swa[-swa_tail:], expected[1][-swa_tail:]))

    def _live_pages(self, ids):
        full = self.rows.full_rows(ids)[::PAGE] // PAGE
        swa = self.rows.swa_rows(ids)[::PAGE] // PAGE
        return full, swa


class TestHostTransferLifecycles(_UnifiedHiCacheCase):
    """D2H sources and H2D targets are owned until their acks."""

    def test_write_through_source_stays_locked_until_the_ack(self):
        self._cache(write_policy="write_through")
        tokens = list(range(1, 17))
        ids, expected = self._insert(tokens, 100)
        full_pages, _ = self._live_pages(ids)
        # Write-through queued the backup at insert; submit it.
        self.cache.flush_pending_backups()
        self.assertTrue(self.controller.has_inflight_device_transfers())
        # The tree lock keeps eviction from turning the source into free pages.
        evicted = self.cache.evict(
            EvictParams(num_tokens=len(tokens), swa_num_tokens=len(tokens))
        )
        self.assertEqual(
            (evicted.num_tokens_evicted, evicted.swa_num_tokens_evicted), (0, 0)
        )
        full_holes = self.allocator.full_attn_allocator._free_phys_pages
        self.assertFalse(bool(torch.isin(full_pages, full_holes).any()))
        # An allocation during the copy takes other pages and leaves the source.
        fresh = self.allocator.alloc(2 * PAGE)
        self.rows.seed(fresh, 600)
        self.assertFalse(
            bool(torch.isin(self.rows.full_rows(fresh), self.rows.full_rows(ids)).any())
        )
        self._assert_rows(ids, expected, swa_tail=self.window)
        self.assertEqual(self.waits, [])

        self.cache.writing_check()
        self.assertFalse(self.controller.has_inflight_device_transfers())
        self.allocator.free(fresh)
        evicted = self.cache.evict(EvictParams(num_tokens=len(tokens)))
        self.assertEqual(evicted.num_tokens_evicted, len(tokens))
        # The host copy is the source data.
        self.rows.full[self.rows.full_rows(ids)] = self.rows.markers(0, len(tokens))
        host = self._match(tokens).last_host_node
        self.assertTrue(self.cache.load_back(host))
        self.cache.ready_to_load_host_cache()
        self.cache.loading_check()
        self._assert_rows(
            self._match(tokens).device_indices, expected, swa_tail=self.window
        )
        self.cache.sanity_check()

    def test_write_back_frees_rows_only_after_the_copy(self):
        self._cache(write_policy="write_back")
        tokens = list(range(1, 17))
        ids, expected = self._insert(tokens, 100)
        full = self.allocator.full_attn_allocator
        engine = self.controller.l2_transfer_engine
        submit = engine.submit_device_to_host
        copies = []

        def owned(sources):
            return all(full.is_slot_allocated(int(i)) for i in sources)

        class _Finish:
            """The copy's finish event; the demote must wait on it first."""

            def __init__(self, event, record):
                self.event, self.record = event, record

            def query(self):
                return self.event.query()

            def synchronize(self):
                self.record["owned_at_sync"] = owned(self.record["sources"])
                self.event.synchronize()

        def recorded(transfers):
            sources = transfers[0].device_indices.clone()
            completion = submit(transfers)
            record = {"sources": sources, "owned_at_copy": owned(sources)}
            copies.append(record)
            return completion._replace(
                finish_event=_Finish(completion.finish_event, record)
            )

        with mock.patch.object(engine, "submit_device_to_host", side_effect=recorded):
            evicted = self.cache.evict(EvictParams(num_tokens=len(tokens)))

        self.assertEqual(evicted.num_tokens_evicted, len(tokens))
        self.assertTrue(copies)
        copied = torch.cat([record["sources"] for record in copies])
        self.assertTrue(torch.equal(copied.sort().values, ids.sort().values))
        for record in copies:
            self.assertTrue(record["owned_at_copy"])
            self.assertTrue(record["owned_at_sync"])
        self.assertFalse(any(full.is_slot_allocated(int(i)) for i in ids))
        host = self._match(tokens).last_host_node
        self.assertTrue(self.cache.tree_core.is_backuped(host))
        self.assertTrue(self.cache.load_back(host))
        self.cache.ready_to_load_host_cache()
        self.cache.loading_check()
        self._assert_rows(
            self._match(tokens).device_indices, expected, swa_tail=self.window
        )
        self.cache.sanity_check()

    def test_load_back_targets_stay_owned_until_the_ack(self):
        self._cache(write_policy="write_through")
        tokens = list(range(1, 17))
        ids, expected = self._insert(tokens, 100)
        self.cache.flush_pending_backups()
        self.cache.writing_check()
        self.cache.evict(EvictParams(num_tokens=len(tokens)))
        host = self._match(tokens).last_host_node
        swa = self.allocator.swa_attn_allocator

        self.assertTrue(self.cache.load_back(host))
        # Reserved, bound into the tree, not queued: SWA cannot move.
        self.assertEqual(swa._pending_hicache_load_pages, self.window // PAGE)
        self.assertTrue(swa.moves_blocked())
        loaded = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).device_indices
        targets = self._live_pages(loaded)
        self.assertEqual(
            self.cache.evict(EvictParams(num_tokens=len(tokens))).num_tokens_evicted, 0
        )

        self.cache.ready_to_load_host_cache()
        # Queued: the reservation is settled, the transfer gate holds the rows.
        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertTrue(self.controller.has_inflight_device_transfers())
        self.assertTrue(swa.moves_blocked())
        fresh = self.allocator.alloc(2 * PAGE)
        self.rows.seed(fresh, 600)
        fresh_pages = self._live_pages(fresh)
        for target, other in zip(targets, fresh_pages):
            self.assertFalse(bool(torch.isin(target, other).any()))
        self.assertEqual(
            self.cache.evict(EvictParams(num_tokens=len(tokens))).num_tokens_evicted, 0
        )
        self.assertEqual(self.waits, [])

        self.cache.loading_check()
        self.assertFalse(self.controller.has_inflight_device_transfers())
        self._assert_rows(loaded, expected, swa_tail=self.window)
        self.assertEqual(self.cache.ongoing_load_back, {})
        self.allocator.free(fresh)
        self.cache.sanity_check()

    def test_buffer_only_source_stays_locked_until_the_d2h_ack(self):
        self._cache(
            write_policy="write_through", tokens=256, host_memory_mode="buffer_only"
        )
        # Buffer-mode backups end in a storage write, which is out of scope:
        # enable the write path and record the storage call instead.
        self.cache.enable_storage = True
        self.controller.prefetch_tokens_occupied = 0
        core, pipeline = self.cache.tree_core, self.cache.buffer_pipeline
        tokens = list(range(1, 17))
        self._insert(tokens, 100)
        node = self._match(tokens).last_device_node
        self.assertTrue(pipeline.pending_write_queue)

        with mock.patch.object(
            self.controller, "write_storage", return_value=1
        ) as write_storage:
            pipeline.flush_pending_writes()
            # Staged and submitted: the source is locked until the D2H ack.
            self.assertTrue(pipeline.ongoing_write_through)
            self.assertGreater(core.get_component_device_lock_ref(node, FULL), 0)
            evicted = self.cache.evict(EvictParams(num_tokens=len(tokens)))
            self.assertEqual(evicted.num_tokens_evicted, 0)
            write_storage.assert_not_called()
            self.assertEqual(self.waits, [])

            self.cache.writing_check()
            # The storage write reads the staging copy, after the ack.
            write_storage.assert_called()
        self.assertEqual(pipeline.ongoing_write_through, {})
        self.assertEqual(core.get_component_device_lock_ref(node, FULL), 0)
        evicted = self.cache.evict(EvictParams(num_tokens=len(tokens)))
        self.assertEqual(evicted.num_tokens_evicted, len(tokens))
        self.cache.sanity_check()


class TestSwaLoadBackReservation(_UnifiedHiCacheCase):
    """Cache-mode SWA load-back loads into physical reservations."""

    window = 3 * PAGE

    def _parent_resident_child_on_host(self):
        self._cache(write_policy="write_through")
        parent_tokens = list(range(1, 9))
        tokens = list(range(1, 17))
        parent_ids, parent_expected = self._insert(parent_tokens, 100)
        child_ids, child_expected = self._insert(tokens, 200)
        self.cache.flush_pending_backups()
        self.cache.writing_check()
        # The leaf goes first; the parent stays resident.
        self.cache.evict(EvictParams(num_tokens=len(tokens) - len(parent_tokens)))
        match = self._match(tokens)
        self.assertEqual(len(match.device_indices), len(parent_tokens))
        return tokens, parent_ids, parent_expected, child_expected, match

    def test_only_the_missing_window_rows_are_reserved_and_bound(self):
        tokens, parent_ids, parent_expected, child_expected, match = (
            self._parent_resident_child_on_host()
        )
        swa = self.allocator.swa_attn_allocator
        parent_swa_rows = self.rows.swa_rows(parent_ids)
        reserve = swa.alloc_physical
        reserved = []

        def recorded(need_size):
            indices = reserve(need_size)
            reserved.append(indices)
            return indices

        entry = self.controller.mem_pool_host.entry_map[PoolName.SWA]
        with mock.patch.object(entry, "device_alloc_fn", side_effect=recorded):
            self.assertTrue(self.cache.load_back(match.last_host_node))
        # The window reaches into the resident parent, which is not reloaded.
        (indices,) = reserved
        self.assertEqual(indices.numel(), 2 * PAGE)
        self.cache.ready_to_load_host_cache()
        self.cache.loading_check()

        loaded = self._match(tokens).device_indices
        child = loaded[len(parent_ids) :]
        self.assertTrue(torch.equal(self.rows.swa_rows(child), indices))
        self.assertTrue(torch.equal(self.rows.swa_rows(parent_ids), parent_swa_rows))
        self._assert_rows(parent_ids, parent_expected, swa_tail=PAGE)
        expected_child = tuple(x[len(parent_ids) :] for x in child_expected)
        self._assert_rows(child, expected_child, swa_tail=2 * PAGE)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])
        self.cache.sanity_check()

    def test_failed_swa_reservation_rolls_back_the_whole_load(self):
        tokens, parent_ids, _, _, match = self._parent_resident_child_on_host()
        full, swa = (
            self.allocator.full_attn_allocator,
            self.allocator.swa_attn_allocator,
        )
        allocated = (full.allocated_count(), swa.allocated_count())

        # The SWA end has no room for the window, even after device eviction.
        entry = self.controller.mem_pool_host.entry_map[PoolName.SWA]
        with mock.patch.object(entry, "device_alloc_fn", return_value=None):
            self.assertFalse(self.cache.load_back(match.last_host_node))

        self.assertEqual((full.allocated_count(), swa.allocated_count()), allocated)
        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertEqual(len(self.controller.load_queue), 0)
        self.assertEqual(self.cache.ongoing_load_back, {})
        self.assertEqual(len(self._match(tokens).device_indices), len(parent_ids))
        self.assertTrue(self.cache.tree_core.is_backuped(match.last_host_node))
        self.assertEqual(self.allocator.verify_byte_accounting(), [])
        self.cache.sanity_check()

    def test_reservation_retries_once_after_device_eviction(self):
        self._cache(write_policy="write_through", tokens=64)
        swa = self.allocator.swa_attn_allocator
        # Tree-owned rows fill the SWA end; once their backups are acked,
        # device eviction may reclaim them.
        fillers = []
        while self.allocator.swa_available_size() >= 2 * PAGE:
            base = 100 * (len(fillers) + 1)
            tokens = list(range(base, base + PAGE))
            fillers.append((tokens, *self._insert(tokens, base)))
        self.cache.flush_pending_backups()
        self.cache.writing_check()
        placed = [self._live_pages(ids)[1] for _, ids, _ in fillers]
        allocated = swa.allocated_count()
        entry = self.controller.mem_pool_host.entry_map[PoolName.SWA]
        reserve, evict = entry.device_alloc_fn, entry.device_evict_fn
        attempts, evictions = [], []

        def recorded_reserve(need_size):
            attempts.append(reserve(need_size))
            return attempts[-1]

        def recorded_evict(need_size):
            evictions.append(need_size)
            return evict(need_size)

        transfer = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(2 * PAGE))
        with (
            mock.patch.object(entry, "device_alloc_fn", side_effect=recorded_reserve),
            mock.patch.object(entry, "device_evict_fn", side_effect=recorded_evict),
        ):
            result = self.controller._resolve_device_transfers(
                [transfer], kv_device_indices=torch.empty(0, dtype=torch.int64)
            )

        self.assertIsNotNone(result)
        self.assertEqual(evictions, [2 * PAGE])
        self.assertEqual(len(attempts), 2)
        self.assertIsNone(attempts[0])
        reserved = transfer.device_indices
        self.assertIs(attempts[1], reserved)
        pages = set((reserved[::PAGE] // PAGE).tolist())
        reclaimed = set()
        for (tokens, ids, expected), before in zip(fillers, placed):
            if len(self._match(tokens).device_indices) == len(tokens):
                # Survivors keep their rows.
                self._assert_rows(ids, expected, swa_tail=PAGE)
                continue
            # The evicted filler is still on host; its SWA page went back to
            # the end and into this reservation.
            self.assertTrue(
                self.cache.tree_core.is_backuped(self._match(tokens).last_host_node)
            )
            reclaimed |= set(before.tolist())
        self.assertTrue(reclaimed)
        self.assertLessEqual(reclaimed, pages)
        # Owned by the reservation alone: counted, and no virtual page maps here.
        page_ids = torch.tensor(sorted(pages))
        self.assertEqual(swa._pending_hicache_load_pages, 2)
        self.assertTrue(bool((swa.physical_to_virtual[page_ids] == -1).all()))
        self.assertFalse(bool(torch.isin(swa.virtual_to_physical, page_ids).any()))
        self.assertEqual(swa.allocated_count(), allocated + (2 - len(reclaimed)) * PAGE)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])

        swa.cancel_physical_reservation(reserved)
        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertEqual(swa.allocated_count(), allocated - len(reclaimed) * PAGE)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])
        self.cache.sanity_check()

    def test_later_pool_failure_cancels_the_swa_reservation(self):
        self._cache(write_policy="write_through")
        swa = self.allocator.swa_attn_allocator
        allocated = swa.allocated_count()
        transfer = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(2 * PAGE))
        # A derived sidecar whose source pool is absent fails after SWA allocated.
        orphan = PoolTransfer(name=PoolName.MAMBA, indices_from_pool=PoolName.INDEXER)

        result = self.controller._resolve_device_transfers(
            [transfer, orphan], kv_device_indices=torch.empty(0, dtype=torch.int64)
        )

        self.assertIsNone(result)
        self.assertIsNone(transfer.device_indices)
        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertEqual(swa.allocated_count(), allocated)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])


class TestBufferOnlyStagedLoad(_UnifiedHiCacheCase):
    """A buffer-only staged load reserves the whole SWA window on the SWA end."""

    window = 3 * PAGE

    def _stage(self, prefix_tokens, span_tokens):
        """Leave the span on host only, as a completed storage fetch does."""
        prefix_ids, prefix_expected = self._insert(prefix_tokens, 100)
        span_ids = self.allocator.alloc(len(span_tokens))
        span_expected = self.rows.seed(span_ids, 200)
        donor = SimpleNamespace(
            rid="donor",
            kv=ReqKvInfo(),
            seqlen=len(prefix_tokens) + len(span_tokens) + 1,
        )
        self.assertIsNotNone(self.req_to_token_pool.alloc([donor]))
        self.req_to_token_pool.write(
            (donor.kv.req_pool_idx, slice(0, donor.seqlen - 1)),
            torch.cat([prefix_ids, span_ids]),
        )
        copied = self.cache.backup_kv_cache(donor)
        self.req_to_token_pool.free(donor)
        self.allocator.free(span_ids)
        # The prefix stays on device; only the span and the SWA window are staged.
        self.cache.host_pool_group.free(copied.host_indices[: len(prefix_tokens)])
        return prefix_ids, prefix_expected, span_expected, copied

    def test_missing_rows_are_bound_and_resident_ones_freed_at_the_ack(self):
        self._cache(
            write_policy="write_through", tokens=256, host_memory_mode="buffer_only"
        )
        swa = self.allocator.swa_attn_allocator
        pipeline = self.cache.buffer_pipeline
        prefix_tokens, span_tokens = list(range(1, 9)), list(range(9, 17))
        tokens = prefix_tokens + span_tokens
        prefix_ids, prefix_expected, span_expected, copied = self._stage(
            prefix_tokens, span_tokens
        )
        (window,) = copied.pool_transfers
        self.assertEqual(len(window.host_indices), self.window)
        staged_kv = copied.host_indices[len(prefix_tokens) :]
        request = CacheRequestHandle("staged", 0)
        occupied = pipeline.host_allocation_units(staged_kv, [window])
        pipeline.staged_prefetches[request] = _StagedPrefetch(
            request=request,
            key_tokens=array("q", tokens),
            extra_key=None,
            cache_salt=None,
            matched_len=len(prefix_tokens),
            num_tokens=len(span_tokens),
            occupied_tokens=occupied,
            host_indices=staged_kv,
            aux_xfers=[window],
            hash_values=["h0", "h1"],
            operation_id=7,
        )
        # Storage attach starts this count; the fetch charged the staging.
        self.controller.prefetch_tokens_occupied = occupied
        match = self._match(tokens)
        req = SimpleNamespace(
            rid=request.rid,
            cache_request_handle=request,
            extra_key=None,
            cache_salt=None,
            prefix_indices=match.device_indices,
            last_node=match.last_device_node,
            kv=SimpleNamespace(cache_protected_len=len(match.device_indices)),
        )
        self.assertTrue(pipeline.prepare_staged_prefetch(req))
        self.assertEqual(
            (req.host_hit_length, req.swa_host_hit_length),
            (len(span_tokens), self.window),
        )
        resident_tail = prefix_ids[-PAGE:]
        resident_rows = self.rows.swa_rows(resident_tail)
        allocated = swa.allocated_count()
        host_available = self.cache.host_pool_group.available_size()

        spliced, _ = self.cache.init_load_back(
            InitLoadBackParams(
                best_match_node=None, host_hit_length=req.host_hit_length, req=req
            )
        )

        self.assertEqual(len(spliced), len(span_tokens))
        # The whole window is reserved; only the new rows are bound to it.
        self.assertEqual(swa._pending_hicache_load_pages, self.window // PAGE)
        self.assertEqual(swa.allocated_count(), allocated + self.window)
        self.assertTrue(torch.equal(self.rows.swa_rows(resident_tail), resident_rows))
        (load,) = pipeline.ongoing_buffer_load_back.values()
        ((pool, redundant),) = load.aux_device_releases
        self.assertEqual(pool, PoolName.SWA)
        self.assertEqual(redundant.numel(), PAGE)
        redundant_pages = redundant[::PAGE] // PAGE
        self.assertTrue(bool((swa.physical_to_virtual[redundant_pages] == -1).all()))
        self.assertFalse(bool(torch.isin(self.rows.swa_rows(spliced), redundant).any()))

        self.cache.ready_to_load_host_cache()
        # Queued: the reservation is settled and the redundant rows wait for the ack.
        self.assertEqual(swa._pending_hicache_load_pages, 0)
        self.assertEqual(swa.allocated_count(), allocated + self.window)
        self.cache.loading_check()

        self.assertEqual(pipeline.ongoing_buffer_load_back, {})
        self.assertEqual(swa.allocated_count(), allocated + len(span_tokens))
        self.assertTrue(bool(torch.isin(redundant_pages, swa._free_phys_pages).all()))
        self.assertGreater(self.cache.host_pool_group.available_size(), host_available)
        self.assertEqual(self.controller.prefetch_tokens_occupied, 0)
        self._assert_rows(spliced, span_expected, swa_tail=len(span_tokens))
        self._assert_rows(prefix_ids, prefix_expected, swa_tail=PAGE)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])
        self.cache.sanity_check()


class TestRetractionRoundTrip(_UnifiedHiCacheCase):
    """Host-pool retraction of a FULL+SWA request on the unified pool."""

    def test_backup_and_restore_carry_rows_and_keep_the_shared_prefix(self):
        self._cache(write_policy="write_back")
        prefix_tokens = list(range(1, 9))
        prefix_ids, prefix_expected = self._insert(prefix_tokens, 100)
        prefix_node = self._match(prefix_tokens).last_device_node
        # The request shares the cached prefix and owns its own suffix rows.
        pin = self.cache.inc_lock_ref(prefix_node).to_dec_params()
        own_ids = self.allocator.alloc(2 * PAGE)
        own_expected = self.rows.seed(own_ids, 200)
        req = SimpleNamespace(rid="retracted", kv=ReqKvInfo(), seqlen=17)
        self.assertIsNotNone(self.req_to_token_pool.alloc([req]))
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, 16)), torch.cat([prefix_ids, own_ids])
        )
        expected = tuple(
            torch.cat(parts) for parts in zip(prefix_expected, own_expected)
        )
        host_available = self.cache.host_pool_group.available_size()

        backup = self.cache.backup_kv_cache(req)
        self.assertIsNotNone(backup)
        self.assertLess(self.cache.host_pool_group.available_size(), host_available)
        self.assertEqual(
            {transfer.name for transfer in backup.pool_transfers or []}, {PoolName.SWA}
        )

        # Retraction releases only the request's rows and its pin on the prefix.
        self.allocator.free(own_ids)
        self.cache.dec_lock_ref(prefix_node, pin)
        reused = self.allocator.alloc(2 * PAGE)
        self.rows.seed(reused, 700)
        self._assert_rows(prefix_ids, prefix_expected, swa_tail=PAGE)
        self.assertTrue(
            torch.equal(self._match(prefix_tokens).device_indices, prefix_ids)
        )

        destination = self.allocator.alloc(16)
        self.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 16)), destination)
        self.cache.restore_kv_cache(req, backup)

        self._assert_rows(destination, expected, swa_tail=self.window)
        self._assert_rows(prefix_ids, prefix_expected, swa_tail=PAGE)
        self.assertEqual(self.cache.host_pool_group.available_size(), host_available)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])
        self.allocator.free(reused)
        self.allocator.free(destination)
        self.req_to_token_pool.free(req)
        self.cache.sanity_check()


class TestLoadBackAdmission(_UnifiedHiCacheCase):
    """PrefillAdder load-back admission over the real cache and shared budget."""

    def _adder(self):
        running = mock.MagicMock()
        running.reqs = []
        return PrefillAdder(
            page_size=PAGE,
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            running_batch=running,
            new_token_ratio=1.0,
            rem_input_tokens=10_000,
            rem_chunk_tokens=None,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )

    def _request(self, tokens):
        match = self._match(tokens)
        req = mock.MagicMock(spec=Req)
        req.rid = "load-back"
        req.cache_request_handle = CacheRequestHandle(req.rid, 0)
        req.priority = 0
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.best_match_node = match.best_match_node
        req.full_untruncated_fill_ids = list(tokens)
        req.output_ids = []
        req.sampling_params = SimpleNamespace(max_new_tokens=PAGE, ignore_eos=False)
        req.host_hit_length = match.host_hit_length
        req.swa_host_hit_length = match.swa_host_hit_length
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.retracted_stain = False
        req.needs_host_load_back.return_value = True
        req.finished.return_value = False
        req.materialized_host_hit_len.return_value = match.host_hit_length
        req.fulfilled_storage_hit_len.return_value = 0
        req.kv = SimpleNamespace(
            cache_protected_len=len(match.device_indices), mamba_pool_idx=None
        )
        req.set_extend_range = mock.MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        return req

    def _locks(self, node):
        core = self.cache.tree_core
        return (
            core.get_component_device_lock_ref(node, FULL),
            core.get_component_host_lock_ref(node, FULL),
            core.get_component_host_lock_ref(node, SWA),
        )

    def _host_pressure(self):
        """A host-only match, a host pool with almost no room, and an unbacked
        device victim whose write-back needs host room."""
        self._cache(write_policy="write_back", tokens=64)
        match_tokens = list(range(1, 17))
        _, expected = self._insert(match_tokens, 100)
        self.cache.evict(EvictParams(num_tokens=len(match_tokens)))
        host = self.cache.host_pool_group
        fillers = []
        while host.available_size() >= 2 * len(match_tokens):
            tokens = list(range(1000 + 100 * len(fillers), 1016 + 100 * len(fillers)))
            self._insert(tokens, 300)
            self.cache.evict(EvictParams(num_tokens=len(tokens)))
            fillers.append(tokens)
        self._insert(list(range(5000, 5040)), 500)
        return match_tokens + list(range(17, 25)), expected, fillers

    def test_host_match_stays_pinned_while_reclaim_evicts_host_leaves(self):
        tokens, expected, fillers = self._host_pressure()
        req = self._request(tokens)
        node = req.best_match_node
        self.assertEqual(req.host_hit_length, 16)
        self.assertEqual(self._locks(node), (0, 0, 0))
        evict_host = self.cache.evict_host
        pins_during_host_eviction = []

        def recorded(*args, **kwargs):
            pins_during_host_eviction.append(self._locks(node)[1:])
            return evict_host(*args, **kwargs)

        with mock.patch.object(self.cache, "evict_host", side_effect=recorded):
            verdict = self._adder().add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )

        self.assertIs(verdict, AddReqResult.CONTINUE)
        # Reclaim had to make host room, and the match was pinned meanwhile.
        self.assertTrue(pins_during_host_eviction)
        for full_pin, swa_pin in pins_during_host_eviction:
            self.assertGreater(full_pin, 0)
            self.assertGreater(swa_pin, 0)
        self.assertEqual(req.host_loaded_length, 16)
        self.cache.ready_to_load_host_cache()
        self.cache.loading_check()
        # The CPU copy survived until the load, so the request sees its data.
        full, swa = self.rows.read(req.prefix_indices)
        self.assertTrue(torch.equal(full, expected[0]))
        self.assertTrue(torch.equal(swa[-self.window :], expected[1][-self.window :]))
        # Host room came from the other host leaves instead.
        self.assertLess(
            sum(self._match(f).host_hit_length for f in fillers), 16 * len(fillers)
        )
        # Only the request's own device lock is left once the load is acked.
        self.assertEqual(self._locks(node), (1, 0, 0))
        self.cache.sanity_check()

    def test_pins_are_released_when_the_load_back_budget_fails(self):
        tokens, _, _ = self._host_pressure()
        req = self._request(tokens)
        node = req.best_match_node
        adder = self._adder()

        with mock.patch.object(
            adder.memory_budget, "prepare_load_back", return_value=False
        ):
            verdict = adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )

        self.assertIs(verdict, AddReqResult.NO_TOKEN)
        self.assertEqual(self._locks(node), (0, 0, 0))
        self.assertEqual(len(self.controller.load_queue), 0)
        self.assertEqual(self._match(tokens).host_hit_length, 16)
        self.cache.sanity_check()

    def test_pins_are_released_when_the_load_back_raises(self):
        tokens, _, _ = self._host_pressure()
        req = self._request(tokens)
        node = req.best_match_node

        with mock.patch.object(
            self.cache, "load_back", side_effect=RuntimeError("load failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                self._adder().add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )

        self.assertEqual(self._locks(node), (0, 0, 0))
        self.assertEqual(self._match(tokens).host_hit_length, 16)
        self.cache.sanity_check()

    def _resident_full_behind_host_swa(self):
        """FULL stays resident under a host-only SWA window, so the host hit
        overstates the FULL slots a load adds."""
        self._cache(write_policy="write_through", tokens=64)
        tokens = list(range(1, 17))
        self._insert(tokens, 100)
        self.cache.flush_pending_backups()
        self.cache.writing_check()
        self.cache.evict(EvictParams(swa_num_tokens=len(tokens)))
        tokens = tokens + list(range(17, 21))
        req = self._request(tokens)
        full_spec = self.cache.tree_core.build_hicache_transfers(
            FULL, req.best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]
        self.assertLess(len(full_spec.host_indices), req.host_hit_length)
        return req, len(full_spec.host_indices)

    def _admit_recording(self, adder, req):
        core = self.cache.tree_core
        build = core.build_hicache_transfers
        prepare = adder.memory_budget.prepare_load_back
        full_specs, full_tokens = [], []

        def recorded_build(component_type, *args, **kwargs):
            if component_type == FULL:
                full_specs.append(args)
            return build(component_type, *args, **kwargs)

        def recorded_prepare(**kwargs):
            full_tokens.append(kwargs["full_tokens"])
            return prepare(**kwargs)

        with (
            mock.patch.object(
                core, "build_hicache_transfers", side_effect=recorded_build
            ),
            mock.patch.object(
                adder.memory_budget, "prepare_load_back", side_effect=recorded_prepare
            ),
        ):
            verdict = adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )
        return verdict, full_specs, full_tokens

    def test_shared_budget_counts_only_the_full_slots_a_load_adds(self):
        req, new_full = self._resident_full_behind_host_swa()
        adder = self._adder()
        extend = len(req.full_untruncated_fill_ids) - req.host_hit_length

        verdict, full_specs, full_tokens = self._admit_recording(adder, req)

        self.assertIs(verdict, AddReqResult.CONTINUE)
        self.assertEqual(len(full_specs), 1)
        self.assertEqual(full_tokens, [new_full + extend + PAGE + PAGE])
        self.cache.ready_to_load_host_cache()
        self.cache.loading_check()
        self.cache.sanity_check()

    def test_budgets_without_full_tokens_skip_the_full_spec(self):
        req, _ = self._resident_full_behind_host_swa()
        adder = self._adder()
        adder.memory_budget = PrefillBudget(self.allocator, self.cache)
        extend = len(req.full_untruncated_fill_ids) - req.host_hit_length

        verdict, full_specs, full_tokens = self._admit_recording(adder, req)

        self.assertIs(verdict, AddReqResult.CONTINUE)
        # The base budget ignores full_tokens, so the host hit bounds it.
        self.assertEqual(full_specs, [])
        self.assertEqual(full_tokens, [req.host_hit_length + extend + PAGE + PAGE])
        self.cache.ready_to_load_host_cache()
        self.cache.loading_check()
        self.cache.sanity_check()


class TestWritePolicySwitch(_UnifiedHiCacheCase):
    """Attaching storage with write_through over a write_back host tree."""

    def _incompatible_tree(self, kind: str):
        self._cache(write_policy="write_back", tokens=64)
        if kind == "aux host rows without FULL":
            tokens = list(range(1, 17))
            self._insert(tokens, 100)
            node = self._match(tokens).last_device_node
            # Back the node up, then drop only its FULL host copy.
            self.cache.backup_node_for_write_back(node)
            self.cache.evict_host(len(tokens), FULL)
        else:  # "host suffix below an unbacked FULL parent"
            self._insert(list(range(1, 9)), 100)
            self._insert(list(range(1, 17)), 200)
            # Write-back backs up only the evicted leaf.
            self.cache.evict(EvictParams(num_tokens=8))
        self.assertFalse(self.cache.tree_core.is_write_through_compatible())

    def _policy_state(self):
        cache, controller = self.cache, self.controller
        return (
            controller.write_policy,
            cache.write_through_threshold,
            cache.is_write_back,
            cache.prefetch_stop_policy,
            cache.enable_storage,
            controller.storage_backend,
            controller.storage_backend_type,
        )

    def _attach(self, **kwargs):
        return self.cache._storage_attachment.attach(
            "mori", hicache_write_policy="write_through", **kwargs
        )

    def test_incompatible_host_trees_are_rejected_without_side_effects(self):
        for kind in (
            "aux host rows without FULL",
            "host suffix below an unbacked FULL parent",
        ):
            with self.subTest(kind=kind):
                self._incompatible_tree(kind)
                before = self._policy_state()
                with mock.patch.object(
                    self.controller, "attach_storage_backend"
                ) as attach_backend:
                    ok, message = self._attach()
                self.assertFalse(ok)
                self.assertIn("from write_back to write_through", message)
                attach_backend.assert_not_called()
                self.assertEqual(self._policy_state(), before)
                self.assertFalse(self.cache.tree_core.is_write_through_compatible())
                self.cache.sanity_check()

    def test_compatible_host_tree_switches_policy(self):
        self._cache(write_policy="write_back", tokens=64)
        tokens = list(range(1, 17))
        self._insert(tokens, 100)
        # A chain backed up from the root down is a valid write-through state.
        core, node, path = (
            self.cache.tree_core,
            self._match(tokens).last_device_node,
            [],
        )
        while node != self.cache.root_node_handle():
            path.append(node)
            node = core.get_parent_node_id(node)
        for node in reversed(path):
            self.assertTrue(self.cache.backup_node_for_write_back(node))
        self.assertTrue(self.cache.tree_core.is_write_through_compatible())
        with mock.patch.object(self.controller, "attach_storage_backend") as attach:
            ok, _ = self._attach()
        self.assertTrue(ok)
        attach.assert_called_once()
        self.assertEqual(self.controller.write_policy, "write_through")
        self.assertFalse(self.cache.is_write_back)
        self.assertEqual(self.cache.write_through_threshold, 1)

    def test_buffer_only_mode_is_not_checked(self):
        # Buffer-only host memory stages transfers and holds no cache tree, so
        # the check does not apply, even after a runtime switch to write_back.
        self._cache(
            write_policy="write_through", tokens=256, host_memory_mode="buffer_only"
        )
        self.assertIsNotNone(self.cache.buffer_pipeline)
        self.cache._storage_attachment._apply_policies(None, "write_back")
        self.assertTrue(self.cache.is_write_back)
        with (
            mock.patch.object(
                self.cache.tree_core, "is_write_through_compatible", return_value=False
            ) as compatible,
            mock.patch.object(self.controller, "attach_storage_backend"),
        ):
            ok, message = self._attach()
        self.assertTrue(ok, message)
        compatible.assert_not_called()
        self.assertEqual(self.controller.write_policy, "write_through")
        self.assertFalse(self.cache.is_write_back)

    def test_runtime_attach_to_an_envelope_arena_fails_before_any_backend(self):
        # The page-envelope host arena exists from startup; a backend that needs
        # separate host pools is refused when it is attached, and the policies
        # applied for the attempt are rolled back.
        self._cache(write_policy="write_back", tokens=64)
        before = self._policy_state()
        with mock.patch(
            "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend"
        ) as create:
            ok, message = self.cache._storage_attachment.attach(
                "file", hicache_write_policy="write_through"
            )
        self.assertFalse(ok)
        self.assertIn("requires separate host pools", message)
        create.assert_not_called()
        self.assertEqual(self._policy_state(), before)


class TestSwaWriteBackEviction(_UnifiedHiCacheCase):
    """SWA eviction of a write-back node copies nothing it does not need."""

    def _record_d2h(self):
        """Patch the D2H submit to record each copy's row counts, then run it."""
        engine = self.controller.l2_transfer_engine
        submit = engine.submit_device_to_host
        copies = []

        def recorded(transfers):
            copies.append([len(t.host_indices) for t in transfers])
            return submit(transfers)

        return (
            mock.patch.object(engine, "submit_device_to_host", side_effect=recorded),
            copies,
        )

    def _parent_and_child(self, *, before_insert=lambda: None):
        self._cache(write_policy="write_back", tokens=64)
        before_insert()
        parent_ids, parent_expected = self._insert(list(range(1, 9)), 100)
        tokens = list(range(1, 17))
        ids, expected = self._insert(tokens, 200)
        # The insert kept the cached parent rows and freed its own duplicates.
        kept = torch.cat([parent_ids, ids[8:]])
        kept_full = torch.cat([parent_expected[0], expected[0][8:]])
        return tokens, kept, kept_full

    def test_hosted_swa_window_is_dropped_without_a_full_write_back(self):
        tokens, ids, expected_full = self._parent_and_child()
        parent = self.cache.tree_core.get_parent_node_id(
            self._match(tokens).last_device_node
        )
        # The internal parent's SWA has a host copy; its FULL has none.
        self.cache.backup_node_for_write_back(parent)
        self.cache.evict_host(8, FULL)
        core = self.cache.tree_core
        self.assertIsNone(core.get_component_host_value(parent, FULL))
        self.assertIsNotNone(core.get_component_host_value(parent, SWA))
        host_available = self.cache.host_pool_group.available_size()
        recording, copies = self._record_d2h()

        with recording:
            evicted = self.cache.evict(EvictParams(swa_num_tokens=8))

        self.assertGreaterEqual(evicted.swa_num_tokens_evicted, 8)
        self.assertEqual(evicted.num_tokens_evicted, 0)
        self.assertEqual(copies, [])
        self.assertEqual(self.cache.host_pool_group.available_size(), host_available)
        self.assertIsNone(core.get_component_host_value(parent, FULL))
        # FULL rows of the parent and the descendant are untouched.
        full, _ = self.rows.read(ids)
        self.assertTrue(torch.equal(full, expected_full))
        self.cache.sanity_check()

    def test_unbacked_swa_window_is_backed_up_before_it_is_dropped(self):
        tokens, ids, expected_full = self._parent_and_child()
        parent = self.cache.tree_core.get_parent_node_id(
            self._match(tokens).last_device_node
        )
        core = self.cache.tree_core
        self.assertIsNone(core.get_component_host_value(parent, SWA))
        recording, copies = self._record_d2h()

        with recording:
            evicted = self.cache.evict(EvictParams(swa_num_tokens=8))

        self.assertGreaterEqual(evicted.swa_num_tokens_evicted, 8)
        self.assertEqual(evicted.num_tokens_evicted, 0)
        self.assertEqual(len(copies), 1)
        self.assertIsNotNone(core.get_component_host_value(parent, SWA))
        full, _ = self.rows.read(ids)
        self.assertTrue(torch.equal(full, expected_full))
        self.cache.sanity_check()

    def _pin_host_full(self, pins):
        """Fill the host with pinned copies until a FULL+SWA page pair no
        longer fits, so the window backup below must fail."""
        host = self.cache.host_pool_group
        while host.available_size() >= 2 * PAGE:
            filler = list(range(1000 + 100 * len(pins), 1000 + 100 * len(pins) + PAGE))
            self._insert(filler, 300)
            self.cache.evict(EvictParams(num_tokens=len(filler)))
            match = self._match(filler)
            self.assertEqual(match.host_hit_length, len(filler))
            node = match.last_host_node
            pins.append((node, self.cache.inc_host_lock_ref(node).to_dec_params()))

    def test_host_pressure_drops_only_the_swa_window(self):
        pins = []
        tokens, ids, expected_full = self._parent_and_child(
            before_insert=lambda: self._pin_host_full(pins)
        )
        core = self.cache.tree_core
        parent = core.get_parent_node_id(self._match(tokens).last_device_node)
        self.assertIsNone(core.get_component_host_value(parent, SWA))
        recording, copies = self._record_d2h()

        with recording:
            evicted = self.cache.evict(EvictParams(swa_num_tokens=8))

        self.assertGreaterEqual(evicted.swa_num_tokens_evicted, 8)
        self.assertEqual(evicted.num_tokens_evicted, 0)
        self.assertEqual(copies, [])
        self.assertIsNone(core.get_component_host_value(parent, SWA))
        self.assertIsNone(core.get_component_host_value(parent, FULL))
        # FULL rows of the parent and the descendant survive the failed backup.
        full, _ = self.rows.read(ids)
        self.assertTrue(torch.equal(full, expected_full))
        for node, params in pins:
            self.cache.dec_host_lock_ref(node, params)
        self.cache.sanity_check()


class TestPerLayerLoadWait(_UnifiedHiCacheCase):
    """Attention reads still wait on the load of their own layer."""

    def test_each_layer_read_waits_on_its_own_load_event(self):
        self._cache(write_policy="write_through")
        tokens = list(range(1, 17))
        self._insert(tokens, 100)
        self.cache.flush_pending_backups()
        self.cache.writing_check()
        self.cache.evict(EvictParams(num_tokens=len(tokens)))
        counter = self.controller.layer_done_counter
        self.kv_pool.register_layer_transfer_counter(counter)
        self.assertTrue(self.cache.load_back(self._match(tokens).last_host_node))

        producer = self.cache.ready_to_load_host_cache()
        # An allocation after the load is queued adds no wait of its own.
        fresh = self.allocator.alloc(2 * PAGE)
        self.assertEqual(self.waits, [])
        counter.set_consumer(producer)
        layer_events = counter.events[producer].load_events
        reads = []
        with mock.patch.object(
            cache_controller.device_module, "current_stream", lambda: _Stream(reads)
        ):
            for layer in range(self.kv_pool.layer_num):
                self.kv_pool.get_key_buffer(layer)
        self.assertEqual(reads, [("wait", event) for event in layer_events])
        self.cache.loading_check()
        self.allocator.free(fresh)
        self.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
