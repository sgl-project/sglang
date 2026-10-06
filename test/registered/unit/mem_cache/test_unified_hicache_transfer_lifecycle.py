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
"""Unified-memory HiCache on the real SWA allocator stack: physical reservations
and the moves they block, the transfer-done event waits, SWA load-back into
reservations, load-back admission, and the write-policy switch check.

Allocation takes only free pages, so it does not wait on HiCache copies; moves
and frees still do. The tests check the rows' contents, not only the bookkeeping.

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

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
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


def _unified_swa_bundle(*, tokens: int = 64):
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
        lazy_compaction=True,
    )


def _unified_swa():
    bundle = _unified_swa_bundle()
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
    """`alloc_physical` / bind / cancel on the SWA end, and the moves it blocks."""

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


class TestFloatSwaReservation(_ConfigCase):
    """The tri-pool's SWA middle floats: it relocates its pages to make room for
    a neighbour. A load's SWA reservation pins it until the load is queued or
    cancelled, as on an end pool."""

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

    def _controller(self) -> HybridCacheController:
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
                    device_evict_fn=None,
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
        self.assertCountEqual(log[:2], [("wait", backup_done), ("wait", load_done)])
        self.assertEqual(log[2], ("move", "swa"))
        self.assertEqual(swa._hicache_transfer_done_events, {})
        self.assertTrue(torch.equal(rows.read(tail)[1], expected_tail[1]))

        # Releasing a reservation also orders after outstanding copies.
        reserved = swa.alloc_physical(PAGE)
        allocator.set_hicache_transfer_done_event(H2D, load_done)
        with _recorded_stream() as waits:
            swa.free_physical(reserved)
        self.assertEqual(waits, [("wait", load_done)])
        allocator.free(head)


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
    ):
        server_args = ServerArgs(
            model_path="dummy",
            page_size=PAGE,
            enable_unified_memory=True,
            hicache_write_policy=write_policy,
            hicache_mem_layout="layer_first",
            hicache_host_memory_mode="cache",
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
        stream.__enter__()
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

    def test_reservation_retries_once_after_device_eviction(self):
        self._cache(write_policy="write_through", tokens=64)
        swa = self.allocator.swa_attn_allocator
        # Tree-owned rows fill the SWA end; once their backups are acked,
        # device eviction may reclaim them.
        fillers = 0
        while self.allocator.swa_available_size() >= 2 * PAGE:
            fillers += 1
            self._insert(
                list(range(100 * fillers, 100 * fillers + PAGE)), 100 * fillers
            )
        self.cache.flush_pending_backups()
        self.cache.writing_check()
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

        # No room on the first try; one device eviction makes it, and the retry's
        # reservation becomes the transfer's destination.
        self.assertIsNotNone(result)
        self.assertEqual(evictions, [2 * PAGE])
        self.assertEqual(len(attempts), 2)
        self.assertIsNone(attempts[0])
        self.assertIs(attempts[1], transfer.device_indices)
        self.assertEqual(swa._pending_hicache_load_pages, 2)

        swa.cancel_physical_reservation(transfer.device_indices)
        self.assertEqual(swa._pending_hicache_load_pages, 0)
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


class TestLoadBackAdmission(_UnifiedHiCacheCase):
    """PrefillAdder load-back admission over the real cache and shared budget."""

    def _adder(self, num_mixed_decode_tokens: int = 0):
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
            num_mixed_decode_tokens=num_mixed_decode_tokens,
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

    def _resident_full_behind_host_swa(self, *, tokens: int = 64):
        """FULL stays resident under a host-only SWA window, so the host hit
        overstates the FULL slots a load adds."""
        self._cache(write_policy="write_through", tokens=tokens)
        tokens = list(range(1, 17))
        _, self.expected = self._insert(tokens, 100)
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

    def test_shared_budget_reclaims_for_the_load_and_pending_demand(self):
        req, new_full = self._resident_full_behind_host_swa(tokens=80)
        # A running request locks its whole path; only its SWA past the window
        # can be reclaimed.
        running = list(range(5000, 5040))
        running_ids, running_expected = self._insert(running, 500)
        self.cache.flush_pending_backups()
        self.cache.writing_check()
        running_node = self._match(running).last_device_node
        pin = self.cache.inc_lock_ref(running_node).to_dec_params()
        reclaimable_swa = self.cache.swa_evictable_size()
        self.assertGreater(reclaimable_swa, 0)
        # The batch's decode tokens are pending demand on both bands. With
        # them, the load fits the shared gap only after that SWA is reclaimed.
        decode = 6 * PAGE
        adder = self._adder(num_mixed_decode_tokens=decode)
        extend = len(req.full_untruncated_fill_ids) - req.host_hit_length
        moves = []
        _record_moves(self.allocator, moves)

        verdict, full_specs, full_tokens = self._admit_recording(adder, req)

        self.assertIs(verdict, AddReqResult.CONTINUE)
        # The FULL ask counts only the slots the load adds; the budget itself
        # adds the pending demand.
        self.assertEqual(len(full_specs), 1)
        self.assertEqual(full_tokens, [new_full + extend + PAGE + PAGE])
        # The queued load blocks every page move. The batch's own extend and
        # the pending decode tokens still fit.
        self.assertEqual(len(self.controller.load_queue), 1)
        self.assertTrue(self.allocator.full_attn_allocator.moves_blocked())
        moved = len(moves)
        own = self.allocator.alloc(PAGE)
        pending = self.allocator.alloc(decode)
        self.assertIsNotNone(own)
        self.assertIsNotNone(pending)
        self.assertEqual(len(moves), moved)
        # That room came from reclaiming the running request's out-of-window
        # SWA; its FULL rows and its window stay.
        self.assertLess(self.cache.swa_evictable_size(), reclaimable_swa)
        self._assert_rows(running_ids, running_expected, swa_tail=self.window)

        self.cache.ready_to_load_host_cache()
        self.cache.loading_check()
        full, swa = self.rows.read(req.prefix_indices)
        self.assertTrue(torch.equal(full, self.expected[0]))
        self.assertTrue(
            torch.equal(swa[-self.window :], self.expected[1][-self.window :])
        )
        self._assert_rows(running_ids, running_expected, swa_tail=self.window)
        # Only the request's own device lock is left once the load is acked.
        self.assertEqual(self._locks(req.best_match_node), (1, 0, 0))
        self.allocator.free(torch.cat([own, pending]))
        self.cache.dec_lock_ref(running_node, pin)
        self.assertEqual(self.allocator.verify_byte_accounting(), [])
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


if __name__ == "__main__":
    unittest.main()
