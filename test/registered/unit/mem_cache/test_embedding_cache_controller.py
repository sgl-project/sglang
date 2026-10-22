"""Unit tests for EmbeddingCacheController paged host pool behavior."""

import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.embedding_cache_controller import (
    EmbeddingCacheController,
    EmbeddingCacheEntry,
    EmbeddingPool,
    EntryState,
    EvictableLRU,
    PageRun,
    RangePageAllocator,
    _build_host_device_transfer_plan,
    _build_storage_transfer_buffers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_pool(num_pages=16, dim=4, page_size=2, modality="vision"):
    total_tokens = num_pages * page_size
    tensor = torch.empty((total_tokens, dim), dtype=torch.float32)
    return EmbeddingPool(
        modality=modality,
        dim=dim,
        dtype=torch.float32,
        page_size=page_size,
        tensor=tensor,
        num_pages=num_pages,
        allocator=RangePageAllocator(num_pages),
        page_bytes=page_size * dim * torch.float32.itemsize,
        pool_size_bytes=total_tokens * dim * torch.float32.itemsize,
        pin_memory=False,
    )


def _make_controller(num_pages=16, dim=4, page_size=2, enable_eviction=True):
    ctrl = EmbeddingCacheController.__new__(EmbeddingCacheController)
    ctrl.tp_world_size = 1
    ctrl.tp_group = None
    ctrl.tp_rank = 0
    ctrl.all_rank_get = False
    ctrl.hidden_dims = {
        Modality.IMAGE: dim,
        Modality.VIDEO: dim,
        Modality.AUDIO: dim,
    }
    ctrl.dtype = torch.float32
    ctrl.element_size = torch.float32.itemsize
    ctrl.enable_eviction = enable_eviction
    ctrl.max_eviction_batch = 10
    ctrl.embedding_store = MagicMock()
    ctrl.total_pool_size_bytes = num_pages * page_size * dim * torch.float32.itemsize
    ctrl.vision_pool = _make_pool(num_pages, dim, page_size)
    ctrl.audio_pool = _make_pool(num_pages, dim, page_size, modality="audio")
    ctrl.pools = {"vision": ctrl.vision_pool, "audio": ctrl.audio_pool}
    ctrl.entries = {}
    ctrl.vision_pool.evictable = EvictableLRU()
    ctrl.audio_pool.evictable = EvictableLRU()
    ctrl.stats = {
        "total_allocated": 0,
        "total_evicted": 0,
        "eviction_count": 0,
        "allocation_failures": 0,
    }
    ctrl.ongoing_prefetch = {}
    ctrl.prefetch_queue = Queue()
    ctrl.insert_queue = Queue()
    ctrl.lock = threading.Lock()
    ctrl.stop_event = threading.Event()
    ctrl.io_thread = MagicMock()
    ctrl.prefetch_tp_group = None
    ctrl._copy_streams = {}
    return ctrl


class TestControllerInitialization(CustomTestCase):
    def test_only_all_rank_get_creates_a_collective_group(self):
        for tp_size, tp_rank, all_rank_get in (
            (1, 0, False),
            (1, 0, True),
            (2, 0, False),
            (2, 1, False),
            (2, 0, True),
            (2, 1, True),
        ):
            with self.subTest(
                tp_size=tp_size, tp_rank=tp_rank, all_rank_get=all_rank_get
            ):
                cpu_group = object()
                parallel = SimpleNamespace(
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    tp_group=SimpleNamespace(cpu_group=cpu_group),
                )
                pools = (_make_pool(), _make_pool(modality="audio"))
                store = MagicMock()
                with (
                    patch(
                        "sglang.srt.mem_cache.embedding_cache_controller.get_parallel",
                        return_value=parallel,
                    ),
                    patch.object(
                        EmbeddingCacheController, "_create_pools", return_value=pools
                    ) as create_pools,
                    patch(
                        "sglang.srt.mem_cache.embedding_cache_controller.threading.Thread"
                    ) as thread,
                    patch.object(
                        torch.distributed,
                        "get_process_group_ranks",
                        return_value=list(range(tp_size)),
                    ) as get_ranks,
                    patch(
                        "sglang.srt.distributed.parallel_state.create_custom_parallel_group"
                    ) as create_group,
                ):
                    ctrl = EmbeddingCacheController(
                        embedding_store=store, all_rank_get=all_rank_get
                    )

                create_pools.assert_called_once_with(pin_memory=True)
                self.assertEqual(store.register_buffer.call_count, 2)
                self.assertIs(
                    store.register_buffer.call_args_list[0].args[0], pools[0].tensor
                )
                self.assertIs(
                    store.register_buffer.call_args_list[1].args[0], pools[1].tensor
                )
                thread.return_value.start.assert_called_once_with()
                if all_rank_get and tp_size > 1:
                    get_ranks.assert_called_once_with(cpu_group)
                    create_group.assert_called_once_with(
                        group_ranks=list(range(tp_size)), backend="gloo"
                    )
                    self.assertIs(ctrl.prefetch_tp_group, create_group.return_value)
                else:
                    get_ranks.assert_not_called()
                    create_group.assert_not_called()
                    self.assertIsNone(ctrl.prefetch_tp_group)

    def test_prefetch_progress_collective_is_opt_in(self):
        for all_rank_get in (False, True):
            with self.subTest(all_rank_get=all_rank_get):
                ctrl = _make_controller()
                ctrl.tp_world_size = 2
                ctrl.all_rank_get = all_rank_get
                ctrl.prefetch_tp_group = object()
                with patch.object(
                    torch.distributed,
                    "all_reduce",
                    side_effect=lambda tensor, **kwargs: tensor.zero_(),
                ) as all_reduce:
                    ready = ctrl.check_prefetch_progress("req")
                self.assertEqual(ready, not all_rank_get)
                if all_rank_get:
                    all_reduce.assert_called_once()
                    self.assertIs(
                        all_reduce.call_args.kwargs["group"], ctrl.prefetch_tp_group
                    )
                    self.assertEqual(
                        all_reduce.call_args.kwargs["op"],
                        torch.distributed.ReduceOp.MIN,
                    )
                else:
                    all_reduce.assert_not_called()


class TestRangePageAllocator(unittest.TestCase):
    def test_prefers_single_contiguous_run(self):
        allocator = RangePageAllocator(num_pages=8)

        runs = allocator.allocate(num_tokens=6, page_size=2)

        self.assertEqual(runs, [PageRun(start=0, length=3)])
        self.assertEqual(allocator.free_ranges, [(3, 5)])

    def test_free_merges_adjacent_ranges(self):
        allocator = RangePageAllocator(num_pages=8)
        first = allocator.allocate(num_tokens=4, page_size=2)
        second = allocator.allocate(num_tokens=4, page_size=2)

        allocator.free(first)
        allocator.free(second)

        self.assertEqual(allocator.free_ranges, [(0, 8)])

    def test_scatter_fallback_returns_physical_order(self):
        allocator = RangePageAllocator(num_pages=4)
        a = allocator.allocate(num_tokens=2, page_size=2)
        b = allocator.allocate(num_tokens=2, page_size=2)
        c = allocator.allocate(num_tokens=2, page_size=2)
        d = allocator.allocate(num_tokens=2, page_size=2)
        allocator.free(a)
        allocator.free(c)

        runs = allocator.allocate(num_tokens=4, page_size=2)

        self.assertEqual(runs, [PageRun(start=0, length=1), PageRun(start=2, length=1)])
        self.assertEqual([run.start for run in runs], sorted(run.start for run in runs))
        self.assertEqual(b, [PageRun(start=1, length=1)])
        self.assertEqual(d, [PageRun(start=3, length=1)])

    def test_allocate_fails_when_total_free_pages_are_insufficient(self):
        allocator = RangePageAllocator(num_pages=2)
        allocator.allocate(num_tokens=4, page_size=2)

        self.assertIsNone(allocator.allocate(num_tokens=2, page_size=2))
        self.assertEqual(allocator.free_pages, 0)


class TestEntryStateAndPins(unittest.TestCase):
    def test_ready_entry_with_no_pins_is_evictable(self):
        entry = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=2,
            dim=4,
            page_runs=[PageRun(0, 1)],
            state=EntryState.READY,
        )

        self.assertTrue(entry.is_evictable())

    def test_ready_entry_with_pin_is_not_evictable(self):
        entry = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=2,
            dim=4,
            page_runs=[PageRun(0, 1)],
            state=EntryState.READY,
        )

        entry.pin()
        self.assertFalse(entry.is_evictable())
        entry.unpin()
        self.assertTrue(entry.is_evictable())

    def test_multiple_pins_require_all_unpins(self):
        entry = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=2,
            dim=4,
            page_runs=[PageRun(0, 1)],
            state=EntryState.READY,
        )

        entry.pin()
        entry.pin()

        self.assertEqual(entry.ref_count, 2)
        self.assertFalse(entry.is_evictable())


class TestEvictableLruInvariant(unittest.TestCase):
    def _insert_entry(
        self,
        ctrl,
        mm_hash,
        modality=Modality.IMAGE,
        state=EntryState.READY,
    ):
        pool = ctrl._get_pool(modality)
        page_runs = pool.allocator.allocate(2, pool.page_size)
        entry = EmbeddingCacheEntry(
            hash=mm_hash,
            modality=modality,
            num_tokens=2,
            dim=pool.dim,
            page_runs=page_runs,
            state=state,
        )
        ctrl.entries[mm_hash] = entry
        return entry

    def test_filling_entry_is_not_in_evictable_lru_until_ready(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)
        entry = self._insert_entry(
            ctrl,
            "h",
            state=EntryState.FILLING,
        )

        self.assertNotIn("h", ctrl.vision_pool.evictable)

        with ctrl.lock:
            ctrl._mark_ready(entry)

        self.assertEqual(list(ctrl.vision_pool.evictable.keys()), ["h"])

    def test_first_read_pin_removes_candidate_and_last_release_reinserts(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)
        entry = self._insert_entry(ctrl, "h")
        with ctrl.lock:
            ctrl._lru_touch("h")

        with ctrl.lock:
            ctrl._pin_read(entry)
            ctrl._pin_read(entry)

        self.assertNotIn("h", ctrl.vision_pool.evictable)

        with ctrl.lock:
            ctrl._unpin_read(entry)
        self.assertNotIn("h", ctrl.vision_pool.evictable)

        with ctrl.lock:
            ctrl._unpin_read(entry)
        self.assertEqual(list(ctrl.vision_pool.evictable.keys()), ["h"])

    def test_evict_for_pool_pops_only_that_pool_candidates(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)
        self._insert_entry(ctrl, "vision_h", modality=Modality.IMAGE)
        self._insert_entry(ctrl, "audio_h", modality=Modality.AUDIO)
        with ctrl.lock:
            ctrl._lru_touch("vision_h")
            ctrl._lru_touch("audio_h")

        required_pages = ctrl.vision_pool.allocator.free_pages + 1
        with ctrl.lock:
            ctrl._evict_for_pool(ctrl.vision_pool, required_pages)

        self.assertNotIn("vision_h", ctrl.entries)
        self.assertIn("audio_h", ctrl.entries)
        self.assertEqual(list(ctrl.audio_pool.evictable.keys()), ["audio_h"])


class TestStoreToPool(unittest.TestCase):
    def test_store_to_pool_async_raises_on_cpu_tensor(self):
        ctrl = _make_controller(num_pages=8, dim=4, page_size=2)
        tensor = torch.empty((2, 4), dtype=torch.float32)

        with self.assertRaises(ValueError):
            ctrl.store_to_pool_async(["h"], [tensor], Modality.IMAGE)

    def test_wait_store_isolates_failed_copy_and_waits_for_the_rest(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)
        entries = []
        for mm_hash in ("first", "failed", "last"):
            page_runs = ctrl.vision_pool.allocator.allocate(2, 2)
            entry = EmbeddingCacheEntry(
                hash=mm_hash,
                modality=Modality.IMAGE,
                num_tokens=2,
                dim=4,
                page_runs=page_runs,
                state=EntryState.FILLING,
            )
            ctrl.entries[mm_hash] = entry
            entries.append(entry)

        handles = [MagicMock(), MagicMock(), MagicMock()]
        handles[1].wait.side_effect = RuntimeError("D2H copy failed")

        with self.assertRaisesRegex(RuntimeError, "D2H copy failed"):
            ctrl.wait_store_to_pool(list(zip(entries, handles)))

        for handle in handles:
            handle.wait.assert_called_once_with()
        self.assertEqual(ctrl.entries["first"].state, EntryState.READY)
        self.assertNotIn("failed", ctrl.entries)
        self.assertEqual(ctrl.entries["last"].state, EntryState.READY)
        self.assertEqual(ctrl.vision_pool.allocator.free_pages, 2)


def _insert_ready_entry(ctrl, mm_hash, tensor, modality=Modality.IMAGE):
    """Manually write tensor into pool pages and create a READY entry."""
    pool = ctrl._get_pool(modality)
    if tensor.ndim != 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    num_tokens = int(tensor.shape[0])
    page_runs = pool.allocator.allocate(num_tokens, pool.page_size)
    copied = 0
    for run in page_runs:
        valid = min(pool.page_size * run.length, num_tokens - copied)
        start = run.start * pool.page_size
        pool.tensor[start : start + valid].copy_(tensor[copied : copied + valid])
        copied += valid
    entry = EmbeddingCacheEntry(
        hash=mm_hash,
        modality=modality,
        num_tokens=num_tokens,
        dim=int(tensor.shape[1]),
        page_runs=page_runs,
        state=EntryState.READY,
    )
    ctrl.entries[mm_hash] = entry
    pool.evictable.touch(mm_hash)
    return entry


class TestMooncakeLifecycle(unittest.TestCase):
    def test_prefetch_creates_filling_entry_and_get_success_marks_ready(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)

        ctrl.prefetch("req", ["h"], [2], Modality.IMAGE)
        op = ctrl.ongoing_prefetch["req"]
        ctrl._finish_get(op, [True])

        entry = ctrl.entries["h"]
        self.assertEqual(entry.state, EntryState.READY)

    def test_prefetch_get_failure_frees_entry(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)

        ctrl.prefetch("req", ["h"], [2], Modality.IMAGE)
        op = ctrl.ongoing_prefetch["req"]
        ctrl._finish_get(op, [False])

        self.assertNotIn("h", ctrl.entries)
        self.assertEqual(ctrl.vision_pool.allocator.free_pages, 4)

    def test_insert_batch_pins_and_releases_on_put(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)
        tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        _insert_ready_entry(ctrl, "h", tensor)

        ctrl.insert_batch(["h"], Modality.IMAGE)
        op = ctrl.insert_queue.get_nowait()

        entry = ctrl.entries["h"]
        self.assertEqual(entry.ref_count, 1)
        self.assertNotIn("h", ctrl.vision_pool.evictable)

        ctrl._finish_put(op, [True])

        self.assertEqual(entry.ref_count, 0)
        self.assertEqual(entry.state, EntryState.READY)
        self.assertIn("h", ctrl.vision_pool.evictable)


class TestGetPoolViews(unittest.TestCase):
    def test_get_pool_views_returns_none_for_filling_entry(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)
        ctrl.entries["h"] = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=2,
            dim=4,
            page_runs=[PageRun(0, 1)],
            state=EntryState.FILLING,
        )

        views = ctrl.get_pool_views(["h"])
        self.assertIsNone(views[0])

    def test_get_pool_views_returns_slices_and_release_unpins(self):
        ctrl = _make_controller(num_pages=4, dim=4, page_size=2)
        tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        _insert_ready_entry(ctrl, "h", tensor)

        views = ctrl.get_pool_views(["h"])
        self.assertIsNotNone(views[0])
        entry = ctrl.entries["h"]
        self.assertEqual(entry.ref_count, 1)

        flat = torch.cat(views[0], dim=0)
        self.assertTrue(torch.equal(flat, tensor))

        ctrl.release_pool_views(["h"])
        self.assertEqual(entry.ref_count, 0)
        self.assertIn("h", ctrl.vision_pool.evictable)


class TestTransferBuffers(unittest.TestCase):
    def test_build_transfer_buffers_for_single_run(self):
        pool = _make_pool(num_pages=8, dim=4, page_size=2)
        entry = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=5,
            dim=4,
            page_runs=[PageRun(2, 3)],
            state=EntryState.READY,
        )

        ptrs, sizes = _build_storage_transfer_buffers(entry, pool)

        self.assertEqual(ptrs, [pool.tensor[4].data_ptr()])
        self.assertEqual(sizes, [5 * 4 * torch.float32.itemsize])

    def test_build_transfer_buffers_for_multiple_runs(self):
        pool = _make_pool(num_pages=8, dim=4, page_size=2)
        entry = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=5,
            dim=4,
            page_runs=[PageRun(0, 1), PageRun(3, 2)],
            state=EntryState.READY,
        )

        ptrs, sizes = _build_storage_transfer_buffers(entry, pool)

        self.assertEqual(ptrs, [pool.tensor[0].data_ptr(), pool.tensor[6].data_ptr()])
        self.assertEqual(
            sizes,
            [
                2 * 4 * torch.float32.itemsize,
                3 * 4 * torch.float32.itemsize,
            ],
        )

    def test_build_h2d_copy_plan_for_fragmented_entry(self):
        pool = _make_pool(num_pages=10, dim=4, page_size=2)
        entry = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=5,
            dim=4,
            page_runs=[PageRun(2, 1), PageRun(7, 2)],
            state=EntryState.READY,
        )

        plan = _build_host_device_transfer_plan(
            entry, pool, src_is_pool=True, dst_token_offset=3
        )

        self.assertEqual(plan, ([4, 14], [3, 5], [2, 3]))

    def test_build_d2h_copy_plan_for_fragmented_entry(self):
        pool = _make_pool(num_pages=10, dim=4, page_size=2)
        entry = EmbeddingCacheEntry(
            hash="h",
            modality=Modality.IMAGE,
            num_tokens=5,
            dim=4,
            page_runs=[PageRun(2, 1), PageRun(7, 2)],
            state=EntryState.READY,
        )

        plan = _build_host_device_transfer_plan(entry, pool, src_is_pool=False)

        self.assertEqual(plan, ([0, 2], [4, 14], [2, 3]))


class TestMooncakeEmbeddingStoreWrappers(unittest.TestCase):
    def test_batch_put_multi_buffers_deduplicates_existing_keys(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
            MooncakeEmbeddingStore,
        )

        store = MooncakeEmbeddingStore.__new__(MooncakeEmbeddingStore)
        store.store = MagicMock()
        store.store.batch_is_exist.return_value = [1, 0]
        store.store.batch_put_from_multi_buffers.return_value = [0]

        results = store.batch_put_from_multi_buffers(
            ["a", "b"],
            [[11], [22]],
            [[4], [4]],
        )

        self.assertEqual(results, [True, True])
        store.store.batch_put_from_multi_buffers.assert_called_once_with(
            ["emb_b"], [[22]], [[4]]
        )

    def test_batch_get_multi_buffers_maps_positive_result_to_true(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
            MooncakeEmbeddingStore,
        )

        store = MooncakeEmbeddingStore.__new__(MooncakeEmbeddingStore)
        store.store = MagicMock()
        store.store.batch_get_into_multi_buffers.return_value = [8, -1]

        results = store.batch_get_into_multi_buffers(
            ["a", "b"],
            [[11], [22]],
            [[4], [4]],
        )

        self.assertEqual(results, [True, False])


if __name__ == "__main__":
    unittest.main()
