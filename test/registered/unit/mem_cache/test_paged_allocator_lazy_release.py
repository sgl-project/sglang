"""need_sort allocators stage released pages and merge only under allocation pressure."""

import os
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

PAGE_SIZE = 2
NUM_PAGES = 16


def _make_allocator(*, need_sort: bool) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _page_ids(indices: torch.Tensor) -> torch.Tensor:
    return indices[::PAGE_SIZE] // PAGE_SIZE


class TestPagedAllocatorLazyRelease(CustomTestCase):
    def test_release_stages_until_free_pages_run_dry(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(24)
        free_pages_before = allocator.free_pages

        with patch.object(torch, "cat", wraps=torch.cat) as cat_mock:
            allocator.free(allocated[:2])
            allocator.free(allocated[4:6])
            self.assertIs(allocator.free_pages, free_pages_before)
            self.assertEqual(allocator.num_staged_pages, 2)
            self.assertEqual(allocator.available_size(), 12)
            cat_mock.assert_not_called()

            # free_pages still covers this: no merge.
            primary = allocator.alloc(8)
            self.assertTrue(torch.equal(_page_ids(primary), torch.arange(13, 17)))
            cat_mock.assert_not_called()

            # Pressure: one merge, staged pages come back sorted.
            reused = allocator.alloc(4)
            self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1, 3])))
            self.assertEqual(cat_mock.call_count, 1)

        self.assertEqual(allocator.num_staged_pages, 0)
        self.assertEqual(allocator.staged_pages, [])

    def test_clear_drops_staged_pages(self):
        allocator = _make_allocator(need_sort=True)
        allocator.free(allocator.alloc(4))

        allocator.clear()
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(allocator.staged_pages, [])

    def test_reuse_guard_waits_only_when_staged_pages_are_needed(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(24)
        event = Mock()
        allocator.set_reuse_guard_event(event)
        allocator.free(allocated[:2])
        # Fresh pages are safe to hand to an asynchronous PD transfer.
        allocator.alloc(8)
        event.synchronize.assert_not_called()

        reused = allocator.alloc(2)
        self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1])))
        event.synchronize.assert_called_once_with()
        allocator.merge_and_sort_free()
        event.synchronize.assert_called_once_with()


class TestPagedAllocatorDebugMode(CustomTestCase):
    def test_debug_mode_accepts_every_spelling_of_the_declared_bool(self):
        """SGLANG_DEBUG_MEMORY_POOL is one switch shared with the KV pool: a
        value that turns on the pool's checks must turn on the allocator's."""
        for value, expected in (("yes", True), ("1", True), ("no", False)):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": value}),
            ):
                self.assertEqual(_make_allocator(need_sort=False).debug_mode, expected)


def _static_token(size=64, *, need_sort=False):
    return TokenToKVPoolAllocator(size, torch.float16, "cpu", None, need_sort)


def _static_swa(*, need_sort=False):
    cache = Mock(spec=BaseSWAKVPool)
    cache.full_kv_pool = cache.swa_kv_pool = None
    cache.swa_req_ring_size = None
    return SWATokenToKVPoolAllocator(64, 64, 1, torch.float16, "cpu", cache, need_sort)


class TestStaticAllocationIdStorage(CustomTestCase):
    def test_retained_ids_do_not_pin_rebuilt_free_lists(self):
        for kind in ("token", "sorted_token", "mamba"):
            with self.subTest(kind=kind):
                allocator = (
                    MambaSlotAllocator(64, "cpu")
                    if kind == "mamba"
                    else _static_token(need_sort=kind == "sorted_token")
                )
                held = []
                for _ in range(8):
                    ids = allocator.alloc(1)
                    held.append(ids[0])
                    churn = allocator.alloc(1)
                    allocator.free(churn)
                    if kind == "sorted_token":
                        allocator.merge_and_sort_free()
                    del ids, churn
                stores = {
                    x.untyped_storage().data_ptr(): x.untyped_storage().nbytes()
                    for x in held
                }
                self.assertEqual(sum(stores.values()), 8 * 8)
                self.assertEqual(allocator.available_size(), 64 - 8)
                rest = allocator.alloc(allocator.available_size())
                all_ids = [int(x) for x in held] + rest.tolist()
                self.assertEqual(sorted(all_ids), list(range(1, 65)))

    def test_zero_and_shortfall_do_not_retain_or_consume_ids(self):
        for allocator in (_static_token(), MambaSlotAllocator(64, "cpu")):
            with self.subTest(allocator=type(allocator).__name__):
                empty = allocator.alloc(0)
                self.assertEqual(empty.untyped_storage().nbytes(), 0)
                with patch.object(
                    torch.Tensor, "clone", side_effect=torch.OutOfMemoryError("output")
                ):
                    self.assertIsNone(allocator.alloc(65))
                    with self.assertRaises(torch.OutOfMemoryError):
                        allocator.alloc(1)
                self.assertEqual(allocator.available_size(), 64)
                self.assertEqual(allocator.alloc(64).tolist(), list(range(1, 65)))

    def test_mamba_group_retains_only_the_owned_batch_and_returns_unused(self):
        allocator = MambaSlotAllocator(64, "cpu")
        held, expected = [], []
        for _ in range(4):
            allocator.alloc_group_begin(4)
            slot = allocator.alloc(1)[0]
            held.append(slot)
            expected.append(int(slot))
            allocator.alloc_group_end()
            churn = allocator.alloc(1)
            allocator.free(churn)
            del churn
            self.assertEqual(slot.untyped_storage().nbytes(), 4 * 8)
            self.assertEqual([int(x) for x in held], expected)
            self.assertEqual(allocator.available_size(), 64 - len(held))
        rest = allocator.alloc(allocator.available_size())
        self.assertEqual(sorted(expected + rest.tolist()), list(range(1, 65)))

    def test_deferred_token_frees_keep_old_results_small_and_reusable(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = _static_token(need_sort=need_sort)
                held = allocator.alloc(1)
                freed = allocator.alloc(63)
                allocator.free_group_begin()
                allocator.free(freed)
                self.assertIsNone(allocator.alloc(1))
                allocator.free_group_end()
                reused = allocator.alloc(63)
                self.assertEqual(reused.tolist(), list(range(2, 65)))
                self.assertEqual(held.tolist(), [1])
                self.assertEqual(held.untyped_storage().nbytes(), 8)

    def test_swa_second_output_failure_restores_full_ids_without_allocating(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = _static_swa(need_sort=need_sort)
                live = allocator.alloc(64)
                allocator.free(live[:2])
                full, swa = allocator.full_attn_allocator, allocator.swa_attn_allocator
                full_ids = (
                    torch.cat((full.free_pages, full.release_pages)).sort().values
                )
                swa_ids = torch.cat((swa.free_pages, swa.release_pages)).sort().values
                mapping = allocator.full_to_swa_index_mapping.clone()
                clone, cat = torch.Tensor.clone, torch.cat
                calls = 0
                failed = False

                def fail_second(tensor, *args, **kwargs):
                    nonlocal calls, failed
                    calls += 1
                    if calls == 2:
                        failed = True
                        raise torch.OutOfMemoryError("SWA output")
                    return clone(tensor, *args, **kwargs)

                def no_allocating_rollback(*args, **kwargs):
                    if failed:
                        raise torch.OutOfMemoryError("rollback tried to allocate")
                    return cat(*args, **kwargs)

                with (
                    patch.object(torch.Tensor, "clone", fail_second),
                    patch.object(torch, "cat", no_allocating_rollback),
                ):
                    with self.assertRaisesRegex(torch.OutOfMemoryError, "SWA output"):
                        allocator.alloc(2)
                self.assertEqual(full.available_size(), 2)
                self.assertEqual(swa.available_size(), 2)
                torch.testing.assert_close(
                    torch.cat((full.free_pages, full.release_pages)).sort().values,
                    full_ids,
                )
                torch.testing.assert_close(
                    torch.cat((swa.free_pages, swa.release_pages)).sort().values,
                    swa_ids,
                )
                torch.testing.assert_close(allocator.full_to_swa_index_mapping, mapping)
                reused = allocator.alloc(2)
                self.assertEqual(reused.tolist(), [1, 2])
                self.assertEqual(
                    allocator.full_to_swa_index_mapping[reused].tolist(), [1, 2]
                )
                self.assertEqual(full.available_size(), 0)
                self.assertEqual(swa.available_size(), 0)


if __name__ == "__main__":
    unittest.main()
