import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
)
from sglang.srt.mem_cache.common import (
    free_kv_row_segments,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _build_swa_allocator(
    page_size: int = 1,
    req_size: int = 8,
    max_context_len: int = 64,
    kv_size: int = 64,
    kv_size_swa: int = 32,
    swa_req_ring_size: int | None = None,
):
    head_num = 8
    head_dim = 128
    num_layers = 24
    global_interval = 4
    dtype = torch.bfloat16
    device = get_device()
    full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
    full_attention_layer_ids_set = set(full_attention_layer_ids)
    swa_attention_layer_ids = [
        i for i in range(num_layers) if i not in full_attention_layer_ids_set
    ]

    req_to_token_pool = ReqToTokenPool(
        size=req_size,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        device=device,
    )
    kv_pool.swa_req_ring_size = swa_req_ring_size
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
        req_to_token_pool=req_to_token_pool,
    )
    return allocator, req_to_token_pool


def _sync_error(fn):
    """The RuntimeError torch raises if `fn` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


def _build_pure_swa_allocator(size_swa: int = 16):
    device = get_device()
    kv_pool = SWAKVPool(
        size=0,
        size_swa=size_swa,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=8,
        head_dim=128,
        swa_attention_layer_ids=list(range(4)),
        full_attention_layer_ids=[],
        device=device,
    )
    return PureSWATokenToKVPoolAllocator(
        size_swa=size_swa,
        page_size=1,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )


def _swa_alloc(allocator, need_size):
    """SWA-pool alloc that also works for page_size > 1 (built-in alloc asserts page_size == 1)."""
    if allocator.page_size == 1:
        return allocator.alloc(need_size)

    assert need_size % allocator.page_size == 0
    full_indices = allocator.full_attn_allocator.alloc(need_size)
    swa_indices = allocator.swa_attn_allocator.alloc(need_size)
    assert full_indices is not None and swa_indices is not None
    allocator.full_to_swa_index_mapping[full_indices] = swa_indices
    return full_indices


class TestSWA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_swa_memory_pool_paged_free_clears_full_page_mapping(self):
        page_size = 4
        allocator, _ = _build_swa_allocator(
            page_size=page_size,
            kv_size=16,
            kv_size_swa=16,
        )

        full_indices = _swa_alloc(allocator, page_size)
        self.assertEqual(allocator.swa_available_size(), 16 - page_size)

        allocator.free_swa(full_indices[:1])
        self.assertEqual(allocator.swa_available_size(), 16)
        self.assertTrue(
            torch.all(
                allocator.full_to_swa_index_mapping[full_indices.to(torch.int64)] == 0
            )
        )

        allocator.free_swa(full_indices[1:2])
        self.assertEqual(allocator.swa_available_size(), 16)

    @unittest.skipUnless(torch.cuda.is_available(), "sync detection needs CUDA")
    def test_clearing_the_mapping_does_not_synchronize(self):
        """Clearing the full-to-SWA mapping must not block the stream; writing a
        host-resident scalar into it does.
        """
        allocator, _ = _build_swa_allocator()
        full_indices = _swa_alloc(allocator, 4)
        mapping = allocator.full_to_swa_index_mapping

        # Warm up outside the window: a first-time cudaMalloc can synchronize on
        # its own, which the detector would report as this call's fault.
        allocator.clear_full_to_swa_mapping(full_indices)

        # Gate on the pre-fix form: a detector blind to this sync class would pass
        # the assert below no matter how the mapping is cleared.
        pre_fix_error = _sync_error(
            lambda: mapping.__setitem__(full_indices.to(torch.int64), 0)
        )
        if pre_fix_error is None:
            self.skipTest("sync debug mode does not flag a blocking H2D copy here")

        self.assertIsNone(
            _sync_error(lambda: allocator.clear_full_to_swa_mapping(full_indices))
        )

    def test_free_swa_group_owns_deferred_indices(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                self._free_swa_group_owns_deferred_indices(page_size)

    def _free_swa_group_owns_deferred_indices(self, page_size):
        allocator, _ = _build_swa_allocator(
            page_size=page_size,
            kv_size=32 * page_size,
            kv_size_swa=32 * page_size,
        )
        index_batches = []
        for size in (2, 3, 1, 4):
            indices = _swa_alloc(allocator, size * page_size)
            assert indices is not None
            index_batches.append(indices)
        original_indices = torch.cat([indices.clone() for indices in index_batches])

        available_before_free = allocator.swa_available_size()
        allocator.free_group_begin()
        for indices in index_batches:
            allocator.free_swa_segment(indices, start_pos=0)

        # The reps were gathered at enqueue time, not from these views.
        self.assertEqual(len(allocator.swa_page_ids_group), len(index_batches))
        self.assertEqual(allocator.swa_available_size(), available_before_free)
        for indices in index_batches:
            indices.zero_()
        allocator.free_group_end()

        self.assertTrue(
            torch.equal(
                allocator.full_to_swa_index_mapping[original_indices.to(torch.int64)],
                torch.zeros_like(original_indices),
            )
        )
        self.assertEqual(
            allocator.swa_available_size(),
            available_before_free + original_indices.numel(),
        )

    def test_free_swa_group_owns_mapping_at_enqueue_time(self):
        allocator, _ = _build_swa_allocator(
            kv_size=8,
            kv_size_swa=8,
        )
        old_full = _swa_alloc(allocator, 1)
        new_full = _swa_alloc(allocator, 1)
        assert old_full is not None and new_full is not None
        old_swa = allocator.full_to_swa_index_mapping[old_full].clone()
        new_swa = allocator.full_to_swa_index_mapping[new_full].clone()

        allocator.free_group_begin()
        allocator.free_swa(old_full)

        # Cache reconciliation can transfer a different SWA slot onto the same
        # full slot before the group flushes. The deferred free still owns the
        # mapping observed above, not this replacement mapping.
        allocator.set_full_to_swa_mapping(old_full, new_swa)
        allocator.clear_full_to_swa_mapping(new_full)
        allocator.free_group_end()

        torch.testing.assert_close(
            allocator.full_to_swa_index_mapping[old_full], new_swa
        )
        self.assertTrue(
            torch.isin(old_swa, allocator.swa_attn_allocator.free_pages).item()
        )
        self.assertFalse(
            torch.isin(new_swa, allocator.swa_attn_allocator.free_pages).item()
        )

    def _build_two_mapped_slots(self, page_size=1):
        allocator, _ = _build_swa_allocator(
            page_size=page_size,
            kv_size=8 * page_size,
            kv_size_swa=8 * page_size,
        )
        old_full = _swa_alloc(allocator, page_size)
        new_full = _swa_alloc(allocator, page_size)
        assert old_full is not None and new_full is not None
        old_swa = allocator.full_to_swa_index_mapping[old_full].clone()
        new_swa = allocator.full_to_swa_index_mapping[new_full].clone()
        return allocator, old_full, new_full, old_swa, new_swa

    def _swa_slot_is_free(self, allocator, swa_index):
        # free_pages holds page ids for page_size > 1 and token ids otherwise,
        # so compare in page space (a no-op divide when page_size == 1).
        swa_pages = swa_index // allocator.page_size
        free_pages = allocator.swa_attn_allocator.free_pages
        return bool(torch.isin(swa_pages, free_pages).all().item())

    def _run_remap_during_free_group(self, allocator, old_full, new_full, new_swa):
        """Queue a combined free, then transfer another SWA slot onto the same
        full slot before the group flushes -- what tombstone recovery does."""
        allocator.free_group_begin()
        allocator.free(old_full)
        allocator.set_full_to_swa_mapping(old_full, new_swa)
        allocator.clear_full_to_swa_mapping(new_full)
        allocator.free_group_end()

    def test_free_group_owns_mapping_at_enqueue_time(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                allocator, old_full, new_full, old_swa, new_swa = (
                    self._build_two_mapped_slots(page_size=page_size)
                )
                available_before = allocator.swa_available_size()

                self._run_remap_during_free_group(
                    allocator, old_full, new_full, new_swa
                )

                self.assertTrue(
                    self._swa_slot_is_free(allocator, old_swa),
                    "the SWA slot owned at enqueue time leaked",
                )
                self.assertFalse(
                    self._swa_slot_is_free(allocator, new_swa),
                    "the replacement SWA slot was freed while still mapped",
                )
                self.assertEqual(
                    allocator.swa_available_size(), available_before + page_size
                )
                # Everything still in use stays reachable through the mapping.
                mapped = allocator.full_to_swa_index_mapping[:-1]
                num_mapped = int((mapped > 0).sum().item())
                num_in_use = (
                    allocator.swa_attn_allocator.size - allocator.swa_available_size()
                )
                self.assertEqual(num_mapped, num_in_use)

    def test_pure_swa_rejects_mapping_edits(self):
        allocator = _build_pure_swa_allocator()
        indices = allocator.alloc(2)
        with self.assertRaises(NotImplementedError):
            allocator.clear_full_to_swa_mapping(indices)
        with self.assertRaises(NotImplementedError):
            allocator.set_full_to_swa_mapping(indices, indices)
        torch.testing.assert_close(
            allocator.full_to_swa_index_mapping[indices], indices
        )


class _SinglePoolAllocator(BaseTokenToKVPoolAllocator):
    """Minimal single-pool allocator: no SWA peer, so the whole range dies
    together whatever the floor says."""

    def __init__(self):
        super().__init__(
            size=16,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index)


class TestFreeFullPartition(CustomTestCase):
    """`free_full` releases only the full side of a hybrid SWA allocator."""

    def setUp(self):
        self.allocator, _ = _build_swa_allocator()
        self.full_baseline = self.allocator.full_available_size()
        self.swa_baseline = self.allocator.swa_available_size()

    def _sizes(self):
        return (
            self.allocator.full_available_size(),
            self.allocator.swa_available_size(),
        )

    def test_free_full_touches_only_the_full_pool(self):
        indices = _swa_alloc(self.allocator, 4)
        # free_full's precondition: the SWA peers are already released.
        self.allocator.free_swa(indices)
        self.assertEqual(self._sizes(), (self.full_baseline - 4, self.swa_baseline))

        self.allocator.free_full(indices)
        self.assertEqual(self._sizes(), (self.full_baseline, self.swa_baseline))

    def test_free_full_is_deferred_inside_a_free_group(self):
        indices = _swa_alloc(self.allocator, 4)
        self.allocator.free_swa(indices)

        self.allocator.free_group_begin()
        self.allocator.free_full(indices)
        self.assertEqual(self.allocator.full_available_size(), self.full_baseline - 4)
        self.allocator.free_group_end()

        self.assertEqual(self.allocator.full_available_size(), self.full_baseline)


class _RowCache:
    """Minimal PrefixCacheTrait host, so free_kv_row can be exercised without
    standing up a whole tree."""

    free_kv_row = BasePrefixCache.free_kv_row

    def __init__(self, allocator, row):
        self.req_to_token_pool = SimpleNamespace(req_to_token=row.unsqueeze(0))
        self.token_to_kv_pool_allocator = allocator
        self.page_size = allocator.page_size


class TestFreeKvRow(CustomTestCase):
    """A kv row is given back split at `swa_evicted_seqlen`: the full side
    whole, the SWA side only from the floor up."""

    def setUp(self):
        self.allocator, _ = _build_swa_allocator()
        self.full_baseline = self.allocator.full_available_size()
        self.swa_baseline = self.allocator.swa_available_size()

    def _sizes(self):
        return (
            self.allocator.full_available_size(),
            self.allocator.swa_available_size(),
        )

    def test_floor_decides_how_much_of_the_swa_side_the_row_frees(self):
        # (start_pos, num_slots, floor, rows whose SWA peers are already gone)
        cases = [
            (0, 4, 4, 4),
            (8, 4, 8, 0),
            (8, 4, 10, 2),
            (8, 4, 4, 0),
        ]
        for start_pos, num_slots, floor, num_dead in cases:
            with self.subTest(start_pos=start_pos, floor=floor):
                indices = _swa_alloc(self.allocator, num_slots)
                # Window eviction already released the peers below the floor.
                if num_dead:
                    self.allocator.free_swa(indices[:num_dead])
                self.assertEqual(
                    self._sizes(),
                    (
                        self.full_baseline - num_slots,
                        self.swa_baseline - num_slots + num_dead,
                    ),
                )
                free_kv_row_segments(
                    self.allocator, [(indices, start_pos)], swa_evicted_seqlen=floor
                )
                self.assertEqual(self._sizes(), (self.full_baseline, self.swa_baseline))

    def test_below_floor_pieces_go_back_through_the_full_side(self):
        allocator, _ = _build_swa_allocator(page_size=4)
        indices = _swa_alloc(allocator, 8)
        allocator.free_swa(indices)
        after_alloc = allocator.full_available_size()

        # Both rows [0, 4) and [4, 8) sit below the floor: full side only.
        with patch.object(
            allocator.full_attn_allocator,
            "free",
            side_effect=AssertionError("full side took the unique path"),
        ):
            free_kv_row_segments(
                allocator,
                [(indices[:4], 0), (indices[4:], 4)],
                swa_evicted_seqlen=8,
            )

        self.assertEqual(allocator.full_available_size(), after_alloc + 8)

    def test_grouped_full_side_frees_defer_and_skip_the_unique_path(self):
        allocator, _ = _build_swa_allocator(page_size=4)
        indices = _swa_alloc(allocator, 12)
        allocator.free_swa(indices[:8])
        after_alloc = allocator.full_available_size()

        with patch.object(
            allocator.full_attn_allocator,
            "free",
            side_effect=AssertionError("full side took the unique path"),
        ):
            allocator.free_group_begin()
            # dead rows [0, 8) and the alive row [8, 12) from one request
            free_kv_row_segments(allocator, [(indices, 0)], swa_evicted_seqlen=8)
            self.assertEqual(allocator.full_available_size(), after_alloc)
            allocator.free_group_end()

        self.assertEqual(allocator.full_available_size(), after_alloc + 12)

    def test_free_kv_row_reads_the_record_row_and_its_floor(self):
        indices = _swa_alloc(self.allocator, 8)
        cache = _RowCache(self.allocator, indices)
        kv = SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=3)
        self.allocator.free_swa(indices[:3])

        cache.free_kv_row(kv, [(1, 5)])

        # Rows [1, 5) go back on the full side; only [3, 5) still had SWA peers
        # to give back, so rows 5-7 keep the 3 SWA slots that are still out.
        self.assertEqual(self._sizes(), (self.full_baseline - 4, self.swa_baseline - 3))

    def test_single_pool_free_kv_row_still_frees_the_whole_range(self):
        allocator = _SinglePoolAllocator()
        cache = _RowCache(allocator, torch.arange(16, dtype=torch.int64))
        kv = SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=4)

        cache.free_kv_row(kv, [(2, 6)])

        self.assertEqual([t.tolist() for t in allocator.freed], [[2, 3], [4, 5]])

        # release_session and _free_kv_aligned dropped their own emptiness
        # guards, so an empty range has to stay a no-op here.
        cache.free_kv_row(kv, [(6, 6)])
        self.assertEqual(len(allocator.freed), 2)


class TestSWAPeerMappedContract(CustomTestCase):
    """page_size 1 gives back every peer the mapping names, without filtering:
    the contract replaces what `swa_indices > 0` used to absorb."""

    def _strict(self):
        return envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.STRICT))

    def _condition_checked_by(self, allocator, indices, start_pos=None):
        """The predicate free_swa hands the async assert, as a python bool."""
        with self._strict():
            with mock.patch.object(torch, "_assert_async") as assert_async:
                if start_pos is None:
                    allocator.free_swa(indices)
                else:
                    allocator.free_swa_segment(indices, start_pos=start_pos)
        return bool(assert_async.call_args.args[0])

    def test_segment_free_flags_a_page_whose_peer_is_already_gone(self):
        allocator, _ = _build_swa_allocator(page_size=4)
        live = _swa_alloc(allocator, 8)
        stale = _swa_alloc(allocator, 8)
        allocator.clear_full_to_swa_mapping(stale)

        self.assertTrue(self._condition_checked_by(allocator, live, start_pos=0))
        self.assertFalse(self._condition_checked_by(allocator, stale, start_pos=0))

    @unittest.skipUnless(torch.cuda.is_available(), "sync detection needs CUDA")
    def test_segment_free_does_not_synchronize_on_pages(self):
        """page_size > 1: page reps by stride replace the page expansion's
        filter and the inner allocator's torch.unique, in and out of a group."""
        ps = 4
        allocator, _ = _build_swa_allocator(page_size=ps)

        def grouped(indices):
            allocator.free_group_begin()
            allocator.free_swa_segment(indices, start_pos=0)
            allocator.free_group_end()

        # Warm up both paths outside the window: a first-time cudaMalloc can
        # synchronize on its own, which the detector would blame on this call.
        allocator.free_swa_segment(_swa_alloc(allocator, 2 * ps), start_pos=0)
        grouped(_swa_alloc(allocator, 2 * ps))
        first = _swa_alloc(allocator, 3 * ps)
        second = _swa_alloc(allocator, 2 * ps)

        # Gate on the pre-fix form: a detector blind to this sync class would pass
        # the asserts below no matter how free_swa derives the pages.
        if _sync_error(lambda: torch.unique(first // ps)) is None:
            self.skipTest("sync debug mode does not flag a data-dependent shape here")

        with self._strict():
            self.assertIsNone(
                _sync_error(
                    lambda: allocator.free_swa_segment(first[: 3 * ps - 1], start_pos=0)
                )
            )
            self.assertIsNone(_sync_error(lambda: grouped(second[: 2 * ps - 1])))

    def test_free_swa_flags_a_slot_whose_peer_is_already_gone(self):
        allocator, _ = _build_swa_allocator()
        live = _swa_alloc(allocator, 4)
        stale = _swa_alloc(allocator, 4)
        # Whoever released the peer left the mapping reading as the padding slot.
        allocator.clear_full_to_swa_mapping(stale)

        self.assertTrue(self._condition_checked_by(allocator, live))
        self.assertFalse(self._condition_checked_by(allocator, stale))

    @unittest.skipUnless(torch.cuda.is_available(), "sync detection needs CUDA")
    def test_free_swa_does_not_synchronize(self):
        """The filter's output shape was data-dependent, so it read a count back
        to the host; the gather that replaced it has a fixed shape."""
        allocator, _ = _build_swa_allocator()
        mapping = allocator.full_to_swa_index_mapping

        # Warm up outside the window: a first-time cudaMalloc can synchronize on
        # its own, which the detector would report as this call's fault.
        allocator.free_swa(_swa_alloc(allocator, 4))
        indices = _swa_alloc(allocator, 4)

        # Gate on the pre-fix form: a detector blind to this sync class would pass
        # the assert below no matter how free_swa reads the mapping.
        peers = mapping[indices]
        if _sync_error(lambda: peers[peers > 0]) is None:
            self.skipTest("sync debug mode does not flag a data-dependent shape here")

        with self._strict():
            self.assertIsNone(_sync_error(lambda: allocator.free_swa(indices)))


@unittest.skipUnless(torch.cuda.is_available(), "paged allocation kernels need CUDA")
class TestSWAReqRingFree(CustomTestCase):
    PS = 256

    def _allocated_ring(self):
        ps = self.PS
        allocator, req_pool = _build_swa_allocator(
            page_size=ps,
            req_size=2,
            max_context_len=4 * ps,
            kv_size=4 * ps,
            kv_size_swa=2 * ps,
            swa_req_ring_size=ps,
        )
        self.assertTrue(allocator.swa_req_ring)
        self.assertIsNotNone(req_pool.alloc_rows(1))
        device = allocator.device
        prefix_cpu = torch.tensor([0], dtype=torch.int64)
        seq_cpu = torch.tensor([2 * ps], dtype=torch.int64)
        # Use the real ring allocation paths: only FULL pages are allocated.
        indices = allocator.alloc_extend(
            prefix_cpu.to(device),
            prefix_cpu,
            seq_cpu.to(device),
            seq_cpu,
            torch.tensor([-1], dtype=torch.int64, device=device),
            2 * ps,
        )
        self.assertIsNotNone(indices)
        decoded = allocator.alloc_decode(
            (seq_cpu + 1).to(device), seq_cpu + 1, indices[-1:]
        )
        self.assertIsNotNone(decoded)
        indices = torch.cat((indices, decoded))
        self.assertTrue(torch.all(allocator.full_to_swa_index_mapping[indices] == 0))
        self.assertEqual(allocator.full_available_size(), ps)
        return allocator, indices

    def test_swa_only_frees_leave_the_paged_pool_untouched(self):
        for segment in (False, True):
            for grouped in (False, True):
                with self.subTest(segment=segment, grouped=grouped):
                    allocator, indices = self._allocated_ring()
                    swa_pages = (
                        allocator.swa_attn_allocator.get_all_free_pages().clone()
                    )
                    swa_available = allocator.swa_available_size()
                    if grouped:
                        allocator.free_group_begin()
                    if segment:
                        allocator.free_swa_segment(indices, start_pos=0)
                    else:
                        allocator.free_swa(indices)
                    self.assertEqual(allocator.swa_free_group, [])
                    self.assertEqual(allocator.swa_page_ids_group, [])
                    if grouped:
                        allocator.free_group_end()
                    self.assertTrue(
                        torch.equal(
                            allocator.swa_attn_allocator.get_all_free_pages(), swa_pages
                        )
                    )
                    self.assertEqual(allocator.swa_available_size(), swa_available)
                    self.assertEqual(allocator.full_available_size(), self.PS)
                    self.assertTrue(
                        torch.all(allocator.full_to_swa_index_mapping[indices] == 0)
                    )

    def test_combined_frees_still_release_full_pages(self):
        for segment in (False, True):
            for grouped in (False, True):
                with self.subTest(segment=segment, grouped=grouped):
                    allocator, indices = self._allocated_ring()
                    swa_pages = (
                        allocator.swa_attn_allocator.get_all_free_pages().clone()
                    )
                    if grouped:
                        allocator.free_group_begin()
                    if segment:
                        allocator.free_segment(indices, start_pos=0)
                    else:
                        allocator.free(indices)
                    if grouped:
                        self.assertEqual(allocator.full_available_size(), self.PS)
                        allocator.free_group_end()
                    self.assertEqual(
                        allocator.full_available_size(), allocator.size_full
                    )
                    self.assertTrue(
                        torch.equal(
                            allocator.swa_attn_allocator.get_all_free_pages(), swa_pages
                        )
                    )
                    full_pages = allocator.full_attn_allocator.get_all_free_pages()
                    self.assertTrue(torch.all(full_pages > 0))
                    self.assertEqual(torch.unique(full_pages).numel(), 4)

    def test_swa_only_frees_do_not_synchronize(self):
        allocator, indices = self._allocated_ring()
        peers = allocator.full_to_swa_index_mapping[indices]
        if _sync_error(lambda: peers[peers > 0]) is None:
            self.skipTest("sync debug mode does not flag a data-dependent shape here")

        with envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.STRICT)):
            for grouped in (False, True):
                with self.subTest(grouped=grouped):
                    if grouped:
                        allocator.free_group_begin()
                    self.assertIsNone(_sync_error(lambda: allocator.free_swa(indices)))
                    self.assertIsNone(
                        _sync_error(
                            lambda: allocator.free_swa_segment(indices, start_pos=0)
                        )
                    )
                    if grouped:
                        self.assertIsNone(_sync_error(allocator.free_group_end))


class TestSWAPageRepsFree(CustomTestCase):
    """page_size > 1: with a start position the SWA side frees one representative
    per page instead of expanding, filtering and dedup'ing through torch.unique."""

    PS = 4

    def _allocator(self):
        allocator, _ = _build_swa_allocator(page_size=self.PS)
        return allocator

    def _sizes(self, allocator):
        return allocator.full_available_size(), allocator.swa_available_size()

    @unittest.skipUnless(torch.cuda.is_available(), "needs a tensor with is_cuda=True")
    def test_free_swa_segment_npu_uses_reference_path(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                allocator, _ = _build_swa_allocator(
                    page_size=page_size,
                    kv_size=8 * page_size,
                    kv_size_swa=8 * page_size,
                )
                available_before = allocator.swa_available_size()
                full_indices = _swa_alloc(allocator, page_size)
                self.assertTrue(full_indices.is_cuda)

                # transfer_to_npu makes NPU tensors report is_cuda=True as well.
                with (
                    patch("sglang.srt.mem_cache.allocator.swa._is_npu", True),
                    patch(
                        "sglang.srt.mem_cache.allocator.swa.get_and_clear_swa_pages",
                        side_effect=AssertionError("NPU free reached Triton"),
                    ),
                ):
                    allocator.free_swa_segment(full_indices[:1], start_pos=0)

                self.assertEqual(allocator.swa_available_size(), available_before)
                self.assertTrue(
                    torch.all(allocator.full_to_swa_index_mapping[full_indices] == 0)
                )

    def test_free_swa_segment_debug_rejects_invalid_page_mappings(self):
        page_size = 4

        def leading_hole(mapping, full_indices, _swa_indices):
            mapping[full_indices[0]] = 0

        def multiple_peers(mapping, full_indices, swa_indices):
            mapping[full_indices[2:page_size]] = swa_indices[
                page_size + 2 : 2 * page_size
            ]

        def duplicate_peer(mapping, full_indices, swa_indices):
            mapping[full_indices[page_size : 2 * page_size]] = swa_indices[:page_size]

        def duplicate_representative(_mapping, full_indices, _swa_indices):
            full_indices[-page_size:] = full_indices[:page_size]

        for name, mutate, num_tokens in (
            ("leading_hole", leading_hole, page_size),
            ("multiple_peers", multiple_peers, page_size),
            ("duplicate_peer", duplicate_peer, 2 * page_size),
            # At page size 4, representatives 0 and 64 belong to separate programs.
            ("duplicate_representative", duplicate_representative, 65 * page_size),
        ):
            with self.subTest(name=name):
                num_allocated_tokens = max(2 * page_size, num_tokens)
                kv_size = max(8 * page_size, num_allocated_tokens)
                allocator, _ = _build_swa_allocator(
                    page_size=page_size,
                    kv_size=kv_size,
                    kv_size_swa=kv_size,
                )
                full_indices = _swa_alloc(allocator, num_allocated_tokens)
                mapping = allocator.full_to_swa_index_mapping
                swa_indices = mapping[full_indices].clone()
                mutate(mapping, full_indices, swa_indices)
                allocator.swa_attn_allocator.debug_mode = True

                # Exercise debug validation without CI's fatal async assertion.
                with (
                    patch.dict(
                        "os.environ",
                        {"SGLANG_INVARIANT_CHECK": str(int(InvariantCheckLevel.OFF))},
                    ),
                    self.assertRaisesRegex(
                        AssertionError, "swa pages do not match the mapped pages"
                    ),
                ):
                    allocator.free_swa_segment(full_indices[:num_tokens], start_pos=0)

    def test_segment_free_releases_the_mapped_pages_for_every_tail(self):
        ps = self.PS
        for num_tokens in (1, ps, ps + 1, 3 * ps - 1, 3 * ps):
            with self.subTest(num_tokens=num_tokens):
                allocator = self._allocator()
                indices = _swa_alloc(allocator, 3 * ps)
                mapping = allocator.full_to_swa_index_mapping
                expected = torch.unique(mapping[indices[:num_tokens]] // ps)
                before = allocator.swa_attn_allocator.free_pages.numel()

                allocator.free_swa_segment(indices[:num_tokens], start_pos=0)

                free_pages = allocator.swa_attn_allocator.free_pages
                freed = free_pages[: free_pages.numel() - before]
                self.assertTrue(torch.equal(torch.sort(freed)[0], expected))
                # The whole last page goes back, and its mapping with it.
                touched = -(num_tokens // -ps) * ps
                self.assertTrue(torch.all(mapping[indices[:touched]] == 0))
                self.assertTrue(torch.all(mapping[indices[touched:]] > 0))


if __name__ == "__main__":
    unittest.main()
