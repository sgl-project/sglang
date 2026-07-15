import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
from sglang.srt.utils.common import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseLegacyFlagForwarding(CustomTestCase):
    def test_dsv4_wrapper_reports_the_wrapped_allocators_legacy_declaration(self):
        """A legacy allocator wrapped by DSV4 HiSparse must not be served the aligned contract."""
        for wrapped_flag in (True, False):
            with self.subTest(wrapped=wrapped_flag):
                allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
                allocator.logical_attn_allocator = SimpleNamespace(
                    uses_legacy_real_length_alloc=wrapped_flag
                )
                self.assertIs(allocator.uses_legacy_real_length_alloc, wrapped_flag)

    def test_dsa_variant_owns_its_sub_pools_and_keeps_the_aligned_contract(self):
        """DSA HiSparse builds both sub-pools itself, so it has no wrapped declaration to inherit."""
        self.assertIs(
            HiSparseTokenToKVPoolAllocator.uses_legacy_real_length_alloc, False
        )


class TestHiSparseLegacyEntriesSurvive(CustomTestCase):
    """A wrapped legacy allocator still needs the real-length machinery it always had.

    HiSparse's alloc_extend is composite -- it takes the private pool and writes
    the mapping. The legacy module therefore cannot reach past the wrapper to the
    logical allocator; it has to keep calling the wrapper. Deleting these entries
    would strand every legacy x HiSparse deployment.
    """

    def test_legacy_prealloc_reaches_the_real_length_entry_not_the_paged_one(self):
        """Handing a real-length caller the page-aligned entry would misread fill_len as need_size."""
        from sglang.srt.disaggregation.decode import alloc_for_decode_prealloc_hisparse

        expected = torch.arange(510, dtype=torch.int64)
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.page_size = 64
        allocator.device = torch.device("cpu")
        allocator.logical_attn_allocator = SimpleNamespace(
            uses_legacy_real_length_alloc=True
        )
        allocator.alloc_logical_only_legacy = MagicMock(return_value=expected)
        allocator.alloc_logical_only = MagicMock(
            side_effect=AssertionError("the aligned entry must not serve a legacy caller")
        )
        req = SimpleNamespace(rid="req-0", kv=None, req_pool_idx=0)
        req_to_token_pool = SimpleNamespace(write=MagicMock())

        result = alloc_for_decode_prealloc_hisparse(
            allocator,
            req_to_token_pool,
            req=req,
            fill_len=510,
            total_prefix_len=0,
            uses_swa_tail=False,
            swa_tail_len=0,
        )

        self.assertIs(result, expected)
        allocator.alloc_logical_only.assert_not_called()
        _, kwargs = allocator.alloc_logical_only_legacy.call_args
        # Real length, unrounded: that is the whole point of the legacy contract.
        self.assertEqual(kwargs["extend_num_tokens"], 510)
        self.assertEqual(req.kv.kv_allocated_len, 510)

    def test_the_composite_extend_entry_is_still_there_for_legacy(self):
        """The legacy module calls alloc_extend on the wrapper; it cannot unwrap to the logical pool."""
        # Checked against the class's own namespace, not hasattr: the base
        # declares alloc_extend / alloc_decode as raising stubs, so hasattr
        # stays true even when the composite override is gone.
        own = vars(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        for name in ("alloc_extend", "alloc_decode", "get_last_loc_hisparse_device"):
            with self.subTest(method=name):
                self.assertIn(
                    name,
                    own,
                    f"{name} is the legacy path's only way to reach the private pool",
                )


def _make_dsv4_allocator(
    *, page_size: int, logical_alloc: MagicMock, hisparse_alloc: MagicMock
) -> DeepSeekV4HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator.compress_ratio = 4
    allocator.page_size = page_size
    allocator.hisparse_page_size = page_size // 4
    allocator.logical_attn_allocator = SimpleNamespace(
        alloc=logical_alloc, free=MagicMock()
    )
    allocator.hisparse_attn_allocator = SimpleNamespace(alloc=hisparse_alloc)
    # The real compression derivation, not a stand-in: it decides how many C4
    # rows a page-aligned logical range yields.
    kvcache = SimpleNamespace(compress_ratio=4)
    kvcache.translate_loc_from_full_to_compressed = (
        HiSparseC4DevicePool.translate_loc_from_full_to_compressed.__get__(kvcache)
    )
    allocator.hisparse_kvcache = kvcache
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        4096, dtype=torch.int64
    )
    return allocator


class TestDeepSeekV4HiSparseCompositeAlloc(CustomTestCase):
    def test_alloc_maps_every_compressed_row_to_a_freshly_allocated_c4_slot(self):
        """The C4 device slots are where prefill KV physically lands, so alloc must reserve and map them."""
        logical_indices = torch.arange(128, 256, dtype=torch.int64)
        # 128 logical tokens at ratio 4 -> exactly 32 C4 rows: 131, 135, ... 255.
        hisparse_indices = torch.arange(700, 732, dtype=torch.int64)
        allocator = _make_dsv4_allocator(
            page_size=64,
            logical_alloc=MagicMock(return_value=logical_indices),
            hisparse_alloc=MagicMock(return_value=hisparse_indices),
        )

        result = allocator.alloc(128)

        self.assertIs(result, logical_indices)
        allocator.logical_attn_allocator.alloc.assert_called_once_with(128)
        allocator.hisparse_attn_allocator.alloc.assert_called_once_with(32)
        mapping = allocator.full_to_hisparse_device_index_mapping
        expected_rows = torch.arange(32, dtype=torch.int64) + 32
        torch.testing.assert_close(mapping[expected_rows], hisparse_indices)
        self.assertEqual(int(mapping.count_nonzero()), 32)

    def test_alloc_rolls_the_logical_range_back_when_the_c4_pool_is_exhausted(self):
        """Keeping the logical range after a C4 failure would leak it for the process lifetime."""
        logical_indices = torch.arange(128, 256, dtype=torch.int64)
        allocator = _make_dsv4_allocator(
            page_size=64,
            logical_alloc=MagicMock(return_value=logical_indices),
            hisparse_alloc=MagicMock(return_value=None),
        )

        self.assertIsNone(allocator.alloc(128))

        allocator.logical_attn_allocator.free.assert_called_once_with(logical_indices)
        self.assertEqual(
            int(allocator.full_to_hisparse_device_index_mapping.count_nonzero()), 0
        )

    def test_alloc_leaves_the_c4_pool_untouched_when_the_logical_pool_is_exhausted(self):
        """Reserving C4 slots for a range that was never allocated would leak them."""
        hisparse_alloc = MagicMock()
        allocator = _make_dsv4_allocator(
            page_size=64,
            logical_alloc=MagicMock(return_value=None),
            hisparse_alloc=hisparse_alloc,
        )

        self.assertIsNone(allocator.alloc(128))

        hisparse_alloc.assert_not_called()
        allocator.logical_attn_allocator.free.assert_not_called()

    def test_alloc_rejects_a_size_that_is_not_a_whole_number_of_pages(self):
        """A partial page would break the whole-page invariant req_to_token addressing relies on."""
        allocator = _make_dsv4_allocator(
            page_size=64,
            logical_alloc=MagicMock(),
            hisparse_alloc=MagicMock(),
        )

        with self.assertRaises(AssertionError):
            allocator.alloc(100)

        allocator.logical_attn_allocator.alloc.assert_not_called()


def _make_dsa_allocator(
    *, page_size: int, logical_alloc: MagicMock, hisparse_alloc: MagicMock
) -> HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.compress_ratio = 1
    allocator.page_size = page_size
    allocator.logical_attn_allocator = SimpleNamespace(
        alloc=logical_alloc, free=MagicMock()
    )
    allocator.hisparse_attn_allocator = SimpleNamespace(alloc=hisparse_alloc)
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        4096, dtype=torch.int64
    )
    return allocator


class TestHiSparseCompositeAlloc(CustomTestCase):
    def test_alloc_serves_a_paged_request_instead_of_refusing_it(self):
        """Paged HiSparse allocation used to be alloc_extend's job; alloc is now the only entry."""
        logical_indices = torch.arange(128, 256, dtype=torch.int64)
        hisparse_indices = torch.arange(700, 828, dtype=torch.int64)
        allocator = _make_dsa_allocator(
            page_size=64,
            logical_alloc=MagicMock(return_value=logical_indices),
            hisparse_alloc=MagicMock(return_value=hisparse_indices),
        )

        result = allocator.alloc(128)

        self.assertIs(result, logical_indices)
        allocator.logical_attn_allocator.alloc.assert_called_once_with(128)
        # compress_ratio == 1, so the device side takes the same count.
        allocator.hisparse_attn_allocator.alloc.assert_called_once_with(128)
        mapping = allocator.full_to_hisparse_device_index_mapping
        torch.testing.assert_close(mapping[logical_indices], hisparse_indices)
        self.assertEqual(int(mapping.count_nonzero()), 128)

    def test_alloc_rolls_the_logical_range_back_when_the_device_pool_is_exhausted(self):
        """Keeping the logical range after a device failure would leak it for the process lifetime."""
        logical_indices = torch.arange(128, 256, dtype=torch.int64)
        allocator = _make_dsa_allocator(
            page_size=64,
            logical_alloc=MagicMock(return_value=logical_indices),
            hisparse_alloc=MagicMock(return_value=None),
        )

        self.assertIsNone(allocator.alloc(128))

        allocator.logical_attn_allocator.free.assert_called_once_with(logical_indices)
        self.assertEqual(
            int(allocator.full_to_hisparse_device_index_mapping.count_nonzero()), 0
        )

    def test_alloc_rejects_a_size_that_is_not_a_whole_number_of_pages(self):
        """A partial page would break the whole-page invariant req_to_token addressing relies on."""
        allocator = _make_dsa_allocator(
            page_size=64,
            logical_alloc=MagicMock(),
            hisparse_alloc=MagicMock(),
        )

        with self.assertRaises(AssertionError):
            allocator.alloc(100)

        allocator.logical_attn_allocator.alloc.assert_not_called()


def _make_dsv4_free_allocator(
    *, page_size: int, wrapped_uses_legacy: bool
) -> DeepSeekV4HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator.page_size = page_size
    allocator.logical_attn_allocator = SimpleNamespace(
        free=MagicMock(), uses_legacy_real_length_alloc=wrapped_uses_legacy
    )
    allocator.is_not_in_free_group = True
    allocator.free_group = []
    return allocator


class TestDeepSeekV4HiSparseFreeGuard(CustomTestCase):
    def test_free_accepts_a_whole_number_of_pages(self):
        """The guard must pass what the page-aligned world actually produces."""
        allocator = _make_dsv4_free_allocator(page_size=64, wrapped_uses_legacy=False)
        free_index = torch.arange(128, dtype=torch.int64)

        allocator.free(free_index)

        allocator.logical_attn_allocator.free.assert_called_once_with(free_index)

    def test_free_rejects_a_partial_page(self):
        """A partial free leaves the rest of the page owned by nobody and never reclaimed."""
        allocator = _make_dsv4_free_allocator(page_size=64, wrapped_uses_legacy=False)

        with self.assertRaises(AssertionError):
            allocator.free(torch.arange(100, dtype=torch.int64))

        allocator.logical_attn_allocator.free.assert_not_called()

    def test_free_rejects_a_partial_page_inside_a_free_group_too(self):
        """Two illegal fragments can concatenate to a legal length, so checking only at group end passes them."""
        allocator = _make_dsv4_free_allocator(page_size=64, wrapped_uses_legacy=False)
        allocator.free_group_begin()

        with self.assertRaises(AssertionError):
            allocator.free(torch.arange(28, dtype=torch.int64))

        self.assertEqual(allocator.free_group, [])

    def test_free_exempts_a_wrapped_legacy_allocator_from_the_page_check(self):
        """A legacy allocator is driven with real token lengths, so whole-page frees are not its contract."""
        allocator = _make_dsv4_free_allocator(page_size=64, wrapped_uses_legacy=True)
        free_index = torch.arange(100, dtype=torch.int64)

        allocator.free(free_index)

        allocator.logical_attn_allocator.free.assert_called_once_with(free_index)


def _make_routing_allocator(
    *, page_size: int, logical_indices: torch.Tensor
) -> HiSparseTokenToKVPoolAllocator:
    """A real HiSparse allocator over sub-pools that record what was asked of them."""
    allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.compress_ratio = 1
    allocator.page_size = page_size
    allocator.logical_attn_allocator = SimpleNamespace(
        alloc=MagicMock(return_value=logical_indices), free=MagicMock()
    )
    allocator.hisparse_attn_allocator = SimpleNamespace(
        alloc=MagicMock(return_value=torch.arange(len(logical_indices)) + 900)
    )
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        4096, dtype=torch.int64
    )
    return allocator


class TestHiSparseDecodeRouting(CustomTestCase):
    """Decode must reach the logical-only entry, and it is the routing that decides that.

    The device slot a decode token writes to is handed out by the coordinator's
    ring before the forward. A composite alloc here would take a private page per
    new logical page and have its mapping overwritten before anything read it:
    never written, never read, and no index left holding it. So this goes through
    the real alloc entries -- testing the overrides directly would not see the
    routing, and the routing is the whole of it.
    """

    def test_decode_takes_no_private_page(self):
        """A private page per decode page drains the private pool ahead of the logical one."""
        from sglang.srt.mem_cache import allocation as allocation_mod

        allocator = _make_routing_allocator(
            page_size=64, logical_indices=torch.arange(64, 128, dtype=torch.int64)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=64))
        batch = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            tree_cache=None,
            maybe_evict_swa=MagicMock(),
            model_config=SimpleNamespace(is_encoder_decoder=False),
            seq_lens_cpu=torch.tensor([64], dtype=torch.int64),
            seq_lens=torch.tensor([64], dtype=torch.int64),
            reqs=[req],
            device=torch.device("cpu"),
            req_to_token_pool=SimpleNamespace(req_to_token=object()),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
        )

        with mock.patch.object(allocation_mod, "_write_new_pages"), mock.patch.object(
            allocation_mod.AssignExtendCacheLocs, "execute_equal_length"
        ):
            allocation_mod.alloc_for_decode(batch, token_per_req=1)

        # Token 64 opens the second page: 64 -> 128, so one page of 64.
        allocator.logical_attn_allocator.alloc.assert_called_once_with(64)
        allocator.hisparse_attn_allocator.alloc.assert_not_called()
        self.assertEqual(int(allocator.full_to_hisparse_device_index_mapping.sum()), 0)
        self.assertEqual(req.kv.kv_allocated_len, 128)

    def test_extend_does_take_private_pages(self):
        """The contrast: prefill KV lands in the private slots, so its alloc must reserve them."""
        from sglang.srt.mem_cache.allocation import _alloc_new_pages

        logical_indices = torch.arange(128, dtype=torch.int64)
        allocator = _make_routing_allocator(
            page_size=64, logical_indices=logical_indices
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)

        out = _alloc_new_pages(tree_cache, need_size=128, oom_label="Prefill")

        self.assertIs(out, logical_indices)
        allocator.logical_attn_allocator.alloc.assert_called_once_with(128)
        allocator.hisparse_attn_allocator.alloc.assert_called_once_with(128)
        self.assertEqual(
            int(allocator.full_to_hisparse_device_index_mapping.count_nonzero()), 128
        )


class _FakeReqToTokenPool:
    def __init__(self, page_size: int, max_tokens: int):
        self.req_to_token = torch.zeros((1, max_tokens), dtype=torch.int64)
        self.writes = []

    def alloc(self, reqs):
        for item in reqs:
            item.req_pool_idx = 0
        return torch.tensor([0], dtype=torch.int64)

    def write(self, indices, values):
        self.writes.append((indices, values))
        self.req_to_token[indices] = values


def _make_prealloc_fixture(
    *, fill_len: int, compress_ratio: int, swa_tail_len: int = 128
) -> SimpleNamespace:
    from sglang.srt.disaggregation.decode import DecodePreallocQueue

    page_size = 64
    alloc_end = ceil_align(fill_len, page_size)
    kv_loc = torch.arange(alloc_end, dtype=torch.int64)
    host_len = alloc_end // compress_ratio
    host_indices = torch.arange(1000, 1000 + host_len, dtype=torch.int64)

    req = SimpleNamespace(
        rid="req-0", origin_input_ids=list(range(fill_len)), output_ids=[], kv=None
    )

    def set_extend_range(start, end):
        req.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

    req.set_extend_range = set_extend_range

    req_to_token_pool = _FakeReqToTokenPool(page_size, alloc_end)
    allocator = SimpleNamespace(
        device=torch.device("cpu"),
        page_size=page_size,
        uses_legacy_real_length_alloc=False,
        available_size=MagicMock(return_value=4096),
        alloc_extend_swa_tail=MagicMock(return_value=kv_loc),
        alloc_logical_only=MagicMock(return_value=kv_loc),
    )
    alloc_paged_token_slots = MagicMock(return_value=host_indices)
    coordinator = SimpleNamespace(
        mem_pool_host=SimpleNamespace(alloc_paged_token_slots=alloc_paged_token_slots),
        req_to_host_pool=object(),
        req_to_host_pool_allocated_len=object(),
        host_token_len=MagicMock(side_effect=lambda length: length // compress_ratio),
    )
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.req_to_token_pool = req_to_token_pool
    queue.token_to_kv_pool_allocator = allocator
    queue.tree_cache = SimpleNamespace(
        evictable_size=MagicMock(return_value=0),
        protected_size=MagicMock(return_value=0),
    )
    queue.scheduler = SimpleNamespace(
        enable_hisparse=True,
        hisparse_coordinator=coordinator,
        server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
    )
    queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
    queue._swa_tail_len = MagicMock(return_value=swa_tail_len)

    return SimpleNamespace(
        queue=queue,
        req=req,
        req_to_token_pool=req_to_token_pool,
        host_indices=host_indices,
        alloc_paged_token_slots=alloc_paged_token_slots,
    )


class TestDeepSeekV4HiSparseAllocator(CustomTestCase):
    def test_forwards_swa_tail_allocation_to_logical_allocator(self):
        """The DSV4 wrapper owns no SWA state, so the swa-tail entry must be a pure forward."""
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        logical_allocator = MagicMock(spec=["alloc_extend_swa_tail"])
        allocator.logical_attn_allocator = logical_allocator

        expected = torch.tensor([8, 9, 10], dtype=torch.int64)
        logical_allocator.alloc_extend_swa_tail.return_value = expected

        result = allocator.alloc_extend_swa_tail(seq_len=512, swa_tail_len=128)

        self.assertIs(result, expected)
        logical_allocator.alloc_extend_swa_tail.assert_called_once_with(
            seq_len=512, swa_tail_len=128
        )

    def test_hisparse_budget_uses_full_logical_capacity_for_swa_tail(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        logical_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=32),
            full_available_size=MagicMock(return_value=512),
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            logical_attn_allocator=logical_allocator
        )
        queue.scheduler = SimpleNamespace(enable_hisparse=True, last_batch=None)
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._need_space_for_single_req = MagicMock(return_value=0)
        queue._active_reserved_tokens = MagicMock(return_value=0)

        budget = queue._allocatable_token_budgets()

        self.assertEqual(budget, 512)
        logical_allocator.full_available_size.assert_called_once_with()
        logical_allocator.available_size.assert_not_called()

    def test_hisparse_prealloc_sizes_the_host_pool_from_the_padded_allocation(self):
        """The host rows must cover the padded logical range the coordinator later reads back."""
        # 510 tokens on a 64-token page pads to 512 logical, i.e. 128 C4 rows.
        # Sizing the host pool from 510 would ask for 127 and drop the last row.
        fixture = _make_prealloc_fixture(fill_len=510, compress_ratio=4)

        result = fixture.queue._pre_alloc(fixture.req)

        self.assertIs(result, fixture.host_indices)
        self.assertEqual(fixture.req.kv.kv_allocated_len, 512)
        args, _ = fixture.alloc_paged_token_slots.call_args
        self.assertEqual(args[-1], 128)

    def test_hisparse_prealloc_uses_swa_tail_for_direct_host_path(self):
        """The direct-to-host path must take the SWA-tail entry, not the plain logical one."""
        fill_len = 512
        swa_tail_len = 128
        fixture = _make_prealloc_fixture(
            fill_len=fill_len, compress_ratio=1, swa_tail_len=swa_tail_len
        )
        allocator = fixture.queue.token_to_kv_pool_allocator

        result = fixture.queue._pre_alloc(fixture.req)

        self.assertIs(result, fixture.host_indices)
        allocator.alloc_extend_swa_tail.assert_called_once()
        allocator.alloc_logical_only.assert_not_called()
        _, kwargs = allocator.alloc_extend_swa_tail.call_args
        self.assertEqual(kwargs["seq_len"], fill_len)
        self.assertEqual(kwargs["swa_tail_len"], swa_tail_len)
        self.assertEqual(fixture.req.kv.swa_evicted_seqlen, fill_len - swa_tail_len)
        self.assertEqual(fixture.req.kv.kv_allocated_len, fill_len)
        self.assertEqual(fixture.req.kv_committed_len, fill_len)
        self.assertEqual(fixture.req.extend_range.length, fill_len)
        self.assertEqual(len(fixture.req_to_token_pool.writes), 1)
        fixture.alloc_paged_token_slots.assert_called_once()


def _make_private_pool_allocator(allocator_cls: type) -> object:
    allocator = object.__new__(allocator_cls)
    allocator.hisparse_attn_allocator = SimpleNamespace(
        free=MagicMock(),
        free_pages_by_any_member_legacy=MagicMock(),
    )
    return allocator


class TestHiSparsePrivatePoolUsesLegacyFree(CustomTestCase):
    def test_generic_private_pool_release_takes_the_legacy_entry(self):
        """The private pool frees an arbitrary member subset, so it must not take the paged entry."""
        allocator = _make_private_pool_allocator(HiSparseTokenToKVPoolAllocator)
        buffer_indices = torch.tensor([5, 0, 9], dtype=torch.int64)

        allocator.free_hisparse_indices(buffer_indices)

        allocator.hisparse_attn_allocator.free.assert_not_called()
        legacy = allocator.hisparse_attn_allocator.free_pages_by_any_member_legacy
        legacy.assert_called_once()
        self.assertEqual(legacy.call_args.args[0].tolist(), [5, 9])

    def test_dsv4_private_pool_release_takes_the_legacy_entry(self):
        """DSV4 keeps its own copy of the private pool release; it must route the same way."""
        allocator = _make_private_pool_allocator(
            DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        buffer_indices = torch.tensor([5, 0, 9], dtype=torch.int64)

        allocator.free_hisparse_indices(buffer_indices)

        allocator.hisparse_attn_allocator.free.assert_not_called()
        legacy = allocator.hisparse_attn_allocator.free_pages_by_any_member_legacy
        legacy.assert_called_once()
        self.assertEqual(legacy.call_args.args[0].tolist(), [5, 9])


if __name__ == "__main__":
    unittest.main()
