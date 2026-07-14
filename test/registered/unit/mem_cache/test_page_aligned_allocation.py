import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import hisparse_coordinator as hisparse_coordinator_module
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache import allocation as allocation_module
from sglang.srt.mem_cache import common as mem_cache_common
from sglang.srt.mem_cache.allocation import (
    _compute_decode_write_locs,
    _DecodeWriteLocs,
    _plan_page_aligned_decode,
    _validate_main_page_aligned_alloc,
    _validate_spec_decode_alloc,
    alloc_for_decode,
    alloc_for_extend,
    alloc_for_spec_decode,
    alloc_paged_token_slots_extend,
)
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.multi_ended_allocator import (
    MultiEndedAllocator,
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_utils as eagle_utils_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPageAlignedAllocation(unittest.TestCase):
    def test_unknown_allocator_rejects_extend_before_mutation(self) -> None:
        """Unknown paged allocators reject extend before any shared state mutation."""
        allocator = _UnsupportedMainAllocator(page_size=4)
        req_to_token_pool = SimpleNamespace(
            alloc=mock.Mock(),
            write=mock.Mock(),
        )
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            req_to_token_pool=req_to_token_pool,
            maybe_evict_swa=mock.Mock(),
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(allocation_module, "alloc_token_slots") as alloc,
            self.assertRaisesRegex(NotImplementedError, "UnsupportedMainAllocator"),
        ):
            alloc_for_extend(batch)

        batch.maybe_evict_swa.assert_not_called()
        req_to_token_pool.alloc.assert_not_called()
        req_to_token_pool.write.assert_not_called()
        alloc.assert_not_called()

    def test_unknown_allocator_rejects_decode_before_mutation(self) -> None:
        """Unknown paged allocators reject decode before eviction or publication."""
        allocator = _UnsupportedMainAllocator(page_size=4)
        req_to_token_pool = SimpleNamespace(write=mock.Mock())
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            req_to_token_pool=req_to_token_pool,
            maybe_evict_swa=mock.Mock(),
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(allocation_module, "alloc_token_slots") as alloc,
            self.assertRaisesRegex(NotImplementedError, "UnsupportedMainAllocator"),
        ):
            alloc_for_decode(batch, token_per_req=1)

        batch.maybe_evict_swa.assert_not_called()
        req_to_token_pool.write.assert_not_called()
        alloc.assert_not_called()

    def test_all_builtin_paged_allocators_support_main_allocation(self) -> None:
        """Every built-in paged allocator explicitly accepts main direct allocation."""
        allocator_types = (
            PagedTokenToKVPoolAllocator,
            MultiEndedAllocator,
            UnifiedMambaTokenToKVPoolAllocator,
            SWATokenToKVPoolAllocator,
            UnifiedSWATokenToKVPoolAllocator,
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )

        for allocator_type in allocator_types:
            with self.subTest(allocator_type=allocator_type.__name__):
                self.assertIn(
                    "validate_main_page_aligned_alloc",
                    allocator_type.__dict__,
                )
                allocator_type.validate_main_page_aligned_alloc(
                    object.__new__(allocator_type)
                )

    def test_main_capability_bypasses_page_one_and_npu(self) -> None:
        """Page-one and NPU legacy paths do not require the main paged hook."""
        page_one_batch = SimpleNamespace(
            tree_cache=SimpleNamespace(
                token_to_kv_pool_allocator=_UnsupportedMainAllocator(page_size=1)
            )
        )
        npu_batch = SimpleNamespace(
            tree_cache=SimpleNamespace(
                token_to_kv_pool_allocator=_UnsupportedMainAllocator(page_size=4)
            )
        )

        with mock.patch.object(allocation_module, "_is_npu", False):
            _validate_main_page_aligned_alloc(page_one_batch)
        with mock.patch.object(allocation_module, "_is_npu", True):
            _validate_main_page_aligned_alloc(npu_batch)

    def test_supported_allocators_expose_spec_decode_capability(self) -> None:
        """Supported allocators explicitly accept speculative decode allocation."""
        explicit_allocator_types = (
            TokenToKVPoolAllocator,
            PagedTokenToKVPoolAllocator,
            SWATokenToKVPoolAllocator,
            MultiEndedAllocator,
        )

        for allocator_type in explicit_allocator_types:
            with self.subTest(allocator_type=allocator_type.__name__):
                self.assertIn("validate_spec_decode_alloc", allocator_type.__dict__)
                allocator_type.validate_spec_decode_alloc(
                    object.__new__(allocator_type)
                )

        self.assertIs(
            PureSWATokenToKVPoolAllocator.validate_spec_decode_alloc,
            SWATokenToKVPoolAllocator.validate_spec_decode_alloc,
        )
        self.assertIs(
            UnifiedSWATokenToKVPoolAllocator.validate_spec_decode_alloc,
            SWATokenToKVPoolAllocator.validate_spec_decode_alloc,
        )

    def test_hisparse_does_not_inherit_spec_decode_capability(self) -> None:
        """HiSparse allocators stay fail-closed for speculative decode."""
        allocator_types = (
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )

        for allocator_type in allocator_types:
            with (
                self.subTest(allocator_type=allocator_type.__name__),
                self.assertRaisesRegex(NotImplementedError, allocator_type.__name__),
            ):
                allocator_type.validate_spec_decode_alloc(
                    object.__new__(allocator_type)
                )

    def test_spec_capability_bypasses_only_npu(self) -> None:
        """Spec decode requires its capability at page one except on NPU."""
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=_UnsupportedMainAllocator(page_size=1)
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            self.assertRaisesRegex(NotImplementedError, "UnsupportedMainAllocator"),
        ):
            _validate_spec_decode_alloc(tree_cache)

        with mock.patch.object(allocation_module, "_is_npu", True):
            _validate_spec_decode_alloc(tree_cache)

    def test_hisparse_rehome_bypasses_page_one_and_npu(self) -> None:
        """HiSparse re-home routing stays exclusive to non-NPU paged allocation."""
        cases = ((1, False), (4, True))

        for page_size, is_npu in cases:
            with self.subTest(page_size=page_size, is_npu=is_npu):
                coordinator = object.__new__(HiSparseCoordinator)
                coordinator.token_to_kv_pool_allocator = SimpleNamespace(
                    page_size=page_size
                )
                coordinator._rehome_page_boundary_owners = mock.Mock()
                coordinator._eager_backup_previous_token = mock.Mock()
                coordinator.is_dsv4_hisparse = True
                coordinator.compress_ratio = 4

                with mock.patch.object(
                    hisparse_coordinator_module,
                    "_is_npu",
                    is_npu,
                ):
                    coordinator.map_last_loc_to_buffer(
                        seq_lens=torch.tensor([1], dtype=torch.int64),
                        out_cache_loc=torch.tensor([1], dtype=torch.int64),
                        req_pool_indices=torch.tensor([0], dtype=torch.int64),
                        seq_lens_cpu=torch.tensor([1], dtype=torch.int64),
                        req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
                    )

                coordinator._rehome_page_boundary_owners.assert_not_called()

    def test_extend_entry_rejects_misalignment_before_eviction(self) -> None:
        """Extend rejects a malformed page request before eviction or allocation."""
        allocator = SimpleNamespace(
            page_size=4,
            alloc_extend=mock.Mock(
                side_effect=AssertionError("allocator must not mutate")
            ),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(allocation_module, "evict_from_tree_cache") as evict,
            self.assertRaisesRegex(AssertionError, "prefix lens"),
        ):
            alloc_paged_token_slots_extend(
                tree_cache=tree_cache,
                prefix_lens=torch.tensor([2], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([2], dtype=torch.int64),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
                last_loc=torch.tensor([11], dtype=torch.int64),
                extend_num_tokens=6,
            )

        evict.assert_not_called()
        allocator.alloc_extend.assert_not_called()

    def test_extend_separates_physical_and_forward_lengths(self) -> None:
        """Extend publishes aligned capacity but gathers only logical tokens."""
        req = SimpleNamespace(
            prefix_indices=torch.tensor([11, 12], dtype=torch.int64),
            kv=SimpleNamespace(kv_allocated_len=4, swa_evicted_seqlen=0),
        )
        req_to_token = torch.zeros((1, 16), dtype=torch.int32)
        req_to_token_pool = SimpleNamespace(
            req_to_token=req_to_token,
            alloc=lambda _: [0],
        )
        allocator = SimpleNamespace(
            page_size=4,
            validate_main_page_aligned_alloc=mock.Mock(),
        )
        batch = SimpleNamespace(
            reqs=[req],
            prefix_lens=[2],
            extend_lens=[3],
            extend_num_tokens=3,
            seq_lens=torch.tensor([5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            req_to_token_pool=req_to_token_pool,
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            token_to_kv_pool_allocator=allocator,
            device=torch.device("cpu"),
            maybe_evict_swa=mock.Mock(),
        )
        physical_slots = torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        logical_slots = torch.tensor([12, 91, 92], dtype=torch.int64)

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(
                allocation_module,
                "alloc_token_slots",
                return_value=physical_slots,
            ) as producer,
            mock.patch.object(allocation_module, "write_cache_indices") as writer,
            mock.patch.object(
                allocation_module,
                "gather_out_cache_loc_extend",
                return_value=logical_slots,
            ) as gather,
        ):
            out_cache_loc, _, _ = alloc_for_extend(batch)

        self.assertEqual(out_cache_loc.numel(), 3)
        self.assertEqual(producer.call_args.kwargs["num_tokens"], 4)
        self.assertEqual(writer.call_args.kwargs["alloc_start_lens_cpu"].tolist(), [4])
        self.assertEqual(writer.call_args.kwargs["alloc_end_lens_cpu"].tolist(), [8])
        self.assertEqual(gather.call_args.kwargs["prefix_lens_cpu"].tolist(), [2])
        self.assertEqual(gather.call_args.kwargs["seq_lens_cpu"].tolist(), [5])
        self.assertEqual(gather.call_args.kwargs["extend_num_tokens"], 3)
        self.assertEqual(req.kv.kv_allocated_len, 8)

    def test_extend_npu_preserves_legacy_paged_routing(self) -> None:
        """NPU extend retains legacy paged allocation and DSV4 state routing."""
        req = SimpleNamespace(
            prefix_indices=torch.tensor([11, 12], dtype=torch.int64),
            kv=SimpleNamespace(kv_allocated_len=2, swa_evicted_seqlen=0),
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 16), dtype=torch.int32),
            alloc=lambda _: [0],
        )
        validate = mock.Mock(side_effect=AssertionError("NPU must bypass main hook"))
        allocator = SimpleNamespace(
            page_size=4,
            validate_main_page_aligned_alloc=validate,
        )
        batch = SimpleNamespace(
            reqs=[req],
            prefix_lens=[2],
            extend_lens=[3],
            extend_num_tokens=3,
            seq_lens=torch.tensor([5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            req_to_token_pool=req_to_token_pool,
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            token_to_kv_pool_allocator=allocator,
            device=torch.device("cpu"),
            maybe_evict_swa=mock.Mock(),
        )
        physical_slots = torch.tensor([101, 102, 103], dtype=torch.int64)
        dsv4_state_lens = object()

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(
                allocation_module,
                "alloc_paged_token_slots_extend",
                return_value=physical_slots,
            ) as legacy_producer,
            mock.patch.object(
                allocation_module, "alloc_token_slots"
            ) as direct_producer,
            mock.patch.object(
                allocation_module,
                "_compute_dsv4_state_lens",
                return_value=dsv4_state_lens,
            ),
            mock.patch.object(allocation_module, "write_cache_indices"),
            mock.patch.object(
                allocation_module,
                "gather_out_cache_loc_extend",
                return_value=physical_slots,
            ),
            mock.patch.object(allocation_module, "maybe_write_dsv4_extend"),
        ):
            alloc_for_extend(batch)

        validate.assert_not_called()
        direct_producer.assert_not_called()
        self.assertEqual(
            legacy_producer.call_args.kwargs["prefix_lens_cpu"].tolist(), [2]
        )
        self.assertEqual(legacy_producer.call_args.kwargs["seq_lens_cpu"].tolist(), [5])
        self.assertEqual(legacy_producer.call_args.kwargs["last_loc"].tolist(), [12])
        self.assertEqual(legacy_producer.call_args.kwargs["extend_num_tokens"], 3)
        self.assertEqual(
            legacy_producer.call_args.kwargs["req_pool_indices"].tolist(),
            [0],
        )
        self.assertIs(
            legacy_producer.call_args.kwargs["dsv4_state_lens"],
            dsv4_state_lens,
        )
        self.assertIs(legacy_producer.call_args.kwargs["batch"], batch)

    def test_decode_direct_allocation_mixes_crossing_and_in_page(self) -> None:
        """Decode allocates one whole page only for the crossing request."""
        batch = _make_decode_batch(
            write_locs=[8, 6],
            allocated_lens=[8, 8],
        )
        allocated_page = torch.tensor([101, 102, 103, 104], dtype=torch.int64)

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(
                allocation_module,
                "alloc_token_slots",
                return_value=allocated_page,
            ) as alloc,
        ):
            out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        self.assertEqual(alloc.call_args.kwargs["num_tokens"], 4)
        self.assertEqual(out_cache_loc.tolist(), [101, 207])
        self.assertEqual([req.kv.kv_allocated_len for req in batch.reqs], [12, 8])

    def test_decode_direct_allocation_skips_zero_crossings(self) -> None:
        """Decode validates capability but allocates nothing for in-page writes."""
        batch = _make_decode_batch(
            write_locs=[5, 7],
            allocated_lens=[8, 8],
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(allocation_module, "alloc_token_slots") as alloc,
        ):
            out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        batch.tree_cache.token_to_kv_pool_allocator.validate_main_page_aligned_alloc.assert_called_once_with()
        alloc.assert_not_called()
        self.assertEqual(out_cache_loc.tolist(), [106, 208])
        self.assertEqual([req.kv.kv_allocated_len for req in batch.reqs], [8, 8])

    def test_decode_plan_mixes_crossing_and_in_page_requests(self) -> None:
        """Crossing requests grow while in-page requests retain capacity."""
        batch = _make_batch([8, 8])
        plan = _plan_page_aligned_decode(
            batch,
            write_locs=_make_write_locs([8, 6]),
            token_per_req=1,
            alloc_page_size=4,
        )

        self.assertEqual(plan.crossing_indices_cpu.tolist(), [0])
        self.assertEqual(plan.allocated_next_cpu.tolist(), [12, 8])

    def test_decode_plan_handles_zero_crossings(self) -> None:
        """A fully in-page batch neither allocates nor advances watermarks."""
        batch = _make_batch([8, 8])
        plan = _plan_page_aligned_decode(
            batch,
            write_locs=_make_write_locs([5, 7]),
            token_per_req=1,
            alloc_page_size=4,
        )

        self.assertEqual(plan.crossing_indices_cpu.numel(), 0)
        self.assertEqual(plan.allocated_next_cpu.tolist(), [8, 8])

    def test_decode_plan_rejects_misaligned_watermark(self) -> None:
        """Paged decode fails before accepting a misaligned watermark."""
        batch = _make_batch([6])

        with self.assertRaisesRegex(AssertionError, "must be page-aligned"):
            _plan_page_aligned_decode(
                batch,
                write_locs=_make_write_locs([6]),
                token_per_req=1,
                alloc_page_size=4,
            )

    def test_decode_plan_rejects_unallocated_gap(self) -> None:
        """A decode write beyond its watermark fails before mutation."""
        batch = _make_batch([4])

        with self.assertRaisesRegex(AssertionError, "exceed allocation watermarks"):
            _plan_page_aligned_decode(
                batch,
                write_locs=_make_write_locs([5]),
                token_per_req=1,
                alloc_page_size=4,
            )

    def test_decode_validation_precedes_eviction(self) -> None:
        """Invalid page state fails before the decode eviction hook mutates state."""
        batch = _make_batch([6])
        batch.seq_lens = torch.tensor([6], dtype=torch.int64)
        batch.seq_lens_cpu = torch.tensor([6], dtype=torch.int64)
        batch.model_config = SimpleNamespace(is_encoder_decoder=False)
        batch.maybe_evict_swa = mock.Mock()
        batch.tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=4,
                validate_main_page_aligned_alloc=mock.Mock(),
            )
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            self.assertRaisesRegex(AssertionError, "must be page-aligned"),
        ):
            alloc_for_decode(batch, token_per_req=1)

        batch.maybe_evict_swa.assert_not_called()

    def test_encoder_decoder_write_locs_use_combined_coordinates(self) -> None:
        """Encoder-decoder positions combine encoder and decoder lengths."""
        batch = SimpleNamespace(
            model_config=SimpleNamespace(is_encoder_decoder=True),
            encoder_lens=torch.tensor([3, 4], dtype=torch.int64),
            encoder_lens_cpu=[3, 4],
            seq_lens=torch.tensor([5, 6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5, 6], dtype=torch.int64),
        )

        write_locs = _compute_decode_write_locs(batch)

        self.assertEqual(write_locs.device.tolist(), [8, 10])
        self.assertEqual(write_locs.cpu.tolist(), [8, 10])
        self.assertEqual(write_locs.cpu.dtype, batch.seq_lens_cpu.dtype)

    def test_plain_dcp_uses_allocator_page_for_request_pool_headroom(self) -> None:
        """Request-pool headroom uses the DCP-adjusted allocator page."""
        server_args = SimpleNamespace(
            dcp_size=4,
            page_size=2,
            max_speculative_num_draft_tokens=0,
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            alloc_page_size=lambda: 8,
        )

        self.assertEqual(ServerArgs.alloc_page_size(server_args), 8)
        self.assertEqual(get_req_to_token_extra_context_len(server_args), 11)

    def test_dcp_spec_reserve_uses_allocator_page(self) -> None:
        """Speculative top-k reserve rounds with the DCP allocator page."""
        server_args = SimpleNamespace(
            dcp_size=4,
            page_size=2,
            max_speculative_num_draft_tokens=4,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=2,
            speculative_num_steps=3,
            alloc_page_size=lambda: 8,
        )

        self.assertEqual(get_alloc_len_per_decode(server_args), 32)

    def test_spec_decode_aligns_local_copy_without_mutating_logical_lens(self) -> None:
        """Spec decode aligns physical locals while preserving caller-owned lens."""
        allocator = SimpleNamespace(
            page_size=4,
            validate_spec_decode_alloc=mock.Mock(),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((2, 16), dtype=torch.int32)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=4))
        cur_kv_lens_cpu = torch.tensor([4], dtype=torch.int64)
        cur_kv_lens = cur_kv_lens_cpu.clone()
        nxt_kv_lens_cpu = torch.tensor([5], dtype=torch.int64)
        nxt_kv_lens = nxt_kv_lens_cpu.clone()
        direct_alloc = mock.Mock(return_value=torch.arange(4, dtype=torch.int64))

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(
                allocation_module,
                "alloc_token_slots",
                direct_alloc,
            ),
            mock.patch.object(allocation_module, "get_last_loc") as get_last_loc,
            mock.patch.object(
                allocation_module, "assign_req_to_token_pool_func"
            ) as writer,
        ):
            alloc_for_spec_decode(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                reqs=[req],
                req_pool_indices=torch.tensor([1], dtype=torch.int64),
                cur_kv_lens=cur_kv_lens,
                cur_kv_lens_cpu=cur_kv_lens_cpu,
                nxt_kv_lens=nxt_kv_lens,
                nxt_kv_lens_cpu=nxt_kv_lens_cpu,
                num_needed_tokens=1,
                batch=SimpleNamespace(device=torch.device("cpu")),
            )

        self.assertEqual(nxt_kv_lens.tolist(), [5])
        self.assertEqual(nxt_kv_lens_cpu.tolist(), [5])
        allocator.validate_spec_decode_alloc.assert_called_once_with()
        direct_alloc.assert_called_once_with(tree_cache=tree_cache, num_tokens=4)
        get_last_loc.assert_not_called()
        self.assertEqual(writer.call_args.args[2].tolist(), [4])
        self.assertEqual(writer.call_args.args[3].tolist(), [8])
        self.assertEqual(writer.call_args.args[4].tolist(), list(range(4)))
        self.assertEqual(req.kv.kv_allocated_len, 8)

    def test_spec_decode_page_one_validates_and_allocates_directly(self) -> None:
        """Page-one spec decode checks capability before direct allocation."""
        allocator = SimpleNamespace(
            page_size=1,
            validate_spec_decode_alloc=mock.Mock(),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 8), dtype=torch.int32)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=2))
        direct_alloc = mock.Mock(return_value=torch.tensor([7], dtype=torch.int64))

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(allocation_module, "alloc_token_slots", direct_alloc),
            mock.patch.object(
                allocation_module, "assign_req_to_token_pool_func"
            ) as writer,
        ):
            alloc_for_spec_decode(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                reqs=[req],
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                cur_kv_lens=torch.tensor([2], dtype=torch.int64),
                cur_kv_lens_cpu=torch.tensor([2], dtype=torch.int64),
                nxt_kv_lens=torch.tensor([3], dtype=torch.int64),
                nxt_kv_lens_cpu=torch.tensor([3], dtype=torch.int64),
                num_needed_tokens=1,
                batch=SimpleNamespace(device=torch.device("cpu")),
            )

        allocator.validate_spec_decode_alloc.assert_called_once_with()
        direct_alloc.assert_called_once_with(tree_cache=tree_cache, num_tokens=1)
        writer.assert_called_once()
        self.assertEqual(req.kv.kv_allocated_len, 3)

    def test_spec_decode_rejects_misaligned_watermark_before_allocation(self) -> None:
        """Spec decode rejects a malformed watermark before any allocator lookup."""
        allocator = SimpleNamespace(
            page_size=4,
            validate_spec_decode_alloc=mock.Mock(),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((2, 16), dtype=torch.int32)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=3))

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(allocation_module, "get_last_loc") as get_last_loc,
            self.assertRaisesRegex(AssertionError, "prefix lens"),
        ):
            alloc_for_spec_decode(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                reqs=[req],
                req_pool_indices=torch.tensor([1], dtype=torch.int64),
                cur_kv_lens=torch.tensor([3], dtype=torch.int64),
                cur_kv_lens_cpu=torch.tensor([3], dtype=torch.int64),
                nxt_kv_lens=torch.tensor([5], dtype=torch.int64),
                nxt_kv_lens_cpu=torch.tensor([5], dtype=torch.int64),
                num_needed_tokens=2,
                batch=SimpleNamespace(device=torch.device("cpu")),
            )

        get_last_loc.assert_not_called()
        self.assertEqual(req.kv.kv_allocated_len, 3)

    def test_spec_decode_rejects_row_overflow_before_allocation(self) -> None:
        """Spec decode validates ragged endpoints before allocating or publishing."""
        allocator = SimpleNamespace(
            page_size=4,
            validate_spec_decode_alloc=mock.Mock(),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 6), dtype=torch.int32)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=4))

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(allocation_module, "alloc_token_slots") as direct_alloc,
            mock.patch.object(
                allocation_module, "assign_req_to_token_pool_func"
            ) as writer,
            self.assertRaisesRegex(AssertionError, "row width"),
        ):
            alloc_for_spec_decode(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                reqs=[req],
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                cur_kv_lens=torch.tensor([4], dtype=torch.int64),
                cur_kv_lens_cpu=torch.tensor([4], dtype=torch.int64),
                nxt_kv_lens=torch.tensor([5], dtype=torch.int64),
                nxt_kv_lens_cpu=torch.tensor([5], dtype=torch.int64),
                num_needed_tokens=1,
                batch=SimpleNamespace(device=torch.device("cpu")),
            )

        direct_alloc.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(req.kv.kv_allocated_len, 4)

    def test_spec_decode_rejects_hisparse_before_positive_or_zero_mutation(
        self,
    ) -> None:
        """HiSparse rejects positive and zero spec allocation before mutation."""
        for nxt_len in (8, 4):
            with self.subTest(nxt_len=nxt_len):
                allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
                allocator.page_size = 4
                tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
                req_to_token_pool = SimpleNamespace(
                    req_to_token=torch.zeros((1, 16), dtype=torch.int32)
                )
                req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=4))

                with (
                    mock.patch.object(allocation_module, "_is_npu", False),
                    mock.patch.object(
                        allocation_module, "alloc_token_slots"
                    ) as direct_alloc,
                    mock.patch.object(
                        allocation_module, "assign_req_to_token_pool_func"
                    ) as writer,
                    self.assertRaisesRegex(
                        NotImplementedError,
                        "HiSparseTokenToKVPoolAllocator",
                    ),
                ):
                    alloc_for_spec_decode(
                        tree_cache=tree_cache,
                        req_to_token_pool=req_to_token_pool,
                        reqs=[req],
                        req_pool_indices=torch.tensor([0], dtype=torch.int64),
                        cur_kv_lens=torch.tensor([4], dtype=torch.int64),
                        cur_kv_lens_cpu=torch.tensor([4], dtype=torch.int64),
                        nxt_kv_lens=torch.tensor([nxt_len], dtype=torch.int64),
                        nxt_kv_lens_cpu=torch.tensor([nxt_len], dtype=torch.int64),
                        num_needed_tokens=nxt_len - 4,
                        batch=SimpleNamespace(device=torch.device("cpu")),
                    )

                direct_alloc.assert_not_called()
                writer.assert_not_called()
                self.assertEqual(req.kv.kv_allocated_len, 4)

    def test_spec_decode_npu_preserves_legacy_dispatch(self) -> None:
        """NPU spec decode bypasses capability and keeps the legacy producer."""
        allocator = SimpleNamespace(page_size=4)
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 16), dtype=torch.int32)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=4))
        legacy_alloc = mock.Mock(return_value=torch.arange(4, dtype=torch.int64))

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(
                allocation_module, "get_last_loc", return_value=torch.tensor([3])
            ) as get_last_loc,
            mock.patch.dict(
                allocation_module.ALLOC_EXTEND_FUNCS, {"npu": legacy_alloc}
            ),
            mock.patch.object(allocation_module, "alloc_token_slots") as direct_alloc,
            mock.patch.object(allocation_module, "assign_req_to_token_pool_func"),
        ):
            alloc_for_spec_decode(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                reqs=[req],
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                cur_kv_lens=torch.tensor([4], dtype=torch.int64),
                cur_kv_lens_cpu=torch.tensor([4], dtype=torch.int64),
                nxt_kv_lens=torch.tensor([8], dtype=torch.int64),
                nxt_kv_lens_cpu=torch.tensor([8], dtype=torch.int64),
                num_needed_tokens=4,
                batch=SimpleNamespace(device=SimpleNamespace(type="npu")),
            )

        get_last_loc.assert_called_once()
        legacy_alloc.assert_called_once()
        direct_alloc.assert_not_called()
        self.assertEqual(req.kv.kv_allocated_len, 8)

    def test_spec_decode_npu_page_one_preserves_token_dispatch(self) -> None:
        """NPU page-one spec decode retains the direct token producer."""
        allocator = SimpleNamespace(page_size=1)
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 8), dtype=torch.int32)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=2))
        direct_alloc = mock.Mock(return_value=torch.tensor([7], dtype=torch.int64))

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(allocation_module, "alloc_token_slots", direct_alloc),
            mock.patch.object(allocation_module, "get_last_loc") as get_last_loc,
            mock.patch.object(
                allocation_module, "assign_req_to_token_pool_func"
            ) as writer,
        ):
            alloc_for_spec_decode(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                reqs=[req],
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                cur_kv_lens=torch.tensor([2], dtype=torch.int64),
                cur_kv_lens_cpu=torch.tensor([2], dtype=torch.int64),
                nxt_kv_lens=torch.tensor([3], dtype=torch.int64),
                nxt_kv_lens_cpu=torch.tensor([3], dtype=torch.int64),
                num_needed_tokens=1,
                batch=SimpleNamespace(device=torch.device("cpu")),
            )

        direct_alloc.assert_called_once_with(tree_cache=tree_cache, num_tokens=1)
        get_last_loc.assert_not_called()
        writer.assert_called_once()
        self.assertEqual(req.kv.kv_allocated_len, 3)

    def test_eagle_validates_spec_capability_before_eviction(self) -> None:
        """EAGLE fails closed before its SWA eviction owner site."""
        events: list[str] = []
        allocator = SimpleNamespace(
            page_size=1,
            validate_spec_decode_alloc=mock.Mock(
                side_effect=lambda: events.append("validate")
            ),
        )
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            token_to_kv_pool_allocator=allocator,
            maybe_evict_swa=mock.Mock(side_effect=lambda: events.append("evict")),
            batch_size=lambda: 0,
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
            reqs=[],
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((0, 1), dtype=torch.int32)
            ),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            device=torch.device("cpu"),
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", False),
            mock.patch.object(eagle_utils_module, "_is_npu", False),
            mock.patch.object(
                eagle_utils_module, "get_alloc_reserve_per_decode", return_value=0
            ),
            mock.patch.object(eagle_utils_module, "alloc_for_spec_decode") as alloc,
        ):
            eagle_utils_module.eagle_prepare_for_decode(batch)

        self.assertEqual(events, ["validate", "evict"])
        alloc.assert_called_once()

    def test_release_accepts_only_the_committed_partial_page(self) -> None:
        """Release permits over-allocation only through the committed page."""
        free = mock.Mock()
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=4,
                free=free,
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(16, dtype=torch.int32).reshape(1, 16)
            ),
            is_chunk_cache=lambda: False,
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=5,
            cache_protected_len=4,
            kv=SimpleNamespace(kv_allocated_len=8),
        )
        server_args = SimpleNamespace(
            page_size=2,
            speculative_algorithm=None,
            strip_thinking_cache=False,
        )

        with (
            mock.patch.object(mem_cache_common, "_is_npu", False),
            mock.patch.object(
                mem_cache_common,
                "get_server_args",
                return_value=server_args,
            ),
        ):
            mem_cache_common._release_overallocated_kv_indices(
                req=req,
                tree_cache=tree_cache,
                effective_committed_len=5,
                unhandled_kv_start=4,
                allocated_end=8,
                allocator_page=4,
            )
            with self.assertRaisesRegex(AssertionError, "Unexpected overallocated"):
                mem_cache_common._release_overallocated_kv_indices(
                    req=req,
                    tree_cache=tree_cache,
                    effective_committed_len=5,
                    unhandled_kv_start=4,
                    allocated_end=9,
                    allocator_page=4,
                )

        free.assert_called_once()
        self.assertEqual(free.call_args.args[0].tolist(), [4, 5, 6, 7])

    def test_release_rejects_invalid_handoff_before_caller_free(self) -> None:
        """Release rejects malformed cache ownership before freeing its suffix."""
        free = mock.Mock()
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=4, free=free),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(16, dtype=torch.int32).reshape(1, 16)
            ),
            is_chunk_cache=lambda: False,
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=5,
            cache_protected_len=4,
            kv=SimpleNamespace(kv_allocated_len=8),
        )
        server_args = SimpleNamespace(
            speculative_algorithm=None,
            strip_thinking_cache=False,
        )

        invalid_handoffs: list[tuple[int, str]] = [
            (3, "allocator-page aligned"),
            (8, "out of range"),
            (0, "precedes protected KV"),
        ]
        for unhandled_kv_start, error_pattern in invalid_handoffs:
            with (
                self.subTest(unhandled_kv_start=unhandled_kv_start),
                mock.patch.object(mem_cache_common, "_is_npu", False),
                mock.patch.object(
                    mem_cache_common,
                    "get_server_args",
                    return_value=server_args,
                ),
                self.assertRaisesRegex(AssertionError, error_pattern),
            ):
                mem_cache_common._release_overallocated_kv_indices(
                    req=req,
                    tree_cache=tree_cache,
                    effective_committed_len=5,
                    unhandled_kv_start=unhandled_kv_start,
                    allocated_end=8,
                    allocator_page=4,
                )

        free.assert_not_called()

    def test_release_allows_chunk_cache_zero_handoff(self) -> None:
        """ChunkCache may hand the entire allocation back to the caller."""
        free = mock.Mock()
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=4, free=free),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(16, dtype=torch.int32).reshape(1, 16)
            ),
            is_chunk_cache=lambda: True,
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=5,
            cache_protected_len=4,
            kv=SimpleNamespace(kv_allocated_len=8),
        )
        server_args = SimpleNamespace(
            speculative_algorithm=None,
            strip_thinking_cache=False,
        )

        with (
            mock.patch.object(mem_cache_common, "_is_npu", False),
            mock.patch.object(
                mem_cache_common,
                "get_server_args",
                return_value=server_args,
            ),
        ):
            mem_cache_common._release_overallocated_kv_indices(
                req=req,
                tree_cache=tree_cache,
                effective_committed_len=5,
                unhandled_kv_start=0,
                allocated_end=8,
                allocator_page=4,
            )

        free.assert_called_once()
        self.assertEqual(free.call_args.args[0].tolist(), list(range(8)))


def _make_batch(allocated_lens: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=allocated_len))
            for allocated_len in allocated_lens
        ]
    )


def _make_write_locs(values: list[int]) -> _DecodeWriteLocs:
    tensor = torch.tensor(values, dtype=torch.int64)
    return _DecodeWriteLocs(device=tensor.clone(), cpu=tensor)


def _make_decode_batch(
    *,
    write_locs: list[int],
    allocated_lens: list[int],
) -> SimpleNamespace:
    req_to_token = torch.zeros((len(write_locs), 16), dtype=torch.int32)
    for index, allocated_len in enumerate(allocated_lens):
        req_to_token[index, :allocated_len] = torch.arange(
            (index + 1) * 100 + 1,
            (index + 1) * 100 + allocated_len + 1,
            dtype=torch.int32,
        )

    def write(indices, values) -> None:
        req_to_token[indices] = values

    allocator = SimpleNamespace(
        page_size=4,
        validate_main_page_aligned_alloc=mock.Mock(),
    )
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(
                kv=SimpleNamespace(kv_allocated_len=allocated_len),
                kv_committed_len=write_loc,
            )
            for write_loc, allocated_len in zip(write_locs, allocated_lens)
        ],
        seq_lens=torch.tensor(write_locs, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(write_locs, dtype=torch.int64),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
        req_to_token_pool=SimpleNamespace(
            req_to_token=req_to_token,
            write=mock.Mock(side_effect=write),
        ),
        req_pool_indices=torch.arange(len(write_locs), dtype=torch.int64),
        device=torch.device("cpu"),
        maybe_evict_swa=mock.Mock(),
    )


class _UnsupportedMainAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, *, page_size: int) -> None:
        self.page_size = page_size

    def clear(self) -> None:
        return None

    def alloc(self, need_size: int) -> torch.Tensor | None:
        raise AssertionError("unsupported allocator must not allocate")

    def free(self, free_index: torch.Tensor) -> None:
        raise AssertionError("unsupported allocator must not free")


if __name__ == "__main__":
    unittest.main()
