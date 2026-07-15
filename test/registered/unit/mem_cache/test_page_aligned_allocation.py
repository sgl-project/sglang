import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu import allocator_npu as allocator_npu_module
from sglang.srt.hardware_backend.npu.dsv4 import (
    dsv4_allocator as dsv4_allocator_module,
)
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
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import dflash_info_v2 as dflash_info_v2_module
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
                    coordinator.map_latest_cache_loc_to_buffer(
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

    def test_extend_npu_routes_complete_authority_to_explicit_entry(self) -> None:
        """NPU extend passes ragged prefixes and DSV4 state to its explicit entry."""
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
        dsv4_allocator = SimpleNamespace(
            compute_dsv4_state_lens_extend=mock.Mock(
                return_value=dsv4_state_lens
            )
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(
                allocator_npu_module,
                "alloc_for_extend_npu",
                return_value=physical_slots,
            ) as npu_entry,
            mock.patch.object(
                allocation_module, "alloc_token_slots"
            ) as direct_producer,
            mock.patch.object(
                allocation_module,
                "_resolve_dsv4_npu_allocator",
                return_value=dsv4_allocator,
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
        self.assertIs(
            npu_entry.call_args.kwargs["prefix_tensors"][0], req.prefix_indices
        )
        self.assertEqual(npu_entry.call_args.kwargs["prefix_lens_cpu"].tolist(), [2])
        self.assertEqual(npu_entry.call_args.kwargs["seq_lens_cpu"].tolist(), [5])
        self.assertEqual(npu_entry.call_args.kwargs["extend_num_tokens"], 3)
        self.assertEqual(
            npu_entry.call_args.kwargs["req_pool_indices"].tolist(),
            [0],
        )
        self.assertIs(
            npu_entry.call_args.kwargs["dsv4_state_lens"],
            dsv4_state_lens,
        )
        self.assertIs(
            npu_entry.call_args.kwargs["dsv4_allocator"],
            dsv4_allocator,
        )
        dsv4_allocator.compute_dsv4_state_lens_extend.assert_called_once_with(
            batch.reqs,
            [5],
        )
        self.assertIs(npu_entry.call_args.kwargs["batch"], batch)

    def test_npu_extend_entry_uses_ragged_prefix_anchors(self) -> None:
        """NPU extend anchors each request from its published ragged prefix."""
        batch = SimpleNamespace(device=torch.device("cpu"))
        prefix_tensors = [
            torch.tensor([11, 12], dtype=torch.int64),
            torch.empty((0,), dtype=torch.int64),
        ]
        expected = torch.tensor([101, 102], dtype=torch.int64)

        with mock.patch.object(
            allocator_npu_module,
            "_alloc_paged_token_slots_extend_npu",
            return_value=expected,
        ) as producer:
            result = allocator_npu_module.alloc_for_extend_npu(
                tree_cache=object(),
                prefix_tensors=prefix_tensors,
                prefix_lens=torch.tensor([2, 0], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([2, 0], dtype=torch.int64),
                seq_lens=torch.tensor([3, 1], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([3, 1], dtype=torch.int64),
                extend_num_tokens=2,
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                dsv4_state_lens=None,
                dsv4_allocator=None,
                batch=batch,
            )

        self.assertIs(result, expected)
        self.assertEqual(producer.call_args.kwargs["last_loc"].tolist(), [12, -1])

    def test_npu_page_one_extend_uses_explicit_direct_entry(self) -> None:
        """NPU page-one extend allocates directly without a continuation anchor."""
        allocator = SimpleNamespace(
            page_size=1,
            alloc=mock.Mock(return_value=torch.tensor([11, 12], dtype=torch.int64)),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)

        with (
            mock.patch.object(allocator_npu_module, "evict_from_tree_cache") as evict,
            mock.patch.object(
                allocator_npu_module,
                "_alloc_paged_token_slots_extend_npu",
                side_effect=AssertionError("page-one must not use an anchor"),
            ),
        ):
            result = allocator_npu_module.alloc_for_extend_npu(
                tree_cache=tree_cache,
                prefix_tensors=[torch.tensor([7], dtype=torch.int64)],
                prefix_lens=torch.tensor([1], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([1], dtype=torch.int64),
                seq_lens=torch.tensor([3], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
                extend_num_tokens=2,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                dsv4_state_lens=None,
                dsv4_allocator=None,
                batch=SimpleNamespace(device=torch.device("cpu")),
            )

        evict.assert_called_once_with(tree_cache, 2)
        allocator.alloc.assert_called_once_with(2)
        self.assertEqual(result.tolist(), [11, 12])

    def test_npu_extend_rejects_hisparse_wrapper_before_mutation(self) -> None:
        """NPU extend rejects the unsupported DSV4 HiSparse wrapper before mutation."""
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        req_to_token_pool = SimpleNamespace(alloc=mock.Mock(), write=mock.Mock())
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            req_to_token_pool=req_to_token_pool,
            maybe_evict_swa=mock.Mock(),
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            self.assertRaisesRegex(RuntimeError, "HiSparse is not supported on NPU"),
        ):
            alloc_for_extend(batch)

        batch.maybe_evict_swa.assert_not_called()
        req_to_token_pool.alloc.assert_not_called()
        req_to_token_pool.write.assert_not_called()

    def test_npu_extend_stashes_direct_dsv4_bundle_identity(self) -> None:
        """NPU extend forwards direct DSV4 state and stashes the exact bundle."""
        full_locations = torch.tensor([101, 102], dtype=torch.int64)
        bundle = DSV4OutCacheLoc(
            out_full_loc=full_locations,
            out_swa_loc=torch.tensor([201, 202], dtype=torch.int64),
            out_c4_loc=torch.tensor([301], dtype=torch.int64),
            out_c128_loc=torch.empty((0,), dtype=torch.int64),
            out_c4_state_loc=torch.tensor([401, 402], dtype=torch.int64),
            out_c128_state_loc=torch.tensor([501, 502], dtype=torch.int64),
        )
        allocator = SimpleNamespace(
            page_size=4,
            alloc_extend=mock.Mock(return_value=bundle),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        batch = SimpleNamespace(
            req_to_token_pool=object(),
            out_cache_loc_dsv4=object(),
        )
        state_lens = object()

        with mock.patch.object(allocator_npu_module, "evict_from_tree_cache"):
            result = allocator_npu_module._alloc_paged_token_slots_extend_npu(
                tree_cache=tree_cache,
                prefix_lens=torch.tensor([2], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([2], dtype=torch.int64),
                seq_lens=torch.tensor([4], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
                last_loc=torch.tensor([9], dtype=torch.int64),
                extend_num_tokens=2,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                dsv4_state_lens=state_lens,
                dsv4_allocator=allocator,
                batch=batch,
            )

        self.assertIs(result, full_locations)
        self.assertIs(batch.out_cache_loc_dsv4, bundle)
        self.assertIs(
            allocator.alloc_extend.call_args.kwargs["dsv4_state_lens"],
            state_lens,
        )

    def test_npu_extend_clears_stale_dsv4_bundle_before_oom(self) -> None:
        """NPU extend clears a stale DSV4 bundle before reporting allocation OOM."""
        allocator = SimpleNamespace(
            page_size=4,
            alloc_extend=mock.Mock(return_value=None),
        )
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            pretty_print=mock.Mock(),
        )
        batch = SimpleNamespace(
            req_to_token_pool=object(),
            out_cache_loc_dsv4=object(),
        )

        with (
            mock.patch.object(allocator_npu_module, "evict_from_tree_cache"),
            mock.patch.object(
                allocator_npu_module,
                "available_and_evictable_str",
                return_value="empty",
            ),
            self.assertRaisesRegex(RuntimeError, "Prefill out of memory"),
        ):
            allocator_npu_module._alloc_paged_token_slots_extend_npu(
                tree_cache=tree_cache,
                prefix_lens=torch.tensor([2], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([2], dtype=torch.int64),
                seq_lens=torch.tensor([4], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
                last_loc=torch.tensor([9], dtype=torch.int64),
                extend_num_tokens=2,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                dsv4_state_lens=object(),
                dsv4_allocator=allocator,
                batch=batch,
            )

        self.assertIsNone(batch.out_cache_loc_dsv4)

    def test_npu_reserve_uses_direct_dsv4_state_authority(self) -> None:
        """NPU reserve computes and forwards state lens through one authority."""
        state_lens = object()
        allocator = SimpleNamespace(
            compute_dsv4_state_lens_reserve=mock.Mock(return_value=state_lens)
        )
        batch = SimpleNamespace(
            reqs=[object()],
            req_to_token_pool=object(),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            token_to_kv_pool_allocator=allocator,
        )
        locations = torch.tensor([101, 102], dtype=torch.int64)
        prefix_lens_cpu = torch.tensor([2], dtype=torch.int64)
        seq_lens_cpu = torch.tensor([4], dtype=torch.int64)

        with (
            mock.patch.object(
                dsv4_allocator_module,
                "alloc_paged_token_slots_extend",
                return_value=locations,
            ) as producer,
            mock.patch.object(dsv4_allocator_module, "maybe_write_dsv4_extend"),
        ):
            result = dsv4_allocator_module.alloc_paged_token_slots_reserve_extend(
                tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
                prefix_lens=torch.tensor([2], dtype=torch.int64),
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=torch.tensor([4], dtype=torch.int64),
                seq_lens_cpu=seq_lens_cpu,
                last_loc=torch.tensor([9], dtype=torch.int64),
                extend_num_tokens=2,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                dsv4_allocator=allocator,
                batch=batch,
            )

        self.assertIs(result, locations)
        allocator.compute_dsv4_state_lens_reserve.assert_called_once_with(
            batch.reqs,
            prefix_lens_cpu,
            seq_lens_cpu,
        )
        self.assertIs(
            producer.call_args.kwargs["dsv4_state_lens"],
            state_lens,
        )
        self.assertIs(
            producer.call_args.kwargs["dsv4_allocator"],
            allocator,
        )

    def test_npu_extend_preserves_outer_publication_order(self) -> None:
        """NPU extend publishes writer, gather, hook, then watermark in order."""
        events: list[str] = []
        req = SimpleNamespace(
            prefix_indices=torch.tensor([11, 12], dtype=torch.int64),
            kv=_TrackedKv(kv_allocated_len=2, events=events),
        )
        allocator = SimpleNamespace(page_size=4)
        batch = SimpleNamespace(
            reqs=[req],
            prefix_lens=[2],
            extend_lens=[2],
            extend_num_tokens=2,
            seq_lens=torch.tensor([4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((1, 8), dtype=torch.int32),
                alloc=mock.Mock(return_value=[0]),
            ),
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            token_to_kv_pool_allocator=allocator,
            device=torch.device("cpu"),
            maybe_evict_swa=mock.Mock(),
        )
        locations = torch.tensor([101, 102], dtype=torch.int64)

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(
                allocation_module,
                "_resolve_dsv4_npu_allocator",
                return_value=None,
            ),
            mock.patch.object(
                allocator_npu_module,
                "alloc_for_extend_npu",
                return_value=locations,
            ),
            mock.patch.object(
                allocation_module,
                "write_cache_indices",
                side_effect=lambda **_: events.append("writer"),
            ),
            mock.patch.object(
                allocation_module,
                "gather_out_cache_loc_extend",
                side_effect=lambda *_, **__: (
                    events.append("gather") or locations
                ),
            ),
            mock.patch.object(
                allocation_module,
                "maybe_write_dsv4_extend",
                side_effect=lambda *_: events.append("hook"),
            ),
        ):
            alloc_for_extend(batch)

        self.assertEqual(events, ["writer", "gather", "hook", "watermark"])

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

    def test_npu_decode_entry_uses_combined_current_anchor(self) -> None:
        """NPU decode looks up the tail at the explicit combined current endpoint."""
        req_to_token = torch.zeros((2, 16), dtype=torch.int32)
        req_to_token[0, 4] = 41
        req_to_token[1, 7] = 73
        batch = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            tree_cache=object(),
        )
        expected = torch.tensor([101, 102], dtype=torch.int64)

        with mock.patch.object(
            allocator_npu_module,
            "_alloc_paged_token_slots_decode_npu",
            return_value=expected,
        ) as producer:
            result = allocator_npu_module.alloc_for_decode_npu(
                batch,
                current_combined_lens=torch.tensor([5, 8], dtype=torch.int64),
                next_combined_lens=torch.tensor([6, 9], dtype=torch.int64),
                next_combined_lens_cpu=torch.tensor([6, 9], dtype=torch.int64),
                token_per_req=1,
                dsv4_state_lens=None,
                dsv4_allocator=None,
            )

        self.assertIs(result, expected)
        self.assertEqual(producer.call_args.kwargs["last_loc"].tolist(), [41, 73])
        self.assertEqual(producer.call_args.kwargs["seq_lens"].tolist(), [6, 9])

    def test_npu_page_one_decode_uses_explicit_direct_entry(self) -> None:
        """NPU page-one decode allocates directly without reading the request table."""
        allocator = SimpleNamespace(
            page_size=1,
            alloc=mock.Mock(return_value=torch.tensor([31, 32], dtype=torch.int64)),
        )
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            req_to_token_pool=SimpleNamespace(
                req_to_token=mock.MagicMock(
                    side_effect=AssertionError("page-one must not read an anchor")
                )
            ),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )

        with (
            mock.patch.object(allocator_npu_module, "evict_from_tree_cache") as evict,
            mock.patch.object(
                allocator_npu_module,
                "get_last_loc",
                side_effect=AssertionError("page-one must not read an anchor"),
            ),
        ):
            result = allocator_npu_module.alloc_for_decode_npu(
                batch,
                current_combined_lens=torch.tensor([4, 5], dtype=torch.int64),
                next_combined_lens=torch.tensor([5, 6], dtype=torch.int64),
                next_combined_lens_cpu=torch.tensor([5, 6], dtype=torch.int64),
                token_per_req=1,
                dsv4_state_lens=None,
                dsv4_allocator=None,
            )

        evict.assert_called_once_with(batch.tree_cache, 2)
        allocator.alloc.assert_called_once_with(2)
        self.assertEqual(result.tolist(), [31, 32])

    def test_npu_decode_clears_stale_dsv4_bundle_before_oom(self) -> None:
        """NPU decode clears a stale DSV4 bundle before reporting allocation OOM."""
        allocator = SimpleNamespace(
            page_size=4,
            alloc_decode=mock.Mock(return_value=None),
        )
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            pretty_print=mock.Mock(),
            evictable_size=mock.Mock(return_value=0),
        )
        batch = SimpleNamespace(
            req_to_token_pool=object(),
            out_cache_loc_dsv4=object(),
        )

        with (
            mock.patch.object(allocator_npu_module, "evict_from_tree_cache"),
            mock.patch.object(
                allocator_npu_module,
                "available_and_evictable_str",
                return_value="empty",
            ),
            self.assertRaisesRegex(RuntimeError, "Decode out of memory"),
        ):
            allocator_npu_module._alloc_paged_token_slots_decode_npu(
                tree_cache=tree_cache,
                seq_lens=torch.tensor([5], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
                last_loc=torch.tensor([8], dtype=torch.int64),
                token_per_req=1,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                dsv4_state_lens=object(),
                dsv4_allocator=allocator,
                batch=batch,
            )

        self.assertIsNone(batch.out_cache_loc_dsv4)

    def test_npu_decode_preserves_outer_publication_order(self) -> None:
        """NPU decode publishes req row, watermark, then DSV4 hook in order."""
        events: list[str] = []
        batch = _make_decode_batch(write_locs=[4], allocated_lens=[4])
        batch.reqs[0].kv = _TrackedKv(kv_allocated_len=4, events=events)
        locations = torch.tensor([101], dtype=torch.int64)

        def write(indices, values) -> None:
            events.append("writer")
            batch.req_to_token_pool.req_to_token[indices] = values

        batch.req_to_token_pool.write.side_effect = write

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(
                allocation_module,
                "_resolve_dsv4_npu_allocator",
                return_value=None,
            ),
            mock.patch.object(
                allocator_npu_module,
                "alloc_for_decode_npu",
                return_value=locations,
            ),
            mock.patch.object(
                allocation_module,
                "maybe_write_dsv4_decode",
                side_effect=lambda *_: events.append("hook"),
            ),
        ):
            alloc_for_decode(batch, token_per_req=1)

        self.assertEqual(events, ["writer", "watermark", "hook"])

    def test_npu_decode_separates_combined_positions_from_decoder_watermarks(
        self,
    ) -> None:
        """NPU decode writes combined positions while advancing decoder watermarks."""
        batch = _make_decode_batch(write_locs=[5, 6], allocated_lens=[5, 6])
        batch.model_config = SimpleNamespace(is_encoder_decoder=True)
        batch.encoder_lens = torch.tensor([3, 4], dtype=torch.int64)
        batch.encoder_lens_cpu = [3, 4]
        expected = torch.tensor([101, 102], dtype=torch.int64)
        dsv4_state_lens = object()
        dsv4_allocator = SimpleNamespace(
            compute_dsv4_state_lens_decode=mock.Mock(
                return_value=dsv4_state_lens
            )
        )

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(
                allocator_npu_module,
                "alloc_for_decode_npu",
                return_value=expected,
            ) as npu_entry,
            mock.patch.object(
                allocation_module,
                "_resolve_dsv4_npu_allocator",
                return_value=dsv4_allocator,
            ),
            mock.patch.object(allocation_module, "maybe_write_dsv4_decode"),
        ):
            result = alloc_for_decode(batch, token_per_req=1)

        self.assertEqual(
            npu_entry.call_args.kwargs["current_combined_lens"].tolist(),
            [8, 10],
        )
        self.assertEqual(
            npu_entry.call_args.kwargs["next_combined_lens"].tolist(),
            [9, 11],
        )
        self.assertEqual(
            npu_entry.call_args.kwargs["next_combined_lens_cpu"].tolist(),
            [9, 11],
        )
        self.assertEqual([req.kv.kv_allocated_len for req in batch.reqs], [6, 7])
        self.assertIs(
            npu_entry.call_args.kwargs["dsv4_state_lens"],
            dsv4_state_lens,
        )
        self.assertIs(
            npu_entry.call_args.kwargs["dsv4_allocator"],
            dsv4_allocator,
        )
        dsv4_allocator.compute_dsv4_state_lens_decode.assert_called_once_with(
            batch.reqs
        )
        self.assertEqual(
            batch.req_to_token_pool.req_to_token[
                batch.req_pool_indices,
                torch.tensor([8, 10], dtype=torch.int64),
            ].tolist(),
            expected.tolist(),
        )
        self.assertEqual(result.tolist(), expected.tolist())

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

    def test_spec_decode_npu_rejects_hisparse_wrapper_before_mutation(self) -> None:
        """NPU spec decode rejects the unsupported wrapper before allocation."""
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 8), dtype=torch.int32)
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=2))

        with (
            mock.patch.object(allocation_module, "_is_npu", True),
            mock.patch.object(allocation_module, "alloc_token_slots") as direct_alloc,
            mock.patch.object(allocation_module, "get_last_loc") as get_last_loc,
            mock.patch.object(
                allocation_module,
                "assign_req_to_token_pool_func",
            ) as writer,
            self.assertRaisesRegex(RuntimeError, "HiSparse is not supported on NPU"),
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

        direct_alloc.assert_not_called()
        get_last_loc.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(req.kv.kv_allocated_len, 2)

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

    def test_eagle_and_dflash_share_spec_allocation_entry(self) -> None:
        """EAGLE and DFLASH route through the same owned allocation entry."""
        self.assertIs(
            eagle_utils_module.alloc_for_spec_decode,
            allocation_module.alloc_for_spec_decode,
        )
        self.assertIs(
            dflash_info_v2_module.alloc_for_spec_decode,
            allocation_module.alloc_for_spec_decode,
        )

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


class _TrackedKv:
    def __init__(self, *, kv_allocated_len: int, events: list[str]) -> None:
        self._kv_allocated_len = kv_allocated_len
        self.swa_evicted_seqlen = 0
        self.events = events

    @property
    def kv_allocated_len(self) -> int:
        return self._kv_allocated_len

    @kv_allocated_len.setter
    def kv_allocated_len(self, value: int) -> None:
        self.events.append("watermark")
        self._kv_allocated_len = value


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
