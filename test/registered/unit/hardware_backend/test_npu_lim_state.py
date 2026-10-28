"""CPU contracts for LIM request state and persistent HBM reservation."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    prepare_prefill_gather_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    resolve_sparse_kv_budget,
    sparse_batch_capacity,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim import (
    LIMDecodeAdapter,
    prepare_lim_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    mtp_compact_topk,
    mtp_selected_kv_view,
    mtp_topk_mask,
    mtp_topk_rows,
    prepare_mtp_rows,
    use_native_mtp_verify,
    validate_mtp_chain,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a")


class TestLIMState(CustomTestCase):
    def test_native_mtp_verify_dispatch_only_for_long_q4_chain(self):
        backend = SimpleNamespace(graph_mode=False)
        manager = SimpleNamespace(
            lim_adapter=SimpleNamespace(mtp_full_visible=True), cache_capacity=12288
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(num_tokens_per_req=4),
            seq_lens_cpu=torch.tensor([12288, 0]),
        )
        self.assertTrue(use_native_mtp_verify(backend, batch, manager))
        batch.seq_lens_cpu = torch.tensor([12287, 12288])
        self.assertFalse(use_native_mtp_verify(backend, batch, manager))
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        backend.graph_mode = True
        self.assertFalse(use_native_mtp_verify(backend, batch, manager))
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        self.assertTrue(use_native_mtp_verify(backend, batch, manager))
        batch.spec_info.num_tokens_per_req = 3
        self.assertFalse(use_native_mtp_verify(backend, batch, manager))

    def test_mtp_chain_validation_rejects_partial_and_out_of_bounds_rows(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            req_pool_indices=torch.tensor([1, 0]),
            seq_lens=torch.tensor([20, 0]),
            positions=torch.tensor([20, 21, 22, 23, 0, 0, 0, 0]),
            out_cache_loc=torch.tensor([1, 2, 3, 4, -1, -1, -1, -1]),
            spec_info=SimpleNamespace(num_tokens_per_req=4),
        )
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            _, positions, live = prepare_mtp_rows(batch, 8, 3, 24)
            validate_mtp_chain(batch, positions, live, 3)
            batch.out_cache_loc[2] = -1
            _, positions, live = prepare_mtp_rows(batch, 8, 3, 24)
            with self.assertRaisesRegex(RuntimeError, "complete contiguous"):
                validate_mtp_chain(batch, positions, live, 3)
            batch.out_cache_loc[2] = 3
            batch.positions[3] = 24  # context bound
            _, positions, live = prepare_mtp_rows(batch, 8, 3, 24)
            with self.assertRaisesRegex(RuntimeError, "complete contiguous"):
                validate_mtp_chain(batch, positions, live, 3)
            batch.positions[3] = 22  # in range but non-contiguous
            _, positions, live = prepare_mtp_rows(batch, 8, 3, 24)
            with self.assertRaisesRegex(RuntimeError, "complete contiguous"):
                validate_mtp_chain(batch, positions, live, 3)
            batch.positions[3] = 23
            batch.req_pool_indices[0] = 0  # active row cannot use padding slot
            _, positions, live = prepare_mtp_rows(batch, 8, 3, 24)
            with self.assertRaisesRegex(RuntimeError, "complete contiguous"):
                validate_mtp_chain(batch, positions, live, 3)

    def test_mtp_tp_padding_does_not_extend_request_windows_or_shrink_k(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            req_pool_indices=torch.tensor([2, 0]),
            seq_lens=torch.tensor([33, 0]),
            positions=torch.tensor([33, 34, 35, 0, 0, 0, 0, 0]),
            out_cache_loc=torch.arange(8),
            spec_info=SimpleNamespace(num_tokens_per_req=3),
        )
        requests, positions, valid = prepare_mtp_rows(batch, 8, 3, 64)
        self.assertEqual(requests.shape, (2, 3))
        self.assertEqual(positions[valid].tolist(), [33, 34, 35])
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        batch.spec_info.num_accept_tokens = torch.tensor([1, 0])
        self.assertEqual(prepare_mtp_rows(batch, 8, 3, 64)[2].sum(), 1)
        sources = torch.arange(2048).view(1, -1)
        padded = mtp_topk_rows(sources, 8)
        self.assertEqual(padded.shape, (8, 2048))
        self.assertTrue(torch.equal(padded[0], sources[0]))
        self.assertTrue((padded[1:] == -1).all())

    def test_mtp_compaction_preserves_vendor_kv_order(self):
        compact = mtp_compact_topk(
            torch.tensor([[0, 11, 5, -1], [2, -1, 8, 1], [1, 2, 3, 4]]),
            torch.tensor([10, 10, 10]),
            torch.tensor([True, True, False]),
        )
        self.assertEqual(
            compact.tolist(), [[0, 5, -1, -1], [2, 8, 1, -1], [-1, -1, -1, -1]]
        )

    def test_mtp_partial_topk_keeps_packed_rows_and_reuses_storage(self):
        for top_k in (1, 3, 8):
            with self.subTest(top_k=top_k):
                workspace = torch.full((4, 8, 1, 3), -99, dtype=torch.int32)
                selected = mtp_selected_kv_view(workspace, 3, top_k)
                self.assertTrue(selected.is_contiguous())
                self.assertEqual(selected.data_ptr(), workspace.data_ptr())
                payload = torch.arange(3 * top_k * 3, dtype=torch.int32)
                selected.copy_(payload.view_as(selected))
                self.assertTrue(
                    torch.equal(workspace.flatten()[: payload.numel()], payload)
                )
                self.assertTrue((workspace.flatten()[payload.numel() :] == -99).all())

    def test_mtp_absolute_positions_and_draft_extend_padding(self):
        # Prefix lengths are unchanged across draft steps. Using seq_lens-1
        # would overwrite a committed token, and padded extend columns must
        # not overwrite a future token with their duplicate/dummy positions.
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([2, 0, 4, 1]),
            seq_lens=torch.tensor([10, 0, 8, 20]),
            positions=torch.tensor([12, 0, 8, 22]),
            out_cache_loc=torch.tensor([100, 0, 102, 103]),
            spec_info=SimpleNamespace(num_accept_tokens=torch.tensor([1, 3, 0, 2])),
        )
        _, positions, valid = prepare_mtp_rows(batch, 4, 4, 64)
        self.assertEqual(positions[valid].tolist(), [12, 22])
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.positions = torch.tensor([10, 11, 12, 0, 0, 0, 8, 9, 10, 20, 21, 22])
        batch.out_cache_loc = torch.arange(12)
        requests, positions, valid = prepare_mtp_rows(batch, 12, 4, 64)
        self.assertEqual(requests[valid].tolist(), [2, 2, 2, 1, 1, 1])
        self.assertEqual(positions[valid].tolist(), [10, 11, 12, 20, 21, 22])
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        _, positions, valid = prepare_mtp_rows(batch, 12, 4, 64)
        self.assertEqual(positions[valid].tolist(), [10, 20, 21])
        batch.out_cache_loc[10] = -1
        _, positions, valid = prepare_mtp_rows(batch, 12, 4, 64)
        self.assertEqual(positions[valid].tolist(), [10, 20])

    def test_mtp_indexshare_masks_future_and_empty_rows(self):
        # Reusing another query's TopK can include future entries in arbitrary
        # columns; selected-KV indices must preserve holes and reject them.
        valid = mtp_topk_mask(
            torch.tensor([[0, 11, 5, -1], [0, 1, 2, 3]]),
            torch.tensor([10, 20]),
            torch.tensor([True, False]),
        )
        self.assertEqual(valid.tolist(), [[True, False, True, False], [False] * 4])

    def test_prefill_gather_shape_ignores_padding_and_checks_host_bounds(self):
        # Host row order and total must match the ragged KV payload, including
        # interior padding. Device-side nonzero would require a stream sync.
        keep, total = prepare_prefill_gather_rows(
            torch.tensor([3, -1, 1, 0]), torch.tensor([4, 2, 3, 0]), 4, 8
        )
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(total, 7)
        keep, total = prepare_prefill_gather_rows(
            torch.tensor([0, -1]), torch.tensor([0, 3]), 4, 8
        )
        self.assertEqual((keep.numel(), total), (0, 0))
        for requests, lengths, error in (
            ([4], [1], "outside sparse pool"),
            ([1], [9], "exceeds sparse context"),
        ):
            with self.subTest(requests=requests, lengths=lengths):
                with self.assertRaisesRegex(RuntimeError, error):
                    prepare_prefill_gather_rows(
                        torch.tensor(requests), torch.tensor(lengths), 4, 8
                    )

    def test_copy_sfa_shares_native_miss_stride_and_masks_interior_padding(self):
        for capacity in (2048, 4096, 8192):
            with self.subTest(capacity=capacity):
                self._check_copy_sfa_metadata(capacity)

    def _check_copy_sfa_metadata(self, capacity):
        batch, top_k = 3, 2048
        context_len = max(4224, capacity + 128)
        ints = lambda *shape: torch.empty(*shape, dtype=torch.int32)
        manager = SimpleNamespace(
            max_context_len=context_len,
            planner_topk_src_ids=ints(batch, 1, top_k),
            planner_topk_dst_slots=ints(batch, 1, top_k),
            planner_topk_miss_counts=ints(batch),
            planner_miss_src_ids=ints(batch, capacity),
            planner_miss_dst_slots=ints(batch, capacity),
            planner_miss_counts=ints(batch),
            planner_actual_seq_lengths_query=torch.arange(
                1, batch + 1, dtype=torch.int32
            ),
            planner_actual_seq_lengths_kv=ints(batch),
            planner_num_cache_tokens=ints(batch),
            hbm_page_offsets=torch.arange(capacity // 128, dtype=torch.int32),
            dram_page_offsets=torch.arange(context_len // 128, dtype=torch.int32),
            planner_hbm_block_table=ints(batch, capacity // 128),
            planner_dram_block_table=ints(batch, context_len // 128),
        )
        adapter = LIMDecodeAdapter.__new__(LIMDecodeAdapter)
        adapter.top_k, adapter.capacity = top_k, capacity
        adapter.fused_copy_metadata = False
        sources = torch.arange(batch * 32768, dtype=torch.int32).view(batch, 32768)
        plan = SimpleNamespace(
            source_ids=torch.arange(top_k, dtype=torch.int32).expand(batch, -1).clone(),
            device_slots=torch.arange(top_k, dtype=torch.int32)
            .expand(batch, -1)
            .clone(),
            topk_misses=torch.tensor([3, 99, 7], dtype=torch.int32),
            miss_sources=sources,
            miss_slots=sources + 1,
            miss_counts=torch.tensor([capacity, 99, 7], dtype=torch.int32),
            valid_rows=torch.tensor([True, False, True]),
            sparse_rows=torch.tensor([True, False, True]),
            request_slots=torch.tensor([3, 4, 1], dtype=torch.int32),
        )
        plan.source_ids[1].fill_(-1)
        metadata = adapter.copy_sfa_metadata(manager, plan)
        self.assertTrue(all(t.is_contiguous() for t in metadata))
        torch.testing.assert_close(metadata[3][0], sources[0])
        torch.testing.assert_close(metadata[3][2], sources[2])
        self.assertEqual(metadata[5].tolist(), [capacity, 0, 7])
        self.assertEqual(metadata[7].tolist(), [capacity, 0, capacity])
        self.assertEqual(metadata[8].tolist(), [capacity, 0, capacity])
        for actual, native in zip(
            metadata[:6],
            (
                plan.source_ids,
                plan.device_slots,
                plan.topk_misses,
                plan.miss_sources,
                plan.miss_slots,
                plan.miss_counts,
            ),
        ):
            self.assertEqual(actual.data_ptr(), native.data_ptr())
        self.assertEqual(metadata[2][1].item(), 0)
        self.assertEqual(
            metadata[9][:, 0].tolist(), [3 * capacity // 128, 0, capacity // 128]
        )
        self.assertEqual(
            metadata[10][:, 0].tolist(), [3 * context_len // 128, 0, context_len // 128]
        )

    def test_full_visible_large_cache_stays_cold_below_capacity(self):
        for capacity in (4096, 8192):
            with self.subTest(capacity=capacity):
                result = prepare_lim_rows(
                    torch.tensor([2048, capacity - 1, capacity, capacity + 1]),
                    torch.tensor([1, 2, 3, 4]),
                    torch.full((6,), -1, dtype=torch.int32),
                    capacity,
                    5,
                    full_visible=True,
                )
                self.assertEqual(result[3].tolist(), [False, False, True, True])
                self.assertEqual(result[5].tolist(), [-3, -3, -2, -2])

    def test_padded_graph_scratch_keeps_request_bounds_and_avoids_tail_workspace(self):
        self._check_padded_graph_scratch(3, 16, 2048)
        self._check_padded_graph_scratch(7, 8, 8192)

    def _check_padded_graph_scratch(self, rows, batch, capacity):
        top_k = 2048
        context = max(4096, capacity + 128)
        key = torch.empty((32, 128, 1, 128), dtype=torch.bfloat16)
        manager = SimpleNamespace(
            indexer_impl="lim",
            size=rows,
            batch_capacity=batch,
            device_cache_size=capacity,
            sparse_context_len=top_k,
            copy_sfa_enabled=True,
            max_context_len=context,
            layer_num=2,
            head_num=1,
            head_dim=576,
            store_dtype=torch.bfloat16,
            device="cpu",
            start_layer=0,
            paged_kv_cache=SimpleNamespace(get_index_k_buffer=lambda _: key),
        )
        with patch(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim.load_fused_lim_operator",
            return_value=None,
        ):
            adapter = LIMDecodeAdapter(manager)
        adapter.is_native = False
        self.assertIsNone(adapter.combined_kv)
        self.assertIsNone(adapter.combined_ids)
        self.assertEqual(adapter.last_prefix.shape, (2, rows + 1))
        self.assertEqual(adapter.source_ids.shape[0], batch)
        req_ids = torch.tensor([1, 2, 99] + [0] * (batch - 3))
        lengths = torch.tensor([2049, 2177, 4097] + [0] * (batch - 3))
        forward_batch = SimpleNamespace(
            req_pool_indices=req_ids,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
        )

        def reference_indexer(**_kwargs):
            return torch.arange(top_k, dtype=torch.int32).expand(batch, 1, -1)

        plan = adapter.select(
            manager,
            query=torch.empty((batch, 32, 128), dtype=torch.bfloat16),
            weights=torch.empty((batch, 32), dtype=torch.bfloat16),
            key=key,
            block_table=torch.zeros((batch, 32), dtype=torch.int32),
            forward_batch=forward_batch,
            layer_id=0,
            reference_indexer=reference_indexer,
        )
        self.assertEqual(plan.request_slots.tolist(), [1, 2] + [rows] * (batch - 2))
        self.assertTrue(torch.all(plan.source_ids[2:] == -1))
        budget = resolve_sparse_kv_budget(
            request_rows=rows,
            graph_batch_capacity=batch,
            context_len=context,
            cache_tokens=capacity,
            num_layers=2,
            top_k=top_k,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
            copy_sfa=True,
        )
        allocated = sum(
            tensor.numel() * tensor.element_size()
            for tensor in vars(adapter).values()
            if isinstance(tensor, torch.Tensor)
        )
        # The native key-scale workspace is charged in the index pool coefficient.
        allocated -= adapter.key_scale.numel() * adapter.key_scale.element_size()
        self.assertEqual(budget.adapter_bytes, allocated)

    def test_graph_budget_grows_scratch_without_growing_residency(self):
        options = dict(
            request_rows=9,
            context_len=32768,
            num_layers=4,
            top_k=2048,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
            copy_sfa=True,
        )
        small = resolve_sparse_kv_budget(**options)
        padded = resolve_sparse_kv_budget(**options, graph_batch_capacity=16)
        self.assertEqual(sparse_batch_capacity(9, 16), 16)
        self.assertEqual(small.device_kv_bytes, padded.device_kv_bytes)
        self.assertEqual(small.slot_map_bytes, padded.slot_map_bytes)
        self.assertEqual(small.host_kv_bytes, padded.host_kv_bytes)
        self.assertEqual(padded.selected_kv_bytes, 16 * 2048 * 1152)
        self.assertGreater(padded.adapter_bytes, small.adapter_bytes)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                **options,
                graph_batch_capacity=16,
                hbm_budget_bytes=small.managed_hbm_bytes,
            )

    def test_mtp_full_visible_budget_reserves_q4_workspace(self):
        options = dict(
            request_rows=4,
            graph_batch_capacity=8,
            context_len=32768,
            cache_tokens=12288,
            num_layers=2,
            top_k=2048,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
            copy_sfa=True,
            copy_sfa_query_heads=32,
        )
        baseline = resolve_sparse_kv_budget(**options)
        mtp = resolve_sparse_kv_budget(**options, mtp_full_visible=True)
        self.assertEqual(
            mtp.adapter_bytes - baseline.adapter_bytes,
            8 * 4 * (2 * 2048 * 4 + 4 + 64 * 4) + 8 * 4,
        )
        self.assertEqual(
            mtp.copy_sfa_bytes - baseline.copy_sfa_bytes, 8 * 4 * 32 * 512 * 2
        )
        fused = resolve_sparse_kv_budget(
            **options, mtp_full_visible=True, fused_copy_sfa_metadata=True
        )
        self.assertEqual(fused.adapter_bytes - mtp.adapter_bytes, (4 + 1 + 8) * 4)
        with self.assertRaisesRegex(ValueError, "capacity"):
            resolve_sparse_kv_budget(
                **{**options, "cache_tokens": 4096}, mtp_full_visible=True
            )
        with self.assertRaisesRegex(ValueError, "native LIM"):
            resolve_sparse_kv_budget(
                **{**options, "copy_sfa": False}, mtp_full_visible=True
            )

    def test_full_visible_q1_transition_padding_and_reuse(self):
        previous = torch.tensor([-1, -1, 2048, 3000, -1], dtype=torch.int32)
        result = prepare_lim_rows(
            torch.tensor([2048, 2049, 2177, 0]),
            torch.tensor([1, 2, 3, 0]),
            previous,
            2048,
            4,
            full_visible=True,
        )
        self.assertEqual(result[2].tolist(), [2048, 2049, 2177, 0])
        self.assertEqual(result[5].tolist(), [-2, -1, -2, -3])
        self.assertEqual(result[0].tolist(), [1, 2, 3, 4])

    def test_page_boundary_and_cold_to_warm(self):
        previous = torch.full((5,), -1, dtype=torch.int32)
        slots = torch.tensor([1, 2, 3])
        lengths = torch.tensor([2048, 2049, 2176])
        result = prepare_lim_rows(lengths, slots, previous, 2048, 4)
        self.assertEqual(result[2].tolist(), [1920, 2048, 2048])
        self.assertEqual(result[5].tolist(), [-3, -2, -2])
        previous[result[0].long()] = result[6]
        result = prepare_lim_rows(lengths + 1, slots, previous, 2048, 4)
        self.assertEqual(result[2].tolist(), [2048, 2048, 2176])
        self.assertEqual(result[5].tolist(), [-2, -1, -1])

    def test_padding_reset_and_rollback(self):
        previous = torch.tensor([-1, 8192, -1, -1, -1], dtype=torch.int32)
        slots = torch.tensor([1, 0, -1, 99])
        result = prepare_lim_rows(
            torch.tensor([4097, 100, 0, 9000]), slots, previous, 4096, 4
        )
        self.assertEqual(result[0].tolist(), [1, 4, 4, 4])
        self.assertEqual(result[1].tolist(), [4097, 0, 0, 0])
        self.assertEqual(result[5].tolist(), [-2, -3, -3, -3])
        previous[1] = -1  # request pool slot was freed/reused
        result = prepare_lim_rows(
            torch.tensor([8193]), torch.tensor([1]), previous, 4096, 4
        )
        self.assertEqual(result[5].tolist(), [-2])

    def test_copy_sfa_budget_includes_planner_and_page_aligned_host_rows(self):
        budget = resolve_sparse_kv_budget(
            request_rows=3,
            context_len=4097,
            num_layers=4,
            top_k=2048,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
            copy_sfa=True,
            copy_sfa_query_heads=8,
        )
        self.assertEqual(budget.cache_tokens, 2048)
        self.assertEqual(budget.host_kv_bytes, 3 * 4 * 4224 * 1152)
        self.assertGreater(budget.copy_sfa_bytes, 4 * 4 * 2048 * 4)
        with self.assertRaisesRegex(ValueError, "requires the native LIM indexer"):
            resolve_sparse_kv_budget(
                request_rows=3,
                context_len=4097,
                num_layers=4,
                top_k=2048,
                kv_bytes_per_token=1152,
                cache_tokens=3072,
                copy_sfa=True,
            )

        for capacity in (4096, 8192):
            with self.subTest(capacity=capacity):
                larger = resolve_sparse_kv_budget(
                    request_rows=3,
                    context_len=4097,
                    num_layers=4,
                    top_k=2048,
                    kv_bytes_per_token=1152,
                    cache_tokens=capacity,
                    indexer_impl="lim",
                    copy_sfa=True,
                    copy_sfa_query_heads=8,
                )
                delta = capacity - 2048
                # Every extra resident slot has KV and owner storage; misses
                # grow to C while attention's selected-K workspace stays fixed.
                expected_extra = (
                    delta * (3 * 4 * 1152 + 4 * 4 * 4 + 3 * 2 * 4)
                    + 4 * (delta // 128) * 4
                )
                self.assertEqual(
                    larger.managed_hbm_bytes - budget.managed_hbm_bytes, expected_extra
                )
                self.assertEqual(larger.selected_kv_bytes, budget.selected_kv_bytes)
                with self.assertRaisesRegex(ValueError, "exceeding"):
                    resolve_sparse_kv_budget(
                        request_rows=3,
                        context_len=4097,
                        num_layers=4,
                        top_k=2048,
                        kv_bytes_per_token=1152,
                        cache_tokens=capacity,
                        indexer_impl="lim",
                        copy_sfa=True,
                        copy_sfa_query_heads=8,
                        hbm_budget_bytes=larger.managed_hbm_bytes - 1,
                    )

    def test_native_budget_accounts_actual_persistent_buffers(self):
        kwargs = dict(
            request_rows=3,
            context_len=32896,
            num_layers=4,
            top_k=2048,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
        )
        budget = resolve_sparse_kv_budget(**kwargs, cache_tokens=4096)
        # C>K is managed by the native pool; no duplicate Python LRU arrays.
        self.assertEqual(budget.policy_bytes, 2048 * 8)
        self.assertEqual(budget.slot_map_bytes, 4 * 4 * 32896 * 4)
        self.assertGreater(budget.adapter_bytes, 3 * (2048 + 128) * 1152)
        exact = resolve_sparse_kv_budget(
            **kwargs, hbm_budget_bytes=budget.managed_hbm_bytes
        )
        self.assertEqual(exact.cache_tokens, 4096)
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            resolve_sparse_kv_budget(**kwargs, cache_tokens=4097)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                **kwargs,
                cache_tokens=4096,
                hbm_budget_bytes=budget.managed_hbm_bytes - 1,
            )


class TestLIMKernelLoad(CustomTestCase):
    """Package registration must replace the optional external binary path."""

    def test_packaged_loader_preserves_in_place_outputs(self):
        package = ModuleType("sgl_kernel_npu")
        package.__path__ = []
        api = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")

        def native(*tensors):
            # Emulate the native boundary, not the framework's tensor ordering.
            tensors[13].copy_(torch.arange(2048, dtype=torch.int32).view(1, 1, -1))
            tensors[18].copy_(tensors[9])

        api.fused_lightning_indexer_manage = native
        modules = {"sgl_kernel_npu": package, api.__name__: api}
        registration = SimpleNamespace(
            fused_lightning_indexer_manage=SimpleNamespace(default=native)
        )
        with (
            patch.dict("sys.modules", modules),
            patch.object(torch.ops, "npu", registration),
        ):
            operator = load_fused_lim_operator()
        ints = lambda *shape: torch.zeros(*shape, dtype=torch.int32)
        ids, counts = ints(1, 1, 2048), ints(1)
        fused_lightning_indexer_manage(
            weights=torch.ones(1, 32, dtype=torch.bfloat16),
            query_scale=torch.ones(1, 32),
            query=torch.ones(1, 32, 128, dtype=torch.bfloat16),
            key_scale=torch.ones(33, 128, 1),
            key=torch.ones(33, 128, 1, 128, dtype=torch.bfloat16),
            block_table=ints(1, 33),
            query_ends=ints(1),
            seq_lens=ints(1),
            prefix_lens=ints(1),
            cache_sizes=torch.tensor([4096], dtype=torch.int32),
            request_state=ints(1),
            request_slots=ints(1),
            slot_pool=ints(1, 4224),
            source_ids=ids,
            device_slots=ints(1, 1, 2048),
            topk_misses=ints(1),
            miss_sources=ints(1, 32768),
            miss_slots=ints(1, 32768),
            miss_counts=counts,
            operator=operator,
        )
        self.assertEqual(counts.item(), 4096)
        torch.testing.assert_close(ids.view(-1), torch.arange(2048, dtype=torch.int32))

    def test_missing_native_registration_fails_before_graph(self):
        package = ModuleType("sgl_kernel_npu")
        package.__path__ = []
        api = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")
        api.fused_lightning_indexer_manage = lambda *args: None
        with (
            patch.dict("sys.modules", {"sgl_kernel_npu": package, api.__name__: api}),
            patch.object(torch.ops, "npu", SimpleNamespace()),
        ):
            with self.assertRaisesRegex(RuntimeError, "matching kernel wheel"):
                load_fused_lim_operator()


if __name__ == "__main__":
    unittest.main()
