"""CPU contracts for LIM request state and persistent HBM reservation."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    prepare_prefill_gather_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    resolve_sparse_kv_budget,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim import (
    LIMDecodeAdapter,
    prepare_lim_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a")


class TestLIMState(CustomTestCase):
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

    def test_copy_sfa_packs_native_miss_stride_and_masks_interior_padding(self):
        batch, top_k = 3, 2048
        ints = lambda *shape: torch.empty(*shape, dtype=torch.int32)
        manager = SimpleNamespace(
            max_context_len=4224,
            planner_topk_src_ids=ints(batch, 1, top_k),
            planner_topk_dst_slots=ints(batch, 1, top_k),
            planner_topk_miss_counts=ints(batch),
            planner_miss_src_ids=ints(batch, top_k),
            planner_miss_dst_slots=ints(batch, top_k),
            planner_miss_counts=ints(batch),
            planner_actual_seq_lengths_query=torch.arange(
                1, batch + 1, dtype=torch.int32
            ),
            planner_actual_seq_lengths_kv=ints(batch),
            planner_num_cache_tokens=ints(batch),
            hbm_page_offsets=torch.arange(16, dtype=torch.int32),
            dram_page_offsets=torch.arange(33, dtype=torch.int32),
            planner_hbm_block_table=ints(batch, 16),
            planner_dram_block_table=ints(batch, 33),
        )
        adapter = LIMDecodeAdapter.__new__(LIMDecodeAdapter)
        adapter.top_k = adapter.capacity = top_k
        sources = torch.arange(batch * 32768, dtype=torch.int32).view(batch, 32768)
        plan = SimpleNamespace(
            source_ids=torch.arange(top_k, dtype=torch.int32).expand(batch, -1).clone(),
            device_slots=torch.arange(top_k, dtype=torch.int32).expand(batch, -1),
            topk_misses=torch.tensor([3, 99, 7], dtype=torch.int32),
            miss_sources=sources,
            miss_slots=sources + 1,
            miss_counts=torch.tensor([3, 99, 7], dtype=torch.int32),
            valid_rows=torch.tensor([True, False, True]),
            sparse_rows=torch.tensor([True, False, True]),
            request_slots=torch.tensor([3, 4, 1], dtype=torch.int32),
        )
        plan.source_ids[1].fill_(-1)
        metadata = adapter.copy_sfa_metadata(manager, plan)
        self.assertTrue(all(t.is_contiguous() for t in metadata))
        torch.testing.assert_close(metadata[3][2], sources[2, :top_k])
        self.assertEqual(metadata[5].tolist(), [3, 0, 7])
        self.assertEqual(metadata[7].tolist(), [2048, 0, 2048])
        self.assertTrue(torch.all(metadata[1][1] == -1))
        self.assertTrue(torch.all(metadata[3][1] == -1))
        self.assertEqual(metadata[9][:, 0].tolist(), [48, 0, 16])
        self.assertEqual(metadata[10][:, 0].tolist(), [99, 0, 33])

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
        with self.assertRaisesRegex(ValueError, "K=C=2048"):
            resolve_sparse_kv_budget(
                request_rows=3,
                context_len=4097,
                num_layers=4,
                top_k=2048,
                kv_bytes_per_token=1152,
                cache_tokens=4096,
                copy_sfa=True,
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
