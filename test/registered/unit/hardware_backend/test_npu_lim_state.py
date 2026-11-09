"""CPU contracts for sparse KV row metadata and native LIM registration."""

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
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    prepare_mtp_rows,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestLIMState(CustomTestCase):
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
