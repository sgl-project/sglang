"""CPU contracts for sparse KV row metadata and native LIM registration."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    sparse_query_width,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.prefill_kv_plan import (
    prepare_prefill_gather_rows,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestLIMState(CustomTestCase):
    def test_sparse_query_width(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, spec_info=None)
        self.assertEqual(sparse_query_width(batch), 1)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = SimpleNamespace(num_tokens_per_req=3)
        self.assertEqual(sparse_query_width(batch), 3)

    def test_prefill_gather_shape_ignores_padding(self):
        # Host row order and total must match the ragged KV payload, including
        # interior padding. Device-side nonzero would require a stream sync.
        keep, total = prepare_prefill_gather_rows(
            torch.tensor([3, -1, 1, 0]), torch.tensor([4, 2, 3, 0])
        )
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(total, 7)
        keep, total = prepare_prefill_gather_rows(
            torch.tensor([0, -1]), torch.tensor([0, 3])
        )
        self.assertEqual((keep.numel(), total), (0, 0))


class TestLIMKernelLoad(CustomTestCase):
    """Package registration must replace the optional external binary path."""

    def test_packaged_loader_preserves_in_place_outputs(self):
        package = ModuleType("sgl_kernel_npu")
        package.__path__ = []
        api = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")

        def native(*tensors, full_visible=False):
            # Emulate the native boundary, not the framework's tensor ordering.
            tensors[11].copy_(torch.arange(2048, dtype=torch.int32).view(1, -1))
            tensors[16].copy_(tensors[7])

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

        def ints(*shape):
            return torch.zeros(*shape, dtype=torch.int32)

        ids, counts = ints(1, 2048), ints(1)
        fused_lightning_indexer_manage(
            weights=torch.ones(1, 32, dtype=torch.bfloat16),
            query=torch.ones(1, 32, 128, dtype=torch.bfloat16),
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
            device_slots=ints(1, 2048),
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
