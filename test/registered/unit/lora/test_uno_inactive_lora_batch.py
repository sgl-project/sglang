"""Regression test for UNO's base-only LoRA routing fast path."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.lora.backend.ascend_backend import AscendLoRABackend
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.speculative.uno_lora import uno_lora_backend_for_device
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _InactiveSkippingBackend:
    skip_inactive_lora_batches = True

    def __init__(self):
        self.batch_info = object()
        self.prepare_called = False

    def reset_batch_state(self):
        self.batch_info = None

    def prepare_lora_batch(self, *args, **kwargs):
        self.prepare_called = True


class TestUnoInactiveLoRABatch(CustomTestCase):
    def test_internal_backend_selection(self):
        self.assertEqual(uno_lora_backend_for_device("npu"), "ascend")
        self.assertEqual(uno_lora_backend_for_device("npu:0"), "ascend")
        self.assertEqual(uno_lora_backend_for_device("cuda"), "uno_cublas")

    def test_ascend_explicit_segments_preserve_uno_row_order(self):
        backend = AscendLoRABackend(max_loras_per_batch=2, device=torch.device("cpu"))
        backend.prepare_lora_token_segments(
            segment_lens=[1, 3, 1, 3],
            weight_indices=[0, 1, 0, 1],
            lora_ranks=[0, 16],
            scalings=[0.0, 2.0],
        )

        info = backend.batch_info
        self.assertIsNotNone(info)
        self.assertEqual(info.seg_lens.tolist(), [1, 3, 1, 3])
        self.assertEqual(info.seg_indptr.tolist(), [0, 1, 4, 5, 8])
        self.assertEqual(info.weight_indices.tolist(), [0, 1, 0, 1])
        self.assertEqual(info.expected_tokens, 8)
        self.assertEqual(info.max_len, 3)

    def test_all_base_batch_clears_stale_routing_before_graph_metadata(self):
        backend = _InactiveSkippingBackend()
        manager = LoRAManager.__new__(LoRAManager)
        manager.lora_backend = backend
        forward_batch = SimpleNamespace(lora_ids=[None], batch_size=1)

        manager.prepare_lora_batch(forward_batch)

        self.assertIsNone(backend.batch_info)
        self.assertFalse(backend.prepare_called)


if __name__ == "__main__":
    unittest.main()
