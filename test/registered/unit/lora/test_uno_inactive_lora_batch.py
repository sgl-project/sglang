"""Regression test for UNO's base-only LoRA routing fast path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.lora.backend.ascend_backend import AscendLoRABackend
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.speculative.uno_lora import (
    init_uno_lora_manager,
    uno_lora_backend_for_device,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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
    def test_recurrent_model_is_rejected_before_loading_or_wrapping_lora(self):
        runner = SimpleNamespace(model_config=object())
        module = "sglang.srt.speculative.uno_lora"
        with (
            patch(module + ".mambaish_config", return_value=object()) as hybrid,
            patch(module + ".LoRAManager") as manager,
            self.assertRaisesRegex(ValueError, "draft-state isolation"),
        ):
            init_uno_lora_manager(runner)
        hybrid.assert_called_once_with(runner.model_config)
        manager.assert_not_called()

    def test_full_attention_model_loads_the_internal_npu_adapter(self):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=object()),
            model=object(),
            load_config=None,
            dtype=torch.bfloat16,
            server_args=object(),
            device="npu",
        )
        module = "sglang.srt.speculative.uno_lora"
        with (
            patch(module + ".mambaish_config", return_value=None),
            patch(
                module + ".get_spec",
                return_value=SimpleNamespace(uno_lora_path="adapter"),
            ),
            patch(
                module + ".LoRARef", return_value=SimpleNamespace(lora_id="internal")
            ),
            patch(module + ".LoRAManager") as manager_cls,
        ):
            manager, adapter_id = init_uno_lora_manager(runner)
        self.assertIs(manager, manager_cls.return_value)
        self.assertEqual(adapter_id, "internal")
        self.assertEqual(manager_cls.call_args.kwargs["lora_backend"], "uno_ascend")
        manager.fetch_new_loras.assert_called_once_with({None, "internal"})

    def test_internal_backend_selection(self):
        self.assertEqual(uno_lora_backend_for_device("npu"), "uno_ascend")
        self.assertEqual(uno_lora_backend_for_device("npu:0"), "uno_ascend")
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
