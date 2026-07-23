"""CPU behavior tests for the shared resident/LoRA MoE finalization path."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.moe.fused_moe_triton.layer as fused_moe_module
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


class TestSglLoraMoeFinalize(unittest.TestCase):
    def test_trim_contiguous_dtype_and_optional_distributed_reduction(self):
        combine_input = object()
        padded_fp32 = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        expected = padded_fp32[:, :3].contiguous()

        for reduce_results in (False, True):
            with self.subTest(reduce_results=reduce_results):
                layer = object.__new__(FusedMoE)
                torch.nn.Module.__init__(layer)
                layer.dispatcher = SimpleNamespace(
                    combine=Mock(return_value=padded_fp32)
                )
                layer.reduce_results = reduce_results
                layer.moe_tp_size = 2
                layer.moe_ep_size = 1

                reduced = expected + 100
                tp_group = object()
                with (
                    patch.object(
                        fused_moe_module,
                        "is_allocation_symmetric",
                        return_value=False,
                    ),
                    patch.object(
                        fused_moe_module,
                        "get_tp_group",
                        return_value=tp_group,
                    ),
                    patch.object(
                        fused_moe_module,
                        "use_symmetric_memory",
                        return_value=nullcontext(),
                    ) as symmetric_memory,
                    patch.object(
                        fused_moe_module,
                        "tensor_model_parallel_all_reduce",
                        return_value=reduced,
                    ) as all_reduce,
                ):
                    result = layer.combine_and_finalize(
                        combine_input=combine_input,
                        origin_hidden_states_dim=3,
                    )

                layer.dispatcher.combine.assert_called_once_with(
                    combine_input=combine_input
                )
                symmetric_memory.assert_called_once_with(tp_group, disabled=True)
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(result.shape, (2, 3))
                self.assertTrue(result.is_contiguous())
                if reduce_results:
                    all_reduce.assert_called_once()
                    reduced_input = all_reduce.call_args.args[0]
                    self.assertEqual(reduced_input.dtype, torch.float32)
                    self.assertTrue(reduced_input.is_contiguous())
                    torch.testing.assert_close(reduced_input, expected)
                    torch.testing.assert_close(result, reduced)
                else:
                    all_reduce.assert_not_called()
                    torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
