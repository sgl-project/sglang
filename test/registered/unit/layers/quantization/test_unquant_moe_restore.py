"""Restoring a TRT-LLM BF16 MoE before a weight update must hand loads the canonical expert layout, in place."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

EXPERTS, HIDDEN, INTERMEDIATE = 2, 4, 3


def _layer_in_block_layout():
    # the TRT-LLM postprocess keeps each expert's bytes in another shape
    def expert_weight(numel):
        return torch.nn.Parameter(
            torch.arange(numel, dtype=torch.bfloat16).reshape(EXPERTS, -1, 2),
            requires_grad=False,
        )

    return SimpleNamespace(
        num_local_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        moe_runner_config=SimpleNamespace(is_gated=True),
        w13_weight=expert_weight(EXPERTS * 2 * INTERMEDIATE * HIDDEN),
        w2_weight=expert_weight(EXPERTS * HIDDEN * INTERMEDIATE),
    )


def _moe_method(use_flashinfer_trtllm_moe):
    method = object.__new__(UnquantizedFusedMoEMethod)
    method.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
    return method


class TestRestoreWeightsBeforeLoading(CustomTestCase):
    def test_trtllm_expert_weights_return_to_the_canonical_layout_in_place(self):
        """Loads copy canonical expert tensors into storage peers write by address, which must not move."""
        layer = _layer_in_block_layout()
        data_ptrs = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())

        _moe_method(use_flashinfer_trtllm_moe=True).restore_weights_before_loading(
            layer
        )

        self.assertEqual(
            tuple(layer.w13_weight.shape), (EXPERTS, 2 * INTERMEDIATE, HIDDEN)
        )
        self.assertEqual(tuple(layer.w2_weight.shape), (EXPERTS, HIDDEN, INTERMEDIATE))
        self.assertEqual(
            (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr()), data_ptrs
        )

    def test_other_moe_backends_are_left_as_they_are(self):
        """Only the TRT-LLM BF16 postprocess changes expert shapes."""
        layer = _layer_in_block_layout()
        shapes_before = (layer.w13_weight.shape, layer.w2_weight.shape)

        _moe_method(use_flashinfer_trtllm_moe=False).restore_weights_before_loading(
            layer
        )

        self.assertEqual((layer.w13_weight.shape, layer.w2_weight.shape), shapes_before)


if __name__ == "__main__":
    unittest.main()
