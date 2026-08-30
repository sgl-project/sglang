"""Regression test for W4AFP8 DeepEP-normal post-reorder scaling."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The function under test is a GPU implementation, but this test replaces every
# launched kernel and only verifies the host-side call contract.  Stub the
# extension symbols so importing the module remains valid on CPU CI runners.
_sgl_kernel_stub = ModuleType("sgl_kernel")
_sgl_kernel_stub.cutlass_w4a8_moe_mm = Mock()
_sgl_kernel_stub.get_cutlass_w4a8_moe_mm_data = Mock()
_sgl_kernel_stub.silu_and_mul = Mock()
with patch.dict(sys.modules, {"sgl_kernel": _sgl_kernel_stub}):
    from sglang.srt.layers.moe import cutlass_w4a8_moe as w4a8_moe

_LAYOUT_SENTINEL = object()


class _KernelLauncher:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, _grid):
        return self.fn


class TestW4AFP8DeepEPNormalPostReorder(CustomTestCase):
    num_tokens, hidden_size, intermediate_size = 2, 8, 4
    num_experts, topk = 2, 2

    def _run_deepep_normal(self, extra_patches):
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        topk_weights = torch.full(
            (self.num_tokens, self.topk), 0.5, dtype=torch.float32
        )
        src2dst = torch.arange(self.num_tokens * self.topk, dtype=torch.int64)

        noop_launcher = _KernelLauncher(lambda *args, **kwargs: None)
        preprocess_result = (
            torch.arange(self.num_tokens * self.topk),
            src2dst,
            torch.empty(0),
        )

        strides = torch.zeros((self.num_experts, 3), dtype=torch.int64)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.int32)
        problem_sizes = torch.zeros((self.num_experts, 3), dtype=torch.int32)
        layer = SimpleNamespace(
            w13_weight=torch.zeros(
                (self.num_experts, self.intermediate_size * 2, self.hidden_size // 2),
                dtype=torch.int8,
            ),
            w2_weight=torch.zeros(
                (self.num_experts, self.hidden_size, self.intermediate_size // 2),
                dtype=torch.int8,
            ),
            w13_weight_scale_inv=torch.ones((self.num_experts, 1, 1)),
            w2_weight_scale_inv=torch.ones((self.num_experts, 1, 1)),
            w13_input_scale=torch.ones(1),
            w2_input_scale=torch.ones(1),
        )

        with (
            patch.object(
                w4a8_moe,
                "deepep_run_moe_deep_preprocess",
                return_value=preprocess_result,
            ),
            patch.object(w4a8_moe, "deepep_permute_triton_kernel", noop_launcher),
            patch.object(
                w4a8_moe,
                "get_cutlass_w4a8_moe_mm_data",
                new=lambda *args, **kwargs: None,
                create=True,
            ),
            patch.object(
                w4a8_moe,
                "cutlass_w4a8_moe_mm",
                new=lambda *args, **kwargs: None,
                create=True,
            ),
            patch.object(
                w4a8_moe,
                "per_tensor_quant_fp8",
                new=lambda *args, **kwargs: None,
            ),
            patch.object(w4a8_moe, "silu_and_mul", new=lambda *args, **kwargs: None),
        ):
            with patch.multiple(w4a8_moe, **extra_patches):
                output = w4a8_moe.cutlass_w4a8_moe_deepep_normal(
                    torch.ones(
                        (self.num_tokens, self.hidden_size), dtype=torch.bfloat16
                    ),
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale_inv,
                    layer.w2_weight_scale_inv,
                    topk_weights,
                    topk_ids,
                    strides,
                    strides,
                    strides,
                    strides,
                    strides,
                    strides,
                    strides,
                    strides,
                    expert_offsets,
                    problem_sizes,
                    problem_sizes,
                    layer.w13_input_scale,
                    layer.w2_input_scale,
                )

        self.assertEqual(output.shape, (self.num_tokens, self.hidden_size))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_gluon_post_reorder_receives_neutral_routed_scale(self):
        """The local reduction is unscaled; DeepEP scales after rank combine."""

        calls = []

        def fake_gluon_post_reorder(
            _down_output,
            output,
            _src2dst,
            _topk_weights,
            _topk,
            _hidden_size,
            routed_scaling_factor,
            *,
            BLOCK_SIZE,
            TOPK,
            layout,
            num_warps,
        ):
            calls.append(1)
            self.assertEqual(routed_scaling_factor, 1.0)
            self.assertEqual(TOPK, self.topk)
            self.assertIs(layout, _LAYOUT_SENTINEL)
            output.zero_()

        self._run_deepep_normal(
            {
                "_is_cuda": True,
                "deepep_post_reorder_gluon_kernel": _KernelLauncher(
                    fake_gluon_post_reorder
                ),
                "gluon_post_reorder_layout": lambda *_args: _LAYOUT_SENTINEL,
            }
        )
        self.assertEqual(len(calls), 1)

    def test_triton_fallback_receives_neutral_routed_scale(self):
        """Off CUDA the Triton kernel runs, still with the neutral scale."""

        calls = []

        def fake_post_reorder(
            _down_output,
            output,
            _src2dst,
            _topk_ids,
            _topk_weights,
            _topk,
            _hidden_size,
            routed_scaling_factor,
            *,
            BLOCK_SIZE,
        ):
            calls.append(1)
            self.assertEqual(routed_scaling_factor, 1.0)
            self.assertEqual(BLOCK_SIZE, 512)
            output.zero_()

        self._run_deepep_normal(
            {
                "_is_cuda": False,
                "deepep_post_reorder_triton_kernel": _KernelLauncher(fake_post_reorder),
            }
        )
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
