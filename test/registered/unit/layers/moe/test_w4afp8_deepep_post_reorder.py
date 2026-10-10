"""Regression test for W4AFP8 DeepEP-normal post-reorder scaling."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# The function under test is a GPU implementation, but this test replaces every
# launched kernel and only verifies the host-side call contract.  Stub the
# extension symbols so importing the module remains valid on CPU CI runners.
_sgl_kernel_stub = ModuleType("sgl_kernel")
_sgl_kernel_stub.cutlass_w4a8_moe_mm = Mock()
_sgl_kernel_stub.get_cutlass_w4a8_moe_mm_data = Mock()
_sgl_kernel_stub.silu_and_mul = Mock()
with patch.dict(sys.modules, {"sgl_kernel": _sgl_kernel_stub}):
    from sglang.srt.layers.moe import cutlass_w4a8_moe as w4a8_moe


class _KernelLauncher:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, _grid):
        return self.fn


class TestW4AFP8DeepEPNormalPostReorder(CustomTestCase):
    def test_post_reorder_receives_neutral_routed_scale(self):
        """The local reduction is unscaled; DeepEP scales after rank combine."""

        num_tokens, hidden_size, intermediate_size = 2, 8, 4
        num_experts, topk = 2, 2
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        topk_weights = torch.full((num_tokens, topk), 0.5, dtype=torch.float32)
        src2dst = torch.arange(num_tokens * topk, dtype=torch.int64)

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
            self.assertEqual(routed_scaling_factor, 1.0)
            self.assertEqual(BLOCK_SIZE, 512)
            output.zero_()

        noop_launcher = _KernelLauncher(lambda *args, **kwargs: None)
        post_reorder_launcher = _KernelLauncher(fake_post_reorder)
        preprocess_result = (
            torch.arange(num_tokens * topk),
            src2dst,
            torch.empty(0),
        )

        strides = torch.zeros((num_experts, 3), dtype=torch.int64)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32)
        problem_sizes = torch.zeros((num_experts, 3), dtype=torch.int32)
        layer = SimpleNamespace(
            w13_weight=torch.zeros(
                (num_experts, intermediate_size * 2, hidden_size // 2),
                dtype=torch.int8,
            ),
            w2_weight=torch.zeros(
                (num_experts, hidden_size, intermediate_size // 2),
                dtype=torch.int8,
            ),
            w13_weight_scale_inv=torch.ones((num_experts, 1, 1)),
            w2_weight_scale_inv=torch.ones((num_experts, 1, 1)),
            w13_input_scale=torch.ones(1),
            w2_input_scale=torch.ones(1),
        )

        quant_mock = Mock()
        with (
            patch.object(
                w4a8_moe,
                "deepep_run_moe_deep_preprocess",
                return_value=preprocess_result,
            ),
            patch.object(w4a8_moe, "deepep_permute_triton_kernel", noop_launcher),
            patch.object(
                w4a8_moe,
                "deepep_post_reorder_triton_kernel",
                post_reorder_launcher,
            ),
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
                new=quant_mock,
            ),
            patch.object(w4a8_moe, "silu_and_mul", new=lambda *args, **kwargs: None),
        ):
            for input_dtype, expected_quant_calls in (
                (torch.bfloat16, 2),
                (torch.float8_e4m3fn, 1),
            ):
                with self.subTest(input_dtype=input_dtype):
                    quant_mock.reset_mock()
                    output = w4a8_moe.cutlass_w4a8_moe_deepep_normal(
                        torch.ones((num_tokens, hidden_size), dtype=input_dtype),
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

                    self.assertEqual(output.shape, (num_tokens, hidden_size))
                    self.assertEqual(output.dtype, torch.bfloat16)
                    # Static-FP8 payloads skip the receive-side requantization;
                    # only the intermediate SiLU*up quant remains.
                    self.assertEqual(quant_mock.call_count, expected_quant_calls)
                    self.assertEqual(
                        quant_mock.call_args.args[0].shape[-1], intermediate_size
                    )

    def test_static_fp8_payload_permuted_via_bf16_view(self):
        """The FP8 payload reuses the exact BF16 element-copy permute path."""

        num_tokens, hidden_size, intermediate_size = 2, 8, 4
        num_experts, topk = 2, 2
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        topk_weights = torch.full((num_tokens, topk), 0.5, dtype=torch.float32)
        src2dst = torch.arange(num_tokens * topk, dtype=torch.int64)

        permute_calls = []

        def fake_permute(a_perm, out, _src2dst, _topk_ids, _unused, _topk, _k, **kw):
            permute_calls.append((a_perm.dtype, tuple(a_perm.shape), out.dtype))

        preprocess_result = (
            torch.arange(num_tokens * topk),
            src2dst,
            torch.empty(0),
        )
        strides = torch.zeros((num_experts, 3), dtype=torch.int64)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32)
        problem_sizes = torch.zeros((num_experts, 3), dtype=torch.int32)

        mm_a_inputs = []

        def fake_mm(_c, a, *_args, **_kwargs):
            mm_a_inputs.append((a.dtype, tuple(a.shape)))

        with (
            patch.object(
                w4a8_moe,
                "deepep_run_moe_deep_preprocess",
                return_value=preprocess_result,
            ),
            patch.object(
                w4a8_moe,
                "deepep_permute_triton_kernel",
                _KernelLauncher(fake_permute),
            ),
            patch.object(
                w4a8_moe,
                "deepep_post_reorder_triton_kernel",
                _KernelLauncher(lambda *args, **kwargs: None),
            ),
            patch.object(
                w4a8_moe,
                "get_cutlass_w4a8_moe_mm_data",
                new=lambda *args, **kwargs: None,
                create=True,
            ),
            patch.object(
                w4a8_moe,
                "cutlass_w4a8_moe_mm",
                new=fake_mm,
                create=True,
            ),
            patch.object(
                w4a8_moe,
                "per_tensor_quant_fp8",
                new=lambda *args, **kwargs: None,
            ),
            patch.object(w4a8_moe, "silu_and_mul", new=lambda *args, **kwargs: None),
        ):
            w4a8_moe.cutlass_w4a8_moe_deepep_normal(
                torch.ones((num_tokens, hidden_size), dtype=torch.float8_e4m3fn),
                torch.zeros(
                    (num_experts, intermediate_size * 2, hidden_size // 2),
                    dtype=torch.int8,
                ),
                torch.zeros(
                    (num_experts, hidden_size, intermediate_size // 2),
                    dtype=torch.int8,
                ),
                torch.ones((num_experts, 1, 1)),
                torch.ones((num_experts, 1, 1)),
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
                torch.ones(1),
                torch.ones(1),
            )

        # The permute kernel sees a BF16 view with half the columns, and the
        # first GEMM receives the restored FP8 tensor with the full hidden size.
        self.assertEqual(
            permute_calls,
            [
                (
                    torch.bfloat16,
                    (num_tokens, hidden_size // 2),
                    torch.bfloat16,
                )
            ],
        )
        self.assertEqual(
            mm_a_inputs[0],
            (torch.float8_e4m3fn, (num_tokens * topk, hidden_size)),
        )


if __name__ == "__main__":
    unittest.main()
