"""CPU contracts for standard [128,128] block-FP8 on Ascend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

from sglang.srt.hardware_backend.npu.quantization import linear_method_npu
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    fp8_grouped_matmul_npu,
    fp8_matmul_npu,
    relayout_npu_block_fp8_weight,
    validate_npu_block_fp8_model_dtype,
    validate_npu_block_fp8_moe_config,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUBlockFP8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner.ascend import (
    pre_permute_deepep_ll_to_ascend,
    pre_permute_deepep_normal_to_ascend,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import fp8, fp8_utils
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeNPUOps:
    def __init__(self):
        self.dynamic_calls = []
        self.quant_calls = []
        self.grouped_calls = []
        self.soft_calls = []

    def npu_dynamic_block_quant(self, value, **kwargs):
        self.dynamic_calls.append((value, kwargs))
        qvalue = torch.empty(value.shape, dtype=torch.float8_e4m3fn)
        scale = torch.ones(value.shape[0], value.shape[1] // 128, dtype=torch.float32)
        return qvalue, scale

    def npu_quant_matmul(self, value, weight, **kwargs):
        self.quant_calls.append((value, weight, kwargs))
        return torch.zeros(value.shape[0], weight.shape[-1], dtype=torch.bfloat16)

    def npu_grouped_matmul(self, **kwargs):
        self.grouped_calls.append(kwargs)
        value = kwargs["x"][0]
        weight = kwargs["weight"][0]
        return [torch.zeros(value.shape[0], weight.shape[-1], dtype=torch.bfloat16)]

    def softfp8_w8a16_matmul(self, value, weight, scale, output_dtype):
        self.soft_calls.append((value, weight, scale, output_dtype))
        return torch.zeros(value.shape[0], weight.shape[-1], dtype=torch.bfloat16)

    def softfp8_w8a16_grouped_matmul(
        self, value, weight, scale, group_list, output_dtype
    ):
        self.soft_calls.append((value, weight, scale, group_list, output_dtype))
        return torch.zeros(value.shape[0], weight.shape[-1], dtype=torch.bfloat16)


class TestNPUBlockFP8Layout(CustomTestCase):
    def test_model_dtype_rejects_fp16_before_runtime(self):
        validate_npu_block_fp8_model_dtype(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "requires model dtype"):
            validate_npu_block_fp8_model_dtype(torch.float16)

    def test_moe_bias_is_rejected_before_runtime(self):
        with self.assertRaisesRegex(ValueError, "expert bias"):
            validate_npu_block_fp8_moe_config(torch.bfloat16, with_bias=True)

    def test_dense_relayout_distinguishes_a5_payload_dtype(self):
        raw = (
            torch.arange(256 * 128, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .reshape(256, 128)
        )
        weight = raw.view(torch.float8_e4m3fn)
        scale = torch.arange(2, dtype=torch.float32).reshape(2, 1)

        a5_weight, a5_scale = relayout_npu_block_fp8_weight(
            weight, scale, [128, 128], before_a5=False
        )
        legacy_weight, legacy_scale = relayout_npu_block_fp8_weight(
            weight, scale, [128, 128], before_a5=True
        )

        self.assertEqual(a5_weight.shape, (128, 256))
        self.assertEqual(a5_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(legacy_weight.dtype, torch.uint8)
        torch.testing.assert_close(a5_weight.view(torch.uint8), legacy_weight)
        torch.testing.assert_close(a5_scale, scale.t())
        torch.testing.assert_close(legacy_scale, scale.t())

    def test_expert_relayout_transposes_weight_and_block_grid(self):
        weight = torch.empty((2, 256, 128), dtype=torch.float8_e4m3fn)
        scale = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1)

        weight, scale = relayout_npu_block_fp8_weight(
            weight, scale, [128, 128], before_a5=False
        )

        self.assertEqual(weight.shape, (2, 128, 256))
        self.assertEqual(scale.shape, (2, 1, 2))
        torch.testing.assert_close(scale, torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]]))

    def test_layout_rejects_unsupported_block_and_scale_shape(self):
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, r"\[128, 128\]"):
            relayout_npu_block_fp8_weight(weight, scale, [64, 128], before_a5=False)
        with self.assertRaisesRegex(ValueError, "scale shape mismatch"):
            relayout_npu_block_fp8_weight(
                weight,
                torch.ones((2, 1), dtype=torch.float32),
                [128, 128],
                before_a5=False,
            )


class TestNPUBlockFP8Ops(CustomTestCase):
    def _patched_runtime(self, ops, *, a5):
        return (
            patch.object(linear_method_npu, "_require_same_npu_device"),
            patch.object(linear_method_npu, "_npu_is_a5_for_tensor", return_value=a5),
            patch.object(linear_method_npu, "_get_npu_ops", return_value=ops),
        )

    def test_dense_a5_dynamic_quant_and_quant_matmul_contract(self):
        ops = _FakeNPUOps()
        x = torch.zeros((2, 128), dtype=torch.bfloat16)
        weight = torch.empty((128, 256), dtype=torch.float8_e4m3fn)
        scale = torch.ones((1, 2), dtype=torch.float32)
        device_patch, generation_patch, ops_patch = self._patched_runtime(ops, a5=True)
        with device_patch, generation_patch, ops_patch:
            output = fp8_matmul_npu(x, weight, [128, 128], scale)

        self.assertEqual(output.shape, (2, 256))
        self.assertEqual(len(ops.dynamic_calls), 1)
        self.assertEqual(ops.dynamic_calls[0][1]["row_block_size"], 1)
        self.assertEqual(ops.dynamic_calls[0][1]["col_block_size"], 128)
        self.assertEqual(ops.quant_calls[0][2]["group_sizes"], (1, 128, 128))
        self.assertIs(ops.quant_calls[0][2]["scale"], scale)

    def test_dense_a5_accepts_prequantized_payload_without_requantizing(self):
        ops = _FakeNPUOps()
        x = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        input_scale = torch.ones((2, 1), dtype=torch.float32)
        weight = torch.empty((128, 256), dtype=torch.float8_e4m3fn)
        scale = torch.ones((1, 2), dtype=torch.float32)
        device_patch, generation_patch, ops_patch = self._patched_runtime(ops, a5=True)
        with device_patch, generation_patch, ops_patch:
            fp8_matmul_npu(
                x,
                weight,
                [128, 128],
                scale,
                input_scale=input_scale,
            )

        self.assertFalse(ops.dynamic_calls)
        self.assertEqual(ops.quant_calls[0][0].data_ptr(), x.data_ptr())

    def test_pre_a5_dense_uses_byte_payload_and_soft_fp8(self):
        ops = _FakeNPUOps()
        x = torch.zeros((2, 128), dtype=torch.bfloat16)
        weight = torch.empty((128, 256), dtype=torch.uint8)
        scale = torch.ones((1, 2), dtype=torch.float32)
        device_patch, generation_patch, ops_patch = self._patched_runtime(ops, a5=False)
        with device_patch, generation_patch, ops_patch:
            output = fp8_matmul_npu(x, weight, [128, 128], scale)

        self.assertEqual(output.shape, (2, 256))
        self.assertEqual(len(ops.soft_calls), 1)
        self.assertFalse(ops.dynamic_calls)

    def test_grouped_a5_preserves_prefill_and_decode_group_list_contracts(self):
        cases = (
            (torch.tensor([2, 3]), 0),  # cumulative offsets (prefill contract)
            (torch.tensor([2, 1]), 1),  # per-expert counts (decode contract)
        )
        for group_list, group_list_type in cases:
            with self.subTest(group_list_type=group_list_type):
                ops = _FakeNPUOps()
                x = torch.zeros((3, 128), dtype=torch.bfloat16)
                weight = torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)
                scale = torch.ones((2, 1, 2), dtype=torch.float32)
                device_patch, generation_patch, ops_patch = self._patched_runtime(
                    ops, a5=True
                )
                with device_patch, generation_patch, ops_patch:
                    output = fp8_grouped_matmul_npu(
                        x,
                        weight,
                        scale,
                        group_list,
                        group_list_type=group_list_type,
                    )

                self.assertEqual(output.shape, (3, 256))
                call = ops.grouped_calls[0]
                torch.testing.assert_close(
                    call["group_list"], group_list.to(torch.int64)
                )
                self.assertEqual(call["group_list_type"], group_list_type)

    def test_grouped_pre_a5_converts_counts_to_cumulative_offsets(self):
        ops = _FakeNPUOps()
        x = torch.zeros((3, 128), dtype=torch.bfloat16)
        weight = torch.empty((2, 128, 256), dtype=torch.uint8)
        scale = torch.ones((2, 1, 2), dtype=torch.float32)
        group_list = torch.tensor([2, 1])
        device_patch, generation_patch, ops_patch = self._patched_runtime(ops, a5=False)
        with device_patch, generation_patch, ops_patch:
            fp8_grouped_matmul_npu(x, weight, scale, group_list, group_list_type=1)

        torch.testing.assert_close(ops.soft_calls[0][3], torch.tensor([2, 3]))

    def test_real_device_guard_rejects_cpu_execution(self):
        with self.assertRaisesRegex(RuntimeError, "resident on an Ascend NPU"):
            fp8_matmul_npu(
                torch.zeros((1, 128), dtype=torch.bfloat16),
                torch.empty((128, 128), dtype=torch.float8_e4m3fn),
                [128, 128],
                torch.ones((1, 1), dtype=torch.float32),
            )


class TestNPUBlockFP8Integration(CustomTestCase):
    def test_compressed_tensors_scheme_does_not_probe_cuda_on_npu(self):
        weight_quant = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy=QuantizationStrategy.BLOCK,
            symmetric=True,
            dynamic=False,
            block_structure=[128, 128],
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        )
        config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="float-quantized",
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )
        with (
            patch.object(compressed_tensors, "_is_npu", True),
            patch.object(
                config,
                "_check_scheme_supported",
                side_effect=AssertionError("CUDA capability must not be queried"),
            ),
        ):
            scheme = config._get_scheme_from_parts(
                weight_quant=weight_quant,
                input_quant=input_quant,
                format="float-quantized",
            )

        self.assertIsInstance(scheme, CompressedTensorsW8A8Fp8)

    def test_dispatch_selects_npu_kernel_before_cuda_backends(self):
        with patch.object(fp8_utils, "_is_npu", True):
            self.assertIs(fp8_utils.dispatch_w8a8_block_fp8_linear(), fp8_matmul_npu)

    def test_moe_kernel_forwards_runner_payload(self):
        kernel = NPUBlockFP8MoEMethod("w13")
        quant_info = SimpleNamespace(
            w13_weight=torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn),
            w13_weight_scale=torch.ones((2, 1, 2), dtype=torch.float32),
            w13_weight_bias=None,
            w13_scale_bias=None,
        )
        expected = torch.zeros((3, 256), dtype=torch.bfloat16)
        with patch(
            "sglang.srt.hardware_backend.npu.quantization.moe_methods."
            "fp8_grouped_matmul_npu",
            return_value=expected,
        ) as grouped:
            actual = kernel.apply(
                quant_info,
                torch.zeros((3, 128), dtype=torch.bfloat16),
                torch.tensor([2, 1]),
                pertoken_scale=None,
                output_dtype=torch.bfloat16,
                weight_prefix="w13",
                group_list_type=1,
            )

        self.assertIs(actual, expected)
        self.assertEqual(grouped.call_args.kwargs["group_list_type"], 1)
        self.assertIsNone(grouped.call_args.kwargs["input_scale"])

    def test_fp8_moe_auto_backend_builds_modular_ascend_kernels(self):
        config = SimpleNamespace(
            use_mxfp8=False,
            weight_block_size=[128, 128],
            is_fp4_experts=False,
            dequant_fp4_to_fp8=False,
        )
        layer = SimpleNamespace()
        runner_config = SimpleNamespace(layer=None)
        fake_runner = SimpleNamespace(runner_backend=MoeRunnerBackend.ASCEND)
        with (
            patch.object(fp8, "_is_npu", True),
            patch.object(
                fp8, "get_moe_runner_backend", return_value=MoeRunnerBackend.AUTO
            ),
            patch.object(fp8, "MoeRunner", return_value=fake_runner) as runner,
        ):
            method = Fp8MoEMethod(config)
            method.create_moe_runner(layer, runner_config)

        self.assertIsInstance(layer.w13_kernel, NPUBlockFP8MoEMethod)
        self.assertIsInstance(layer.w2_kernel, NPUBlockFP8MoEMethod)
        self.assertIs(runner_config.layer, layer)
        runner.assert_called_once_with(MoeRunnerBackend.ASCEND, runner_config)

    def test_deepep_normal_and_ll_payloads_share_ascend_runner_contract(self):
        hidden_states = torch.zeros((3, 128), dtype=torch.bfloat16)
        topk_ids = torch.zeros((1, 2), dtype=torch.int64)
        topk_weights = torch.ones((1, 2), dtype=torch.float32)

        normal = DeepEPNormalDispatchOutput(
            hidden_states,
            None,
            topk_ids,
            topk_weights,
            [2, 1],
        )
        normal_input = pre_permute_deepep_normal_to_ascend(normal, None, None, {})
        torch.testing.assert_close(normal_input.expert_tokens, torch.tensor([2, 1]))
        self.assertIsNone(normal_input.hidden_states_scale)
        self.assertEqual(normal_input.group_list_type, 1)

        low_latency = DeepEPLLDispatchOutput(
            hidden_states,
            None,
            topk_ids,
            topk_weights,
            torch.tensor([1, 2], dtype=torch.int32),
            3,
        )
        ll_input = pre_permute_deepep_ll_to_ascend(low_latency, None, None, {})
        torch.testing.assert_close(ll_input.expert_tokens, torch.tensor([1, 2]))
        self.assertIsNone(ll_input.hidden_states_scale)
        self.assertEqual(ll_input.group_list_type, 1)


if __name__ == "__main__":
    unittest.main()
