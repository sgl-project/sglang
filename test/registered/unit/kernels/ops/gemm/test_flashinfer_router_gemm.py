"""CPU-only tests for the FlashInfer router GEMM adapter."""

import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

router_gemm = importlib.import_module("sglang.kernels.ops.gemm.flashinfer_router_gemm")

EXPECTED_OP_NAMES = {
    (7168, 128, torch.bfloat16): "mm_M1_16_K7168_N128",
    (7168, 256, torch.float32): "mm_M1_16_K7168_N256",
    (6144, 256, torch.float32): "mm_M1_16_K6144_N256",
    (7168, 256, torch.bfloat16): "mm_M1_16_K7168_N256_bf16",
    (7168, 384, torch.float32): "mm_M1_16_K7168_N384",
    (7168, 384, torch.bfloat16): "mm_M1_16_K7168_N384_bf16",
    (7168, 896, torch.float32): "mm_M1_16_K7168_N896",
    (7168, 896, torch.bfloat16): "mm_M1_16_K7168_N896_bf16",
}


class TestFlashInferRouterGemm(CustomTestCase):
    def setUp(self):
        super().setUp()
        router_gemm._resolve_flashinfer_router_gemm_op.cache_clear()

    def tearDown(self):
        router_gemm._resolve_flashinfer_router_gemm_op.cache_clear()
        super().tearDown()

    def _mock_flashinfer_gemm(self, *, missing=()):
        flashinfer = ModuleType("flashinfer")
        flashinfer.__path__ = []
        gemm = ModuleType("flashinfer.gemm")
        op_mocks = {}

        def make_op():
            def op(_hidden_states, _router_weights_t, output, **_kwargs):
                output.fill_(3)

            return MagicMock(side_effect=op)

        for op_name in EXPECTED_OP_NAMES.values():
            if op_name not in missing:
                op_mocks[op_name] = make_op()
                setattr(gemm, op_name, op_mocks[op_name])

        flashinfer.gemm = gemm
        return (
            patch.dict(
                sys.modules,
                {"flashinfer": flashinfer, "flashinfer.gemm": gemm},
            ),
            op_mocks,
        )

    def test_api_name_mapping(self):
        self.assertEqual(router_gemm._ROUTER_GEMM_OP_NAMES, EXPECTED_OP_NAMES)

    def test_supports_only_fixed_shapes_token_range_and_arches(self):
        module_patch, _ = self._mock_flashinfer_gemm()
        with module_patch:
            for hidden_dim, num_experts, out_dtype in EXPECTED_OP_NAMES:
                for num_tokens in (1, 16):
                    for device_sm in (90, 100, 103, 107):
                        with self.subTest(
                            hidden_dim=hidden_dim,
                            num_experts=num_experts,
                            out_dtype=out_dtype,
                            num_tokens=num_tokens,
                            device_sm=device_sm,
                        ):
                            self.assertTrue(
                                router_gemm.is_flashinfer_router_gemm_supported(
                                    num_tokens,
                                    hidden_dim,
                                    num_experts,
                                    out_dtype,
                                    device_sm,
                                )
                            )

            unsupported = [
                (0, 7168, 256, torch.float32, 90),
                (17, 7168, 256, torch.float32, 90),
                (1, 5120, 256, torch.float32, 90),
                (1, 7168, 257, torch.float32, 90),
                (1, 7168, 256, torch.float16, 90),
                (1, 7168, 256, torch.float32, 89),
                (1, 7168, 256, torch.float32, 120),
            ]
            for case in unsupported:
                with self.subTest(case=case):
                    self.assertFalse(
                        router_gemm.is_flashinfer_router_gemm_supported(*case)
                    )

    def test_missing_flashinfer_api_is_not_supported(self):
        op_name = EXPECTED_OP_NAMES[(7168, 384, torch.float32)]
        module_patch, _ = self._mock_flashinfer_gemm(missing={op_name})
        with module_patch:
            self.assertFalse(
                router_gemm.is_flashinfer_router_gemm_supported(
                    1, 7168, 384, torch.float32, 90
                )
            )

    def test_missing_flashinfer_package_is_not_supported(self):
        with patch.dict(
            sys.modules,
            {"flashinfer": None, "flashinfer.gemm": None},
        ):
            self.assertFalse(
                router_gemm.is_flashinfer_router_gemm_supported(
                    1, 7168, 384, torch.float32, 90
                )
            )

    def test_adapter_allocates_output_and_transposes_router_weights(self):
        hidden_states = torch.empty((2, 7168), dtype=torch.bfloat16)
        router_weights = torch.empty((384, 7168), dtype=torch.bfloat16)
        op_name = EXPECTED_OP_NAMES[(7168, 384, torch.bfloat16)]
        module_patch, op_mocks = self._mock_flashinfer_gemm()

        with (
            module_patch,
            patch.object(router_gemm, "is_arch_support_pdl", return_value=True),
        ):
            output = router_gemm.flashinfer_router_gemm(
                hidden_states,
                router_weights,
                out_dtype=torch.bfloat16,
            )

        self.assertEqual(output.shape, (2, 384))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.all(output == 3))
        op_mocks[op_name].assert_called_once()
        args = op_mocks[op_name].call_args.args
        kwargs = op_mocks[op_name].call_args.kwargs
        self.assertIs(args[0], hidden_states)
        self.assertEqual(args[1].shape, (7168, 384))
        self.assertEqual(args[1].stride(), (1, 7168))
        self.assertEqual(args[1].data_ptr(), router_weights.data_ptr())
        self.assertIs(args[2], output)
        self.assertEqual(kwargs, {"launch_with_pdl": True})

    def test_preallocated_output_dtype_selects_flashinfer_op(self):
        hidden_states = torch.empty((2, 7168), dtype=torch.bfloat16)
        router_weights = torch.empty((384, 7168), dtype=torch.bfloat16)
        output = torch.empty((2, 384), dtype=torch.float32)
        fp32_op_name = EXPECTED_OP_NAMES[(7168, 384, torch.float32)]
        bf16_op_name = EXPECTED_OP_NAMES[(7168, 384, torch.bfloat16)]
        module_patch, op_mocks = self._mock_flashinfer_gemm()

        with (
            module_patch,
            patch.object(router_gemm, "is_arch_support_pdl", return_value=False),
        ):
            result = router_gemm.flashinfer_router_gemm(
                hidden_states,
                router_weights,
                out_dtype=torch.bfloat16,
                output=output,
            )

        self.assertIs(result, output)
        self.assertEqual(result.dtype, torch.float32)
        op_mocks[fp32_op_name].assert_called_once()
        op_mocks[bf16_op_name].assert_not_called()
        self.assertEqual(
            op_mocks[fp32_op_name].call_args.kwargs,
            {"launch_with_pdl": False},
        )


if __name__ == "__main__":
    unittest.main()
