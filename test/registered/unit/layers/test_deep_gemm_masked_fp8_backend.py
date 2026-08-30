"""CPU coverage for the selectable masked DeepGEMM FP8 backend."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.quantization.per_token_group_quant import (
    _allocate_outputs,
    _infer_scale_layout,
)
from sglang.srt.layers.deep_gemm_wrapper import entrypoint
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_runner


class _FakeTensor:
    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self


class TestDeepGemmMaskedFp8Backend(unittest.TestCase):
    def setUp(self):
        self.lhs = (
            _FakeTensor((256, 256, 7168)),
            _FakeTensor((256, 256, 56), torch.int32),
        )
        self.rhs = (
            _FakeTensor((256, 4096, 7168)),
            _FakeTensor((256, 32, 56), torch.int32),
        )
        self.lhs_api_scale = _FakeTensor((256, 256, 56), torch.float32)
        self.rhs_api_scale = _FakeTensor((256, 32, 56), torch.float32)
        self.out = _FakeTensor((256, 256, 4096))
        self.masked_m = _FakeTensor((256,))

    def _run(self, backend):
        run = Mock(return_value=self.out)
        flashinfer = types.ModuleType("flashinfer")
        flashinfer_gemm = types.ModuleType("flashinfer.gemm")
        flashinfer_gemm.batch_deepgemm_fp8_nt_groupwise = run
        flashinfer.gemm = flashinfer_gemm
        unpack = Mock(side_effect=[self.lhs_api_scale, self.rhs_api_scale])
        with (
            patch.object(entrypoint, "DEEPGEMM_MASKED_FP8_BACKEND", backend),
            patch.object(entrypoint, "_sanity_check_input"),
            patch.object(entrypoint, "_ensure_cuda", side_effect=lambda pair: pair),
            patch.object(entrypoint, "_unpack_packed_ue8m0_scale", unpack),
            patch.dict(
                sys.modules,
                {"flashinfer": flashinfer, "flashinfer.gemm": flashinfer_gemm},
            ),
        ):
            result = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                self.lhs,
                self.rhs,
                self.out,
                self.masked_m,
                expected_m=1,
            )
        self.assertIs(result, self.out)
        expected_lhs_scale = self.lhs[1] if backend == "cake" else self.lhs_api_scale
        expected_rhs_scale = self.rhs[1] if backend == "cake" else self.rhs_api_scale
        run.assert_called_once_with(
            self.lhs[0],
            self.rhs[0],
            expected_lhs_scale,
            expected_rhs_scale,
            self.masked_m,
            1,
            out=self.out,
            backend="deepgemm" if backend == "flashinfer" else "cake",
        )
        if backend == "cake":
            unpack.assert_not_called()
        else:
            self.assertEqual(unpack.call_count, 2)

    def test_flashinfer_reference_route(self):
        self._run("flashinfer")

    def test_cake_route(self):
        self._run("cake")

    def test_non_native_backend_rejects_recipe_and_overlap(self):
        with (
            patch.object(entrypoint, "DEEPGEMM_MASKED_FP8_BACKEND", "cake"),
            patch.object(entrypoint, "_sanity_check_input"),
            patch.object(entrypoint, "_ensure_cuda", side_effect=lambda pair: pair),
        ):
            with self.assertRaisesRegex(ValueError, "FP4/MXFP8 recipes"):
                entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                    self.lhs,
                    self.rhs,
                    self.out,
                    self.masked_m,
                    expected_m=1,
                    recipe_a=(1, 32),
                )
            with self.assertRaisesRegex(ValueError, "GEMM overlap"):
                entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                    self.lhs,
                    self.rhs,
                    self.out,
                    self.masked_m,
                    expected_m=1,
                    overlap_args=object(),
                )

    def test_packed_backend_keeps_packed_activation_quantization(self):
        gateup = torch.empty((2, 3, 16), dtype=torch.bfloat16)
        masked_m = torch.tensor([1, 2], dtype=torch.int32)
        expected = (object(), object())
        with (
            patch.object(
                deep_gemm_runner.deep_gemm_wrapper,
                "DEEPGEMM_SCALE_UE8M0",
                True,
            ),
            patch.object(
                deep_gemm_runner.deep_gemm_wrapper,
                "DEEPGEMM_MASKED_FP8_PACKED_SCALES",
                True,
            ),
            patch.object(
                deep_gemm_runner,
                "per_token_group_quant",
                return_value=expected,
            ) as quant,
        ):
            result = deep_gemm_runner._varlen_deep_gemm_silu_mul_quant(
                gateup,
                masked_m,
                group_size=4,
                topk=1,
                num_real_tokens=2,
            )

        self.assertIs(result, expected)
        self.assertTrue(quant.call_args.kwargs["scale_ue8m0"])
        self.assertTrue(quant.call_args.kwargs["fuse_silu_and_mul"])
        self.assertIs(quant.call_args.kwargs["masked_m"], masked_m)
        self.assertTrue(quant.call_args.kwargs["column_major_scales"])

    def test_batch_backend_writes_row_major_float_ue8m0_directly(self):
        gateup = torch.empty((2, 3, 16), dtype=torch.bfloat16)
        masked_m = torch.tensor([1, 2], dtype=torch.int32)
        expected = (object(), object())
        with (
            patch.object(
                deep_gemm_runner.deep_gemm_wrapper,
                "DEEPGEMM_SCALE_UE8M0",
                True,
            ),
            patch.object(
                deep_gemm_runner.deep_gemm_wrapper,
                "DEEPGEMM_MASKED_FP8_PACKED_SCALES",
                False,
            ),
            patch.object(
                deep_gemm_runner,
                "per_token_group_quant",
                return_value=expected,
            ) as quant,
        ):
            result = deep_gemm_runner._varlen_deep_gemm_silu_mul_quant(
                gateup,
                masked_m,
                group_size=4,
                topk=1,
                num_real_tokens=2,
            )

        self.assertIs(result, expected)
        self.assertTrue(quant.call_args.kwargs["scale_ue8m0"])
        self.assertTrue(quant.call_args.kwargs["fuse_silu_and_mul"])
        self.assertFalse(quant.call_args.kwargs["column_major_scales"])
        self.assertTrue(quant.call_args.kwargs["unpacked_ue8m0_scales"])
        self.assertIs(quant.call_args.kwargs["masked_m"], masked_m)

    def test_unpack_packed_ue8m0_activation_scale_is_lossless(self):
        exponents = torch.tensor(
            [[[[120, 121, 122, 123], [124, 125, 126, 127]]]],
            dtype=torch.uint8,
        ).reshape(1, 1, 8)
        packed = exponents.view(torch.int32)

        unpacked = entrypoint._unpack_packed_ue8m0_scale(packed, collapse_mn=False)

        expected = (exponents.to(torch.int32) << 23).view(torch.float32)
        torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)
        self.assertEqual(unpacked.shape, (1, 1, 8))
        self.assertTrue(unpacked.is_contiguous())

    def test_unpack_packed_ue8m0_weight_scale_collapses_repeated_rows(self):
        first = torch.tensor([120, 121, 122, 123], dtype=torch.uint8)
        second = torch.tensor([124, 125, 126, 127], dtype=torch.uint8)
        exponents = torch.cat((first.repeat(128, 1), second.repeat(128, 1)), dim=0)
        packed = exponents.reshape(1, 256, 4).view(torch.int32)

        unpacked = entrypoint._unpack_packed_ue8m0_scale(packed, collapse_mn=True)

        expected_exp = torch.stack((first, second)).reshape(1, 2, 4)
        expected = (expected_exp.to(torch.int32) << 23).view(torch.float32)
        torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)
        self.assertEqual(unpacked.shape, (1, 2, 4))
        self.assertTrue(unpacked.is_contiguous())

    def test_unpacked_ue8m0_scale_layout_is_row_major_float32(self):
        scale = torch.empty((2, 3, 4), dtype=torch.float32)
        self.assertEqual(
            _infer_scale_layout(scale, scale_ue8m0=True, num_groups=4),
            (True, True, True),
        )

        source = torch.empty((2, 256), dtype=torch.bfloat16)
        output_q, output_s = _allocate_outputs(
            source,
            group_size=128,
            out_dtype=torch.float8_e4m3fn,
            scale_ue8m0=True,
            column_major_scales=False,
            fuse_silu_and_mul=False,
            unpacked_ue8m0_scales=True,
        )
        self.assertEqual(output_q.shape, (2, 256))
        self.assertEqual(output_s.shape, (2, 2))
        self.assertEqual(output_s.dtype, torch.float32)
        self.assertTrue(output_s.is_contiguous())

    def test_batch_backend_forces_masked_layout(self):
        with (
            patch.object(
                deep_gemm_runner.deep_gemm_wrapper,
                "DEEPGEMM_MASKED_FP8_BACKEND",
                "cake",
            ),
            deep_gemm_runner.envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("auto"),
        ):
            self.assertTrue(
                deep_gemm_runner._should_use_masked_standard_layout(None, None, None)
            )

        with (
            patch.object(
                deep_gemm_runner.deep_gemm_wrapper,
                "DEEPGEMM_MASKED_FP8_BACKEND",
                "cake",
            ),
            deep_gemm_runner.envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("compact"),
        ):
            with self.assertRaisesRegex(ValueError, "require.*masked"):
                deep_gemm_runner._should_use_masked_standard_layout(None, None, None)


if __name__ == "__main__":
    unittest.main()
