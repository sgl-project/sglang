import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.deep_gemm_wrapper import entrypoint
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_runner


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def is_contiguous(self):
        return True


class TestDeepGemmMaskedFp8Backend(unittest.TestCase):
    def setUp(self):
        self.lhs = (_FakeTensor((256, 256, 7168)), _FakeTensor((256, 256, 56)))
        self.rhs = (_FakeTensor((256, 4096, 7168)), _FakeTensor((256, 32, 56)))
        self.out = _FakeTensor((256, 256, 4096))
        self.masked_m = _FakeTensor((256,))

    def _run(self, backend):
        run = Mock(return_value=self.out)
        flashinfer = types.ModuleType("flashinfer")
        flashinfer_gemm = types.ModuleType("flashinfer.gemm")
        flashinfer_gemm.batch_deepgemm_fp8_nt_groupwise = run
        flashinfer.gemm = flashinfer_gemm
        with (
            patch.object(entrypoint, "DEEPGEMM_MASKED_FP8_BACKEND", backend),
            patch.object(entrypoint, "_sanity_check_input"),
            patch.object(entrypoint, "_ensure_cuda", side_effect=lambda pair: pair),
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
        run.assert_called_once_with(
            self.lhs[0],
            self.rhs[0],
            self.lhs[1],
            self.rhs[1],
            self.masked_m,
            1,
            out=self.out,
            backend="deepgemm" if backend == "flashinfer" else "cake",
        )

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

    def test_standard_scale_backend_quantizes_down_input_row_major(self):
        gateup = torch.empty((2, 3, 16), dtype=torch.bfloat16)
        masked_m = torch.tensor([1, 2], dtype=torch.int32)
        expected = (object(), object())
        with (
            patch.object(
                deep_gemm_runner.deep_gemm_wrapper,
                "DEEPGEMM_MASKED_FP8_STANDARD_SCALES",
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
        self.assertFalse(quant.call_args.kwargs["column_major_scales"])


if __name__ == "__main__":
    unittest.main()
