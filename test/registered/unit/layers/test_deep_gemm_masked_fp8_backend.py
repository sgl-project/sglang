import sys
import types
import unittest
from unittest.mock import Mock, patch

from sglang.srt.layers.deep_gemm_wrapper import entrypoint


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


if __name__ == "__main__":
    unittest.main()
