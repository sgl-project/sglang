import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    dispatch_w8a8_block_fp8_linear,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")


class TestFp8Utils(unittest.TestCase):

    @patch("sglang.srt.layers.quantization.fp8_utils.get_fp8_gemm_runner_backend")
    @patch("sglang.srt.layers.quantization.fp8_utils.is_sm120_supported")
    @patch(
        "sglang.srt.layers.quantization.fp8_utils._check_cutlass_block_fp8_hardware_support"
    )
    def test_auto_dispatch_sm120(
        self, mock_cutlass_support, mock_is_sm120, mock_get_backend
    ):

        mock_get_backend.return_value = Fp8GemmRunnerBackend.AUTO
        mock_is_sm120.return_value = True
        mock_cutlass_support.return_value = True

        with patch(
            "sglang.srt.layers.quantization.fp8_utils.deep_gemm_wrapper"
        ) as mock_deepgemm:

            mock_deepgemm.ENABLE_JIT_DEEPGEMM = True

            func = dispatch_w8a8_block_fp8_linear()

            self.assertEqual(
                func.__name__, "cutlass_w8a8_block_fp8_linear_with_fallback"
            )

    @patch("sglang.srt.layers.quantization.fp8_utils.get_fp8_gemm_runner_backend")
    @patch("sglang.srt.layers.quantization.fp8_utils.is_sm120_supported")
    def test_explicit_cutlass_dispatch(self, mock_is_sm120, mock_get_backend):

        mock_get_backend.return_value = Fp8GemmRunnerBackend.CUTLASS

        with patch(
            "sglang.srt.layers.quantization.fp8_utils._check_cutlass_block_fp8_hardware_support",
            return_value=True,
        ):
            func = dispatch_w8a8_block_fp8_linear()
            self.assertEqual(
                func.__name__, "cutlass_w8a8_block_fp8_linear_with_fallback"
            )


if __name__ == "__main__":
    unittest.main()
