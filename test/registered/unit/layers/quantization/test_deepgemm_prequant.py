import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepGemmPrequant(unittest.TestCase):
    def _run_prequantized_input(self, scale_ue8m0):
        q_input = torch.zeros((3, 128), dtype=fp8_dtype)
        scale_dtype = torch.int32 if scale_ue8m0 else torch.float32
        input_scale = torch.empty((1, 4), dtype=scale_dtype).transpose(0, 1)[:3]
        input_scale.fill_(1)
        weight = torch.zeros((64, 128), dtype=fp8_dtype)
        weight_scale = torch.ones((1, 1), dtype=torch.float32)
        matmul_output = torch.zeros((3, 64), dtype=torch.bfloat16)

        with (
            patch.object(
                fp8_utils.deep_gemm_wrapper,
                "DEEPGEMM_SCALE_UE8M0",
                scale_ue8m0,
            ),
            patch.object(fp8_utils, "sglang_per_token_group_quant_fp8") as quantizer,
            patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_deepgemm",
                return_value=matmul_output,
            ) as matmul,
        ):
            output = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback(
                input=q_input,
                weight=weight,
                block_size=[128, 128],
                weight_scale=weight_scale,
                input_scale=input_scale,
            )

        quantizer.assert_not_called()
        matmul.assert_called_once()
        args = matmul.call_args.args
        self.assertEqual(args[0].data_ptr(), q_input.data_ptr())
        self.assertIs(args[1], weight)
        self.assertIs(args[2], input_scale)
        self.assertIs(args[3], weight_scale)
        self.assertEqual(args[4], [128, 128])
        self.assertEqual(matmul.call_args.kwargs["output_dtype"], torch.bfloat16)
        self.assertEqual(output.shape, (3, 64))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_prequantized_fp32_scale_input_skips_quantizer(self):
        self._run_prequantized_input(scale_ue8m0=False)

    def test_prequantized_ue8m0_scale_input_skips_quantizer(self):
        self._run_prequantized_input(scale_ue8m0=True)


if __name__ == "__main__":
    unittest.main(verbosity=3)
