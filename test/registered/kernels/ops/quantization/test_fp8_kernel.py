import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    static_quant_fp8,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.fp8 import TestFP8Base

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")


_is_cuda = is_cuda()


class TestPerTokenGroupQuantFP8(TestFP8Base):
    def test_per_token_group_quant_fp8(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return

        A, A_quant_gt, scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        A_quant, scale = per_token_group_quant_fp8(
            x=A.to(torch.bfloat16), group_size=self.group_size
        )
        torch.testing.assert_close(scale, scale_gt)
        diff = (A_quant.to(torch.float16) - A_quant_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


class TestFlashinferBmmPrequantizedInput(unittest.TestCase):
    def test_prequantized_rows_are_not_quantized_twice(self):
        if not _is_cuda or torch.cuda.get_device_capability()[0] < 9:
            return

        from sglang.srt.layers.quantization.fp8_utils import (
            apply_fp8_linear_bmm_flashinfer,
        )

        generator = torch.Generator(device="cuda").manual_seed(1)
        rows, intermediate_size, hidden_size = 224, 128, 4096
        gate_up = (
            torch.randn(
                rows,
                2 * intermediate_size,
                device="cuda",
                generator=generator,
            )
            * 2
        ).to(torch.bfloat16)
        weight = (
            torch.randn(
                hidden_size, intermediate_size, device="cuda", generator=generator
            )
            * 0.05
        ).to(torch.float8_e4m3fn)
        weight_scale = torch.tensor([0.01], device="cuda")
        input_scale = torch.tensor([0.02], device="cuda")
        gate, up = gate_up.chunk(2, dim=-1)
        activation = torch.nn.functional.silu(gate) * up
        expected = apply_fp8_linear_bmm_flashinfer(
            activation, weight.t(), weight_scale, input_scale
        )
        quantized, _ = static_quant_fp8(activation, input_scale)
        actual = apply_fp8_linear_bmm_flashinfer(
            quantized,
            weight.t(),
            weight_scale,
            input_scale,
            pre_quant_output_dtype=activation.dtype,
        )

        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )


if __name__ == "__main__":
    unittest.main()
