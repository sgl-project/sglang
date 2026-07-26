import unittest

import torch

from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA JIT")
class TestDsv4SiluAndMulValidRows(unittest.TestCase):
    def test_device_valid_rows_skip_unused_capacity(self):
        rows = 32
        valid = 7
        intermediate = 256
        gate_up = torch.randn(
            (rows, intermediate * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference = torch.empty(
            (rows, intermediate),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        reference_scale = torch.empty(
            (rows, intermediate // 128),
            dtype=torch.float32,
            device="cuda",
        )
        silu_and_mul_contig_post_quant(
            input=gate_up,
            output=reference,
            output_scale=reference_scale,
            quant_group_size=128,
        )

        output = torch.zeros_like(reference)
        output_scale = torch.zeros_like(reference_scale)
        valid_rows = torch.tensor([valid], dtype=torch.int32, device="cuda")
        silu_and_mul_contig_post_quant(
            input=gate_up,
            output=output,
            output_scale=output_scale,
            quant_group_size=128,
            valid_rows=valid_rows,
        )
        torch.testing.assert_close(
            output[:valid].view(torch.uint8),
            reference[:valid].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output_scale[:valid],
            reference_scale[:valid],
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(output[valid:].view(torch.uint8) == 0))
        self.assertTrue(torch.all(output_scale[valid:] == 0))


if __name__ == "__main__":
    unittest.main()
