import unittest

import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TestCPUQuantOps(CustomTestCase):
    def test_per_token_quant_int8_python_dispatch_cpu(self):
        x = torch.randn(4, 33, dtype=torch.bfloat16)

        x_q, scales, x_sum = per_token_quant_int8(x, cal_sum=True)
        ref_q, ref_scales = torch.ops.sgl_kernel.per_token_quant_int8_cpu(
            x.contiguous()
        )

        torch.testing.assert_close(x_q, ref_q)
        torch.testing.assert_close(scales, ref_scales)
        torch.testing.assert_close(x_sum, x.sum(dim=-1))


if __name__ == "__main__":
    unittest.main()
