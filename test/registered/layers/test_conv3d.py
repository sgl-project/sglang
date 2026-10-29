import unittest

import torch
from torch import nn

from sglang.srt.layers.conv import Conv3dLayer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestConv3dBiasRounding(CustomTestCase):
    def test_linear_optimization_matches_cuda_convolution_bias_rounding(self):
        for dtype, half_ulp in (
            (torch.bfloat16, 2**-8),
            (torch.float16, 2**-11),
        ):
            with self.subTest(dtype=dtype):
                # The dot product is halfway between adjacent representable
                # values at 1.0. Adding bias inside GEMM crosses that midpoint;
                # cuDNN rounds the convolution before adding bias.
                layer = Conv3dLayer(
                    1, 16, kernel_size=(1, 1, 2), stride=(1, 1, 2), bias=True
                ).to(device="cuda", dtype=dtype)
                reference = nn.Conv3d(
                    1, 16, kernel_size=(1, 1, 2), stride=(1, 1, 2), bias=True
                ).to(device="cuda", dtype=dtype)
                with torch.no_grad():
                    layer.weight[..., 0].fill_(1.0)
                    layer.weight[..., 1].fill_(half_ulp)
                    layer.bias.fill_(half_ulp)
                    reference.load_state_dict(layer.state_dict())
                    x = torch.ones(2, 1, 1, 1, 4, device="cuda", dtype=dtype)
                    expected = reference(x)
                    torch.testing.assert_close(
                        expected, torch.ones_like(expected), rtol=0, atol=0
                    )
                    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
