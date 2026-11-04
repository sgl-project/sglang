import unittest

import torch

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    y = x * (127 / absmax)
    # libdevice round is half away from zero, unlike torch.round.
    x_q = (torch.sign(y) * torch.floor(y.abs() + 0.5)).to(torch.int8)
    return x_q, absmax / 127, x.sum(dim=-1)


class TestPerTokenQuantInt8(CustomTestCase):
    def test_matches_reference(self):
        device = get_device()
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((1, 7), (33, 4096), (4, 5, 1000)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = torch.randn(shape, dtype=dtype, device=device) * 3
                    x_q, scale, x_sum = per_token_quant_int8(x, cal_sum=True)
                    ref_q, ref_scale, ref_sum = _reference(x)
                    self.assertEqual(x_q.dtype, torch.int8)
                    self.assertLessEqual(
                        (x_q.int() - ref_q.int()).abs().max().item(), 1
                    )
                    torch.testing.assert_close(scale, ref_scale)
                    torch.testing.assert_close(
                        x_sum.float(), ref_sum, rtol=1e-2, atol=1e-2
                    )

    def test_rounds_half_away_from_zero(self):
        # absmax 127 makes scale 1, so the kernel rounds these values unchanged.
        x = torch.tensor([[127.0, 0.5, -0.5, 1.5, -1.5, 2.5]], device=get_device())
        x_q, _ = per_token_quant_int8(x)
        self.assertEqual(x_q.tolist(), [[127, 1, -1, 2, -2, 3]])


if __name__ == "__main__":
    unittest.main()
