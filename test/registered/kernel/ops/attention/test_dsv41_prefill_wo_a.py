import unittest

import torch

from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "The prefill WO-A path targets Blackwell",
)
class TestPrefillWoA(CustomTestCase):
    def test_exact_output_and_contiguous_layout(self):
        torch.manual_seed(911)
        weight = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
        for rows in (4096, 4097, 65536):
            with self.subTest(rows=rows):
                # Match the attention backend's 64 padded heads, 16 local heads.
                backing = torch.randn(
                    rows, 64, 512, device="cuda", dtype=torch.bfloat16
                )
                x = backing[:, :16].view(rows, 2, 4096)
                expected = torch.einsum("tgd,grd->tgr", x, weight)
                actual = _apply_wo_a_bf16_matmul(
                    x, weight, is_decode=False, is_prefill=True
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.flatten(1).data_ptr(), actual.data_ptr())


if __name__ == "__main__":
    unittest.main()
