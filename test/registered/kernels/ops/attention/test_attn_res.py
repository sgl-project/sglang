import unittest

import torch

from sglang.kernels.ops.attention.attn_res import attn_res_fused_tma
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168


class TestAttentionResidual(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_attn_residual_and_prefix_write(self):
        generator = torch.Generator(device="cuda").manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device="cuda")

        num_tokens, num_bank_rows, num_valid_bank_rows = 5, 8, 5
        prefix = randn(num_tokens, _HIDDEN_SIZE).to(torch.bfloat16)
        bank = randn(num_tokens, num_bank_rows, _HIDDEN_SIZE).to(torch.bfloat16)
        combine_weight = (randn(_HIDDEN_SIZE) * _HIDDEN_SIZE**-0.5).to(torch.bfloat16)
        output_weight = (1 + 0.1 * randn(_HIDDEN_SIZE)).to(torch.bfloat16)
        output = torch.empty_like(prefix)

        rows = torch.cat(
            [
                bank[:, :num_valid_bank_rows].float(),
                prefix.unsqueeze(1).float(),
            ],
            dim=1,
        )
        rms = torch.rsqrt(rows.square().mean(-1) + 1e-6)
        scores = (rows * combine_weight.float()).sum(-1) * rms
        mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows).sum(1)
        expected = (
            mixed
            * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
            * output_weight.float()
        )

        attn_res_fused_tma(
            prefix,
            bank,
            combine_weight,
            output_weight,
            output,
            num_valid_bank_rows,
            1e-6,
            write_prefix=True,
        )

        torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=4e-2)
        self.assertTrue(torch.equal(bank[:, num_valid_bank_rows], prefix))


if __name__ == "__main__":
    unittest.main()
