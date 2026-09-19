import unittest

import torch

from sglang.kernels.ops.activation import situ_and_mul_masked
from sglang.kernels.ops.moe import (
    situ_and_mul_masked_post_quant_mxfp4,
    situ_and_mul_post_quant_mxfp4,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BETA = 4.0
_LINEAR_BETA = 25.0


def _situ_reference(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    return (
        _BETA
        * torch.tanh(gate / _BETA)
        * torch.sigmoid(gate)
        * _LINEAR_BETA
        * torch.tanh(up / _LINEAR_BETA)
    )


class TestKimiK3MaskedSituKernel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_situ_and_mul_masked(self):
        generator = torch.Generator(device="cuda").manual_seed(3)
        num_experts, num_tokens, hidden_size, topk = 8, 128, 3072, 16
        gate_up = torch.randn(
            num_experts,
            num_tokens,
            2 * hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        masked_m = torch.tensor(
            [0, 128, 64, 128, 128, 128, 128, 128],
            device="cuda",
            dtype=torch.int32,
        )
        sentinel = 42.0
        output = torch.full(
            (num_experts, num_tokens, hidden_size),
            sentinel,
            device="cuda",
            dtype=torch.bfloat16,
        )

        returned = situ_and_mul_masked(
            gate_up,
            output,
            masked_m,
            beta=_BETA,
            linear_beta=_LINEAR_BETA,
            topk=topk,
            expected_m=2,
        )
        expected = _situ_reference(gate_up).to(torch.bfloat16)

        self.assertIs(returned, output)
        for expert in range(num_experts):
            valid_tokens = int(masked_m[expert].item())
            torch.testing.assert_close(
                output[expert, :valid_tokens].float(),
                expected[expert, :valid_tokens].float(),
                rtol=2e-2,
                atol=4e-2,
            )
            self.assertTrue(bool((output[expert, valid_tokens:] == sentinel).all()))

    def test_situ_and_mul_masked_post_quant_mxfp4(self):
        generator = torch.Generator(device="cuda").manual_seed(4)
        num_experts, num_tokens, hidden_size, topk = 8, 128, 3072, 16
        gate_up = torch.randn(
            num_experts,
            num_tokens,
            2 * hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        masked_m = torch.tensor(
            [0, 128, 64, 128, 128, 128, 128, 128],
            device="cuda",
            dtype=torch.int32,
        )
        output = torch.full(
            (num_experts, num_tokens, hidden_size // 2),
            0x7F,
            device="cuda",
            dtype=torch.uint8,
        )
        output_scale = torch.full(
            (num_experts, hidden_size // 64, num_tokens),
            0x7F7F,
            device="cuda",
            dtype=torch.uint16,
        )

        situ_and_mul_masked_post_quant_mxfp4(
            gate_up,
            output,
            output_scale,
            masked_m,
            _BETA,
            _LINEAR_BETA,
            topk=topk,
            expected_m=2,
        )

        compact_input = torch.cat(
            [
                gate_up[expert, : int(masked_m[expert].item())]
                for expert in range(num_experts)
            ]
        )
        compact_output = torch.empty(
            (compact_input.shape[0], hidden_size // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        compact_scale = torch.empty(
            (hidden_size // 64, compact_input.shape[0]),
            device="cuda",
            dtype=torch.uint16,
        )
        situ_and_mul_post_quant_mxfp4(
            compact_input,
            compact_output,
            compact_scale,
            _BETA,
            _LINEAR_BETA,
        )

        compact_offset = 0
        for expert in range(num_experts):
            valid_tokens = int(masked_m[expert].item())
            self.assertTrue(
                torch.equal(
                    output[expert, :valid_tokens],
                    compact_output[compact_offset : compact_offset + valid_tokens],
                )
            )
            self.assertTrue(
                torch.equal(
                    output_scale[expert, :, :valid_tokens],
                    compact_scale[:, compact_offset : compact_offset + valid_tokens],
                )
            )
            self.assertTrue(bool((output[expert, valid_tokens:] == 0x7F).all()))
            self.assertTrue(
                bool((output_scale[expert, :, valid_tokens:] == 0x7F7F).all())
            )
            compact_offset += valid_tokens


if __name__ == "__main__":
    unittest.main()
