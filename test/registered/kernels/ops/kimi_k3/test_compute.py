import unittest

import torch

from sglang.kernels.ops.kimi_k3 import (
    situ_and_mul,
    situ_and_mul_masked_post_quant,
)
from sglang.kernels.ops.kimi_k3.attn_res import attn_res_fused_tma
from sglang.kernels.ops.kimi_k3.mla_output_gate import (
    covered,
    kimi_k3_mla_output_gate,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168
_GROUP_SIZE = 128
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


def _unpack_ue8m0_scales(packed, num_groups):
    num_experts, groups_per_word, num_tokens = packed.shape
    exponents = packed.contiguous().view(torch.uint8)
    exponents = exponents.view(num_experts, groups_per_word, num_tokens, 4)
    exponents = exponents.permute(0, 2, 1, 3).reshape(
        num_experts, num_tokens, num_groups
    )
    return torch.exp2(exponents.float() - 127.0)


class TestKimiK3ComputeKernels(CustomTestCase):
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

    def test_mla_output_gate(self):
        generator = torch.Generator(device="cuda").manual_seed(1)
        shape = (5, 12, 128)
        x = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

        self.assertTrue(covered(x, gate))
        expected = x * torch.sigmoid(gate).to(torch.bfloat16)
        self.assertTrue(torch.equal(kimi_k3_mla_output_gate(x, gate), expected))

    def test_situ_and_mul(self):
        generator = torch.Generator(device="cuda").manual_seed(2)
        hidden_size = 1024
        storage = torch.randn(
            (7, 2 * hidden_size + 16),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate_up = storage[:, : 2 * hidden_size]
        output = torch.empty(
            (gate_up.shape[0], hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        returned = situ_and_mul(gate_up, output, beta=_BETA, linear_beta=_LINEAR_BETA)

        self.assertIs(returned, output)
        torch.testing.assert_close(
            returned.float(),
            _situ_reference(gate_up).to(torch.bfloat16).float(),
            rtol=2e-2,
            atol=4e-2,
        )

    def test_situ_mul_quant(self):
        torch.cuda.manual_seed_all(3)
        num_experts, num_tokens, hidden_size, topk = 8, 32, 1024, 16
        gate_up = (
            torch.randn(
                num_experts,
                num_tokens,
                2 * hidden_size,
                device="cuda",
                dtype=torch.float32,
            )
            * 2.0
        ).to(torch.bfloat16)
        masked_m = torch.randint(
            0,
            num_tokens + 1,
            (num_experts,),
            device="cuda",
            dtype=torch.int32,
        )
        masked_m[0] = 0
        masked_m[-1] = num_tokens

        output = torch.full(
            (num_experts, num_tokens, hidden_size),
            0x7F,
            device="cuda",
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        num_groups = hidden_size // _GROUP_SIZE
        output_scale = torch.zeros(
            (num_experts, num_groups // 4, num_tokens),
            device="cuda",
            dtype=torch.int32,
        )

        situ_and_mul_masked_post_quant(
            input=gate_up,
            output=output,
            output_scale=output_scale,
            quant_group_size=_GROUP_SIZE,
            masked_m=masked_m,
            beta=_BETA,
            linear_beta=_LINEAR_BETA,
            scale_ue8m0=True,
            topk=topk,
            transposed=True,
        )

        scales = _unpack_ue8m0_scales(output_scale, num_groups)
        expanded_scales = scales.repeat_interleave(_GROUP_SIZE, dim=-1)
        dequantized = output.float() * expanded_scales
        expected = _situ_reference(gate_up)
        error_bound = expanded_scales * 17.0
        raw_output = output.view(torch.uint8)
        for expert in range(num_experts):
            valid_tokens = int(masked_m[expert].item())
            self.assertTrue(
                bool(
                    (
                        (
                            dequantized[expert, :valid_tokens]
                            - expected[expert, :valid_tokens]
                        ).abs()
                        <= error_bound[expert, :valid_tokens]
                    ).all()
                )
            )
            self.assertTrue(bool((raw_output[expert, valid_tokens:] == 0x7F).all()))


if __name__ == "__main__":
    unittest.main()
