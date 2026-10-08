import unittest

import torch

from sglang.kernels.ops.moe import situ_and_mul_masked_post_quant
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

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


class TestSituAndMulQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

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
