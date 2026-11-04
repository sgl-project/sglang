"""SiTU-and-mul with the fused cast (`run_with_cast`) on ROCm: the shared experts'
activation and the routed input's fp32 -> bf16 cast off one fp32 fused-front output."""

import unittest

import torch

from sglang.kernels.ops.activation import situ_and_mul
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

_BETA, _LINEAR_BETA = 4.0, 25.0
# Kimi-K3.1 TP8 fused front: [shared gate_up | router logits | routed input]
_SIZES = [1536, 896, 3584]


def _situ_reference(gate_up):
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (
        _BETA
        * torch.tanh(gate / _BETA)
        * torch.sigmoid(gate)
        * _LINEAR_BETA
        * torch.tanh(up / _LINEAR_BETA)
    )


@unittest.skipUnless(torch.version.hip, "the cast path serves Kimi-K3.1's ROCm MoE")
class TestSituAndMulCast(CustomTestCase):
    def test_cast_rides_the_activation_launch(self):
        """Off row-strided views of the fp32 front, the activation must equal what `run`
        alone gives and the routed input must land as its exact bf16 rounding, also
        when the cast items start mid-block (1 and 3 tokens) or span many blocks."""
        for num_tokens in (1, 3, 257):
            g = torch.Generator(device="cuda").manual_seed(num_tokens)
            front = 3 * torch.randn(
                (num_tokens, sum(_SIZES)), generator=g, device="cuda"
            )
            gate_up, _, routed = torch.split(front, _SIZES, dim=-1)
            routed_bf16 = torch.full(
                routed.shape, float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            act = situ_and_mul(
                gate_up, None, _BETA, _LINEAR_BETA, (routed, routed_bf16)
            )
            msg = f"{num_tokens} tokens"
            self.assertTrue(torch.equal(routed_bf16, routed.to(torch.bfloat16)), msg)
            self.assertTrue(
                torch.equal(act, situ_and_mul(gate_up, None, _BETA, _LINEAR_BETA)),
                msg,
            )
            torch.testing.assert_close(
                act.float(),
                _situ_reference(gate_up).to(torch.bfloat16).float(),
                rtol=2e-2,
                atol=4e-2,
                msg=msg,
            )


if __name__ == "__main__":
    unittest.main()
