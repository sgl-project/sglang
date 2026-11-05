import unittest

import torch

from sglang.kernels.ops.activation import situ_and_mul
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# backend-specific: the ROCm MoE front fuses routed-input casting with SiTU.
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

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


class TestSituAndMul(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.version.hip or get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

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


@unittest.skipUnless(torch.version.hip, "the cast path serves Kimi-K3.1's ROCm MoE")
class TestSituAndMulCast(CustomTestCase):
    def test_cast_rides_the_activation_launch(self):
        """The strided routed slice rounds exactly while SiTU keeps its output."""
        sizes = [1536, 896, 3584]  # shared gate/up, router, routed input
        # The cast begins mid-block at 1/3 tokens and spans many blocks at 257.
        for num_tokens in (1, 3, 257):
            g = torch.Generator(device="cuda").manual_seed(num_tokens)
            front = 3 * torch.randn(
                (num_tokens, sum(sizes)), generator=g, device="cuda"
            )
            gate_up, _, routed = torch.split(front, sizes, dim=-1)
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
