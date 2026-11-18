"""The Grok router softcap c * tanh(x / c) must stay finite for large finite logits,
in both the CUDA-core and the tensor-core kernel."""

import unittest

import torch

from sglang.kernels.ops.moe.router import fused_moe_router_shim
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

SOFTCAP = 30.0
TOPK = 2
NUM_EXPERTS = 8
HIDDEN = 256


class TestMoeRouterSoftcap(CustomTestCase):
    def test_large_logits_stay_finite(self):
        # 1 token takes the CUDA-core kernel, 512 tokens the tensor-core kernel
        for bs in (1, 512):
            with self.subTest(bs=bs):
                x = torch.zeros(bs, HIDDEN, device="cuda")
                x[:, 0] = 1
                weight = torch.zeros(NUM_EXPERTS, HIDDEN, device="cuda")
                weight[:, 0] = torch.arange(NUM_EXPERTS, dtype=torch.float32)
                # |x| / c = 50 overflows exp(2x/c) in fp32 for both signs
                weight[0, 0] = 1500
                weight[1, 0] = -1500

                weights, ids = fused_moe_router_shim(
                    moe_softcapping=SOFTCAP,
                    hidden_states=x,
                    gating_output=weight,
                    topk=TOPK,
                    renormalize=False,
                )

                logits = x.double() @ weight.double().T
                expected = (
                    (SOFTCAP * torch.tanh(logits / SOFTCAP)).softmax(-1).topk(TOPK)
                )
                torch.testing.assert_close(
                    weights.double(), expected.values, atol=1e-6, rtol=1e-5
                )
                self.assertTrue(torch.equal(ids.long(), expected.indices))


if __name__ == "__main__":
    unittest.main()
