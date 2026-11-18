"""
Unit tests for the fused sigmoid-gate top-k + logsigmoid renorm.

The renorm has to stay in the log domain: normalizing sigmoid(x) by its own
sum is 0/0 -- NaN for the whole token -- as soon as every active logit
underflows the fp32 sigmoid. Regression test for sgl-project/sglang#51102.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import sigmoid_gate_topk_renorm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=20, suite="base-a-jit-kernel-unit-test-amd")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _reference(logits, k, n_shared, bias, route_scale, global_scale):
    """fp64 reference: sigmoid+bias top-k, then logsigmoid renorm over k++shared."""
    n_routed = logits.shape[1] - n_shared
    routed = logits[:, :n_routed].double()
    ids = torch.topk(torch.sigmoid(routed) + bias.double(), k, dim=-1).indices
    active = routed.gather(-1, ids)
    if n_shared:
        active = torch.cat([active, logits[:, n_routed:].double()], dim=-1)
    log_probs = F.logsigmoid(active)
    weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))
    return ids, weights * (route_scale * global_scale.double())


class TestSigmoidGateTopKRenorm(CustomTestCase):
    def _check(self, logits, k, n_shared, dtype=torch.float32):
        logits = logits.to(_DEVICE, dtype)
        n_routed = logits.shape[1] - n_shared
        bias = torch.randn(n_routed, device=_DEVICE, dtype=torch.float32) * 0.1
        global_scale = torch.tensor([0.7], device=_DEVICE)
        route_scale = 1.3

        routed_w, ids, shared_w, _ = sigmoid_gate_topk_renorm(
            logits, k, n_shared, route_scale, global_scale, bias
        )
        ref_ids, ref_w = _reference(
            logits, k, n_shared, bias, route_scale, global_scale
        )

        observed_w = torch.cat([routed_w, shared_w], dim=-1)
        self.assertEqual(
            sorted(ids.tolist()), sorted(ref_ids.tolist()), "selected experts differ"
        )
        # torch.topk already returns the experts in descending score order, which is
        # the order the kernel stores them in, so the weights line up as they are.
        atol, rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-6, 1e-5)
        torch.testing.assert_close(
            observed_w.float(), ref_w.float(), atol=atol, rtol=rtol
        )
        total = observed_w.float().sum(-1)
        torch.testing.assert_close(
            total,
            torch.full_like(total, route_scale * 0.7),
            atol=atol,
            rtol=rtol,
        )

    def test_matches_reference(self):
        cases = [
            (1, 258, 6, 2),
            (7, 258, 6, 2),
            (64, 258, 6, 2),
            (3, 64, 4, 0),
            (33, 40, 6, 2),
        ]
        for tokens, experts, k, n_shared in cases:
            for scale in (1.0, 30.0):
                with self.subTest(tokens=tokens, experts=experts, k=k, scale=scale):
                    logits = torch.randn(tokens, experts) * scale
                    self._check(logits, k, n_shared)

    def test_matches_reference_bf16_logits(self):
        self._check(torch.randn(5, 258) * 10.0, 6, 2, dtype=torch.bfloat16)

    def test_no_nan_when_every_active_logit_underflows(self):
        """sgl-project/sglang#51102: 0/0 renorm turned the whole token into NaN."""
        for value in (-100.0, -300.0):
            with self.subTest(value=value):
                self._check(torch.full((2, 258), value), 6, 2)
                routed_w, _, shared_w, _ = sigmoid_gate_topk_renorm(
                    torch.full((2, 258), value, device=_DEVICE),
                    6,
                    2,
                    1.0,
                    torch.ones(1, device=_DEVICE),
                    torch.zeros(256, device=_DEVICE),
                )
                weights = torch.cat([routed_w, shared_w], dim=-1)
                self.assertTrue(torch.isfinite(weights).all(), msg=str(weights))
                torch.testing.assert_close(
                    weights, torch.full_like(weights, 1.0 / 8.0), atol=1e-6, rtol=1e-6
                )

    def test_underflow_in_one_row_does_not_leak_into_another(self):
        logits = torch.randn(2, 258)
        logits[0] = -300.0
        self._check(logits, 6, 2)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "the fused gate JIT kernel is CUDA-only",
)
class TestInklingGateTopKRenormV2(CustomTestCase):
    """The production CUDA JIT path shares the epilogue the Triton kernel had."""

    def test_jit_renorm_underflow(self):
        try:
            from sglang.kernels.ops.moe.inkling_gate_topk_renorm import (
                inkling_gate_topk_renorm_v2,
            )
        except Exception as error:  # the JIT build is optional outside CI images
            self.skipTest(f"fused gate JIT unavailable: {error}")

        for value in (-100.0, -300.0):
            with self.subTest(value=value):
                # The JIT gate kernel reads 32B-aligned rows, so feed it the
                # production [tokens, 264]-padded view sliced to [:, :258].
                padded = torch.full((2, 264), value, device="cuda")
                logits = padded[:, :258]
                bias = torch.zeros(256, device="cuda")
                global_scale = torch.ones(1, device="cuda")
                routed_w, _, shared_w, _ = inkling_gate_topk_renorm_v2(
                    logits, bias, global_scale, 1.0
                )
                weights = torch.cat([routed_w, shared_w], dim=-1)
                self.assertTrue(torch.isfinite(weights).all(), msg=str(weights))
                torch.testing.assert_close(
                    weights, torch.full_like(weights, 1.0 / 8.0), atol=1e-5, rtol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
