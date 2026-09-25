"""The ROCm decode router gate must reproduce aiter's topk_gating bit for bit."""

import unittest

import torch

from sglang.kernels.ops.moe.rocm_router_gate import (
    ROCM_ROUTER_MAX_TOKENS,
    rocm_router_gate,
    rocm_router_gemv_split_k,
    rocm_router_reduce_partials,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


try:
    from aiter import topk_gating as aiter_topk_gating
except ImportError:  # aiter is absent off ROCm
    aiter_topk_gating = None


NUM_EXPERTS = 384
HIDDEN = 5120
TOPK = 6
ROUTED_SCALING = 1.5


def _aiter_gate(logits, bias, topk, renorm, rsf):
    weights = torch.empty(
        logits.shape[0], topk, dtype=torch.float32, device=logits.device
    )
    ids = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=logits.device)
    aiter_topk_gating(
        weights, ids, logits, bias, renorm, rsf, score_func="sqrtsoftplus"
    )
    return weights, ids


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and aiter_topk_gating is not None,
    "ROCm gfx95 + aiter only",
)
class TestRocmRouterGate(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.bias_bf16 = (
            torch.randn(NUM_EXPERTS, device=self.device, generator=self.gen) * 0.5
        ).to(torch.bfloat16)

    def _randn(self, *shape, scale=1.0):
        return torch.randn(*shape, device=self.device, generator=self.gen) * scale

    def _assert_same_gate(self, logits, bias, renorm=True, rsf=ROUTED_SCALING, msg=""):
        ref_w, ref_i = _aiter_gate(logits, bias, TOPK, renorm, rsf)
        out_w, out_i = rocm_router_gate(logits, bias, TOPK, renorm, rsf)
        self.assertTrue(torch.equal(ref_i, out_i), f"ids {msg}")
        self.assertTrue(torch.equal(ref_w, out_w), f"weights {msg}")

    def test_gate_matches_aiter_on_ties_and_non_finite_logits(self):
        """Tie order and the NaN / +-inf handling are aiter topk_gating's, bit for bit."""
        zero_bias = torch.zeros(NUM_EXPERTS, device=self.device, dtype=torch.bfloat16)
        levels = torch.randint(
            0, 8, (512, NUM_EXPERTS), device=self.device, generator=self.gen
        ).float()
        self._assert_same_gate(levels - 3, self.bias_bf16, msg="8 levels")
        self._assert_same_gate(
            torch.zeros(16, NUM_EXPERTS, device=self.device), zero_bias, msg="all equal"
        )
        for bad in (float("-inf"), float("nan"), float("inf")):
            logits = self._randn(16, NUM_EXPERTS, scale=3.0)
            logits[:, ::7] = bad
            self._assert_same_gate(logits, self.bias_bf16, msg=str(bad))

    def test_fused_split_k_gate_matches_reduce_then_gate(self):
        """The decode router passes the GEMV partials straight to the gate: its in-launch
        reduce must write the same logits as rocm_router_reduce_partials, and gate them
        to the same weights and ids."""
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        x = self._randn(ROCM_ROUTER_MAX_TOKENS, HIDDEN).to(torch.bfloat16)
        partials = rocm_router_gemv_split_k(x, weight)
        ref_logits = torch.empty(
            ROCM_ROUTER_MAX_TOKENS, NUM_EXPERTS, device=self.device
        )
        rocm_router_reduce_partials(partials, ref_logits)
        ref_w, ref_i = rocm_router_gate(
            ref_logits, self.bias_bf16, TOPK, True, ROUTED_SCALING
        )
        logits = torch.full_like(ref_logits, float("nan"))
        out_w, out_i = rocm_router_gate(
            logits, self.bias_bf16, TOPK, True, ROUTED_SCALING, partials=partials
        )
        self.assertTrue(torch.equal(logits, ref_logits))
        self.assertTrue(torch.equal(out_i, ref_i))
        self.assertTrue(torch.equal(out_w, ref_w))

    def test_gemv_accuracy_and_batch_invariance(self):
        """The split-K GEMV is within fp32 rounding of fp64, and every M runs the same
        16-row tile, so a row's result does not depend on the batch."""
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        x = self._randn(ROCM_ROUTER_MAX_TOKENS, HIDDEN).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(ROCM_ROUTER_MAX_TOKENS, NUM_EXPERTS, device=self.device)
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        part = torch.empty(17, NUM_EXPERTS, device=self.device)
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x[:17], weight), part)
        self.assertTrue(torch.equal(part, full[:17]))


if __name__ == "__main__":
    unittest.main()
