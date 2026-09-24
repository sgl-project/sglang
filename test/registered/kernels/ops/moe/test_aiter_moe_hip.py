"""The ROCm decode router gate must reproduce aiter's topk_gating bit for bit."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-kernel-test-1-gpu-amd-mi35x")


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
        from sglang.kernels.ops.moe.rocm_router_gate import (
            ROCM_ROUTER_MAX_TOKENS,
            rocm_router_gate,
            rocm_router_gemv_split_k,
            rocm_router_max_tokens,
            rocm_router_reduce_partials,
        )

        self.max_tokens = ROCM_ROUTER_MAX_TOKENS
        self.gate = rocm_router_gate
        self.gemv = rocm_router_gemv_split_k
        self.max_tokens_for = rocm_router_max_tokens
        self.reduce = rocm_router_reduce_partials
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.bias_bf16 = (
            torch.randn(NUM_EXPERTS, device=self.device, generator=self.gen) * 0.5
        ).to(torch.bfloat16)

    def _randn(self, *shape, scale=1.0):
        return torch.randn(*shape, device=self.device, generator=self.gen) * scale

    def _assert_same_gate(self, logits, bias, renorm=True, rsf=ROUTED_SCALING, msg=""):
        for topk in (TOPK,):
            ref_w, ref_i = _aiter_gate(logits, bias, topk, renorm, rsf)
            out_w, out_i = self.gate(logits, bias, topk, renorm, rsf)
            self.assertTrue(torch.equal(ref_i, out_i), f"ids {msg} topk {topk}")
            self.assertTrue(torch.equal(ref_w, out_w), f"weights {msg} topk {topk}")

    def test_gate_matches_aiter_on_ties(self):
        zero_bias = torch.zeros(NUM_EXPERTS, device=self.device, dtype=torch.bfloat16)
        for num_tokens in (512,):
            levels = torch.randint(
                0, 8, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            self._assert_same_gate(levels - 3, self.bias_bf16, msg="8 levels")
            self._assert_same_gate(
                torch.zeros(num_tokens, NUM_EXPERTS, device=self.device),
                zero_bias,
                msg="all equal",
            )
            few = torch.full(
                (num_tokens, NUM_EXPERTS), float("-inf"), device=self.device
            )
            few[:, :3] = 1.0
            self._assert_same_gate(few, zero_bias, msg="3 finite experts")

    def test_gate_matches_aiter_non_finite(self):
        for num_tokens in (16,):
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[logits < 0] = float("-inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="-inf")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, ::7] = float("nan")
            self._assert_same_gate(logits, self.bias_bf16, msg="nan")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, 5] = float("inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="+inf")

    def test_gemv_accuracy_batch_invariance_and_repeatability(self):
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        x = self._randn(self.max_tokens, HIDDEN).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(self.max_tokens, NUM_EXPERTS, device=self.device)
        self.reduce(self.gemv(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        for num_tokens in (1, 17, 64):
            rows = x[:num_tokens]
            out = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
            self.reduce(self.gemv(rows, weight), out)
            self.assertTrue(
                torch.equal(out, full[:num_tokens]), f"batch of {num_tokens}"
            )
        # The same rows moved to other positions of the batch.
        shifted = torch.roll(x, shifts=17, dims=0)
        out_shifted = torch.empty_like(full)
        self.reduce(self.gemv(shifted, weight), out_shifted)
        self.assertTrue(torch.equal(out_shifted, torch.roll(full, 17, 0)))


if __name__ == "__main__":
    unittest.main()
