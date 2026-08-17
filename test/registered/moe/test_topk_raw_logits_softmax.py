"""Regression test: the raw-logits softmax top-k must return probabilities.

`fused_topk_softmax_torch_raw_logits` feeds the flashinfer_trtllm_routed MoE
runner backend (scoring_func="softmax", no correction bias, no custom routing).
Its packed top-k weights are applied directly as the trtllm-gen MoE combine
scale, so they must be softmax probabilities in BOTH renormalize modes.

With renormalize=False the HF reference (Qwen2MoE-style gates with
norm_topk_prob=False) applies softmax over ALL experts and gathers the top-k
WITHOUT renormalizing over the selection. The old implementation returned the
raw gathered router logits instead: arbitrary signed values that silently
corrupted the MoE output from the first router layer on, with no error raised
anywhere (the non-routed trtllm bf16 path asserts "Renormalize is required",
so the False case was never considered on the routed path).
"""

import unittest

import torch

from sglang.srt.layers.moe.topk import fused_topk_softmax_torch_raw_logits
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")

NUM_EXPERTS = 256
TOPK = 8


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFusedTopkSoftmaxRawLogits(CustomTestCase):
    device = "cuda"

    def _run(self):
        torch.manual_seed(2026)
        hidden = torch.randn(128, 64, device=self.device, dtype=torch.bfloat16)
        gating = torch.randn(128, NUM_EXPERTS, device=self.device, dtype=torch.bfloat16)
        w_false, ids_false = fused_topk_softmax_torch_raw_logits(
            hidden, gating, TOPK, renormalize=False
        )
        w_true, ids_true = fused_topk_softmax_torch_raw_logits(
            hidden, gating, TOPK, renormalize=True
        )
        return gating, (w_false, ids_false), (w_true, ids_true)

    def test_renormalize_false_gathers_full_softmax(self):
        # HF reference: softmax over ALL experts, top-k gathered, NOT renormalized
        gating, (w, ids), _ = self._run()
        ref = torch.softmax(gating.float(), dim=-1).gather(1, ids.long())
        torch.testing.assert_close(w, ref)
        # probabilities in [0, 1] (raw logits would be arbitrary signed values)
        self.assertTrue(((w >= 0) & (w <= 1)).all().item())
        self.assertTrue((w.sum(dim=-1) <= 1.0 + 1e-6).all().item())

    def test_renormalize_true_softmax_over_selection(self):
        # unchanged semantics: softmax over the gathered logits
        gating, _, (w, ids) = self._run()
        ref = torch.softmax(gating.float().gather(1, ids.long()), dim=-1)
        # this arm's code path is untouched: require exact equality
        self.assertTrue(torch.equal(w, ref))
        torch.testing.assert_close(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)))

    def test_selection_on_logits_equals_selection_on_probs(self):
        # softmax is monotonic: selecting on the logits must pick the same
        # experts as selecting on the full-softmax probabilities. Run in
        # float32 so bf16 boundary ties cannot flip tie-breaking across
        # backends.
        torch.manual_seed(2026)
        hidden = torch.randn(8, 64, device=self.device, dtype=torch.float32)
        gating = torch.randn(8, NUM_EXPERTS, device=self.device, dtype=torch.float32)
        _, ids = fused_topk_softmax_torch_raw_logits(
            hidden, gating, TOPK, renormalize=False
        )
        probs = torch.softmax(gating, dim=-1)
        ref_ids = torch.topk(probs, TOPK, dim=-1, sorted=False)[1]
        self.assertTrue(
            torch.equal(ids.sort(dim=-1).values.long(), ref_ids.sort(dim=-1).values)
        )


if __name__ == "__main__":
    unittest.main()
