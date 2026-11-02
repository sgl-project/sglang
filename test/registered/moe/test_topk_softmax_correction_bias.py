"""Softmax routing with a selection bias (ERNIE-4.5 ``moe_statics``).

The reference router selects experts by ``softmax(logits) + bias`` and weights
them by the unbiased probabilities. The CUDA softmax top-k kernels take no bias,
so a biased softmax gate must not be routed through them.
"""

import unittest

import torch

from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")

NUM_EXPERTS = 64
TOPK = 6
HIDDEN = 64


def _reference(logits, bias, topk):
    probs = logits.float().softmax(dim=-1)
    ids = (probs + bias).topk(topk, dim=-1).indices
    weights = probs.gather(1, ids)
    return weights / weights.sum(dim=-1, keepdim=True), ids


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestTopkSoftmaxCorrectionBias(CustomTestCase):
    DEVICE = "cuda"

    def _inputs(self, num_tokens=128):
        torch.manual_seed(0)
        hidden = torch.randn(
            num_tokens, HIDDEN, device=self.DEVICE, dtype=torch.bfloat16
        )
        logits = torch.randn(num_tokens, NUM_EXPERTS, device=self.DEVICE)
        # like ERNIE-4.5-21B-A3B: a large common offset plus a small per-expert spread
        bias = 62.85 + 0.01 * torch.randn(NUM_EXPERTS, device=self.DEVICE)
        return hidden, logits, bias

    def test_inputs_depend_on_bias(self):
        _, logits, bias = self._inputs()
        _, with_bias = _reference(logits, bias, TOPK)
        _, without_bias = _reference(logits, torch.zeros_like(bias), TOPK)
        self.assertFalse(
            torch.equal(with_bias.sort(dim=-1).values, without_bias.sort(dim=-1).values)
        )

    def test_select_experts_applies_bias(self):
        hidden, logits, bias = self._inputs()
        out = select_experts(
            hidden,
            logits,
            TopKConfig(top_k=TOPK, renormalize=True, correction_bias=bias),
        )
        ref_weights, ref_ids = _reference(logits, bias, TOPK)
        ids, order = out.topk_ids.long().sort(dim=-1)
        ref_ids, ref_order = ref_ids.sort(dim=-1)
        torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            out.topk_weights.gather(1, order), ref_weights.gather(1, ref_order)
        )


if __name__ == "__main__":
    unittest.main()
