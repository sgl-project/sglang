"""Softmax routing with a selection bias (ERNIE-4.5 ``moe_statics``).

The reference router (transformers ``Ernie4_5_MoeTopKRouter``) ranks experts by
``softmax(logits) + bias`` in fp32 and weights the chosen experts by their
unbiased probabilities, renormalized. ``fused_topk`` must not drop the bias, nor
apply it to the logits as the fused router's default softmax mode does.
"""

import unittest

import torch

from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, stage="base-b", runner_config="1-gpu-small-amd")

NUM_TOKENS = 512
# ERNIE-4.5-21B-A3B: 64 routed experts, top-6
NUM_EXPERTS = 64
TOPK = 6
# Rows whose reference top-k is decided by less than ~3 fp32 ulps at the bias offset;
# the fused kernel's exp may order them differently.
NEAR_TIE = 1e-5


def _reference(logits, bias, topk):
    probs = logits.float().softmax(dim=-1)
    ids = (probs + bias.float().view(1, -1)).topk(topk, dim=-1).indices
    weights = probs.gather(1, ids)
    return weights / weights.sum(dim=-1, keepdim=True), ids


def _sorted_by_id(weights, ids):
    ids, order = ids.long().sort(dim=-1)
    return weights.float().gather(1, order), ids


def _decided_rows(logits, bias, topk):
    biased = logits.float().softmax(dim=-1) + bias.float().view(1, -1)
    top = biased.topk(topk + 1, dim=-1).values
    return top[:, topk - 1] - top[:, topk] > NEAR_TIE


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestTopkSoftmaxCorrectionBias(CustomTestCase):
    DEVICE = "cuda"

    def _inputs(self, logits_dtype):
        gen = torch.Generator(device=self.DEVICE).manual_seed(0)
        logits = torch.randn(
            NUM_TOKENS, NUM_EXPERTS, generator=gen, device=self.DEVICE
        ).to(logits_dtype)
        # Like ERNIE-4.5-21B-A3B layer 1: a large common offset and a small
        # per-expert spread. Only the spread changes the selection.
        bias = 62.85 + 1e-3 * torch.randn(
            NUM_EXPERTS, generator=gen, device=self.DEVICE
        )
        hidden = torch.zeros(NUM_TOKENS, 8, dtype=torch.bfloat16, device=self.DEVICE)
        return hidden, logits, bias

    def _route(self, hidden, logits, bias, torch_native=False):
        config = TopKConfig(
            top_k=TOPK,
            renormalize=True,
            correction_bias=bias,
            torch_native=torch_native,
        )
        with get_context().override_server_args():
            out = select_experts(
                hidden_states=hidden, router_logits=logits, topk_config=config
            )
        return out.topk_weights, out.topk_ids

    def _assert_same_routing(self, weights, ids, ref_weights, ref_ids, rows=None):
        weights, ids = _sorted_by_id(weights, ids)
        ref_weights, ref_ids = _sorted_by_id(ref_weights, ref_ids)
        if rows is not None:
            self.assertGreater(rows.float().mean().item(), 0.97)
            weights, ids, ref_weights, ref_ids = (
                t[rows] for t in (weights, ids, ref_weights, ref_ids)
            )
        torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, ref_weights)

    def test_bias_changes_the_reference_selection(self):
        # Guard: without this the other tests could pass with the bias ignored.
        _, logits, bias = self._inputs(torch.float32)
        _, with_bias = _reference(logits, bias, TOPK)
        _, without_bias = _reference(logits, torch.zeros_like(bias), TOPK)
        changed = (
            (with_bias.sort(dim=-1).values != without_bias.sort(dim=-1).values)
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
        self.assertGreater(changed, 0.05)

    def test_select_experts_matches_reference(self):
        for logits_dtype in (torch.bfloat16, torch.float32):
            # ERNIE-4.5 passes a (1, E) bias on CUDA, ERNIE-4.5-VL an (E,) one
            for bias_shape in ((NUM_EXPERTS,), (1, NUM_EXPERTS)):
                for torch_native in (False, True):
                    with self.subTest(
                        logits_dtype=logits_dtype,
                        bias_shape=bias_shape,
                        torch_native=torch_native,
                    ):
                        hidden, logits, bias = self._inputs(logits_dtype)
                        bias = bias.view(bias_shape)
                        weights, ids = self._route(hidden, logits, bias, torch_native)
                        if not torch_native:
                            self.assertEqual(ids.dtype, torch.int32)
                        ref_weights, ref_ids = _reference(logits, bias, TOPK)
                        self._assert_same_routing(
                            weights,
                            ids,
                            ref_weights,
                            ref_ids,
                            rows=_decided_rows(logits, bias, TOPK),
                        )

    def test_zero_bias_matches_unbiased_routing(self):
        # fp32 logits: random bf16 rows often tie at the top-k boundary.
        hidden, logits, bias = self._inputs(torch.float32)
        zero_bias = torch.zeros_like(bias)
        weights, ids = self._route(hidden, logits, zero_bias)
        ref_weights, ref_ids = _reference(logits, zero_bias, TOPK)
        self._assert_same_routing(weights, ids, ref_weights, ref_ids)
        # and the fused kernel that serves gates without a bias agrees
        kernel_weights, kernel_ids = self._route(hidden, logits, None)
        self._assert_same_routing(weights, ids, kernel_weights, kernel_ids)


if __name__ == "__main__":
    unittest.main()
