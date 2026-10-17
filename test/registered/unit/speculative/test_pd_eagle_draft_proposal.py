"""A PD decode's first EAGLE draft input carries a rejection-sampling proposal.

PD prefill sends the first draft token, its probability and hidden state, but
not the distribution the token was drawn from. With rejection sampling on
(the ROCm default for EAGLE top-k 1), the decode's first draft step stacks
that missing proposal with the steps it drafts itself.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import eagle_disaggregation
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
    pd_first_draft_proposal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 11


def _spec(rejection_sampling=True, multi_layer=False):
    return SimpleNamespace(
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        enable_multi_layer_eagle=multi_layer,
        speculative_use_rejection_sampling=rejection_sampling,
    )


def _batch(tokens):
    # One entry per request: a token, or the token chain of multi-layer EAGLE.
    chains = [list(t) if isinstance(t, tuple) else [t] for t in tokens]
    reqs = [
        SimpleNamespace(
            output_topk_p=[0.25] * len(chain),
            output_topk_index=chain,
            hidden_states_tensor=torch.full((4,), float(i)),
            output_dsa_topk_indices=None,
        )
        for i, chain in enumerate(chains)
    ]
    return SimpleNamespace(
        reqs=reqs,
        device="cpu",
        enable_overlap=False,
        # A model without DSA seed metadata: no MTP index sharing.
        model_config=SimpleNamespace(vocab_size=VOCAB, hf_config=SimpleNamespace()),
    )


class TestPdEagleDraftProposal(CustomTestCase):
    def build(self, spec, tokens=(3, 7)):
        with patch.object(eagle_disaggregation, "get_spec", return_value=spec):
            return build_eagle_disagg_draft_input(
                _batch(tokens), torch.tensor([1, 2]), future_map=None
            )

    def test_rejection_sampling_gets_one_hot_proposal_at_drafted_token(self):
        spec_info = self.build(_spec())
        expected = torch.zeros((2, VOCAB))
        expected[0, 3] = expected[1, 7] = 1.0
        self.assertEqual(spec_info.draft_probs.dtype, torch.float32)
        torch.testing.assert_close(spec_info.draft_probs, expected)
        torch.testing.assert_close(spec_info.topk_index, torch.tensor([[3], [7]]))

    def test_proposal_matches_target_vocab_width(self):
        # eagle_sample refuses a proposal whose width differs from the target's.
        spec_info = self.build(_spec())
        self.assertEqual(spec_info.draft_probs.shape[-1], VOCAB)

    def test_no_proposal_without_rejection_sampling(self):
        self.assertIsNone(self.build(_spec(rejection_sampling=False)).draft_probs)
        self.assertIsNone(
            self.build(_spec(rejection_sampling=False, multi_layer=True)).draft_probs
        )

    def test_multi_layer_gets_one_hot_proposal_per_chain_step(self):
        # Multi-layer EAGLE verifies the received chain with no draft forward,
        # so it needs a (b, num_steps, vocab) proposal; None fails the verify.
        chains = ((3, 5, 9), (7, 0, 10))
        spec_info = self.build(_spec(multi_layer=True), tokens=chains)
        expected = torch.zeros((2, 3, VOCAB))
        for row, chain in enumerate(chains):
            for step, token in enumerate(chain):
                expected[row, step, token] = 1.0
        self.assertEqual(spec_info.draft_probs.dtype, torch.float32)
        torch.testing.assert_close(spec_info.draft_probs, expected)
        torch.testing.assert_close(spec_info.topk_index, torch.tensor(chains))

    def test_one_hot_proposal_keeps_verify_exact(self):
        # Chain verify of one token: accept X if coin * q(X) < p(X), otherwise
        # resample from (p - q)+. Enumerating X ~ q_draft and both outcomes, the
        # committed token's distribution must equal p for any drafting q.
        p = torch.tensor([0.5, 0.3, 0.2])
        q_draft = torch.tensor([0.1, 0.6, 0.3])
        committed = torch.zeros(3)
        for x in range(3):
            q = pd_first_draft_proposal(torch.tensor([[x]]), 3)[0, 0]
            accept = torch.clamp(p[x] / q[x], max=1.0)
            residual = torch.clamp(p - q, min=0.0)
            residual = residual / residual.sum()
            committed[x] += q_draft[x] * accept
            committed += q_draft[x] * (1 - accept) * residual
        torch.testing.assert_close(committed, p)


if __name__ == "__main__":
    unittest.main()
