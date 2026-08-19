import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MaskFillingHead:
    def __init__(self, *, num_drafts: int) -> None:
        self.num_drafts = num_drafts

    def sample_block(
        self,
        base_logits,
        *,
        first_prev_tokens,
        hidden_states,
        sampler,
    ):
        del first_prev_tokens, hidden_states, sampler
        bs, _, vocab_size = base_logits.shape
        draft_tokens = torch.arange(bs * self.num_drafts).view(bs, self.num_drafts)
        corrected_logits = torch.arange(
            bs * self.num_drafts * vocab_size, dtype=base_logits.dtype
        ).view(bs, self.num_drafts, vocab_size)
        return draft_tokens, corrected_logits


class _MaskFillingModel:
    def __init__(self, *, num_drafts: int, vocab_size: int) -> None:
        self.markov_head = _MaskFillingHead(num_drafts=num_drafts)
        self.lm_head = SimpleNamespace(
            org_vocab_size=vocab_size,
            weight=torch.empty(vocab_size, 2),
        )

    def compute_base_logits(self, hidden_states):
        return (
            torch.zeros(hidden_states.shape[0], self.lm_head.org_vocab_size),
            None,
        )


def _confidence(*, draft_hidden, anchor_tokens, draft_tokens, confidence_tap):
    del draft_hidden, anchor_tokens, confidence_tap
    return torch.ones(draft_tokens.shape, dtype=torch.float32)


class TestDsparkDraftSampler(CustomTestCase):
    def test_mask_filling_outputs_use_draft_width(self):
        """A gamma-row mask-filling block emits gamma-1 sampler outputs."""
        gamma = 4
        num_drafts = gamma - 1
        max_bs = 2
        vocab_size = 7
        shared_out = torch.empty(max_bs * num_drafts, dtype=torch.int64)
        sampler = DsparkDraftSampler(
            model=_MaskFillingModel(num_drafts=num_drafts, vocab_size=vocab_size),
            gamma=gamma,
            num_drafts=num_drafts,
            max_bs=max_bs,
            device="cpu",
            confidence_fn=_confidence,
            out=shared_out,
            folded_sampling=True,
        )

        sampler(
            hidden_states=torch.zeros(max_bs * gamma, 2),
            input_ids=torch.arange(max_bs * gamma),
        )

        self.assertIs(sampler.out, shared_out)
        self.assertEqual(sampler.out.shape, (max_bs * num_drafts,))
        self.assertEqual(
            sampler.corrected_out.shape,
            (max_bs * num_drafts, vocab_size),
        )
        self.assertEqual(sampler.confidence_out.shape, (max_bs, num_drafts))
        self.assertTrue(torch.equal(sampler.out, torch.arange(max_bs * num_drafts)))
        self.assertTrue(
            torch.equal(
                sampler.corrected_out,
                torch.arange(
                    max_bs * num_drafts * vocab_size, dtype=torch.float32
                ).view(max_bs * num_drafts, vocab_size),
            )
        )
        self.assertTrue(
            torch.equal(sampler.confidence_out, torch.ones(max_bs, num_drafts))
        )


if __name__ == "__main__":
    unittest.main()
