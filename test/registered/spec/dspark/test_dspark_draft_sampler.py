import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft import sample_draft_block
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BS, _GAMMA, _HIDDEN, _VOCAB, _RANK = 3, 4, 16, 64, 8


def _single_rank_tp_sync() -> SpecTpSync:
    return SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0))


class _DraftModel:
    """BF16 base logits plus an FP32 Markov bias, as in the DeepSeek-V4 draft
    head, so the corrected step logits are FP32."""

    sample_from_anchor = True

    def __init__(self):
        gen = torch.Generator().manual_seed(0)
        self.head_weight = torch.randn(_VOCAB, _HIDDEN, generator=gen).bfloat16()
        self.lm_head = SimpleNamespace(org_vocab_size=_VOCAB, weight=self.head_weight)
        self.markov_head = VanillaMarkov(vocab_size=_VOCAB, markov_rank=_RANK)
        with torch.no_grad():
            for param in self.markov_head.parameters():
                param.copy_(torch.randn(param.shape, generator=gen))

    def compute_base_logits(self, hidden):
        return hidden.bfloat16() @ self.head_weight.T, None


class TestFoldedCorrectedLogits(CustomTestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(1)
        self.model = _DraftModel()
        self.hidden = torch.randn(_BS * _GAMMA, _HIDDEN, generator=gen)
        self.input_ids = torch.randint(0, _VOCAB, (_BS * _GAMMA,), generator=gen)
        self.anchor = self.input_ids.view(_BS, _GAMMA)[:, 0]
        self.sampler = DsparkDraftSampler(
            model=self.model,
            gamma=_GAMMA,
            max_bs=_BS,
            device="cpu",
            tp_sync=_single_rank_tp_sync(),
        )

    def _corrected(self):
        return self.sampler.corrected_out[: _BS * _GAMMA].view(_BS, _GAMMA, -1)

    def test_sampled_rows_keep_the_logits_they_were_drawn_from(self):
        sampling_info = SimpleNamespace(
            temperatures=torch.full((_BS, 1), 0.7),
            top_ks=torch.full((_BS,), 1 << 30, dtype=torch.int32),
        )
        self.sampler.stage_sampling_params(bs=_BS, sampling_info=sampling_info)
        self.sampler(self.hidden, self.input_ids)

        tokens = self.sampler.out[: _BS * _GAMMA].view(_BS, _GAMMA)
        base = self.model.compute_base_logits(self.hidden)[0].view(_BS, _GAMMA, -1)
        corrected = self._corrected()
        self.assertEqual(corrected.dtype, torch.float32)
        prev = self.anchor
        for step in range(_GAMMA):
            step_logits = self.model.markov_head.apply_step_logits(
                base[:, step], token_ids=prev, hidden_states=None
            )
            self.assertEqual(step_logits.dtype, torch.float32)
            self.assertTrue(torch.equal(corrected[:, step], step_logits))
            prev = tokens[:, step]

    def test_greedy_rows_match_the_eager_proposal(self):
        self.sampler.stage_sampling_params(bs=_BS, sampling_info=None)
        self.sampler(self.hidden, self.input_ids)

        base = self.model.compute_base_logits(self.hidden)[0].view(_BS, _GAMMA, -1)
        eager = sample_draft_block(
            base_logits=base,
            anchor_tokens=self.anchor,
            draft_hidden=self.hidden.view(_BS, _GAMMA, -1),
            sampling_info=None,
            markov_head=self.model.markov_head,
            device="cpu",
            tp_sync=_single_rank_tp_sync(),
        )
        tokens = self.sampler.out[: _BS * _GAMMA].view(_BS, _GAMMA)
        self.assertTrue(torch.equal(tokens, eager.draft_tokens))
        self.assertTrue(torch.equal(self._corrected(), eager.corrected_logits))


if __name__ == "__main__":
    unittest.main()
