"""Draft sampling overrides on the DSpark markov draft: the default path keeps
the fused in-kernel sample, and a top-k/top-p override samples from the cut-off
q that verification rebuilds from the kept step logits."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.sampling.draft_sampling import build_draft_probs
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dspark_components import dspark_draft, dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    draft_tp_rng_sites,
    resolve_draft_params,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, RANK, BS, GAMMA = 211, 8, 3, 4


def _sampling_info():
    # A sampling request with its own cutoffs, a plain one, and a greedy one.
    return SimpleNamespace(
        temperatures=torch.tensor([[0.7], [1.5], [1.0]]),
        top_ks=torch.tensor([2, TOP_K_ALL, 1], dtype=torch.int32),
        top_ps=torch.tensor([0.8, 0.9, 1.0]),
        is_all_greedy=False,
        is_any_greedy=True,
    )


def _noise():
    return DraftNoise(key=3, offset=torch.zeros(1, dtype=torch.int64))


def _tp_sync():
    return SimpleNamespace(sync=lambda _site, values: values)


class TestDsparkDraftOverrides(unittest.TestCase):
    def setUp(self):
        self.spec = SimpleNamespace(
            speculative_draft_temperature=None,
            speculative_draft_top_k=None,
            speculative_draft_top_p=None,
        )
        self.enterContext(
            mock.patch("sglang.srt.runtime_context.get_spec", return_value=self.spec)
        )
        # CPU exercises the multinomial fallback; Gumbel's row argmax is GPU-only.
        self.enterContext(envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False))
        torch.manual_seed(0)
        self.head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(torch.bfloat16)
        self.base = torch.randn(BS, GAMMA, VOCAB).to(torch.bfloat16)
        self.hidden = torch.randn(BS, GAMMA, 16).to(torch.bfloat16)
        self.anchor = torch.randint(0, VOCAB, (BS,))

    def _draft(self):
        return sample_draft_block(
            base_logits=self.base,
            anchor_tokens=self.anchor,
            draft_hidden=self.hidden,
            sampling_info=_sampling_info(),
            markov_head=self.head,
            noise=_noise(),
            device="cpu",
            tp_sync=_tp_sync(),
            greedy_draft=False,
        )

    def test_cutoff_redraw_declares_the_draft_an_rng_site(self):
        # The fused kernel's Philox noise matches across ranks; the cutoff
        # redraw does not, so it must join the rng sync preset.
        self.assertEqual(draft_tp_rng_sites(greedy_draft=False), frozenset())
        self.spec.speculative_draft_top_p = 0.5
        self.assertEqual(
            draft_tp_rng_sites(greedy_draft=False), {SpecTpSyncSite.DSPARK_DRAFT}
        )
        self.assertEqual(draft_tp_rng_sites(greedy_draft=True), frozenset())

    def test_default_draft_keeps_the_fused_sample_and_no_cutoffs(self):
        with mock.patch.object(
            dspark_draft, "build_draft_probs", side_effect=AssertionError
        ):
            block = self._draft()
        self.assertIsNone(block.cutoffs)
        params = resolve_draft_params(
            sampling_info=_sampling_info(), bs=BS, device="cpu"
        )
        self.assertIsNone(params.cutoffs)
        torch.testing.assert_close(params.temperatures, torch.tensor([0.7, 1.5, 1.0]))

    def test_draft_temperature_replaces_every_sampling_row(self):
        self.spec.speculative_draft_temperature = 0.4
        params = resolve_draft_params(
            sampling_info=_sampling_info(), bs=BS, device="cpu"
        )
        torch.testing.assert_close(params.temperatures, torch.full((BS,), 0.4))
        # greedy_mask stays the target's: it picks the accept rule.
        self.assertEqual(params.greedy_mask.tolist(), [False, False, True])
        torch.testing.assert_close(
            params.step_temperatures, torch.tensor([0.4, 0.4, 0.0])
        )

    def test_cutoffs_sample_from_the_truncated_q_that_verify_rebuilds(self):
        self.spec.speculative_draft_top_k = 2
        self.spec.speculative_draft_top_p = 0.5
        block = self._draft()
        cutoffs = block.cutoffs
        self.assertEqual(cutoffs.top_ks.tolist(), [2, 2, 1])
        torch.testing.assert_close(cutoffs.top_ps, torch.full((BS,), 0.5))
        self.assertEqual((cutoffs.apply_top_k, cutoffs.apply_top_p), (True, True))
        q = build_draft_probs(block.corrected_logits, cutoffs)
        # Every drafted token lies inside its row's cut-off support.
        self.assertTrue(bool((q.gather(-1, block.draft_tokens[..., None]) > 0).all()))
        self.assertTrue(bool(((q > 0).sum(-1) <= 2).all()))
        # The greedy row takes the argmax of its corrected logits.
        self.assertTrue(
            torch.equal(
                block.draft_tokens[2], block.corrected_logits[2].float().argmax(-1)
            )
        )
        # Verification rebuilds that exact q from the kept logits.
        zeros = torch.zeros(BS, dtype=torch.int32)
        selected = SimpleNamespace(correct_len=zeros, bonus=zeros, cap_trim_lens=zeros)
        with (
            mock.patch.object(
                dspark_verify.AcceptGreedy, "execute", return_value=(zeros,) * 3
            ),
            mock.patch.object(
                dspark_verify.AcceptSampling, "execute", return_value=(zeros,) * 3
            ) as sampling,
            mock.patch.object(
                dspark_verify.SelectMixedAccept, "execute", return_value=selected
            ),
        ):
            dspark_verify.accept_draft_tokens(
                candidates=torch.zeros(BS, GAMMA + 1, dtype=torch.long),
                target_logits=torch.zeros(BS * (GAMMA + 1), VOCAB),
                draft_block=block,
                sampling_info=_sampling_info(),
                draft_input=None,
                gamma=GAMMA,
                verify_num_draft_tokens=GAMMA + 1,
            )
        torch.testing.assert_close(sampling.call_args.kwargs["draft_probs"], q)
        # A q handed in (the verify builds it once for the accept and the
        # estimator) is scored as is, never rebuilt.
        given = q * 0.5
        with (
            mock.patch.object(
                dspark_verify.AcceptGreedy, "execute", return_value=(zeros,) * 3
            ),
            mock.patch.object(
                dspark_verify.AcceptSampling, "execute", return_value=(zeros,) * 3
            ) as sampling,
            mock.patch.object(
                dspark_verify.SelectMixedAccept, "execute", return_value=selected
            ),
            mock.patch.object(
                dspark_verify, "cutoff_draft_probs", side_effect=AssertionError
            ),
        ):
            dspark_verify.accept_draft_tokens(
                candidates=torch.zeros(BS, GAMMA + 1, dtype=torch.long),
                target_logits=torch.zeros(BS * (GAMMA + 1), VOCAB),
                draft_block=block,
                sampling_info=_sampling_info(),
                draft_input=None,
                gamma=GAMMA,
                verify_num_draft_tokens=GAMMA + 1,
                draft_probs=given,
            )
        self.assertIs(sampling.call_args.kwargs["draft_probs"], given)

    def test_folded_sampler_stages_cutoffs_for_replay(self):
        self.spec.speculative_draft_top_k = 3
        model = SimpleNamespace(
            markov_head=self.head,
            sample_from_anchor=True,
            lm_head=SimpleNamespace(org_vocab_size=VOCAB),
            confidence_head=None,
            # The hidden rows stand in for the base logits.
            compute_base_logits=lambda hidden: (hidden, None),
        )
        sampler = DsparkDraftSampler(
            model=model,
            gamma=GAMMA,
            max_bs=BS + 1,
            device="cpu",
            noise=_noise(),
            tp_sync=_tp_sync(),
            greedy_draft=False,
        )
        sampler.step_logits = torch.empty((BS + 1, GAMMA, VOCAB), dtype=torch.bfloat16)
        sampler.stage_sampling_params(bs=BS, sampling_info=_sampling_info())
        cutoffs = sampler.params.cutoffs
        self.assertEqual(cutoffs.top_ks[:BS].tolist(), [3, 3, 1])
        self.assertEqual((cutoffs.apply_top_k, cutoffs.apply_top_p), (True, False))
        sampler(self.base.view(BS * GAMMA, VOCAB), self.anchor.repeat_interleave(GAMMA))
        tokens = sampler.out[: BS * GAMMA].view(BS, GAMMA)
        q = build_draft_probs(sampler.step_logits[:BS], cutoffs.slice(BS))
        self.assertTrue(bool((q.gather(-1, tokens[..., None]) > 0).all()))
        self.assertTrue(bool(((q > 0).sum(-1) <= 3).all()))


if __name__ == "__main__":
    unittest.main()
