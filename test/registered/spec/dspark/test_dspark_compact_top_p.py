import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    accept_draft_tokens,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"
_VOCAB, _STRIDE = 32000, 6
_VERIFY_LENS = [6, 3, 1, 4]
_BS = len(_VERIFY_LENS)
# Top-p keeps heads 0 and 1 and drops head 2 for every request's temperature/top-p.
_HEAD_PROBS = [0.6, 0.37, 0.03]
# Head index drafted on each verified row: reject on row 2 (req 0), reject on
# the last verified row (req 1), accept every verified row (reqs 2 and 3).
_DRAFT_HEADS = [[0, 0, 2, 0, 0], [0, 0, 2, 0, 0], [0] * 5, [0] * 5]


def _sampling_info(**overrides):
    info = SimpleNamespace(
        temperatures=torch.tensor([[1.0], [0.7], [1.0], [1.3]], device=DEVICE),
        top_ps=torch.tensor([0.95, 0.9, 0.85, 0.8], device=DEVICE),
        top_ks=torch.full((_BS,), _VOCAB, dtype=torch.int32, device=DEVICE),
        need_top_k_sampling=False,
        need_top_p_sampling=True,
        is_all_greedy=False,
        is_any_greedy=False,
        has_custom_logit_processor=False,
        acc_linear_penalties=None,
        penalizer_orchestrator=None,
        grammar_mask=None,
        logit_bias=None,
    )
    for name, value in overrides.items():
        setattr(info, name, value)
    return info


def _compact_batch(vocab=_VOCAB, verify_lens=_VERIFY_LENS):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    logits = torch.randn(_BS, _STRIDE, vocab, generator=gen, device=DEVICE)
    perm = torch.randperm(vocab, generator=gen, device=DEVICE)
    heads = perm[: _BS * _STRIDE * 3].view(_BS, _STRIDE, 3)
    head_logits = 25.0 + torch.tensor(_HEAD_PROBS, device=DEVICE).log()
    logits.scatter_(2, heads, head_logits.expand(_BS, _STRIDE, 3))
    real = (
        torch.arange(_STRIDE, device=DEVICE)[None, :]
        < torch.tensor(verify_lens, device=DEVICE)[:, None]
    )
    logits = torch.where(real[..., None], logits, torch.zeros_like(logits))
    head_drafts = heads[:, :-1].gather(
        2, torch.tensor(_DRAFT_HEADS, device=DEVICE)[..., None]
    )
    pad_drafts = perm[-_BS * (_STRIDE - 1) :].view(_BS, _STRIDE - 1)
    drafts = torch.where(real[:, :-1], head_drafts.squeeze(-1), pad_drafts)
    candidates = torch.cat(
        [torch.zeros(_BS, 1, dtype=torch.long, device=DEVICE), drafts], dim=1
    )
    # Uniform draft probs: verified drafts accept iff top-p keeps them, and
    # drafts on uniform padding rows accept.
    draft_block = DraftBlockResult(
        draft_tokens=drafts,
        corrected_logits=torch.zeros(_BS, _STRIDE - 1, vocab, device=DEVICE),
        greedy_mask=torch.zeros(_BS, dtype=torch.bool, device=DEVICE),
        temperatures=torch.ones(_BS, dtype=torch.float32, device=DEVICE),
    )
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor(verify_lens, device=DEVICE),
        graph_num_tokens=sum(verify_lens),
    )
    return logits.view(_BS * _STRIDE, vocab), candidates, heads, draft_block, layout


def _accept(batch, *, compact_padding, sampling_info, seed):
    logits, candidates, _, draft_block, layout = batch
    torch.manual_seed(seed)
    return accept_draft_tokens(
        candidates=candidates,
        target_logits=logits,
        draft_block=draft_block,
        sampling_info=sampling_info,
        draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
        gamma=_STRIDE - 1,
        verify_num_draft_tokens=_STRIDE,
        cutoff_layout=layout,
        compact_padding=compact_padding,
    )


class TestCompactVerifyTopP(CustomTestCase):
    def test_target_probs_match_full_renorm(self):
        for vocab in (_VOCAB, 129280):
            with self.subTest(vocab=vocab):
                logits, _, heads, _, layout = _compact_batch(vocab)
                kwargs = dict(
                    next_token_logits=logits,
                    sampling_info=_sampling_info(),
                    draft_token_num=_STRIDE,
                    bs=_BS,
                )
                ref = build_speculative_verify_target_probs(**kwargs)
                got = build_speculative_verify_target_probs(
                    **kwargs, padding_verify_lens=layout.verify_lens
                )
                real = (
                    torch.arange(_STRIDE, device=DEVICE)[None, :]
                    < layout.verify_lens[:, None].long()
                )
                torch.testing.assert_close(got[real], ref[real], rtol=0, atol=0)
                torch.testing.assert_close(got[~real], ref[~real], rtol=1e-5, atol=0)
                head_probs = got.gather(2, heads)[real]
                self.assertTrue(bool((head_probs[:, :2] > 0).all()))
                self.assertTrue(bool((head_probs[:, 2] == 0).all()))

    def test_full_width_layout_matches_probs_and_peak_memory(self):
        logits, _, _, _, layout = _compact_batch(129280, [_STRIDE] * _BS)
        kwargs = dict(
            next_token_logits=logits,
            sampling_info=_sampling_info(),
            draft_token_num=_STRIDE,
            bs=_BS,
        )

        def build(**extra):
            build_speculative_verify_target_probs(**kwargs, **extra)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.memory_allocated()
            probs = build_speculative_verify_target_probs(**kwargs, **extra)
            torch.cuda.synchronize()
            return probs, torch.cuda.max_memory_allocated() - start

        ref, ref_peak = build()
        got, got_peak = build(padding_verify_lens=layout.verify_lens)
        self.assertTrue(torch.equal(got, ref))
        self.assertLessEqual(got_peak, ref_peak + (1 << 20))

    def test_accept_depths_and_cap_trim_match_full_renorm(self):
        batch = _compact_batch()
        heads = batch[2]
        for seed in range(4):
            ref = _accept(
                batch, compact_padding=False, sampling_info=_sampling_info(), seed=seed
            )
            got = _accept(
                batch, compact_padding=True, sampling_info=_sampling_info(), seed=seed
            )
            for name, g, r in zip(("correct_len", "bonus", "cap_trim"), got, ref):
                self.assertTrue(torch.equal(g, r), f"{name} seed={seed}: {g} vs {r}")
            correct_len, bonus, cap_trim = got
            self.assertEqual(correct_len.tolist(), [2, 2, 0, 3])
            self.assertEqual(cap_trim.tolist(), [0, 0, 5, 2])
            for i in (0, 1):
                self.assertIn(int(bonus[i]), heads[i, 2, :2].tolist())
            self.assertEqual(int(bonus[2]), int(heads[2, 0, 0]))
            self.assertEqual(int(bonus[3]), int(heads[3, 3, 0]))

    def test_logit_bias_keeps_full_renorm(self):
        logits, candidates, heads, draft_block, layout = _compact_batch()
        pad = (
            torch.arange(_STRIDE - 1, device=DEVICE)[None, :]
            >= torch.tensor(_VERIFY_LENS, device=DEVICE)[:, None]
        )
        used = set(heads.flatten().tolist()) | set(candidates.flatten().tolist())
        boosted = min(set(range(_VOCAB)) - used)
        # Top-p keeps only the boosted token on padding rows. The padding drafts
        # are near-impossible under the draft, so only top-p rejects them.
        bias = torch.zeros(_BS, _VOCAB, device=DEVICE)
        bias[:, boosted] = 16.0
        logits = (logits.view(_BS, _STRIDE, _VOCAB) + bias[:, None]).view_as(logits)
        draft_block = DraftBlockResult(
            draft_tokens=draft_block.draft_tokens,
            corrected_logits=draft_block.corrected_logits.scatter(
                2, candidates[:, 1:, None], (pad * -30.0)[..., None]
            ),
            greedy_mask=draft_block.greedy_mask,
            temperatures=draft_block.temperatures,
        )
        info = _sampling_info(
            temperatures=torch.ones(_BS, 1, device=DEVICE), logit_bias=bias
        )
        batch = (logits, candidates, heads, draft_block, layout)
        ref = _accept(batch, compact_padding=False, sampling_info=info, seed=0)
        got = _accept(batch, compact_padding=True, sampling_info=info, seed=0)
        for g, r in zip(got, ref):
            self.assertTrue(torch.equal(g, r), f"{g} vs {r}")
        self.assertEqual(ref[2].tolist(), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
