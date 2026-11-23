"""TOP DeepSeek-V4.1 DSpark: what the admission refuses instead of verifying with another accept rule, and that drafts
are drawn under the key of the verify row they are matched to."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    DraftRaceKey,
    draft_markov_block,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.srt.true_on_policy.dsv41_dspark import race_accept, validate_dspark

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_CONFIG = SimpleNamespace(num_nextn_predict_layers=3)


def _spec(**overrides):
    fields = dict(
        speculative_algorithm="DSPARK",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=6,
        speculative_dspark_block_size=5,
        speculative_use_rejection_sampling=False,
        speculative_use_block_verification=False,
        speculative_draft_model_path=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _validate(spec, mode=RaggedVerifyMode.STATIC):
    with patch(
        "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
        return_value=mode,
    ):
        validate_dspark(_CONFIG, spec, "/models/v41")


class TestDsv41Dspark(CustomTestCase):
    def test_eagle_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "DSpark"):
            _validate(_spec(speculative_algorithm="EAGLE"))

    def test_another_block_size_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "block size 5"):
            _validate(
                _spec(speculative_dspark_block_size=7, speculative_num_draft_tokens=8)
            )

    def test_ragged_verify_is_refused(self):
        """The TOP attention program keys verify rows by a fixed draft count."""
        with self.assertRaisesRegex(NotImplementedError, "static chain"):
            _validate(_spec(), mode=RaggedVerifyMode.COMPACT)

    def test_rejection_sampling_is_refused(self):
        """Rejection sampling emits other tokens than non-speculative TOP decode would draw."""
        with self.assertRaisesRegex(NotImplementedError, "race"):
            _validate(_spec(speculative_use_rejection_sampling=True))


VOCAB, GAMMA = 64, 5


def _fake_race(logprobs, seeds, positions):
    # A CPU stand-in for the race: a pure function of (row argmax, seed, position).
    winner = torch.argmax(logprobs.float(), dim=-1)
    return ((winner + 7 * seeds.long() + 13 * positions.long()) % VOCAB).unsqueeze(1)


class _ZeroBiasMarkovHead:
    vocab_size = VOCAB
    vocab_shard = None

    def init_state(self, draft_hidden):
        return None

    def step_bias(self, prev_tokens, *, hidden_states, state):
        return torch.zeros(len(prev_tokens), VOCAB), state

    def step_logits_dtype(self, base_dtype):
        return torch.float32


class TestDsv41DsparkRaceDraft(CustomTestCase):
    """Draft step j is raced under verify row j's key (the anchor is row 0), so a draft whose distribution equals the
    target's is always accepted, and the emitted tokens are race_accept's targets whatever the drafts."""

    # TOP temperature scaling asks whether a CUDA graph is capturing; no device here.
    @patch("torch.cuda.is_current_stream_capturing", return_value=False)
    def _accept(self, shift, _capturing):
        torch.manual_seed(0)
        bs, verify = 3, GAMMA + 1
        seeds = torch.tensor([11, 22, 33], dtype=torch.int64)
        temperatures = torch.tensor([0.8, 1.0, 0.6], dtype=torch.float32)
        positions_2d = torch.tensor([[40], [7], [100]]) + torch.arange(verify)
        target_logits = torch.randn(bs * verify, VOCAB).to(torch.bfloat16)
        anchor = torch.tensor([3, 4, 5])
        fake = SimpleNamespace(sample_from_logprobs=_fake_race)
        with patch.dict(
            sys.modules, {"sglang.srt.true_on_policy.exponential_race": fake}
        ):
            drafts, _ = draft_markov_block(
                markov_head=_ZeroBiasMarkovHead(),
                # Draft step j's distribution is verify row j's.
                base_logits=target_logits.view(bs, verify, VOCAB)[:, :GAMMA].float(),
                anchor_tokens=anchor,
                draft_hidden=None,
                step_temperatures=temperatures,
                noise=DraftNoise(key=1, offset=torch.zeros(1, dtype=torch.int64)),
                tp_sync=SimpleNamespace(sync=lambda site, tokens: tokens),
                race_key=DraftRaceKey(
                    seeds=seeds, positions=positions_2d[:, shift : shift + GAMMA]
                ),
            )
            correct_len, bonus, _ = race_accept(
                candidates=torch.cat([anchor[:, None], drafts], dim=1),
                target_logits=target_logits,
                sampling_info=SimpleNamespace(
                    temperatures=temperatures[:, None], sampling_seed=seeds
                ),
                positions=positions_2d.reshape(-1),
                verify_num_draft_tokens=verify,
            )
        return correct_len

    def test_matched_draft_is_always_accepted(self):
        self.assertEqual(self._accept(0).tolist(), [GAMMA] * 3)

    def test_draft_keyed_one_row_late_is_not(self):
        self.assertEqual(self._accept(1).tolist(), [0] * 3)


if __name__ == "__main__":
    unittest.main()
