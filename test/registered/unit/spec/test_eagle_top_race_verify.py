"""Under TOP, EAGLE verify draws each row's target exactly as non-spec decode draws that token.

Draft-dependent rejection sampling keyed by the verify-window start made a retracted request diverge
from its uninterrupted run. Verify targets must depend only on the row's logits, its request's seed and
temperature, and the row's input-token position.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_utils import _top_race_verify_targets
from sglang.srt.true_on_policy.sampling import scale_logits_for_sampling
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 64
DRAFT_TOKEN_NUM = 3


def _fake_race(logprobs, seeds, positions):
    # A CPU stand-in for the race: a pure function of (row logprobs, seed, position).
    winner = torch.argmax(logprobs.float(), dim=-1)
    return ((winner + 7 * seeds.long() + 13 * positions.long()) % VOCAB).unsqueeze(1)


class TestTopRaceVerifyTargets(CustomTestCase):
    # TOP temperature scaling asks whether a CUDA graph is capturing; no device here.
    @patch("torch.cuda.is_current_stream_capturing", return_value=False)
    def test_verify_targets_equal_non_spec_draws(self, _capturing):
        torch.manual_seed(0)
        seeds = torch.tensor([1234, 98765], dtype=torch.int64)
        temperatures = torch.tensor([[1.0], [0.7]], dtype=torch.float32)
        # Request 0 verifies input positions 10..12, request 1 positions 41..43.
        positions = torch.tensor([10, 11, 12, 41, 42, 43], dtype=torch.int64)
        logits = torch.randn(2 * DRAFT_TOKEN_NUM, VOCAB).to(torch.bfloat16)
        logits_before = logits.clone()

        fake_module = SimpleNamespace(sample_from_logprobs=_fake_race)
        with patch.dict(
            sys.modules, {"sglang.srt.true_on_policy.exponential_race": fake_module}
        ):
            targets = _top_race_verify_targets(
                next_token_logits=logits,
                temperatures=temperatures,
                sampling_seed=seeds,
                positions=positions,
                draft_token_num=DRAFT_TOKEN_NUM,
            )

        self.assertEqual(targets.shape, (2, DRAFT_TOKEN_NUM))
        self.assertEqual(targets.dtype, torch.int64)
        # Logprob return reads the verify logits afterwards.
        self.assertTrue(torch.equal(logits, logits_before))
        for row in range(2 * DRAFT_TOKEN_NUM):
            req = row // DRAFT_TOKEN_NUM
            # The non-spec sampler's draw for this token: one row, its own request's key.
            logprobs = torch.log_softmax(
                scale_logits_for_sampling(
                    logits_before[row : row + 1].clone(), temperatures[req : req + 1]
                ),
                dim=-1,
            )
            expected = _fake_race(
                logprobs, seeds[req : req + 1], positions[row : row + 1]
            )
            self.assertEqual(
                targets[req, row % DRAFT_TOKEN_NUM].item(),
                expected.item(),
                msg=f"row {row}",
            )

    def test_missing_seed_fails_instead_of_falling_back(self):
        with self.assertRaises(ValueError):
            _top_race_verify_targets(
                next_token_logits=torch.zeros(
                    DRAFT_TOKEN_NUM, VOCAB, dtype=torch.bfloat16
                ),
                temperatures=torch.ones(1, 1),
                sampling_seed=None,
                positions=torch.arange(DRAFT_TOKEN_NUM),
                draft_token_num=DRAFT_TOKEN_NUM,
            )


class TestRejectionSamplingIsNeverTopVerify(CustomTestCase):
    def test_top_refuses_rejection_sampling_targets(self):
        """Draft-dependent targets would make a retracted request diverge from its uninterrupted run."""
        with patch(
            "sglang.srt.true_on_policy.config.is_true_on_policy_enabled",
            return_value=True,
        ):
            with self.assertRaisesRegex(NotImplementedError, r"\[top\]"):
                eagle_utils._rejection_sampling_target_probs(
                    torch.zeros(1, VOCAB), torch.ones(1, 1)
                )

    def test_native_rejection_sampling_keeps_the_sampling_softmax(self):
        logits, temperatures = torch.zeros(1, VOCAB), torch.ones(1, 1)
        with (
            patch(
                "sglang.srt.true_on_policy.config.is_true_on_policy_enabled",
                return_value=False,
            ),
            patch.object(eagle_utils, "sampling_softmax") as softmax,
        ):
            probs = eagle_utils._rejection_sampling_target_probs(logits, temperatures)
        softmax.assert_called_once_with(logits, temperatures=temperatures)
        self.assertIs(probs, softmax.return_value)


if __name__ == "__main__":
    unittest.main()
