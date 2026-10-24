import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, GAMMA, VOCAB = 3, 4, 50


class TestGreedyDraftAccept(CustomTestCase):
    def test_sampling_accept_reads_a_one_hot_draft(self):
        # The sampling accept gets a one-hot draft at the target vocab width.
        torch.manual_seed(0)
        draft_tokens = torch.randint(0, VOCAB, (BS, GAMMA))
        block = DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=None,
            greedy_mask=torch.zeros(BS, dtype=torch.bool),
            temperatures=torch.ones(BS),
            greedy_draft=True,
        )
        seen = {}

        def fake_accept(**kwargs):
            seen["draft_probs"] = kwargs["draft_probs"]
            zeros = torch.zeros(BS, dtype=torch.int64)
            return zeros, zeros, zeros

        with mock.patch.object(
            dspark_verify.AcceptSampling, "execute", side_effect=fake_accept
        ):
            dspark_verify.accept_draft_tokens(
                candidates=torch.zeros(BS, GAMMA + 1, dtype=torch.int64),
                target_logits=torch.randn(BS * (GAMMA + 1), VOCAB),
                draft_block=block,
                sampling_info=SimpleNamespace(is_all_greedy=False, is_any_greedy=False),
                draft_input=None,
                gamma=GAMMA,
                verify_num_draft_tokens=GAMMA + 1,
            )
        probs = seen["draft_probs"]
        self.assertEqual(probs.shape, (BS, GAMMA, VOCAB))
        expected = torch.nn.functional.one_hot(draft_tokens, VOCAB).float()
        self.assertTrue(torch.equal(probs, expected))


if __name__ == "__main__":
    unittest.main()
