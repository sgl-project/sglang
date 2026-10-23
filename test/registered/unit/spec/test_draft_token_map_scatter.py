import unittest

import torch

from sglang.srt.speculative.spec_utils import (
    scatter_draft_probs_to_target_vocab,
    validate_draft_token_map,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestScatterDraftProbs(unittest.TestCase):
    def test_places_mass_at_mapped_ids_and_zero_elsewhere(self):
        token_map = torch.tensor([5, 1, 7])
        q = torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]])
        out = scatter_draft_probs_to_target_vocab(q, token_map, 9)
        self.assertEqual(out.shape, (2, 9))
        self.assertTrue(torch.equal(out[:, [5, 1, 7]], q))
        rest = [i for i in range(9) if i not in (5, 1, 7)]
        self.assertTrue((out[:, rest] == 0).all())
        torch.testing.assert_close(out.sum(-1), q.sum(-1))

    def test_stacked_steps(self):
        token_map = torch.tensor([3, 0])
        q = torch.rand(4, 3, 2).softmax(-1)
        out = scatter_draft_probs_to_target_vocab(q, token_map, 6)
        self.assertEqual(out.shape, (4, 3, 6))
        self.assertTrue(torch.equal(out[..., 3], q[..., 0]))
        self.assertTrue(torch.equal(out[..., 0], q[..., 1]))
        self.assertTrue((out[..., [1, 2, 4, 5]] == 0).all())

    def test_sampled_token_maps_to_the_proposed_target_id(self):
        # The draft samples reduced index i with prob q[i] and commits token_map[i];
        # the scattered q must give that target id the same probability.
        token_map = torch.tensor([8, 2, 5])
        q = torch.tensor([[0.2, 0.5, 0.3]])
        out = scatter_draft_probs_to_target_vocab(q, token_map, 10)
        for i in range(3):
            self.assertEqual(out[0, token_map[i]].item(), q[0, i].item())

    def test_matches_dtype_and_device(self):
        q = torch.rand(2, 4, dtype=torch.float64)
        out = scatter_draft_probs_to_target_vocab(q, torch.arange(4), 7)
        self.assertEqual(out.dtype, torch.float64)


class TestValidateTokenMap(unittest.TestCase):
    def test_accepts_injective_in_range(self):
        validate_draft_token_map(torch.tensor([9, 0, 4]), 10)

    def test_rejects_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "outside the target vocab"):
            validate_draft_token_map(torch.tensor([1, 10]), 10)
        with self.assertRaisesRegex(ValueError, "outside the target vocab"):
            validate_draft_token_map(torch.tensor([-1, 3]), 10)

    def test_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_draft_token_map(torch.tensor([2, 2, 3]), 10)

    def test_rejects_empty(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_draft_token_map(torch.tensor([], dtype=torch.int64), 10)


if __name__ == "__main__":
    unittest.main()
