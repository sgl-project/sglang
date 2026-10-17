import unittest

import torch

from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPSpecRelayInput(unittest.TestCase):
    def test_degenerate_contains_only_the_logical_proposal(self):
        relay = PPSpecRelayInput.degenerate(
            rids=["a", "b"],
            bonus_tokens=torch.tensor([11, 22]),
            num_draft_tokens=4,
            speculative_num_steps=3,
        )

        self.assertEqual(relay.configuration, (3, 4))
        self.assertEqual(tuple(relay.tokens.shape), (2, 4))
        torch.testing.assert_close(
            relay.tokens,
            torch.tensor([[11, 0, 0, 0], [22, 0, 0, 0]]),
        )

    def test_filter_and_reindex_preserve_configuration(self):
        relay = PPSpecRelayInput(
            rids=["a", "b", "c"],
            tokens=torch.arange(12).reshape(3, 4),
            speculative_num_steps=3,
        )

        relay.filter_batch(torch.tensor([2, 0]), [2, 0])
        reindexed = relay.reindex(["a", "c"])

        self.assertEqual(relay.configuration, (3, 4))
        self.assertEqual(reindexed.configuration, (3, 4))
        torch.testing.assert_close(reindexed.tokens[:, 0], torch.tensor([0, 8]))

    def test_merge_converts_new_degenerate_rows_to_running_configuration(self):
        left = PPSpecRelayInput.degenerate(["a"], torch.tensor([1]), 4)
        right = PPSpecRelayInput.degenerate(["b"], torch.tensor([2]), 6)

        left.merge_batch(right)

        self.assertEqual(left.configuration, (3, 4))
        self.assertEqual(left.rids, ["a", "b"])
        torch.testing.assert_close(left.tokens[1], torch.tensor([2, 0, 0, 0]))

    def test_merge_rejects_drafted_rows_from_another_configuration(self):
        left = PPSpecRelayInput.degenerate(["a"], torch.tensor([1]), 4)
        right = PPSpecRelayInput(
            ["b"],
            torch.tensor([[2, 3, 4, 5, 6, 7]]),
            parents=torch.tensor([[-1, 0, 1, 2, 3]]),
            top_scores=torch.tensor([[0, 1, 2, 3, 4]]),
            speculative_num_steps=5,
        )

        with self.assertRaisesRegex(ValueError, "drafted.*different"):
            left.merge_batch(right)

    def test_adopt_transition_degenerates_uncovered_rows(self):
        current = PPSpecRelayInput(
            rids=["a", "new"],
            tokens=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
            speculative_num_steps=3,
        )
        relayed = PPSpecRelayInput(
            rids=["a"],
            tokens=torch.tensor([[30, 31, 32, 33, 34, 35]]),
            parents=torch.tensor([[-1, 0, 1, 2, 3]]),
            top_scores=torch.tensor([[0, 1, 2, 3, 4]]),
            speculative_num_steps=5,
        )

        current.adopt(relayed)

        self.assertEqual(current.configuration, (5, 6))
        torch.testing.assert_close(
            current.tokens,
            torch.tensor([[30, 31, 32, 33, 34, 35], [20, 0, 0, 0, 0, 0]]),
        )
        torch.testing.assert_close(
            current.parents,
            torch.tensor([[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]]),
        )


if __name__ == "__main__":
    unittest.main()
