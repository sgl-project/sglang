import unittest

from sglang.srt.speculative.racer.automaton import RacerAutomaton
from sglang.srt.speculative.racer.draft_provider import RacerDraftProvider
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _parents(mask):
    parents = [-1] * len(mask)
    for i, row in enumerate(mask):
        ancestors = [j for j in range(i) if row[j]]
        if ancestors:
            parents[i] = ancestors[-1]
    return parents


class TestRacerLogitsFill(CustomTestCase):
    def test_empty_token_bin_grows_zero_placeholders_to_k(self):
        automaton = RacerAutomaton(ngram=3, topk=2, max_nodes=128)
        tokens, mask = automaton.retrieve(root_token=9, max_num_draft=4)

        self.assertEqual(tokens[0], 9)
        self.assertEqual(tokens[1:], [0, 0, 0])
        self.assertEqual(len(mask), 4)
        self.assertTrue(mask[0][0])
        self.assertEqual(_parents(mask), [-1, 0, 1, 2])
        self.assertEqual(automaton._last_real_count, 4)

    def test_logits_tree_fills_padding_without_replacing_retrieval(self):
        automaton = RacerAutomaton(ngram=3, topk=2, max_nodes=128)
        automaton.sync_history([1, 2, 3, 1, 2, 4, 1, 2])
        automaton.update_logits([3], [[50, 60]])

        tokens, mask = automaton.retrieve(root_token=3, max_num_draft=8)

        self.assertEqual(len(tokens), 8)
        self.assertEqual(tokens[0], 3)
        retrieval_prefix = [t for t in tokens if t not in (0, 50, 60)]
        self.assertGreaterEqual(len(retrieval_prefix), 1)
        self.assertIn(50, tokens)
        self.assertTrue(all(mask[i][i] for i in range(8)))

    def test_reuses_existing_retrieval_edge_then_expands(self):
        automaton = RacerAutomaton(ngram=2, topk=1, max_nodes=128, min_depth=1)
        automaton.sync_history([10, 11, 12])
        automaton.update_logits(
            [11, 12],
            [[12], [13]],
        )

        tokens, mask = automaton.retrieve(root_token=11, max_num_draft=4)
        self.assertEqual(tokens[0], 11)
        self.assertIn(12, tokens)
        self.assertIn(13, tokens)
        self.assertEqual(len(tokens), 4)
        self.assertTrue(mask[0][0])

    def test_missing_outgoing_edge_does_not_invent_children(self):
        automaton = RacerAutomaton(ngram=3, topk=4, max_nodes=128)
        automaton.update_logits([7], [[8]])

        tokens, mask = automaton.retrieve(root_token=7, max_num_draft=6)
        self.assertEqual(tokens[0], 7)
        self.assertEqual(tokens[1], 8)
        self.assertEqual(tokens[2:], [0, 0, 0, 0])
        self.assertEqual(automaton._last_real_count, 6)
        self.assertEqual(_parents(mask)[1], 0)
        # Token 8 has no TokenBin row, so it grows a 0-chain rather than
        # inventing extra copy-logit siblings.
        self.assertEqual(_parents(mask)[2], 1)

    def test_update_logits_keeps_zero_placeholder_copy_logits(self):
        provider = RacerDraftProvider(
            draft_token_num=4,
            ngram=3,
            topk=2,
            max_nodes=128,
        )
        tokens, _ = provider.batch_get(
            req_ids=["r0"],
            batch_tokens=[[1, 2, 3]],
            total_lens=[3],
        )
        self.assertEqual(int(tokens[0]), 3)
        self.assertTrue(all(int(t) == 0 for t in tokens[1:]))

        provider.update_logits(
            ["r0"],
            tokens,
            [
                [11, 12],
                [99, 98],
                [97, 96],
                [95, 94],
            ],
        )
        state = provider._state("r0")
        self.assertEqual(state._token_bin[3], [11, 12])
        # The 0-chain is part of the proposal tree, so the last placeholder
        # row becomes TokenBin[0] and can expand on the next round.
        self.assertEqual(state._token_bin[0], [95, 94])

    def test_provider_always_emits_fixed_k(self):
        provider = RacerDraftProvider(
            draft_token_num=8,
            ngram=3,
            topk=2,
            max_nodes=128,
            stats_enabled=True,
        )
        tokens, mask = provider.batch_get(
            req_ids=["r0"],
            batch_tokens=[[1, 2, 3, 1, 2, 4, 1, 2, 3]],
            total_lens=[9],
        )
        self.assertEqual(tokens.shape, (8,))
        self.assertEqual(mask.shape, (64,))
        stats = provider.consume_last_batch_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(
            stats[0]["padding_nodes"] + stats[0]["nodes_before_padding"],
            8,
        )


if __name__ == "__main__":
    unittest.main()
