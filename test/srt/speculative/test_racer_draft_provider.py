import numpy as np

from sglang.srt.speculative.racer.automaton import RacerAutomaton
from sglang.srt.speculative.racer.draft_provider import RacerDraftProvider


def test_racer_automaton_returns_fixed_budget_tree():
    automaton = RacerAutomaton(ngram=3, topk=2, max_nodes=128)
    automaton.sync_history([1, 2, 3, 1, 2, 4, 1, 2])
    automaton.update_logits([3, 4], [[5, 6], [7, 8]])

    tokens, mask = automaton.retrieve(root_token=3, max_num_draft=8)

    assert len(tokens) == 8
    assert len(mask) == 8
    assert all(len(row) == 8 for row in mask)
    assert tokens[0] == 3
    assert mask[0][0]


def test_racer_provider_batches_fixed_width_trees():
    provider = RacerDraftProvider(
        draft_token_num=4,
        ngram=3,
        topk=2,
        max_nodes=128,
    )

    tokens, mask = provider.batch_get(
        req_ids=["r0", "r1"],
        batch_tokens=[
            [1, 2, 3, 1, 2, 4],
            [7, 8, 9, 7, 8, 10],
        ],
        total_lens=[6, 6],
    )

    assert tokens.shape == (8,)
    assert mask.shape == (32,)
    assert tokens.dtype == np.int64
    assert mask.dtype == np.bool_


def test_copy_logits_feed_future_tree():
    provider = RacerDraftProvider(
        draft_token_num=4,
        ngram=3,
        topk=2,
        max_nodes=128,
    )

    tokens, _ = provider.batch_get(
        req_ids=["r0"],
        batch_tokens=[[1, 2, 3, 4]],
        total_lens=[4],
    )
    provider.update_logits(
        ["r0"],
        tokens,
        np.asarray(
            [
                [11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
            ],
            dtype=np.int64,
        ),
    )

    next_tokens, next_mask = provider.batch_get(
        req_ids=["r0"],
        batch_tokens=[[1, 2, 3, 4, int(tokens[0])]],
        total_lens=[5],
    )

    assert next_tokens.shape == (4,)
    assert next_mask.shape == (16,)
