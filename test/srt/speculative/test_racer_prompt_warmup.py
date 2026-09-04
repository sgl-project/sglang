import numpy as np

from sglang.srt.speculative.racer.draft_provider import RacerDraftProvider


def test_prompt_warmup_splits_requests_and_latest_token_row_wins():
    provider = RacerDraftProvider(
        draft_token_num=4,
        ngram=2,
        topk=3,
        max_nodes=128,
        stats_enabled=False,
    )

    # req-a has token 11 twice; the later prompt position must overwrite the
    # earlier TokenBin row, matching the original C++ adj_matrix assignment.
    tokens = np.asarray([10, 11, 11, 20, 21], dtype=np.int64)
    topk = np.asarray(
        [
            [100, 101, 102],
            [110, 111, 112],
            [120, 121, 122],
            [200, 201, 202],
            [210, 211, 212],
        ],
        dtype=np.int64,
    )

    provider.seed_prompt_logits(
        ["req-a", "req-b"],
        tokens,
        topk,
        [3, 2],
    )

    state_a = provider._state("req-a")
    state_b = provider._state("req-b")

    assert state_a._token_bin_row(10) == [100, 101, 102]
    assert state_a._token_bin_row(11) == [120, 121, 122]
    assert state_b._token_bin_row(20) == [200, 201, 202]
    assert state_b._token_bin_row(21) == [210, 211, 212]
