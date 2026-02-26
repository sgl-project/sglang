import sys

import pytest

from sglang.srt.debug_utils.comparator.aligner.token_aligner.planner import (
    _match_sequences,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.seq_info_builder import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    SeqId,
    SGLangSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


class TestBuildTokenIndexSGLangThd:
    """Tests for SGLang thd token index building."""

    def test_single_step_prefill(self):
        """Single prefill step with two sequences."""
        side_aux = TokenAlignerGlobalAux(
            step_auxs={
                0: TokenAlignerStepAux(
                    input_ids=[10, 20, 30, 40, 50],
                    positions=[0, 1, 2, 0, 1],
                    seq_lens=[3, 2],
                    seq_ids=[SGLangSeqId(rid="A"), SGLangSeqId(rid="B")],
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 2

        seq_a = index.sequences[SGLangSeqId(rid="A")]
        assert seq_a.input_ids == [10, 20, 30]
        assert seq_a.positions == [0, 1, 2]
        assert seq_a.locator.token_index_in_step == [0, 1, 2]

        seq_b = index.sequences[SGLangSeqId(rid="B")]
        assert seq_b.input_ids == [40, 50]
        assert seq_b.positions == [0, 1]
        assert seq_b.locator.token_index_in_step == [3, 4]


class TestBuildTokenIndexMegatronThd:
    """Tests for Megatron thd token index building."""

    def test_single_step_two_sequences(self):
        """Single step with two sequences in thd layout."""
        side_aux = TokenAlignerGlobalAux(
            step_auxs={
                0: TokenAlignerStepAux(
                    input_ids=[10, 20, 30, 40, 50],
                    positions=[0, 1, 2, 0, 1],
                    seq_lens=[3, 2],
                    seq_ids=[
                        PositionalSeqId(step=0, seq_index=0),
                        PositionalSeqId(step=0, seq_index=1),
                    ],
                ),
            },
            framework="megatron",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[PositionalSeqId(step=0, seq_index=0)]
        assert seq0.input_ids == [10, 20, 30]
        assert seq0.positions == [0, 1, 2]
        assert seq0.locator.token_index_in_step == [0, 1, 2]

        seq1 = index.sequences[PositionalSeqId(step=0, seq_index=1)]
        assert seq1.input_ids == [40, 50]
        assert seq1.positions == [0, 1]
        assert seq1.locator.token_index_in_step == [3, 4]


class TestMatchSequences:
    """Tests for _match_sequences: for each y, find matching x."""

    def test_exact_match_simple(self):
        """Identical input_ids on both sides → all matched."""
        matched = _match_seqs(
            x={0: (10, 20, 30), 1: (40, 50)},
            y={0: (10, 20, 30), 1: (40, 50)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(0), S(0)), (S(1), S(1))}

    def test_exact_match_different_order(self):
        """Sequences in different order still match by content."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (40, 50)},
            y={0: (40, 50), 1: (10, 20)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(1), S(0)), (S(0), S(1))}

    def test_exact_match_different_seq_ids(self):
        """Seq IDs don't need to correspond — matching is by content."""
        matched = _match_seqs(
            x={5: (10, 20), 9: (30, 40)},
            y={2: (30, 40), 7: (10, 20)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(9), S(2)), (S(5), S(7))}

    def test_no_match(self):
        """Completely different input_ids → no matches."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (99, 88)},
        )
        assert matched == []

    def test_empty_sides(self):
        """Empty x or y → no matches."""
        assert _match_seqs(x={}, y={0: (10,)}) == []
        assert _match_seqs(x={0: (10,)}, y={}) == []
        assert _match_seqs(x={}, y={}) == []

    def test_x_has_more_sequences(self):
        """Extra x sequences are ignored (no y needs them)."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (30, 40), 2: (50, 60)},
            y={0: (30, 40)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(1), S(0))}

    def test_y_has_more_sequences(self):
        """Extra y sequences remain unmatched."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (10, 20), 1: (30, 40), 2: (50, 60)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(0), S(0))}

    def test_one_x_not_reused(self):
        """Each x can only be claimed once, even if multiple y want it."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (10, 20), 1: (10, 20)},
        )
        assert len(matched) == 1

    def test_ambiguous_all_matched(self):
        """Multiple identical sequences on both sides → all paired (greedy 1:1)."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (10, 20), 2: (10, 20)},
            y={0: (10, 20), 1: (10, 20), 2: (10, 20)},
        )
        S = _int_to_seq_id
        assert len(matched) == 3
        x_ids = {m[0] for m in matched}
        y_ids = {m[1] for m in matched}
        assert x_ids == {S(0), S(1), S(2)}
        assert y_ids == {S(0), S(1), S(2)}

    def test_prefix_x_shorter(self):
        """x has fewer tokens (prefix of y) → prefix match."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (10, 20, 30)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(0), S(0))}

    def test_prefix_y_shorter(self):
        """y has fewer tokens (prefix of x) → prefix match."""
        matched = _match_seqs(
            x={0: (10, 20, 30)},
            y={0: (10, 20)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(0), S(0))}

    def test_prefix_picks_longest(self):
        """Among multiple prefix candidates, picks the one with longest overlap."""
        matched = _match_seqs(
            x={0: (10,), 1: (10, 20, 30)},
            y={0: (10, 20, 30, 40)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(1), S(0))}

    def test_exact_preferred_over_prefix(self):
        """Exact match is tried first, even if a longer prefix candidate exists."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (10, 20, 30)},
            y={0: (10, 20)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(0), S(0))}

    def test_prefix_fallback_after_exact(self):
        """Exact matches consume sequences, remaining use prefix match."""
        matched = _match_seqs(
            x={0: (10, 20, 30), 1: (40, 50)},
            y={0: (10, 20, 30), 1: (40, 50, 60)},
        )
        S = _int_to_seq_id
        assert len(matched) == 2
        matched_set = _matched_ids(matched)
        assert (S(0), S(0)) in matched_set
        assert (S(1), S(1)) in matched_set

    def test_single_token_sequences(self):
        """Single-token sequences can match."""
        matched = _match_seqs(
            x={0: (42,)},
            y={0: (42,)},
        )
        S = _int_to_seq_id
        assert _matched_ids(matched) == {(S(0), S(0))}

    def test_no_partial_overlap_without_prefix(self):
        """Overlapping content that isn't a prefix → no match."""
        matched = _match_seqs(
            x={0: (10, 20, 30)},
            y={0: (20, 30, 40)},
        )
        assert matched == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _int_to_seq_id(k: int) -> SeqId:
    """Convert an int key to a SeqId for test convenience."""
    return SGLangSeqId(rid=str(k))


def _make_index(
    *,
    sequences: dict[int, tuple[int, ...]],
    layout: str = "thd",
) -> TokenAlignerSeqsInfo:
    """Create a TokenAlignerSeqsInfo from simplified input_ids-only specification."""
    records: dict[SeqId, TokenAlignerSeqInfo] = {}
    for k, input_ids in sequences.items():
        num_tokens = len(input_ids)
        records[_int_to_seq_id(k)] = TokenAlignerSeqInfo(
            input_ids=list(input_ids),
            positions=list(range(num_tokens)),
            locator=TokenLocator(
                token_index_in_step=list(range(num_tokens)),
            ),
        )
    return TokenAlignerSeqsInfo(sequences=records, layout=layout)


def _make_seq_info_dict(
    sequences: dict[int, tuple[int, ...]],
) -> dict[SeqId, TokenAlignerSeqInfo]:
    """Create a dict of TokenAlignerSeqInfo from {int_key: input_ids_tuple}."""
    result: dict[SeqId, TokenAlignerSeqInfo] = {}
    for k, input_ids in sequences.items():
        num_tokens = len(input_ids)
        result[_int_to_seq_id(k)] = TokenAlignerSeqInfo(
            input_ids=list(input_ids),
            positions=list(range(num_tokens)),
            locator=TokenLocator(
                token_index_in_step=list(range(num_tokens)),
            ),
        )
    return result


def _match_seqs(
    *,
    x: dict[int, tuple[int, ...]],
    y: dict[int, tuple[int, ...]],
) -> list[tuple[SeqId, SeqId]]:
    """Shorthand: build SeqInfo dicts and call _match_sequences."""
    return _match_sequences(
        seqs=Pair(x=_make_seq_info_dict(x), y=_make_seq_info_dict(y))
    )


def _matched_ids(matched: list[tuple[SeqId, SeqId]]) -> set[tuple[SeqId, SeqId]]:
    """Convert matched pairs list to set for order-independent comparison."""
    return set(matched)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
