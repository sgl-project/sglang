from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple, Optional

from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    SeqId,
    TokenAlignerPlan,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def compute_token_aligner_plan(
    seqs_info_pair: Pair[TokenAlignerSeqsInfo],
) -> TokenAlignerPlan:
    """Compute a token alignment plan from two side token seqs_info_pair."""
    matched_pairs: list[tuple[SeqId, SeqId]] = _match_sequences(
        seqs=Pair(x=seqs_info_pair.x.sequences, y=seqs_info_pair.y.sequences)
    )

    _empty = TokenLocator(steps=[], token_index_in_step=[])
    locator_x: TokenLocator = _empty
    locator_y: TokenLocator = _empty

    for seq_id_x, seq_id_y in matched_pairs:
        rec: Pair[TokenAlignerSeqInfo] = Pair(
            x=seqs_info_pair.x.sequences[seq_id_x],
            y=seqs_info_pair.y.sequences[seq_id_y],
        )

        # positions is validated to be [0, 1, ..., N-1], so position == index
        # and the common range is simply [0, min(len_x, len_y)).
        common_len: int = min(len(rec.x.positions), len(rec.y.positions))

        x_ids = rec.x.input_ids[:common_len]
        y_ids = rec.y.input_ids[:common_len]
        assert x_ids == y_ids, f"{seq_id_x=} {seq_id_y=} {x_ids=} {y_ids=}"

        locator_x = locator_x + TokenLocator(
            steps=rec.x.locator.steps[:common_len],
            token_index_in_step=rec.x.locator.token_index_in_step[:common_len],
        )
        locator_y = locator_y + TokenLocator(
            steps=rec.y.locator.steps[:common_len],
            token_index_in_step=rec.y.locator.token_index_in_step[:common_len],
        )

    return TokenAlignerPlan(
        locators=Pair(x=locator_x, y=locator_y),
        layouts=seqs_info_pair.map(lambda s: s.layout),
    )


# -------------------- Sequence matcher --------------------


def _match_sequences(
    seqs: Pair[dict[SeqId, TokenAlignerSeqInfo]],
) -> list[tuple[SeqId, SeqId]]:
    """For each y (target) sequence, find a matching x (baseline) sequence.

    Two-pass: exact match first, then prefix match for remaining.
    """
    x_lookup: dict[tuple[int, ...], list[SeqId]] = defaultdict(list)
    for seq_id, rec in seqs.x.items():
        x_lookup[tuple(rec.input_ids)].append(seq_id)

    claimed_x_ids: set[SeqId] = set()
    matched_seq_id_pairs: list[tuple[SeqId, SeqId]] = []

    for seq_id_y in sorted(seqs.y.keys()):
        seq_y: TokenAlignerSeqInfo = seqs.y[seq_id_y]

        matched_x: Optional[SeqId] = _find_matching_x_exact(
            seq_y=seq_y, x_lookup=x_lookup, claimed_x_ids=claimed_x_ids
        )
        if matched_x is None:
            matched_x = _find_matching_x_prefix(
                seq_y=seq_y, x_seqs=seqs.x, claimed_x_ids=claimed_x_ids
            )

        if matched_x is not None:
            matched_seq_id_pairs.append((matched_x, seq_id_y))
            claimed_x_ids.add(matched_x)

    return matched_seq_id_pairs


def _find_matching_x_exact(
    *,
    seq_y: TokenAlignerSeqInfo,
    x_lookup: dict[tuple[int, ...], list[SeqId]],
    claimed_x_ids: set[SeqId],
) -> Optional[SeqId]:
    """Find an x sequence with identical input_ids."""
    ids_y_key: tuple[int, ...] = tuple(seq_y.input_ids)
    candidates: list[SeqId] = x_lookup.get(ids_y_key, [])
    for candidate in candidates:
        if candidate not in claimed_x_ids:
            return candidate
    return None


class _PrefixCandidate(NamedTuple):
    seq_id_x: SeqId
    overlap_len: int


def _find_matching_x_prefix(
    *,
    seq_y: TokenAlignerSeqInfo,
    x_seqs: dict[SeqId, TokenAlignerSeqInfo],
    claimed_x_ids: set[SeqId],
) -> Optional[SeqId]:
    """Find the x sequence with the longest prefix relationship to y."""
    ids_y: list[int] = seq_y.input_ids
    candidates: list[_PrefixCandidate] = [
        _PrefixCandidate(
            seq_id_x=seq_id_x, overlap_len=min(len(seq_x.input_ids), len(ids_y))
        )
        for seq_id_x, seq_x in x_seqs.items()
        if seq_id_x not in claimed_x_ids and _is_prefix_pair(seq_x.input_ids, ids_y)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.overlap_len).seq_id_x


def _is_prefix_pair(a: list[int], b: list[int]) -> bool:
    """True if a is a prefix of b, or b is a prefix of a."""
    shorter_len: int = min(len(a), len(b))
    return a[:shorter_len] == b[:shorter_len]
