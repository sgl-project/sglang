"""Backend-local contracts for linear MLX speculative verification.

The first Gemma 4 MTP implementation deliberately proposes a single linear
token rather than a tree.  Keeping the acceptance policy here makes it easy to
test without importing MLX, model weights, or scheduler state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MlxVerifySegment:
    """One request's fixed-width draft row and target verification results.

    ``draft_tokens`` contains ``draft_slot_width`` entries.  Only the explicit
    ``valid_draft_count`` prefix is a draft; every remaining entry must be the
    ``-1`` padding sentinel.  ``target_token_ids`` has one row for the pending
    root query plus one row for each valid draft.
    """

    request_id: str
    draft_tokens: tuple[int, ...]
    valid_draft_count: int
    target_token_ids: tuple[int, ...]
    invalid_draft_count: int | None = None
    verification_query_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "draft_tokens", tuple(self.draft_tokens))
        object.__setattr__(self, "target_token_ids", tuple(self.target_token_ids))

        width = len(self.draft_tokens)
        valid = self.valid_draft_count
        if valid < 0:
            raise ValueError("valid_draft_count must be non-negative")
        if valid > width:
            raise ValueError(
                f"valid_draft_count {valid} exceeds draft row width {width}"
            )

        valid_prefix = self.draft_tokens[:valid]
        if any(token < 0 for token in valid_prefix):
            raise ValueError(
                "negative draft token found inside the declared valid prefix"
            )
        padded_suffix = self.draft_tokens[valid:]
        if any(token != -1 for token in padded_suffix):
            raise ValueError("every unused draft slot must contain exactly -1")

        if self.invalid_draft_count is not None:
            if self.invalid_draft_count < 0:
                raise ValueError("invalid_draft_count must be non-negative")
            if valid + self.invalid_draft_count != width:
                raise ValueError(
                    "valid_draft_count + invalid_draft_count must equal the "
                    "fixed draft row width"
                )

        expected_queries = valid + 1
        if len(self.target_token_ids) != expected_queries:
            raise ValueError(
                "target verification rows must equal valid_draft_count + 1: "
                f"expected {expected_queries}, got {len(self.target_token_ids)}"
            )
        if self.verification_query_count is not None and (
            self.verification_query_count != expected_queries
        ):
            raise ValueError(
                "verification_query_count must equal valid_draft_count + 1"
            )
        if any(token < 0 for token in self.target_token_ids):
            raise ValueError("target verification token IDs must be non-negative")

    @property
    def draft_slot_width(self) -> int:
        return len(self.draft_tokens)

    @property
    def valid_draft_tokens(self) -> tuple[int, ...]:
        """Return only the explicit valid prefix; padding never escapes here."""

        return self.draft_tokens[: self.valid_draft_count]


@dataclass(frozen=True)
class MlxVerifyDecision:
    """Lossless greedy acceptance decision for one request."""

    request_id: str
    emitted_token_ids: tuple[int, ...]
    accepted_draft_count: int
    committed_query_count: int
    seed_hidden_row_index: int

    @property
    def accept_len(self) -> int:
        """Scheduler-visible emitted width (accepted drafts plus one target)."""

        return len(self.emitted_token_ids)


def verify_greedy_segment(segment: MlxVerifySegment) -> MlxVerifyDecision:
    """Accept the longest exact draft prefix and append the target continuation."""

    drafts = segment.valid_draft_tokens
    targets = segment.target_token_ids
    accepted = 0
    for draft, target in zip(drafts, targets):
        if draft != target:
            break
        accepted += 1

    emitted = drafts[:accepted] + (targets[accepted],)
    # Query row zero is the pending previously-emitted token.  Each accepted
    # draft makes one additional verification query part of the visible cache.
    committed_queries = accepted + 1
    return MlxVerifyDecision(
        request_id=segment.request_id,
        emitted_token_ids=emitted,
        accepted_draft_count=accepted,
        committed_query_count=committed_queries,
        # The newly sampled mismatch/bonus has not been queried.  The assistant
        # seed therefore uses the last committed query row.
        seed_hidden_row_index=committed_queries - 1,
    )


def verify_greedy_segments(
    segments: Iterable[MlxVerifySegment],
    *,
    expected_request_ids: Sequence[str] | None = None,
) -> tuple[MlxVerifyDecision, ...]:
    """Verify an ordered batch while rejecting ambiguous request alignment."""

    materialized = tuple(segments)
    request_ids = tuple(segment.request_id for segment in materialized)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("verification segments contain duplicate request IDs")
    if expected_request_ids is not None and request_ids != tuple(expected_request_ids):
        raise ValueError(
            "verification request ordering does not match target row ordering: "
            f"segments={request_ids}, target={tuple(expected_request_ids)}"
        )
    return tuple(verify_greedy_segment(segment) for segment in materialized)


def build_linear_verify_queries(
    root_token: int, segment: MlxVerifySegment
) -> tuple[int, ...]:
    """Build target queries after validated padding has been removed."""

    if root_token < 0:
        raise ValueError("root verification token must be non-negative")
    return (root_token,) + segment.valid_draft_tokens
