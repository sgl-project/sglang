"""Pure token-space planning and the native input-logprob offset contract."""

import math
from dataclasses import dataclass

from .protocol import RawSystemOneError, TokenLogprob


def invalid_scores(message: str):
    return RawSystemOneError("incomplete_scores", message, 500)


@dataclass(frozen=True)
class TokenPlan:
    sequences: tuple[tuple[int, ...], ...]
    original_to_unique: tuple[int, ...]
    common_length: int


def plan_tokens(sequences: list[list[int]]) -> TokenPlan:
    unique = tuple(sorted({tuple(ids) for ids in sequences}, key=lambda x: (len(x), x)))
    lookup = {ids: i for i, ids in enumerate(unique)}
    common = 0
    for tokens in zip(*unique):
        if len(set(tokens)) != 1:
            break
        common += 1
    return TokenPlan(unique, tuple(lookup[tuple(x)] for x in sequences), common)


def partition_candidates(
    candidates: list[int],
    sequences: tuple[tuple[int, ...], ...],
    max_candidates: int,
    target_tokens: int,
    hard_tokens: int,
) -> list[list[int]]:
    """Count full inputs, including cached tokens; oversized inputs are singletons."""
    batches, batch, tokens = [], [], 0
    for candidate in candidates:
        size = len(sequences[candidate])
        if size > hard_tokens:
            raise RawSystemOneError(
                "token_budget_exceeded",
                "A candidate exceeds the admission token limit.",
            )
        if batch and (len(batch) == max_candidates or tokens + size > target_tokens):
            batches.append(batch)
            batch, tokens = [], 0
        batch.append(candidate)
        tokens += size
        if tokens >= target_tokens:
            batches.append(batch)
            batch, tokens = [], 0
    if batch:
        batches.append(batch)
    return batches


def native_rows(
    ids: tuple[int, ...], start: int, rows: list
) -> tuple[TokenLogprob, ...]:
    """Map the *final, accumulated* native rows, never token-ID-search them.

    SchedulerLogprobResultProcessor returns ids[start:] with an initial None
    at absolute position start, then scores for start+1 onward. Chunked prefill
    is accumulated by the manager before its non-streaming batch yields.
    For a branch start=K-1, that None is a boundary placeholder; shared records
    already contain its real score. Only position zero is null in API output.
    """
    if not isinstance(rows, (list, tuple)) or len(rows) != len(ids) - start:
        raise invalid_scores("Native input logprobs do not cover the requested span.")
    mapped = []
    for offset, row in enumerate(rows):
        position = start + offset
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise invalid_scores("Malformed native input-logprob row.")
        value, token_id, _ = row
        if type(token_id) is not int or token_id != ids[position]:
            raise invalid_scores("Native token IDs do not match the supplied sequence.")
        if offset == 0:
            if value is not None:
                raise invalid_scores("Native span must begin with a null boundary row.")
        elif value is None:
            raise invalid_scores("A scoreable token is missing its log-probability.")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise invalid_scores("Malformed native token log-probability.")
        elif not math.isfinite(value):
            raise RawSystemOneError(
                "non_finite_scores", "Native token log-probability is not finite.", 500
            )
        mapped.append(TokenLogprob(position=position, token_id=token_id, logprob=value))
    return tuple(mapped)


def aggregate(
    ids: tuple[int, ...], records: tuple[TokenLogprob, ...]
) -> tuple[float, int]:
    if len(ids) < 2 or len(records) != len(ids):
        raise invalid_scores("Complete candidate scores are required.")
    for position, (token_id, record) in enumerate(zip(ids, records)):
        if record.position != position or record.token_id != token_id:
            raise invalid_scores("Candidate token positions or IDs do not match.")
        if position == 0:
            if record.logprob is not None:
                raise invalid_scores("The initial token must be unscored.")
        elif record.logprob is None or not math.isfinite(record.logprob):
            raise invalid_scores("Every non-initial token must have a finite score.")
    try:
        total = math.fsum(record.logprob for record in records[1:])
    except OverflowError:
        total = math.inf
    if not math.isfinite(total):
        raise RawSystemOneError(
            "non_finite_scores", "Candidate sum is not finite.", 500
        )
    return total, len(ids) - 1
