"""Enhanced multi-turn KL divergence test helpers."""

from __future__ import annotations

from typing import Callable

from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
    get_input_ids,
)

__all__ = [
    # Cache assertion callbacks
    "default_prefill_cache_assert",
    "default_decode_cache_assert",
    "make_mamba_prefill_assert",
    "make_mamba_decode_assert",
    # Enhanced test helpers
    "test_input_output_logprobs_match_helper",
    "test_input_output_logprobs_match_prefill_cache_hit_helper",
    "test_input_output_logprobs_match_decode_cache_hit_helper",
    # Internal helpers (for custom inline tests)
    "_replay_and_compare_kl",
    # Re-exports from kl_test_utils
    "get_input_ids",
    "_generate",
    "_flush_cache",
    "_extract_output_logprobs",
]


# =============================================================================
# Cache assertion callbacks
# =============================================================================
# Prefill signature: (result, prefix_len, label) -> None
# Decode  signature: (result, history_len, output_len, label) -> None


def default_prefill_cache_assert(result: dict, prefix_len: int, label: str):
    """Standard radix cache: cached_tokens == prefix_len."""
    actual = result["meta_info"]["cached_tokens"]
    assert (
        actual == prefix_len
    ), f"{label}: expected cached_tokens={prefix_len}, got {actual}"


def default_decode_cache_assert(
    result: dict, history_len: int, output_len: int, label: str
):
    """Standard radix cache: cached_tokens == history_len + output_len."""
    expected = history_len + output_len
    actual = result["meta_info"]["cached_tokens"]
    assert (
        actual == expected
    ), f"{label}: expected cached_tokens={expected}, got {actual}"


def make_mamba_prefill_assert(chunk_size: int = 64) -> Callable:
    """Mamba: cached_tokens in [rounded_down - chunk_size, rounded_down]."""

    def _check(result: dict, prefix_len: int, label: str):
        actual = result["meta_info"]["cached_tokens"]
        upper = (prefix_len // chunk_size) * chunk_size
        lower = max(0, upper - chunk_size)
        assert (
            lower <= actual <= upper
        ), f"{label}: expected cached_tokens in [{lower}, {upper}], got {actual}"

    return _check


def make_mamba_decode_assert(track_interval: int = 16) -> Callable:
    """Mamba: cached_tokens = floor((history+output-1)/interval)*interval."""

    def _check(result: dict, history_len: int, output_len: int, label: str):
        actual = result["meta_info"]["cached_tokens"]
        if output_len <= 0:
            expected = history_len
        else:
            expected = (
                (history_len + output_len - 1) // track_interval
            ) * track_interval
        assert (
            actual >= expected
        ), f"{label}: expected cached_tokens={expected}, got {actual}"

    return _check


# =============================================================================
# Internal helpers
# =============================================================================


def _replay_and_compare_kl(
    base_url: str,
    model_name: str,
    kl_threshold: float,
    replay_input_ids: list[list[int]],
    output_logprobs: list[list[float]],
    label: str,
    batch_size: int = 1,
):
    """Flush cache, run replay prefill in batches, compare KL divergence."""
    all_input_logprobs = []
    for start in range(0, len(replay_input_ids), batch_size):
        end = start + batch_size
        all_input_logprobs.extend(
            _get_input_logprobs(
                base_url,
                replay_input_ids[start:end],
                output_logprobs[start:end],
            )
        )
    acc = {model_name: {"kl_div": kl_threshold}}
    compare_kl_divergence(all_input_logprobs, output_logprobs, acc, model_name, label)


def _interleave_order(n: int, branches_per_group: int) -> list[int] | None:
    """Build interleaved submission order for branch stress testing.

    Given n items grouped into groups of branches_per_group, returns indices
    that interleave branches across groups: [g0b0, g1b0, ..., g0b1, g1b1, ...].

    Returns None if no interleaving is needed.
    """
    if branches_per_group <= 0 or branches_per_group >= n:
        return None
    num_groups = n // branches_per_group
    order = [
        g * branches_per_group + b
        for b in range(branches_per_group)
        for g in range(num_groups)
    ]
    # Append remainder indices not covered by complete groups
    for i in range(num_groups * branches_per_group, n):
        order.append(i)
    return order


def _generate_maybe_interleaved(base_url, inputs, max_new_tokens, order=None):
    """Generate with optional interleaved submission order.

    Submits inputs reordered by ``order``, then maps results back to the
    original order so the caller always sees results[i] corresponds to
    inputs[i].
    """
    if order is None:
        return _generate(base_url, inputs, max_new_tokens, return_logprob=True)
    ordered = [inputs[i] for i in order]
    results = _generate(base_url, ordered, max_new_tokens, return_logprob=True)
    unordered = [None] * len(results)
    for idx, orig in enumerate(order):
        unordered[orig] = results[idx]
    return unordered


# =============================================================================
# Helper 1: test_input_output_logprobs_match_helper
# =============================================================================


def test_input_output_logprobs_match_helper(
    base_url: str,
    model_name: str,
    kl_threshold: float,
    input_ids: list[list[int]],
    *,
    label: str = "logprobs_match",
    max_new_tokens: int = 256,
    # --- Multi-turn ---
    # turn_suffixes[t][i] = suffix tokens for sample i at turn t+1
    turn_suffixes: list[list[list[int]]] | None = None,
    # --- Cache assertion (for turns > 0) ---
    assert_decode_cached_tokens: Callable | None = None,
    replay_batch_size: int = 1,
):
    """Verify decode logprobs match prefill replay.

    Single-turn (turn_suffixes=None):
      flush -> generate(input_ids) -> replay -> KL

    Multi-turn (turn_suffixes provided):
      flush -> generate turn 0 ->
      for t in range(len(turn_suffixes)):
        input = accumulated + output + suffix[t] -> generate ->
        assert_decode_cached_tokens (optional) ->
      replay last turn -> KL

    Multi-branch: caller passes input_ids where multiple entries share
    a prefix.
    """
    n = len(input_ids)
    num_turns = 1 + (len(turn_suffixes) if turn_suffixes else 0)
    print(f"[{label}] {n} samples, {num_turns} turns, max_new_tokens={max_new_tokens}")

    _flush_cache(base_url)

    current_input = list(input_ids)
    last_outputs = None
    prev_input_lens = [0] * n
    prev_output_lens = [0] * n

    for turn in range(num_turns):
        if turn > 0:
            suffixes = turn_suffixes[turn - 1]
            current_input = [
                current_input[i] + last_outputs[i] + suffixes[i] for i in range(n)
            ]

        results = _generate(
            base_url, current_input, max_new_tokens, return_logprob=True
        )
        assert len(results) == n

        if turn > 0 and assert_decode_cached_tokens:
            for i, result in enumerate(results):
                assert_decode_cached_tokens(
                    result,
                    prev_input_lens[i],
                    prev_output_lens[i],
                    f"{label}[turn{turn}][{i}]",
                )

        last_outputs = [r["output_ids"] for r in results]
        prev_input_lens = [len(current_input[i]) for i in range(n)]
        prev_output_lens = [len(last_outputs[i]) for i in range(n)]

    # Replay last turn
    replay_ids = [current_input[i] + results[i]["output_ids"] for i in range(n)]
    output_lps = [_extract_output_logprobs(r) for r in results]

    _replay_and_compare_kl(
        base_url,
        model_name,
        kl_threshold,
        replay_ids,
        output_lps,
        label=label,
        batch_size=replay_batch_size,
    )


# =============================================================================
# Helper 2: test_input_output_logprobs_match_prefill_cache_hit_helper
# =============================================================================


def test_input_output_logprobs_match_prefill_cache_hit_helper(
    base_url: str,
    model_name: str,
    kl_threshold: float,
    input_ids: list[list[int]] | None = None,
    *,
    # --- Multi-branch: explicit prefix/full split ---
    prefix_input_ids: list[list[int]] | None = None,
    full_input_ids: list[list[int]] | None = None,
    label: str = "prefill_cache_hit",
    max_new_tokens: int = 256,
    # --- Multi-turn: additional turns after the cache-hit generation ---
    turn_suffixes: list[list[list[int]]] | None = None,
    # --- Cache assertions ---
    assert_prefill_cached_tokens: Callable | None = None,  # turn 0
    assert_decode_cached_tokens: Callable | None = None,  # turns > 0
    # --- Interleaving for branch stress ---
    branches_per_group: int = 0,
    replay_batch_size: int = 1,
):
    """Verify logprobs when prefill cache is hit.

    Original (input_ids only, backward compat):
      flush -> seed(input_ids) -> generate(input_ids, cache hit) -> replay -> KL

    Multi-branch (prefix_input_ids + full_input_ids):
      flush -> seed(prefixes) -> generate(fulls, prefix cache hit) ->
      assert_prefill_cached_tokens -> replay -> KL

    Multi-turn (+ turn_suffixes):
      ... after prefill cache-hit turn, additional turns:
      input = accumulated + output + suffix -> generate ->
      assert_decode_cached_tokens -> replay last turn -> KL

    Interleaving (branches_per_group > 0):
      Reorders submission for decode-cache-hit turns to interleave branches
      across groups, stressing the radix tree with competing branches.
    """
    # Resolve inputs: backward compat with input_ids-only
    if input_ids is not None and prefix_input_ids is None:
        prefix_input_ids = input_ids
        full_input_ids = input_ids
    assert prefix_input_ids is not None and full_input_ids is not None
    assert len(prefix_input_ids) == len(full_input_ids)

    if assert_prefill_cached_tokens is None:
        assert_prefill_cached_tokens = default_prefill_cache_assert

    n = len(full_input_ids)
    num_turns = 1 + (len(turn_suffixes) if turn_suffixes else 0)
    order = _interleave_order(n, branches_per_group)
    print(f"[{label}] {n} samples, {num_turns} turns, max_new_tokens={max_new_tokens}")

    # Seed cache with prefixes
    _flush_cache(base_url)
    _generate(base_url, prefix_input_ids, max_new_tokens=0)

    # Turn 0: prefill cache hit (NOT interleaved, matching original behavior)
    results = _generate(base_url, full_input_ids, max_new_tokens, return_logprob=True)
    assert len(results) == n

    for i, result in enumerate(results):
        assert_prefill_cached_tokens(
            result, len(prefix_input_ids[i]), f"{label}[turn0][{i}]"
        )

    current_input = list(full_input_ids)
    last_outputs = [r["output_ids"] for r in results]
    prev_input_lens = [len(full_input_ids[i]) for i in range(n)]
    prev_output_lens = [len(last_outputs[i]) for i in range(n)]

    # Additional turns: decode cache hits (interleaved if order is set)
    if turn_suffixes:
        if assert_decode_cached_tokens is None:
            assert_decode_cached_tokens = default_decode_cache_assert

        for t, suffixes in enumerate(turn_suffixes):
            current_input = [
                current_input[i] + last_outputs[i] + suffixes[i] for i in range(n)
            ]
            results = _generate_maybe_interleaved(
                base_url, current_input, max_new_tokens, order
            )
            assert len(results) == n

            for i, result in enumerate(results):
                assert_decode_cached_tokens(
                    result,
                    prev_input_lens[i],
                    prev_output_lens[i],
                    f"{label}[turn{t + 1}][{i}]",
                )

            last_outputs = [r["output_ids"] for r in results]
            prev_input_lens = [len(current_input[i]) for i in range(n)]
            prev_output_lens = [len(last_outputs[i]) for i in range(n)]

    # Replay last turn
    replay_ids = [current_input[i] + results[i]["output_ids"] for i in range(n)]
    output_lps = [_extract_output_logprobs(r) for r in results]

    _replay_and_compare_kl(
        base_url,
        model_name,
        kl_threshold,
        replay_ids,
        output_lps,
        label=label,
        batch_size=replay_batch_size,
    )


# =============================================================================
# Helper 3: test_input_output_logprobs_match_decode_cache_hit_helper
# =============================================================================


def test_input_output_logprobs_match_decode_cache_hit_helper(
    base_url: str,
    model_name: str,
    kl_threshold: float,
    first_turn_input_ids: list[list[int]],
    *,
    # --- Multi-turn ---
    # turn_suffixes[t][i] = suffix for sample i at turn t+2
    turn_suffixes: list[list[list[int]]],
    label: str = "decode_cache_hit",
    max_new_tokens: int = 256,
    # --- Cache assertion ---
    assert_decode_cached_tokens: Callable | None = None,
    # --- Interleaving ---
    branches_per_group: int = 0,
    replay_batch_size: int = 1,
):
    """Verify logprobs when decode cache is hit.

    2-turn (turn_suffixes has 1 entry):
      flush -> generate turn 1 ->
      turn 2: input = turn1 + output + suffix -> generate ->
      assert_decode_cached_tokens -> replay -> KL

    Multi-turn (turn_suffixes has N entries):
      flush -> generate turn 1 ->
      for each turn t: input = accumulated + output + suffix[t] -> generate ->
      assert_decode_cached_tokens -> replay last turn -> KL

    Multi-branch: caller duplicates first_turn_input_ids entries and provides
    different suffixes per branch. Use branches_per_group for interleaved
    submission to stress the radix tree.
    """
    assert (
        len(turn_suffixes) >= 1
    ), "turn_suffixes must have at least 1 entry (for turn 2)"
    if assert_decode_cached_tokens is None:
        assert_decode_cached_tokens = default_decode_cache_assert

    n = len(first_turn_input_ids)
    num_turns = 1 + len(turn_suffixes)
    order = _interleave_order(n, branches_per_group)
    print(f"[{label}] {n} samples, {num_turns} turns, max_new_tokens={max_new_tokens}")

    # Turn 1: populate cache, no assertion, no interleaving
    _flush_cache(base_url)
    results = _generate(
        base_url, first_turn_input_ids, max_new_tokens, return_logprob=True
    )
    assert len(results) == n

    current_input = list(first_turn_input_ids)
    last_outputs = [r["output_ids"] for r in results]
    prev_input_lens = [len(first_turn_input_ids[i]) for i in range(n)]
    prev_output_lens = [len(last_outputs[i]) for i in range(n)]

    # Turns 2..N: decode cache hits (interleaved if order is set)
    for t, suffixes in enumerate(turn_suffixes):
        current_input = [
            current_input[i] + last_outputs[i] + suffixes[i] for i in range(n)
        ]
        results = _generate_maybe_interleaved(
            base_url, current_input, max_new_tokens, order
        )
        assert len(results) == n

        for i, result in enumerate(results):
            assert_decode_cached_tokens(
                result,
                prev_input_lens[i],
                prev_output_lens[i],
                f"{label}[turn{t + 1}][{i}]",
            )

        last_outputs = [r["output_ids"] for r in results]
        prev_input_lens = [len(current_input[i]) for i in range(n)]
        prev_output_lens = [len(last_outputs[i]) for i in range(n)]

    # Replay last turn
    replay_ids = [current_input[i] + results[i]["output_ids"] for i in range(n)]
    output_lps = [_extract_output_logprobs(r) for r in results]

    _replay_and_compare_kl(
        base_url,
        model_name,
        kl_threshold,
        replay_ids,
        output_lps,
        label=label,
        batch_size=replay_batch_size,
    )
