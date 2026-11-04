# SPDX-License-Identifier: Apache-2.0
"""Byte model for runtime memory that lives outside the KV pool and graph memory.

Post-capture KV sizing measures everything allocated by the end of graph capture.
What it cannot see are the per-step allocations of serving: the sampling state the
running batch keeps across steps, and the eager tails of the decode and prefill
forwards. The reserve covers the worst legal combination of them, so it is bounded by
the running batch size and the prefill chunk size rather than by the request mix.
"""

from __future__ import annotations

from typing import Optional, Sequence

import msgspec

from sglang.srt.environ import envs

_FP32_BYTES = 4

# fp32 [num_requests, vocab] rows the running batch keeps across steps: 4 penalizer
# states (frequency, presence, repetition, min_new_tokens), 6 penalty accumulators
# (additive and scaling sums of up to three forward copies the overlap scheduler
# keeps alive) and 1 logit_bias.
_PERSISTENT_FP32_ROWS = 11

# fp32 [num_requests, vocab] rows a decode step adds: eager-decode logits plus the
# two min-p renormalizations, or log-probs.
_DECODE_FP32_ROWS = 3

# Allocator slack and lazily grown workspaces; arbitrary.
_TRANSIENT_SLACK_BYTES = 1 << 30

# Arbitrary: per-request reserves may claim at most 1/10 of the KV budget before the
# running cap backs off below the decode graph ceiling.
_KV_BUDGET_SHARE_DENOMINATOR = 10


class TransientReserve(msgspec.Struct, frozen=True, kw_only=True):
    persistent_bytes: int
    decode_bytes: int
    prefill_bytes: int
    fragmentation_bytes: int

    @property
    def total_bytes(self) -> int:
        # Prefill and decode forwards run one at a time, but under interleaving the
        # caching allocator splits the blocks one tail freed for the other's
        # tensors, so both tails stay reserved beside the batch's sampling state.
        return (
            self.persistent_bytes
            + self.decode_bytes
            + self.prefill_bytes
            + self.fragmentation_bytes
            + _TRANSIENT_SLACK_BYTES
        )


def transient_reserve(
    *,
    vocab_size: int,
    max_running_requests: int,
    chunked_prefill_size: Optional[int],
    enable_deterministic_inference: bool,
    prefill_graph_bytes: int,
) -> TransientReserve:
    """``prefill_graph_bytes`` is the measured prefill graph pool, which bounds an
    eager forward of one chunk."""
    row_bytes = _FP32_BYTES * vocab_size
    logprob_tokens = max(chunked_prefill_size or 0, 0)
    if envs.SGLANG_ENABLE_LOGPROB_CHUNK.get():
        logprob_tokens = min(logprob_tokens, envs.SGLANG_LOGPROB_CHUNK_SIZE.get())
    # The input-logprob tail holds the chunk's fp32 logits and the gathered logprob
    # rows, plus their log-softmax off the fast path.
    full_log_softmax = (
        not envs.SGLANG_ENABLE_FAST_INPUT_LOGPROBS.get()
        or enable_deterministic_inference
    )
    logprob_rows = 3 if full_log_softmax else 2
    return TransientReserve(
        persistent_bytes=_PERSISTENT_FP32_ROWS * row_bytes * max_running_requests,
        decode_bytes=_DECODE_FP32_ROWS * row_bytes * max_running_requests,
        prefill_bytes=max(
            prefill_graph_bytes, logprob_rows * row_bytes * logprob_tokens
        ),
        # A freed tail block that smaller live tensors split stays reserved, so the
        # next tail needs one more block of the largest tail allocation.
        fragmentation_bytes=row_bytes * max(max_running_requests, logprob_tokens),
    )


def graph_bounded_running_cap(
    *,
    ceiling: int,
    capture_bs: Sequence[int],
    kv_budget_bytes: int,
    vocab_size: int,
    context_len: int,
    page_size: int,
) -> int:
    """Largest decode capture size up to ``ceiling`` whose per-request reserve, the
    sampling state and decode tail plus its graph state, fits in 1/10 of the KV
    budget."""
    # Graph state per request: one shared fp32 logits row and one attention index
    # row, counted as int64 to cover the triton backend.
    graph_state_bytes = _FP32_BYTES * vocab_size + 8 * -(-context_len // page_size)
    per_request_bytes = (
        _PERSISTENT_FP32_ROWS + _DECODE_FP32_ROWS
    ) * _FP32_BYTES * vocab_size + graph_state_bytes
    fit = kv_budget_bytes // (_KV_BUDGET_SHARE_DENOMINATOR * per_request_bytes)
    limit = min(ceiling, fit)
    return max((bs for bs in capture_bs if bs <= limit), default=min(capture_bs))
