"""Steady-state metrics for online serving benchmarks.

This module deliberately owns only the interval-selection and interval-throughput
logic.  The regular serving benchmark remains responsible for issuing requests
and reporting its full-run metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence, Tuple

import numpy as np


class RequestOutput(Protocol):
    """The subset of a serving request result needed by this module."""

    success: bool
    start_time: float
    latency: float
    ttft: float
    itl: List[float]
    output_len: int
    generated_text: str


class InputRequest(Protocol):
    prompt_len: int


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> List[Any]: ...


@dataclass(frozen=True)
class SteadyStateWindow:
    start: float
    end: float
    duration: float
    concurrency_threshold: int
    peak_concurrency: int


@dataclass(frozen=True)
class SteadyStateMetrics:
    concurrency_ratio: float
    window_start: float
    window_end: float
    duration: float
    concurrency_threshold: int
    completed: int
    total_input: Optional[int]
    input_throughput: Optional[float]
    total_output: float
    total_output_retokenized: float
    output_throughput: float
    output_throughput_retokenized: float
    average_concurrency: float
    peak_concurrency: int
    peak_output_throughput: float


def find_steady_state_window(
    outputs: Sequence[RequestOutput], concurrency_ratio: float
) -> SteadyStateWindow:
    """Select the longest continuous interval above the concurrency threshold."""
    if not 0 < concurrency_ratio <= 1:
        raise ValueError("steady-state concurrency ratio must be in (0, 1]")

    successful = [output for output in outputs if output.success and output.latency > 0]
    if not successful:
        raise ValueError("no successful requests with positive latency")

    events = {}
    for output in successful:
        events[output.start_time] = events.get(output.start_time, 0) + 1
        end_time = output.start_time + output.latency
        events[end_time] = events.get(end_time, 0) - 1

    concurrency = 0
    intervals = []
    event_times = sorted(events)
    for index, event_time in enumerate(event_times[:-1]):
        concurrency += events[event_time]
        next_time = event_times[index + 1]
        if next_time > event_time:
            intervals.append((event_time, next_time, concurrency))

    peak_concurrency = max((item[2] for item in intervals), default=0)
    threshold = max(1, int(np.ceil(peak_concurrency * concurrency_ratio)))

    spans: List[Tuple[float, float]] = []
    span_start: Optional[float] = None
    span_end: Optional[float] = None
    for start, end, active_requests in intervals:
        if active_requests >= threshold:
            if span_start is None:
                span_start = start
            span_end = end
        elif span_start is not None and span_end is not None:
            spans.append((span_start, span_end))
            span_start = span_end = None
    if span_start is not None and span_end is not None:
        spans.append((span_start, span_end))

    if not spans:
        raise ValueError("no steady-state measurement window could be determined")

    # max() keeps the first span when durations tie, making selection deterministic.
    window_start, window_end = max(spans, key=lambda span: span[1] - span[0])
    return SteadyStateWindow(
        start=window_start,
        end=window_end,
        duration=window_end - window_start,
        concurrency_threshold=threshold,
        peak_concurrency=peak_concurrency,
    )


def _token_timestamps(output: RequestOutput) -> List[float]:
    timestamps = [output.start_time + output.ttft]
    for inter_token_latency in output.itl:
        timestamps.append(timestamps[-1] + inter_token_latency)
    return timestamps


def _output_tokens_in_window(
    outputs: Sequence[RequestOutput],
    output_lens: Sequence[int],
    retokenized_output_lens: Sequence[int],
    window: SteadyStateWindow,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    output_tokens = 0.0
    retokenized_tokens = 0.0
    token_events: List[Tuple[float, float]] = []

    for output, output_len, retokenized_len in zip(
        outputs, output_lens, retokenized_output_lens
    ):
        if not output.success or output.latency <= 0:
            continue

        timestamps = _token_timestamps(output)
        timestamps_in_window = [
            timestamp
            for timestamp in timestamps
            if window.start <= timestamp <= window.end
        ]
        fraction_in_window = len(timestamps_in_window) / len(timestamps)
        output_tokens += output_len * fraction_in_window
        retokenized_tokens += retokenized_len * fraction_in_window

        event_weight = output_len / len(timestamps)
        token_events.extend(
            (timestamp, event_weight) for timestamp in timestamps_in_window
        )

    return output_tokens, retokenized_tokens, token_events


def calculate_steady_state_metrics(
    outputs: Sequence[RequestOutput],
    tokenizer: Tokenizer,
    concurrency_ratio: float,
    input_requests: Optional[Sequence[InputRequest]] = None,
) -> SteadyStateMetrics:
    """Calculate token throughput metrics for the steady-state window.

    Input tokens do not have per-token timestamps. Their throughput therefore uses
    an arrival-based definition: prompt tokens from successful requests that start
    inside the half-open measurement window ``[start, end)`` divided by its duration.
    ``None`` is reported when aligned input requests are unavailable, as in the
    current multi-turn serving benchmark.
    """
    window = find_steady_state_window(outputs, concurrency_ratio)

    if input_requests is not None and len(input_requests) != len(outputs):
        raise ValueError("input requests and outputs must have the same length")

    if input_requests is None:
        total_input = None
        input_throughput = None
    else:
        total_input = sum(
            request.prompt_len
            for request, output in zip(input_requests, outputs)
            if output.success and window.start <= output.start_time < window.end
        )
        input_throughput = total_input / window.duration

    output_lens = [output.output_len if output.success else 0 for output in outputs]
    retokenized_output_lens = [
        (
            len(tokenizer.encode(output.generated_text, add_special_tokens=False))
            if output.success
            else 0
        )
        for output in outputs
    ]
    total_output, total_output_retokenized, token_events = _output_tokens_in_window(
        outputs,
        output_lens,
        retokenized_output_lens,
        window,
    )

    successful = [output for output in outputs if output.success and output.latency > 0]
    completed = sum(
        window.start <= output.start_time + output.latency <= window.end
        for output in successful
    )
    overlap_duration = sum(
        max(
            0.0,
            min(output.start_time + output.latency, window.end)
            - max(output.start_time, window.start),
        )
        for output in successful
    )

    duration_seconds = max(1, int(np.ceil(window.duration)))
    tokens_per_second = np.zeros(duration_seconds)
    for token_time, token_weight in token_events:
        elapsed = max(0.0, token_time - window.start)
        # A token exactly on a second boundary belongs to the second that just
        # ended. In particular, a token at window.end must not be folded into
        # a bucket that already contains the preceding second's tokens.
        bucket = max(0, int(np.ceil(elapsed)) - 1)
        bucket = min(bucket, duration_seconds - 1)
        tokens_per_second[bucket] += token_weight

    return SteadyStateMetrics(
        concurrency_ratio=concurrency_ratio,
        window_start=window.start,
        window_end=window.end,
        duration=window.duration,
        concurrency_threshold=window.concurrency_threshold,
        completed=completed,
        total_input=total_input,
        input_throughput=input_throughput,
        total_output=total_output,
        total_output_retokenized=total_output_retokenized,
        output_throughput=total_output / window.duration,
        output_throughput_retokenized=total_output_retokenized / window.duration,
        average_concurrency=overlap_duration / window.duration,
        peak_concurrency=window.peak_concurrency,
        peak_output_throughput=float(np.max(tokens_per_second)),
    )


def steady_state_output_throughput(
    outputs: Sequence[RequestOutput],
    output_lens: Sequence[int],
    retokenized_output_lens: Sequence[int],
    concurrency_ratio: float,
) -> Tuple[float, float, float, int]:
    """Compatibility helper for callers interested only in output throughput."""
    window = find_steady_state_window(outputs, concurrency_ratio)
    output_tokens, retokenized_tokens, _ = _output_tokens_in_window(
        outputs, output_lens, retokenized_output_lens, window
    )
    return (
        output_tokens / window.duration,
        retokenized_tokens / window.duration,
        window.duration,
        window.concurrency_threshold,
    )
