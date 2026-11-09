"""Early-stop-aware accounting for one-batch streaming benchmarks.

Steady-state window = last request's first token -> first request's finish.
Token counts come from the cumulative meta_info["completion_tokens"] (the
server may coalesce chunks); boundary counts are interpolated per request so
same-step chunk delivery order cannot skew the window.
"""

from typing import List, Optional, Tuple

import msgspec

# (arrival time, cumulative completion tokens) of one delivered chunk.
_Obs = Tuple[float, int]


class SteadyStateWindow(msgspec.Struct, frozen=True):
    """Full-batch decode window: every request started, none finished yet."""

    start: float
    end: float
    output_tokens: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def output_throughput(self) -> float:
        return self.output_tokens / self.duration


def validate_finish_reason(finish_reason: dict, *, ignore_eos: bool) -> None:
    """Raise on any finish reason the benchmark must not silently accept."""
    finish_type = finish_reason["type"]
    if finish_type == "length":
        return
    if finish_type == "stop":
        # The scheduler's NaN detector reports as a "stop" match; never accept it.
        if finish_reason["matched"] == "NaN happened":
            raise RuntimeError(f"Request hit NaN logits: {finish_reason}.")
        if not ignore_eos:
            return
        raise RuntimeError(
            f"Request stopped early despite ignore_eos=True: {finish_reason}."
        )
    raise RuntimeError(f"Unexpected finish reason: {finish_reason}.")


class _Boundary:
    """Per-request observations bracketing one boundary instant."""

    def __init__(self, time: float, before: List[Optional[_Obs]]):
        self.time = time
        self.before = before
        self.after: List[Optional[_Obs]] = [None] * len(before)

    def observe(self, index: int, obs: _Obs) -> None:
        if self.after[index] is None:
            self.after[index] = obs

    def tokens_at_boundary(self, index: int) -> float:
        """Tokens of `index` at `time`, interpolated between its chunks."""
        before = self.before[index]
        if before is None:
            return 0.0
        t_before, c_before = before
        after = self.after[index]
        if after is None or self.time <= t_before:
            return float(c_before)
        t_after, c_after = after
        if self.time >= t_after or t_after <= t_before:
            return float(c_after)
        return c_before + (c_after - c_before) * (self.time - t_before) / (
            t_after - t_before
        )

    def total_tokens(self) -> float:
        return sum(self.tokens_at_boundary(i) for i in range(len(self.before)))


class BatchStreamRecorder:
    """Tracks per-request progress of one batched streaming /generate call."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.all_started_time: Optional[float] = None
        self._last_obs: List[Optional[_Obs]] = [None] * batch_size
        self._first_token_time: List[Optional[float]] = [None] * batch_size
        self._completion_tokens: List[int] = [0] * batch_size
        self._finish_types: List[Optional[str]] = [None] * batch_size
        self._num_started = 0
        self._t0: Optional[_Boundary] = None
        self._t1: Optional[_Boundary] = None

    def record_chunk(
        self,
        *,
        index: int,
        completion_tokens: int,
        finish_type: Optional[str],
        now: float,
    ) -> None:
        obs = (now, completion_tokens)
        for boundary in (self._t0, self._t1):
            if boundary is not None:
                boundary.observe(index, obs)
        self._last_obs[index] = obs
        self._completion_tokens[index] = completion_tokens
        if self._first_token_time[index] is None and completion_tokens > 0:
            self._first_token_time[index] = now
            self._num_started += 1
            if self._num_started == self.batch_size:
                self.all_started_time = now
                self._t0 = _Boundary(time=now, before=list(self._last_obs))
        if finish_type is not None and self._finish_types[index] is None:
            self._finish_types[index] = finish_type
            if self._t1 is None:
                self._t1 = _Boundary(time=now, before=list(self._last_obs))

    def missing_indices(self) -> List[int]:
        return [i for i, t in enumerate(self._first_token_time) if t is None]

    @property
    def total_output_tokens(self) -> int:
        return sum(self._completion_tokens)

    @property
    def num_early_stopped(self) -> int:
        return sum(1 for t in self._finish_types if t == "stop")

    @property
    def tokens_before_all_started(self) -> Optional[float]:
        """Batch tokens at the last request's first token; None before then."""
        if self._t0 is None:
            return None
        return self._t0.total_tokens()

    def steady_state_window(self) -> Optional[SteadyStateWindow]:
        """None when the batch never decoded at full size for a nonzero span."""
        if self._t0 is None or self._t1 is None:
            return None
        if self._t1.time <= self._t0.time:
            return None
        output_tokens = self._t1.total_tokens() - self._t0.total_tokens()
        if output_tokens <= 0:
            return None
        return SteadyStateWindow(
            start=self._t0.time,
            end=self._t1.time,
            output_tokens=output_tokens,
        )
