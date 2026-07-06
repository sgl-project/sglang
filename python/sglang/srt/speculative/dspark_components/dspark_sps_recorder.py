from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional

import msgspec

SPS_RECORD_MAX_STEP_INTERVAL_SECONDS = 1.0
SPS_RECORD_MAX_RECORDS = 200_000


class SpsStepRecord(msgspec.Struct, frozen=True):
    forward_ct: int
    num_running_reqs: int
    num_verify_tokens: int
    step_time: float
    verify_tokens_local: int = -1
    verify_tokens_dp_synced: int = -1
    verify_tokens_graph_key: int = -1


class SpsDataRecorder:
    def __init__(
        self,
        *,
        max_records: int = SPS_RECORD_MAX_RECORDS,
        max_step_interval: float = SPS_RECORD_MAX_STEP_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_records < 1:
            raise ValueError(f"max_records must be >= 1, got {max_records}.")
        self._max_step_interval = max_step_interval
        self._clock = clock
        self._records: deque[SpsStepRecord] = deque(maxlen=max_records)
        self._prev_stamp: Optional[tuple[float, int, int, int, int, int, int]] = None

    def observe_decode_step(
        self,
        *,
        forward_ct: int,
        num_running_reqs: int,
        num_verify_tokens: int,
        verify_tokens_local: int = -1,
        verify_tokens_dp_synced: int = -1,
        verify_tokens_graph_key: int = -1,
    ) -> None:
        now = self._clock()
        prev = self._prev_stamp
        self._prev_stamp = (
            now,
            forward_ct,
            num_running_reqs,
            num_verify_tokens,
            verify_tokens_local,
            verify_tokens_dp_synced,
            verify_tokens_graph_key,
        )
        if prev is None:
            return
        (
            prev_time,
            prev_forward_ct,
            prev_num_running_reqs,
            prev_num_verify_tokens,
            prev_verify_tokens_local,
            prev_verify_tokens_dp_synced,
            prev_verify_tokens_graph_key,
        ) = prev
        step_time = now - prev_time
        if not (0.0 < step_time <= self._max_step_interval):
            return
        self._records.append(
            SpsStepRecord(
                forward_ct=prev_forward_ct,
                num_running_reqs=prev_num_running_reqs,
                num_verify_tokens=prev_num_verify_tokens,
                step_time=step_time,
                verify_tokens_local=prev_verify_tokens_local,
                verify_tokens_dp_synced=prev_verify_tokens_dp_synced,
                verify_tokens_graph_key=prev_verify_tokens_graph_key,
            )
        )

    def note_non_decode_step(self) -> None:
        self._prev_stamp = None

    def dump_records(self) -> list[dict]:
        return [msgspec.to_builtins(record) for record in self._records]
