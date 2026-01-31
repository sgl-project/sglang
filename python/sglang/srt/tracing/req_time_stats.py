from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List

from sglang.srt.tracing.clock import now_mono_s


class ReqTimePoint(IntEnum):
    received = 0
    request_sent_to_scheduler = 1
    scheduled = 2
    first_token = 3
    response_sent = 4
    finished = 5


@dataclass
class ReqTimeStatsBase:
    """
    Per request time stats are stored as a fixed list of monotonic timestamps in seconds.
    Each entry corresponds to a milestone in ReqTimePoint.
    Durations are computed as ts[end] minus ts[start].
    """

    ts_mono_s: List[float]

    @classmethod
    def create(cls) -> "ReqTimeStatsBase":
        return cls(ts_mono_s=[float("nan")] * len(ReqTimePoint))

    def mark(self, point: ReqTimePoint) -> None:
        i = int(point)
        cur = self.ts_mono_s[i]
        if cur == cur:
            return
        self.ts_mono_s[i] = now_mono_s()

    def mark_at(self, point: ReqTimePoint, ts_mono_s: float) -> None:
        i = int(point)
        cur = self.ts_mono_s[i]
        if cur == cur:
            return
        self.ts_mono_s[i] = ts_mono_s

    def duration_s(self, start: ReqTimePoint, end: ReqTimePoint) -> float:
        return self.ts_mono_s[int(end)] - self.ts_mono_s[int(start)]

    def to_log_record(self) -> dict:
        return {"req_time_ts_mono_s": list(self.ts_mono_s)}
