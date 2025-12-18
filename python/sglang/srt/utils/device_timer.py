from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import torch


class DeviceTimer:
    def __init__(self, reporter: Callable[[str, float], None]):
        self._intervals: Deque[_TimingInterval] = deque()
        self._reporter = reporter

    @contextmanager
    def wrap(self, category: str):
        self._intervals.append(_TimingInterval.create())
        try:
            yield
        finally:
            self._intervals[-1].end(category=category)
            self._report()

    def _report(self):
        while len(self._intervals) > 0:
            interval = self._intervals[0]
            if not interval.end_event.query():
                break

            self._intervals.popleft()
            self._reporter(interval.category, interval.elapsed_time() / 1000.0)


@dataclass
class _TimingInterval:
    start_event: torch.cuda.Event
    end_event: Optional[torch.cuda.Event] = None
    category: Optional[str] = None

    @staticmethod
    def create():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return _TimingInterval(start_event=start_event)

    def end(self, category: str):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

        assert self.end_event is None
        self.end_event = end_event
        self.category = category

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)
