from typing import Deque, Optional, Callable

import torch
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


@contextmanager
def time_device_forward_pass(forward_mode: ForwardMode):
    if not _ENABLE_METRICS_DEVICE_TIMER:
        yield
        return

    category = "forward_" + forward_mode.name.lower()
    with TODO.wrap(category=category):
        yield


class DeviceTimer:
    _DELAY_THRESHOLD = 2

    @dataclass
    class Interval:
        start_event: torch.cuda.Event
        end_event: Optional[torch.cuda.Event] = None
        category: Optional[str] = None

        @staticmethod
        def create():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return DeviceTimer.Interval(start_event=start_event)

        def end(self, category: str):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()

            assert self.end_event is None
            self.end_event = end_event
            self.category = category

    def __init__(self, reporter: Callable[[str, float], None]):
        self._intervals: Deque[DeviceTimer.Interval] = deque()
        self._reporter = reporter

    @contextmanager
    def wrap(self, category: str):
        self._intervals.append(DeviceTimer.Interval.create())
        try:
            yield
        finally:
            self._intervals[-1].end(category=category)
            self._report()

    def _report(self):
        while len(self._intervals) >= self._DELAY_THRESHOLD:
            interval = self._intervals.popleft()
            interval.end
            TODO
