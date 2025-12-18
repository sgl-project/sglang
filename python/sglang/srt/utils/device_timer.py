from typing import Deque, Optional

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
    @dataclass
    class Interval:
        start: torch.cuda.Event
        end: Optional[torch.cuda.Event] = None
        category: Optional[str] = None

    def __init__(self):
        self._intervals: Deque[DeviceTimer.Interval] = deque()

    @contextmanager
    def wrap(self, category: str):
        self._start()
        try:
            yield
        finally:
            self._end(category=category)

    def _start(self):
        self._intervals.append(DeviceTimer.Interval(start=torch.cuda.Event()))

    def _end(self, category: str):
        interval = self._intervals[-1]
        assert interval.end is None
        interval.end = torch.cuda.Event()
        interval.category = category
