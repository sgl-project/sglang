from typing import Deque

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
        end: torch.cuda.Event

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
        TODO

    def _end(self, category: str):
        TODO
