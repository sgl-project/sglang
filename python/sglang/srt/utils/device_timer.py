from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import torch


def device_timer_ctx(timer: Optional["DeviceTimer"], category: str):
    """Timing context for one forward segment; no-op when the timer is absent.

    A segment that skips this stays out of the fwd_occupancy numerator while
    still counting in its wall-clock denominator, i.e. reads as GPU idle.
    """
    if timer is None:
        return nullcontext()
    return timer.wrap(metadata={"category": category})


class DeviceTiming:
    """Elapsed span between a group's first and last existing CUDA events.

    Completion is driven by DeviceTimer's nonblocking event queries. The callback
    may be registered after completion (e.g. when CPU result processing catches up).
    No events are moved or added. Work outside these boundaries is not measured.
    A group spanning different streams has no ordered boundary pair and reports
    None rather than silently claiming a complete iteration duration.
    """

    def __init__(self):
        self.num_intervals = 0
        self._pending = 0
        self._sealed = False
        self._elapsed = 0.0
        self._callback = None
        self._first_interval = None
        self._last_interval = None
        self._same_stream = True

    def when_ready(self, callback: Callable[[Optional[float]], None]):
        self._callback = callback
        self._notify()

    def _notify(self):
        if self._sealed and self._pending == 0 and self._callback is not None:
            callback, self._callback = self._callback, None
            if self._first_interval is None:
                callback(0.0)
            elif not self._same_stream:
                callback(None)
            else:
                callback(
                    self._first_interval.start_event.elapsed_time(
                        self._last_interval.end_event
                    )
                    / 1000.0
                )


class DeviceTimer:
    def __init__(self, reporter: Optional[Callable] = None):
        self._intervals: Deque[_TimingInterval] = deque()
        self._reporters: List[Callable] = [] if reporter is None else [reporter]
        self._in_wrap = False
        self._capture: Optional[DeviceTiming] = None

    @contextmanager
    def capture(self):
        """Group intervals launched in this scope without recording new events."""
        assert self._capture is None, "DeviceTimer.capture is not re-entrant"
        timing = DeviceTiming()
        self._capture = timing
        try:
            yield timing
        finally:
            self._capture = None
            timing._sealed = True
            timing._notify()

    def add_reporter(self, reporter: Callable):
        self._reporters.append(reporter)

    @contextmanager
    def wrap(self, metadata: Dict):
        # Not re-entrant: a nested wrap would end the wrong interval and leave
        # an un-ended one at the head of the queue for _report() to trip over.
        assert not self._in_wrap, "DeviceTimer.wrap is not re-entrant"
        interval = _TimingInterval.create()
        interval.capture = self._capture
        if interval.capture is not None:
            timing = interval.capture
            timing.num_intervals += 1
            timing._pending += 1
            if timing._first_interval is None:
                timing._first_interval = interval
            elif interval.stream != timing._first_interval.stream:
                timing._same_stream = False
            timing._last_interval = interval
        self._intervals.append(interval)
        self._in_wrap = True
        try:
            yield
        finally:
            self._in_wrap = False
            interval.end(metadata=metadata)
            self._report()

    def _report(self):
        while len(self._intervals) > 0:
            interval = self._intervals[0]
            if not interval.end_event.query():
                break

            self._intervals.popleft()
            elapsed = interval.elapsed_time() / 1000.0
            for reporter in self._reporters:
                reporter(t=elapsed, **interval.metadata)
            if interval.capture is not None:
                timing = interval.capture
                # Groups retain their boundary intervals; avoid an ownership cycle.
                interval.capture = None
                timing._elapsed += elapsed
                timing._pending -= 1
                timing._notify()


class GapTimer(DeviceTimer):
    """Measures GPU idle gaps between consecutive uses of a stream.

    Where DeviceTimer.wrap() measures the duration *inside* a block,
    GapTimer.wrap() measures the time *between* consecutive blocks
    (gap = next_block_start - last_block_end).
    """

    def __init__(self, reporter: Callable):
        super().__init__(reporter)
        self._pending: Optional[_TimingInterval] = None

    @contextmanager
    def wrap(self, metadata: Dict):
        if self._pending is not None:
            self._pending.end(metadata=metadata)
            self._intervals.append(self._pending)
            self._pending = None
            self._report()
        try:
            yield
        finally:
            self._pending = _TimingInterval.create()

    def cancel(self):
        """Discard a pending gap (e.g. server went idle)."""
        self._pending = None


@dataclass
class _TimingInterval:
    start_event: torch.cuda.Event
    end_event: Optional[torch.cuda.Event] = None
    metadata: Optional[Dict] = None
    capture: Optional[DeviceTiming] = None
    stream: Optional[torch.cuda.Stream] = None

    @staticmethod
    def create():
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        return _TimingInterval(start_event=start_event, stream=stream)

    def end(self, metadata: Dict):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

        assert self.end_event is None
        self.end_event = end_event
        self.metadata = metadata

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)
