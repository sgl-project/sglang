from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Protocol

import torch


def device_timer_ctx(timer: Optional["DeviceTimer"], category: str):
    """Timing context for one forward segment; no-op when the timer is absent.

    A segment that skips this stays out of the fwd_occupancy numerator while
    still counting in its wall-clock denominator, i.e. reads as GPU idle.
    """
    if timer is None:
        return nullcontext()
    return timer.wrap(metadata={"category": category})


class TimingObserver(Protocol):
    """Optional consumer of existing intervals; no timing or publication policy."""

    def on_interval_start(self, interval: "_TimingInterval") -> None: ...

    def on_interval_ready(self, interval: "_TimingInterval") -> None: ...


class DeviceTimer:
    def __init__(self, reporter: Optional[Callable] = None):
        self._intervals: Deque[_TimingInterval] = deque()
        self._reporters: List[Callable] = [] if reporter is None else [reporter]
        self._in_wrap = False
        self._observer: Optional[TimingObserver] = None

    @contextmanager
    def capture(self, observer: TimingObserver):
        """Observe intervals created in this scope, including later completion."""
        assert self._observer is None, "DeviceTimer.capture is not re-entrant"
        self._observer = observer
        try:
            yield
        finally:
            self._observer = None

    def add_reporter(self, reporter: Callable):
        self._reporters.append(reporter)

    @contextmanager
    def wrap(self, metadata: Dict):
        # Not re-entrant: a nested wrap would end the wrong interval and leave
        # an un-ended one at the head of the queue for _report() to trip over.
        assert not self._in_wrap, "DeviceTimer.wrap is not re-entrant"
        interval = _TimingInterval.create(track_stream=self._observer is not None)
        interval.observer = self._observer
        if interval.observer is not None:
            interval.observer.on_interval_start(interval)
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
            if interval.observer is not None:
                # Observers can retain intervals; break the back-reference first.
                observer, interval.observer = interval.observer, None
                observer.on_interval_ready(interval)


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
    observer: Optional[TimingObserver] = None
    stream: Optional[torch.cuda.Stream] = None

    @staticmethod
    def create(track_stream: bool = False):
        stream = torch.cuda.current_stream() if track_stream else None
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return _TimingInterval(start_event=start_event, stream=stream)

    def end(self, metadata: Dict):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

        assert self.end_event is None
        self.end_event = end_event
        self.metadata = metadata

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)
