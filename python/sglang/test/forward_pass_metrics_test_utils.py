"""CPU event fixtures for FPM tests."""

from contextlib import contextmanager

from sglang.srt.observability.forward_pass_metrics import FpmTiming


class FakeEvent:
    def __init__(self, timestamp):
        self.timestamp = timestamp

    def elapsed_time(self, end):
        return end.timestamp - self.timestamp


class FakeInterval:
    def __init__(self, milliseconds, ready=False, start=0, stream=0):
        self.milliseconds = milliseconds
        self.ready = ready
        self.end_event = self
        self.start_event = FakeEvent(start)
        self.timestamp = start + milliseconds
        self.stream = stream
        self.observer = None
        self.metadata = None

    def end(self, metadata):
        self.metadata = metadata

    def query(self):
        return self.ready

    def elapsed_time(self):
        return self.milliseconds


@contextmanager
def capture_timing(timer):
    timing = FpmTiming()
    try:
        with timer.capture(timing):
            yield timing
    finally:
        timing.seal()
