"""Result-owned FPM spans over existing GPU timing intervals."""

import logging
from functools import wraps
from typing import Callable, Optional

from msgspec.structs import replace

from sglang.srt.utils.device_timer import DeviceTimer

logger = logging.getLogger(__name__)


class FpmTiming:
    """First-start to last-end GPU span on one stream, including segment gaps."""

    def __init__(self):
        self.num_intervals = 0
        self._pending = 0
        self._sealed = False
        self._callback = None
        self._first_interval = None
        self._last_interval = None
        self._same_stream = True

    def on_interval_start(self, interval):
        self.num_intervals += 1
        self._pending += 1
        if self._first_interval is None:
            self._first_interval = interval
        elif interval.stream != self._first_interval.stream:
            self._same_stream = False
        self._last_interval = interval

    def on_interval_ready(self, interval):
        self._pending -= 1
        self._notify()

    def seal(self):
        self._sealed = True
        self._notify()

    def when_ready(self, callback: Callable[[Optional[float]], None]):
        self._callback = callback
        self._notify()

    def _notify(self):
        if not self._sealed or self._pending or self._callback is None:
            return
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

    def publish_when_ready(self, metrics, publisher):
        # Hold only immutable scalar statistics, never a live batch/request.
        def publish(elapsed):
            if elapsed is None:
                logger.warning(
                    "FPM timing spans multiple CUDA streams; skipping ambiguous span"
                )
                return
            publisher.publish(replace(metrics, wall_time=elapsed))

        self.when_ready(publish)


def wrap_forward_with_fpm(forward: Callable, timer: DeviceTimer) -> Callable:
    """Installed on the scheduler instance only when that rank enables FPM."""

    @wraps(forward)
    def timed_forward(batch, *args, **kwargs):
        # PREBUILT can recursively dispatch an inner idle forward for DP MLP
        # synchronization. Let that real forward own the capture and result.
        if batch.forward_mode.is_prebuilt():
            return forward(batch, *args, **kwargs)
        timing = FpmTiming()
        try:
            with timer.capture(timing):
                result = forward(batch, *args, **kwargs)
        finally:
            timing.seal()
        result.fpm_timing = timing
        return result

    return timed_forward
