"""Release retired overlap snapshots outside the scheduler thread."""

import queue
import threading


class DeferredOverlapRelease:
    """Delete retired overlap snapshots on a background thread."""

    def __init__(self):
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run,
            name="uocr-overlap-release",
            daemon=True,
        )
        self._thread.start()

    def defer(self, value) -> None:
        """Queue a retired snapshot for background release."""
        self._queue.put(value)

    def _run(self) -> None:
        while True:
            value = self._queue.get()
            del value
