from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HostCopyDone:
    """Drop-in for a ``copy_done`` CUDA event, backed by a worker thread.

    Carries the copy's exception as well, so ``synchronize()`` can abort the
    consumer instead of letting it read invalid host tensors.
    """

    def __init__(self):
        self._done = threading.Event()
        self.error: Optional[BaseException] = None

    def record(self, *args, **kwargs) -> None:
        # No-op for parity with torch.cuda.Event; set_done is the real signal.
        pass

    def set_done(self, error: Optional[BaseException] = None) -> None:
        self.error = error
        self._done.set()

    def synchronize(self) -> None:
        """Block until the copy completes; re-raise if it failed."""
        self._done.wait()
        if self.error is not None:
            raise RuntimeError(
                "Async device->host copy failed; destination tensors are invalid"
            ) from self.error

    def query(self) -> bool:
        """True once the copy has completed (successfully or not)."""
        return self._done.is_set()


class AsyncD2HCopyWorker:
    """Runs blocking device->host copies on a dedicated daemon thread.

    Under NVIDIA Confidential Computing a D2H ``cudaMemcpyAsync`` is forced
    synchronous and blocks at issue, stalling the submitting thread; this moves
    the copy off it so the overlap scheduler keeps launching the next step.

    ``d2h_copy_stream`` is private on purpose: on a shared stream the blanket
    ``synchronize()`` below would also block on whatever the caller queued after
    submitting, re-coupling the caller to the copy.
    """

    def __init__(self, device_module):
        self.device_module = device_module
        # Private stream, created and owned here so nothing outside this worker
        # can enqueue onto it (see class docstring). Created on the caller's
        # thread, so it lands on the current device.
        self.d2h_copy_stream = device_module.Stream()
        self._device_index = device_module.current_device()
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="sglang-d2h-copy-worker", daemon=True
        )
        self._thread.start()

    def submit(self, copy_fn: Callable[[], None]) -> HostCopyDone:
        """Record readiness on the CURRENT stream and enqueue the copy.

        Must be called with the stream that produced the copy sources current.
        """
        src_ready = self.device_module.Event()
        src_ready.record()
        done = HostCopyDone()
        self._queue.put((src_ready, copy_fn, done))
        return done

    def _loop(self):
        self.device_module.set_device(self._device_index)
        while True:
            item = self._queue.get()
            if item is None:
                return
            src_ready, copy_fn, done = item
            error = None
            try:
                # Event-sync, not a stream wait: a stream wait would re-expose
                # the caller's thread to the blocking copy.
                src_ready.synchronize()
                with self.device_module.stream(self.d2h_copy_stream):
                    copy_fn()
                self.d2h_copy_stream.synchronize()
            except Exception as e:
                logger.exception("AsyncD2HCopyWorker copy failed")
                error = e
            finally:
                # Signal on failure too, so the caller never hangs.
                done.set_done(error=error)

    def shutdown(self, timeout: float = 2.0):
        """Signal the worker to stop and join it (best-effort, bounded wait)."""
        if not self._thread.is_alive():
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
