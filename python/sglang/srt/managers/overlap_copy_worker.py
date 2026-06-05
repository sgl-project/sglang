from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class AsyncD2HCopyWorker:
    """Runs the per-step result Device->Host readback on a dedicated daemon thread.

    Under NVIDIA Confidential Computing (bounce-buffer CC), a device-to-host
    ``cudaMemcpyAsync`` is forced synchronous and blocks AT ISSUE (the host
    destination is staged through an encrypted bounce buffer). Issuing it inline
    on the scheduler thread stalls the overlap pipeline: the scheduler cannot
    launch the next step's CUDA graph until the copy returns (~one decode step).

    This worker performs the (still-blocking) copy off the scheduler thread so
    the scheduler keeps issuing work. It mirrors TensorRT-LLM PR #8463:
      - the scheduler thread records a CUDA event after the producing work and
        hands (event, copy_fn, done) here, then returns immediately;
      - this worker ``cudaEventSynchronize``-s on that event (event-sync, NOT
        stream-wait, so the blocking copy does not stall the scheduler's CUDA
        API calls), runs the copy on the dedicated copy stream, blocks until it
        completes, and sets ``done``;
      - the scheduler later waits on ``done`` (worker-done => copy-done).

    The copy itself stays synchronous under CC; it is merely non-blocking *to the
    scheduler thread*, restoring overlap.
    """

    def __init__(self, device_module, copy_stream):
        self.device_module = device_module
        self.copy_stream = copy_stream
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="sglang-d2h-copy-worker", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        src_ready: "torch.cuda.Event",  # noqa: F821 - annotation only
        copy_fn: Callable[[], None],
        done: threading.Event,
    ):
        """Enqueue a readback. ``src_ready`` must already be recorded on the
        stream that produces the copy sources; ``copy_fn`` performs the actual
        ``.to("cpu", ...)`` copies; ``done`` is set when they have completed."""
        self._queue.put((src_ready, copy_fn, done))

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            src_ready, copy_fn, done = item
            try:
                # Wait until the producing forward+sample has materialized the
                # source tensors. Event-sync (cudaEventSynchronize), not a
                # stream wait — see class docstring / TRT-LLM #8463.
                src_ready.synchronize()
                # Issue the copies on a dedicated stream owned by this thread.
                # PyTorch's current stream is thread-local, so this does not
                # affect the scheduler thread's stream.
                with self.device_module.stream(self.copy_stream):
                    copy_fn()
                self.copy_stream.synchronize()
            except Exception:
                logger.exception("AsyncD2HCopyWorker readback failed")
            finally:
                done.set()

    def shutdown(self, timeout: float = 2.0):
        """Signal the worker to stop and join it (best-effort, bounded wait)."""
        if not self._thread.is_alive():
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
