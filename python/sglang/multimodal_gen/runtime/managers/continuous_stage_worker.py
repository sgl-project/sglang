# SPDX-License-Identifier: Apache-2.0
"""Async encode/decode workers for continuous diffusion batching.

Text-encode admission and VAE-decode completion each run on their own
worker thread with their own side CUDA stream so the scheduler can keep
stepping the packed denoising batch while both stages overlap.

Design points:

- Bounded job queues: ``submit`` raises :class:`StageQueueFull` when a
  worker's queue is at capacity, and callers apply backpressure (requests
  stay queued / completed states stay in a backlog) instead of running the
  work inline.
- CUDA-event handoff: producers record an event at submit time and the
  worker stream waits on exactly that event (not on everything enqueued to
  the default stream afterwards). Each result carries a completion event;
  results consumed on the GPU make the consumer stream wait on it, while
  host-consumed kinds (finalize) synchronize before being returned.
- Lifecycle states: each worker moves NEW -> RUNNING -> DRAINING -> STOPPED
  (or FAILED); a dead worker surfaces synthetic error results for its
  in-flight tickets instead of hanging pollers.
- Robust shutdown: sentinels, bounded joins, and error results for any job
  that never ran.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Result kinds consumed on the host (synchronize before returning); other
# kinds are consumed on the scheduler's GPU stream (stream wait, no host sync).
_HOST_CONSUMED_KINDS = frozenset({"finalize"})

_DEFAULT_QUEUE_DEPTH = 8
_BLOCK_POLL_INTERVAL_S = 0.1
_DEFAULT_BLOCK_TIMEOUT_S = 600.0


class StageQueueFull(Exception):
    """The bounded stage queue is full; apply backpressure and retry later."""


class WorkerState(Enum):
    NEW = "new"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class StageJob:
    kind: str  # "encode" | "finalize"
    ticket: Any
    payload: Callable[[], Any]
    # Recorded on the producer stream at submit; the worker stream waits on
    # it so the job only depends on work enqueued *before* submission.
    submit_event: Any = None


@dataclass(slots=True)
class StageResult:
    kind: str
    ticket: Any
    value: Any = None
    error: BaseException | None = None
    # Recorded on the worker stream after the job; consumers order against it.
    done_event: Any = None


@dataclass(slots=True)
class _WorkerHandle:
    kind: str
    jobs: queue.Queue[StageJob | None]
    thread: threading.Thread
    stream: Any = None
    state: WorkerState = WorkerState.NEW
    # Tickets submitted but not yet returned; used to error out jobs when a
    # worker dies or is shut down before running them.
    in_flight: dict[Any, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def async_stages_supported(server_args: Any) -> bool:
    """Return True if encode/decode can run async without component offload."""
    if not bool(getattr(server_args, "cb_async_stages", True)):
        return False
    offload_flags = (
        "dit_cpu_offload",
        "dit_layerwise_offload",
        "text_encoder_cpu_offload",
        "image_encoder_cpu_offload",
        "vae_cpu_offload",
    )
    return not any(getattr(server_args, flag, False) for flag in offload_flags)


class AsyncContinuousStageWorker:
    """Separate encode and finalize workers with bounded queues.

    The public surface (``submit`` / ``try_submit`` / ``poll_results`` /
    ``pending`` / ``shutdown``) is thread-safe for a single producer (the
    scheduler loop) and hides the per-kind workers behind one result queue.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        queue_depth: int = _DEFAULT_QUEUE_DEPTH,
        kinds: tuple[str, ...] = ("encode", "finalize"),
    ) -> None:
        self._queue_depth = max(1, int(queue_depth))
        self._device = device
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda and device is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._results: queue.Queue[StageResult] = queue.Queue()
        self._pending_lock = threading.Lock()
        self._pending_by_kind: dict[str, int] = {kind: 0 for kind in kinds}
        self._shutdown = False
        self._workers: dict[str, _WorkerHandle] = {}
        for kind in kinds:
            self._workers[kind] = self._start_worker(kind)

    def _start_worker(self, kind: str) -> _WorkerHandle:
        stream = torch.cuda.Stream(device=self._device) if self._use_cuda else None
        handle = _WorkerHandle(
            kind=kind,
            jobs=queue.Queue(maxsize=self._queue_depth),
            thread=threading.Thread(
                name=f"cb-stage-{kind}",
                daemon=True,
                target=self._run_worker,
                args=(kind,),
            ),
            stream=stream,
        )
        self._workers[kind] = handle
        handle.state = WorkerState.RUNNING
        handle.thread.start()
        return handle

    # ------------------------------------------------------------------ #
    # Producer API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def pending(self) -> int:
        with self._pending_lock:
            return sum(self._pending_by_kind.values())

    def pending_for(self, kind: str) -> int:
        with self._pending_lock:
            return self._pending_by_kind.get(kind, 0)

    def worker_state(self, kind: str) -> WorkerState:
        return self._workers[kind].state

    def can_submit(self, kind: str, count: int = 1) -> bool:
        """Whether ``count`` more jobs fit in this worker's bounded queue."""
        if self._shutdown:
            return False
        handle = self._workers.get(kind)
        if handle is None or handle.state is not WorkerState.RUNNING:
            return False
        return (self._queue_depth - handle.jobs.qsize()) >= count

    def submit(self, kind: str, ticket: Any, fn: Callable[[], Any]) -> None:
        """Queue one job; raises StageQueueFull when at capacity."""
        if self._shutdown:
            raise RuntimeError("stage worker is shut down")
        handle = self._workers.get(kind)
        if handle is None:
            raise ValueError(f"unknown stage kind: {kind}")
        if handle.state is not WorkerState.RUNNING:
            raise RuntimeError(f"stage worker {kind} is {handle.state.value}")
        submit_event = None
        if handle.stream is not None:
            submit_event = torch.cuda.Event()
            submit_event.record(torch.cuda.current_stream(self._device))
        job = StageJob(kind=kind, ticket=ticket, payload=fn, submit_event=submit_event)
        with handle.lock:
            handle.in_flight[ticket] = kind
        try:
            handle.jobs.put_nowait(job)
        except queue.Full:
            with handle.lock:
                handle.in_flight.pop(ticket, None)
            raise StageQueueFull(
                f"stage {kind} queue is full ({self._queue_depth} jobs)"
            ) from None
        with self._pending_lock:
            self._pending_by_kind[kind] = self._pending_by_kind.get(kind, 0) + 1

    def try_submit(self, kind: str, ticket: Any, fn: Callable[[], Any]) -> bool:
        """Like submit() but returns False instead of raising on backpressure."""
        try:
            self.submit(kind, ticket, fn)
            return True
        except (StageQueueFull, RuntimeError):
            return False

    # ------------------------------------------------------------------ #
    # Consumer API                                                         #
    # ------------------------------------------------------------------ #

    def poll_results(
        self,
        block_one: bool = False,
        block_timeout_s: float = _DEFAULT_BLOCK_TIMEOUT_S,
    ) -> list[StageResult]:
        """Drain finished results, optionally blocking for one.

        Blocking wakes early when every worker with in-flight jobs dies, in
        which case synthetic error results are returned for those tickets.
        """
        results: list[StageResult] = []
        if block_one and self.pending > 0:
            waited = 0.0
            while waited < block_timeout_s:
                try:
                    results.append(self._results.get(timeout=_BLOCK_POLL_INTERVAL_S))
                    break
                except queue.Empty:
                    waited += _BLOCK_POLL_INTERVAL_S
                    dead = self._collect_dead_worker_results()
                    if dead:
                        results.extend(dead)
                        break
            else:
                logger.error(
                    "Timed out waiting %.0fs for an async stage result",
                    block_timeout_s,
                )
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        results.extend(self._collect_dead_worker_results())
        return [self._finish_result(result) for result in results]

    def _finish_result(self, result: StageResult) -> StageResult:
        handle = self._workers.get(result.kind)
        if handle is not None:
            with handle.lock:
                handle.in_flight.pop(result.ticket, None)
        with self._pending_lock:
            current = self._pending_by_kind.get(result.kind, 0)
            self._pending_by_kind[result.kind] = max(0, current - 1)
        done_event = result.done_event
        if done_event is not None:
            if result.kind in _HOST_CONSUMED_KINDS:
                # The caller serializes this value on the host right away.
                done_event.synchronize()
            else:
                # The caller keeps using the value on the scheduler stream.
                torch.cuda.current_stream(self._device).wait_event(done_event)
        return result

    def _collect_dead_worker_results(self) -> list[StageResult]:
        """Turn in-flight jobs of dead workers into error results."""
        results: list[StageResult] = []
        for handle in self._workers.values():
            if handle.state in (WorkerState.RUNNING, WorkerState.DRAINING):
                if handle.thread.is_alive():
                    continue
                handle.state = WorkerState.FAILED
            if handle.state not in (WorkerState.FAILED, WorkerState.STOPPED):
                continue
            with handle.lock:
                tickets = list(handle.in_flight.items())
                handle.in_flight.clear()
            for ticket, kind in tickets:
                results.append(
                    StageResult(
                        kind=kind,
                        ticket=ticket,
                        error=RuntimeError(
                            f"stage worker {handle.kind} "
                            f"{handle.state.value} before running the job"
                        ),
                    )
                )
        return results

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def shutdown(self, timeout_s: float = 60.0) -> None:
        """Stop both workers; error out anything that never ran."""
        self._shutdown = True
        for handle in self._workers.values():
            if handle.state is WorkerState.RUNNING:
                handle.state = WorkerState.DRAINING
                try:
                    handle.jobs.put(None, timeout=timeout_s)
                except queue.Full:
                    logger.error(
                        "Could not deliver shutdown sentinel to %s worker",
                        handle.kind,
                    )
        for handle in self._workers.values():
            handle.thread.join(timeout=timeout_s)
            if handle.thread.is_alive():
                logger.error(
                    "Stage worker %s did not stop within %.0fs",
                    handle.kind,
                    timeout_s,
                )
                handle.state = WorkerState.FAILED
            elif handle.state is not WorkerState.FAILED:
                handle.state = WorkerState.STOPPED
        # Surface any jobs that never ran so pollers do not wait forever.
        for result in self._collect_dead_worker_results():
            self._results.put(result)

    # ------------------------------------------------------------------ #
    # Worker loop                                                          #
    # ------------------------------------------------------------------ #

    def _run_worker(self, kind: str) -> None:
        handle = self._workers[kind]
        try:
            if self._use_cuda and self._device is not None:
                torch.cuda.set_device(self._device)
            while True:
                job = handle.jobs.get()
                if job is None:
                    handle.state = WorkerState.STOPPED
                    return
                result = self._run_job(handle, job)
                # The job ran; it must not be errored again if the worker
                # stops before this result is polled.
                with handle.lock:
                    handle.in_flight.pop(job.ticket, None)
                self._results.put(result)
        except BaseException:  # noqa: BLE001 - worker death is surfaced to pollers
            handle.state = WorkerState.FAILED
            logger.error("Stage worker %s crashed", kind, exc_info=True)
            raise

    def _run_job(self, handle: _WorkerHandle, job: StageJob) -> StageResult:
        result = StageResult(kind=job.kind, ticket=job.ticket)
        try:
            with torch.no_grad():
                if handle.stream is not None:
                    if job.submit_event is not None:
                        # Wait only for work enqueued before submission.
                        handle.stream.wait_event(job.submit_event)
                    else:
                        handle.stream.wait_stream(
                            torch.cuda.default_stream(self._device)
                        )
                    with torch.cuda.stream(handle.stream):
                        result.value = job.payload()
                    done_event = torch.cuda.Event()
                    done_event.record(handle.stream)
                    result.done_event = done_event
                else:
                    result.value = job.payload()
        except BaseException as e:  # noqa: BLE001 - forwarded to caller
            logger.error("Async %s stage job failed: %s", job.kind, e, exc_info=True)
            result.error = e
            result.done_event = None
        return result
