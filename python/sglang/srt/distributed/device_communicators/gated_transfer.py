# SPDX-License-Identifier: Apache-2.0
"""Gated background transfers: pause any chunked transfer for foreground ops.

A background bulk transfer that overlaps the forward (a KV-cache gather over
NVLink, host <-> device copies over PCIe) slows any latency-critical forward
op that shares its link, and nothing preempts work already on the GPU. Gating
splits the transfer into chunks that each wait on a pause word, so a
foreground op stops the transfer at the next chunk boundary and restarts it
without resubmitting anything::

    chain = TransferChain(device, get_transfer_gate(Link.PCIE_H2D, device))
    chain.run(chunks, after=torch.cuda.current_stream(), done=landed)
    ...
    with paused_background_transfers(Link.PCIE_H2D):
        latency_critical_op()  # the chunks wait at their next gate
    ...
    chain.wait(torch.cuda.current_stream(), landed)  # before reading the data

A chunk is any callable ``chunk(stream)`` (``ChunkOp``). It runs with
``stream`` current, so it may enqueue its work through any API that uses the
current stream (``tensor.copy_``, ``PyNcclCommunicator`` calls, the HiCache
transfer kernels) or through ``stream`` explicitly. Gating therefore does not
care what moves the bytes: ``copy_chunks`` / ``copy_tensor_chunks`` build
copy-engine chunks (D2D, P2P, pinned H2D / D2H), and a new gated transfer
only implements its chunks, for example one NCCL call per chunk. A chunk must
not allocate, sync or wait on the host: it may be enqueued behind a parked
gate. The chunk boundary is the only preemption point, so the chunk size
bounds how long a pause takes to land.

Two-sided chunks (NCCL): a rank's pause stops only that rank's chunks. A peer
whose own chunk already passed its gate keeps that chunk in flight, waiting
inside NCCL for the paused rank, for the whole pause: on a stream memop for
copy-engine collectives (which holds its stream's hardware queue, see below),
on spinning CTAs for SM kernels (which take SMs from the peer's forward).
Prefer copy-engine collectives and small chunks. Give gated chunks a
communicator no foreground op uses: NCCL orders every op of a communicator
through one internal stream, so a parked chunk would also hold a foreground
collective issued on it later.

Gate (``TransferGate``, one per link and device): a pause word in pinned,
device-mapped host memory. A pause writes a nonzero sequence number and a
resume writes 0, both with ``cuStreamWriteValue32`` in the order of the
bracketing stream; every chunk is preceded by ``cuStreamWaitValue32(== 0)``.
One host thread at a time owns a paused gate; a bracket on another thread
leaves it alone, so a resume is only ever written on the stream that paused.

Chain (``TransferChain``): a stream may evaluate a wait ahead of the work
queued before it, which would let several chunks pass their gates before a
pause lands. Chunks therefore alternate between the chain's two streams, and
each waits for an event recorded after what precedes it (the first chunk:
the run's setup ops and the whole previous run): a gate is read only once the
work before it has retired, so at most one chunk drains after a pause. The
chain creates its streams itself (``cuStreamCreate``): a pooled
``torch.cuda.Stream`` may be the very stream the forward runs on.

In flight: a bracket pauses only gates whose chains have unretired chunks
(``cuEventQuery`` on each chain's gated tail; a host flag keeps the drained
case free), so owners need not arm or disarm anything. The skip is sound when
the chain's ``after`` stream is the bracketing stream: a run enqueued after
the check then starts after the bracketed op.

Hazards. A parked gate wait holds the hardware queue of its stream, and later
work of every stream mapped to that queue (``CUDA_DEVICE_MAX_CONNECTIONS``)
stalls behind it. Anything the resume depends on that lands behind a parked
wait therefore hangs the forward:
- the bracketing stream, or a peer's stream its collective needs, sharing a
  hardware queue with a parked wait (a gate, or a two-sided chunk waiting
  for a paused peer);
- a module loaded lazily (a kernel's first launch, an NCCL lazy connect) or a
  device-wide sync (``cudaFree`` when an allocation misses the cache) inside a
  bracket: bracket the communication call, after its outputs exist;
- a stream waiting on the chain inside the bracket (``TransferChain.wait``
  refuses that).
Each of these ends as a pause that never resumes, so a watchdog thread
(``SGLANG_GATED_TRANSFER_WATCHDOG_MS``) forces a gate open with a host store
when the same pause has been held for that long, and logs it: the transfer then
overlaps the foreground op, a slowdown instead of a hang. The host-mapped word
costs about 1 us more per pause/resume pair than a device word (H200).

Brackets are eager-only: no-ops under ``torch.compile`` tracing and stream
capture, and never part of a captured graph. Chains are not capturable either
(a captured chain loses the one-chunk drain bound).
"""

from __future__ import annotations

import ctypes
import enum
import logging
import threading
import time
import weakref
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# cuda.h: CUstreamWaitValue_flags / CUstreamWriteValue_flags / CUevent_flags /
# CUstream_flags / CUresult / CUstreamCaptureMode / CUmemhostalloc flags
CU_STREAM_WAIT_VALUE_EQ = 0x1
# CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER is rejected (CUDA_ERROR_INVALID_VALUE)
# on B200 / driver 13.4, so every write keeps the default fence.
CU_STREAM_WRITE_VALUE_DEFAULT = 0x0
_CU_EVENT_DISABLE_TIMING = 0x2
_CU_STREAM_NON_BLOCKING = 0x1
_CU_STREAM_CAPTURE_MODE_RELAXED = 2
_CU_MEMHOSTALLOC_PORTABLE = 0x1
_CU_MEMHOSTALLOC_DEVICEMAP = 0x2
_CUDA_SUCCESS = 0
_CUDA_ERROR_NOT_READY = 600


class CudaDriver:
    """The driver entry points gated transfers need, bound once via ctypes
    (about 2 us per call, mostly driver time). The ``_v2`` names matter: the
    bare stream memop symbols are the legacy, non-capturable ones. ``bind``
    adds an entry point for a transfer that needs more."""

    _instance: Optional[CudaDriver] = None

    def __init__(self):
        lib = ctypes.CDLL("libcuda.so.1")
        void_p, u64, u32, uint = (
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        )
        self._get_error_name = lib.cuGetErrorName
        self._get_error_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._lib = lib
        bind = self.bind
        self.wait32 = bind("cuStreamWaitValue32_v2", [void_p, u64, u32, uint])
        self.write32 = bind("cuStreamWriteValue32_v2", [void_p, u64, u32, uint])
        self.memcpy = bind("cuMemcpyAsync", [u64, u64, ctypes.c_size_t, void_p])
        self.mem_host_alloc = bind(
            "cuMemHostAlloc", [ctypes.POINTER(void_p), ctypes.c_size_t, uint]
        )
        self.mem_host_get_device_pointer = bind(
            "cuMemHostGetDevicePointer_v2", [ctypes.POINTER(u64), void_p, uint]
        )
        self.stream_create = bind("cuStreamCreate", [ctypes.POINTER(void_p), uint])
        self.stream_destroy = bind("cuStreamDestroy_v2", [void_p])
        self.stream_is_capturing = bind(
            "cuStreamIsCapturing", [void_p, ctypes.POINTER(ctypes.c_int)]
        )
        self.exchange_capture_mode = bind(
            "cuThreadExchangeStreamCaptureMode", [ctypes.POINTER(ctypes.c_int)]
        )
        self.event_create = bind("cuEventCreate", [ctypes.POINTER(void_p), uint])
        self.event_destroy = bind("cuEventDestroy_v2", [void_p])
        self.event_record = bind("cuEventRecord", [void_p, void_p])
        self.event_query = bind("cuEventQuery", [void_p])
        self.stream_wait_event = bind("cuStreamWaitEvent", [void_p, void_p, uint])

    def bind(self, name: str, argtypes) -> Callable[..., int]:
        """The driver entry point ``name`` returning a ``CUresult``."""
        fn = getattr(self._lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    @classmethod
    def get(cls) -> CudaDriver:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def check(self, result: int, what: str) -> None:
        if result != _CUDA_SUCCESS:
            name = ctypes.c_char_p()
            self._get_error_name(result, ctypes.byref(name))
            raise RuntimeError(
                f"{what} failed: {(name.value or b'?').decode()} ({result})"
            )

    def create_event(self) -> int:
        event = ctypes.c_void_p()
        self.check(
            self.event_create(ctypes.byref(event), _CU_EVENT_DISABLE_TIMING),
            "cuEventCreate",
        )
        return event.value

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self.check(
            self.stream_create(ctypes.byref(stream), _CU_STREAM_NON_BLOCKING),
            "cuStreamCreate",
        )
        return stream.value

    def is_capturing(self, stream_handle: int) -> bool:
        status = ctypes.c_int()
        self.check(
            self.stream_is_capturing(stream_handle, ctypes.byref(status)),
            "cuStreamIsCapturing",
        )
        return status.value != 0

    def events_done(self, events: Sequence[int]) -> bool:
        """Whether every event has completed. Queries in relaxed capture
        mode: a query is otherwise an unsafe call that invalidates a global
        capture running on another thread."""
        mode = ctypes.c_int(_CU_STREAM_CAPTURE_MODE_RELAXED)
        self.check(self.exchange_capture_mode(ctypes.byref(mode)), "capture mode")
        try:
            for event in events:
                result = self.event_query(event)
                if result == _CUDA_ERROR_NOT_READY:
                    return False
                self.check(result, "cuEventQuery")
            return True
        finally:
            self.exchange_capture_mode(ctypes.byref(mode))


class Link(enum.Flag):
    """The interconnect a transfer loads. A gate belongs to one link; a
    foreground bracket names the links its op needs. PCIe is full duplex, so
    each direction is its own link."""

    NVLINK = 1
    PCIE_H2D = 2
    PCIE_D2H = 4
    PCIE = PCIE_H2D | PCIE_D2H


# One chunk of a background transfer: enqueues its work on the given stream.
# It must not allocate, sync or wait on the host (it may be enqueued behind a
# parked gate).
ChunkOp = Callable[[torch.cuda.Stream], None]


class CopyChunk:
    """A chunk made of copy-engine copies, ``(dst, src, nbytes)`` each, all
    behind one gate."""

    __slots__ = ("copies",)

    def __init__(self, copies: Tuple[Tuple[int, int, int], ...]):
        self.copies = copies

    def __call__(self, stream: torch.cuda.Stream) -> None:
        drv = CudaDriver.get()
        for dst, src, nbytes in self.copies:
            drv.check(
                drv.memcpy(dst, src, nbytes, stream.cuda_stream),
                "transfer chunk (cuMemcpyAsync)",
            )


class TransferGate:
    """The pause word of one link's background transfers on one device.

    Use ``get_transfer_gate``: brackets and the watchdog reach gates through
    its registry.
    """

    def __init__(self, link: Link, device_index: int):
        self.link = link
        self.device_index = device_index
        self._drv = CudaDriver.get()
        # Pinned, device-mapped host memory: the GPU waits on it and the
        # watchdog opens it with a plain store, no stream needed. Never freed
        # (the registry lives as long as the process).
        host = ctypes.c_void_p()
        device_ptr = ctypes.c_uint64()
        with torch.cuda.device(device_index):
            self._drv.check(
                self._drv.mem_host_alloc(
                    ctypes.byref(host),
                    64,
                    _CU_MEMHOSTALLOC_PORTABLE | _CU_MEMHOSTALLOC_DEVICEMAP,
                ),
                "cuMemHostAlloc",
            )
            self._drv.check(
                self._drv.mem_host_get_device_pointer(
                    ctypes.byref(device_ptr), host, 0
                ),
                "cuMemHostGetDevicePointer",
            )
        self._word = ctypes.c_uint32.from_address(host.value)
        self._word.value = 0
        self.address = device_ptr.value
        self._seq = 0
        self._lock = threading.Lock()
        self._chains: List[weakref.ref] = []
        # True from a chain's run until a query sees every gated tail retired.
        self._maybe_pending = False
        self._runs = 0
        # Bracket state, owned by one thread at a time.
        self._pause_depth = 0
        self._pause_stream: Optional[torch.cuda.Stream] = None
        self._pause_thread: Optional[int] = None
        # Watchdog: (sequence number, first seen at) of a held pause.
        self._held: Optional[Tuple[int, float]] = None
        self.forced_opens = 0

    def pause(self, stream: torch.cuda.Stream) -> None:
        """Stop this gate's chunks at their next boundary, in ``stream``
        order. Each pause writes a fresh nonzero sequence number."""
        self._seq = self._seq % 0xFFFFFFFF + 1
        self._write(self._seq, stream)

    def resume(self, stream: torch.cuda.Stream) -> None:
        self._write(0, stream)

    def _write(self, value: int, stream: torch.cuda.Stream) -> None:
        self._drv.check(
            self._drv.write32(
                stream.cuda_stream, self.address, value, CU_STREAM_WRITE_VALUE_DEFAULT
            ),
            "pause word (cuStreamWriteValue32)",
        )

    def enqueue_wait(self, stream_handle: int) -> None:
        """Make ``stream_handle`` wait until the gate is open."""
        self._drv.check(
            self._drv.wait32(stream_handle, self.address, 0, CU_STREAM_WAIT_VALUE_EQ),
            "pause gate (cuStreamWaitValue32)",
        )

    def word(self) -> int:
        """The pause word as the GPU last wrote it (0: open)."""
        return self._word.value

    def force_open(self) -> None:
        """Open the gate from the host, behind every stream's back."""
        self._word.value = 0

    def _add_chain(self, chain: TransferChain) -> None:
        with self._lock:
            self._chains = [ref for ref in self._chains if ref() is not None]
            self._chains.append(weakref.ref(chain))

    def _remove_chain(self, chain: TransferChain) -> None:
        with self._lock:
            self._chains = [
                ref for ref in self._chains if ref() is not None and ref() is not chain
            ]

    def _note_run(self) -> None:
        with self._lock:
            self._runs += 1
            self._maybe_pending = True

    def has_pending(self) -> bool:
        """Whether a chain on this gate may still have gated chunks to run."""
        if not self._maybe_pending:
            return False
        with self._lock:
            runs = self._runs
            # Queried under the lock, with the chains held: close() destroys
            # a chain's events only after leaving the list, and a collected
            # chain only once nothing references it.
            chains = [chain for chain in (ref() for ref in self._chains) if chain]
            if not self._drv.events_done([chain._gated_tail for chain in chains]):
                return True
            # A chain run on another thread since the snapshot stays pending.
            if self._runs == runs:
                self._maybe_pending = False
            return False


_gates: List[TransferGate] = []
_gate_index = {}
_registry_lock = threading.Lock()


def get_transfer_gate(link: Link, device_index: Optional[int] = None) -> TransferGate:
    """The process-wide gate of ``link`` on ``device_index`` (default: the
    current device), created on first use. Owners create theirs at init."""
    assert link in (
        Link.NVLINK,
        Link.PCIE_H2D,
        Link.PCIE_D2H,
    ), f"one link per gate, got {link}"
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (link, device_index)
    with _registry_lock:
        gate = _gate_index.get(key)
        if gate is None:
            gate = TransferGate(link, device_index)
            _gate_index[key] = gate
            # Brackets iterate this list without the lock; replace, never
            # mutate it.
            global _gates
            _gates = _gates + [gate]
            _ensure_watchdog()
        return gate


# ---- watchdog -------------------------------------------------------------------

_watchdog: Optional[threading.Thread] = None


def _watchdog_timeout_s() -> float:
    from sglang.srt.environ import envs

    return envs.SGLANG_GATED_TRANSFER_WATCHDOG_MS.get() / 1000.0


def _ensure_watchdog() -> None:
    global _watchdog
    if _watchdog is not None or _watchdog_timeout_s() <= 0:
        return
    _watchdog = threading.Thread(
        target=_watch_gates, name="gated-transfer-watchdog", daemon=True
    )
    _watchdog.start()


def _watch_gates() -> None:
    """Force open any gate whose same pause has been held for the timeout.
    Every hang a pause can cause holds its sequence number forever, while a
    resumed pause gives way to 0 within one foreground op."""
    while True:
        timeout = _watchdog_timeout_s()
        if timeout <= 0:
            return
        time.sleep(max(timeout / 4, 0.001))
        now = time.monotonic()
        for gate in _gates:
            check_gate_held(gate, now, timeout)


def check_gate_held(gate: TransferGate, now: float, timeout: float) -> bool:
    """One watchdog sample of ``gate``; True if it forced the gate open."""
    seq = gate.word()
    held = gate._held
    if not seq:
        gate._held = None
        return False
    if held is None or held[0] != seq:
        gate._held = (seq, now)
        return False
    if now - held[1] < timeout:
        return False
    gate.force_open()
    gate._held = None
    gate.forced_opens += 1
    logger.error(
        "Gated transfer watchdog: the %s gate on cuda:%d stayed paused for "
        "%.0f ms (pause #%d); forcing it open. The paused transfer now "
        "overlaps the foreground op. Some work its resume depends on is stuck "
        "behind a parked chunk: a hardware queue shared with a gated stream "
        "(raise CUDA_DEVICE_MAX_CONNECTIONS), a lazy module load or a device "
        "sync inside a pause bracket, or a wait on the paused transfer.",
        gate.link.name,
        gate.device_index,
        (now - held[1]) * 1000,
        seq,
    )
    return True


# ---- chains -----------------------------------------------------------------------


class TransferChain:
    """Runs a background transfer's chunks (any callables) on the chain's own
    streams, each chunk behind ``gate`` if one is given (at most one chunk
    drains after a pause); without a gate, a plain transfer stream.

    One chain per transfer owner; ``run`` calls serialize on the chain's
    primary stream and must come from one host thread at a time.
    """

    def __init__(self, device_index: int, gate: Optional[TransferGate] = None):
        assert gate is None or gate.device_index == device_index
        self.gate = gate
        self.device_index = device_index
        self._drv = drv = CudaDriver.get()
        with torch.cuda.device(device_index):
            handles = [drv.create_stream() for _ in range(2 if gate else 1)]
            self._streams = tuple(
                torch.cuda.ExternalStream(h, device=torch.device("cuda", device_index))
                for h in handles
            )
            # _done[k % 2]: recorded after chunk k, waited on by chunk k + 1.
            self._done = (drv.create_event(), drv.create_event())
            self._after = drv.create_event()
            self._gated_tail = drv.create_event()
            self._last = drv.create_event()
        self._handles = tuple(handles)
        self._closed = False
        self._finalizer = weakref.finalize(
            self,
            _destroy_chain,
            drv,
            self._handles,
            (*self._done, self._after, self._gated_tail, self._last),
        )
        # The driver context may be gone at interpreter exit.
        self._finalizer.atexit = False
        if gate is not None:
            gate._add_chain(self)

    @property
    def stream(self) -> torch.cuda.Stream:
        """The primary stream: setup, finish and ``done`` run on it."""
        assert not self._closed
        return self._streams[0]

    def run(
        self,
        ops: Sequence[ChunkOp],
        after: torch.cuda.Stream,
        *,
        setup: Sequence[ChunkOp] = (),
        finish: Sequence[ChunkOp] = (),
        done: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Enqueue one transfer, ordered after the work ``after`` has queued
        so far: ``setup`` ops, then ``ops`` (the chunks, any ``ChunkOp``
        callables, each behind the gate), then ``finish`` ops; record
        ``done`` once everything has run. Setup and finish ops run ungated on
        the primary stream. Nothing is enqueued on ``after``: consumers wait
        through ``wait``."""
        assert not self._closed
        drv = self._drv
        after_handle = after.cuda_stream
        if drv.is_capturing(after_handle):
            raise RuntimeError("background transfers cannot be graph-captured")
        primary = self._handles[0]
        stream = self._streams[0]
        drv.check(drv.event_record(self._after, after_handle), "cuEventRecord")
        drv.check(drv.stream_wait_event(primary, self._after, 0), "cuStreamWaitEvent")
        # Every op runs with its stream current, so ops written against the
        # current stream (torch ops, PyNcclCommunicator) need no adapting.
        caller_stream = torch.cuda.current_stream(self.device_index)
        try:
            torch.cuda.set_stream(stream)
            for op in setup:
                op(stream)
            if self.gate is None:
                for op in ops:
                    op(stream)
            elif ops:
                self._run_gated(ops)
            for op in finish:
                op(stream)
        finally:
            torch.cuda.set_stream(caller_stream)
            # Also when an op raised: what ran so far is still tracked.
            drv.check(drv.event_record(self._last, primary), "cuEventRecord")
        if done is not None:
            done.record(stream)

    def _run_gated(self, ops: Sequence[ChunkOp]) -> None:
        drv = self._drv
        gate = self.gate
        handles, streams, done = self._handles, self._streams, self._done
        wait_event, record, wait32 = (
            drv.stream_wait_event,
            drv.event_record,
            drv.wait32,
        )
        word = gate.address
        # Every gate is read behind an event wait: chunk k runs on stream
        # (k + 1) % 2 and waits for the event of what precedes it, so chunk 0
        # (on the second stream) waits for the setup ops and the whole
        # previous run, which all ran on the primary stream.
        drv.check(record(done[0], handles[0]), "chunk chain (cuEventRecord)")
        last = 0
        try:
            for k, op in enumerate(ops):
                i = (k + 1) & 1
                handle = handles[i]
                result = wait_event(handle, done[i ^ 1], 0)
                if result:
                    drv.check(result, "chunk chain (cuStreamWaitEvent)")
                result = wait32(handle, word, 0, CU_STREAM_WAIT_VALUE_EQ)
                if result:
                    drv.check(result, "pause gate (cuStreamWaitValue32)")
                torch.cuda.set_stream(streams[i])
                last = i
                op(streams[i])
                result = record(done[i], handle)
                if result:
                    drv.check(result, "chunk chain (cuEventRecord)")
        finally:
            torch.cuda.set_stream(streams[0])
            # Every chunk is ordered before the last; join it if it ran on the
            # second stream.
            if last:
                drv.check(
                    wait_event(handles[0], done[1], 0),
                    "chunk chain join (cuStreamWaitEvent)",
                )
            drv.check(
                record(self._gated_tail, handles[0]), "gated tail (cuEventRecord)"
            )
            gate._note_run()

    def wait(
        self, stream: torch.cuda.Stream, event: Optional[torch.cuda.Event] = None
    ) -> None:
        """Make ``stream`` wait for ``event`` (a ``done`` of a run) or the last
        run. Refused while this thread holds the chain's gate paused: the
        waiting stream would block on chunks parked behind its own pause."""
        assert not self._closed
        gate = self.gate
        if (
            gate is not None
            and gate._pause_depth
            and gate._pause_thread == threading.get_ident()
        ):
            raise RuntimeError(
                "waiting on a gated transfer inside a bracket that paused it "
                "would deadlock"
            )
        if event is not None:
            stream.wait_event(event)
        else:
            self._drv.check(
                self._drv.stream_wait_event(stream.cuda_stream, self._last, 0),
                "cuStreamWaitEvent",
            )

    def retired(self) -> bool:
        """Whether the last run's chunks have all completed."""
        assert not self._closed
        tail = self._gated_tail if self.gate is not None else self._last
        return self._drv.events_done([tail])

    def close(self) -> None:
        """Release the chain's streams and events once its work completes."""
        if not self._closed:
            self._closed = True
            if self.gate is not None:
                self.gate._remove_chain(self)
            self._finalizer()


def _destroy_chain(drv: CudaDriver, streams, events) -> None:
    # Both calls defer the release until queued work completes.
    for event in events:
        drv.event_destroy(event)
    for stream in streams:
        drv.stream_destroy(stream)


# ---- foreground brackets ------------------------------------------------------

_NOT_PAUSED = nullcontext()


class _PausedTransfers:
    __slots__ = ("_links", "_paused")

    def __init__(self, links: Link):
        self._links = links
        self._paused: List[TransferGate] = []

    def __enter__(self):
        me = threading.get_ident()
        paused = self._paused
        capturing = None
        try:
            for gate in _gates:
                if not gate.link & self._links:
                    continue
                owner = gate._pause_thread
                if owner is not None:
                    if owner == me:
                        gate._pause_depth += 1
                        paused.append(gate)
                    # Paused by another thread, which resumes it.
                    continue
                if not gate._maybe_pending:
                    continue
                if capturing is None:
                    capturing = torch.cuda.is_current_stream_capturing()
                if capturing or not gate.has_pending():
                    continue
                with gate._lock:
                    if gate._pause_thread is not None:
                        continue
                    gate._pause_thread = me
                stream = torch.cuda.current_stream(gate.device_index)
                gate._pause_stream = stream
                gate._pause_depth = 1
                paused.append(gate)
                gate.pause(stream)
        except BaseException:
            _release(paused)
            raise

    def __exit__(self, *exc):
        _release(self._paused)
        return False


def _release(paused: List[TransferGate]) -> None:
    """Undo ``paused`` (innermost last). Every gate is released even if a
    resume raises; the first error is raised after the loop."""
    error = None
    while paused:
        gate = paused.pop()
        gate._pause_depth -= 1
        if gate._pause_depth:
            continue
        stream = gate._pause_stream
        gate._pause_stream = None
        gate._pause_thread = None
        # After the bracketed op, on the stream that issued the pause.
        try:
            gate.resume(stream)
        except BaseException as e:
            error = error or e
    if error is not None:
        raise error


def background_transfers_pending(links: Link) -> bool:
    """Host-side hint: whether a bracket on ``links`` may pause anything now
    (some gate has chunks that may still be in flight). Lets a caller skip
    work it does only to make a bracket safe."""
    return any(
        gate.link & links and (gate._maybe_pending or gate._pause_depth)
        for gate in _gates
    )


def paused_background_transfers(links: Link) -> Union[_PausedTransfers, nullcontext]:
    """Bracket a latency-critical op issued on the current stream: every
    background transfer on ``links`` with chunks in flight stops at its next
    chunk boundary for the op's duration.

    Bracket only the communication call: nothing inside may allocate (unless
    a cached block is sure to serve it), sync, load a module, or wait on a
    paused transfer (see the module docstring). Nested brackets pause and
    resume each gate once.
    """
    if torch.compiler.is_compiling() or not _gates:
        return _NOT_PAUSED
    return _PausedTransfers(links)


# ---- copy chunks ----------------------------------------------------------------


def chunk_spans(n: int, chunk: Optional[int]) -> List[Tuple[int, int]]:
    """``(offset, length)`` pieces of ``[0, n)``, ``chunk`` long except the
    last (one piece if ``chunk`` is None)."""
    assert chunk is None or chunk > 0
    if n <= 0:
        return []
    step = n if not chunk else chunk
    return [(off, min(step, n - off)) for off in range(0, n, step)]


def copy_chunks(
    dst: int, src: int, nbytes: int, chunk_bytes: Optional[int]
) -> List[CopyChunk]:
    """Copy-engine copies of ``nbytes`` from ``src`` to ``dst`` in
    ``chunk_bytes`` pieces. Any pair of unified addresses works: device,
    peer or pinned host. Pageable host memory does not: its copies block the
    host, which would hang behind a parked gate."""
    return [
        CopyChunk(((dst + off, src + off, n),))
        for off, n in chunk_spans(nbytes, chunk_bytes)
    ]


def copy_tensor_chunks(
    dst: torch.Tensor, src: torch.Tensor, chunk_bytes: Optional[int]
) -> List[CopyChunk]:
    """``copy_chunks`` between two contiguous tensors of equal size (H2D,
    D2H, D2D or P2P); host tensors must be pinned. The caller keeps both
    alive until the transfer's ``done``."""
    nbytes = dst.numel() * dst.element_size()
    assert nbytes == src.numel() * src.element_size(), "size mismatch"
    assert dst.is_contiguous() and src.is_contiguous()
    for t in (dst, src):
        assert t.is_cuda or t.is_pinned(), "host side of a gated copy must be pinned"
    return copy_chunks(dst.data_ptr(), src.data_ptr(), nbytes, chunk_bytes)
