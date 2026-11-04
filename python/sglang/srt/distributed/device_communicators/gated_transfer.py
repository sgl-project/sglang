# SPDX-License-Identifier: Apache-2.0
"""Gated background transfers: pause any chunked transfer for foreground ops.

A background bulk transfer that overlaps the forward (a KV-cache gather over
NVLink, host <-> device copies over PCIe) slows any latency-critical op that
shares its link, and nothing preempts work already on the GPU. A
``TransferChain`` runs a transfer as chunks that each wait on its link's
pause word, so a foreground bracket stops every transfer on the link at the
next chunk boundary and restarts it without resubmitting anything::

    chain = TransferChain(Link.PCIE_H2D)  # once per owner, on its device
    chain.run(copy_chunks(dst, src, 16 << 20))  # after the current stream
    ...
    with paused_background_transfers(Link.PCIE_H2D):
        latency_critical_op()  # the chunks wait at their next boundary
    ...
    chain.wait()  # the current stream waits for the transfer to land

A chunk is any callable ``chunk(stream)`` (``ChunkOp``), run with ``stream``
current: it may enqueue its work through any API that uses the current
stream (``tensor.copy_``, ``PyNcclCommunicator`` calls, the HiCache transfer
kernels) or through ``stream`` itself, and ``copy_chunks`` builds copy-engine
chunks. A chunk must not allocate, sync or wait on the host, since it may be
enqueued behind a paused gate. The chunk size bounds how long a pause takes
to land. Chunks that are NCCL calls pause only this rank's side: a peer
whose chunk already passed its gate waits inside NCCL for the whole pause
(on SMs for kernel collectives), so prefer copy-engine collectives, small
chunks, and a communicator no foreground op uses (NCCL orders every op of a
communicator).

How it works:
- Gate: one pause word per link and device, in pinned, device-mapped host
  memory. A bracket writes a fresh nonzero sequence number to pause and 0 to
  resume, with ``cuStreamWriteValue32`` in the order of the bracketed op's
  stream. Only the thread that paused a gate resumes it; a bracket on
  another thread leaves that gate alone. A bracket pauses only gates on the
  current device with a run in flight (a host flag keeps the drained case
  free). A run issued once the bracketed op is enqueued, after the
  bracketing stream (the default ``after``), starts behind the op; a run
  issued meanwhile from another thread, or after another stream, may
  overlap a bracket that found its gate drained.
- Chain: a stream may evaluate a wait ahead of the work queued before it,
  which would let several chunks pass their gates before a pause lands.
  Chunks alternate between the chain's two streams and each waits for an
  event of what precedes it (the first chunk: the setup ops and the whole
  previous run), so a gate is read only once the work before it retired and
  at most one chunk per chain drains after a pause. The chain creates its
  streams (``cuStreamCreate``; a pooled ``torch.cuda.Stream`` may be the
  forward's own stream) and never destroys them: torch may still record
  events on a stream a chunk used (freeing a pinned block a ``tensor.copy_``
  used), so a chain lives as long as the process.

Hazards: a parked gate wait holds its stream's hardware queue, and later
work of every stream mapped to that queue (``CUDA_DEVICE_MAX_CONNECTIONS``)
stalls behind it. Anything the resume depends on that lands behind a parked
wait hangs the forward: the bracketing stream (or a peer's stream its
collective needs) sharing a hardware queue with a parked wait, a lazy module
load (a kernel's first launch, an NCCL lazy connect) or a device-wide sync
(``cudaFree`` when an allocation misses the cache) inside a bracket, or a
stream waiting on the paused transfer inside the bracket (``wait`` refuses
that on the pausing thread). So bracket only the communication call, after
its outputs exist. A watchdog thread (``SGLANG_GATED_TRANSFER_WATCHDOG_MS``)
forces open, with a host store, a gate whose same pause has been held that
long, and logs it: the transfer then overlaps the foreground op, a slowdown
instead of a hang.

Brackets are eager-only: no-ops under ``torch.compile`` tracing and stream
capture. Chains refuse capture (a captured chain loses the drain bound).
"""

from __future__ import annotations

import ctypes
import enum
import logging
import threading
import time
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from sglang.srt.utils.cuda_vmm_utils import check_drv

try:
    from cuda.bindings import driver as cu
except ImportError:  # a build without cuda-python: no transfer can run
    cu = None

logger = logging.getLogger(__name__)

_WAIT_EQ = 0x1  # cuda.h: CU_STREAM_WAIT_VALUE_EQ


class Link(enum.Flag):
    """The interconnect a transfer loads. PCIe is full duplex, so each
    direction is its own link."""

    NVLINK = 1
    PCIE_H2D = 2
    PCIE_D2H = 4
    PCIE = PCIE_H2D | PCIE_D2H


# One chunk of a background transfer: enqueues its work on the given stream,
# which is also current while it runs.
ChunkOp = Callable[[torch.cuda.Stream], None]


def _create_event():
    return check_drv(
        cu.cuEventCreate(cu.CUevent_flags.CU_EVENT_DISABLE_TIMING), "cuEventCreate"
    )


def _events_done(events) -> bool:
    """Whether every event has completed. Queries in relaxed capture mode: a
    query is otherwise an unsafe call that invalidates a global capture
    running on another thread."""
    mode = check_drv(
        cu.cuThreadExchangeStreamCaptureMode(
            cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        ),
        "capture mode",
    )
    try:
        for event in events:
            result = cu.cuEventQuery(event)
            if result[0] == cu.CUresult.CUDA_ERROR_NOT_READY:
                return False
            check_drv(result, "cuEventQuery")
        return True
    finally:
        cu.cuThreadExchangeStreamCaptureMode(mode)


# ---- gates --------------------------------------------------------------------


class _Gate:
    """The pause word of one link's chains on one device."""

    def __init__(self, link: Link, device: int):
        self.link = link
        self.device = device
        # Host memory the GPU waits on, so the watchdog can open it with a
        # plain store. Never freed: gates live as long as the process.
        with torch.cuda.device(device):
            host = check_drv(
                cu.cuMemHostAlloc(
                    64, cu.CU_MEMHOSTALLOC_PORTABLE | cu.CU_MEMHOSTALLOC_DEVICEMAP
                ),
                "cuMemHostAlloc",
            )
            address = check_drv(
                cu.cuMemHostGetDevicePointer(host, 0), "cuMemHostGetDevicePointer"
            )
        self.address = int(address)
        self._word = ctypes.c_uint32.from_address(host)
        self._word.value = 0
        self._seq = 0
        self._lock = threading.Lock()
        # Every chain's last-run event, and whether one may be in flight: set
        # by a run, cleared by a query that sees them all retired.
        self._chain_events = []
        self._maybe_pending = False
        # The bracket that paused the gate: its thread resumes it, on the
        # stream that paused it.
        self._pause_thread: Optional[int] = None
        self._pause_stream: Optional[torch.cuda.Stream] = None
        # Watchdog: (sequence number, first seen at) of a held pause.
        self._held: Optional[Tuple[int, float]] = None
        self.forced_opens = 0

    def pause(self, stream: torch.cuda.Stream) -> None:
        """Stop the chunks at their next boundary, in ``stream`` order. Each
        pause writes a fresh nonzero sequence number."""
        self._seq = self._seq % 0xFFFFFFFF + 1
        self._write(self._seq, stream)

    def resume(self, stream: torch.cuda.Stream) -> None:
        self._write(0, stream)

    def _write(self, value: int, stream: torch.cuda.Stream) -> None:
        # Flags 0 keep the memory barrier; NO_MEMORY_BARRIER is rejected on
        # B200 / driver 13.4.
        check_drv(
            cu.cuStreamWriteValue32(stream.cuda_stream, self.address, value, 0),
            "pause word (cuStreamWriteValue32)",
        )

    def word(self) -> int:
        """The pause word as the GPU last wrote it (0: open)."""
        return self._word.value

    def force_open(self) -> None:
        """Open the gate from the host, behind every stream's back."""
        self._word.value = 0

    def note_run(self) -> None:
        with self._lock:
            self._maybe_pending = True

    def has_pending(self) -> bool:
        """Whether a chain on this gate may still have a run in flight."""
        if not self._maybe_pending:
            return False
        with self._lock:
            # A chain notes a run under this lock once it has recorded it, so
            # no run noted meanwhile is cleared.
            self._maybe_pending = not _events_done(self._chain_events)
            return self._maybe_pending


# Brackets and the watchdog iterate this tuple without the lock; it is
# replaced, never mutated.
_gates: Tuple[_Gate, ...] = ()
_registry_lock = threading.Lock()


def _get_gate(link: Link, device: int) -> _Gate:
    global _gates
    with _registry_lock:
        for gate in _gates:
            if gate.link == link and gate.device == device:
                return gate
        gate = _Gate(link, device)
        _gates = _gates + (gate,)
        _start_watchdog()
        return gate


# ---- watchdog -------------------------------------------------------------------

_watchdog: Optional[threading.Thread] = None


def _watchdog_timeout_s() -> float:
    from sglang.srt.environ import envs

    return envs.SGLANG_GATED_TRANSFER_WATCHDOG_MS.get() / 1000.0


def _start_watchdog() -> None:
    global _watchdog
    if _watchdog is None and _watchdog_timeout_s() > 0:
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
            _check_held(gate, now, timeout)


def _check_held(gate: _Gate, now: float, timeout: float) -> bool:
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
        gate.device,
        (now - held[1]) * 1000,
        seq,
    )
    return True


# ---- chains -----------------------------------------------------------------------


class TransferChain:
    """Runs background transfers (any ``ChunkOp`` chunks) on the chain's own
    streams, each chunk behind the pause word of ``link``; ``link=None``
    never pauses (one plain transfer stream).

    One chain per transfer owner, for the life of the process (its streams
    and events are never released). It belongs to the device current at
    construction: call ``run`` and ``wait`` with that device current, from
    one host thread at a time.
    """

    def __init__(self, link: Optional[Link]):
        assert link in (None, Link.NVLINK, Link.PCIE_H2D, Link.PCIE_D2H), (
            f"a chain loads one link, got {link}"
        )
        self.link = link
        self.device = torch.cuda.current_device()
        with torch.cuda.device(self.device):
            self._handles = tuple(
                check_drv(
                    cu.cuStreamCreate(cu.CUstream_flags.CU_STREAM_NON_BLOCKING),
                    "cuStreamCreate",
                )
                for _ in range(1 if link is None else 2)
            )
            self._streams = tuple(
                torch.cuda.ExternalStream(
                    int(h), device=torch.device("cuda", self.device)
                )
                for h in self._handles
            )
            # _chunk_done[i]: after the last chunk on stream i; the next chunk,
            # on the other stream, waits for it.
            self._chunk_done = (_create_event(), _create_event())
            self._after = _create_event()
            self._last = _create_event()
        self._gate = None
        if link is not None:
            self._gate = _get_gate(link, self.device)
            with self._gate._lock:
                self._gate._chain_events.append(self._last)

    def run(
        self,
        chunks: Sequence[ChunkOp],
        *,
        setup: Sequence[ChunkOp] = (),
        finish: Sequence[ChunkOp] = (),
        done: Optional[torch.cuda.Event] = None,
        after: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Enqueue one transfer behind the work queued so far on ``after``
        (default: the current stream): the ``setup`` ops, the ``chunks``
        (each behind the gate), then the ``finish`` ops, all ungated but the
        chunks; then record ``done``. ``after`` only gets an event recorded.

        If an op raises, the work enqueued before it stays tracked: ``done``
        is still recorded, and ``wait`` and the brackets still see it.
        """
        self._check_device()
        caller = torch.cuda.current_stream(self.device)
        after_handle = (caller if after is None else after).cuda_stream
        if check_drv(cu.cuStreamIsCapturing(after_handle), "cuStreamIsCapturing"):
            raise RuntimeError("background transfers cannot be graph-captured")
        primary = self._streams[0]
        check_drv(cu.cuEventRecord(self._after, after_handle), "cuEventRecord")
        check_drv(
            cu.cuStreamWaitEvent(self._handles[0], self._after, 0),
            "cuStreamWaitEvent",
        )
        gated = self._gate is not None and bool(chunks)
        try:
            # Every op runs with its stream current, so ops written against
            # the current stream need no adapting.
            torch.cuda.set_stream(primary)
            for op in setup:
                op(primary)
            if gated:
                self._run_gated(chunks)
            else:
                for op in chunks:
                    op(primary)
            for op in finish:
                op(primary)
        finally:
            torch.cuda.set_stream(caller)
            check_drv(cu.cuEventRecord(self._last, self._handles[0]), "cuEventRecord")
            # Before done: brackets must see the run even if done.record raises.
            if gated:
                self._gate.note_run()
            if done is not None:
                done.record(primary)

    def _run_gated(self, chunks: Sequence[ChunkOp]) -> None:
        handles, streams, chunk_done = self._handles, self._streams, self._chunk_done
        wait_event, record, wait32 = (
            cu.cuStreamWaitEvent,
            cu.cuEventRecord,
            cu.cuStreamWaitValue32,
        )
        word = self._gate.address
        # Chunk k runs on stream (k + 1) % 2 behind the event of what
        # precedes it, so chunk 0 (on the second stream) waits for the setup
        # ops and the whole previous run, which all ran on the primary.
        check_drv(record(chunk_done[0], handles[0]), "chunk chain (cuEventRecord)")
        i = 0
        try:
            for k, op in enumerate(chunks):
                i = (k + 1) & 1
                handle = handles[i]
                result = wait_event(handle, chunk_done[i ^ 1], 0)
                if result[0]:
                    check_drv(result, "chunk chain (cuStreamWaitEvent)")
                result = wait32(handle, word, 0, _WAIT_EQ)
                if result[0]:
                    check_drv(result, "pause gate (cuStreamWaitValue32)")
                torch.cuda.set_stream(streams[i])
                op(streams[i])
                result = record(chunk_done[i], handle)
                if result[0]:
                    check_drv(result, "chunk chain (cuEventRecord)")
        finally:
            torch.cuda.set_stream(streams[0])
            # The last chunk ran on the second stream: join that stream at
            # its end (also when the chunk raised after enqueuing work, so
            # never recorded its event).
            if i:
                check_drv(
                    record(chunk_done[1], handles[1]), "chunk chain (cuEventRecord)"
                )
                check_drv(
                    wait_event(handles[0], chunk_done[1], 0),
                    "chunk chain (cuStreamWaitEvent)",
                )

    def wait(
        self,
        done: Optional[torch.cuda.Event] = None,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Make ``stream`` (default: the current stream) wait for ``done``
        (a run's ``done`` event; default: the last run). Refused inside a
        bracket of this thread that paused this chain: the stream would wait
        on chunks parked behind its own pause."""
        if self._gate is not None and self._gate._pause_thread == threading.get_ident():
            raise RuntimeError(
                "waiting on a gated transfer inside the bracket that paused it "
                "would deadlock"
            )
        self._check_device()
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        if done is not None:
            stream.wait_event(done)
        else:
            check_drv(
                cu.cuStreamWaitEvent(stream.cuda_stream, self._last, 0),
                "cuStreamWaitEvent",
            )

    def _check_device(self) -> None:
        assert torch.cuda.current_device() == self.device, "use a chain on its device"


# ---- foreground brackets ------------------------------------------------------

_NOT_PAUSED = nullcontext()


class _PausedTransfers:
    __slots__ = ("_links", "_paused")

    def __init__(self, links: Link):
        self._links = links
        self._paused: List[_Gate] = []

    def __enter__(self):
        me = threading.get_ident()
        device = stream = None
        try:
            for gate in _gates:
                # A paused gate is resumed by the bracket that paused it: an
                # enclosing one on this thread, or one on another thread.
                if (
                    not gate.link & self._links
                    or not gate._maybe_pending
                    or gate._pause_thread is not None
                ):
                    continue
                if device is None:
                    if torch.cuda.is_current_stream_capturing():
                        return
                    device = torch.cuda.current_device()
                if gate.device != device or not gate.has_pending():
                    continue
                if stream is None:
                    stream = torch.cuda.current_stream(device)
                with gate._lock:
                    if gate._pause_thread is not None:
                        continue
                    gate._pause_thread = me
                gate._pause_stream = stream
                self._paused.append(gate)
                gate.pause(stream)
        except BaseException:
            _release(self._paused)
            raise

    def __exit__(self, *exc):
        _release(self._paused)
        return False


def _release(paused: List[_Gate]) -> None:
    """Resume the gates a bracket paused, on the stream that paused each,
    after the bracketed op. Every gate is released even if a resume raises
    (the first error is raised after the loop), and each stays owned until
    its resume is enqueued: a pause another thread enqueued before it would
    be undone by it."""
    error = None
    while paused:
        gate = paused.pop()
        try:
            gate.resume(gate._pause_stream)
        except BaseException as e:
            error = error or e
        finally:
            gate._pause_stream = None
            gate._pause_thread = None
    if error is not None:
        raise error


def paused_background_transfers(links: Link) -> Union[_PausedTransfers, nullcontext]:
    """Bracket a latency-critical op issued on the current stream: every
    background transfer on ``links`` with chunks in flight on this device
    stops at its next chunk boundary for the op's duration.

    Bracket only the communication call: nothing inside may allocate (unless
    a cached block is sure to serve it), sync, load a module, or wait on a
    paused transfer (see the module docstring). Nested brackets pause and
    resume each gate once.
    """
    if torch.compiler.is_compiling() or not _gates:
        return _NOT_PAUSED
    return _PausedTransfers(links)


# ---- copy chunks ----------------------------------------------------------------


class CopyChunk:
    """A chunk that copies ``nbytes`` from ``src`` to ``dst`` on the copy
    engine. Any unified addresses work (device, peer or pinned host) but
    pageable host memory: its copies block the host, which would hang behind
    a paused gate."""

    __slots__ = ("dst", "src", "nbytes")

    def __init__(self, dst: int, src: int, nbytes: int):
        self.dst, self.src, self.nbytes = dst, src, nbytes

    def __call__(self, stream: torch.cuda.Stream) -> None:
        check_drv(
            cu.cuMemcpyAsync(self.dst, self.src, self.nbytes, stream.cuda_stream),
            "transfer chunk (cuMemcpyAsync)",
        )


def chunk_spans(n: int, chunk: Optional[int]) -> List[Tuple[int, int]]:
    """``(offset, length)`` pieces of ``[0, n)``, ``chunk`` long except the
    last (one piece if ``chunk`` is None)."""
    assert chunk is None or chunk > 0
    step = chunk or n
    return [(off, min(step, n - off)) for off in range(0, n, step)] if n > 0 else []


def copy_chunks(
    dst: torch.Tensor, src: torch.Tensor, chunk_bytes: Optional[int] = None
) -> List[CopyChunk]:
    """Chunks copying ``src`` into ``dst``, ``chunk_bytes`` each (one chunk if
    None): contiguous tensors of equal size, in any direction (H2D, D2H, D2D,
    P2P); host tensors must be pinned. Keep both alive until the transfer is
    done."""
    nbytes = dst.numel() * dst.element_size()
    assert nbytes == src.numel() * src.element_size(), "size mismatch"
    assert dst.is_contiguous() and src.is_contiguous()
    for t in (dst, src):
        assert t.is_cuda or t.is_pinned(), "host side of a gated copy must be pinned"
    d, s = dst.data_ptr(), src.data_ptr()
    return [
        CopyChunk(d + off, s + off, n) for off, n in chunk_spans(nbytes, chunk_bytes)
    ]
