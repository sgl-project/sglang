# SPDX-License-Identifier: Apache-2.0
"""Gated background transfers: pause any chunked transfer for foreground ops.

A background bulk transfer that overlaps the forward (a KV-cache gather over
NVLink, host <-> device copies over PCIe) slows any latency-critical op that
shares its link, and nothing preempts work already on the GPU. A
``TransferChain`` runs a transfer as chunks that each wait on its link's
pause word, so a foreground bracket stops every transfer on the link at the
next chunk boundary and restarts it without resubmitting anything::

    chain = TransferChain(Link.PCIE_H2D)  # once per owner, on its device
    landed = chain.run(copy_chunks(dst, src, 16 << 20))  # returns at once
    ...
    with paused_background_transfers(Link.PCIE_H2D):
        latency_critical_op()  # the chunks wait at their next boundary
    ...
    chain.wait(landed)  # the current stream waits for the transfer to land

A chunk is any callable ``chunk(stream)`` (``ChunkOp``), run with ``stream``
current: it may enqueue its work through any API that uses the current
stream (``tensor.copy_``, ``PyNcclCommunicator`` calls, the HiCache transfer
kernels) or through ``stream`` itself, and ``copy_chunks`` builds copy-engine
chunks. A chunk may run after ``run`` returned, on the chain's feeder thread
(below), so it must not depend on state its caller scopes around ``run``
(a chunk calling NCCL enters ``change_state`` itself). It must not allocate,
sync or wait on the host, since it may be enqueued behind a paused gate, and
should enqueue little (a copy, a collective): the chain bounds the ops it
keeps queued, not what each one enqueues. The chunk size bounds how long a
pause takes to land. Chunks that are NCCL calls
pause only this rank's side: a peer whose chunk already passed its gate
waits inside NCCL for the whole pause (on SMs for kernel collectives), so
prefer copy-engine collectives, small chunks, and a communicator no
foreground op uses (NCCL orders every op of a communicator).

How it works:
- Gate: one pause word per link and device, in pinned, device-mapped host
  memory. A bracket writes a fresh nonzero sequence number to pause and 0 to
  resume, with ``cuStreamWriteValue32`` in the order of the bracketed op's
  stream. Only the thread that paused a gate resumes it; a bracket on
  another thread leaves that gate alone. A bracket pauses only gates on the
  current device with a run that has not landed. A run issued once the
  bracketed op is enqueued, after the bracketing stream (the default
  ``after``), starts behind the op; a run issued meanwhile from another
  thread, or after another stream, may overlap a bracket that found its gate
  drained.
- Chain: a stream may evaluate a wait ahead of the work queued before it,
  which would let several chunks pass their gates before a pause lands.
  Chunks alternate between the chain's two streams and each waits for an
  event of the op before it (the first chunk of a run: its setup ops and the
  whole previous run), so a gate is read only once the work before it
  retired and at most one chunk per chain drains after a pause. The chain
  creates its streams (``cuStreamCreate``; a pooled ``torch.cuda.Stream`` may
  be the forward's own stream) and never destroys them: torch may still
  record events on a stream a chunk used (freeing a pinned block a
  ``tensor.copy_`` used), so a chain lives as long as the process.
- Submission: a chain keeps at most ``_WINDOW`` ops queued on the GPU, since
  a stream queue full of chunks parked behind a pause blocks the thread
  that enqueues more. ``run`` enqueues at most a window of ops itself, and
  the chain's feeder thread enqueues the rest, in order, as earlier ops
  retire: ``run`` never waits for the GPU, and does not push back on a
  caller that outruns its link either (its runs queue up on the host).
  After a run's last op the GPU writes the run's ticket to a host-mapped
  word, which ``query`` reads. ``wait`` makes a stream wait for it only
  once the whole run is enqueued: a stream waiting on work not enqueued yet
  can hold a hardware queue, or the driver, that the rest of the run needs.
  So ``wait`` blocks the host while a run longer than the window is still
  queued, behind a pause or a busy link: issue whatever its ops wait on
  first. If the chain's own driver calls fail, it fails for good: its
  pending runs are dropped and raise instead of landing.

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

import collections
import ctypes
import enum
import functools
import logging
import threading
import time
from contextlib import nullcontext
from typing import Callable, Deque, List, NoReturn, Optional, Sequence, Tuple, Union

import torch

from sglang.srt.utils.cuda_vmm_utils import check_drv

try:
    from cuda.bindings import driver as cu
except ImportError:  # a build without cuda-python: no transfer can run
    cu = None

logger = logging.getLogger(__name__)

# cuda.h CUstreamWaitValue_flags: EQ, and GEQ ((int32_t)(*addr - value) >= 0).
_WAIT_EQ = 0x1
_WAIT_GEQ = 0x0
# Ops a chain keeps queued on the GPU. Its streams take about a thousand
# single-copy chunks before enqueueing more blocks the host (B200); this
# leaves room for chunks that enqueue more work. Arbitrary below that.
_WINDOW = 64


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


def _create_event(blocking: bool = False):
    flags = cu.CUevent_flags.CU_EVENT_DISABLE_TIMING
    if blocking:  # a host wait sleeps instead of spinning a core
        flags |= cu.CUevent_flags.CU_EVENT_BLOCKING_SYNC
    return check_drv(cu.cuEventCreate(flags), "cuEventCreate")


def _relax_capture_mode():
    """Let this thread query and wait on events while another thread runs a
    global-mode capture, which those calls would otherwise invalidate.
    Returns the previous mode."""
    return check_drv(
        cu.cuThreadExchangeStreamCaptureMode(
            cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        ),
        "capture mode",
    )


def _event_done(event) -> bool:
    mode = _relax_capture_mode()
    try:
        result = cu.cuEventQuery(event)
        if result[0] == cu.CUresult.CUDA_ERROR_NOT_READY:
            return False
        check_drv(result, "cuEventQuery")
        return True
    finally:
        cu.cuThreadExchangeStreamCaptureMode(mode)


def _reached(ticket: int, target: int) -> bool:
    """Whether the 32-bit run ticket ``ticket`` is at or past ``target``."""
    return (ticket - target) & 0xFFFFFFFF < 0x80000000


def _bind_context(device: int) -> None:
    """Make ``device``'s primary context current if this thread has none:
    torch binds it at a thread's first runtime call, and a driver call here
    may come first."""
    result = cu.cuCtxGetCurrent()
    if not result[0] and int(result[1]):
        return
    check_drv(result, "cuCtxGetCurrent")
    handle = check_drv(cu.cuDeviceGet(device), "cuDeviceGet")
    context = check_drv(cu.cuDevicePrimaryCtxRetain(handle), "cuDevicePrimaryCtxRetain")
    check_drv(cu.cuCtxSetCurrent(context), "cuCtxSetCurrent")


def _host_word() -> Tuple[ctypes.c_uint32, int]:
    """A zeroed uint32 in pinned, device-mapped host memory, never freed, and
    its device address: streams wait on it or write it, the host reads or
    writes it directly. Needs the device's context current."""
    host = check_drv(
        cu.cuMemHostAlloc(
            64, cu.CU_MEMHOSTALLOC_PORTABLE | cu.CU_MEMHOSTALLOC_DEVICEMAP
        ),
        "cuMemHostAlloc",
    )
    address = check_drv(
        cu.cuMemHostGetDevicePointer(host, 0), "cuMemHostGetDevicePointer"
    )
    word = ctypes.c_uint32.from_address(host)
    word.value = 0
    return word, int(address)


# ---- gates --------------------------------------------------------------------


class _Gate:
    """The pause word of one link's chains on one device."""

    def __init__(self, link: Link, device: int):
        self.link = link
        self.device = device
        # The watchdog opens it with a plain host store.
        self._word, self.address = _host_word()
        self._seq = 0
        # Replaced, never mutated: brackets read it without the lock.
        self._chains: Tuple[TransferChain, ...] = ()
        self._lock = threading.Lock()
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

    def busy(self) -> bool:
        """Whether a chain on this gate has a run that has not landed."""
        for chain in self._chains:
            if chain._landed.value != chain._issued:
                return True
        return False


# Brackets and the watchdog iterate this tuple without the lock; it is
# replaced, never mutated.
_gates: Tuple[_Gate, ...] = ()
_registry_lock = threading.Lock()


def _add_chain(link: Link, device: int, chain: TransferChain) -> _Gate:
    """The gate of ``link`` on ``device``, created on first use, with
    ``chain`` added to it."""
    global _gates
    with _registry_lock:
        gate = next((g for g in _gates if g.link == link and g.device == device), None)
        if gate is None:
            gate = _Gate(link, device)
            _gates = _gates + (gate,)
            _start_watchdog()
        gate._chains = gate._chains + (chain,)
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


class _Run:
    """A run's ops, enqueued in order: the wait for its ``after`` work, the
    setup ops, the chunks (``gated``), the finish ops, the ticket write."""

    __slots__ = ("ops", "gated", "ticket", "next")

    def __init__(self, ops: List[ChunkOp], gated: range, ticket: int):
        self.ops, self.gated, self.ticket = ops, gated, ticket
        self.next = 0  # the next op to enqueue


class TransferChain:
    """Runs background transfers (any ``ChunkOp`` chunks) on the chain's own
    streams, each chunk behind the pause word of ``link``; ``link=None``
    never pauses (one plain transfer stream).

    One chain per transfer owner, for the life of the process (its streams,
    events and feeder thread are never released). It belongs to the device
    current at construction: call ``run`` and ``wait`` with that device
    current, and ``run`` from one host thread at a time.
    """

    def __init__(self, link: Optional[Link]):
        assert link in (None, Link.NVLINK, Link.PCIE_H2D, Link.PCIE_D2H), (
            f"a chain loads one link, got {link}"
        )
        self.link = link
        self.device = torch.cuda.current_device()
        _bind_context(self.device)
        self._handles = tuple(
            check_drv(
                cu.cuStreamCreate(cu.CUstream_flags.CU_STREAM_NON_BLOCKING),
                "cuStreamCreate",
            )
            for _ in range(1 if link is None else 2)
        )
        self._streams = tuple(
            torch.cuda.ExternalStream(int(h), device=torch.device("cuda", self.device))
            for h in self._handles
        )
        # _ring[j % _WINDOW] is recorded after op j: the next op, on the
        # other stream, waits for it, and op j + _WINDOW is enqueued once it
        # has completed. Only the events _make_room waits on block: recording
        # a blocking event costs ~1 us more on the host (B200).
        half = _WINDOW // 2
        self._ring = tuple(
            _create_event(blocking=j % half == half - 1) for j in range(_WINDOW)
        )
        self._spare_events: List = []  # `after` events whose waits are enqueued
        # The ticket of the last run that landed, written by the GPU.
        self._landed, self._landed_address = _host_word()
        self._issued = 0  # the last run's ticket
        self._enqueued_run = 0  # the last run with all its ops enqueued
        self._enqueued = 0  # ops enqueued on the GPU
        self._retired = 0  # of which known to have completed
        self._stream = 0  # the stream of the last op enqueued
        self._pending: Deque[_Run] = collections.deque()  # not fully enqueued
        # _pending, _issued, _enqueued_run, _error, _failed, the wakeups.
        self._lock = threading.Lock()
        self._more = threading.Condition(self._lock)
        self._run_enqueued = threading.Condition(self._lock)
        self._submitting = threading.Lock()  # held while enqueuing ops
        self._error: Optional[BaseException] = None  # an op's, raised once
        self._failed: Optional[BaseException] = None  # the chain's, for good
        self._gate = None if link is None else _add_chain(link, self.device, self)
        self._feeder = threading.Thread(
            target=self._feed, name="gated-transfer-feeder", daemon=True
        )
        self._feeder.start()

    def run(
        self,
        chunks: Sequence[ChunkOp],
        *,
        setup: Sequence[ChunkOp] = (),
        finish: Sequence[ChunkOp] = (),
        after: Optional[torch.cuda.Stream] = None,
    ) -> int:
        """Enqueue one transfer behind the work queued so far on ``after``
        (default: the current stream): the ``setup`` ops, the ``chunks``
        (each behind the gate), then the ``finish`` ops, all ungated but the
        chunks. Returns its ticket for ``wait`` and ``query``: a 32-bit
        count, valid for the next 2^31 runs.

        Never waits for the GPU or the feeder thread, and enqueues at most a
        window of ops itself: the feeder thread enqueues the rest, in order,
        as earlier ones retire. An op that raises skips the rest of its run,
        whose ticket still lands once the work enqueued before has run; the
        error is raised here, or by the next ``run`` or ``wait`` if the
        feeder thread hit it.
        """
        self._use_device()
        self._raise_feeder_error()
        caller = torch.cuda.current_stream(self.device)
        after_handle = (caller if after is None else after).cuda_stream
        if check_drv(cu.cuStreamIsCapturing(after_handle), "cuStreamIsCapturing"):
            raise RuntimeError("background transfers cannot be graph-captured")
        try:
            event = self._spare_events.pop()
        except IndexError:
            event = _create_event()
        check_drv(cu.cuEventRecord(event, after_handle), "cuEventRecord")
        ticket = (self._issued + 1) & 0xFFFFFFFF
        first = 1 + len(setup)
        run = _Run(
            [
                functools.partial(self._wait_after, event),
                *setup,
                *chunks,
                *finish,
                functools.partial(self._write_ticket, ticket),
            ],
            range(first, first + len(chunks) if self._gate is not None else first),
            ticket,
        )
        with self._lock:
            if self._failed is not None:
                self._raise_failed()
            self._issued = ticket
            self._pending.append(run)
        error = None
        if self._submitting.acquire(blocking=False):  # else the feeder is at it
            try:
                error = self._enqueue_pending(block=False, budget=_WINDOW)
            finally:
                self._submitting.release()
                torch.cuda.set_stream(caller)
        if self._pending:
            with self._lock:
                self._more.notify()
        if error is not None:
            raise error
        return ticket

    def wait(
        self,
        ticket: Optional[int] = None,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Make ``stream`` (default: the current stream) wait until the run
        of ``ticket`` (default: the last run) has landed. Blocks the host
        until that run is fully enqueued (see the module docstring). Refused
        inside a bracket of this thread that paused this chain: the stream
        would wait on chunks parked behind its own pause."""
        if self._gate is not None and self._gate._pause_thread == threading.get_ident():
            raise RuntimeError(
                "waiting on a gated transfer inside the bracket that paused it "
                "would deadlock"
            )
        self._use_device()
        ticket = self._issued if ticket is None else ticket
        with self._lock:
            if not _reached(self._issued, ticket):
                raise ValueError(f"ticket {ticket} is not one of the last 2^31 runs")
            while not _reached(self._enqueued_run, ticket):
                if self._failed is not None:
                    self._raise_failed()
                self._run_enqueued.wait()
        self._raise_feeder_error()
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        check_drv(
            cu.cuStreamWaitValue32(
                stream.cuda_stream, self._landed_address, ticket, _WAIT_GEQ
            ),
            "cuStreamWaitValue32",
        )

    def query(self, ticket: Optional[int] = None) -> bool:
        """Whether the run of ``ticket`` (default: the last run) has landed."""
        ticket = self._issued if ticket is None else ticket
        if _reached(self._landed.value, ticket):
            return True
        if self._failed is not None and not _reached(self._enqueued_run, ticket):
            self._raise_failed()
        return False

    def _use_device(self) -> None:
        assert torch.cuda.current_device() == self.device, "use a chain on its device"
        _bind_context(self.device)

    def _raise_failed(self) -> NoReturn:
        raise RuntimeError(
            "the gated transfer chain failed and dropped its pending runs"
        ) from self._failed

    def _raise_feeder_error(self) -> None:
        if self._error is None:
            return
        with self._lock:
            error, self._error = self._error, None
        if error is not None:
            raise RuntimeError(
                "a gated transfer op failed on the feeder thread"
            ) from error

    def _enqueue_pending(
        self, block: bool, budget: Optional[int] = None
    ) -> Optional[BaseException]:
        """Enqueue pending ops, oldest first, while the window has room
        (``block``: wait for room; the feeder thread) and ``budget`` lasts.
        An op that raises skips the rest of its run but the ticket write; the
        first such error is returned, or on the feeder thread recorded before
        the run counts as enqueued. If the chain's own driver calls fail, or
        an interrupt leaves its state unknown, the chain fails (``_fail``)
        and the error is raised."""
        error = None
        try:
            while self._pending:
                run = self._pending[0]
                ops, last = run.ops, len(run.ops) - 1
                while run.next <= last:
                    if budget is not None:
                        if budget == 0:
                            return error
                        budget -= 1
                    full = self._enqueued - self._retired >= _WINDOW
                    if full and not self._make_room(block):
                        return error
                    i = run.next
                    run.next = i + 1
                    failed = self._enqueue(ops[i], i in run.gated)
                    if failed is None:
                        continue
                    if i == 0 or i == last:  # the chain's own driver calls
                        raise failed
                    if block:
                        with self._lock:
                            self._error = self._error or failed
                        logger.error(
                            "Gated transfer: an op failed on the feeder thread",
                            exc_info=failed,
                        )
                    error = error or failed
                    run.next = last
                with self._lock:
                    self._pending.popleft()
                    self._enqueued_run = run.ticket
                    self._run_enqueued.notify_all()
            return error
        except BaseException as e:
            self._fail(e)
            raise

    def _fail(self, error: BaseException) -> None:
        """Fail the chain for good (the caller holds ``_submitting``): drop
        every pending run; ``run`` raises from now on, and so do ``wait``
        and ``query`` on a dropped run. No stream waits on a dropped run:
        ``wait`` enqueues only behind a fully enqueued one."""
        with self._lock:
            self._failed = self._failed or error
            self._pending.clear()
            self._run_enqueued.notify_all()

    def _make_room(self, block: bool) -> bool:
        """Whether the oldest half of the window has completed (``block``:
        wait for it). The window fills at a whole number of halves, so the
        op waited on is the last of a half, whose event blocks."""
        target = self._enqueued - _WINDOW // 2 - 1
        event = self._ring[target % _WINDOW]
        if block:
            check_drv(cu.cuEventSynchronize(event), "cuEventSynchronize")
        elif not _event_done(event):
            return False
        self._retired = target + 1
        return True

    def _enqueue(self, op: ChunkOp, gated: bool) -> Optional[BaseException]:
        """Enqueue one op: chunks alternate between the two streams and wait
        at the gate, the other ops run on the primary stream. An op on
        another stream than the op before it first waits for that op, so a
        gate is read only once the work before it retired. Returns what the
        op raised; the chain's own driver calls raise."""
        i = 1 - self._stream if gated else 0
        handle, ring, j = self._handles[i], self._ring, self._enqueued
        if i != self._stream:
            result = cu.cuStreamWaitEvent(handle, ring[(j - 1) % _WINDOW], 0)
            if result[0]:
                check_drv(result, "chain (cuStreamWaitEvent)")
            self._stream = i
        if gated:
            result = cu.cuStreamWaitValue32(handle, self._gate.address, 0, _WAIT_EQ)
            if result[0]:
                check_drv(result, "pause gate (cuStreamWaitValue32)")
        stream = self._streams[i]
        torch.cuda.set_stream(stream)
        failed = None
        try:
            op(stream)
        except BaseException as e:  # SystemExit too: it would kill the feeder
            failed = e
        # Also after an op that raised: the work it enqueued is tracked.
        result = cu.cuEventRecord(ring[j % _WINDOW], handle)
        if result[0]:
            check_drv(result, "chain (cuEventRecord)")
        self._enqueued = j + 1
        return failed

    def _wait_after(self, event, stream: torch.cuda.Stream) -> None:
        result = cu.cuStreamWaitEvent(stream.cuda_stream, event, 0)
        if result[0]:
            check_drv(result, "after (cuStreamWaitEvent)")
        self._spare_events.append(event)

    def _write_ticket(self, ticket: int, stream: torch.cuda.Stream) -> None:
        result = cu.cuStreamWriteValue32(
            stream.cuda_stream, self._landed_address, ticket, 0
        )
        if result[0]:
            check_drv(result, "ticket (cuStreamWriteValue32)")

    def _feed(self) -> None:
        """The feeder thread: enqueues what ``run`` left pending, waiting for
        the window to drain. Ops run here with the chain's stream current."""
        try:
            torch.cuda.set_device(self.device)
            _bind_context(self.device)
            _relax_capture_mode()
        except BaseException as e:
            logger.error("Gated transfer: the feeder thread failed", exc_info=e)
            with self._submitting:
                self._fail(e)
            return
        while True:
            with self._lock:
                while not self._pending:
                    self._more.wait()
            with self._submitting:
                try:
                    self._enqueue_pending(block=True)
                except BaseException as e:  # the chain failed
                    logger.error("Gated transfer: the feeder thread failed", exc_info=e)
                    return


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
                    or gate._pause_thread is not None
                    or not gate.busy()
                ):
                    continue
                if device is None:
                    if torch.cuda.is_current_stream_capturing():
                        return
                    device = torch.cuda.current_device()
                if gate.device != device:
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
    background transfer on ``links`` with a run in flight on this device
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
    P2P); host tensors must be pinned. Keep both alive until the transfer
    has landed."""
    nbytes = dst.numel() * dst.element_size()
    assert nbytes == src.numel() * src.element_size(), "size mismatch"
    assert dst.is_contiguous() and src.is_contiguous()
    for t in (dst, src):
        assert t.is_cuda or t.is_pinned(), "host side of a gated copy must be pinned"
    d, s = dst.data_ptr(), src.data_ptr()
    return [
        CopyChunk(d + off, s + off, n) for off, n in chunk_spans(nbytes, chunk_bytes)
    ]
