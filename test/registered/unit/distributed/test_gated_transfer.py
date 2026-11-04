"""Gated background transfers (``gated_transfer``).

- CPU: foreground bracket bookkeeping over fake gates (nesting, link and
  device filtering, one owner thread kept until the resume is enqueued,
  exception-safe entry and exit, the in-flight skip) and the watchdog's
  sampling.
- 1 GPU: ``TransferChain`` over copy-engine chunks in every direction (D2D,
  pinned H2D, D2H), dedicated streams that outlive torch's host copies, a
  held first chunk, the one-chunk drain bound (within a run, across runs,
  with the pause landing mid-copy), chunks raising on either stream, the
  bracket's link filter and ``wait`` guard on real gates, the capture refusal,
  and the watchdog forcing a stuck gate open.
- 2-4 GPUs: NCCL broadcast and allgather calls gated as plain chunk
  callables, and a paused root holding every rank.

While a gate is parked, no test enqueues GPU work it then waits for: a parked
wait holds its stream's hardware queue, and any stream may share it (the
hazard the module documents; a default-stream read and a first kernel launch
both hung while writing these tests). The tests flip pause words and blockers
from the host (both live in pinned memory) and check landed bytes in pinned
host buffers.
"""

import itertools
import subprocess
import sys
import textwrap
import threading
import time
import unittest
import unittest.mock

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators import gated_transfer
from sglang.srt.distributed.device_communicators.gated_transfer import (
    Link,
    TransferChain,
    chunk_spans,
    copy_chunks,
    paused_background_transfers,
)
from sglang.srt.environ import envs
from sglang.srt.utils.cuda_vmm_utils import check_drv
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=90, stage="base-c", runner_config="4-gpu-b200")

_MiB = 1 << 20
_WAIT_EQ = 0x1


# =============================================================================
# CPU: bracket bookkeeping over fake gates.
# =============================================================================


def _fake_gate(link, device=0):
    """A gate that records pause / resume instead of writing device memory."""
    gate = gated_transfer._Gate.__new__(gated_transfer._Gate)
    gate.link, gate.device = link, device
    gate._lock = threading.Lock()
    gate._maybe_pending = True
    gate._pause_stream = gate._pause_thread = None
    gate.calls = []
    gate.pending = True
    gate.has_pending = lambda: gate.pending
    gate.pause = lambda stream: gate.calls.append(("pause", stream))
    gate.resume = lambda stream: gate.calls.append(("resume", stream))
    return gate


class TestBracketBookkeeping(CustomTestCase):
    def setUp(self):
        self.nvlink = _fake_gate(Link.NVLINK)
        self.h2d = _fake_gate(Link.PCIE_H2D)
        self.d2h = _fake_gate(Link.PCIE_D2H)
        self.stream = object()
        self.paused = [("pause", self.stream)]
        self.paused_and_resumed = [("pause", self.stream), ("resume", self.stream)]
        for owner, attr, value in (
            (gated_transfer, "_gates", (self.nvlink, self.h2d, self.d2h)),
            (torch.cuda, "current_device", lambda: 0),
            (torch.cuda, "current_stream", lambda device=None: self.stream),
            (torch.cuda, "is_current_stream_capturing", lambda: False),
        ):
            patcher = unittest.mock.patch.object(owner, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_nested_brackets_pause_and_resume_once(self):
        """An inner bracket's exit must not resume the outer bracket's op."""
        with paused_background_transfers(Link.NVLINK):
            with paused_background_transfers(Link.NVLINK):
                pass
            self.assertEqual(self.nvlink.calls, self.paused)
        self.assertEqual(self.nvlink.calls, self.paused_and_resumed)
        self.assertIsNone(self.nvlink._pause_thread)

    def test_bracket_pauses_only_its_links(self):
        with paused_background_transfers(Link.NVLINK):
            pass
        self.assertEqual(self.h2d.calls + self.d2h.calls, [])
        with paused_background_transfers(Link.PCIE_H2D):
            with paused_background_transfers(Link.NVLINK | Link.PCIE):
                # The inner bracket pauses only what is not paused yet.
                self.assertEqual(self.h2d.calls, self.paused)
                self.assertEqual(self.d2h.calls, self.paused)
            self.assertEqual(self.d2h.calls, self.paused_and_resumed)
            self.assertEqual(self.h2d.calls, self.paused)
        self.assertEqual(len(self.nvlink.calls), 4)
        self.assertEqual(self.h2d.calls, self.paused_and_resumed)

    def test_bracket_pauses_only_gates_of_the_current_device(self):
        """Another device's gate is not ordered with the bracketed op."""
        other = _fake_gate(Link.NVLINK, device=1)
        other.has_pending = None  # never queried
        with unittest.mock.patch.object(gated_transfer, "_gates", (other,)):
            with paused_background_transfers(Link.NVLINK):
                pass
        self.assertEqual(other.calls, [])

    def test_drained_gate_is_not_paused(self):
        """No chunk in flight: the bracket costs no stream write."""
        self.nvlink.pending = False
        with paused_background_transfers(Link.NVLINK):
            pass
        self.nvlink._maybe_pending = False
        self.nvlink.has_pending = None  # the host flag answers first
        with paused_background_transfers(Link.NVLINK):
            pass
        self.assertEqual(self.nvlink.calls, [])

    def test_no_pause_under_stream_capture(self):
        with unittest.mock.patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: True
        ):
            with paused_background_transfers(Link.NVLINK):
                pass
        self.assertEqual(self.nvlink.calls, [])

    def test_another_threads_pause_is_left_alone(self):
        """Only the pausing thread resumes a gate, on its own stream."""
        other_inside, release = threading.Event(), threading.Event()

        def other():
            with paused_background_transfers(Link.NVLINK):
                other_inside.set()
                release.wait(10)

        thread = threading.Thread(target=other)
        thread.start()
        other_inside.wait(10)
        with paused_background_transfers(Link.NVLINK):
            self.assertEqual(self.nvlink._pause_thread, thread.ident)
        self.assertEqual(self.nvlink.calls, self.paused)
        release.set()
        thread.join(10)
        self.assertEqual(self.nvlink.calls, self.paused_and_resumed)

    def test_exception_inside_bracket_still_resumes(self):
        with self.assertRaises(ValueError):
            with paused_background_transfers(Link.NVLINK):
                raise ValueError
        self.assertEqual(self.nvlink.calls, self.paused_and_resumed)
        self.assertIsNone(self.nvlink._pause_thread)

    def test_failed_entry_resumes_what_it_paused(self):
        """A bracket whose entry raises never reaches __exit__."""

        def broken(stream):
            raise RuntimeError("driver error")

        self.h2d.pause = broken
        with self.assertRaises(RuntimeError):
            with paused_background_transfers(Link.NVLINK | Link.PCIE):
                pass
        self.assertEqual(self.nvlink.calls, self.paused_and_resumed)
        for gate in (self.nvlink, self.h2d, self.d2h):
            self.assertIsNone(gate._pause_thread)

    def test_failed_resume_still_releases_every_gate(self):
        """A resume that raises must not leave the other gates paused and
        owned by this thread."""

        def broken(stream):
            raise RuntimeError("driver error")

        with self.assertRaises(RuntimeError):
            with paused_background_transfers(Link.NVLINK | Link.PCIE):
                self.h2d.resume = broken
        self.assertEqual(self.nvlink.calls, self.paused_and_resumed)
        self.assertEqual(self.d2h.calls, self.paused_and_resumed)
        for gate in (self.nvlink, self.h2d, self.d2h):
            self.assertIsNone(gate._pause_thread)

    def test_gate_stays_owned_until_its_resume_is_enqueued(self):
        """Released any earlier, another thread could pause the gate before
        the resume is enqueued, and the late resume would undo that pause."""
        owners = []
        self.nvlink.resume = lambda stream: owners.append(self.nvlink._pause_thread)
        with paused_background_transfers(Link.NVLINK):
            pass
        self.assertEqual(owners, [threading.get_ident()])
        self.assertIsNone(self.nvlink._pause_thread)

    def test_no_bracket_without_gates_or_while_compiling(self):
        """Dynamo traces the bracket away (it constant-folds is_compiling)."""
        with unittest.mock.patch.object(gated_transfer, "_gates", ()):
            self.assertIs(
                paused_background_transfers(Link.NVLINK), gated_transfer._NOT_PAUSED
            )
        with unittest.mock.patch.object(torch.compiler, "is_compiling", lambda: True):
            self.assertIs(
                paused_background_transfers(Link.NVLINK), gated_transfer._NOT_PAUSED
            )

    def test_chunk_spans(self):
        self.assertEqual(chunk_spans(10, 4), [(0, 4), (4, 4), (8, 2)])
        self.assertEqual(chunk_spans(10, None), [(0, 10)])
        # Empty, chunked or not: no chunks, never a zero-step range error.
        self.assertEqual(chunk_spans(0, None), [])
        self.assertEqual(chunk_spans(0, 4), [])


class TestWatchdogSample(CustomTestCase):
    def test_forces_open_only_the_same_pause_held_past_the_timeout(self):
        gate = gated_transfer._Gate.__new__(gated_transfer._Gate)
        gate.link, gate.device = Link.NVLINK, 0
        gate._held, gate.forced_opens = None, 0
        word = [0]
        gate.word = lambda: word[0]
        gate.force_open = lambda: word.__setitem__(0, 0)
        check = gated_transfer._check_held
        self.assertFalse(check(gate, 0.0, 1.0))  # open
        word[0] = 7
        self.assertFalse(check(gate, 1.0, 1.0))  # first seen
        word[0] = 8  # a new pause: the clock restarts
        self.assertFalse(check(gate, 1.9, 1.0))
        self.assertFalse(check(gate, 2.5, 1.0))
        with self.assertLogs(gated_transfer.logger, level="ERROR"):
            self.assertTrue(check(gate, 3.0, 1.0))
        self.assertEqual((word[0], gate.forced_opens), (0, 1))


# =============================================================================
# 1 GPU: chains over copy-engine chunks.
# =============================================================================


class _HostWord:
    """A pinned int32 that a stream can wait on (UVA: the host pointer is the
    device address) and the host can flip without any GPU queue."""

    def __init__(self, value=0):
        self.tensor = torch.full((1,), value, dtype=torch.int32).pin_memory()
        self.address = self.tensor.data_ptr()

    def set(self, value):
        self.tensor[0] = value


class _Blocked:
    """A chunk op held until the host releases a word. It first sets
    ``reached``, so the host can tell the stream got past the chunk's gate."""

    def __init__(self, op, word):
        self.op, self.word = op, word
        self.reached = _HostWord()

    def __call__(self, stream):
        cu, handle = gated_transfer.cu, stream.cuda_stream
        check_drv(cu.cuStreamWriteValue32(handle, self.reached.address, 1, 0), "marker")
        check_drv(
            cu.cuStreamWaitValue32(handle, self.word.address, 1, _WAIT_EQ), "blocker"
        )
        self.op(stream)

    def wait_reached(self):
        assert _wait_until(lambda: int(self.reached.tensor[0]) == 1)


def _hold(stream, word):
    """Hold ``stream`` until the host sets ``word`` to 1."""
    check_drv(
        gated_transfer.cu.cuStreamWaitValue32(
            stream.cuda_stream, word.address, 1, _WAIT_EQ
        ),
        "hold",
    )


def _landed(host, chunk):
    """Nonzero bytes of every ``chunk`` of a pinned host buffer."""
    return [
        int(host[i : i + chunk].count_nonzero()) for i in range(0, host.numel(), chunk)
    ]


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def _retired(chain):
    """Whether the chain's last run has completed."""
    return gated_transfer._events_done([chain._last])


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestTransferChainSingleGpu(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        # Fresh gates (the registry is process-wide); no watchdog unless asked.
        for attr, value in (("_gates", ()), ("_start_watchdog", lambda: None)):
            patcher = unittest.mock.patch.object(gated_transfer, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.after = torch.cuda.Stream(self.device)
        self.control = torch.cuda.Stream(self.device)

    def _src(self, nbytes):
        # Nonzero bytes, so a landed chunk is visible.
        return torch.randint(1, 255, (nbytes,), dtype=torch.uint8, device=self.device)

    def _pinned(self, nbytes):
        return torch.zeros(nbytes, dtype=torch.uint8).pin_memory()

    def test_copies_in_every_direction(self):
        n = 3 * _MiB + 123  # a ragged last chunk
        dev = self._src(n)
        host = torch.randint(0, 255, (n,), dtype=torch.uint8).pin_memory()
        for link in (None, Link.PCIE_H2D):
            chain = TransferChain(link)
            for chunk in (None, _MiB, 2 * _MiB):  # 1, 4 and 2 chunks
                for dst, src in (
                    (torch.empty_like(dev), dev),  # D2D
                    (torch.empty_like(dev), host),  # H2D
                    (self._pinned(n), dev),  # D2H
                ):
                    with self.subTest(link=link, chunk=chunk, dst=dst.device):
                        torch.cuda.synchronize()
                        done = torch.cuda.Event()
                        chain.run(
                            copy_chunks(dst, src, chunk), after=self.after, done=done
                        )
                        done.synchronize()
                        self.assertTrue(torch.equal(dst.cpu(), src.cpu()))

    def test_pageable_host_memory_is_rejected(self):
        """A pageable copy blocks the host, which would hang behind a gate."""
        with self.assertRaises(AssertionError):
            copy_chunks(self._src(16), torch.zeros(16, dtype=torch.uint8), None)

    def test_chain_streams_are_not_pooled(self):
        """A pooled torch stream can be the forward's own stream."""
        chain = TransferChain(Link.NVLINK)
        handles = {s.cuda_stream for s in chain._streams}
        pooled = {torch.cuda.Stream(self.device).cuda_stream for _ in range(80)}
        self.assertEqual(len(handles), 2)
        self.assertFalse(handles & pooled)

    def test_run_follows_its_after_stream_and_wait_joins(self):
        """No chunk starts before the work queued on ``after`` (default: the
        current stream), and ``wait`` makes the current stream wait for the
        last run, whichever chain stream ran its last chunk."""
        chain = TransferChain(Link.NVLINK)
        src = self._src(4 * _MiB)
        consumer = torch.cuda.Stream(self.device)
        # The last chunk on the second stream, then on the primary.
        for explicit, n_chunks in itertools.product((False, True), (3, 4)):
            n = n_chunks * _MiB
            dst = self._pinned(n)
            blocker = _HostWord()
            torch.cuda.synchronize()
            _hold(self.after, blocker)  # work queued on `after`, held by the host
            try:
                if explicit:  # issued from another current stream
                    chain.run(copy_chunks(dst, src[:n], _MiB), after=self.after)
                else:
                    with torch.cuda.stream(self.after):
                        chain.run(copy_chunks(dst, src[:n], _MiB))
                time.sleep(0.05)
                ahead = int(dst.count_nonzero())
            finally:
                blocker.set(1)
            with torch.cuda.stream(consumer):
                chain.wait()
                landed = dst.to(self.device, non_blocking=True)
                equal = torch.equal(landed, src[:n])
            case = f"explicit after: {explicit}, {n_chunks} chunks"
            self.assertEqual(ahead, 0, case)
            self.assertTrue(equal, case)

    def test_stream_ordered_pause_writes_a_sequence_and_resume_zero(self):
        gate = TransferChain(Link.NVLINK)._gate
        gate.pause(self.control)
        self.control.synchronize()
        first = gate.word()
        gate.resume(self.control)
        self.control.synchronize()
        self.assertEqual(gate.word(), 0)
        gate.pause(self.control)
        self.control.synchronize()
        self.assertNotIn(gate.word(), (0, first))
        gate.force_open()
        self.assertEqual(gate.word(), 0)

    def test_paused_gate_holds_the_first_chunk(self):
        chain = TransferChain(Link.NVLINK)
        gate = chain._gate
        gate._word.value = 5  # paused from the host
        src = self._src(4 * _MiB)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        chain.run(copy_chunks(dst, src, _MiB), after=self.after)
        time.sleep(0.1)
        self.assertEqual(_landed(dst, _MiB), [0, 0, 0, 0])
        self.assertFalse(_retired(chain))
        self.assertTrue(gate.has_pending())
        gate.force_open()
        self.assertTrue(_wait_until(lambda: _retired(chain)))
        self.assertTrue(torch.equal(dst, src.cpu()))
        self.assertFalse(gate.has_pending())
        self.assertFalse(gate._maybe_pending)

    def test_at_most_the_inflight_chunk_drains_after_a_pause(self):
        """Chunk 0 is held by a blocker; a pause lands; releasing chunk 0 must
        not let chunk 1 pass its gate, because chunk 1's gate is read only
        after chunk 0 retired."""
        chain = TransferChain(Link.NVLINK)
        blocker = _HostWord()
        src = self._src(4 * _MiB)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        ops = copy_chunks(dst, src, _MiB)
        ops[0] = _Blocked(ops[0], blocker)
        chain.run(ops, after=self.after)
        ops[0].wait_reached()  # chunk 0's gate is behind us
        chain._gate._word.value = 9
        blocker.set(1)
        time.sleep(0.1)
        self.assertEqual(_landed(dst, _MiB), [_MiB, 0, 0, 0])
        chain._gate.force_open()
        self.assertTrue(_wait_until(lambda: _retired(chain)))
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_back_to_back_runs_drain_one_chunk(self):
        """A run's first chunk reads the gate only after the previous run
        retired: with the previous run's last chunk in flight when a pause
        lands, the next run's chunk does not pass its gate."""
        chain = TransferChain(Link.NVLINK)
        blocker = _HostWord()
        srcs = [self._src(_MiB) for _ in range(2)]
        dsts = [self._pinned(_MiB) for _ in range(2)]
        torch.cuda.synchronize()
        first = copy_chunks(dsts[0], srcs[0], None)
        first[0] = _Blocked(first[0], blocker)
        chain.run(first, after=self.after)
        chain.run(copy_chunks(dsts[1], srcs[1], None), after=self.after)
        first[0].wait_reached()
        chain._gate._word.value = 4
        blocker.set(1)
        time.sleep(0.1)
        self.assertEqual([_landed(d, _MiB) for d in dsts], [[_MiB], [0]])
        chain._gate.force_open()
        self.assertTrue(_wait_until(lambda: _retired(chain)))
        self.assertTrue(torch.equal(dsts[1], srcs[1].cpu()))

    def test_pause_during_a_running_chunk_holds_the_next(self):
        """The case the event chaining exists for: the pause lands while
        chunk 0's (long) copy runs; chunk 1's gate must not have been read
        ahead of it."""
        chain = TransferChain(Link.NVLINK)
        blocker = _HostWord()
        chunk = 512 * _MiB  # ~10 ms over PCIe: the pause lands mid-copy
        src = self._src(2 * chunk)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        ops = copy_chunks(dst, src, chunk)
        ops[0] = _Blocked(ops[0], blocker)
        done = torch.cuda.Event()
        chain.run(ops, after=self.after, done=done)
        ops[0].wait_reached()
        blocker.set(1)
        chain._gate._word.value = 6
        time.sleep(0.3)
        self.assertEqual(int(dst[chunk:].count_nonzero()), 0)
        self.assertEqual(int(dst[:chunk].count_nonzero()), chunk)
        chain._gate.force_open()
        self.assertTrue(_wait_until(done.query))
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_current_stream_callables_are_gated(self):
        """A chunk written against the current stream, with no stream
        handling of its own, runs on the chain's stream behind the gate; the
        caller's current stream is restored afterwards."""
        chain = TransferChain(Link.NVLINK)
        src = self._src(2 * _MiB)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        chain._gate._word.value = 2
        caller = torch.cuda.current_stream()
        chunks = [
            lambda stream, a=a: dst[a : a + _MiB].copy_(
                src[a : a + _MiB], non_blocking=True
            )
            for a in (0, _MiB)
        ]
        chain.run(chunks, after=self.after)
        self.assertEqual(torch.cuda.current_stream(), caller)
        time.sleep(0.1)
        self.assertEqual(_landed(dst, _MiB), [0, 0])
        chain._gate.force_open()
        self.assertTrue(_wait_until(lambda: _retired(chain)))
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_setup_and_finish_run_around_the_chunks(self):
        """Setup lands before chunk 0 reads the gate; finish runs after the
        last chunk, and both ``done`` and the last run follow the finish ops,
        also in a run with no chunks (a broadcast's non-root ranks)."""
        chain = TransferChain(Link.NVLINK)
        src = self._src(3 * _MiB)
        stage = torch.zeros_like(src)
        dst = torch.zeros_like(src)
        out = torch.zeros_like(src)
        consumer = torch.cuda.Stream(self.device)
        # Three chunks, two, none; waiting on done, then on the last run.
        for chunk, use_done in itertools.product((_MiB, 3 * _MiB // 2, 0), (1, 0)):
            for t in (stage, dst, out):
                t.zero_()
            torch.cuda.synchronize()
            done = torch.cuda.Event()
            chain.run(
                copy_chunks(dst, stage, chunk) if chunk else [],
                setup=[lambda stream: stage.copy_(src)],
                finish=[lambda stream: out.copy_(dst if chunk else stage)],
                done=done,
                after=self.after,
            )
            chain.wait(done if use_done else None, stream=consumer)
            with torch.cuda.stream(consumer):
                equal = torch.equal(out, src)
            self.assertTrue(equal, f"chunk {chunk}, wait on done: {use_done}")

    def test_raising_chunk_keeps_what_it_enqueued_tracked(self):
        """A chunk that raises aborts the run, but everything enqueued before
        the error, including work the raising chunk enqueued on either chain
        stream (chunks 0, 2, ... run on the second), still holds back
        ``done``, ``wait`` and the brackets' in-flight check. The caller's
        current stream is restored."""
        chain = TransferChain(Link.NVLINK)
        src = self._src(_MiB)
        caller = torch.cuda.current_stream()
        for failing in (0, 1, 2):
            with self.subTest(failing_chunk=failing):
                blocker = _HostWord()
                dst = self._pinned(_MiB)
                torch.cuda.synchronize()
                held = _Blocked(lambda s: dst.copy_(src, non_blocking=True), blocker)

                def enqueue_then_raise(stream):
                    held(stream)
                    raise ValueError("chunk failed after enqueuing work")

                ops = [lambda stream: None] * failing + [enqueue_then_raise]
                done = torch.cuda.Event()
                with self.assertRaises(ValueError):
                    chain.run(ops, after=self.after, done=done)
                self.assertEqual(torch.cuda.current_stream(), caller)
                try:
                    held.wait_reached()
                    self.assertFalse(done.query())
                    self.assertFalse(_retired(chain))
                    self.assertTrue(chain._gate.has_pending())
                finally:
                    blocker.set(1)  # a failed check must not leave it parked
                self.assertTrue(_wait_until(done.query))
                self.assertTrue(_wait_until(lambda: _retired(chain)))
                self.assertTrue(torch.equal(dst, src.cpu()))

    def test_unrecordable_done_still_leaves_the_run_in_flight(self):
        """A ``done`` that cannot be recorded (e.g. another device's event)
        must not hide the run's chunks from the brackets."""
        chain = TransferChain(Link.NVLINK)
        blocker = _HostWord()
        src = self._src(_MiB)
        dst = torch.zeros_like(src)
        torch.cuda.synchronize()
        ops = copy_chunks(dst, src)
        ops[0] = _Blocked(ops[0], blocker)

        class Unrecordable:
            def record(self, stream):
                raise RuntimeError("event of another device")

        try:
            with self.assertRaises(RuntimeError):
                chain.run(ops, done=Unrecordable(), after=self.after)
            self.assertTrue(chain._gate.has_pending())
        finally:
            blocker.set(1)
        self.assertTrue(_wait_until(lambda: _retired(chain)))

    def test_chain_is_used_on_its_device(self):
        """The default streams are the chain device's: a call made with
        another device current is refused, an explicit stream included."""
        chain = TransferChain(Link.NVLINK)
        other = lambda: chain.device + 1
        with unittest.mock.patch.object(torch.cuda, "current_device", other):
            for call in (lambda: chain.run([]), lambda: chain.wait(stream=self.after)):
                with self.assertRaises(AssertionError):
                    call()

    def test_ungated_chain_tracks_its_last_run(self):
        chain = TransferChain(None)
        self.assertEqual(len(chain._streams), 1)
        blocker = _HostWord()
        src = self._src(_MiB)
        dst = torch.zeros_like(src)
        torch.cuda.synchronize()
        ops = copy_chunks(dst, src, None)
        ops[0] = _Blocked(ops[0], blocker)
        chain.run(ops, after=self.after)
        ops[0].wait_reached()
        self.assertFalse(_retired(chain))
        blocker.set(1)
        self.assertTrue(_wait_until(lambda: _retired(chain)))

    def test_chain_streams_outlive_torch_host_copies(self):
        """A pinned ``tensor.copy_`` in a chunk makes torch record the
        chain's stream on the host block, and freeing the block records an
        event on that stream. A chain that destroyed its streams once it was
        collected crashed the process there."""
        code = textwrap.dedent(
            """
            import gc
            import torch
            from sglang.srt.distributed.device_communicators.gated_transfer import (
                TransferChain,
            )

            src = torch.ones(1 << 20, dtype=torch.uint8, device="cuda")
            dst = torch.zeros(1 << 20, dtype=torch.uint8).pin_memory()
            chain = TransferChain(None)
            copy = lambda stream: dst.copy_(src, non_blocking=True)
            chain.run([copy])
            torch.cuda.synchronize()
            del chain, copy
            gc.collect()
            del dst  # torch records an event on the chain's stream here
            torch.cuda.synchronize()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=300
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode()[-2000:])

    def test_bracket_pauses_in_flight_gates_of_its_link_only(self):
        nvlink, h2d = TransferChain(Link.NVLINK), TransferChain(Link.PCIE_H2D)
        blocker = _HostWord()
        src = self._src(2 * _MiB)
        dsts = [torch.zeros_like(src) for _ in range(2)]
        dones = [torch.cuda.Event() for _ in range(2)]
        torch.cuda.synchronize()
        for chain, dst, done in zip((nvlink, h2d), dsts, dones):
            ops = copy_chunks(dst, src, _MiB)
            ops[0] = _Blocked(ops[0], blocker)
            chain.run(ops, after=self.after, done=done)
        with torch.cuda.stream(self.control):
            with paused_background_transfers(Link.NVLINK):
                self.assertEqual(nvlink._gate._pause_thread, threading.get_ident())
                self.assertIsNone(h2d._gate._pause_thread)
                # Harmless: the chain owns its streams; its chunks just park.
                nvlink.run([], after=self.after)
                with self.assertRaises(RuntimeError):
                    # The waiting stream would block on its own pause.
                    nvlink.wait(dones[0])
                h2d.wait(dones[1])  # another link's gate: fine
            self.assertIsNone(nvlink._gate._pause_thread)
        blocker.set(1)
        for done in dones:
            self.assertTrue(_wait_until(done.query))
        self.control.synchronize()
        # Including the empty run issued inside the bracket.
        self.assertTrue(_wait_until(lambda: _retired(nvlink) and _retired(h2d)))
        for dst in dsts:
            self.assertTrue(torch.equal(dst, src))
        # Drained: the next bracket pauses nothing.
        with torch.cuda.stream(self.control):
            with paused_background_transfers(Link.NVLINK | Link.PCIE):
                self.assertIsNone(nvlink._gate._pause_thread)
                self.assertIsNone(h2d._gate._pause_thread)

    def test_watchdog_forces_a_stuck_pause_open(self):
        """A pause whose resume never comes: the watchdog opens the gate and
        the parked chain completes."""
        chain = TransferChain(Link.NVLINK)
        gate = chain._gate
        src = self._src(2 * _MiB)
        dst = self._pinned(src.numel())
        gate.pause(self.control)  # no resume follows
        self.control.synchronize()
        chain.run(copy_chunks(dst, src, _MiB), after=self.after)
        time.sleep(0.05)
        self.assertFalse(_retired(chain))
        stop = threading.Event()

        def watchdog():
            # One gate, a 50 ms timeout, the production sampling.
            while not stop.is_set():
                time.sleep(0.0125)
                gated_transfer._check_held(gate, time.monotonic(), 0.05)

        thread = threading.Thread(target=watchdog, daemon=True)
        with self.assertLogs(gated_transfer.logger, level="ERROR"):
            thread.start()
            self.assertTrue(_wait_until(lambda: _retired(chain), timeout=5))
            stop.set()
            thread.join(2)
        self.assertEqual(gate.forced_opens, 1)
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_watchdog_thread_reads_its_timeout_from_the_env(self):
        with envs.SGLANG_GATED_TRANSFER_WATCHDOG_MS.override(0):
            gated_transfer._watch_gates()  # returns at once when disabled

    def test_capturing_after_stream_is_refused(self):
        chain = TransferChain(Link.NVLINK)
        src = self._src(_MiB)
        dst = torch.zeros_like(src)
        stream = torch.cuda.Stream(self.device)
        dst.add_(1)  # load the kernel outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with self.assertRaises(RuntimeError):
                chain.run(copy_chunks(dst, src))
            with torch.cuda.stream(self.after), self.assertRaises(RuntimeError):
                chain.run(copy_chunks(dst, src), after=stream)
            dst.add_(1)  # keep the capture valid and non-empty


# =============================================================================
# 2-4 GPUs: NCCL chunks are just callables.
# =============================================================================

_WORLD = 4 if torch.cuda.device_count() >= 4 else 2


def _init(rank, port):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=_WORLD
    )
    from sglang.srt.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )

    # A communicator of the gated chunks alone: NCCL orders every op of a
    # communicator, so a parked chunk would also hold any other op on it.
    return PyNcclCommunicator(
        group=torch.distributed.group.WORLD, device=torch.device("cuda", rank)
    )


def _run_nccl_chunks(rank, port):
    comm = _init(rank, port)  # the gated chunks' own communicator
    from sglang.srt.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )

    foreground = PyNcclCommunicator(
        group=torch.distributed.group.WORLD, device=torch.device("cuda", rank)
    )
    chain = TransferChain(Link.NVLINK)
    gate = chain._gate
    after = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    root = _WORLD - 1
    n = 1 << 20  # fp32 elements per chunk
    # Connect before anything can park: a lazy connect behind a parked gate
    # hangs (see the module docstring).
    for c in (comm, foreground):
        with c.change_state(enable=True):
            c.broadcast(torch.zeros(n, device="cuda"), 0)
            c.all_gather(
                torch.zeros(_WORLD * n, device="cuda"), torch.zeros(n, device="cuda")
            )
            c.all_reduce(torch.zeros(n, device="cuda"))
    torch.cuda.synchronize()

    def wait(event):
        assert _wait_until(event.query), f"[rank {rank}] gated NCCL chunks hung"

    def run(chunks):
        """The chunks call PyNcclCommunicator on the current stream: the chain
        makes it theirs."""
        torch.cuda.synchronize()
        torch.distributed.barrier()
        done = torch.cuda.Event()
        with comm.change_state(enable=True):
            chain.run(chunks, after=after, done=done)
        chain.wait(done, stream=consumer)
        read = torch.cuda.Event()
        read.record(consumer)
        return read

    def run_paused(paused, chunks):
        """Rank ``paused`` (one whose data every rank needs) is paused from
        the host before the chunks are enqueued: no rank completes until it
        resumes."""
        if rank == paused:
            gate._word.value = 3
        read = run(chunks)
        time.sleep(0.2)
        stalled = not read.query()
        torch.distributed.barrier()  # host only: no GPU work behind a parked gate
        if rank == paused:
            gate.force_open()
        wait(read)
        assert stalled, f"[rank {rank}] chunks ran through rank {paused}'s pause"

    # Broadcast from the last rank, four chunks, then with the root paused.
    data = torch.empty((4, n), device="cuda")
    broadcast = [lambda stream, i=i: comm.broadcast(data[i], root) for i in range(4)]
    for paused in (None, root):
        data.fill_(float(rank + 1))
        if paused is None:
            wait(run(broadcast))
        else:
            run_paused(paused, broadcast)
        assert bool((data == root + 1).all()), f"[rank {rank}] broadcast"

    # A sequence of allgathers, one chunk each, then with rank 0 paused.
    inputs = [torch.full((n,), float(10 * rank + i), device="cuda") for i in range(3)]
    outputs = [torch.zeros(_WORLD * n, device="cuda") for _ in range(3)]
    gathers = [
        lambda stream, o=o, x=x: comm.all_gather(o, x) for o, x in zip(outputs, inputs)
    ]
    for paused in (None, 0):
        for out in outputs:
            out.zero_()
        if paused is None:
            wait(run(gathers))
        else:
            run_paused(paused, gathers)
        for i, out in enumerate(outputs):
            for r in range(_WORLD):
                got = out.view(_WORLD, n)[r]
                assert bool((got == 10 * r + i).all()), f"[rank {rank}] gather {i}"

    # The public bracket: chunks in flight (chunk 0 held by a host word), a
    # foreground all-reduce on the other communicator pauses and resumes the
    # gate in its own stream's order. With the watchdog off, a resume that
    # never ran leaves the chunks parked and fails the wait below.
    blocker = _HostWord()
    for out in outputs:
        out.zero_()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    held = list(gathers)
    held[0] = _Blocked(held[0], blocker)
    done = torch.cuda.Event()
    with comm.change_state(enable=True):
        chain.run(held, after=after, done=done)
    assert gate.has_pending()
    x = torch.full((n,), float(rank + 1), device="cuda")
    pauses = gate._seq
    with foreground.change_state(enable=True):
        with paused_background_transfers(Link.NVLINK):
            foreground.all_reduce(x)
    assert gate._seq == pauses + 1 and gate._pause_thread is None
    blocker.set(1)  # host only: nothing waits on the GPU before the release
    chain.wait(done, stream=consumer)
    read = torch.cuda.Event()
    read.record(consumer)
    wait(read)
    torch.cuda.synchronize()
    assert bool((x == sum(range(1, _WORLD + 1))).all()), f"[rank {rank}] all-reduce"
    for i, out in enumerate(outputs):
        assert bool((out.view(_WORLD, n)[0] == i).all()), f"[rank {rank}] gather {i}"
    assert gate.word() == 0
    torch.distributed.barrier()


@unittest.skipIf(torch.cuda.device_count() < 2, "needs 2 CUDA devices")
class TestGatedNcclChunksMultiGpu(CustomTestCase):
    def test_gated_nccl_chunks(self):
        """NCCL needs no gate support of its own: plain PyNcclCommunicator
        calls are passed in as chunks. A paused rank holds every rank, and a
        foreground collective's bracket pauses and resumes them."""
        # No watchdog: a pause that never resumes must fail, not be rescued.
        with envs.SGLANG_GATED_TRANSFER_WATCHDOG_MS.override(0):
            mp.spawn(_run_nccl_chunks, args=(29831,), nprocs=_WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
