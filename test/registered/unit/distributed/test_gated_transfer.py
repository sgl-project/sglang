"""Gated background transfers (``gated_transfer``).

- CPU: foreground bracket bookkeeping over fake gates (nesting, link
  filtering, one owner thread, exception-safe entry and exit, the in-flight
  skip) and the watchdog's sampling.
- 1 GPU: ``TransferChain`` over copy-engine chunks in every direction (D2D,
  pinned H2D, D2H), dedicated streams, a held first chunk, the one-chunk drain
  bound (within a run, across runs, with the pause landing mid-copy), the
  bracket's link filter and ``wait`` guard on real gates, ``close``, the
  capture refusal, and the watchdog forcing a stuck gate open.
- 2-4 GPUs: NCCL broadcast and allgather calls gated as plain chunk
  callables, and a paused root holding every rank.

While a gate is parked, no test enqueues GPU work it then waits for: a parked
wait holds its stream's hardware queue, and any stream may share it (the
hazard the module documents; a default-stream read and a first kernel launch
both hung while writing these tests). The tests flip pause words and blockers
from the host (both live in pinned memory) and check landed bytes in pinned
host buffers.
"""

import threading
import time
import unittest
import unittest.mock

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators import gated_transfer
from sglang.srt.distributed.device_communicators.gated_transfer import (
    CU_STREAM_WAIT_VALUE_EQ,
    CudaDriver,
    Link,
    TransferChain,
    TransferGate,
    background_transfers_pending,
    check_gate_held,
    chunk_spans,
    copy_chunks,
    copy_tensor_chunks,
    get_transfer_gate,
    paused_background_transfers,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=90, stage="base-c", runner_config="4-gpu-b200")

_MiB = 1 << 20


# =============================================================================
# CPU: bracket bookkeeping over fake gates.
# =============================================================================


def _fake_gate(link, pending=True):
    """A gate that records pause / resume instead of writing device memory."""
    gate = TransferGate.__new__(TransferGate)
    gate.link = link
    gate.device_index = 0
    gate._lock = threading.Lock()
    gate._maybe_pending = pending
    gate._runs = 0
    gate._pause_depth = 0
    gate._pause_stream = None
    gate._pause_thread = None
    gate.calls = []
    gate.pending = pending
    gate.has_pending = lambda: gate.pending
    gate.pause = lambda stream: gate.calls.append(("pause", stream))
    gate.resume = lambda stream: gate.calls.append(("resume", stream))
    return gate


class TestBracketBookkeeping(CustomTestCase):
    def setUp(self):
        self.nvlink = _fake_gate(Link.NVLINK)
        self.h2d = _fake_gate(Link.PCIE_H2D)
        self.d2h = _fake_gate(Link.PCIE_D2H)
        for owner, attr, value in (
            (gated_transfer, "_gates", [self.nvlink, self.h2d, self.d2h]),
            (torch.cuda, "current_stream", lambda device=None: "stream"),
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
            self.assertEqual(self.nvlink.calls, [("pause", "stream")])
        self.assertEqual(self.nvlink.calls, [("pause", "stream"), ("resume", "stream")])
        self.assertEqual(self.nvlink._pause_depth, 0)
        self.assertIsNone(self.nvlink._pause_thread)

    def test_bracket_pauses_only_its_links(self):
        with paused_background_transfers(Link.NVLINK):
            pass
        self.assertEqual(self.h2d.calls + self.d2h.calls, [])
        with paused_background_transfers(Link.PCIE_H2D):
            pass
        self.assertEqual(self.d2h.calls, [])
        with paused_background_transfers(Link.NVLINK | Link.PCIE):
            self.assertEqual(self.d2h.calls, [("pause", "stream")])
        self.assertEqual(len(self.nvlink.calls), 4)
        self.assertEqual(len(self.h2d.calls), 4)

    def test_inner_bracket_on_another_link_pauses_that_link_only(self):
        with paused_background_transfers(Link.NVLINK):
            with paused_background_transfers(Link.PCIE_H2D):
                self.assertEqual(self.h2d.calls, [("pause", "stream")])
            self.assertEqual(
                self.h2d.calls, [("pause", "stream"), ("resume", "stream")]
            )
            self.assertEqual(self.nvlink.calls, [("pause", "stream")])

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
            self.assertEqual(self.nvlink._pause_depth, 1)
        self.assertEqual(self.nvlink.calls, [("pause", "stream")])
        release.set()
        thread.join(10)
        self.assertEqual(self.nvlink.calls, [("pause", "stream"), ("resume", "stream")])

    def test_exception_inside_bracket_still_resumes(self):
        with self.assertRaises(ValueError):
            with paused_background_transfers(Link.NVLINK):
                raise ValueError
        self.assertEqual(self.nvlink.calls[-1], ("resume", "stream"))
        self.assertEqual(self.nvlink._pause_depth, 0)

    def test_failed_entry_resumes_what_it_paused(self):
        """A bracket whose entry raises never reaches __exit__."""

        def broken(stream):
            raise RuntimeError("driver error")

        self.h2d.pause = broken
        with self.assertRaises(RuntimeError):
            with paused_background_transfers(Link.NVLINK | Link.PCIE):
                pass
        self.assertEqual(self.nvlink.calls, [("pause", "stream"), ("resume", "stream")])
        for gate in (self.nvlink, self.h2d, self.d2h):
            self.assertEqual(gate._pause_depth, 0)
            self.assertIsNone(gate._pause_thread)

    def test_failed_resume_still_releases_every_gate(self):
        """A resume that raises must not leave the other gates paused and
        owned by this thread."""

        def broken(stream):
            raise RuntimeError("driver error")

        with self.assertRaises(RuntimeError):
            with paused_background_transfers(Link.NVLINK | Link.PCIE):
                self.h2d.resume = broken
        self.assertEqual(self.nvlink.calls, [("pause", "stream"), ("resume", "stream")])
        self.assertEqual(self.d2h.calls, [("pause", "stream"), ("resume", "stream")])
        for gate in (self.nvlink, self.h2d, self.d2h):
            self.assertEqual(gate._pause_depth, 0)
            self.assertIsNone(gate._pause_thread)

    def test_no_gates_no_bracket(self):
        with unittest.mock.patch.object(gated_transfer, "_gates", []):
            self.assertIs(
                paused_background_transfers(Link.NVLINK), gated_transfer._NOT_PAUSED
            )

    def test_no_bracket_while_compiling(self):
        """Dynamo traces the bracket away (it constant-folds is_compiling)."""
        with unittest.mock.patch.object(torch.compiler, "is_compiling", lambda: True):
            self.assertIs(
                paused_background_transfers(Link.NVLINK), gated_transfer._NOT_PAUSED
            )
        self.assertEqual(self.nvlink.calls, [])

    def test_pending_hint_follows_links_and_pauses(self):
        self.assertTrue(background_transfers_pending(Link.NVLINK))
        self.nvlink._maybe_pending = False
        self.assertFalse(background_transfers_pending(Link.NVLINK))
        self.assertTrue(background_transfers_pending(Link.NVLINK | Link.PCIE))
        # A paused gate stays relevant until its bracket resumes it.
        self.nvlink._pause_depth = 1
        self.assertTrue(background_transfers_pending(Link.NVLINK))

    def test_chunk_spans_and_copy_chunks(self):
        self.assertEqual(chunk_spans(10, 4), [(0, 4), (4, 4), (8, 2)])
        self.assertEqual(chunk_spans(10, None), [(0, 10)])
        # Empty, chunked or not: no chunks, never a zero-step range error.
        self.assertEqual(chunk_spans(0, None), [])
        self.assertEqual(chunk_spans(0, 4), [])
        self.assertEqual(
            [op.copies for op in copy_chunks(1000, 2000, 10, 4)],
            [((1000, 2000, 4),), ((1004, 2004, 4),), ((1008, 2008, 2),)],
        )


class TestWatchdogSample(CustomTestCase):
    def test_forces_open_only_the_same_pause_held_past_the_timeout(self):
        gate = TransferGate.__new__(TransferGate)
        gate.link, gate.device_index = Link.NVLINK, 0
        gate._held, gate.forced_opens = None, 0
        word = [0]
        gate.word = lambda: word[0]
        gate.force_open = lambda: word.__setitem__(0, 0)
        self.assertFalse(check_gate_held(gate, 0.0, 1.0))  # open
        word[0] = 7
        self.assertFalse(check_gate_held(gate, 1.0, 1.0))  # first seen
        word[0] = 8  # a new pause: the clock restarts
        self.assertFalse(check_gate_held(gate, 1.9, 1.0))
        self.assertFalse(check_gate_held(gate, 2.5, 1.0))
        with self.assertLogs(gated_transfer.logger, level="ERROR"):
            self.assertTrue(check_gate_held(gate, 3.0, 1.0))
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
        drv = CudaDriver.get()
        drv.check(drv.write32(stream.cuda_stream, self.reached.address, 1, 0), "marker")
        drv.check(
            drv.wait32(
                stream.cuda_stream, self.word.address, 1, CU_STREAM_WAIT_VALUE_EQ
            ),
            "blocker",
        )
        self.op(stream)

    def wait_reached(self):
        assert _wait_until(lambda: int(self.reached.tensor[0]) == 1)


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


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestTransferChainSingleGpu(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        # Fresh gates (the registry is process-wide); no watchdog unless asked.
        for owner, attr, value in (
            (gated_transfer, "_gates", []),
            (gated_transfer, "_gate_index", {}),
            (gated_transfer, "_ensure_watchdog", lambda: None),
        ):
            patcher = unittest.mock.patch.object(owner, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.after = torch.cuda.Stream(self.device)
        self.control = torch.cuda.Stream(self.device)

    def _chain(self, link=Link.NVLINK, gated=True):
        gate = get_transfer_gate(link, self.device.index) if gated else None
        chain = TransferChain(self.device.index, gate)
        self.addCleanup(chain.close)
        return chain

    def _src(self, nbytes):
        # Nonzero bytes, so a landed chunk is visible.
        return torch.randint(1, 255, (nbytes,), dtype=torch.uint8, device=self.device)

    def _pinned(self, nbytes):
        return torch.zeros(nbytes, dtype=torch.uint8).pin_memory()

    def test_copies_in_every_direction(self):
        n = 3 * _MiB + 123  # a ragged last chunk
        dev = self._src(n)
        host = torch.randint(0, 255, (n,), dtype=torch.uint8).pin_memory()
        for gated in (False, True):
            chain = self._chain(Link.PCIE_H2D, gated=gated)
            for chunk in (None, _MiB, 2 * _MiB):  # 1, 4 and 2 chunks
                for dst, src in (
                    (torch.empty_like(dev), dev),  # D2D
                    (torch.empty_like(dev), host),  # H2D
                    (self._pinned(n), dev),  # D2H
                ):
                    with self.subTest(gated=gated, chunk=chunk, dst=dst.device):
                        torch.cuda.synchronize()
                        done = torch.cuda.Event()
                        chain.run(
                            copy_tensor_chunks(dst, src, chunk), self.after, done=done
                        )
                        done.synchronize()
                        self.assertTrue(torch.equal(dst.cpu(), src.cpu()))

    def test_pageable_host_memory_is_rejected(self):
        dev = torch.zeros(16, dtype=torch.uint8, device=self.device)
        with self.assertRaises(AssertionError):
            copy_tensor_chunks(dev, torch.zeros(16, dtype=torch.uint8), None)

    def test_chain_streams_are_not_pooled(self):
        """A pooled torch stream can be the forward's own stream."""
        chain = self._chain()
        handles = {s.cuda_stream for s in chain._streams}
        pooled = {torch.cuda.Stream(self.device).cuda_stream for _ in range(80)}
        self.assertEqual(len(handles), 2)
        self.assertFalse(handles & pooled)

    def test_run_is_ordered_after_the_after_stream_and_waits_join(self):
        """The first chunk waits for ``after``'s queued work; a consumer that
        waits on ``done`` sees every chunk, whichever stream ran last."""
        chunk = 256 * _MiB
        src = self._src(4 * chunk)
        dst = torch.zeros_like(src)
        stage = torch.zeros_like(src)
        chain = self._chain()
        consumer = torch.cuda.Stream(self.device)
        for n_chunks in (3, 4):  # the last chunk on the primary, then the other
            n = n_chunks * chunk
            dst.zero_()
            stage.zero_()
            torch.cuda.synchronize()
            with torch.cuda.stream(self.after):
                stage[:n].copy_(src[:n])  # the chunks read what `after` wrote
            done = torch.cuda.Event()
            chain.run(
                copy_tensor_chunks(dst[:n], stage[:n], chunk), self.after, done=done
            )
            chain.wait(consumer, done)
            with torch.cuda.stream(consumer):
                equal = torch.equal(dst[:n], src[:n])
            self.assertTrue(equal, f"{n_chunks} chunks")

    def test_stream_ordered_pause_writes_a_sequence_and_resume_zero(self):
        gate = self._chain().gate
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
        chain = self._chain()
        gate = chain.gate
        gate._word.value = 5  # paused from the host
        src = self._src(4 * _MiB)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        chain.run(copy_tensor_chunks(dst, src, _MiB), self.after)
        time.sleep(0.1)
        self.assertEqual(_landed(dst, _MiB), [0, 0, 0, 0])
        self.assertFalse(chain.retired())
        self.assertTrue(gate.has_pending())
        gate.force_open()
        self.assertTrue(_wait_until(chain.retired))
        self.assertTrue(torch.equal(dst, src.cpu()))
        self.assertFalse(gate.has_pending())
        self.assertFalse(gate._maybe_pending)

    def test_at_most_the_inflight_chunk_drains_after_a_pause(self):
        """Chunk 0 is held by a blocker; a pause lands; releasing chunk 0 must
        not let chunk 1 pass its gate, because chunk 1's gate is read only
        after chunk 0 retired."""
        chain = self._chain()
        gate = chain.gate
        blocker = _HostWord()
        src = self._src(4 * _MiB)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        ops = copy_tensor_chunks(dst, src, _MiB)
        ops[0] = _Blocked(ops[0], blocker)
        chain.run(ops, self.after)
        ops[0].wait_reached()  # chunk 0's gate is behind us
        gate._word.value = 9
        blocker.set(1)
        time.sleep(0.1)
        self.assertEqual(_landed(dst, _MiB), [_MiB, 0, 0, 0])
        gate.force_open()
        self.assertTrue(_wait_until(chain.retired))
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_back_to_back_runs_drain_one_chunk(self):
        """A run's first chunk reads the gate only after the previous run
        retired: with the previous run's last chunk in flight when a pause
        lands, the next run's chunk does not pass its gate."""
        chain = self._chain()
        gate = chain.gate
        blocker = _HostWord()
        srcs = [self._src(_MiB) for _ in range(2)]
        dsts = [self._pinned(_MiB) for _ in range(2)]
        torch.cuda.synchronize()
        first = copy_tensor_chunks(dsts[0], srcs[0], None)
        first[0] = _Blocked(first[0], blocker)
        chain.run(first, self.after)
        chain.run(copy_tensor_chunks(dsts[1], srcs[1], None), self.after)
        first[0].wait_reached()
        gate._word.value = 4
        blocker.set(1)
        time.sleep(0.1)
        self.assertEqual([_landed(d, _MiB) for d in dsts], [[_MiB], [0]])
        gate.force_open()
        self.assertTrue(_wait_until(chain.retired))
        self.assertTrue(torch.equal(dsts[1], srcs[1].cpu()))

    def test_current_stream_callables_are_gated(self):
        """A chunk written against the current stream, with no stream
        handling of its own, runs on the chain's stream behind the gate; the
        caller's current stream is restored afterwards."""
        chain = self._chain()
        gate = chain.gate
        src = self._src(2 * _MiB)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        gate._word.value = 2
        caller = torch.cuda.current_stream()
        chunks = [
            lambda stream, a=a: dst[a : a + _MiB].copy_(
                src[a : a + _MiB], non_blocking=True
            )
            for a in (0, _MiB)
        ]
        chain.run(chunks, self.after)
        self.assertEqual(torch.cuda.current_stream(), caller)
        time.sleep(0.1)
        self.assertEqual(_landed(dst, _MiB), [0, 0])
        gate.force_open()
        self.assertTrue(_wait_until(chain.retired))
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_setup_and_finish_run_around_the_chunks(self):
        """Setup lands before chunk 0 reads the gate; finish runs after the
        last chunk, and ``wait`` without an event joins the whole run."""
        chain = self._chain()
        src = self._src(3 * _MiB)
        stage = torch.zeros_like(src)
        dst = torch.zeros_like(src)
        out = torch.zeros_like(src)
        consumer = torch.cuda.Stream(self.device)
        torch.cuda.synchronize()
        for chunk in (_MiB, 3 * _MiB // 2):  # three chunks, then two
            for t in (stage, dst, out):
                t.zero_()
            torch.cuda.synchronize()
            chain.run(
                copy_tensor_chunks(dst, stage, chunk),
                self.after,
                setup=[lambda stream: stage.copy_(src)],
                finish=[lambda stream: out.copy_(dst)],
            )
            chain.wait(consumer)
            with torch.cuda.stream(consumer):
                equal = torch.equal(out, src)
            self.assertTrue(equal, f"chunk {chunk}")

    def test_raising_chunk_keeps_earlier_chunks_tracked(self):
        """A chunk that raises aborts the run, but the chunks already
        enqueued stay visible to brackets, ``retired`` and ``wait``."""
        chain = self._chain()
        gate = chain.gate
        src = self._src(_MiB)
        dst = self._pinned(_MiB)
        torch.cuda.synchronize()
        gate._word.value = 7
        caller = torch.cuda.current_stream()

        def broken(stream):
            raise ValueError("chunk failed")

        with self.assertRaises(ValueError):
            chain.run(copy_tensor_chunks(dst, src, None) + [broken], self.after)
        self.assertEqual(torch.cuda.current_stream(), caller)
        self.assertTrue(gate.has_pending())
        self.assertFalse(chain.retired())
        gate.force_open()
        self.assertTrue(_wait_until(chain.retired))
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_ungated_chain_tracks_its_last_run(self):
        chain = self._chain(gated=False)
        blocker = _HostWord()
        src = self._src(_MiB)
        dst = torch.zeros_like(src)
        torch.cuda.synchronize()
        ops = copy_tensor_chunks(dst, src, None)
        ops[0] = _Blocked(ops[0], blocker)
        chain.run(ops, self.after)
        ops[0].wait_reached()
        self.assertFalse(chain.retired())
        blocker.set(1)
        self.assertTrue(_wait_until(chain.retired))

    def test_closed_chain_refuses_use(self):
        chain = self._chain()
        gate = chain.gate
        chain.close()
        self.assertEqual(gate._chains, [])
        for use in (chain.retired, lambda: chain.wait(self.after)):
            with self.assertRaises(AssertionError):
                use()

    def test_pause_during_a_running_chunk_holds_the_next(self):
        """The case the event chaining exists for: the pause lands while
        chunk 0's (long) copy runs; chunk 1's gate must not have been read
        ahead of it."""
        chain = self._chain()
        gate = chain.gate
        blocker = _HostWord()
        chunk = 512 * _MiB  # ~10 ms over PCIe: the pause lands mid-copy
        src = self._src(2 * chunk)
        dst = self._pinned(src.numel())
        torch.cuda.synchronize()
        ops = copy_tensor_chunks(dst, src, chunk)
        ops[0] = _Blocked(ops[0], blocker)
        done = torch.cuda.Event()
        chain.run(ops, self.after, done=done)
        ops[0].wait_reached()
        blocker.set(1)
        gate._word.value = 6
        time.sleep(0.3)
        self.assertEqual(int(dst[chunk:].count_nonzero()), 0)
        self.assertEqual(int(dst[:chunk].count_nonzero()), chunk)
        gate.force_open()
        self.assertTrue(_wait_until(done.query))
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_bracket_pauses_in_flight_gates_of_its_link_only(self):
        nvlink, h2d = self._chain(Link.NVLINK), self._chain(Link.PCIE_H2D)
        blocker = _HostWord()
        src = self._src(2 * _MiB)
        dsts = [torch.zeros_like(src) for _ in range(2)]
        dones = [torch.cuda.Event() for _ in range(2)]
        torch.cuda.synchronize()
        for chain, dst, done in zip((nvlink, h2d), dsts, dones):
            ops = copy_tensor_chunks(dst, src, _MiB)
            ops[0] = _Blocked(ops[0], blocker)
            chain.run(ops, self.after, done=done)
        with torch.cuda.stream(self.control):
            with paused_background_transfers(Link.NVLINK):
                self.assertEqual(nvlink.gate._pause_depth, 1)
                self.assertEqual(h2d.gate._pause_depth, 0)
                # Harmless: the chain owns its streams; its chunks just park.
                nvlink.run([], self.after)
                with self.assertRaises(RuntimeError):
                    # The waiting stream would block on its own pause.
                    nvlink.wait(self.control, dones[0])
                h2d.wait(self.control, dones[1])  # another link's gate: fine
            self.assertEqual(nvlink.gate._pause_depth, 0)
        blocker.set(1)
        for done in dones:
            self.assertTrue(_wait_until(done.query))
        self.control.synchronize()
        for dst in dsts:
            self.assertTrue(torch.equal(dst, src))
        # Drained: the next bracket pauses nothing.
        with torch.cuda.stream(self.control):
            with paused_background_transfers(Link.NVLINK | Link.PCIE):
                self.assertEqual(nvlink.gate._pause_depth, 0)
                self.assertEqual(h2d.gate._pause_depth, 0)

    def test_watchdog_forces_a_stuck_pause_open(self):
        """A pause whose resume never comes: the watchdog opens the gate and
        the parked chain completes."""
        chain = self._chain()
        gate = chain.gate
        src = self._src(2 * _MiB)
        dst = self._pinned(src.numel())
        gate.pause(self.control)  # no resume follows
        self.control.synchronize()
        chain.run(copy_tensor_chunks(dst, src, _MiB), self.after)
        time.sleep(0.05)
        self.assertFalse(chain.retired())
        stop = threading.Event()

        def watchdog():
            # One gate, a 50 ms timeout, the production sampling.
            while not stop.is_set():
                time.sleep(0.0125)
                check_gate_held(gate, time.monotonic(), 0.05)

        thread = threading.Thread(target=watchdog, daemon=True)
        with self.assertLogs(gated_transfer.logger, level="ERROR"):
            thread.start()
            self.assertTrue(_wait_until(chain.retired, timeout=5))
            stop.set()
            thread.join(2)
        self.assertEqual(gate.forced_opens, 1)
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_watchdog_thread_reads_its_timeout_from_the_env(self):
        with envs.SGLANG_GATED_TRANSFER_WATCHDOG_MS.override(0):
            gated_transfer._watch_gates()  # returns at once when disabled

    def test_capturing_after_stream_is_refused(self):
        chain = self._chain()
        src = self._src(_MiB)
        dst = torch.zeros_like(src)
        stream = torch.cuda.Stream(self.device)
        dst.add_(1)  # load the kernel outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with self.assertRaises(RuntimeError):
                chain.run(copy_tensor_chunks(dst, src, None), stream)
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
    gate = get_transfer_gate(Link.NVLINK)
    chain = TransferChain(rank, gate)
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
            chain.run(chunks, after, done=done)
        chain.wait(consumer, done)
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
        chain.run(held, after, done=done)
    assert gate.has_pending()
    x = torch.full((n,), float(rank + 1), device="cuda")
    pauses = gate._seq
    with foreground.change_state(enable=True):
        with paused_background_transfers(Link.NVLINK):
            foreground.all_reduce(x)
    assert gate._seq == pauses + 1 and gate._pause_depth == 0
    blocker.set(1)  # host only: nothing waits on the GPU before the release
    chain.wait(consumer, done)
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
