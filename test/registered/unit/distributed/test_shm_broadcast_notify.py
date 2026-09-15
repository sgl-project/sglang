"""Unit tests for the idle-reader notify channel in shm_broadcast.MessageQueue.

MessageQueue is exercised in-process: the writer and the reader share the
same ShmRingBuffer through the exported Handle, and rendezvous via
wait_until_ready on a background thread.
"""

import os
import threading
import time
import unittest
from unittest.mock import patch

import sglang.srt.distributed.device_communicators.shm_broadcast as shm_broadcast
from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _make_queue(n_local_reader=1):
    """A writer plus `n_local_reader` local readers, handshaken and ready."""
    writer = MessageQueue(
        n_reader=n_local_reader,
        n_local_reader=n_local_reader,
        local_reader_ranks=list(range(1, n_local_reader + 1)),
    )
    handle = writer.export_handle()
    readers = [
        MessageQueue.create_from_handle(handle, rank=r)
        for r in range(1, n_local_reader + 1)
    ]
    # wait_until_ready rendezvous: the writer collects subscription messages
    # from every reader before publishing READY, so run it off-thread.
    t = threading.Thread(target=writer.wait_until_ready)
    t.start()
    for reader in readers:
        reader.wait_until_ready()
    t.join(timeout=10)
    assert not t.is_alive(), "writer wait_until_ready did not finish"
    return writer, readers


class TestMessageQueueNotify(CustomTestCase):
    def test_broadcast_roundtrip(self):
        writer, (reader,) = _make_queue()
        writer.enqueue({"a": 1})
        self.assertEqual(reader.dequeue(), {"a": 1})

    def test_enqueue_sends_notify_ping(self):
        writer, (reader,) = _make_queue()
        writer.enqueue("x")
        ready = reader.notify_poller.poll(2000)
        self.assertTrue(ready, "reader was not notified of the write")
        self.assertEqual(reader.local_notify_socket.recv(), b"")

    def test_idle_reader_blocks_instead_of_spinning(self):
        # With no writes and busy_loop_s elapsed, _wait_for_write must block
        # on the notify socket until the deadline rather than sched_yield.
        _, (reader,) = _make_queue()
        yields = []

        def counting_yield():
            yields.append(1)
            os.sched_yield()

        with patch.object(shm_broadcast.os, "sched_yield", counting_yield):
            reader._wait_for_write(deadline=time.monotonic() + 0.3, busy_loop_s=0.0)
        self.assertEqual(yields, [])

    def test_idle_reader_wakes_on_enqueue(self):
        writer, (reader,) = _make_queue()
        got = []

        def read():
            with envs.SGLANG_RINGBUFFER_BUSY_LOOP_S.override(0.0):
                got.append(reader.dequeue())

        t = threading.Thread(target=read)
        t.start()
        # Let the reader go idle well past the busy-loop window so it is
        # parked on the notify socket rather than spinning.
        time.sleep(0.5)
        writer.enqueue("wake")
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "reader did not wake on enqueue")
        self.assertEqual(got, ["wake"])

    def test_barrier_rendezvous(self):
        writer, (reader,) = _make_queue()
        released = []
        t = threading.Thread(target=lambda: released.append(reader.barrier()))
        t.start()
        # A reader that arrives first must block until the writer collects
        # its ack and publishes the release.
        time.sleep(0.5)
        self.assertTrue(t.is_alive(), "reader barrier did not block")
        writer.barrier()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "reader barrier did not release")
        self.assertEqual(released, [None])

    def test_barrier_waits_for_all_readers(self):
        writer, (r0, r1) = _make_queue(n_local_reader=2)
        done = []
        t0 = threading.Thread(target=lambda: done.append(r0.barrier()))
        tw = threading.Thread(target=lambda: done.append(writer.barrier()))
        t0.start()
        tw.start()
        # With r1's ack outstanding the writer must not release early.
        time.sleep(0.5)
        self.assertTrue(tw.is_alive(), "writer released before all readers arrived")
        self.assertTrue(t0.is_alive())
        done.append(r1.barrier())
        t0.join(timeout=5)
        tw.join(timeout=5)
        self.assertFalse(t0.is_alive())
        self.assertFalse(tw.is_alive())
        self.assertEqual(len(done), 3)

    def test_all_gather_object_rank_order(self):
        writer, (r0, r1) = _make_queue(n_local_reader=2)
        got = {}
        threads = [
            threading.Thread(
                target=lambda r=r, i=i: got.setdefault(
                    i, r.all_gather_object(f"obj{i}")
                )
            )
            for i, r in enumerate([r0, r1], start=1)
        ]
        for t in threads:
            t.start()
        got[0] = writer.all_gather_object("obj0")
        for t in threads:
            t.join(timeout=5)
            self.assertFalse(t.is_alive(), "reader all_gather_object did not finish")
        for gathered in got.values():
            self.assertEqual(gathered, ["obj0", "obj1", "obj2"])


if __name__ == "__main__":
    unittest.main()
