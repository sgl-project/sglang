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


if __name__ == "__main__":
    unittest.main()
