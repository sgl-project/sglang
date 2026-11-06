"""Local broadcaster port selection and real concurrent XPUB handshakes."""

import socket
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import zmq

from sglang.srt.distributed.device_communicators import shm_broadcast as shm
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestShmBroadcastPortWindows(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(envs.SGLANG_PORT.override(43000))
        self.enterContext(envs.SGLANG_SHM_PORTS_PER_RANK.override(0))
        self.initialized = self.enterContext(
            patch.object(shm.dist, "is_initialized", return_value=True)
        )
        self.rank = self.enterContext(
            patch.object(shm.dist, "get_rank", return_value=0)
        )

    def test_default_range_does_not_query_distributed_rank(self):
        self.rank.side_effect = AssertionError("rank must not be queried")
        self.assertEqual(shm._get_local_subscribe_port_range(), (43000, 43008))
        self.initialized.assert_not_called()
        with envs.SGLANG_PORT.override(None):
            self.assertIsNone(shm._get_local_subscribe_port_range())

    def test_global_ranks_have_disjoint_windows(self):
        with envs.SGLANG_SHM_PORTS_PER_RANK.override(128):
            windows = []
            for rank in (0, 2, 4, 6, 8, 14):
                self.rank.return_value = rank
                window = shm._get_local_subscribe_port_range()
                self.assertEqual(window, (43000 + rank * 128, 43000 + (rank + 1) * 128))
                windows.append(window)
            self.assertTrue(all(a[1] <= b[0] for a, b in zip(windows, windows[1:])))
            self.rank.assert_called_with()

    def test_standalone_queue_uses_rank_zero(self):
        self.initialized.return_value = False
        self.rank.side_effect = AssertionError("default process group is absent")
        with envs.SGLANG_SHM_PORTS_PER_RANK.override(128):
            self.assertEqual(shm._get_local_subscribe_port_range(), (43000, 43128))

    def test_last_tcp_port_is_valid(self):
        with (
            envs.SGLANG_PORT.override(65408),
            envs.SGLANG_SHM_PORTS_PER_RANK.override(128),
        ):
            self.assertEqual(shm._get_local_subscribe_port_range(), (65408, 65536))

    def test_invalid_ranges_fail_before_allocating_resources(self):
        for base, width, rank in (
            (None, 128, 0),
            (43000, -1, 0),
            (0, 128, 0),
            (-1, 128, 0),
            (65530, 0, 0),
            (65500, 128, 0),
            (65000, 128, 6),
        ):
            with (
                self.subTest(base=base, width=width, rank=rank),
                envs.SGLANG_PORT.override(base),
                envs.SGLANG_SHM_PORTS_PER_RANK.override(width),
                patch.object(shm, "ShmRingBuffer") as buffer,
                patch.object(shm, "Context") as context,
            ):
                self.rank.return_value = rank
                with self.assertRaises(ValueError):
                    shm.MessageQueue(n_reader=1, n_local_reader=1)
                buffer.assert_not_called()
                context.assert_not_called()

    def test_queue_without_local_readers_does_not_need_a_local_range(self):
        with (
            envs.SGLANG_PORT.override(None),
            envs.SGLANG_SHM_PORTS_PER_RANK.override(128),
            patch.object(shm, "Context") as context,
        ):
            queue = shm.MessageQueue(n_reader=0, n_local_reader=0)
            self.assertIsNone(queue.export_handle().local_subscribe_port)
            context.return_value.socket.assert_not_called()

    def test_constructor_preserves_zmq_default_port_selection(self):
        with (
            envs.SGLANG_PORT.override(None),
            patch.object(shm, "Context") as context,
            patch.object(shm, "ShmRingBuffer"),
        ):
            shm.MessageQueue(n_reader=1, n_local_reader=1)
            context.return_value.socket.return_value.bind_to_random_port.assert_called_once_with(
                "tcp://127.0.0.1"
            )

    def test_concurrent_writers_complete_real_ready_handshakes(self):
        # Choose a dynamic unprivileged base; leave room for all four windows.
        for _ in range(32):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                base = probe.getsockname()[1]
            if base + 7 * 128 <= 65536:
                break
        else:
            self.fail("Could not select a base with room for the test port windows")

        ranks = threading.local()
        self.rank.side_effect = lambda: ranks.value

        def exercise(rank):
            ranks.value = rank
            for iteration in range(8):
                queues = []
                try:
                    writer = shm.MessageQueue(
                        n_reader=1, n_local_reader=1, max_chunk_bytes=64, max_chunks=1
                    )
                    queues.append(writer)
                    handle = writer.export_handle()
                    self.assertTrue(
                        base + rank * 128
                        <= handle.local_subscribe_port
                        < base + (rank + 1) * 128
                    )
                    reader = shm.MessageQueue.create_from_handle(handle, rank=0)
                    queues.append(reader)
                    for queue in queues:
                        queue.local_socket.setsockopt(zmq.RCVTIMEO, 3000)
                        queue.local_socket.setsockopt(zmq.SNDTIMEO, 3000)
                    writer.wait_until_ready()
                    reader.wait_until_ready()
                    message = f"rank={rank},iteration={iteration}".encode()
                    writer.local_socket.send(message)
                    self.assertEqual(reader.local_socket.recv(), message)
                finally:
                    contexts = {queue.local_socket.context for queue in queues}
                    for queue in queues:
                        queue.local_socket.close(linger=0)
                    for context in contexts:
                        context.term()
            return 8

        with (
            envs.SGLANG_PORT.override(base),
            envs.SGLANG_SHM_PORTS_PER_RANK.override(128),
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            futures = [executor.submit(exercise, rank) for rank in (0, 2, 4, 6)]
            self.assertEqual(sum(future.result(timeout=40) for future in futures), 32)


if __name__ == "__main__":
    unittest.main()
