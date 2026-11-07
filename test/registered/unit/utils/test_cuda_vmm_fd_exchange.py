"""Real SCM_RIGHTS transport; only rendezvous and timeout barriers are injected."""

import errno
import os
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from sglang.srt.utils import cuda_vmm_utils as vmm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@unittest.skipUnless(
    hasattr(socket, "SCM_RIGHTS") and hasattr(socket, "SOCK_SEQPACKET"),
    "Requires UNIX SOCK_SEQPACKET/SCM_RIGHTS",
)
class TestPosixFdCleanup(CustomTestCase):
    def run_exchange(self, mode):
        received = []
        peer_errors = []
        proceed = threading.Event()
        first_received = threading.Event()
        second_receive = threading.Event()
        peer_done = threading.Event()
        original_recv = vmm._recv_fd
        original_send = vmm._send_fd
        with tempfile.TemporaryDirectory(prefix="sgl_fd_test_") as directory:
            peer_path = os.path.join(directory, "peer.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
                listener.bind(peer_path)
                listener.listen(1)
                listener.settimeout(3)
                with open(os.devnull, "rb") as exported:

                    def peer(path):
                        try:
                            with socket.socket(
                                socket.AF_UNIX, socket.SOCK_SEQPACKET
                            ) as client:
                                client.settimeout(3)
                                client.connect(path)
                                vmm._send_fd(client, exported.fileno(), 1, 0)
                                connection, _ = listener.accept()
                                connection.close()
                                if mode == "receive_error":
                                    client.send(b"bad header")
                                elif mode == "duplicate":
                                    vmm._send_fd(client, exported.fileno(), 1, 0)
                                elif mode == "late_receive":
                                    if not proceed.wait(3):
                                        raise RuntimeError(
                                            "late peer barrier timed out"
                                        )
                                    vmm._send_fd(client, exported.fileno(), 1, 1)
                        except BaseException as error:
                            peer_errors.append(error)
                        finally:
                            peer_done.set()

                    peer_thread = None
                    receiver_thread = None

                    def gather(paths, local_path, *, group):
                        nonlocal peer_thread
                        paths[:] = [local_path, peer_path]
                        # Capture the actual rendezvous path, not a fake socket.
                        peer_thread = threading.Thread(target=peer, args=(local_path,))
                        peer_thread.start()

                    calls = 0

                    def receive(sock):
                        nonlocal calls, receiver_thread
                        receiver_thread = threading.current_thread()
                        calls += 1
                        if mode == "late_receive" and calls == 2:
                            second_receive.set()
                            if not proceed.wait(3):
                                raise RuntimeError("receive barrier timed out")
                        packet = original_recv(sock)
                        if packet is not None:
                            received.append(packet[2])
                            first_received.set()
                        return packet

                    def send(sock, fd, rank, base_idx):
                        if mode == "send_error" and rank == 0:
                            if not first_received.wait(3):
                                raise RuntimeError("first receive barrier timed out")
                            raise RuntimeError("injected send failure")
                        original_send(sock, fd, rank, base_idx)

                    result = None
                    try:
                        with (
                            patch.object(vmm.dist, "all_gather_object", gather),
                            patch.object(vmm, "_recv_fd", receive),
                            patch.object(vmm, "_send_fd", send),
                            patch.object(vmm, "_FD_SEND_TIMEOUT_S", 0.2),
                        ):
                            if mode == "success":
                                result = vmm.exchange_posix_fds(None, 0, 2, [], [0, 1])
                                self.assertEqual(set(result), {(1, 0)})
                                os.fstat(result[(1, 0)])
                            else:
                                message = {
                                    "late_receive": "timed out",
                                    "send_error": "injected send failure",
                                    "mismatch": "exchange mismatch",
                                }.get(mode, "receive failed")
                                with self.assertRaisesRegex(RuntimeError, message):
                                    vmm.exchange_posix_fds(
                                        None,
                                        0,
                                        2,
                                        [exported.fileno()]
                                        if mode == "send_error"
                                        else [],
                                        [0, 2] if mode == "mismatch" else [0, 1],
                                    )
                            if mode == "late_receive":
                                self.assertTrue(second_receive.is_set())
                            proceed.set()
                            self.assertTrue(peer_done.wait(3))
                            peer_thread.join(3)
                            if receiver_thread is not None:
                                receiver_thread.join(3)
                                self.assertFalse(receiver_thread.is_alive())
                            # A second packet is processed after the failed caller has returned.
                            if mode == "late_receive":
                                self.assertEqual(len(received), 2)
                            self.assertEqual(peer_errors, [])
                            if result is None:
                                self.assertGreaterEqual(len(received), 1)
                                for fd in received:
                                    with self.assertRaises(OSError) as caught:
                                        os.fstat(fd)
                                    self.assertEqual(
                                        caught.exception.errno, errno.EBADF
                                    )
                            os.fstat(exported.fileno())
                    finally:
                        proceed.set()
                        if peer_thread is not None:
                            peer_thread.join(3)
                        if receiver_thread is not None:
                            receiver_thread.join(3)
                        for fd in set(received):
                            try:
                                os.close(fd)
                            except OSError:
                                pass

    def test_success_transfers_fd_ownership(self):
        self.run_exchange("success")

    def test_receive_error_closes_accumulated_fds(self):
        self.run_exchange("receive_error")

    def test_duplicate_closes_both_fds(self):
        self.run_exchange("duplicate")

    def test_timeout_closes_accumulated_and_late_fds(self):
        self.run_exchange("late_receive")

    def test_send_error_closes_accumulated_fds(self):
        self.run_exchange("send_error")

    def test_mismatch_closes_accumulated_fds(self):
        self.run_exchange("mismatch")


if __name__ == "__main__":
    unittest.main()
