"""Clock-domain and framing contracts on real ZMQ sockets, without a server."""

import unittest
from unittest.mock import patch

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct, ipc_metrics
from sglang.srt.managers.io_struct import AbortReq

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DOMAIN = b"a" * 24


class TestIpcMetrics(CustomTestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.addCleanup(self.context.destroy, linger=0)
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.setsockopt(zmq.RCVTIMEO, 5000)
        self.receiver.bind("inproc://delivery")
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.SNDTIMEO, 5000)
        self.sender.connect("inproc://delivery")

    def test_both_codecs_timestamp_before_serialization(self):
        """Moving the clock read after encoding silently omits sender work."""
        for pickle_ipc in (False, True):
            with self.subTest(pickle_ipc=pickle_ipc):
                clock = [100]
                encoder_owner = io_struct.pickle if pickle_ipc else io_struct
                encoder_name = "dumps" if pickle_ipc else "msgpack_encode"
                encode = getattr(encoder_owner, encoder_name)

                def slow_encode(*args, **kwargs):
                    clock[0] = 200
                    return encode(*args, **kwargs)

                message = AbortReq(rid="before-serialization")
                with (
                    patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                    patch.object(encoder_owner, encoder_name, side_effect=slow_encode),
                    patch.object(
                        ipc_metrics.time, "monotonic_ns", side_effect=lambda: clock[0]
                    ),
                ):
                    ipc_metrics.sock_send_with_timestamp(
                        self.sender, message, clock_domain=DOMAIN
                    )
                    stamp, received = ipc_metrics.sock_recv_with_timestamp(
                        self.receiver, expected_clock_domain=DOMAIN
                    )
                self.assertEqual(received, message)
                self.assertEqual(stamp, 100)
                self.assertFalse(self.receiver.getsockopt(zmq.RCVMORE))

    def test_unverified_metadata_preserves_payload_and_next_message(self):
        """Bad metadata must neither sample a foreign clock nor desync framing."""
        timestamp = (123).to_bytes(8, "big")
        metadata_cases = [
            [],  # Original single-frame IPC.
            [timestamp],  # Old, unverified timestamp-only frame.
            [timestamp + b"\x01" + b"b" * 24],
            [timestamp + b"\x02" + DOMAIN],
            [b"short"],
            [timestamp + b"\x01" + DOMAIN, b"extra"],
        ]
        for pickle_ipc in (False, True):
            for frames in metadata_cases:
                with (
                    self.subTest(pickle_ipc=pickle_ipc, frames=frames),
                    patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                ):
                    message = AbortReq(rid="legacy-or-invalid")
                    io_struct.sock_send(
                        self.sender, message, flags=zmq.SNDMORE if frames else 0
                    )
                    for index, frame in enumerate(frames):
                        self.sender.send(
                            frame, flags=zmq.SNDMORE if index + 1 < len(frames) else 0
                        )
                    next_message = AbortReq(rid="next")
                    io_struct.sock_send(self.sender, next_message)
                    stamp, received = ipc_metrics.sock_recv_with_timestamp(
                        self.receiver, expected_clock_domain=DOMAIN
                    )
                    self.assertIsNone(stamp)
                    self.assertEqual(received, message)
                    self.assertEqual(io_struct.sock_recv(self.receiver), next_message)

    def test_unknown_receiver_domain_does_not_trust_valid_metadata(self):
        message = AbortReq(rid="unknown-receiver")
        ipc_metrics.sock_send_with_timestamp(self.sender, message, clock_domain=DOMAIN)
        self.assertEqual(
            ipc_metrics.sock_recv_with_timestamp(
                self.receiver, expected_clock_domain=None
            ),
            (None, message),
        )

    def test_domain_tracks_boot_and_time_namespace(self):
        with (
            patch.object(ipc_metrics.sys, "platform", "linux"),
            patch.object(
                ipc_metrics.Path,
                "read_text",
                return_value="00000000-0000-0000-0000-000000000001",
            ) as boot,
            patch.object(
                ipc_metrics.os, "readlink", return_value="time:[123]"
            ) as namespace,
        ):
            original = ipc_metrics.read_clock_domain()
            self.assertEqual(len(original), 24)
            self.assertEqual(ipc_metrics.read_clock_domain(), original)
            namespace.return_value = "time:[124]"
            self.assertNotEqual(ipc_metrics.read_clock_domain(), original)
            namespace.return_value = "time:[123]"
            boot.return_value = "00000000-0000-0000-0000-000000000002"
            self.assertNotEqual(ipc_metrics.read_clock_domain(), original)
            namespace.side_effect = OSError("unavailable")
            self.assertIsNone(ipc_metrics.read_clock_domain())
        with patch.object(ipc_metrics.sys, "platform", "darwin"):
            self.assertIsNone(ipc_metrics.read_clock_domain())


if __name__ == "__main__":
    unittest.main()
