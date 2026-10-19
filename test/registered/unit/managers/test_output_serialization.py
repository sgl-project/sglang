import asyncio
import gc
import unittest
from unittest.mock import patch

import pybase64
import torch
import zmq
import zmq.asyncio

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct  # noqa: E402
from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOutputSerialization(unittest.TestCase):
    def test_base64_preserves_bytes_and_missing_items(self):
        tensor = torch.arange(120, dtype=torch.int32).reshape(5, 4, 6)
        items = [tensor, tensor.transpose(0, 1), torch.empty((0, 2)), None]
        expected = [
            pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")
            if item is not None
            else None
            for item in items
        ]
        encode = DetokenizerManager._b64_encode_per_request
        self.assertEqual(encode(items), expected)
        self.assertIsNone(encode(None))
        self.assertEqual(encode([]), [])

    def make_output(self):
        return io_struct.BatchEmbeddingOutput(
            rids=["large", "missing", "empty"],
            finished_reasons=[None] * 3,
            embeddings=[[], [], []],
            prompt_tokens=[0] * 3,
            cached_tokens=[0] * 3,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            pooled_hidden_states=[
                torch.arange(262144, dtype=torch.int32).reshape(512, 512).T,
                None,
                torch.empty((0, 2), dtype=torch.int32),
            ],
        )

    def check_output(self, received, original):
        # The receive Frame is gone. Decoded tensors must still own writable data.
        gc.collect()
        self.assertEqual(received.rids, original.rids)
        self.assertIsNone(received.pooled_hidden_states[1])
        for i in (0, 2):
            self.assertTrue(
                torch.equal(
                    received.pooled_hidden_states[i], original.pooled_hidden_states[i]
                )
            )
        received.pooled_hidden_states[0].add_(1)
        self.assertFalse(
            torch.equal(
                received.pooled_hidden_states[0], original.pooled_hidden_states[0]
            )
        )

    def test_sync_ipc_wire_compatibility_and_tensor_lifetime(self):
        context = zmq.Context()
        sender = context.socket(zmq.PAIR)
        receiver = context.socket(zmq.PAIR)
        sender.setsockopt(zmq.SNDTIMEO, 5000)
        receiver.setsockopt(zmq.RCVTIMEO, 5000)
        sender.bind("inproc://output-sync")
        receiver.connect("inproc://output-sync")
        try:
            for use_pickle in (False, True):
                with (
                    self.subTest(pickle=use_pickle),
                    patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle),
                ):
                    original = self.make_output()
                    io_struct.sock_send(sender, original)
                    self.check_output(io_struct.sock_recv(receiver), original)
            with patch.object(io_struct, "_USE_PICKLE_IPC", False):
                original = self.make_output()
                sender.send(io_struct.msgpack_encode(original))
                self.check_output(io_struct.sock_recv(receiver), original)
                io_struct.sock_send(sender, original)
                self.check_output(io_struct.msgpack_decode(receiver.recv()), original)
                blob = b"\x00\xff" * 65536
                io_struct.sock_send(sender, blob)
                self.assertEqual(io_struct.sock_recv(receiver), blob)
        finally:
            sender.close(linger=0)
            receiver.close(linger=0)
            context.term()

    def test_async_ipc_tensor_lifetime(self):
        async def run():
            context = zmq.asyncio.Context()
            sender = context.socket(zmq.PAIR)
            receiver = context.socket(zmq.PAIR)
            sender.bind("inproc://output-async")
            receiver.connect("inproc://output-async")
            try:
                for use_pickle in (False, True):
                    with (
                        self.subTest(pickle=use_pickle),
                        patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle),
                    ):
                        original = self.make_output()
                        await io_struct.async_sock_send(sender, original)
                        received = await io_struct.async_sock_recv(receiver)
                        self.check_output(received, original)
            finally:
                sender.close(linger=0)
                receiver.close(linger=0)
                context.term()

        asyncio.run(asyncio.wait_for(run(), timeout=10))


if __name__ == "__main__":
    unittest.main()
