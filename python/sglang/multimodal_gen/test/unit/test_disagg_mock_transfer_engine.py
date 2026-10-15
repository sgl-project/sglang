# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the non-RDMA MockTransferEngine fallback."""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport import engine as engine_module
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    MockTransferEngine,
    MooncakeDiffusionEngine,
    create_transfer_engine,
)


class TestMockTransferEngine(unittest.TestCase):
    def test_round_trip_delivers_bytes_at_matching_offset(self):
        sender_pool = torch.zeros(4096, dtype=torch.uint8)
        receiver_pool = torch.zeros(4096, dtype=torch.uint8)

        sender = MockTransferEngine(hostname="127.0.0.1")
        receiver = MockTransferEngine(hostname="127.0.0.1")
        self.addCleanup(sender.deregister_buffer, sender_pool.data_ptr())
        self.addCleanup(receiver.deregister_buffer, receiver_pool.data_ptr())

        sender.register_buffer(sender_pool.data_ptr(), sender_pool.numel())
        receiver.register_buffer(receiver_pool.data_ptr(), receiver_pool.numel())

        payload = torch.arange(256, dtype=torch.uint8)
        src_offset, dst_offset = 128, 512
        sender_pool[src_offset : src_offset + len(payload)] = payload

        ret = sender.transfer_sync(
            receiver.session_id,
            sender_pool.data_ptr() + src_offset,
            receiver_pool.data_ptr() + dst_offset,
            len(payload),
        )

        self.assertEqual(ret, 0)
        torch.testing.assert_close(
            receiver_pool[dst_offset : dst_offset + len(payload)], payload
        )
        self.assertTrue(torch.all(receiver_pool[:dst_offset] == 0))
        self.assertTrue(torch.all(receiver_pool[dst_offset + len(payload) :] == 0))

    def test_write_outside_registered_buffer_is_rejected(self):
        sender_pool = torch.zeros(64, dtype=torch.uint8)
        receiver_pool = torch.zeros(64, dtype=torch.uint8)

        sender = MockTransferEngine(hostname="127.0.0.1")
        receiver = MockTransferEngine(hostname="127.0.0.1")
        self.addCleanup(sender.deregister_buffer, sender_pool.data_ptr())
        self.addCleanup(receiver.deregister_buffer, receiver_pool.data_ptr())

        sender.register_buffer(sender_pool.data_ptr(), sender_pool.numel())
        receiver.register_buffer(receiver_pool.data_ptr(), receiver_pool.numel())

        ret = sender.transfer_sync(
            receiver.session_id,
            sender_pool.data_ptr(),
            receiver_pool.data_ptr() + receiver_pool.numel(),  # one past the end
            16,
        )

        self.assertNotEqual(ret, 0)

    def test_batch_delivers_all_tensors_over_one_connection(self):
        sender_pool = torch.zeros(4096, dtype=torch.uint8)
        receiver_pool = torch.zeros(4096, dtype=torch.uint8)

        sender = MockTransferEngine(hostname="127.0.0.1")
        receiver = MockTransferEngine(hostname="127.0.0.1")
        self.addCleanup(sender.deregister_buffer, sender_pool.data_ptr())
        self.addCleanup(receiver.deregister_buffer, receiver_pool.data_ptr())

        sender.register_buffer(sender_pool.data_ptr(), sender_pool.numel())
        receiver.register_buffer(receiver_pool.data_ptr(), receiver_pool.numel())

        payloads = [
            torch.arange(i * 16, (i + 1) * 16, dtype=torch.uint8) for i in range(4)
        ]
        src_offsets = [0, 64, 128, 192]
        dst_offsets = [512, 768, 1024, 1280]
        for payload, src_offset in zip(payloads, src_offsets):
            sender_pool[src_offset : src_offset + len(payload)] = payload

        ret = sender.batch_transfer_sync(
            receiver.session_id,
            [sender_pool.data_ptr() + o for o in src_offsets],
            [receiver_pool.data_ptr() + o for o in dst_offsets],
            [len(p) for p in payloads],
        )

        self.assertEqual(ret, 0)
        for payload, dst_offset in zip(payloads, dst_offsets):
            torch.testing.assert_close(
                receiver_pool[dst_offset : dst_offset + len(payload)], payload
            )

    def test_auto_backend_falls_back_to_mock_without_mooncake(self):
        with patch.object(engine_module, "_check_mooncake", return_value=False):
            engine = create_transfer_engine(hostname="127.0.0.1", backend="auto")
        self.addCleanup(engine.deregister_buffer, 0)
        self.assertIsInstance(engine, MockTransferEngine)
        self.assertFalse(engine.supports_gpu_direct)

    def test_auto_backend_prefers_mooncake_once_it_imports(self):
        with (
            patch.object(engine_module, "_check_mooncake", return_value=True),
            patch.object(MooncakeDiffusionEngine, "__init__", return_value=None),
        ):
            engine = create_transfer_engine(hostname="127.0.0.1", backend="auto")
        self.assertIsInstance(engine, MooncakeDiffusionEngine)


if __name__ == "__main__":
    unittest.main()
