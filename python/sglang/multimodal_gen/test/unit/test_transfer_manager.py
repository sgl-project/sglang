# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DiffusionTransferManager transfer logic."""

import unittest

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    MockTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
    PendingReceive,
    StagedTransfer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocMsg,
    TransferPushedMsg,
    TransferPushMsg,
    TransferStagedMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)


def _make_manager(
    pool_size: int = 16 * 1024 * 1024,
    min_block: int = 1024 * 1024,
    session_id: str | None = None,
) -> DiffusionTransferManager:
    """Create a manager with mock engine and pinned buffer."""
    engine = MockTransferEngine(session_id=session_id)
    buffer = TransferTensorBuffer(
        pool_size=pool_size,
        min_block_size=min_block,
        role_name="test",
    )
    return DiffusionTransferManager(engine=engine, buffer=buffer)


class TestStaging(unittest.TestCase):
    """Test D2H staging of GPU tensors to TransferBuffer."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_stage_single_tensor(self):
        mgr = _make_manager()
        t = torch.randn(4, 8)
        staged = mgr.stage_tensors("r1", {"data": t})

        self.assertIsNotNone(staged)
        self.assertIsInstance(staged, StagedTransfer)
        self.assertEqual(staged.request_id, "r1")
        self.assertIn("data", staged.manifest)
        self.assertIsNotNone(staged.slot)

    def test_stage_multiple_tensors(self):
        mgr = _make_manager()
        tensors = {
            "latents": torch.randn(2, 4, 16, 16),
            "embeds": torch.randn(2, 77, 768),
        }
        staged = mgr.stage_tensors("r1", tensors)

        self.assertIsNotNone(staged)
        self.assertIn("latents", staged.manifest)
        self.assertIn("embeds", staged.manifest)

    def test_stage_with_scalar_fields(self):
        mgr = _make_manager()
        staged = mgr.stage_tensors(
            "r1",
            {"t": torch.randn(4)},
            scalar_fields={"guidance_scale": 7.5, "request_id": "r1"},
        )
        self.assertEqual(staged.scalar_fields["guidance_scale"], 7.5)

    def test_stage_empty_tensors(self):
        mgr = _make_manager()
        staged = mgr.stage_tensors("r1", {})
        self.assertIsNotNone(staged)
        self.assertIsNone(staged.slot)  # No data allocated
        self.assertEqual(staged.manifest, {})

    def test_stage_none_values(self):
        mgr = _make_manager()
        staged = mgr.stage_tensors("r1", {"a": None, "b": None})
        self.assertIsNotNone(staged)

    def test_free_staged(self):
        mgr = _make_manager()
        mgr.stage_tensors("r1", {"t": torch.randn(4, 8)})
        mgr.free_staged("r1")
        # Should be idempotent
        mgr.free_staged("r1")

    def test_get_staged_info(self):
        mgr = _make_manager()
        mgr.stage_tensors("r1", {"t": torch.randn(4)})
        info = mgr.get_staged_info("r1")
        self.assertIsNotNone(info)
        self.assertEqual(info.request_id, "r1")

        self.assertIsNone(mgr.get_staged_info("nonexistent"))


class TestReceive(unittest.TestCase):
    """Test receive slot allocation and tensor loading."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_allocate_receive_slot(self):
        mgr = _make_manager()
        pending = mgr.allocate_receive_slot("r1", 1024 * 1024)

        self.assertIsNotNone(pending)
        self.assertIsInstance(pending, PendingReceive)
        self.assertEqual(pending.request_id, "r1")
        self.assertIsNotNone(pending.slot)

    def test_allocate_too_large(self):
        mgr = _make_manager(pool_size=1024 * 1024)
        pending = mgr.allocate_receive_slot("r1", 2 * 1024 * 1024)
        self.assertIsNone(pending)

    def test_get_receive_slot_addr(self):
        mgr = _make_manager()
        mgr.allocate_receive_slot("r1", 1024 * 1024)

        addr = mgr.get_receive_slot_addr("r1")
        self.assertIsNotNone(addr)
        self.assertEqual(addr, mgr.pool_data_ptr + mgr.get_receive_slot_offset("r1"))

    def test_free_receive_slot(self):
        mgr = _make_manager()
        mgr.allocate_receive_slot("r1", 1024 * 1024)
        mgr.free_receive_slot("r1")
        # Should be idempotent
        mgr.free_receive_slot("r1")
        self.assertIsNone(mgr.get_receive_slot_addr("r1"))


class TestTransfer(unittest.TestCase):
    """Test end-to-end transfer between two managers."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_full_transfer_cycle(self):
        """Test: sender stages → RDMA push → receiver loads."""
        sender = _make_manager(session_id="sender-1")
        receiver = _make_manager(session_id="receiver-1")

        # 1. Sender stages tensors (D2H)
        original = torch.randn(2, 4, 8, 8)
        staged = sender.stage_tensors("r1", {"latents": original})
        self.assertIsNotNone(staged)

        # 2. Receiver allocates slot
        pending = receiver.allocate_receive_slot("r1", staged.slot.size)
        self.assertIsNotNone(pending)

        # 3. Sender pushes via RDMA to receiver's slot
        dest_addr = receiver.get_receive_slot_addr("r1")
        ok = sender.push_to_peer(
            "r1",
            dest_session_id=receiver.session_id,
            dest_addr=dest_addr,
            transfer_size=staged.slot.size,
        )
        self.assertTrue(ok)

        # 4. Receiver loads tensors (H2D)
        loaded = receiver.load_tensors("r1", staged.manifest, device="cpu")
        self.assertIn("latents", loaded)
        torch.testing.assert_close(loaded["latents"], original)

        # 5. Cleanup
        sender.free_staged("r1")
        receiver.free_receive_slot("r1")

    def test_transfer_multiple_tensors(self):
        """Test transfer with multiple tensor fields."""
        sender = _make_manager(session_id="s")
        receiver = _make_manager(session_id="r")

        originals = {
            "embeds": torch.randn(2, 77, 768),
            "latents": torch.randn(2, 4, 32, 32),
        }
        staged = sender.stage_tensors("r1", originals)

        pending = receiver.allocate_receive_slot("r1", staged.slot.size)
        dest_addr = receiver.get_receive_slot_addr("r1")

        ok = sender.push_to_peer("r1", "r", dest_addr, staged.slot.size)
        self.assertTrue(ok)

        loaded = receiver.load_tensors("r1", staged.manifest, device="cpu")
        torch.testing.assert_close(loaded["embeds"], originals["embeds"])
        torch.testing.assert_close(loaded["latents"], originals["latents"])

        sender.free_staged("r1")
        receiver.free_receive_slot("r1")

    def test_push_without_staging_fails(self):
        sender = _make_manager(session_id="s")
        ok = sender.push_to_peer("nonexistent", "dest", 0, 100)
        self.assertFalse(ok)

    def test_load_without_allocation_raises(self):
        receiver = _make_manager(session_id="r")
        with self.assertRaises(ValueError):
            receiver.load_tensors("nonexistent", {}, device="cpu")

    def test_concurrent_transfers(self):
        """Multiple requests transferred concurrently."""
        sender = _make_manager(session_id="s", pool_size=64 * 1024 * 1024)
        receiver = _make_manager(session_id="r", pool_size=64 * 1024 * 1024)

        for i in range(4):
            rid = f"r{i}"
            original = torch.randn(2, 4, 8, 8)
            staged = sender.stage_tensors(rid, {"data": original})
            pending = receiver.allocate_receive_slot(rid, staged.slot.size)
            dest_addr = receiver.get_receive_slot_addr(rid)

            ok = sender.push_to_peer(rid, "r", dest_addr, staged.slot.size)
            self.assertTrue(ok)

            loaded = receiver.load_tensors(rid, staged.manifest, device="cpu")
            torch.testing.assert_close(loaded["data"], original)

            sender.free_staged(rid)
            receiver.free_receive_slot(rid)


class TestTransferProtocol(unittest.TestCase):
    """Test transfer protocol message encoding/decoding."""

    def test_encode_decode_staged(self):
        msg = TransferStagedMsg(
            request_id="r1",
            data_size=1024,
            manifest={"latents": [{"offset": 0, "shape": [4], "dtype": "float32"}]},
            session_id="session-1",
            pool_ptr=0x1000,
            slot_offset=0,
        )
        frames = encode_transfer_msg(msg)
        self.assertEqual(len(frames), 2)

        decoded = decode_transfer_msg(frames)
        self.assertEqual(decoded["msg_type"], "transfer_staged")
        self.assertEqual(decoded["request_id"], "r1")
        self.assertEqual(decoded["data_size"], 1024)

    def test_encode_decode_alloc(self):
        msg = TransferAllocMsg(request_id="r1", data_size=2048, source_role="encoder")
        frames = encode_transfer_msg(msg)
        decoded = decode_transfer_msg(frames)
        self.assertEqual(decoded["msg_type"], "transfer_alloc")
        self.assertEqual(decoded["source_role"], "encoder")

    def test_encode_decode_push(self):
        msg = TransferPushMsg(
            request_id="r1",
            dest_session_id="sess-2",
            dest_addr=0x2000,
            transfer_size=4096,
        )
        frames = encode_transfer_msg(msg)
        decoded = decode_transfer_msg(frames)
        self.assertEqual(decoded["dest_session_id"], "sess-2")
        self.assertEqual(decoded["dest_addr"], 0x2000)

    def test_is_transfer_message(self):
        transfer_frames = encode_transfer_msg(TransferPushedMsg(request_id="r1"))
        self.assertTrue(is_transfer_message(transfer_frames))

        # Non-transfer message (e.g., tensor multipart starting with JSON)
        non_transfer = [b'{"tensor_descriptors": []}', b"data"]
        self.assertFalse(is_transfer_message(non_transfer))

    def test_decode_invalid_raises(self):
        with self.assertRaises(ValueError):
            decode_transfer_msg([b"not-transfer", b"{}"])


class TestCapacity(unittest.TestCase):
    """Test capacity reporting."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_free_slots_count(self):
        mgr = _make_manager(pool_size=16 * 1024 * 1024)
        count = mgr.free_slots_count(4 * 1024 * 1024)
        self.assertGreater(count, 0)

    def test_session_and_pool_properties(self):
        mgr = _make_manager(session_id="test-prop")
        self.assertEqual(mgr.session_id, "test-prop")
        self.assertGreater(mgr.pool_data_ptr, 0)
        self.assertEqual(mgr.pool_size, 16 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
