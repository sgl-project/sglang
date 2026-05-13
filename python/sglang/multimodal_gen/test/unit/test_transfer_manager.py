# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DiffusionTransferManager transfer logic."""

import ctypes
import os
import queue
import socket
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferMetaBuffer,
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
    MockTransferEngine,
    create_transfer_engine,
    resolve_transfer_backend,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
    PendingPeerSend,
    PendingReceive,
    StagedTransfer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAbortMsg,
    TransferAllocMsg,
    TransferAllocRejectMsg,
    TransferPeerInfoMsg,
    TransferPushedMsg,
    TransferStagedMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)


def _make_manager(
    testcase: unittest.TestCase | None = None,
    pool_size: int = 16 * 1024 * 1024,
    min_block: int = 1024 * 1024,
    session_id: str | None = None,
) -> DiffusionTransferManager:
    engine = MockTransferEngine(session_id=session_id)
    buffer = TransferTensorBuffer(
        pool_size=pool_size,
        min_block_size=min_block,
        role_name="test",
    )
    meta_buffer = TransferMetaBuffer(
        slot_count=4,
        slot_size=64 * 1024,
        role_name="test",
    )
    manager = DiffusionTransferManager(
        engine=engine,
        buffer=buffer,
        meta_buffer=meta_buffer,
        host_id="host-a",
    )
    if testcase is not None:
        testcase.addCleanup(manager.cleanup)
    return manager


def _reserve_tcp_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


class TestStaging(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)

    def test_stage_single_tensor(self):
        mgr = _make_manager(self)
        staged = mgr.stage_tensors("r1", {"data": torch.randn(4, 8)})
        self.assertIsInstance(staged, StagedTransfer)
        self.assertIn("data", staged.manifest)

    def test_stage_with_scalar_fields(self):
        mgr = _make_manager(self)
        staged = mgr.stage_tensors(
            "r1",
            {"t": torch.randn(4)},
            scalar_fields={"guidance_scale": 7.5, "request_id": "r1"},
        )
        self.assertEqual(staged.scalar_fields["guidance_scale"], 7.5)

    def test_free_staged_is_idempotent(self):
        mgr = _make_manager(self)
        mgr.stage_tensors("r1", {"t": torch.randn(4, 8)})
        mgr.free_staged("r1")
        mgr.free_staged("r1")
        self.assertIsNone(mgr.get_staged_info("r1"))


class TestReceiveReadyValidation(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)

    def test_validate_receive_ready_checks_session_offsets_and_sizes(self):
        mgr = _make_manager(self, session_id="receiver-session")
        pending = mgr.allocate_receive_slot("req-ready", 4096, 1024)
        self.assertIsNotNone(pending)

        self.assertIsNone(
            mgr.validate_receive_ready(
                "req-ready",
                dest_session_id="receiver-session",
                dest_slot_offset=pending.slot.offset,
                dest_meta_slot_offset=pending.meta_slot.offset,
                data_size=4096,
                meta_size=1024,
            )
        )
        self.assertIn(
            "receiver session mismatch",
            mgr.validate_receive_ready("req-ready", dest_session_id="old-session"),
        )
        self.assertIn(
            "metadata slot offset mismatch",
            mgr.validate_receive_ready(
                "req-ready",
                dest_meta_slot_offset=pending.meta_slot.offset + 1,
            ),
        )
        self.assertIn(
            "metadata size exceeds",
            mgr.validate_receive_ready(
                "req-ready", meta_size=pending.meta_slot.size + 1
            ),
        )


class TestReceive(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)

    def test_allocate_receive_slot(self):
        mgr = _make_manager(self)
        pending = mgr.allocate_receive_slot("r1", 1024 * 1024, 4096)
        self.assertIsInstance(pending, PendingReceive)
        self.assertIsNotNone(mgr.get_receive_slot_addr("r1"))
        self.assertIsNotNone(mgr.get_receive_meta_addr("r1"))

    def test_free_receive_slot_is_idempotent(self):
        mgr = _make_manager(self)
        mgr.allocate_receive_slot("r1", 1024 * 1024, 4096)
        mgr.free_receive_slot("r1")
        mgr.free_receive_slot("r1")
        self.assertIsNone(mgr.get_receive_slot_addr("r1"))


class TestTransfer(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)

    def test_full_transfer_cycle(self):
        sender = _make_manager(self, session_id="sender-1")
        receiver = _make_manager(self, session_id="receiver-1")
        original = torch.randn(2, 4, 8, 8)
        staged = sender.stage_tensors(
            "r1",
            {"latents": original},
            scalar_fields={"request_id": "r1", "guidance_scale": 7.5},
        )
        receiver.allocate_receive_slot("r1", staged.slot.size, staged.meta_size)

        ok = sender.push_to_peer(
            "r1",
            dest_session_id=receiver.session_id,
            dest_addr=receiver.get_receive_slot_addr("r1"),
            transfer_size=staged.slot.size,
        )
        self.assertTrue(ok)
        ok_meta = sender._engine.transfer_sync(
            receiver.session_id,
            sender.meta_pool_ptr + staged.meta_slot.offset,
            receiver.get_receive_meta_addr("r1"),
            staged.meta_size,
        )
        self.assertEqual(ok_meta, 0)

        loaded, scalar_fields, _ = receiver.load_transfer_async("r1", device="cpu")
        torch.testing.assert_close(loaded["latents"], original)
        self.assertEqual(scalar_fields["request_id"], "r1")
        self.assertEqual(scalar_fields["guidance_scale"], 7.5)

    def test_duplicate_peer_info_is_ignored_after_queueing(self):
        sender = _make_manager(self, session_id="sender-dup")
        sender.stage_tensors("dup-1", {"latents": torch.randn(1, 4, 8, 8)})
        sender._send_queues = [queue.Queue()]
        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "dup-1",
                "dest_session_id": "receiver",
                "dest_addr": 123,
                "transfer_size": 456,
            }
        )
        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "dup-1",
                "dest_session_id": "receiver",
                "dest_addr": 999,
                "transfer_size": 456,
            }
        )

        self.assertEqual(sender._send_queues[0].qsize(), 1)
        self.assertEqual(sender._pending_peer_sends["dup-1"].dest_addr, 123)

    def test_send_executor_completes_and_dedupes_terminal_state(self):
        sender = _make_manager(self, session_id="sender-exec")
        receiver = _make_manager(self, session_id="receiver-exec")
        callback = unittest.mock.MagicMock()
        sender._on_send_completion = callback
        sender._send_executors = [ThreadPoolExecutor(max_workers=2)]
        sender._send_queues = [queue.Queue()]

        staged = sender.stage_tensors("r1", {"latents": torch.randn(1, 4, 8, 8)})
        receiver.allocate_receive_slot("r1", staged.slot.size, staged.meta_size)
        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "r1",
                "dest_session_id": receiver.session_id,
                "dest_addr": receiver.get_receive_slot_addr("r1"),
                "transfer_size": staged.slot.size,
                "meta_dest_addr": receiver.get_receive_meta_addr("r1"),
                "meta_transfer_size": staged.meta_size,
                "receiver_control_endpoint": "tcp://receiver-ctrl",
            }
        )

        sender._submit_send_task(sender._send_queues[0].get_nowait(), 0)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if sender._drain_send_completions():
                break
            time.sleep(0.01)

        sender._send_executors[0].shutdown(wait=True)
        sender._send_executors = []

        callback.assert_called_once()
        self.assertEqual(sender._terminal_send_states["r1"], "success")
        self.assertIsNone(sender.get_staged_info("r1"))

        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "r1",
                "dest_session_id": receiver.session_id,
                "dest_addr": receiver.get_receive_slot_addr("r1"),
                "transfer_size": staged.slot.size,
            }
        )
        self.assertNotIn("r1", sender._pending_peer_sends)

    def test_send_failure_retries_before_terminal_success(self):
        sender = _make_manager(self, session_id="sender-retry")
        callback = unittest.mock.MagicMock()
        sender._on_send_completion = callback
        sender._send_executors = [ThreadPoolExecutor(max_workers=1)]
        sender._send_queues = [queue.Queue()]
        sender.stage_tensors("retry-1", {"latents": torch.randn(1, 4, 8, 8)})
        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "retry-1",
                "dest_session_id": "receiver",
                "dest_addr": 0x1234,
                "transfer_size": 4096,
            }
        )

        attempts = {"count": 0}

        def flaky_send(request_id, peer_info):
            attempts["count"] += 1
            if attempts["count"] < 3:
                return False, "transient failure"
            return True, None

        sender._execute_send = flaky_send

        while callback.call_count == 0:
            if sender._send_queues[0].qsize() > 0:
                sender._submit_send_task(sender._send_queues[0].get_nowait(), 0)
            sender._drain_send_completions()
            if attempts["count"] > 5:
                break
            time.sleep(0.01)

        sender._send_executors[0].shutdown(wait=True)
        sender._send_executors = []

        self.assertEqual(attempts["count"], 3)
        callback.assert_called_once()
        self.assertTrue(callback.call_args[0][3])
        self.assertEqual(sender._terminal_send_states["retry-1"], "success")

    def test_send_failure_exhausts_retries_before_terminal_failure(self):
        sender = _make_manager(self, session_id="sender-retry-fail")
        callback = unittest.mock.MagicMock()
        sender._on_send_completion = callback
        sender._send_executors = [ThreadPoolExecutor(max_workers=1)]
        sender._send_queues = [queue.Queue()]
        sender.stage_tensors("retry-fail-1", {"latents": torch.randn(1, 4, 8, 8)})
        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "retry-fail-1",
                "dest_session_id": "receiver",
                "dest_addr": 0x1234,
                "transfer_size": 4096,
            }
        )

        sender._execute_send = lambda request_id, peer_info: (False, "hard failure")

        while callback.call_count == 0:
            if sender._send_queues[0].qsize() > 0:
                sender._submit_send_task(sender._send_queues[0].get_nowait(), 0)
            sender._drain_send_completions()
            time.sleep(0.01)

        sender._send_executors[0].shutdown(wait=True)
        sender._send_executors = []

        callback.assert_called_once()
        self.assertFalse(callback.call_args[0][3])
        self.assertEqual(sender._terminal_send_states["retry-fail-1"], "failed")
        self.assertIsNone(sender.get_staged_info("retry-fail-1"))

    def test_same_host_local_copy_path_moves_data_and_meta(self):
        sender = _make_manager(self, session_id="sender-local")
        receiver = _make_manager(self, session_id="receiver-local")
        staged = sender.stage_tensors(
            "local-1",
            {"latents": torch.randn(1, 4, 8, 8)},
            scalar_fields={"request_id": "local-1"},
        )
        receiver.allocate_receive_slot("local-1", staged.slot.size, staged.meta_size)
        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "local-1",
                "dest_session_id": receiver.session_id,
                "dest_addr": receiver.get_receive_slot_addr("local-1"),
                "transfer_size": staged.slot.size,
                "meta_dest_addr": receiver.get_receive_meta_addr("local-1"),
                "meta_transfer_size": staged.meta_size,
                "receiver_host_id": "host-a",
                "receiver_supports_local_copy": True,
                "dest_shm_name": receiver.data_shm_name,
                "dest_shm_offset": receiver.get_receive_slot_offset("local-1"),
                "meta_dest_shm_name": receiver.meta_shm_name,
                "meta_dest_shm_offset": receiver.get_receive_meta_offset("local-1"),
            }
        )

        success, error_msg = sender._execute_send(
            "local-1", sender._pending_peer_sends["local-1"]
        )
        self.assertTrue(success)
        self.assertIsNone(error_msg)
        loaded, scalars, _ = receiver.load_transfer_async("local-1", device="cpu")
        self.assertIn("latents", loaded)
        self.assertEqual(scalars["request_id"], "local-1")

    def test_same_host_local_copy_fails_if_meta_copy_fails(self):
        sender = _make_manager(self, session_id="sender-local-fail")
        receiver = _make_manager(self, session_id="receiver-local-fail")
        staged = sender.stage_tensors(
            "local-fail-1",
            {"latents": torch.randn(1, 4, 8, 8)},
            scalar_fields={"request_id": "local-fail-1"},
        )
        receiver.allocate_receive_slot(
            "local-fail-1", staged.slot.size, staged.meta_size
        )
        sender._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "local-fail-1",
                "dest_session_id": receiver.session_id,
                "dest_addr": receiver.get_receive_slot_addr("local-fail-1"),
                "transfer_size": staged.slot.size,
                "meta_dest_addr": receiver.get_receive_meta_addr("local-fail-1"),
                "meta_transfer_size": staged.meta_size,
                "receiver_host_id": "host-a",
                "receiver_supports_local_copy": True,
                "dest_shm_name": receiver.data_shm_name,
                "dest_shm_offset": receiver.get_receive_slot_offset("local-fail-1"),
                "meta_dest_shm_name": receiver.meta_shm_name,
                "meta_dest_shm_offset": receiver.get_receive_meta_offset(
                    "local-fail-1"
                ),
            }
        )
        original_local_copy = sender._local_copy
        call_count = {"n": 0}

        def flaky_local_copy(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return False
            return original_local_copy(*args, **kwargs)

        sender._local_copy = flaky_local_copy

        success, error_msg = sender._execute_send(
            "local-fail-1", sender._pending_peer_sends["local-fail-1"]
        )
        self.assertFalse(success)
        self.assertIn("local shared-memory copy failed", error_msg)

    def test_missing_staged_payload_reports_failure(self):
        mgr = _make_manager(self, session_id="sender-missing")
        callback = unittest.mock.MagicMock()
        mgr._on_send_completion = callback
        mgr._send_executors = [ThreadPoolExecutor(max_workers=1)]
        mgr._send_queues = [queue.Queue()]
        mgr._register_peer_send(
            {
                "msg_type": "transfer_peer_info",
                "request_id": "missing-staged",
                "dest_session_id": "receiver",
                "dest_addr": 0,
                "transfer_size": 1024,
                "meta_transfer_size": 64,
            }
        )

        mgr._submit_send_task("missing-staged", 0)
        mgr._drain_send_completions()

        callback.assert_called_once()
        self.assertEqual(mgr._terminal_send_states["missing-staged"], "failed")
        mgr._send_executors[0].shutdown(wait=True)
        mgr._send_executors = []

    def test_abort_request_frees_staged_and_dynamic_receive_and_tombstones(self):
        mgr = _make_manager(self, session_id="sender-abort")
        staged = mgr.stage_tensors(
            "abort-1",
            {"latents": torch.randn(1, 4, 8, 8)},
            scalar_fields={"request_id": "abort-1"},
        )
        pending = mgr.allocate_receive_slot(
            "abort-1", staged.slot.size, staged.meta_size
        )
        self.assertIsNotNone(pending)

        mgr.abort_request("abort-1")

        self.assertTrue(mgr.is_request_aborted("abort-1"))
        self.assertIsNone(mgr.get_staged_info("abort-1"))
        self.assertIsNone(mgr.get_receive_slot_addr("abort-1"))
        self.assertEqual(mgr._terminal_send_states["abort-1"], "aborted")


class TestCleanup(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)

    def test_cleanup_waits_for_receive_thread_before_closing_control_socket(self):
        mgr = _make_manager(session_id="cleanup-order")
        events = []
        fake_thread = unittest.mock.Mock()
        fake_thread.join.side_effect = lambda timeout=None: events.append("join")
        fake_thread.is_alive.return_value = False
        fake_socket = unittest.mock.Mock()
        fake_socket.close.side_effect = lambda linger=0: events.append("close")
        mgr._running = True
        mgr._receive_thread = fake_thread
        mgr._control_pull = fake_socket

        mgr.cleanup()

        self.assertEqual(events[:2], ["join", "close"])
        fake_socket.close.assert_called_once_with(linger=0)

    def test_cleanup_stops_background_receive_loop_before_socket_close(self):
        mgr = _make_manager(session_id="cleanup-background")
        context = zmq.Context(io_threads=1)
        cleaned_up = False

        try:
            mgr.start_background_loops(
                context,
                _reserve_tcp_endpoint(),
                start_send_loop=False,
            )
            receive_thread = mgr._receive_thread
            self.assertIsNotNone(receive_thread)

            deadline = time.time() + 1.0
            while time.time() < deadline and not receive_thread.is_alive():
                time.sleep(0.01)
            self.assertTrue(receive_thread.is_alive())

            mgr.cleanup()
            cleaned_up = True

            self.assertFalse(receive_thread.is_alive())
            self.assertIsNone(mgr._control_pull)
        finally:
            if not cleaned_up:
                mgr.cleanup()
            context.destroy(linger=0)


class TestTransferProtocol(unittest.TestCase):
    def test_encode_decode_staged(self):
        msg = TransferStagedMsg(
            request_id="r1",
            data_size=1024,
            meta_size=256,
            session_id="session-1",
            pool_ptr=0x1000,
            slot_offset=0,
            meta_pool_ptr=0x2000,
            meta_slot_offset=128,
        )
        decoded = decode_transfer_msg(encode_transfer_msg(msg))
        self.assertEqual(decoded["msg_type"], "transfer_staged")
        self.assertEqual(decoded["request_id"], "r1")
        self.assertEqual(decoded["meta_size"], 256)

    def test_encode_decode_alloc(self):
        msg = TransferAllocMsg(
            request_id="r1",
            data_size=2048,
            meta_size=128,
            source_role="encoder",
            source_host_id="host-a",
        )
        decoded = decode_transfer_msg(encode_transfer_msg(msg))
        self.assertEqual(decoded["msg_type"], "transfer_alloc")
        self.assertEqual(decoded["source_role"], "encoder")
        self.assertEqual(decoded["meta_size"], 128)

    def test_encode_decode_peer_info(self):
        msg = TransferPeerInfoMsg(
            request_id="r1",
            dest_session_id="sess-2",
            dest_addr=0x2000,
            transfer_size=4096,
            meta_dest_addr=0x3000,
            meta_transfer_size=512,
            receiver_host_id="host-a",
            receiver_supports_local_copy=True,
        )
        decoded = decode_transfer_msg(encode_transfer_msg(msg))
        self.assertEqual(decoded["msg_type"], "transfer_peer_info")
        self.assertEqual(decoded["dest_addr"], 0x2000)
        self.assertEqual(decoded["meta_dest_addr"], 0x3000)

    def test_is_transfer_message(self):
        transfer_frames = encode_transfer_msg(TransferPushedMsg(request_id="r1"))
        self.assertTrue(is_transfer_message(transfer_frames))
        self.assertFalse(is_transfer_message([b'{"tensor_descriptors": []}', b"data"]))

    def test_encode_decode_alloc_reject(self):
        msg = TransferAllocRejectMsg(
            request_id="r1",
            receiver_role="denoiser",
            receiver_instance=1,
            retryable=True,
            reason="busy",
        )
        decoded = decode_transfer_msg(encode_transfer_msg(msg))
        self.assertEqual(decoded["msg_type"], "transfer_alloc_reject")
        self.assertTrue(decoded["retryable"])
        self.assertEqual(decoded["receiver_instance"], 1)

    def test_encode_decode_abort(self):
        msg = TransferAbortMsg(
            request_id="r1",
            reason="timeout",
            source="timeout",
        )
        decoded = decode_transfer_msg(encode_transfer_msg(msg))
        self.assertEqual(decoded["msg_type"], "transfer_abort")
        self.assertEqual(decoded["reason"], "timeout")


class TestSendRuntimeConfig(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)

    def test_default_runtime_config_uses_single_queue_and_fallback_workers(self):
        mgr = _make_manager(self)
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            queue_count, total_workers, worker_counts = (
                mgr._resolve_send_runtime_config(3)
            )
        self.assertEqual(queue_count, 1)
        self.assertEqual(total_workers, 3)
        self.assertEqual(worker_counts, [3])

    def test_queue_only_override_promotes_total_workers(self):
        mgr = _make_manager(self)
        with unittest.mock.patch.dict(
            os.environ,
            {"SGLANG_DIFFUSION_DISAGG_SEND_QUEUE_SIZE": "4"},
            clear=True,
        ):
            queue_count, total_workers, worker_counts = (
                mgr._resolve_send_runtime_config(1)
            )
        self.assertEqual(queue_count, 4)
        self.assertEqual(total_workers, 4)
        self.assertEqual(worker_counts, [1, 1, 1, 1])

    def test_explicit_thread_pool_override_is_evenly_distributed(self):
        mgr = _make_manager(self)
        with unittest.mock.patch.dict(
            os.environ,
            {
                "SGLANG_DIFFUSION_DISAGG_SEND_QUEUE_SIZE": "3",
                "SGLANG_DIFFUSION_DISAGG_SEND_THREAD_POOL_SIZE": "8",
            },
            clear=True,
        ):
            queue_count, total_workers, worker_counts = (
                mgr._resolve_send_runtime_config(1)
            )
        self.assertEqual(queue_count, 3)
        self.assertEqual(total_workers, 8)
        self.assertEqual(worker_counts, [3, 3, 2])

    def test_invalid_thread_pool_smaller_than_queue_count_fails(self):
        mgr = _make_manager(self)
        with unittest.mock.patch.dict(
            os.environ,
            {
                "SGLANG_DIFFUSION_DISAGG_SEND_QUEUE_SIZE": "3",
                "SGLANG_DIFFUSION_DISAGG_SEND_THREAD_POOL_SIZE": "2",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                mgr._resolve_send_runtime_config(1)

    def test_same_downstream_identity_maps_to_same_queue(self):
        mgr = _make_manager(self)
        mgr._send_queues = [queue.Queue() for _ in range(4)]
        first = PendingPeerSend(
            request_id="r1",
            receiver_instance=7,
            dest_session_id="sess-a",
        )
        second = PendingPeerSend(
            request_id="r2",
            receiver_instance=7,
            dest_session_id="sess-a",
        )
        self.assertEqual(
            mgr._select_send_queue_idx(first),
            mgr._select_send_queue_idx(second),
        )


# Consolidated from test_transfer_engine.py.
class TestMockTransferEngine(unittest.TestCase):
    """Test MockTransferEngine simulated RDMA transfers."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_session_id_unique(self):
        e1 = MockTransferEngine()
        e2 = MockTransferEngine()
        self.assertNotEqual(e1.session_id, e2.session_id)

    def test_custom_session_id(self):
        e = MockTransferEngine(session_id="test-123")
        self.assertEqual(e.session_id, "test-123")

    def test_register_and_deregister(self):
        e = MockTransferEngine()
        # Should not raise
        e.register_buffer(0x1000, 4096)
        e.deregister_buffer(0x1000)
        # Double deregister should be safe
        e.deregister_buffer(0x1000)

    def test_transfer_sync_copies_data(self):
        """Verify that transfer_sync copies data between memory regions."""
        e1 = MockTransferEngine(session_id="src")
        e2 = MockTransferEngine(session_id="dst")

        # Allocate source and destination buffers
        src_buf = (ctypes.c_byte * 1024)()
        dst_buf = (ctypes.c_byte * 1024)()

        # Fill source with pattern
        for i in range(1024):
            src_buf[i] = i % 128

        src_addr = ctypes.addressof(src_buf)
        dst_addr = ctypes.addressof(dst_buf)

        e1.register_buffer(src_addr, 1024)
        e2.register_buffer(dst_addr, 1024)

        # Transfer from src to dst
        ret = e1.transfer_sync("dst", src_addr, dst_addr, 1024)
        self.assertEqual(ret, 0)

        # Verify data was copied
        for i in range(1024):
            self.assertEqual(dst_buf[i], i % 128)

    def test_batch_transfer_sync(self):
        """Verify batch transfer copies multiple regions."""
        e1 = MockTransferEngine(session_id="src")
        e2 = MockTransferEngine(session_id="dst")

        buf1_src = (ctypes.c_byte * 256)()
        buf2_src = (ctypes.c_byte * 256)()
        buf1_dst = (ctypes.c_byte * 256)()
        buf2_dst = (ctypes.c_byte * 256)()

        for i in range(256):
            buf1_src[i] = 42
            buf2_src[i] = 99

        ret = e1.batch_transfer_sync(
            "dst",
            [ctypes.addressof(buf1_src), ctypes.addressof(buf2_src)],
            [ctypes.addressof(buf1_dst), ctypes.addressof(buf2_dst)],
            [256, 256],
        )
        self.assertEqual(ret, 0)
        self.assertEqual(buf1_dst[0], 42)
        self.assertEqual(buf2_dst[0], 99)

    def test_is_base_transfer_engine(self):
        e = MockTransferEngine()
        self.assertIsInstance(e, BaseTransferEngine)

    def test_reset_clears_state(self):
        MockTransferEngine(session_id="a")
        MockTransferEngine(session_id="b")
        self.assertEqual(len(MockTransferEngine._registry), 2)
        MockTransferEngine.reset()
        self.assertEqual(len(MockTransferEngine._registry), 0)


class TestCreateTransferEngine(unittest.TestCase):
    """Test factory function."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_force_mock(self):
        engine = create_transfer_engine(force_mock=True)
        self.assertIsInstance(engine, MockTransferEngine)

    def test_returns_base_interface(self):
        engine = create_transfer_engine(force_mock=True)
        self.assertIsInstance(engine, BaseTransferEngine)

    def test_auto_backend_uses_mock_for_loopback(self):
        self.assertEqual(
            resolve_transfer_backend("auto", hostname="127.0.0.1"),
            "mock",
        )
        self.assertEqual(
            resolve_transfer_backend("auto", hostname="localhost"),
            "mock",
        )

    def test_auto_backend_prefers_mooncake_for_rdma_or_non_loopback(self):
        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.transport.engine._check_mooncake",
            return_value=True,
        ):
            self.assertEqual(
                resolve_transfer_backend("auto", hostname="10.0.0.5"),
                "mooncake",
            )
            self.assertEqual(
                resolve_transfer_backend(
                    "auto", hostname="127.0.0.1", ib_device="mlx5_0"
                ),
                "mooncake",
            )

    def test_auto_backend_non_loopback_fails_fast_without_mooncake(self):
        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.transport.engine._check_mooncake",
            return_value=False,
        ):
            self.assertEqual(
                resolve_transfer_backend("auto", hostname="10.0.0.5"),
                "mooncake",
            )
            with self.assertRaisesRegex(RuntimeError, "Mooncake transfer backend"):
                create_transfer_engine(backend="auto", hostname="10.0.0.5")

    def test_explicit_mooncake_backend_requires_availability(self):
        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.transport.engine._check_mooncake",
            return_value=False,
        ), self.assertRaisesRegex(RuntimeError, "Mooncake transfer backend"):
            create_transfer_engine(backend="mooncake")


if __name__ == "__main__":
    unittest.main()
