# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Scheduler transfer integration (Phase 7b).

Tests the transfer message handling in the scheduler's pool mode event loop:
- Transfer frame detection
- transfer_alloc handling (allocate receive slot)
- transfer_push handling (RDMA push to peer)
- transfer_ready handling (load tensors, run compute)
- Encoder transfer staging
- Full encoder→denoiser transfer flow
"""

import json
import unittest
from unittest.mock import MagicMock

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    MockTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TRANSFER_MAGIC,
    TransferAllocMsg,
    TransferMsgType,
    TransferPushMsg,
    encode_transfer_msg,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler


class TestSchedulerTransferFrameDetection(unittest.TestCase):
    """Test the static _is_transfer_frames method."""

    def test_transfer_frames_detected(self):
        frames = encode_transfer_msg(TransferAllocMsg(request_id="r1", data_size=1024))
        self.assertTrue(Scheduler._is_transfer_frames(frames))

    def test_non_transfer_frames_not_detected(self):
        frames = [b"some_data", b"more_data"]
        self.assertFalse(Scheduler._is_transfer_frames(frames))

    def test_empty_frames_not_detected(self):
        self.assertFalse(Scheduler._is_transfer_frames([]))

    def test_single_frame_not_detected(self):
        self.assertFalse(Scheduler._is_transfer_frames([b"data"]))


class TestSchedulerTransferAlloc(unittest.TestCase):
    """Test _handle_transfer_alloc."""

    def setUp(self):
        MockTransferEngine.reset()
        self.engine = MockTransferEngine()
        self.buffer = TransferTensorBuffer(pool_size=1 * 1024 * 1024)
        self.tm = DiffusionTransferManager(engine=self.engine, buffer=self.buffer)

        # Mock scheduler
        self.scheduler = object.__new__(Scheduler)
        self.scheduler._transfer_manager = self.tm
        self.scheduler._disagg_role = MagicMock()
        self.scheduler._disagg_role.value = "denoising"
        self.scheduler._disagg_metrics = None

        # Mock result push socket
        self.scheduler._pool_result_push = MagicMock()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_alloc_sends_allocated_response(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-001",
            "data_size": 4096,
            "source_role": "encoder",
        }
        self.scheduler._handle_transfer_alloc(msg)

        # Verify transfer_allocated was sent
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        self.assertEqual(sent_frames[0], TRANSFER_MAGIC)

        reply = json.loads(sent_frames[1])
        self.assertEqual(reply["msg_type"], TransferMsgType.ALLOCATED)
        self.assertEqual(reply["request_id"], "req-001")
        self.assertEqual(reply["session_id"], self.engine.session_id)
        self.assertGreater(reply["slot_size"], 0)

    def test_alloc_creates_receive_slot(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-002",
            "data_size": 8192,
        }
        self.scheduler._handle_transfer_alloc(msg)

        # Verify slot was allocated in transfer manager
        addr = self.tm.get_receive_slot_addr("req-002")
        self.assertIsNotNone(addr)


class TestSchedulerTransferPush(unittest.TestCase):
    """Test _handle_transfer_push."""

    def setUp(self):
        MockTransferEngine.reset()
        self.engine = MockTransferEngine()
        self.buffer = TransferTensorBuffer(pool_size=1 * 1024 * 1024)
        self.tm = DiffusionTransferManager(engine=self.engine, buffer=self.buffer)

        self.scheduler = object.__new__(Scheduler)
        self.scheduler._transfer_manager = self.tm
        self.scheduler._disagg_role = MagicMock()
        self.scheduler._disagg_role.value = "encoder"
        self.scheduler._disagg_metrics = None
        self.scheduler._pool_result_push = MagicMock()
        self.scheduler._rdma_push_queue = None

    def tearDown(self):
        MockTransferEngine.reset()

    def test_push_sends_pushed_response(self):
        # Stage some data first
        tensor = torch.randn(4, 4)
        staged = self.tm.stage_tensors("req-push-1", {"t": tensor})
        self.assertIsNotNone(staged)

        # Create a destination buffer to push to
        dest_engine = MockTransferEngine()
        dest_buffer = TransferTensorBuffer(pool_size=1 * 1024 * 1024)
        dest_tm = DiffusionTransferManager(engine=dest_engine, buffer=dest_buffer)
        dest_pending = dest_tm.allocate_receive_slot("req-push-1", staged.slot.size)

        msg = {
            "msg_type": TransferMsgType.PUSH,
            "request_id": "req-push-1",
            "dest_session_id": dest_engine.session_id,
            "dest_addr": dest_tm.pool_data_ptr + dest_pending.slot.offset,
            "transfer_size": staged.slot.size,
        }

        self.scheduler._handle_transfer_push(msg)

        # Verify transfer_pushed was sent
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        reply = json.loads(sent_frames[1])
        self.assertEqual(reply["msg_type"], TransferMsgType.PUSHED)
        self.assertEqual(reply["request_id"], "req-push-1")

    def test_push_frees_staged_slot(self):
        tensor = torch.randn(4, 4)
        staged = self.tm.stage_tensors("req-push-2", {"t": tensor})
        self.assertIsNotNone(staged)

        msg = {
            "msg_type": TransferMsgType.PUSH,
            "request_id": "req-push-2",
            "dest_session_id": "mock-dest",
            "dest_addr": self.buffer.pool_data_ptr,
            "transfer_size": staged.slot.size,
        }
        self.scheduler._handle_transfer_push(msg)

        # Staged slot should be freed
        info = self.tm.get_staged_info("req-push-2")
        self.assertIsNone(info)


class TestSchedulerTransferReady(unittest.TestCase):
    """Test _handle_transfer_ready with mock compute."""

    def setUp(self):
        MockTransferEngine.reset()
        self.engine = MockTransferEngine()
        self.buffer = TransferTensorBuffer(pool_size=2 * 1024 * 1024)
        self.tm = DiffusionTransferManager(engine=self.engine, buffer=self.buffer)

        self.scheduler = object.__new__(Scheduler)
        self.scheduler._transfer_manager = self.tm
        self.scheduler._disagg_metrics = None
        self.scheduler._pool_result_push = MagicMock()
        self.scheduler.gpu_id = 0

    def tearDown(self):
        MockTransferEngine.reset()

    def test_build_disagg_req(self):
        """Test that _build_disagg_req correctly reconstructs a Req."""
        scalar_fields = {
            "request_id": "req-ready-1",
            "prompt": "test prompt",
            "num_inference_steps": 20,
        }
        tensors = {
            "prompt_embeds": torch.randn(1, 16, 64),
        }

        req = self.scheduler._build_disagg_req(scalar_fields, tensors)

        self.assertEqual(req.request_id, "req-ready-1")
        self.assertEqual(req.prompt, "test prompt")
        self.assertEqual(req.num_inference_steps, 20)
        self.assertTrue(torch.equal(req.prompt_embeds, tensors["prompt_embeds"]))


class TestSchedulerTransferEncoderStaging(unittest.TestCase):
    """Test encoder transfer staging (_disagg_encoder_transfer_stage)."""

    def setUp(self):
        MockTransferEngine.reset()
        self.engine = MockTransferEngine()
        self.buffer = TransferTensorBuffer(pool_size=2 * 1024 * 1024)
        self.tm = DiffusionTransferManager(engine=self.engine, buffer=self.buffer)

        self.scheduler = object.__new__(Scheduler)
        self.scheduler._transfer_manager = self.tm
        self.scheduler._disagg_role = MagicMock()
        self.scheduler._disagg_role.value = "encoder"
        self.scheduler._disagg_metrics = None
        self.scheduler._pool_result_push = MagicMock()
        self.scheduler._transfer_stream = None

    def tearDown(self):
        MockTransferEngine.reset()

    def test_encoder_transfer_stage_sends_staged_msg(self):
        tensor_fields = {
            "prompt_embeds": torch.randn(1, 8, 32),
            "latents": torch.randn(1, 4, 16, 16),
        }
        scalar_fields = {
            "request_id": "req-enc-1",
            "guidance_scale": 7.5,
        }

        self.scheduler._disagg_encoder_transfer_stage(
            "req-enc-1", tensor_fields, scalar_fields
        )

        # Verify transfer_staged was sent
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        self.assertEqual(sent_frames[0], TRANSFER_MAGIC)

        staged_msg = json.loads(sent_frames[1])
        self.assertEqual(staged_msg["msg_type"], "transfer_staged")
        self.assertEqual(staged_msg["request_id"], "req-enc-1")
        self.assertEqual(staged_msg["session_id"], self.engine.session_id)
        self.assertGreater(staged_msg["data_size"], 0)
        self.assertIn("prompt_embeds", staged_msg["manifest"])
        self.assertIn("latents", staged_msg["manifest"])
        self.assertEqual(staged_msg["scalar_fields"]["request_id"], "req-enc-1")
        self.assertEqual(staged_msg["scalar_fields"]["guidance_scale"], 7.5)

    def test_encoder_transfer_stage_data_in_buffer(self):
        """Verify staged data is actually in the buffer."""
        tensor = torch.randn(1, 4, 8, 8)
        self.scheduler._disagg_encoder_transfer_stage(
            "req-enc-2", {"latents": tensor}, {"request_id": "req-enc-2"}
        )

        # Verify data is staged in TransferManager
        info = self.tm.get_staged_info("req-enc-2")
        self.assertIsNotNone(info)
        self.assertIn("latents", info.manifest)


class TestSchedulerTransferMessageDispatch(unittest.TestCase):
    """Test _handle_transfer_msg dispatch."""

    def setUp(self):
        MockTransferEngine.reset()
        self.engine = MockTransferEngine()
        self.buffer = TransferTensorBuffer(pool_size=1 * 1024 * 1024)
        self.tm = DiffusionTransferManager(engine=self.engine, buffer=self.buffer)

        self.scheduler = object.__new__(Scheduler)
        self.scheduler._transfer_manager = self.tm
        self.scheduler._disagg_role = MagicMock()
        self.scheduler._disagg_role.value = "denoising"
        self.scheduler._disagg_metrics = None
        self.scheduler._pool_result_push = MagicMock()
        self.scheduler._rdma_push_queue = None
        self.scheduler.gpu_id = 0

    def tearDown(self):
        MockTransferEngine.reset()

    def test_dispatch_alloc_message(self):
        """Verify transfer_alloc is routed to _handle_transfer_alloc."""
        alloc_msg = TransferAllocMsg(request_id="req-d1", data_size=2048)
        frames = encode_transfer_msg(alloc_msg)

        self.scheduler._handle_transfer_msg(frames)

        # Should have sent transfer_allocated
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent = json.loads(
            self.scheduler._pool_result_push.send_multipart.call_args[0][0][1]
        )
        self.assertEqual(sent["msg_type"], TransferMsgType.ALLOCATED)

    def test_dispatch_push_message(self):
        """Verify transfer_push is routed to _handle_transfer_push."""
        # Stage data first
        tensor = torch.randn(2, 2)
        staged = self.tm.stage_tensors("req-d2", {"t": tensor})

        push_msg = TransferPushMsg(
            request_id="req-d2",
            dest_session_id="mock",
            dest_addr=self.buffer.pool_data_ptr,
            transfer_size=staged.slot.size,
        )
        frames = encode_transfer_msg(push_msg)
        self.scheduler._handle_transfer_msg(frames)

        sent = json.loads(
            self.scheduler._pool_result_push.send_multipart.call_args[0][0][1]
        )
        self.assertEqual(sent["msg_type"], TransferMsgType.PUSHED)


class TestSchedulerTransferEndToEnd(unittest.TestCase):
    """Test a full transfer cycle: encoder stages → denoiser receives."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_encoder_to_denoiser_transfer_data(self):
        """Simulate the encoder staging data, then a denoiser allocating
        and receiving it via RDMA (mock), verifying data integrity."""

        # --- Encoder side ---
        enc_engine = MockTransferEngine()
        enc_buffer = TransferTensorBuffer(pool_size=2 * 1024 * 1024)
        enc_tm = DiffusionTransferManager(engine=enc_engine, buffer=enc_buffer)

        # Stage encoder output
        original_tensor = torch.randn(1, 4, 16, 16)
        staged = enc_tm.stage_tensors(
            "e2e-req-1",
            {"latents": original_tensor},
            {"request_id": "e2e-req-1", "guidance_scale": 7.5},
        )
        self.assertIsNotNone(staged)

        # --- Denoiser side ---
        den_engine = MockTransferEngine()
        den_buffer = TransferTensorBuffer(pool_size=2 * 1024 * 1024)
        den_tm = DiffusionTransferManager(engine=den_engine, buffer=den_buffer)

        # Allocate receive slot
        pending = den_tm.allocate_receive_slot("e2e-req-1", staged.slot.size)
        self.assertIsNotNone(pending)

        # --- Encoder RDMA push ---
        dest_addr = den_tm.pool_data_ptr + pending.slot.offset
        success = enc_tm.push_to_peer(
            "e2e-req-1",
            dest_session_id=den_engine.session_id,
            dest_addr=dest_addr,
            transfer_size=staged.slot.size,
        )
        self.assertTrue(success)

        # --- Denoiser loads tensors ---
        tensors = den_tm.load_tensors("e2e-req-1", staged.manifest, device="cpu")
        self.assertIn("latents", tensors)
        self.assertTrue(torch.allclose(tensors["latents"], original_tensor, atol=1e-6))

        # Cleanup
        enc_tm.free_staged("e2e-req-1")
        den_tm.free_receive_slot("e2e-req-1")


class TestSchedulerTransferInit(unittest.TestCase):
    """Test _init_disagg_transfer_manager."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_init_creates_transfer_manager(self):
        """Verify _init_disagg_transfer_manager creates a working manager."""
        scheduler = object.__new__(Scheduler)
        scheduler._transfer_manager = None
        scheduler._pool_result_push = MagicMock()
        scheduler.gpu_id = 0

        # Mock server_args — use power-of-2 size to avoid BuddyAllocator rounding
        scheduler.server_args = MagicMock()
        scheduler.server_args.disagg_transfer_pool_size = 1 * 1024 * 1024
        scheduler.server_args.disagg_p2p_hostname = "127.0.0.1"
        scheduler.server_args.disagg_ib_device = None

        # Mock disagg role
        scheduler._disagg_role = MagicMock()
        scheduler._disagg_role.value = "encoder"

        scheduler._init_disagg_transfer_manager()

        self.assertIsNotNone(scheduler._transfer_manager)
        self.assertEqual(scheduler._transfer_manager.pool_size, 1 * 1024 * 1024)

    def test_init_sends_register_message(self):
        """Verify _init_disagg_transfer_manager sends transfer_register to DS."""
        scheduler = object.__new__(Scheduler)
        scheduler._transfer_manager = None
        scheduler._pool_result_push = MagicMock()
        scheduler.gpu_id = 0

        scheduler.server_args = MagicMock()
        scheduler.server_args.disagg_transfer_pool_size = 256 * 1024
        scheduler.server_args.disagg_p2p_hostname = "127.0.0.1"
        scheduler.server_args.disagg_ib_device = None
        scheduler._disagg_role = MagicMock()
        scheduler._disagg_role.value = "denoising"

        scheduler._init_disagg_transfer_manager()

        # Should have sent register message
        scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = scheduler._pool_result_push.send_multipart.call_args[0][0]
        self.assertEqual(sent_frames[0], TRANSFER_MAGIC)

        reg = json.loads(sent_frames[1])
        self.assertEqual(reg["msg_type"], TransferMsgType.REGISTER)
        self.assertEqual(reg["role"], "denoising")
        self.assertGreater(reg["pool_size"], 0)


class TestMultiRankGating(unittest.TestCase):
    """Test Phase 7c multi-rank gating: non-rank-0 skips ZMQ sends."""

    def _make_scheduler(self, gpu_id=0, role="encoder"):
        """Create a minimal mock Scheduler for multi-rank tests."""
        scheduler = object.__new__(Scheduler)
        scheduler.gpu_id = gpu_id
        scheduler._disagg_role = MagicMock()
        scheduler._disagg_role.value = role
        scheduler._disagg_metrics = None
        # Transfer is always on
        scheduler._transfer_manager = None
        scheduler._pool_result_push = MagicMock() if gpu_id == 0 else None
        scheduler._pool_work_pull = MagicMock() if gpu_id == 0 else None
        scheduler.worker = MagicMock()
        scheduler.worker.local_rank = gpu_id
        scheduler.server_args = MagicMock()
        scheduler.server_args.disagg_transfer_pool_size = 1 * 1024 * 1024
        scheduler.server_args.disagg_p2p_hostname = "127.0.0.1"
        return scheduler

    def test_init_sockets_skipped_non_rank0(self):
        """Non-rank-0 should not create ZMQ sockets."""
        scheduler = self._make_scheduler(gpu_id=1)
        scheduler._disagg_mode = True
        scheduler.server_args.disagg_pool_work_port = 5000
        scheduler.server_args.disagg_pool_result_port = 5001
        scheduler.server_args.disagg_ds_host = "127.0.0.1"

        scheduler._init_disagg_sockets()

        # Sockets should remain None
        self.assertIsNone(scheduler._pool_work_pull)
        self.assertIsNone(scheduler._pool_result_push)

    def test_init_transfer_manager_skipped_non_rank0(self):
        """Non-rank-0 should not create TransferManager."""
        scheduler = self._make_scheduler(gpu_id=2)

        scheduler._init_disagg_transfer_manager()

        self.assertIsNone(scheduler._transfer_manager)

    def _make_encoder_frames(self):
        """Create pickled frames for encoder step tests."""
        import pickle

        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        req = object.__new__(Req)
        req.__dict__.update(
            {
                "request_id": "test-1",
                "guidance_scale": 7.5,
                "num_inference_steps": 20,
            }
        )
        return [b"req_id", pickle.dumps([req])], req

    def test_encoder_step_no_send_on_non_rank0(self):
        """Non-rank-0 encoder should not crash on send (push is None)."""
        scheduler = self._make_scheduler(gpu_id=1, role="encoder")
        frames, req = self._make_encoder_frames()
        scheduler.worker.execute_forward.return_value = req

        send_fn = MagicMock()

        scheduler._disagg_encoder_step(
            send_fn,
            frames=frames,
        )

        # send_fn should NOT be called (push is None)
        send_fn.assert_not_called()

    def test_handle_transfer_non_rank0_skips_alloc(self):
        """Non-rank-0 should skip transfer_alloc messages (no-op)."""
        scheduler = self._make_scheduler(gpu_id=1, role="denoising")

        alloc_frames = encode_transfer_msg(
            TransferAllocMsg(request_id="r1", data_size=1024)
        )
        # Should not raise
        scheduler._handle_transfer_non_rank0(alloc_frames)

    def test_handle_transfer_non_rank0_ready_calls_execute(self):
        """Non-rank-0 should call execute_forward on transfer_ready."""
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        scheduler = self._make_scheduler(gpu_id=1, role="denoising")
        scheduler._disagg_role = RoleType.DENOISER
        scheduler.worker.pipeline.get_module.return_value = MagicMock()

        ready_msg = {
            "msg_type": TransferMsgType.READY,
            "request_id": "r1",
            "scalar_fields": {
                "request_id": "r1",
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
            },
        }
        ready_frames = [
            TRANSFER_MAGIC,
            json.dumps(ready_msg, separators=(",", ":")).encode("utf-8"),
        ]

        scheduler._handle_transfer_non_rank0(ready_frames)

        scheduler.worker.execute_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
