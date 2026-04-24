# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Scheduler transfer integration."""

import pickle
import queue
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    _PendingInboundTransfer,
    estimate_transfer_bytes,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferMetaBuffer,
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    MockTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocMsg,
    TransferMsgType,
    decode_transfer_msg,
    encode_transfer_msg,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage


class TestSchedulerTransferFrameDetection(unittest.TestCase):
    def test_transfer_frames_detected(self):
        frames = encode_transfer_msg(
            TransferAllocMsg(request_id="r1", data_size=1024, meta_size=128)
        )
        self.assertTrue(Scheduler._is_transfer_frames(frames))

    def test_non_transfer_frames_not_detected(self):
        self.assertFalse(Scheduler._is_transfer_frames([b"some_data", b"more_data"]))


class _SchedulerHarness:
    @staticmethod
    def make(role: RoleType) -> Scheduler:
        scheduler = object.__new__(Scheduler)
        scheduler._disagg_role = role
        scheduler._disagg_metrics = None
        scheduler._pool_result_push = MagicMock()
        scheduler._preallocated_slots = {}
        scheduler._control_queue = queue.Queue()
        scheduler._transferring_queue = queue.Queue()
        scheduler._prefetch_queue = scheduler._transferring_queue
        scheduler._swapping_queue = queue.Queue()
        scheduler._compute_ready_queue = queue.Queue()
        scheduler._swap_out_queue = deque()
        scheduler._send_ready_queue = deque()
        scheduler._outbound_staging_retry_queue = deque()
        scheduler._swap_in_stream = None
        scheduler._compute_stream = None
        scheduler._swap_out_stream = None
        scheduler._pending_transfer_reconfigure = None
        scheduler._transfer_reconfigured = False
        scheduler._warmup_inbound_sizes = {}
        scheduler._aborted_request_ids = {}
        scheduler._disagg_timeout_s = 600.0
        scheduler._running = True
        scheduler.context = MagicMock()
        scheduler.gpu_id = 0
        scheduler.worker = SimpleNamespace(
            local_rank=0,
            pipeline=SimpleNamespace(get_module=lambda name: None),
        )
        scheduler.server_args = SimpleNamespace(
            disagg_instance_id=1,
            pool_control_advertised_endpoint="tcp://receiver-ctrl",
            pool_control_endpoint="tcp://receiver-ctrl",
            pool_work_endpoint=None,
            disagg_p2p_hostname="127.0.0.1",
            disagg_ib_device=None,
            disagg_transfer_backend="auto",
            disagg_transfer_pin_memory="auto",
            sp_degree=1,
            tp_size=1,
            enable_cfg_parallel=False,
            resolved_role_device=lambda: "cpu",
        )
        return scheduler


class TestTransferEngineGpuSelection(unittest.TestCase):
    def test_rank0_transfer_engine_uses_physical_gpu_id(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler.worker.local_rank = 4
        scheduler.server_args.pool_control_advertised_endpoint = None
        scheduler.server_args.pool_control_endpoint = None
        scheduler.server_args.resolved_role_device = lambda: "cuda"

        fake_engine = MagicMock()
        fake_engine.session_id = "session-1"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-1"
        fake_manager.pool_data_ptr = 123
        fake_manager.pool_size = 4096
        fake_manager.meta_pool_ptr = 456
        fake_manager.meta_pool_size = 1024
        fake_manager.data_shm_name = None
        fake_manager.meta_shm_name = None
        fake_manager.host_id = "host-a"

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ) as create_engine, patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
            return_value=MagicMock(),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer",
            return_value=MagicMock(),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager()

        create_engine.assert_called_once_with(
            hostname="127.0.0.1",
            gpu_id=4,
            ib_device=None,
            backend="mock",
            force_mock=True,
        )
        scheduler._pool_result_push.send_multipart.assert_called_once()

    def test_cuda_role_enables_transfer_pin_memory_by_default(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.pool_control_advertised_endpoint = None
        scheduler.server_args.pool_control_endpoint = None
        scheduler.server_args.resolved_role_device = lambda: "cuda"

        fake_engine = MagicMock()
        fake_engine.session_id = "session-pin"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-pin"
        fake_manager.pool_data_ptr = 123
        fake_manager.pool_size = 4096
        fake_manager.meta_pool_ptr = 456
        fake_manager.meta_pool_size = 1024
        fake_manager.data_shm_name = "data-shm"
        fake_manager.meta_shm_name = "meta-shm"
        fake_manager.host_id = "host-pin"

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
            return_value=MagicMock(),
        ) as tensor_buffer_cls, patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer",
            return_value=MagicMock(),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager()

        kwargs = tensor_buffer_cls.call_args.kwargs
        self.assertTrue(kwargs["pin_memory"])
        self.assertFalse(kwargs["pin_memory_strict"])

    def test_required_transfer_pin_memory_enables_strict_mode(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.pool_control_advertised_endpoint = None
        scheduler.server_args.pool_control_endpoint = None
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler.server_args.disagg_transfer_pin_memory = "required"

        fake_engine = MagicMock()
        fake_engine.session_id = "session-pin-required"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-pin-required"
        fake_manager.pool_data_ptr = 123
        fake_manager.pool_size = 4096
        fake_manager.meta_pool_ptr = 456
        fake_manager.meta_pool_size = 1024
        fake_manager.data_shm_name = "data-shm"
        fake_manager.meta_shm_name = "meta-shm"
        fake_manager.host_id = "host-pin"

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
            return_value=MagicMock(),
        ) as tensor_buffer_cls, patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer",
            return_value=MagicMock(),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager()

        kwargs = tensor_buffer_cls.call_args.kwargs
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["pin_memory_strict"])

    def test_cpu_role_does_not_pin_transfer_memory(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler.server_args.pool_control_advertised_endpoint = None
        scheduler.server_args.pool_control_endpoint = None
        scheduler.server_args.disagg_transfer_pin_memory = "required"
        scheduler.server_args.resolved_role_device = lambda: "cpu"

        fake_engine = MagicMock()
        fake_engine.session_id = "session-cpu"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-cpu"
        fake_manager.pool_data_ptr = 123
        fake_manager.pool_size = 4096
        fake_manager.meta_pool_ptr = 456
        fake_manager.meta_pool_size = 1024
        fake_manager.data_shm_name = "data-shm"
        fake_manager.meta_shm_name = "meta-shm"
        fake_manager.host_id = "host-cpu"

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
            return_value=MagicMock(),
        ) as tensor_buffer_cls, patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer",
            return_value=MagicMock(),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager()

        kwargs = tensor_buffer_cls.call_args.kwargs
        self.assertFalse(kwargs["pin_memory"])
        self.assertFalse(kwargs["pin_memory_strict"])

    def test_explicit_mooncake_backend_is_forwarded_without_force_mock(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler.server_args.disagg_transfer_backend = "mooncake"
        scheduler.server_args.disagg_p2p_hostname = "10.0.0.5"
        scheduler.server_args.disagg_ib_device = "mlx5_0"
        scheduler.server_args.pool_control_advertised_endpoint = None
        scheduler.server_args.pool_control_endpoint = None

        fake_engine = MagicMock()
        fake_engine.session_id = "session-2"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-2"
        fake_manager.pool_data_ptr = 123
        fake_manager.pool_size = 4096
        fake_manager.meta_pool_ptr = 456
        fake_manager.meta_pool_size = 1024
        fake_manager.data_shm_name = None
        fake_manager.meta_shm_name = None
        fake_manager.host_id = "host-b"

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ) as create_engine, patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
            return_value=MagicMock(),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer",
            return_value=MagicMock(),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager()

        create_engine.assert_called_once_with(
            hostname="10.0.0.5",
            gpu_id=0,
            ib_device="mlx5_0",
            backend="mooncake",
            force_mock=False,
        )


class TestTransferManagerPreallocation(unittest.TestCase):
    def test_inbound_startup_register_defers_preallocation_until_calibration(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.disagg_max_slots_per_instance = 4

        fake_engine = MagicMock()
        fake_engine.session_id = "session-startup"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-startup"
        fake_manager.pool_data_ptr = 123
        fake_manager.pool_size = 4096
        fake_manager.meta_pool_ptr = 456
        fake_manager.meta_pool_size = 1024
        fake_manager.data_shm_name = "data-shm"
        fake_manager.meta_shm_name = "meta-shm"
        fake_manager.host_id = "host-a"
        fake_buffer = MagicMock()
        fake_meta_buffer = MagicMock()

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
            return_value=fake_buffer,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer",
            return_value=fake_meta_buffer,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager()

        fake_buffer.allocate.assert_not_called()
        fake_meta_buffer.allocate.assert_not_called()
        sent_frames = scheduler._pool_result_push.send_multipart.call_args[0][0]
        register_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(register_msg["preallocated_slots"], [])
        self.assertEqual(scheduler._preallocated_slots, {})

    def test_inbound_calibrated_rebuild_keeps_receive_slots_dynamic(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.disagg_max_slots_per_instance = 3

        fake_engine = MagicMock()
        fake_engine.session_id = "session-calibrated"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-calibrated"
        fake_manager.pool_data_ptr = 1024
        fake_manager.pool_size = 16384
        fake_manager.meta_pool_ptr = 2048
        fake_manager.meta_pool_size = 4096
        fake_manager.data_shm_name = "data-shm"
        fake_manager.meta_shm_name = "meta-shm"
        fake_manager.host_id = "host-b"
        fake_buffer = MagicMock()
        fake_buffer.allocate.side_effect = [
            SimpleNamespace(offset=0, size=8192),
            SimpleNamespace(offset=8192, size=8192),
            SimpleNamespace(offset=16384, size=8192),
        ]
        fake_meta_buffer = MagicMock()
        fake_meta_buffer.allocate.side_effect = [
            SimpleNamespace(offset=0, size=1024),
            SimpleNamespace(offset=1024, size=1024),
            SimpleNamespace(offset=2048, size=1024),
        ]

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
            return_value=fake_buffer,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer",
            return_value=fake_meta_buffer,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager(
                measured_transfer_bytes=8192,
                measured_meta_bytes=1024,
            )

        fake_buffer.allocate.assert_not_called()
        fake_meta_buffer.allocate.assert_not_called()
        self.assertEqual(scheduler._preallocated_slots, {})
        sent_frames = scheduler._pool_result_push.send_multipart.call_args[0][0]
        register_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(register_msg["preallocated_slots"], [])


class _TrackedStreamContext:
    def __init__(self, state, stream):
        self._state = state
        self._stream = stream

    def __enter__(self):
        self._state["active"] = True
        self._state["stream"] = self._stream
        return self

    def __exit__(self, exc_type, exc, tb):
        self._state["active"] = False
        return False


class TestSchedulerTransferAlloc(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)
        self.engine = MockTransferEngine(session_id="receiver-session")
        self.buffer = TransferTensorBuffer(pool_size=1 * 1024 * 1024, role_name="test")
        self.meta_buffer = TransferMetaBuffer(
            slot_count=2, slot_size=64 * 1024, role_name="test"
        )
        self.tm = DiffusionTransferManager(
            engine=self.engine,
            buffer=self.buffer,
            meta_buffer=self.meta_buffer,
            host_id="host-a",
        )
        self.addCleanup(self.tm.cleanup)
        self.scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        self.scheduler._transfer_manager = self.tm
        self.tm.send_direct_message = MagicMock()

    def test_alloc_sends_peer_info_to_upstream_with_meta_and_local_copy(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-alloc-1",
            "data_size": 4096,
            "meta_size": 2048,
            "source_control_endpoint": "tcp://upstream-ctrl",
            "source_host_id": "host-a",
        }

        self.scheduler._handle_transfer_alloc(msg)

        self.assertIsNotNone(self.tm.get_receive_slot_addr("req-alloc-1"))
        self.assertIsNotNone(self.tm.get_receive_meta_addr("req-alloc-1"))
        self.tm.send_direct_message.assert_called_once()
        endpoint, peer_msg = self.tm.send_direct_message.call_args[0]
        self.assertEqual(endpoint, "tcp://upstream-ctrl")
        self.assertEqual(peer_msg.msg_type, TransferMsgType.PEER_INFO)
        self.assertEqual(peer_msg.request_id, "req-alloc-1")
        self.assertEqual(peer_msg.dest_session_id, self.engine.session_id)
        self.assertEqual(peer_msg.receiver_control_endpoint, "tcp://receiver-ctrl")
        self.assertEqual(peer_msg.meta_transfer_size, 2048)
        self.assertEqual(peer_msg.receiver_host_id, "host-a")
        self.assertTrue(peer_msg.receiver_supports_local_copy)
        self.assertEqual(peer_msg.dest_shm_name, self.tm.data_shm_name)
        self.assertEqual(peer_msg.meta_dest_shm_name, self.tm.meta_shm_name)
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        accepted_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(accepted_msg["msg_type"], TransferMsgType.ALLOC_ACCEPTED)
        self.assertEqual(accepted_msg["receiver_role"], RoleType.DENOISER.value)
        self.assertEqual(accepted_msg["request_id"], "req-alloc-1")
        self.assertEqual(accepted_msg["receiver_session_id"], self.engine.session_id)
        self.assertEqual(accepted_msg["data_size"], 4096)
        self.assertEqual(accepted_msg["meta_size"], 2048)
        self.assertIsInstance(accepted_msg["receiver_meta_slot_offset"], int)

    def test_duplicate_alloc_reuses_existing_pending_receive_slot(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-alloc-dup",
            "data_size": 4096,
            "meta_size": 2048,
            "source_control_endpoint": "tcp://upstream-ctrl",
            "source_host_id": "host-a",
        }

        self.scheduler._handle_transfer_alloc(msg)
        first_slot_offset = self.tm.get_receive_slot_offset("req-alloc-dup")
        first_meta_offset = self.tm.get_receive_meta_offset("req-alloc-dup")

        self.tm.send_direct_message.reset_mock()
        self.scheduler._pool_result_push.send_multipart.reset_mock()
        self.scheduler._handle_transfer_alloc(msg)

        self.assertEqual(
            self.tm.get_receive_slot_offset("req-alloc-dup"), first_slot_offset
        )
        self.assertEqual(
            self.tm.get_receive_meta_offset("req-alloc-dup"), first_meta_offset
        )
        self.tm.send_direct_message.assert_called_once()
        _endpoint, peer_msg = self.tm.send_direct_message.call_args[0]
        self.assertEqual(peer_msg.dest_shm_offset, first_slot_offset)
        self.assertEqual(peer_msg.meta_dest_shm_offset, first_meta_offset)
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        accepted_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(accepted_msg["msg_type"], TransferMsgType.ALLOC_ACCEPTED)
        self.assertEqual(accepted_msg["receiver_slot_offset"], first_slot_offset)
        self.assertEqual(accepted_msg["receiver_meta_slot_offset"], first_meta_offset)

    def test_alloc_rejects_stale_receiver_session(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-stale-session",
            "data_size": 4096,
            "meta_size": 512,
            "receiver_session_id": "old-receiver-session",
            "source_control_endpoint": "tcp://upstream-ctrl",
            "source_host_id": "host-a",
        }

        self.scheduler._handle_transfer_alloc(msg)

        self.assertIsNone(self.tm.get_receive_slot_addr("req-stale-session"))
        self.tm.send_direct_message.assert_not_called()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        reply = decode_transfer_msg(sent_frames)
        self.assertEqual(reply["msg_type"], TransferMsgType.ALLOC_REJECT)
        self.assertTrue(reply["retryable"])
        self.assertEqual(reply["reason"], "receiver session changed before allocation")
        self.assertEqual(reply["receiver_session_id"], self.engine.session_id)

    def test_missing_source_control_endpoint_reports_non_retryable_alloc_reject(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-alloc-2",
            "data_size": 4096,
            "meta_size": 512,
        }

        self.scheduler._handle_transfer_alloc(msg)

        self.assertIsNone(self.tm.get_receive_slot_addr("req-alloc-2"))
        self.tm.send_direct_message.assert_not_called()
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        reply = decode_transfer_msg(sent_frames)
        self.assertEqual(reply["msg_type"], TransferMsgType.ALLOC_REJECT)
        self.assertFalse(reply["retryable"])
        self.assertEqual(reply["reason"], "missing source control endpoint")
        self.assertEqual(reply["request_id"], "req-alloc-2")

    def test_allocate_receive_slot_failure_reports_retryable_alloc_reject(self):
        self.tm.allocate_receive_slot = MagicMock(return_value=None)
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-alloc-3",
            "data_size": 4096,
            "meta_size": 512,
            "source_control_endpoint": "tcp://upstream-ctrl",
            "source_host_id": "host-a",
        }

        self.scheduler._handle_transfer_alloc(msg)

        self.assertIsNone(self.tm.get_receive_slot_addr("req-alloc-3"))
        self.tm.send_direct_message.assert_not_called()
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        reply = decode_transfer_msg(sent_frames)
        self.assertEqual(reply["msg_type"], TransferMsgType.ALLOC_REJECT)
        self.assertTrue(reply["retryable"])
        self.assertEqual(reply["reason"], "receiver failed to allocate slot")
        self.assertEqual(reply["request_id"], "req-alloc-3")


class TestSchedulerPrefetchQueues(unittest.TestCase):
    def test_direct_callbacks_queue_work_without_inline_execution(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._handle_transfer_ready = MagicMock()
        scheduler._handle_transfer_failed = MagicMock()

        ready_msg = {"msg_type": TransferMsgType.READY, "request_id": "ready-1"}
        failed_msg = {"msg_type": TransferMsgType.FAILED, "request_id": "fail-1"}

        scheduler._on_direct_transfer_ready(ready_msg)
        scheduler._on_direct_transfer_failed(failed_msg)

        self.assertEqual(scheduler._transferring_queue.get_nowait(), ready_msg)
        self.assertEqual(scheduler._control_queue.get_nowait(), failed_msg)
        scheduler._handle_transfer_ready.assert_not_called()
        scheduler._handle_transfer_failed.assert_not_called()

    def test_encoder_ignores_ready_callback(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transferring_queue = None
        scheduler._handle_transfer_ready = MagicMock()

        scheduler._on_direct_transfer_ready(
            {"msg_type": TransferMsgType.READY, "request_id": "ignored"}
        )

        scheduler._handle_transfer_ready.assert_not_called()

    def test_control_queue_is_serviced_before_transferring_queue(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        order = []

        scheduler._handle_transfer_alloc = lambda msg: order.append(
            ("alloc", msg["request_id"])
        )
        scheduler._prefetch_transfer_ready = lambda msg: (
            order.append(("ready", msg["request_id"]))
            or _PendingInboundTransfer(
                request_id=msg["request_id"],
                role_name="DENOISER",
                scalar_fields={},
                tensors={},
                load_event=None,
                prealloc_slot_id=None,
            )
        )

        scheduler._control_queue.put(
            {"msg_type": TransferMsgType.ALLOC, "request_id": "alloc-first"}
        )
        scheduler._transferring_queue.put(
            {"msg_type": TransferMsgType.READY, "request_id": "ready-second"}
        )

        self.assertTrue(scheduler._process_transfer_control_queue())
        self.assertTrue(scheduler._process_prefetch_queue_once())
        self.assertEqual(order, [("alloc", "alloc-first"), ("ready", "ready-second")])
        self.assertEqual(
            scheduler._swapping_queue.get_nowait().request_id, "ready-second"
        )

    def test_swapping_queue_only_advances_ready_loads(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)

        class _FakeEvent:
            def __init__(self):
                self.ready = False

            def query(self):
                return self.ready

        event = _FakeEvent()
        item = _PendingInboundTransfer(
            request_id="r1",
            role_name="DENOISER",
            scalar_fields={},
            tensors={},
            load_event=event,
            prealloc_slot_id=None,
        )
        scheduler._swapping_queue.put(item)
        self.assertFalse(scheduler._process_swapping_queue_once())
        self.assertTrue(scheduler._compute_ready_queue.empty())
        event.ready = True
        self.assertTrue(scheduler._process_swapping_queue_once())
        self.assertEqual(scheduler._compute_ready_queue.get_nowait().request_id, "r1")


class TestSchedulerTransferReady(unittest.TestCase):
    def setUp(self):
        self.scheduler = _SchedulerHarness.make(RoleType.DECODER)
        self.scheduler._prefetch_transfer_ready = MagicMock(
            return_value=_PendingInboundTransfer(
                request_id="ready-direct",
                role_name="DECODER",
                scalar_fields={},
                tensors={},
                load_event=None,
                prealloc_slot_id=None,
            )
        )
        self.scheduler._wait_transfer_event_on_compute_stream = MagicMock()
        self.scheduler._run_prefetched_compute_item = MagicMock()
        self.scheduler._transferring_queue = None

    def test_handle_transfer_ready_reuses_prefetched_compute_path(self):
        msg = {
            "msg_type": TransferMsgType.READY,
            "request_id": "ready-direct",
        }

        self.scheduler._handle_transfer_ready(msg)

        self.scheduler._wait_transfer_event_on_compute_stream.assert_called_once()
        self.scheduler._run_prefetched_compute_item.assert_called_once()
        item = self.scheduler._run_prefetched_compute_item.call_args[0][0]
        self.assertEqual(item.request_id, "ready-direct")


class TestSchedulerTransferReadyValidation(unittest.TestCase):
    def test_prefetch_rejects_stale_ready_before_metadata_read(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._fail_inbound_transfer = MagicMock()

        class _FakeManager:
            session_id = "new-session"

            def validate_receive_ready(self, *args, **kwargs):
                return "receiver session mismatch"

            def load_transfer_async(self, *args, **kwargs):
                raise AssertionError("metadata should not be read for stale READY")

        scheduler._transfer_manager = _FakeManager()
        msg = {
            "msg_type": TransferMsgType.READY,
            "request_id": "stale-ready",
            "dest_session_id": "old-session",
            "prealloc_slot_id": 7,
        }

        item = scheduler._prefetch_transfer_ready(msg)

        self.assertIsNone(item)
        scheduler._fail_inbound_transfer.assert_called_once()
        args = scheduler._fail_inbound_transfer.call_args[0]
        self.assertEqual(args[0], "stale-ready")
        self.assertIn("Transfer ready validation failed", args[1])
        self.assertEqual(args[2], 7)

    def test_prefetch_metadata_magic_failure_is_reported_gracefully(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._fail_inbound_transfer = MagicMock()

        class _FakeManager:
            session_id = "receiver-session"

            def validate_receive_ready(self, *args, **kwargs):
                return None

            def load_transfer_async(self, *args, **kwargs):
                raise ValueError("Invalid transfer metadata magic")

        scheduler._transfer_manager = _FakeManager()
        msg = {
            "msg_type": TransferMsgType.READY,
            "request_id": "bad-metadata",
            "dest_session_id": "receiver-session",
        }

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.time.sleep"
        ):
            item = scheduler._prefetch_transfer_ready(msg)

        self.assertIsNone(item)
        scheduler._fail_inbound_transfer.assert_called_once()
        self.assertIn(
            "Invalid transfer metadata",
            scheduler._fail_inbound_transfer.call_args[0][1],
        )

    def test_invalid_denoiser_scalar_fields_fail_before_compute(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._fail_inbound_transfer = MagicMock()
        scheduler._release_pending_receive = MagicMock()
        scheduler._build_disagg_compute_req = MagicMock()
        item = _PendingInboundTransfer(
            request_id="bad-denoiser",
            role_name="DENOISER",
            scalar_fields={"request_id": "bad-denoiser", "num_inference_steps": None},
            tensors={},
            load_event=None,
            prealloc_slot_id=11,
        )

        scheduler._run_prefetched_compute_item(item, is_multi_rank=False)

        scheduler._fail_inbound_transfer.assert_called_once()
        self.assertIn(
            "invalid num_inference_steps",
            scheduler._fail_inbound_transfer.call_args[0][1],
        )
        scheduler._release_pending_receive.assert_not_called()
        scheduler._build_disagg_compute_req.assert_not_called()


class TestSchedulerTransferStreams(unittest.TestCase):
    def test_wait_transfer_event_on_compute_stream_binds_event_to_compute_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        stream_state = {"active": False, "stream": None}
        current_stream = MagicMock()

        def _wait_event(event):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)

        current_stream.wait_event.side_effect = _wait_event

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.current_stream",
            return_value=current_stream,
        ):
            scheduler._wait_transfer_event_on_compute_stream(object())

        current_stream.wait_event.assert_called_once()

    def test_prefetch_uses_swap_in_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._swap_in_stream = object()
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.load_transfer_async.return_value = ({}, {}, None)

        scheduler._prefetch_transfer_ready({"request_id": "stream-in-1"})

        scheduler._transfer_manager.load_transfer_async.assert_called_once_with(
            "stream-in-1",
            device="cpu",
            stream=scheduler._swap_in_stream,
        )

    def test_encoder_staging_uses_swap_out_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._swap_out_stream = object()
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )

        scheduler._disagg_encoder_transfer_stage(
            "stream-out-1",
            {"latents": torch.randn(1, 4, 4, 4)},
            {"request_id": "stream-out-1"},
        )

        scheduler._transfer_manager.stage_tensors_async.assert_called_once_with(
            request_id="stream-out-1",
            tensor_fields=unittest.mock.ANY,
            scalar_fields={"request_id": "stream-out-1"},
            stream=scheduler._swap_out_stream,
        )

    def test_sender_staging_failure_retries_without_client_error(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transfer_manager = MagicMock()
        staged = SimpleNamespace(
            transfer_size=1,
            meta_size=1,
            scalar_fields={},
            slot=None,
            meta_slot=None,
        )
        scheduler._transfer_manager.stage_tensors_async.return_value = (staged, None)

        enqueued = scheduler._finalize_disagg_encoder_stage(
            "stage-retry",
            None,
            None,
            {"latents": torch.randn(1, 4, 4, 4)},
            {"request_id": "stage-retry"},
        )

        self.assertFalse(enqueued)
        self.assertEqual(len(scheduler._outbound_staging_retry_queue), 1)
        scheduler._pool_result_push.send_multipart.assert_not_called()

        self.assertTrue(scheduler._process_outbound_staging_retry_once())

        scheduler._transfer_manager.stage_tensors_async.assert_called_once_with(
            request_id="stage-retry",
            tensor_fields=unittest.mock.ANY,
            scalar_fields={"request_id": "stage-retry"},
            stream=scheduler._swap_out_stream,
        )
        self.assertEqual(len(scheduler._outbound_staging_retry_queue), 0)
        self.assertEqual(len(scheduler._swap_out_queue), 1)
        self.assertEqual(scheduler._swap_out_queue[0].request_id, "stage-retry")

    def test_staging_backpressure_defers_new_alloc_control(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._handle_transfer_alloc = MagicMock()
        scheduler._handle_transfer_abort = MagicMock()
        scheduler._control_queue.put(
            {"msg_type": TransferMsgType.ALLOC, "request_id": "new-work"}
        )
        scheduler._control_queue.put(
            {"msg_type": TransferMsgType.ABORT, "request_id": "abort-me"}
        )

        self.assertTrue(scheduler._process_transfer_control_queue(allow_new_work=False))

        scheduler._handle_transfer_alloc.assert_not_called()
        scheduler._handle_transfer_abort.assert_called_once()
        deferred = scheduler._control_queue.get_nowait()
        self.assertEqual(deferred["request_id"], "new-work")

    def test_encoder_forward_runs_on_compute_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        scheduler._swap_out_stream = object()
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )
        stream_state = {"active": False, "stream": None}

        def _forward(batch, return_req=False):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)
            return batch[0]

        scheduler.worker.execute_forward = MagicMock(side_effect=_forward)
        req = Req(request_id="enc-compute")
        frames = [b"enc-compute", pickle.dumps([req])]

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ):
            scheduler._disagg_encoder_step(MagicMock(), frames=frames)

        scheduler.worker.execute_forward.assert_called_once()
        scheduler._transfer_manager.stage_tensors_async.assert_called_once()

    def test_encoder_abort_after_stage_cleans_local_transfer_state(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )
        scheduler._is_request_aborted = MagicMock(return_value=True)
        scheduler._warmup_inbound_sizes["enc-abort"] = (32, 16)
        scheduler.worker.execute_forward = MagicMock(
            return_value=Req(request_id="enc-abort")
        )
        frames = [b"enc-abort", pickle.dumps([Req(request_id="enc-abort")])]

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.extract_transfer_fields",
            return_value=({"latents": torch.randn(1, 1)}, {"request_id": "enc-abort"}),
        ):
            scheduler._disagg_encoder_step(MagicMock(), frames=frames)

        scheduler._transfer_manager.abort_request.assert_called_once_with("enc-abort")
        self.assertNotIn("enc-abort", scheduler._warmup_inbound_sizes)

    def test_denoiser_abort_after_stage_cleans_local_transfer_state(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )
        scheduler._enqueue_outbound_transfer = MagicMock()
        scheduler._is_request_aborted = MagicMock(side_effect=[False, True])
        scheduler._warmup_inbound_sizes["den-abort"] = (64, 32)
        scheduler.worker.execute_forward = MagicMock(
            return_value=Req(request_id="den-abort")
        )

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.extract_transfer_fields",
            return_value=({"latents": torch.randn(1, 1)}, {"request_id": "den-abort"}),
        ):
            scheduler._disagg_denoiser_compute(
                Req(request_id="den-abort"),
                "den-abort",
                "DENOISER",
            )

        scheduler._transfer_manager.abort_request.assert_called_once_with("den-abort")
        scheduler._enqueue_outbound_transfer.assert_not_called()
        self.assertNotIn("den-abort", scheduler._warmup_inbound_sizes)

    def test_decoder_forward_runs_on_compute_stream_and_waits_before_send(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        stream_state = {"active": False, "stream": None}
        current_stream = MagicMock()

        def _forward(batch):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)
            return SimpleNamespace(
                output=torch.zeros(1),
                audio=None,
                audio_sample_rate=None,
                error=None,
            )

        scheduler.worker.execute_forward = MagicMock(side_effect=_forward)

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.current_stream",
            return_value=current_stream,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.send_tensors"
        ) as mock_send_tensors:
            scheduler._disagg_decoder_compute(
                Req(request_id="dec-compute"),
                "dec-compute",
                "DECODER",
            )

        current_stream.wait_stream.assert_called_once_with(scheduler._compute_stream)
        mock_send_tensors.assert_called_once()

    def test_follower_compute_uses_compute_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        stream_state = {"active": False, "stream": None}

        def _forward(batch, return_req=False):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)
            return batch[0]

        scheduler.worker.execute_forward = MagicMock(side_effect=_forward)

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ):
            scheduler._disagg_compute_non_rank0({"request_id": "follower-compute"})

        scheduler.worker.execute_forward.assert_called_once()


class TestSchedulerWarmupCalibration(unittest.TestCase):
    def test_calibrated_pool_size_uses_rounded_buddy_slot(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler.server_args.disagg_max_slots_per_instance = 32
        scheduler.server_args.disagg_transfer_redundancy = 1.25
        scheduler.server_args.disagg_transfer_pool_size = 256 * 1024 * 1024
        expected_slot_size = 64 * 1024 * 1024
        expected_pool_size = int(expected_slot_size * 32 * 1.25)

        fake_engine = MagicMock()
        fake_engine.session_id = "session-sized"
        fake_manager = MagicMock()
        fake_manager.session_id = "session-sized"
        fake_manager.pool_data_ptr = 123
        fake_manager.pool_size = expected_pool_size
        fake_manager.meta_pool_ptr = 456
        fake_manager.meta_pool_size = 1024
        fake_manager.data_shm_name = None
        fake_manager.meta_shm_name = None
        fake_manager.host_id = "host-sized"
        fake_manager.free_slots_count.return_value = 32

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.create_transfer_engine",
            return_value=fake_engine,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferTensorBuffer",
        ) as buffer_cls, patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.TransferMetaBuffer"
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.DiffusionTransferManager",
            return_value=fake_manager,
        ):
            scheduler._init_disagg_transfer_manager(
                measured_transfer_bytes=37_356_032,
                measured_meta_bytes=1024,
            )

        buffer_cls.assert_called_once()
        buffer_kwargs = buffer_cls.call_args.kwargs
        self.assertEqual(buffer_kwargs["pool_size"], expected_pool_size)
        self.assertEqual(buffer_kwargs["device"], "cpu")
        self.assertEqual(buffer_kwargs["role_name"], RoleType.ENCODER.value)
        self.assertFalse(buffer_kwargs["pin_memory"])
        self.assertFalse(buffer_kwargs["pin_memory_strict"])
        sent_frames = scheduler._pool_result_push.send_multipart.call_args[0][0]
        register_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(register_msg["capacity_slot_size"], expected_slot_size)
        self.assertEqual(register_msg["capacity_slots"], 32)

    def test_schedule_transfer_reconfigure_keeps_max_sizes(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transfer_manager = MagicMock()

        scheduler._schedule_transfer_reconfigure(128, 32)
        scheduler._schedule_transfer_reconfigure(64, 256)

        self.assertEqual(
            scheduler._pending_transfer_reconfigure,
            {"transfer_bytes": 128, "meta_bytes": 256},
        )

    def test_apply_pending_transfer_reconfigure_rebuilds_idle_manager(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        manager = MagicMock()
        manager.has_active_transfers.return_value = False
        scheduler._transfer_manager = manager
        scheduler._pending_transfer_reconfigure = {
            "transfer_bytes": 4096,
            "meta_bytes": 1024,
        }
        scheduler._preallocated_slots = {"stale": object()}
        scheduler._init_disagg_transfer_manager = MagicMock()

        rebuilt = scheduler._maybe_apply_pending_transfer_reconfigure()

        self.assertTrue(rebuilt)
        manager.cleanup.assert_called_once()
        scheduler._init_disagg_transfer_manager.assert_called_once_with(
            measured_transfer_bytes=4096,
            measured_meta_bytes=1024,
        )
        self.assertIsNone(scheduler._pending_transfer_reconfigure)
        self.assertTrue(scheduler._transfer_reconfigured)
        self.assertEqual(scheduler._preallocated_slots, {})

    def test_apply_pending_transfer_reconfigure_waits_for_local_queues(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        manager = MagicMock()
        manager.has_active_transfers.return_value = False
        scheduler._transfer_manager = manager
        scheduler._pending_transfer_reconfigure = {
            "transfer_bytes": 4096,
            "meta_bytes": 1024,
        }
        scheduler._control_queue.put({"msg_type": TransferMsgType.READY})
        scheduler._init_disagg_transfer_manager = MagicMock()

        rebuilt = scheduler._maybe_apply_pending_transfer_reconfigure()

        self.assertFalse(rebuilt)
        manager.cleanup.assert_not_called()
        scheduler._init_disagg_transfer_manager.assert_not_called()

    def test_encoder_warmup_send_completion_schedules_reconfigure(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.send_direct_message = MagicMock()
        scheduler._schedule_transfer_reconfigure = MagicMock()

        scheduler._on_direct_send_completion(
            "warmup-enc",
            SimpleNamespace(
                receiver_control_endpoint="tcp://receiver",
                prealloc_slot_id=3,
            ),
            SimpleNamespace(
                scalar_fields={"is_warmup": True},
                transfer_size=8192,
                meta_size=512,
            ),
            True,
            None,
        )

        scheduler._schedule_transfer_reconfigure.assert_called_once_with(8192, 512)

    def test_denoiser_warmup_send_completion_uses_max_inbound_and_outbound(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.send_direct_message = MagicMock()
        scheduler._schedule_transfer_reconfigure = MagicMock()
        scheduler._warmup_inbound_sizes["warmup-den"] = (2048, 1024)

        scheduler._on_direct_send_completion(
            "warmup-den",
            SimpleNamespace(
                receiver_control_endpoint="tcp://receiver",
                prealloc_slot_id=5,
            ),
            SimpleNamespace(
                scalar_fields={"is_warmup": True},
                transfer_size=4096,
                meta_size=256,
            ),
            True,
            None,
        )

        scheduler._schedule_transfer_reconfigure.assert_called_once_with(4096, 1024)
        self.assertNotIn("warmup-den", scheduler._warmup_inbound_sizes)

    def test_decoder_warmup_compute_schedules_reconfigure_from_inbound_sizes(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler._release_pending_receive = MagicMock()
        scheduler._build_disagg_compute_req = MagicMock(
            return_value=Req(request_id="warmup-dec")
        )
        scheduler._disagg_decoder_compute = MagicMock()
        scheduler._schedule_transfer_reconfigure = MagicMock()
        scheduler._warmup_inbound_sizes["warmup-dec"] = (1536, 384)
        item = _PendingInboundTransfer(
            request_id="warmup-dec",
            role_name="DECODER",
            scalar_fields={"request_id": "warmup-dec", "is_warmup": True},
            tensors={"latents": torch.randn(1, 4, 4, 4)},
            load_event=None,
            prealloc_slot_id=9,
        )

        scheduler._run_prefetched_compute_item(item, is_multi_rank=False)

        scheduler._schedule_transfer_reconfigure.assert_called_once_with(1536, 384)
        self.assertNotIn("warmup-dec", scheduler._warmup_inbound_sizes)


class TestSchedulerTensorDistribution(unittest.TestCase):
    def test_build_disagg_req_routes_sampling_fields_through_sampling_params(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)

        req = scheduler._build_disagg_req(
            {
                "request_id": "req-sampling",
                "num_inference_steps": 50,
                "width": 1280,
                "height": 704,
                "seed": 123,
            },
            {},
        )

        self.assertEqual(req.request_id, "req-sampling")
        self.assertEqual(req.sampling_params.request_id, "req-sampling")
        self.assertEqual(req.num_inference_steps, 50)
        self.assertEqual(req.sampling_params.num_inference_steps, 50)
        self.assertEqual(req.width, 1280)
        self.assertEqual(req.height, 704)
        self.assertIsNotNone(req.generator)

    def test_denoiser_stage_managed_sp_shards_and_marks_req(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.sp_degree = 2
        scheduler.server_args.pipeline_config = SimpleNamespace(
            shard_latents_for_sp=MagicMock(
                side_effect=lambda req, tensor: (tensor[..., :1].clone(), True)
            )
        )
        scheduler._broadcast_tensor_payload_to_all_ranks = MagicMock(
            side_effect=lambda payload: payload
        )

        latents = torch.randn(1, 2, 4, 4)
        image_latent = torch.randn(1, 2, 4, 4)
        prompt_embeds = torch.randn(1, 8, 16)

        req = scheduler._build_disagg_compute_req(
            {"request_id": "req-sp-1", "enable_sequence_shard": False},
            {
                "latents": latents,
                "image_latent": image_latent,
                "prompt_embeds": prompt_embeds,
            },
        )

        self.assertEqual(tuple(req.latents.shape), (1, 2, 4, 1))
        self.assertEqual(tuple(req.image_latent.shape), (1, 2, 4, 1))
        self.assertEqual(tuple(req.prompt_embeds.shape), tuple(prompt_embeds.shape))
        self.assertEqual(
            getattr(req, "_disagg_pre_sharded_fields"),
            ("image_latent", "latents"),
        )
        self.assertEqual(
            scheduler.server_args.pipeline_config.shard_latents_for_sp.call_count, 2
        )

    def test_denoiser_model_managed_sp_keeps_full_inputs(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.sp_degree = 2
        scheduler.server_args.pipeline_config = SimpleNamespace(
            shard_latents_for_sp=MagicMock()
        )
        scheduler._broadcast_tensor_payload_to_all_ranks = MagicMock(
            side_effect=lambda payload: payload
        )

        latents = torch.randn(1, 2, 4, 4)
        req = scheduler._build_disagg_compute_req(
            {"request_id": "req-sp-2", "enable_sequence_shard": True},
            {"latents": latents},
        )

        self.assertTrue(torch.equal(req.latents, latents))
        self.assertFalse(hasattr(req, "_disagg_pre_sharded_fields"))
        scheduler.server_args.pipeline_config.shard_latents_for_sp.assert_not_called()

    def test_decoder_parallel_decode_keeps_full_latents(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler.server_args.sp_degree = 2
        scheduler._broadcast_tensor_payload_to_all_ranks = MagicMock(
            side_effect=lambda payload: payload
        )
        scheduler.worker.pipeline = SimpleNamespace(
            get_module=lambda name: (
                SimpleNamespace(use_parallel_decode=True) if name == "vae" else None
            )
        )

        latents = torch.randn(1, 4, 4, 4, 4)
        req = scheduler._build_disagg_compute_req(
            {"request_id": "req-dec-1"},
            {"latents": latents},
        )

        self.assertTrue(torch.equal(req.latents, latents))
        self.assertFalse(hasattr(req, "_disagg_pre_sharded_fields"))


class TestDenoisingStagePreShardedInputs(unittest.TestCase):
    def test_preprocess_sp_latents_skips_disagg_pre_sharded_fields(self):
        batch = Req(request_id="req-stage-1")
        batch.latents = torch.randn(1, 4, 4, 4, 4)
        batch.image_latent = torch.randn(1, 4, 4, 4, 4)
        batch._disagg_pre_sharded_fields = ("latents", "image_latent")

        pipeline_config = SimpleNamespace(shard_latents_for_sp=MagicMock())
        stage = object.__new__(DenoisingStage)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_sp_world_size",
            return_value=2,
        ):
            DenoisingStage._preprocess_sp_latents(
                stage,
                batch,
                SimpleNamespace(pipeline_config=pipeline_config),
            )

        self.assertTrue(batch.did_sp_shard_latents)
        pipeline_config.shard_latents_for_sp.assert_not_called()


class TestSchedulerTransferEncoderStaging(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)
        self.engine = MockTransferEngine(session_id="encoder-session")
        self.buffer = TransferTensorBuffer(pool_size=2 * 1024 * 1024, role_name="test")
        self.meta_buffer = TransferMetaBuffer(
            slot_count=2, slot_size=64 * 1024, role_name="test"
        )
        self.tm = DiffusionTransferManager(
            engine=self.engine,
            buffer=self.buffer,
            meta_buffer=self.meta_buffer,
            host_id="host-a",
        )
        self.addCleanup(self.tm.cleanup)
        self.scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        self.scheduler._transfer_manager = self.tm

    def test_encoder_transfer_stage_enqueues_then_sends_staged_msg(self):
        tensor_fields = {
            "prompt_embeds": torch.randn(1, 8, 32),
            "latents": torch.randn(1, 4, 16, 16),
        }
        scalar_fields = {"request_id": "req-enc-1", "guidance_scale": 7.5}

        self.scheduler._disagg_encoder_transfer_stage(
            "req-enc-1", tensor_fields, scalar_fields
        )

        self.assertEqual(len(self.scheduler._swap_out_queue), 1)
        self.assertTrue(self.scheduler._process_swap_out_queue_once())
        self.assertTrue(self.scheduler._process_send_ready_queue_once())

        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        staged_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(staged_msg["msg_type"], TransferMsgType.STAGED)
        self.assertEqual(staged_msg["request_id"], "req-enc-1")
        self.assertEqual(staged_msg["session_id"], self.engine.session_id)
        self.assertGreater(staged_msg["meta_size"], 0)


# Consolidated from test/multimodal_gen/test_disagg_transfer_sizing_unittest.py.
class TestDisaggTransferSizing(unittest.TestCase):
    def test_transfer_size_alignment_matches_scheduler_estimator(self):
        tensor_fields = {
            "latents": [
                torch.zeros(3, dtype=torch.float16),
                torch.zeros(7, dtype=torch.float16),
            ],
            "prompt_embeds": torch.zeros(5, dtype=torch.float32),
        }

        expected = 1536

        self.assertEqual(
            DiffusionTransferManager._estimate_transfer_size(tensor_fields),
            expected,
        )
        self.assertEqual(estimate_transfer_bytes(tensor_fields), expected)

    def test_transfer_size_estimator_handles_empty_payload(self):
        tensor_fields = {"latents": None}

        self.assertEqual(
            DiffusionTransferManager._estimate_transfer_size(tensor_fields),
            0,
        )
        self.assertEqual(estimate_transfer_bytes(tensor_fields), 0)


if __name__ == "__main__":
    unittest.main()
