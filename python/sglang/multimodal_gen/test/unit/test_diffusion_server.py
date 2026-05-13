# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DiffusionServer pool-based pipeline orchestrator."""

import pickle
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.multimodal_gen.runtime.disaggregation.diffusion_server import (
    DiffusionServer,
    _TransferRequestState,
)
from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    MaxFreeSlotsFirst,
    PoolDispatcher,
)
from sglang.multimodal_gen.runtime.disaggregation.metrics import DisaggMetrics
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
    TransferPhase,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
)
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
    _is_skip_broadcast,
    _should_broadcast_encoder_idle_skip,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocAcceptedMsg,
    TransferAllocRejectMsg,
    TransferPeerInfoMsg,
    TransferPushedMsg,
    TransferRegisterMsg,
    TransferStagedMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import GetDisaggStatsReq
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestDiffusionServerInit(unittest.TestCase):
    def test_basic_init(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19900",
            encoder_work_endpoints=["tcp://127.0.0.1:19901"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19902"],
            decoder_work_endpoints=["tcp://127.0.0.1:19903"],
            encoder_result_endpoint="tcp://127.0.0.1:19904",
            denoiser_result_endpoint="tcp://127.0.0.1:19905",
            decoder_result_endpoint="tcp://127.0.0.1:19906",
            max_slots_per_instance=3,
        )
        self.addCleanup(server.stop)
        self.assertEqual(server._encoder_free_slots, [3])
        self.assertEqual(server._denoiser_free_slots, [3])
        self.assertEqual(server._decoder_free_slots, [3])


class TestDiffusionServerClientRequests(unittest.TestCase):
    def setUp(self):
        self.server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19800",
            encoder_work_endpoints=["tcp://127.0.0.1:19801"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19802"],
            decoder_work_endpoints=["tcp://127.0.0.1:19803"],
            encoder_result_endpoint="tcp://127.0.0.1:19804",
            denoiser_result_endpoint="tcp://127.0.0.1:19805",
            decoder_result_endpoint="tcp://127.0.0.1:19806",
        )
        self.addCleanup(self.server.stop)

    def _frontend_with_payload(self, payload):
        frontend = MagicMock()
        frontend.recv_multipart.return_value = [b"client", b"", payload]
        return frontend

    def _response_from_frontend(self, frontend):
        frames = frontend.send_multipart.call_args[0][0]
        return pickle.loads(frames[-1])

    def test_stats_request_returns_output_batch(self):
        frontend = self._frontend_with_payload(pickle.dumps(GetDisaggStatsReq()))

        self.server._handle_client_request(frontend)

        response = self._response_from_frontend(frontend)
        self.assertIsNone(response.error)
        self.assertEqual(response.output["role"], "diffusion_server")

    def test_bad_pickle_returns_error_response(self):
        frontend = self._frontend_with_payload(b"not-a-pickle")

        self.server._handle_client_request(frontend)

        response = self._response_from_frontend(frontend)
        self.assertIn("deserialize", response.error)

    def test_unsupported_dict_request_returns_error_response(self):
        frontend = self._frontend_with_payload(pickle.dumps({"method": "ping"}))

        self.server._handle_client_request(frontend)

        response = self._response_from_frontend(frontend)
        self.assertIn("Unsupported request type", response.error)

    def test_duplicate_request_id_returns_error_response(self):
        self.server._tracker.submit("dup-req")
        frontend = self._frontend_with_payload(
            pickle.dumps(SimpleNamespace(request_id="dup-req", metrics=None))
        )

        self.server._handle_client_request(frontend)

        response = self._response_from_frontend(frontend)
        self.assertIn("Duplicate request_id", response.error)


class TestDiffusionServerTransferProtocol(unittest.TestCase):
    def setUp(self):
        self.server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19910",
            encoder_work_endpoints=["tcp://127.0.0.1:19911"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19912"],
            decoder_work_endpoints=["tcp://127.0.0.1:19913"],
            encoder_result_endpoint="tcp://127.0.0.1:19914",
            denoiser_result_endpoint="tcp://127.0.0.1:19915",
            decoder_result_endpoint="tcp://127.0.0.1:19916",
            max_slots_per_instance=2,
        )
        self.addCleanup(self.server.stop)
        self.server._encoder_pushes = [MagicMock()]
        self.server._denoiser_pushes = [MagicMock()]
        self.server._decoder_pushes = [MagicMock()]

    def _submit_running_request(self, request_id: str, state: RequestState):
        record = self.server._tracker.submit(request_id)
        if state == RequestState.ENCODER_RUNNING:
            record.encoder_instance = 0
        elif state in (
            RequestState.DENOISING_RUNNING,
            RequestState.DENOISING_DONE,
            RequestState.DENOISING_WAITING,
        ):
            record.encoder_instance = 0
            record.denoiser_instance = 0
        elif state in (RequestState.DECODER_RUNNING, RequestState.DECODER_WAITING):
            record.encoder_instance = 0
            record.denoiser_instance = 0
            record.decoder_instance = 0
        record.state = state

    def test_transfer_register_tracks_host_meta_and_ignores_prealloc(self):
        reg_msg = TransferRegisterMsg(
            role="denoiser",
            instance_id=0,
            session_id="den-session-0",
            pool_ptr=0x7F000000,
            pool_size=16 * 1024 * 1024,
            meta_pool_ptr=0x8F000000,
            meta_pool_size=128 * 1024,
            control_endpoint="tcp://den-ctrl",
            host_id="host-a",
            supports_local_copy=True,
            data_shm_name="data-shm",
            meta_shm_name="meta-shm",
            preallocated_slots=[
                {
                    "slot_id": 4,
                    "offset": 256,
                    "size": 4096,
                    "addr": 0x7F000100,
                    "meta_offset": 128,
                    "meta_size": 2048,
                    "meta_addr": 0x8F000080,
                }
            ],
        )
        self.server._handle_transfer_result(
            encode_transfer_msg(reg_msg), RoleType.DENOISER
        )

        peer = self.server._denoiser_peers[0]
        self.assertEqual(peer["control_endpoint"], "tcp://den-ctrl")
        self.assertEqual(peer["host_id"], "host-a")
        self.assertTrue(peer["supports_local_copy"])
        self.assertEqual(peer["meta_pool_ptr"], 0x8F000000)
        self.assertEqual(peer["free_preallocated_slots"], [])

    def test_transfer_register_updates_effective_capacity(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19920",
            encoder_work_endpoints=["tcp://127.0.0.1:19921"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19922"],
            decoder_work_endpoints=["tcp://127.0.0.1:19923"],
            encoder_result_endpoint="tcp://127.0.0.1:19924",
            denoiser_result_endpoint="tcp://127.0.0.1:19925",
            decoder_result_endpoint="tcp://127.0.0.1:19926",
            max_slots_per_instance=32,
        )
        self.addCleanup(server.stop)
        server._denoiser_free_slots[0] = 31

        server._handle_transfer_result(
            encode_transfer_msg(
                TransferRegisterMsg(
                    role="denoiser",
                    instance_id=0,
                    session_id="den-capacity",
                    pool_size=256 * 1024 * 1024,
                    capacity_slots=4,
                    capacity_slot_size=64 * 1024 * 1024,
                )
            ),
            RoleType.DENOISER,
        )

        self.assertEqual(server._denoiser_capacity_limits[0], 4)
        self.assertEqual(server._denoiser_free_slots[0], 3)
        self.assertEqual(server._denoiser_peers[0]["capacity_slots"], 4)
        self.assertEqual(
            server._denoiser_peers[0]["capacity_slot_size"],
            64 * 1024 * 1024,
        )

    def test_transfer_staged_dispatches_dynamic_alloc_with_meta_and_host(self):
        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferRegisterMsg(
                    role="encoder",
                    instance_id=0,
                    session_id="enc-session-0",
                    pool_ptr=0x1000,
                    pool_size=16 * 1024 * 1024,
                    meta_pool_ptr=0x1800,
                    meta_pool_size=128 * 1024,
                    control_endpoint="tcp://enc-ctrl",
                    host_id="host-a",
                )
            ),
            RoleType.ENCODER,
        )
        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferRegisterMsg(
                    role="denoiser",
                    instance_id=0,
                    session_id="den-session-0",
                    pool_ptr=0x2000,
                    pool_size=16 * 1024 * 1024,
                    meta_pool_ptr=0x2800,
                    meta_pool_size=128 * 1024,
                    control_endpoint="tcp://den-ctrl",
                    host_id="host-a",
                    preallocated_slots=[
                        {
                            "slot_id": 1,
                            "offset": 512,
                            "size": 4096,
                            "addr": 0x2000 + 512,
                            "meta_offset": 128,
                            "meta_size": 2048,
                            "meta_addr": 0x2800 + 128,
                        }
                    ],
                )
            ),
            RoleType.DENOISER,
        )
        self._submit_running_request("r1", RequestState.ENCODER_RUNNING)

        staged_msg = TransferStagedMsg(
            request_id="r1",
            data_size=4096,
            meta_size=2048,
            session_id="enc-session-0",
            pool_ptr=0x1000,
            slot_offset=0,
            meta_pool_ptr=0x1800,
            meta_slot_offset=64,
        )
        self.server._handle_transfer_result(
            encode_transfer_msg(staged_msg), RoleType.ENCODER
        )
        self.server._drain_denoiser_tta()

        sent_frames = self.server._denoiser_pushes[0].send_multipart.call_args[0][0]
        alloc_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(alloc_msg["msg_type"], "transfer_alloc")
        self.assertEqual(alloc_msg["source_control_endpoint"], "tcp://enc-ctrl")
        self.assertEqual(alloc_msg["source_host_id"], "host-a")
        self.assertEqual(alloc_msg["receiver_session_id"], "den-session-0")
        self.assertEqual(alloc_msg["meta_size"], 2048)
        self.assertIsNone(alloc_msg.get("preallocated_slot"))
        self.assertEqual(
            self.server._tracker.get("r1").state,
            RequestState.DENOISING_WAITING,
        )

    def test_transfer_staged_keeps_encoder_slot_busy_until_push_and_starts_wait_timer(
        self,
    ):
        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferRegisterMsg(
                    role="encoder",
                    instance_id=0,
                    session_id="enc-session-0",
                    pool_ptr=0x1000,
                    pool_size=16 * 1024 * 1024,
                    meta_pool_ptr=0x1800,
                    meta_pool_size=128 * 1024,
                    control_endpoint="tcp://enc-ctrl",
                    host_id="host-a",
                )
            ),
            RoleType.ENCODER,
        )
        self._submit_running_request("r-stage", RequestState.ENCODER_RUNNING)
        self.server._encoder_free_slots[0] = 0
        staged_msg = TransferStagedMsg(
            request_id="r-stage",
            data_size=4096,
            meta_size=2048,
            session_id="enc-session-0",
            pool_ptr=0x1000,
            slot_offset=0,
            meta_pool_ptr=0x1800,
            meta_slot_offset=64,
        )

        self.server._handle_transfer_result(
            encode_transfer_msg(staged_msg), RoleType.ENCODER
        )

        self.assertEqual(self.server._encoder_free_slots[0], 0)
        self.assertEqual(
            self.server._tracker.get("r-stage").state,
            RequestState.DENOISING_WAITING,
        )
        self.assertIsNotNone(
            self.server._transfer_state["r-stage"].downstream_wait_since
        )
        self.assertEqual(
            self.server._transfer_state["r-stage"].transfer_phase,
            TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT,
        )

    def test_transfer_pushed_releases_sender_slot_once_and_starts_running(self):
        self._submit_running_request("r-pushed", RequestState.DENOISING_WAITING)
        self.server._tracker.update_instances("r-pushed", denoiser_instance=0)
        self.server._encoder_free_slots[0] = 0
        self.server._transfer_state["r-pushed"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_slot_released=False,
            alloc_accepted=True,
        )

        pushed = encode_transfer_msg(
            TransferPushedMsg(request_id="r-pushed", success=True)
        )
        self.server._handle_transfer_result(pushed, RoleType.ENCODER)
        self.server._handle_transfer_result(pushed, RoleType.ENCODER)

        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.assertEqual(
            self.server._tracker.get("r-pushed").state,
            RequestState.DENOISING_RUNNING,
        )

    def test_stale_transfer_pushed_is_ignored(self):
        self._submit_running_request("r-stale-pushed", RequestState.DENOISING_WAITING)
        self.server._tracker.update_instances("r-stale-pushed", denoiser_instance=0)
        self.server._encoder_free_slots[0] = 0
        self.server._transfer_state["r-stale-pushed"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_session_id="new-sender-session",
            receiver_session_id="new-receiver-session",
            sender_slot_released=False,
            alloc_accepted=True,
        )

        pushed = encode_transfer_msg(
            TransferPushedMsg(
                request_id="r-stale-pushed",
                success=True,
                source_session_id="old-sender-session",
                dest_session_id="new-receiver-session",
                receiver_role=RoleType.DENOISER.value,
                receiver_instance=0,
            )
        )

        self.server._handle_transfer_result(pushed, RoleType.ENCODER)

        self.assertEqual(self.server._encoder_free_slots[0], 0)
        self.assertFalse(
            self.server._transfer_state["r-stale-pushed"].transfer_completion_processed
        )
        self.assertEqual(
            self.server._tracker.get("r-stale-pushed").state,
            RequestState.DENOISING_WAITING,
        )

    def test_fatal_alloc_reject_releases_sender_slot(self):
        self._submit_running_request("r-fatal", RequestState.DENOISING_WAITING)
        self.server._pending["r-fatal"] = b"client"
        self.server._frontend = MagicMock()
        self.server._send_abort = MagicMock()
        self.server._encoder_free_slots[0] = 0
        self.server._tracker.update_instances("r-fatal", denoiser_instance=0)
        self.server._transfer_state["r-fatal"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_control_endpoint="tcp://enc-ctrl",
            sender_slot_released=False,
            downstream_wait_since=1.0,
        )

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocRejectMsg(
                    request_id="r-fatal",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                    retryable=False,
                    reason="fatal-busy",
                )
            ),
            RoleType.DENOISER,
        )

        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.server._send_abort.assert_called_once()
        self.assertIsNone(self.server._tracker.get("r-fatal"))

    def test_retryable_alloc_reject_requeues_request(self):
        self._submit_running_request("r-retry", RequestState.DENOISING_WAITING)
        self.server._tracker.update_instances("r-retry", denoiser_instance=0)
        self.server._denoiser_free_slots[0] = 0
        self.server._dispatcher.select_denoiser_with_capacity = MagicMock(
            return_value=1
        )
        p2p = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            downstream_wait_since=1.0,
        )
        self.server._transfer_state["r-retry"] = p2p

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocRejectMsg(
                    request_id="r-retry",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                    retryable=True,
                    reason="busy",
                )
            ),
            RoleType.DENOISER,
        )

        self.assertIn("r-retry", self.server._transfer_state)
        self.assertEqual(len(self.server._denoiser_tta), 1)
        self.assertEqual(self.server._denoiser_tta[0].request_id, "r-retry")
        self.assertEqual(
            self.server._tracker.get("r-retry").state,
            RequestState.DENOISING_WAITING,
        )
        self.assertGreater(p2p.next_downstream_retry_at, 0.0)
        self.assertEqual(p2p.downstream_retry_attempts, 1)

    def test_retryable_alloc_reject_without_alternative_retries_after_backoff(self):
        self._submit_running_request("r-no-alt", RequestState.DENOISING_WAITING)
        self.server._pending["r-no-alt"] = b"client"
        self.server._frontend = MagicMock()
        self.server._send_abort = MagicMock()
        self.server._encoder_free_slots[0] = 0
        self.server._denoiser_free_slots[0] = 0
        self.server._tracker.update_instances("r-no-alt", denoiser_instance=0)
        self.server._transfer_state["r-no-alt"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_control_endpoint="tcp://enc-ctrl",
            downstream_wait_since=1.0,
        )

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocRejectMsg(
                    request_id="r-no-alt",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                    retryable=True,
                    reason="busy",
                )
            ),
            RoleType.DENOISER,
        )

        self.server._send_abort.assert_not_called()
        self.server._frontend.send_multipart.assert_not_called()
        self.assertEqual(self.server._encoder_free_slots[0], 0)
        self.assertEqual(len(self.server._denoiser_tta), 1)
        self.assertIn("r-no-alt", self.server._transfer_state)
        self.assertIsNotNone(self.server._tracker.get("r-no-alt"))
        p2p = self.server._transfer_state["r-no-alt"]
        self.assertEqual(p2p.transfer_phase, TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT)
        self.assertIn(0, p2p.rejected_instances)
        retry_at = p2p.next_downstream_retry_at

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=retry_at - 0.001,
        ):
            self.server._drain_denoiser_tta()

        self.server._denoiser_pushes[0].send_multipart.assert_not_called()
        self.assertEqual(len(self.server._denoiser_tta), 1)

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=retry_at + 0.001,
        ):
            self.server._drain_denoiser_tta()

        self.server._denoiser_pushes[0].send_multipart.assert_called_once()
        self.assertEqual(len(self.server._denoiser_tta), 0)
        self.assertEqual(p2p.transfer_phase, TransferPhase.WAITING_ALLOC_RESULT)
        self.assertNotIn(0, p2p.rejected_instances)

    def test_retryable_decoder_alloc_reject_single_instance_retries_after_backoff(self):
        self._submit_running_request("r-decoder-retry", RequestState.DECODER_WAITING)
        self.server._tracker.update_instances("r-decoder-retry", decoder_instance=0)
        self.server._decoder_free_slots[0] = 0
        self.server._transfer_state["r-decoder-retry"] = _TransferRequestState(
            sender_role=RoleType.DENOISER.value,
            receiver_role=RoleType.DECODER.value,
            sender_instance=0,
            receiver_instance=0,
            downstream_wait_since=1.0,
        )

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocRejectMsg(
                    request_id="r-decoder-retry",
                    receiver_role=RoleType.DECODER.value,
                    receiver_instance=0,
                    retryable=True,
                    reason="busy",
                )
            ),
            RoleType.DECODER,
        )

        p2p = self.server._transfer_state["r-decoder-retry"]
        retry_at = p2p.next_downstream_retry_at
        self.assertEqual(len(self.server._decoder_tta), 1)
        self.assertIn(0, p2p.rejected_instances)

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=retry_at + 0.001,
        ):
            self.server._drain_decoder_tta()

        self.server._decoder_pushes[0].send_multipart.assert_called_once()
        self.assertEqual(len(self.server._decoder_tta), 0)
        self.assertEqual(p2p.transfer_phase, TransferPhase.WAITING_ALLOC_RESULT)
        self.assertNotIn(0, p2p.rejected_instances)

    def test_decoder_tta_backoff_head_does_not_block_ready_entry(self):
        self._submit_running_request("r-cooling", RequestState.DECODER_WAITING)
        self._submit_running_request("r-ready", RequestState.DECODER_WAITING)
        self.server._decoder_free_slots[0] = 1
        cooling = _TransferRequestState(
            sender_role=RoleType.DENOISER.value,
            sender_instance=0,
            next_downstream_retry_at=100.0,
        )
        ready = _TransferRequestState(
            sender_role=RoleType.DENOISER.value,
            sender_instance=0,
        )
        cooling.rejected_instances[0] = 0
        self.server._transfer_state["r-cooling"] = cooling
        self.server._transfer_state["r-ready"] = ready
        self.server._enqueue_role_wait(self.server._decoder_tta, "r-cooling", cooling)
        self.server._enqueue_role_wait(self.server._decoder_tta, "r-ready", ready)

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=10.0,
        ):
            self.server._drain_decoder_tta()

        self.server._decoder_pushes[0].send_multipart.assert_called_once()
        sent_frames = self.server._decoder_pushes[0].send_multipart.call_args[0][0]
        alloc_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(alloc_msg["request_id"], "r-ready")
        self.assertEqual(len(self.server._decoder_tta), 1)
        self.assertEqual(self.server._decoder_tta[0].request_id, "r-cooling")

    def test_alloc_accepted_stops_downstream_wait_timer(self):
        self._submit_running_request("r-accept", RequestState.DENOISING_WAITING)
        p2p = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            receiver_session_id="den-session",
            downstream_wait_since=123.0,
            downstream_retry_attempts=2,
            next_downstream_retry_at=999.0,
        )
        p2p.rejected_instances[0] = 0
        self.server._transfer_state["r-accept"] = p2p

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocAcceptedMsg(
                    request_id="r-accept",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                    receiver_session_id="den-session",
                    receiver_slot_offset=256,
                    receiver_slot_size=4096,
                    receiver_meta_slot_offset=64,
                    receiver_meta_slot_size=2048,
                )
            ),
            RoleType.DENOISER,
        )

        self.assertTrue(self.server._transfer_state["r-accept"].alloc_accepted)
        self.assertEqual(
            self.server._transfer_state["r-accept"].receiver_slot_offset, 256
        )
        self.assertEqual(
            self.server._transfer_state["r-accept"].receiver_meta_slot_offset, 64
        )
        self.assertIsNone(self.server._transfer_state["r-accept"].downstream_wait_since)
        self.assertEqual(
            self.server._transfer_state["r-accept"].transfer_phase,
            TransferPhase.SENDING,
        )
        self.assertEqual(self.server._transfer_state["r-accept"].rejected_instances, {})
        self.assertEqual(
            self.server._transfer_state["r-accept"].downstream_retry_attempts,
            0,
        )
        self.assertEqual(
            self.server._transfer_state["r-accept"].next_downstream_retry_at,
            0.0,
        )

    def test_stale_alloc_accepted_is_ignored(self):
        self._submit_running_request("r-stale-accept", RequestState.DENOISING_WAITING)
        p2p = _TransferRequestState(
            receiver_role=RoleType.DENOISER.value,
            receiver_instance=0,
            receiver_session_id="new-session",
            downstream_wait_since=123.0,
        )
        self.server._transfer_state["r-stale-accept"] = p2p

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocAcceptedMsg(
                    request_id="r-stale-accept",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                    receiver_session_id="old-session",
                )
            ),
            RoleType.DENOISER,
        )

        self.assertFalse(self.server._transfer_state["r-stale-accept"].alloc_accepted)
        self.assertEqual(
            self.server._transfer_state["r-stale-accept"].transfer_phase,
            TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT,
        )

    def test_alloc_result_waits_for_role_ack_until_downstream_timeout(self):
        self._submit_running_request("r-alloc-timeout", RequestState.DENOISING_WAITING)
        self.server._tracker.update_instances("r-alloc-timeout", denoiser_instance=0)
        self.server._denoiser_free_slots[0] = 0
        self.server._downstream_wait_timeout_s = 100.0
        self.server._transfer_state["r-alloc-timeout"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            receiver_pool_ptr=0x3000,
            receiver_slot_offset=256,
            receiver_slot_size=4096,
            receiver_meta_pool_ptr=0x3800,
            receiver_meta_slot_offset=64,
            receiver_meta_slot_size=2048,
            meta_size=2048,
            transfer_phase=TransferPhase.WAITING_ALLOC_RESULT,
            handoff_started_at=0.0,
            phase_started_at=0.0,
            downstream_wait_since=0.0,
        )
        self.server._denoiser_peers[0] = {
            "free_preallocated_slots": [],
        }

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=10.0,
        ):
            self.server._handle_timeouts()

        self.assertIn("r-alloc-timeout", self.server._transfer_state)
        self.assertEqual(len(self.server._denoiser_tta), 0)
        self.assertEqual(self.server._denoiser_free_slots[0], 0)
        self.assertEqual(
            self.server._transfer_state["r-alloc-timeout"].transfer_phase,
            TransferPhase.WAITING_ALLOC_RESULT,
        )
        self.assertEqual(
            self.server._transfer_state["r-alloc-timeout"].receiver_role,
            RoleType.DENOISER.value,
        )
        self.assertEqual(
            self.server._transfer_state["r-alloc-timeout"].receiver_instance,
            0,
        )

    def test_downstream_wait_timeout_aborts_sender_only_and_times_out(self):
        self._submit_running_request("r-timeout", RequestState.DENOISING_WAITING)
        self.server._pending["r-timeout"] = b"client"
        self.server._frontend = MagicMock()
        self.server._send_abort = MagicMock()
        self.server._downstream_wait_timeout_s = 1.0
        self.server._encoder_free_slots[0] = 0
        self.server._transfer_state["r-timeout"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            sender_instance=0,
            sender_control_endpoint="tcp://enc-ctrl",
            transfer_phase=TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT,
            handoff_started_at=0.0,
            phase_started_at=0.0,
            downstream_wait_since=0.0,
        )

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=10.0,
        ):
            self.server._handle_timeouts()

        self.server._send_abort.assert_called_once()
        _args, kwargs = self.server._send_abort.call_args
        self.assertTrue(kwargs["to_sender"])
        self.assertFalse(kwargs["to_receiver"])
        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.assertIsNone(self.server._tracker.get("r-timeout"))

    def test_denoiser_error_releases_sender_and_receiver_slots(self):
        self._submit_running_request("r-den-error", RequestState.DENOISING_RUNNING)
        self.server._pending["r-den-error"] = b"client"
        self.server._frontend = MagicMock()
        self.server._tracker.update_instances("r-den-error", denoiser_instance=0)
        self.server._encoder_free_slots[0] = 0
        self.server._denoiser_free_slots[0] = 0
        self.server._transfer_state["r-den-error"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_slot_released=False,
            receiver_slot_released=False,
        )

        self.server._handle_transfer_done(
            {"request_id": "r-den-error", "error": "metadata invalid"},
            RoleType.DENOISER,
        )

        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.assertEqual(self.server._denoiser_free_slots[0], 1)
        self.assertIsNone(self.server._tracker.get("r-den-error"))

    def test_denoiser_done_keeps_slot_busy_for_decoder_handoff_and_enqueues_once(self):
        self._submit_running_request("r-done", RequestState.DENOISING_RUNNING)
        self.server._denoiser_peers[0] = {
            "control_endpoint": "tcp://den-ctrl",
            "host_id": "host-a",
            "free_preallocated_slots": [],
        }
        self.server._decoder_peers[0] = {
            "control_endpoint": "tcp://dec-ctrl",
            "host_id": "host-a",
            "free_preallocated_slots": [],
        }
        self.server._encoder_free_slots[0] = 0
        self.server._denoiser_free_slots[0] = 0
        self.server._transfer_state["r-done"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_slot_released=False,
            receiver_pool_ptr=0x3000,
            receiver_slot_offset=256,
            receiver_slot_size=4096,
            receiver_meta_pool_ptr=0x3800,
            receiver_meta_slot_offset=64,
            receiver_meta_slot_size=2048,
            meta_size=2048,
        )

        done_msg = {
            "request_id": "r-done",
            "staged_for_decoder": True,
            "session_id": "den-session",
            "pool_ptr": 0x5000,
            "slot_offset": 128,
            "meta_pool_ptr": 0x5800,
            "meta_slot_offset": 32,
            "data_size": 2048,
            "meta_size": 1024,
        }

        self.server._handle_transfer_done(done_msg, RoleType.DENOISER)
        self.server._handle_transfer_done(done_msg, RoleType.DENOISER)
        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferPushedMsg(
                    request_id="r-done",
                    success=True,
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                )
            ),
            RoleType.ENCODER,
        )

        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.assertEqual(self.server._denoiser_free_slots[0], 0)
        self.assertEqual(self.server._denoiser_peers[0]["free_preallocated_slots"], [])
        self.assertEqual(len(self.server._decoder_tta), 1)
        self.assertEqual(
            self.server._tracker.get("r-done").state,
            RequestState.DECODER_WAITING,
        )
        self.assertFalse(self.server._transfer_state["r-done"].sender_slot_released)

    def test_second_hop_push_releases_denoiser_slot_once(self):
        self._submit_running_request("r-second-push", RequestState.DECODER_WAITING)
        self.server._tracker.update_instances(
            "r-second-push", denoiser_instance=0, decoder_instance=0
        )
        self.server._denoiser_free_slots[0] = 0
        self.server._transfer_state["r-second-push"] = _TransferRequestState(
            sender_role=RoleType.DENOISER.value,
            receiver_role=RoleType.DECODER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_slot_released=False,
            alloc_accepted=True,
        )

        pushed = encode_transfer_msg(
            TransferPushedMsg(request_id="r-second-push", success=True)
        )
        self.server._handle_transfer_result(pushed, RoleType.DENOISER)
        self.server._handle_transfer_result(pushed, RoleType.DENOISER)

        self.assertEqual(self.server._denoiser_free_slots[0], 1)
        self.assertEqual(
            self.server._tracker.get("r-second-push").state,
            RequestState.DECODER_RUNNING,
        )


# Consolidated from test_request_state.py.
class TestRequestState(unittest.TestCase):
    """Test RequestState enum and transitions."""

    def test_all_states_defined(self):
        expected = {
            "PENDING",
            "ENCODER_WAITING",
            "ENCODER_RUNNING",
            "ENCODER_DONE",
            "DENOISING_WAITING",
            "DENOISING_RUNNING",
            "DENOISING_DONE",
            "DECODER_WAITING",
            "DECODER_RUNNING",
            "DONE",
            "FAILED",
            "TIMED_OUT",
        }
        actual = {s.name for s in RequestState}
        self.assertEqual(actual, expected)


class TestRequestTracker(unittest.TestCase):
    """Test RequestTracker lifecycle management."""

    def test_submit_and_get(self):
        tracker = RequestTracker()
        record = tracker.submit("r1")
        self.assertEqual(record.request_id, "r1")
        self.assertEqual(record.state, RequestState.PENDING)
        self.assertFalse(record.is_terminal())
        self.assertIn(RequestState.PENDING.value, record.state_timestamps)

        got = tracker.get("r1")
        self.assertIs(got, record)

    def test_duplicate_submit_raises(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        with self.assertRaises(ValueError):
            tracker.submit("r1")

    def test_full_lifecycle(self):
        tracker = RequestTracker()
        tracker.submit("r1")

        tracker.transition("r1", RequestState.ENCODER_RUNNING, encoder_instance=0)
        self.assertEqual(tracker.get("r1").state, RequestState.ENCODER_RUNNING)
        self.assertEqual(tracker.get("r1").encoder_instance, 0)

        tracker.transition("r1", RequestState.ENCODER_DONE)
        tracker.transition("r1", RequestState.DENOISING_RUNNING, denoiser_instance=1)
        tracker.transition("r1", RequestState.DENOISING_DONE)
        tracker.transition("r1", RequestState.DECODER_RUNNING, decoder_instance=2)
        tracker.transition("r1", RequestState.DONE)

        record = tracker.get("r1")
        self.assertEqual(record.state, RequestState.DONE)
        self.assertTrue(record.is_terminal())
        self.assertEqual(record.encoder_instance, 0)
        self.assertEqual(record.denoiser_instance, 1)
        self.assertEqual(record.decoder_instance, 2)

    def test_invalid_transition_raises(self):
        tracker = RequestTracker()
        tracker.submit("r1")

        # Cannot go from PENDING to DENOISING_RUNNING
        with self.assertRaises(ValueError):
            tracker.transition("r1", RequestState.DENOISING_RUNNING)

    def test_fail_from_any_active_state(self):
        for start_state in [
            RequestState.ENCODER_RUNNING,
            RequestState.ENCODER_DONE,
            RequestState.DENOISING_RUNNING,
        ]:
            tracker = RequestTracker()
            tracker.submit("r1")
            # Walk to start_state
            if start_state == RequestState.ENCODER_RUNNING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
            elif start_state == RequestState.ENCODER_DONE:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
            elif start_state == RequestState.DENOISING_RUNNING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
                tracker.transition("r1", RequestState.DENOISING_RUNNING)

            tracker.transition("r1", RequestState.FAILED, error="test error")
            record = tracker.get("r1")
            self.assertTrue(record.is_terminal())
            self.assertEqual(record.error, "test error")

    def test_timeout_from_active_state(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.transition("r1", RequestState.ENCODER_RUNNING)
        tracker.transition("r1", RequestState.TIMED_OUT)
        self.assertTrue(tracker.get("r1").is_terminal())

    def test_timeout_from_terminal_raises(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.transition("r1", RequestState.ENCODER_RUNNING)
        tracker.transition("r1", RequestState.FAILED)
        with self.assertRaises(ValueError):
            tracker.transition("r1", RequestState.TIMED_OUT)

    def test_unknown_request_raises(self):
        tracker = RequestTracker()
        with self.assertRaises(ValueError):
            tracker.transition("nonexistent", RequestState.ENCODER_RUNNING)

    def test_remove(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        record = tracker.remove("r1")
        self.assertIsNotNone(record)
        self.assertIsNone(tracker.get("r1"))
        self.assertIsNone(tracker.remove("r1"))  # Already removed

    def test_snapshot(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.submit("r2")
        tracker.transition("r1", RequestState.ENCODER_RUNNING)

        snap = tracker.snapshot()
        self.assertEqual(snap["total"], 2)
        self.assertEqual(snap["active"], 2)
        self.assertIn("pending", snap["by_state"])
        self.assertIn("encoder_running", snap["by_state"])

    def test_elapsed(self):
        tracker = RequestTracker()
        record = tracker.submit("r1")
        self.assertGreater(record.elapsed_s(), 0.0)

    def test_waiting_states_lifecycle(self):
        """Test full lifecycle with WAITING states (capacity-aware dispatch)."""
        tracker = RequestTracker()
        tracker.submit("r1")

        # PENDING → ENCODER_WAITING (queued in TTA)
        tracker.transition("r1", RequestState.ENCODER_WAITING)
        self.assertEqual(tracker.get("r1").state, RequestState.ENCODER_WAITING)

        # ENCODER_WAITING → ENCODER_RUNNING (slot available)
        tracker.transition("r1", RequestState.ENCODER_RUNNING, encoder_instance=0)
        self.assertEqual(tracker.get("r1").state, RequestState.ENCODER_RUNNING)

        tracker.transition("r1", RequestState.ENCODER_DONE)

        # ENCODER_DONE → DENOISING_WAITING (all denoisers full)
        tracker.transition("r1", RequestState.DENOISING_WAITING)
        self.assertEqual(tracker.get("r1").state, RequestState.DENOISING_WAITING)

        # DENOISING_WAITING → DENOISING_RUNNING
        tracker.transition("r1", RequestState.DENOISING_RUNNING, denoiser_instance=1)

        tracker.transition("r1", RequestState.DENOISING_DONE)

        # DENOISING_DONE → DECODER_WAITING
        tracker.transition("r1", RequestState.DECODER_WAITING)
        self.assertEqual(tracker.get("r1").state, RequestState.DECODER_WAITING)

        # DECODER_WAITING → DECODER_RUNNING
        tracker.transition("r1", RequestState.DECODER_RUNNING, decoder_instance=0)

        tracker.transition("r1", RequestState.DONE)
        self.assertTrue(tracker.get("r1").is_terminal())

    def test_fail_from_waiting_states(self):
        """WAITING states can transition to FAILED."""
        for waiting_state in [
            RequestState.ENCODER_WAITING,
            RequestState.DENOISING_WAITING,
            RequestState.DECODER_WAITING,
        ]:
            tracker = RequestTracker()
            tracker.submit("r1")

            # Walk to the waiting state
            if waiting_state == RequestState.ENCODER_WAITING:
                tracker.transition("r1", RequestState.ENCODER_WAITING)
            elif waiting_state == RequestState.DENOISING_WAITING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
                tracker.transition("r1", RequestState.DENOISING_WAITING)
            elif waiting_state == RequestState.DECODER_WAITING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
                tracker.transition("r1", RequestState.DENOISING_RUNNING)
                tracker.transition("r1", RequestState.DENOISING_DONE)
                tracker.transition("r1", RequestState.DECODER_WAITING)

            tracker.transition("r1", RequestState.FAILED, error="timeout")
            self.assertTrue(tracker.get("r1").is_terminal())

    def test_skip_waiting_when_capacity_available(self):
        """When capacity is available, skip WAITING and go directly to RUNNING."""
        tracker = RequestTracker()
        tracker.submit("r1")

        # PENDING → ENCODER_RUNNING directly (skip ENCODER_WAITING)
        tracker.transition("r1", RequestState.ENCODER_RUNNING, encoder_instance=0)
        tracker.transition("r1", RequestState.ENCODER_DONE)

        # ENCODER_DONE → DENOISING_RUNNING directly
        tracker.transition("r1", RequestState.DENOISING_RUNNING, denoiser_instance=0)
        tracker.transition("r1", RequestState.DENOISING_DONE)

        # DENOISING_DONE → DECODER_RUNNING directly
        tracker.transition("r1", RequestState.DECODER_RUNNING, decoder_instance=0)
        tracker.transition("r1", RequestState.DONE)

        self.assertTrue(tracker.get("r1").is_terminal())

    def test_timeout_from_waiting_state(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.transition("r1", RequestState.ENCODER_WAITING)
        tracker.transition("r1", RequestState.TIMED_OUT)
        self.assertTrue(tracker.get("r1").is_terminal())

    def test_state_and_event_timestamps_recorded(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.transition("r1", RequestState.ENCODER_WAITING)
        tracker.transition("r1", RequestState.ENCODER_RUNNING)
        tracker.mark_event("r1", "completion_signal_time_s", 123.45)

        record = tracker.get("r1")
        self.assertIn(RequestState.ENCODER_WAITING.value, record.state_timestamps)
        self.assertIn(RequestState.ENCODER_RUNNING.value, record.state_timestamps)
        self.assertEqual(record.event_timestamps["completion_signal_time_s"], 123.45)
        self.assertGreater(record.last_transition_time_s, 0.0)


# Consolidated from test_disagg_metrics.py.
class TestDisaggMetrics(unittest.TestCase):
    """Test DisaggMetrics tracking and snapshot."""

    def test_initial_snapshot(self):
        """Fresh metrics should have zero counts."""
        m = DisaggMetrics(role="encoder")
        s = m.snapshot()
        self.assertEqual(s.role, "encoder")
        self.assertEqual(s.requests_completed, 0)
        self.assertEqual(s.requests_failed, 0)
        self.assertEqual(s.requests_in_flight, 0)
        self.assertEqual(s.requests_timed_out, 0)
        self.assertEqual(s.queue_depth, 0)
        self.assertAlmostEqual(s.throughput_rps, 0.0)
        self.assertGreater(s.uptime_s, 0.0)

    def test_request_lifecycle(self):
        """Start -> complete should increment counts and track latency."""
        m = DisaggMetrics(role="denoising")
        m.record_request_start("req-001")

        s = m.snapshot()
        self.assertEqual(s.requests_in_flight, 1)
        self.assertEqual(s.requests_completed, 0)

        time.sleep(0.05)
        m.record_request_complete("req-001")

        s = m.snapshot()
        self.assertEqual(s.requests_in_flight, 0)
        self.assertEqual(s.requests_completed, 1)
        self.assertGreater(s.last_latency_s, 0.04)
        self.assertGreater(s.avg_latency_s, 0.04)
        self.assertGreater(s.max_latency_s, 0.04)

    def test_multiple_requests(self):
        """Track multiple concurrent requests."""
        m = DisaggMetrics(role="decoder")

        m.record_request_start("r1")
        m.record_request_start("r2")
        m.record_request_start("r3")
        self.assertEqual(m.snapshot().requests_in_flight, 3)

        m.record_request_complete("r1")
        m.record_request_complete("r2")
        self.assertEqual(m.snapshot().requests_in_flight, 1)
        self.assertEqual(m.snapshot().requests_completed, 2)

        m.record_request_failed("r3")
        self.assertEqual(m.snapshot().requests_in_flight, 0)
        self.assertEqual(m.snapshot().requests_failed, 1)

    def test_timeout_tracking(self):
        m = DisaggMetrics(role="encoder")
        m.record_request_start("r1")
        m.record_request_timeout("r1")

        s = m.snapshot()
        self.assertEqual(s.requests_timed_out, 1)
        self.assertEqual(s.requests_in_flight, 0)

    def test_queue_depth(self):
        m = DisaggMetrics(role="encoder")
        m.update_queue_depth(5)
        self.assertEqual(m.snapshot().queue_depth, 5)
        m.update_queue_depth(0)
        self.assertEqual(m.snapshot().queue_depth, 0)

    def test_throughput(self):
        """Throughput should reflect completions within the window."""
        m = DisaggMetrics(role="denoising")
        # Complete 5 requests
        for i in range(5):
            m.record_request_start(f"r{i}")
            m.record_request_complete(f"r{i}")

        s = m.snapshot()
        self.assertEqual(s.requests_completed, 5)
        # Should have positive throughput (5 completions in last 60s window)
        self.assertGreater(s.throughput_rps, 0.0)

    def test_to_dict(self):
        """Snapshot should serialize to a dict with all expected keys."""
        m = DisaggMetrics(role="encoder")
        m.record_request_start("r1")
        m.record_request_complete("r1")

        d = m.snapshot().to_dict()
        expected_keys = {
            "role",
            "requests_completed",
            "requests_failed",
            "requests_in_flight",
            "requests_timed_out",
            "queue_depth",
            "last_latency_s",
            "avg_latency_s",
            "max_latency_s",
            "throughput_rps",
            "uptime_s",
        }
        self.assertEqual(set(d.keys()), expected_keys)
        self.assertEqual(d["role"], "encoder")
        self.assertEqual(d["requests_completed"], 1)

    def test_max_latency_tracks_worst_case(self):
        m = DisaggMetrics(role="encoder")

        m.record_request_start("fast")
        m.record_request_complete("fast")
        fast_latency = m.snapshot().max_latency_s

        m.record_request_start("slow")
        time.sleep(0.1)
        m.record_request_complete("slow")

        s = m.snapshot()
        self.assertGreater(s.max_latency_s, fast_latency)
        self.assertGreater(s.max_latency_s, 0.09)


# Consolidated from test/multimodal_gen/test_disagg_control_plane_unittest.py.
class _FakeFrontend:
    def __init__(self):
        self.sent = []

    def send_multipart(self, frames):
        self.sent.append(frames)


class _EncoderLoopDummy(SchedulerDisaggMixin):
    def __init__(self, *, rank0: bool, messages=None):
        self.server_args = SimpleNamespace(
            sp_degree=1,
            tp_size=2,
            enable_cfg_parallel=False,
        )
        self.gpu_id = 0 if rank0 else 1
        self._disagg_role = RoleType.ENCODER
        self._running = True
        self._consecutive_error_count = 0
        self._max_consecutive_errors = 3
        self.broadcasts = []
        self.steps = []
        self.cleaned = False
        self._messages = list(messages or [])
        self._pool_work_pull = object()

    def _process_outbound_staging_retry_once(self):
        return False

    def _process_swap_out_queue_once(self):
        return False

    def _process_send_ready_queue_once(self):
        return False

    def _maybe_apply_pending_transfer_reconfigure(self):
        return False

    def _has_pending_outbound_staging_retry(self):
        return False

    def _try_recv_work_noblock(self):
        return None

    def _broadcast_to_all_ranks(self, data):
        if self.gpu_id == 0:
            self.broadcasts.append(data)
            if _is_skip_broadcast(data):
                self._running = False
            return data
        msg = self._messages.pop(0)
        self.broadcasts.append(msg)
        return msg

    def _disagg_encoder_step(self, send_tensors_fn, frames):
        del send_tensors_fn
        self.steps.append(frames)

    def _cleanup_disagg(self):
        self.cleaned = True


def _make_diffusion_server(*, timeout_s=600.0, downstream_wait_timeout_s=120.0):
    server = DiffusionServer(
        frontend_endpoint="inproc://frontend-test",
        encoder_work_endpoints=["inproc://encoder-work"],
        denoiser_work_endpoints=["inproc://denoiser-work"],
        decoder_work_endpoints=["inproc://decoder-work"],
        encoder_result_endpoint="inproc://encoder-result",
        denoiser_result_endpoint="inproc://denoiser-result",
        decoder_result_endpoint="inproc://decoder-result",
        timeout_s=timeout_s,
        downstream_wait_timeout_s=downstream_wait_timeout_s,
        max_slots_per_instance=1,
    )
    server._frontend = _FakeFrontend()
    return server


class TestDisaggControlPlane(unittest.TestCase):
    def test_transfer_register_round_trip_preserves_direct_connect_fields(self):
        msg = TransferRegisterMsg(
            role="encoder",
            instance_id=7,
            session_id="sess-1",
            pool_ptr=1234,
            pool_size=4096,
            control_endpoint="tcp://10.0.0.1:9001",
            work_endpoint="tcp://10.0.0.1:9000",
            rank0_only=True,
            role_device="cpu",
            preallocated_slots=[
                {
                    "offset": 512,
                    "size": 2048,
                    "slot_id": 3,
                    "addr": 1746,
                }
            ],
        )

        frames = encode_transfer_msg(msg)
        decoded = decode_transfer_msg(frames)

        self.assertTrue(is_transfer_message(frames))
        self.assertEqual(decoded["role"], "encoder")
        self.assertEqual(decoded["instance_id"], 7)
        self.assertEqual(decoded["control_endpoint"], "tcp://10.0.0.1:9001")
        self.assertEqual(decoded["work_endpoint"], "tcp://10.0.0.1:9000")
        self.assertTrue(decoded["rank0_only"])
        self.assertEqual(decoded["role_device"], "cpu")
        self.assertEqual(decoded["preallocated_slots"][0]["slot_id"], 3)

    def test_transfer_peer_info_round_trip_preserves_receiver_fields(self):
        msg = TransferPeerInfoMsg(
            request_id="req-1",
            dest_session_id="dst-session",
            dest_addr=987654,
            transfer_size=8192,
            receiver_role="decoder",
            receiver_instance=2,
            receiver_control_endpoint="tcp://10.0.0.3:9101",
            prealloc_slot_id=5,
        )

        decoded = decode_transfer_msg(encode_transfer_msg(msg))

        self.assertEqual(decoded["request_id"], "req-1")
        self.assertEqual(decoded["transfer_size"], 8192)
        self.assertEqual(decoded["receiver_role"], "decoder")
        self.assertEqual(decoded["receiver_instance"], 2)
        self.assertEqual(
            decoded["receiver_control_endpoint"],
            "tcp://10.0.0.3:9101",
        )
        self.assertEqual(decoded["prealloc_slot_id"], 5)

    def test_server_args_resolve_control_endpoint_and_role_device(self):
        args = object.__new__(ServerArgs)
        args.scheduler_port = 31000
        args.host = "0.0.0.0"
        args.disagg_p2p_hostname = "10.1.2.3"
        args.num_gpus = 0
        args.disagg_role_device = "auto"

        self.assertEqual(
            args.derive_pool_control_endpoint(),
            "tcp://0.0.0.0:31001",
        )
        self.assertEqual(
            args.derive_pool_control_advertised_endpoint(),
            "tcp://10.1.2.3:31001",
        )
        self.assertEqual(args.resolved_role_device(), "cpu")

        args.num_gpus = 4
        self.assertEqual(args.resolved_role_device(), "cuda")

        args.disagg_role_device = "cpu"
        self.assertEqual(args.resolved_role_device(), "cpu")

    def test_max_free_slots_policy_uses_explicit_capacity(self):
        policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=4)
        self.assertEqual(policy.select(active_counts=[4, 1, 2]), 1)
        self.assertIsNone(policy.select_with_capacity([0, 0, 0]))

        dispatcher = PoolDispatcher(
            num_encoders=1,
            num_denoisers=3,
            num_decoders=1,
            policy_name="max_free_slots",
            max_slots_per_instance=4,
        )
        self.assertEqual(dispatcher.select_denoiser(active_counts=[4, 1, 2]), 1)

    def test_encoder_module_filtering_requires_decoder_opt_in(self):
        module_names = ["text_encoder", "vae", "transformer", "scheduler"]

        filtered_default = filter_modules_for_role(module_names, RoleType.ENCODER)
        filtered_with_decoder = filter_modules_for_role(
            module_names,
            RoleType.ENCODER,
            extra_allowed_modules={"vae"},
        )

        self.assertEqual(filtered_default, ["text_encoder", "scheduler"])
        self.assertEqual(
            filtered_with_decoder,
            ["text_encoder", "vae", "scheduler"],
        )

    def test_encoder_idle_skip_helpers(self):
        self.assertTrue(_is_skip_broadcast(("skip",)))
        self.assertFalse(_is_skip_broadcast(("encoder_work", [])))
        self.assertFalse(_should_broadcast_encoder_idle_skip(1.0, 1.005))
        self.assertTrue(_should_broadcast_encoder_idle_skip(1.0, 1.011))

    def test_encoder_rank0_broadcasts_idle_skip(self):
        dummy = _EncoderLoopDummy(rank0=True)

        dummy._disagg_encoder_rank0_event_loop()

        self.assertIn(("skip",), dummy.broadcasts)
        self.assertIsNone(dummy.broadcasts[-1])
        self.assertTrue(dummy.cleaned)

    def test_encoder_follower_ignores_idle_skip(self):
        dummy = _EncoderLoopDummy(rank0=False, messages=[("skip",), None])

        dummy._disagg_encoder_non_rank0_event_loop()

        self.assertEqual(dummy.steps, [])
        self.assertTrue(dummy.cleaned)

    def test_denoiser_done_without_decoder_payload_fails_request(self):
        server = _make_diffusion_server()
        request_id = "req-missing-decoder-payload"
        server._pending[request_id] = b"client"
        server._tracker.submit(request_id)
        server._tracker.transition(
            request_id, RequestState.ENCODER_RUNNING, encoder_instance=0
        )
        server._tracker.transition(request_id, RequestState.ENCODER_DONE)
        server._tracker.transition(request_id, RequestState.DENOISING_WAITING)
        server._tracker.transition(
            request_id, RequestState.DENOISING_RUNNING, denoiser_instance=0
        )
        server._encoder_free_slots[0] = 0
        server._denoiser_free_slots[0] = 0
        server._transfer_state[request_id] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            sender_instance=0,
            sender_slot_released=True,
            receiver_role=RoleType.DENOISER.value,
            receiver_instance=0,
            transfer_phase=TransferPhase.RUNNING_DOWNSTREAM,
        )

        server._handle_transfer_done(
            {"request_id": request_id, "staged_for_decoder": False},
            RoleType.DENOISER,
        )

        self.assertIsNone(server._tracker.get(request_id))
        self.assertNotIn(request_id, server._pending)
        self.assertNotIn(request_id, server._transfer_state)
        self.assertEqual(server._denoiser_free_slots[0], 1)
        self.assertEqual(len(server._frontend.sent), 1)

    def test_global_timeout_cleans_transfer_state_and_returns_error(self):
        server = _make_diffusion_server(timeout_s=0.1)
        request_id = "req-global-timeout"
        server._pending[request_id] = b"client"
        record = server._tracker.submit(request_id)
        server._tracker.transition(
            request_id, RequestState.ENCODER_RUNNING, encoder_instance=0
        )
        server._tracker.transition(request_id, RequestState.ENCODER_DONE)
        server._tracker.transition(request_id, RequestState.DENOISING_WAITING)
        server._tracker.transition(
            request_id, RequestState.DENOISING_RUNNING, denoiser_instance=0
        )
        record.submit_time -= 1.0
        server._encoder_free_slots[0] = 0
        server._denoiser_free_slots[0] = 0
        server._transfer_state[request_id] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            sender_instance=0,
            sender_slot_released=True,
            receiver_role=RoleType.DENOISER.value,
            receiver_instance=0,
            transfer_phase=TransferPhase.RUNNING_DOWNSTREAM,
        )

        server._handle_timeouts()

        self.assertIsNone(server._tracker.get(request_id))
        self.assertNotIn(request_id, server._pending)
        self.assertNotIn(request_id, server._transfer_state)
        self.assertEqual(server._denoiser_free_slots[0], 1)
        self.assertEqual(len(server._frontend.sent), 1)


if __name__ == "__main__":
    unittest.main()
