# SPDX-License-Identifier: Apache-2.0
"""Central request router for disaggregated diffusion pipelines."""

import json
import logging
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass

import zmq

from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    PoolDispatcher,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    unpack_tensors,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocMsg,
    TransferMsgType,
    TransferPushMsg,
    TransferReadyMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket

logger = logging.getLogger(__name__)


@dataclass
class _EncoderTTAEntry:
    request_id: str
    client_identity: bytes
    payload: bytes


@dataclass
class _TransferRequestState:
    sender_session_id: str = ""
    sender_pool_ptr: int = 0
    sender_slot_offset: int = 0
    data_size: int = 0
    manifest: dict = None
    scalar_fields: dict = None
    receiver_session_id: str = ""
    receiver_pool_ptr: int = 0
    receiver_slot_offset: int = 0
    sender_instance: int = -1
    receiver_instance: int = -1
    prealloc_slot_id: int | None = None

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.scalar_fields is None:
            self.scalar_fields = {}


@dataclass
class _RoleTTAEntry:
    request_id: str
    transfer_state: _TransferRequestState | None = None


class DiffusionServer:
    """Global pipeline orchestrator for N:M:K disaggregated diffusion.

    Capacity-aware dispatch with FreeBufferSlots per instance and TTA queues.
    """

    def __init__(
        self,
        frontend_endpoint: str,
        encoder_work_endpoints: list[str],
        denoiser_work_endpoints: list[str],
        decoder_work_endpoints: list[str],
        encoder_result_endpoint: str,
        denoiser_result_endpoint: str,
        decoder_result_endpoint: str,
        dispatch_policy_name: str = "round_robin",
        timeout_s: float = 600.0,
        encoder_capacity: int = 4,
        denoiser_capacity: int = 2,
        decoder_capacity: int = 4,
        p2p_mode: bool = True,
    ):
        self._frontend_endpoint = frontend_endpoint
        self._encoder_work_endpoints = encoder_work_endpoints
        self._denoiser_work_endpoints = denoiser_work_endpoints
        self._decoder_work_endpoints = decoder_work_endpoints
        self._encoder_result_endpoint = encoder_result_endpoint
        self._denoiser_result_endpoint = denoiser_result_endpoint
        self._decoder_result_endpoint = decoder_result_endpoint

        self._num_encoders = len(encoder_work_endpoints)
        self._num_denoisers = len(denoiser_work_endpoints)
        self._num_decoders = len(decoder_work_endpoints)
        self._timeout_s = timeout_s

        self._tracker = RequestTracker()
        self._dispatcher = PoolDispatcher(
            num_encoders=self._num_encoders,
            num_denoisers=self._num_denoisers,
            num_decoders=self._num_decoders,
            policy_name=dispatch_policy_name,
        )

        self._context = zmq.Context(io_threads=2)
        self._running = False
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

        self._pending: dict[str, bytes] = {}  # request_id -> client ZMQ identity
        self._lock = threading.Lock()

        # FreeBufferSlots per instance
        self._encoder_free_slots = [encoder_capacity] * self._num_encoders
        self._denoiser_free_slots = [denoiser_capacity] * self._num_denoisers
        self._decoder_free_slots = [decoder_capacity] * self._num_decoders

        # TTA queues per role type
        self._encoder_tta: deque[_EncoderTTAEntry] = deque()
        self._denoiser_tta: deque[_RoleTTAEntry] = deque()
        self._decoder_tta: deque[_RoleTTAEntry] = deque()

        self._transfer_mode = p2p_mode
        self._transfer_state: dict[str, _TransferRequestState] = {}

        # Per-instance registration: instance_idx -> {session_id, pool_ptr, pool_size}
        # Keyed by the same index used to build the PUSH work-socket list
        # (i.e. the index into --encoder/denoiser/decoder-urls). The index is
        # resolved from the registering instance's work_endpoint so the control
        # plane (work PUSH) and the data plane (RDMA session_id / pool_ptr /
        # preallocated slots) stay consistent regardless of startup order.
        self._encoder_peers: dict[int, dict] = {}
        self._denoiser_peers: dict[int, dict] = {}
        self._decoder_peers: dict[int, dict] = {}

        # work_endpoint -> index lookup tables, built from the --*-urls args
        self._encoder_endpoint_to_idx = {
            ep: i for i, ep in enumerate(encoder_work_endpoints)
        }
        self._denoiser_endpoint_to_idx = {
            ep: i for i, ep in enumerate(denoiser_work_endpoints)
        }
        self._decoder_endpoint_to_idx = {
            ep: i for i, ep in enumerate(decoder_work_endpoints)
        }

    @property
    def tracker(self) -> RequestTracker:
        return self._tracker

    @property
    def dispatcher(self) -> PoolDispatcher:
        return self._dispatcher

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            name="DiffusionServer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "DiffusionServer started: frontend=%s, "
            "%d encoder(s), %d denoiser(s), %d decoder(s), policy=%s, "
            "capacity=(%d/%d/%d)",
            self._frontend_endpoint,
            self._num_encoders,
            self._num_denoisers,
            self._num_decoders,
            type(self._dispatcher.encoder_policy).__name__,
            self._encoder_free_slots[0] if self._encoder_free_slots else 0,
            self._denoiser_free_slots[0] if self._denoiser_free_slots else 0,
            self._decoder_free_slots[0] if self._decoder_free_slots else 0,
        )

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until the event loop has bound all sockets, or *timeout* elapses."""
        return self._ready.wait(timeout=timeout)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _event_loop(self) -> None:
        frontend, _ = get_zmq_socket(
            self._context, zmq.ROUTER, self._frontend_endpoint, bind=True
        )

        encoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._encoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            encoder_pushes.append(sock)

        denoiser_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._denoiser_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            denoiser_pushes.append(sock)

        decoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._decoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            decoder_pushes.append(sock)

        encoder_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._encoder_result_endpoint, bind=True
        )
        denoiser_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._denoiser_result_endpoint, bind=True
        )
        decoder_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._decoder_result_endpoint, bind=True
        )

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(encoder_result_pull, zmq.POLLIN)
        poller.register(denoiser_result_pull, zmq.POLLIN)
        poller.register(decoder_result_pull, zmq.POLLIN)

        self._encoder_pushes = encoder_pushes
        self._denoiser_pushes = denoiser_pushes
        self._decoder_pushes = decoder_pushes
        self._frontend = frontend

        self._ready.set()

        all_sockets = (
            [frontend, encoder_result_pull, denoiser_result_pull, decoder_result_pull]
            + encoder_pushes
            + denoiser_pushes
            + decoder_pushes
        )

        try:
            while self._running:
                events = dict(poller.poll(timeout=10))

                self._handle_timeouts()

                if frontend in events:
                    self._handle_client_request(frontend)

                if encoder_result_pull in events:
                    self._handle_role_result(encoder_result_pull, RoleType.ENCODER)

                if denoiser_result_pull in events:
                    self._handle_role_result(denoiser_result_pull, RoleType.DENOISER)

                if decoder_result_pull in events:
                    self._handle_role_result(decoder_result_pull, RoleType.DECODER)

                self._drain_all_queues()

        except Exception:
            logger.exception("DiffusionServer event loop error")
        finally:
            for sock in all_sockets:
                sock.close()
            self._context.destroy(linger=0)

    def _handle_role_result(self, result_pull: zmq.Socket, role: RoleType) -> None:
        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        if is_transfer_message(frames):
            self._handle_transfer_result(frames, role)
            return

        if role == RoleType.DECODER:
            self._handle_decoder_result_frames(frames)
        else:
            # Non-transfer frames from encoder/denoiser are error results
            # sent via send_tensors (e.g., _disagg_error).
            self._handle_role_error_frames(frames, role)

    def _handle_role_error_frames(self, frames: list, role: RoleType) -> None:
        """Handle non-transfer error results from encoder/denoiser roles."""
        try:
            tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")
        except Exception as e:
            logger.warning(
                "DiffusionServer: failed to unpack non-transfer frames from %s: %s",
                role.value,
                e,
            )
            return

        request_id = scalar_fields.get("request_id")
        disagg_error = scalar_fields.get("_disagg_error")

        if request_id and disagg_error:
            logger.error(
                "DiffusionServer: %s error for %s: %s",
                role.value,
                request_id,
                disagg_error,
            )
            self._complete_with_error(request_id, f"{role.value} error: {disagg_error}")
        elif request_id:
            logger.warning(
                "DiffusionServer: non-transfer frames from %s for %s without error",
                role.value,
                request_id,
            )
        else:
            logger.warning(
                "DiffusionServer: non-transfer frames from %s without request_id",
                role.value,
            )

    def _handle_client_request(self, frontend: zmq.Socket) -> None:
        try:
            parts = frontend.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(parts) < 3:
            return

        client_identity = parts[0]
        payload = parts[-1]

        try:
            reqs = pickle.loads(payload)
        except (pickle.UnpicklingError, EOFError):
            logger.warning("DiffusionServer: failed to deserialize request")
            return

        if not isinstance(reqs, list):
            reqs = [reqs]

        req = reqs[0]

        if isinstance(req, dict) or not hasattr(req, "request_id"):
            # Send empty reply so REQ socket doesn't hang
            try:
                frontend.send_multipart(
                    [client_identity, b"", pickle.dumps({"status": "ignored"})],
                    zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            return

        request_id = getattr(req, "request_id", None)
        if request_id is None:
            request_id = f"ds-{time.monotonic()}"

        try:
            self._tracker.submit(request_id)
        except ValueError:
            logger.warning("DiffusionServer: duplicate request_id %s", request_id)
            return

        with self._lock:
            self._pending[request_id] = client_identity

        try:
            self._tracker.transition(request_id, RequestState.ENCODER_WAITING)
        except ValueError:
            pass
        self._encoder_tta.append(
            _EncoderTTAEntry(
                request_id=request_id,
                client_identity=client_identity,
                payload=payload,
            )
        )
        logger.debug(
            "DiffusionServer: queued %s to encoder_tta",
            request_id,
        )

    def _handle_decoder_result_frames(self, frames: list) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: decoder result missing request_id")
            return

        logger.debug("DiffusionServer: decoder result %s", request_id)
        record = self._tracker.get(request_id)
        if record and record.decoder_instance is not None:
            self._decoder_free_slots[record.decoder_instance] += 1

        tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")

        output_batch = OutputBatch(
            output=tensor_fields.get("output"),
            audio=tensor_fields.get("audio"),
            audio_sample_rate=scalar_fields.get("audio_sample_rate"),
            error=scalar_fields.get("error"),
        )

        try:
            if output_batch.error:
                self._tracker.transition(
                    request_id, RequestState.FAILED, error=output_batch.error
                )
            else:
                self._tracker.transition(request_id, RequestState.DONE)
        except ValueError:
            pass

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            logger.warning(
                "DiffusionServer: no pending client for decoder result %s",
                request_id,
            )
            self._tracker.remove(request_id)
            return

        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send result for %s: %s",
                request_id,
                e,
            )

        logger.debug("DiffusionServer: returned result for %s", request_id)
        self._transfer_state.pop(request_id, None)
        self._tracker.remove(request_id)

    def _dispatch_to_encoder(
        self, request_id: str, payload: bytes, encoder_idx: int
    ) -> None:
        self._encoder_free_slots[encoder_idx] -= 1

        try:
            self._tracker.transition(
                request_id,
                RequestState.ENCODER_RUNNING,
                encoder_instance=encoder_idx,
            )
        except ValueError:
            pass

        self._encoder_pushes[encoder_idx].send_multipart(
            [request_id.encode("utf-8"), payload]
        )
        logger.debug(
            "DiffusionServer: dispatched %s to encoder[%d] (free=%d)",
            request_id,
            encoder_idx,
            self._encoder_free_slots[encoder_idx],
        )

    def _drain_all_queues(self) -> None:
        self._drain_encoder_tta()
        self._drain_denoiser_tta()
        self._drain_decoder_tta()

    def _drain_encoder_tta(self) -> None:
        while self._encoder_tta:
            idx = self._dispatcher.select_encoder_with_capacity(
                self._encoder_free_slots
            )
            if idx is None:
                break
            entry = self._encoder_tta.popleft()
            self._dispatch_to_encoder(entry.request_id, entry.payload, idx)

    def _drain_denoiser_tta(self) -> None:
        while self._denoiser_tta:
            idx = self._dispatcher.select_denoiser_with_capacity(
                self._denoiser_free_slots
            )
            if idx is None:
                break
            entry = self._denoiser_tta.popleft()
            self._transfer_dispatch_to_denoiser(
                entry.request_id, entry.transfer_state, idx
            )

    def _drain_decoder_tta(self) -> None:
        while self._decoder_tta:
            idx = self._dispatcher.select_decoder_with_capacity(
                self._decoder_free_slots
            )
            if idx is None:
                break
            entry = self._decoder_tta.popleft()
            self._transfer_dispatch_to_decoder(
                entry.request_id, entry.transfer_state, idx
            )

    def _extract_request_id(self, frames: list) -> str | None:
        try:
            metadata = json.loads(frames[0])
            return metadata.get("scalar_fields", {}).get("request_id")
        except (json.JSONDecodeError, IndexError, TypeError):
            return None

    def _complete_with_error(self, request_id: str, error_msg: str) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        logger.error("DiffusionServer: %s — %s", request_id, error_msg)

        try:
            self._tracker.transition(request_id, RequestState.FAILED, error=error_msg)
        except ValueError:
            pass

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._tracker.remove(request_id)
            return

        error_batch = OutputBatch(error=error_msg)
        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(error_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send error for %s: %s",
                request_id,
                e,
            )

        self._tracker.remove(request_id)

    def _handle_timeouts(self) -> None:
        timed_out = self._tracker.find_timed_out(self._timeout_s)
        for request_id in timed_out:
            # Free the slot for the timed-out request
            record = self._tracker.get(request_id)
            if record:
                self._free_slot_for_record(record)

            self._complete_with_error(
                request_id,
                f"DiffusionServer timeout: request {request_id} "
                f"not completed within {self._timeout_s}s",
            )

        if timed_out:
            timed_set = set(timed_out)
            self._encoder_tta = deque(
                e for e in self._encoder_tta if e.request_id not in timed_set
            )
            self._denoiser_tta = deque(
                e for e in self._denoiser_tta if e.request_id not in timed_set
            )
            self._decoder_tta = deque(
                e for e in self._decoder_tta if e.request_id not in timed_set
            )

    def _free_slot_for_record(self, record) -> None:
        if (
            record.state in (RequestState.ENCODER_RUNNING, RequestState.ENCODER_DONE)
            and record.encoder_instance is not None
        ):
            self._encoder_free_slots[record.encoder_instance] += 1
        if (
            record.state
            in (RequestState.DENOISING_RUNNING, RequestState.DENOISING_DONE)
            and record.denoiser_instance is not None
        ):
            self._denoiser_free_slots[record.denoiser_instance] += 1
        if (
            record.state == RequestState.DECODER_RUNNING
            and record.decoder_instance is not None
        ):
            self._decoder_free_slots[record.decoder_instance] += 1

    def _handle_transfer_result(self, frames: list, role: RoleType) -> None:
        try:
            msg = decode_transfer_msg(frames)
        except (ValueError, Exception) as e:
            logger.error("DiffusionServer: failed to decode transfer message: %s", e)
            return

        msg_type = msg.get("msg_type")

        if msg_type == TransferMsgType.REGISTER:
            self._handle_transfer_register(msg)
        elif msg_type == TransferMsgType.STAGED:
            self._handle_transfer_staged(msg)
        elif msg_type == TransferMsgType.ALLOCATED:
            self._handle_transfer_allocated(msg)
        elif msg_type == TransferMsgType.PUSHED:
            self._handle_transfer_pushed(msg)
        elif msg_type == TransferMsgType.DONE:
            self._handle_transfer_done(msg, role)
        else:
            logger.warning("DiffusionServer: unknown transfer msg_type=%s", msg_type)

    def _handle_transfer_register(self, msg: dict) -> None:
        try:
            role = RoleType.from_string(msg.get("role", ""))
        except ValueError:
            logger.warning(
                "DiffusionServer transfer: unknown role in register: %s",
                msg.get("role"),
            )
            return

        work_endpoint = msg.get("work_endpoint", "")
        if role == RoleType.ENCODER:
            endpoint_to_idx = self._encoder_endpoint_to_idx
            peers = self._encoder_peers
        elif role == RoleType.DENOISER:
            endpoint_to_idx = self._denoiser_endpoint_to_idx
            peers = self._denoiser_peers
        elif role == RoleType.DECODER:
            endpoint_to_idx = self._decoder_endpoint_to_idx
            peers = self._decoder_peers
        else:
            logger.warning(
                "DiffusionServer transfer: unsupported role in register: %s", role
            )
            return

        idx = endpoint_to_idx.get(work_endpoint)
        if idx is None:
            # Fail loudly: without a URL match, the control plane (work PUSH)
            # and data plane (RDMA dest) would drift silently.
            logger.error(
                "DiffusionServer transfer: register for role=%s with unknown "
                "work_endpoint=%r (known=%s); dropping registration",
                role.value,
                work_endpoint,
                list(endpoint_to_idx.keys()),
            )
            return

        info = {
            "session_id": msg.get("session_id", ""),
            "pool_ptr": msg.get("pool_ptr", 0),
            "pool_size": msg.get("pool_size", 0),
            "work_endpoint": work_endpoint,
        }
        prealloc = msg.get("preallocated_slots", [])
        info["free_preallocated_slots"] = list(prealloc)
        peers[idx] = info

        logger.info(
            "DiffusionServer transfer: registered %s[%d] work_endpoint=%s "
            "session=%s pool_ptr=%#x prealloc=%d",
            role,
            idx,
            work_endpoint,
            info["session_id"],
            info["pool_ptr"],
            len(prealloc),
        )

    def _handle_transfer_staged(self, msg: dict) -> None:
        request_id = msg["request_id"]
        logger.debug("DiffusionServer transfer: encoder staged %s", request_id)
        record = self._tracker.get(request_id)
        encoder_idx = record.encoder_instance if record else 0

        p2p = _TransferRequestState(
            sender_session_id=msg.get("session_id", ""),
            sender_pool_ptr=msg.get("pool_ptr", 0),
            sender_slot_offset=msg.get("slot_offset", 0),
            data_size=msg.get("data_size", 0),
            manifest=msg.get("manifest", {}),
            scalar_fields=msg.get("scalar_fields", {}),
            sender_instance=encoder_idx,
        )
        self._transfer_state[request_id] = p2p

        # Encoder slot freed later in _handle_transfer_pushed after RDMA completes
        try:
            self._tracker.transition(request_id, RequestState.ENCODER_DONE)
        except ValueError:
            pass

        try:
            self._tracker.transition(request_id, RequestState.DENOISING_WAITING)
        except ValueError:
            pass
        self._denoiser_tta.append(
            _RoleTTAEntry(request_id=request_id, transfer_state=p2p)
        )

    def _try_fast_path_push(
        self,
        request_id: str,
        p2p: _TransferRequestState,
        receiver_peer_info: dict,
        sender_pushes: list,
        receiver_role_label: str,
        receiver_idx: int,
    ) -> bool:
        """Try to dispatch via a pre-allocated receive slot (fast path).

        If the receiver already registered a free prealloc slot large enough
        for this transfer, claim it and send a ``TransferPushMsg`` directly
        to the sender so RDMA can start immediately. Returns True when the
        fast path is used; False when the caller must fall back to the
        round-trip alloc path.
        """
        free_slots = receiver_peer_info.get("free_preallocated_slots", [])
        if not (free_slots and free_slots[0].get("size", 0) >= p2p.data_size):
            return False

        slot_info = free_slots.pop(0)
        p2p.receiver_session_id = receiver_peer_info.get("session_id", "")
        p2p.receiver_pool_ptr = receiver_peer_info.get("pool_ptr", 0)
        p2p.receiver_slot_offset = slot_info["offset"]
        p2p.prealloc_slot_id = slot_info.get("slot_id")

        push_msg = TransferPushMsg(
            request_id=request_id,
            dest_session_id=p2p.receiver_session_id,
            dest_addr=slot_info["addr"],
            transfer_size=p2p.data_size,
        )
        sender_pushes[p2p.sender_instance].send_multipart(encode_transfer_msg(push_msg))
        logger.debug(
            "DiffusionServer transfer: fast-path push to %s[%d] for %s "
            "(prealloc slot %s, %d bytes)",
            receiver_role_label,
            receiver_idx,
            request_id,
            slot_info.get("slot_id"),
            p2p.data_size,
        )
        return True

    def _send_slow_path_alloc(
        self,
        request_id: str,
        p2p: _TransferRequestState,
        receiver_pushes: list,
        receiver_idx: int,
        source_role: str,
    ) -> None:
        """Ask the receiver to allocate a slot (slow path).

        Used when the receiver has no free prealloc slot large enough. The
        receiver will respond with ``transfer_allocated``; see
        :meth:`_handle_transfer_allocated`.
        """
        alloc_msg = TransferAllocMsg(
            request_id=request_id,
            data_size=p2p.data_size,
            source_role=source_role,
        )
        receiver_pushes[receiver_idx].send_multipart(encode_transfer_msg(alloc_msg))

    def _transfer_dispatch_to_denoiser(
        self, request_id: str, p2p: _TransferRequestState, denoiser_idx: int
    ) -> None:
        self._denoiser_free_slots[denoiser_idx] -= 1
        p2p.receiver_instance = denoiser_idx

        try:
            self._tracker.transition(
                request_id,
                RequestState.DENOISING_RUNNING,
                denoiser_instance=denoiser_idx,
            )
        except ValueError:
            pass

        peer_info = self._denoiser_peers.get(denoiser_idx, {})
        if not self._try_fast_path_push(
            request_id=request_id,
            p2p=p2p,
            receiver_peer_info=peer_info,
            sender_pushes=self._encoder_pushes,
            receiver_role_label="denoiser",
            receiver_idx=denoiser_idx,
        ):
            self._send_slow_path_alloc(
                request_id=request_id,
                p2p=p2p,
                receiver_pushes=self._denoiser_pushes,
                receiver_idx=denoiser_idx,
                source_role="encoder",
            )

    def _handle_transfer_allocated(self, msg: dict) -> None:
        request_id = msg["request_id"]
        p2p = self._transfer_state.get(request_id)
        if p2p is None:
            logger.warning(
                "DiffusionServer transfer: no state for allocated %s", request_id
            )
            return

        p2p.receiver_session_id = msg.get("session_id", "")
        p2p.receiver_pool_ptr = msg.get("pool_ptr", 0)
        p2p.receiver_slot_offset = msg.get("slot_offset", 0)

        dest_addr = p2p.receiver_pool_ptr + p2p.receiver_slot_offset
        push_msg = TransferPushMsg(
            request_id=request_id,
            dest_session_id=p2p.receiver_session_id,
            dest_addr=dest_addr,
            transfer_size=p2p.data_size,
        )

        sender_idx = p2p.sender_instance
        record = self._tracker.get(request_id)
        if record and record.state in (
            RequestState.DECODER_RUNNING,
            RequestState.DECODER_WAITING,
        ):
            self._denoiser_pushes[sender_idx].send_multipart(
                encode_transfer_msg(push_msg)
            )
        else:
            self._encoder_pushes[sender_idx].send_multipart(
                encode_transfer_msg(push_msg)
            )

    def _handle_transfer_pushed(self, msg: dict) -> None:
        request_id = msg["request_id"]
        logger.debug("DiffusionServer transfer: pushed %s", request_id)
        p2p = self._transfer_state.get(request_id)
        if p2p is None:
            logger.warning(
                "DiffusionServer transfer: no state for pushed %s", request_id
            )
            return

        # Use record state (not sender_idx) to determine sender role,
        # because encoder and denoiser can share the same instance index.
        record = self._tracker.get(request_id)
        if record and record.state in (
            RequestState.DENOISING_RUNNING,
            RequestState.DENOISING_WAITING,
            RequestState.DENOISING_DONE,
        ):
            if record.encoder_instance is not None:
                self._encoder_free_slots[record.encoder_instance] += 1
        elif record and record.state in (
            RequestState.DECODER_RUNNING,
            RequestState.DECODER_WAITING,
        ):
            if record.denoiser_instance is not None:
                self._denoiser_free_slots[record.denoiser_instance] += 1

        scalar_fields = dict(p2p.scalar_fields) if p2p.scalar_fields else {}
        if p2p.prealloc_slot_id is not None:
            scalar_fields["_prealloc_slot_id"] = p2p.prealloc_slot_id
        ready_msg = TransferReadyMsg(
            request_id=request_id,
            manifest=p2p.manifest,
            slot_offset=p2p.receiver_slot_offset,
            scalar_fields=scalar_fields,
        )

        receiver_idx = p2p.receiver_instance
        record = self._tracker.get(request_id)
        if record and record.state in (
            RequestState.DENOISING_RUNNING,
            RequestState.DENOISING_WAITING,
        ):
            self._denoiser_pushes[receiver_idx].send_multipart(
                encode_transfer_msg(ready_msg)
            )
        elif record and record.state in (
            RequestState.DECODER_RUNNING,
            RequestState.DECODER_WAITING,
        ):
            self._decoder_pushes[receiver_idx].send_multipart(
                encode_transfer_msg(ready_msg)
            )

        logger.debug(
            "DiffusionServer transfer: notified receiver for %s (data ready)",
            request_id,
        )

    def _recycle_prealloc_slot(
        self, p2p: _TransferRequestState, role: RoleType
    ) -> None:
        if p2p is None or p2p.prealloc_slot_id is None:
            return
        receiver_idx = p2p.receiver_instance
        if role == RoleType.DENOISER:
            peer_info = self._denoiser_peers.get(receiver_idx, {})
        elif role == RoleType.DECODER:
            peer_info = self._decoder_peers.get(receiver_idx, {})
        else:
            return
        free_list = peer_info.get("free_preallocated_slots", [])
        free_list.append(
            {
                "offset": p2p.receiver_slot_offset,
                "size": p2p.data_size,
                "slot_id": p2p.prealloc_slot_id,
                "addr": p2p.receiver_pool_ptr + p2p.receiver_slot_offset,
            }
        )
        p2p.prealloc_slot_id = None

    def _handle_transfer_done(self, msg: dict, role: RoleType) -> None:
        request_id = msg.get("request_id", "")
        logger.debug(
            "DiffusionServer transfer: done %s role=%s",
            request_id,
            role.value,
        )
        error = msg.get("error")
        p2p = self._transfer_state.get(request_id)

        if role == RoleType.DENOISER:
            record = self._tracker.get(request_id)

            if p2p is not None:
                self._recycle_prealloc_slot(p2p, RoleType.DENOISER)

            if error:
                if record and record.denoiser_instance is not None:
                    self._denoiser_free_slots[record.denoiser_instance] += 1
                self._complete_with_error(request_id, f"Denoiser error: {error}")
                return

            try:
                self._tracker.transition(request_id, RequestState.DENOISING_DONE)
            except ValueError:
                pass

            if p2p is not None and msg.get("staged_for_decoder"):
                # Denoiser slot freed later in _handle_transfer_pushed
                p2p.sender_session_id = msg.get("session_id", "")
                p2p.sender_pool_ptr = msg.get("pool_ptr", 0)
                p2p.sender_slot_offset = msg.get("slot_offset", 0)
                p2p.data_size = msg.get("data_size", 0)
                p2p.manifest = msg.get("manifest", {})
                p2p.scalar_fields = msg.get("scalar_fields", {})
                p2p.sender_instance = record.denoiser_instance if record else 0

                try:
                    self._tracker.transition(request_id, RequestState.DECODER_WAITING)
                except ValueError:
                    pass
                self._decoder_tta.append(
                    _RoleTTAEntry(request_id=request_id, transfer_state=p2p)
                )
            else:
                if record and record.denoiser_instance is not None:
                    self._denoiser_free_slots[record.denoiser_instance] += 1

        elif role == RoleType.DECODER:
            if p2p is not None:
                self._recycle_prealloc_slot(p2p, RoleType.DECODER)

            record = self._tracker.get(request_id)
            if record and record.decoder_instance is not None:
                self._decoder_free_slots[record.decoder_instance] += 1

            if error:
                self._complete_with_error(request_id, f"Decoder error: {error}")
            else:
                try:
                    self._tracker.transition(request_id, RequestState.DONE)
                except ValueError:
                    pass

                self._transfer_return_to_client_from_msg(request_id, msg)

            self._transfer_state.pop(request_id, None)

    def _transfer_dispatch_to_decoder(
        self, request_id: str, p2p: _TransferRequestState, decoder_idx: int
    ) -> None:
        self._decoder_free_slots[decoder_idx] -= 1
        p2p.receiver_instance = decoder_idx

        try:
            self._tracker.transition(
                request_id,
                RequestState.DECODER_RUNNING,
                decoder_instance=decoder_idx,
            )
        except ValueError:
            pass

        peer_info = self._decoder_peers.get(decoder_idx, {})
        if not self._try_fast_path_push(
            request_id=request_id,
            p2p=p2p,
            receiver_peer_info=peer_info,
            sender_pushes=self._denoiser_pushes,
            receiver_role_label="decoder",
            receiver_idx=decoder_idx,
        ):
            self._send_slow_path_alloc(
                request_id=request_id,
                p2p=p2p,
                receiver_pushes=self._decoder_pushes,
                receiver_idx=decoder_idx,
                source_role="denoiser",
            )

    def _transfer_return_to_client_from_msg(self, request_id: str, msg: dict) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._tracker.remove(request_id)
            return

        output_batch = OutputBatch(error=msg.get("error"))

        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer transfer: failed to send result for %s: %s",
                request_id,
                e,
            )
        self._tracker.remove(request_id)

    def get_stats(self) -> dict:
        with self._lock:
            pending_count = len(self._pending)
        return {
            "role": "diffusion_server",
            "transfer_mode": self._transfer_mode,
            "num_encoders": self._num_encoders,
            "num_denoisers": self._num_denoisers,
            "num_decoders": self._num_decoders,
            "pending_requests": pending_count,
            "dispatch_policy": type(self._dispatcher.encoder_policy).__name__,
            "encoder_free_slots": list(self._encoder_free_slots),
            "denoiser_free_slots": list(self._denoiser_free_slots),
            "decoder_free_slots": list(self._decoder_free_slots),
            "encoder_tta_depth": len(self._encoder_tta),
            "denoiser_tta_depth": len(self._denoiser_tta),
            "decoder_tta_depth": len(self._decoder_tta),
            "transfer_active_transfers": len(self._transfer_state),
            "encoder_peers": len(self._encoder_peers),
            "denoiser_peers": len(self._denoiser_peers),
            "decoder_peers": len(self._decoder_peers),
            "tracker": self._tracker.snapshot(),
        }
