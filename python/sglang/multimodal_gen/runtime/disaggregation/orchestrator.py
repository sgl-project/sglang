# SPDX-License-Identifier: Apache-2.0
"""Central request router for disaggregated diffusion pipelines.

Supports both the classic 3-role (encoder/denoiser/decoder) topology and
arbitrary N-group topologies via ``PlacementGroupConfig``.
"""

import json
import logging
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass

import zmq

from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    GroupDispatcher,
)
from sglang.multimodal_gen.runtime.disaggregation.placement_group import (
    PlacementGroupConfig,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
)
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
class _FirstGroupTTAEntry:
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
    sender_group: str = ""
    receiver_group: str = ""
    prealloc_slot_id: int | None = None

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.scalar_fields is None:
            self.scalar_fields = {}


@dataclass
class _GroupTTAEntry:
    request_id: str
    transfer_state: _TransferRequestState | None = None


class DiffusionServer:
    """Global pipeline orchestrator for N-group disaggregated diffusion.

    Capacity-aware dispatch with FreeBufferSlots per instance and TTA queues.
    Supports arbitrary group topologies via ``PlacementGroupConfig``.
    """

    def __init__(
        self,
        frontend_endpoint: str,
        group_config: PlacementGroupConfig,
        group_work_endpoints: dict[str, list[str]],
        group_result_endpoints: dict[str, str],
        dispatch_policy_name: str = "round_robin",
        timeout_s: float = 600.0,
        group_capacities: dict[str, int] | None = None,
        p2p_mode: bool = True,
    ):
        self._frontend_endpoint = frontend_endpoint
        self._group_config = group_config
        self._timeout_s = timeout_s
        self._transfer_mode = p2p_mode

        self._group_chain: list[str] = group_config.group_chain()
        self._first_group: str = self._group_chain[0]
        self._last_group: str = self._group_chain[-1]
        self._next_group: dict[str, str | None] = {}
        for i, name in enumerate(self._group_chain):
            self._next_group[name] = (
                self._group_chain[i + 1] if i + 1 < len(self._group_chain) else None
            )

        default_capacity = 4
        capacities = group_capacities or {}

        group_instances: dict[str, int] = {}
        self._group_work_endpoints: dict[str, list[str]] = {}
        self._group_result_endpoints: dict[str, str] = {}
        self._group_free_slots: dict[str, list[int]] = {}
        self._group_first_tta: deque[_FirstGroupTTAEntry] = deque()
        self._group_tta: dict[str, deque[_GroupTTAEntry]] = {}
        self._group_peers: dict[str, dict[int, dict]] = {}
        self._group_endpoint_to_idx: dict[str, dict[str, int]] = {}

        for name in self._group_chain:
            eps = group_work_endpoints.get(name, [])
            self._group_work_endpoints[name] = eps
            self._group_result_endpoints[name] = group_result_endpoints.get(name, "")
            cap = capacities.get(name, default_capacity)
            self._group_free_slots[name] = [cap] * len(eps)
            self._group_tta[name] = deque()
            self._group_peers[name] = {}
            self._group_endpoint_to_idx[name] = {ep: i for i, ep in enumerate(eps)}
            group_instances[name] = len(eps)

        self._tracker = RequestTracker()
        self._dispatcher = GroupDispatcher(
            group_instances=group_instances,
            policy_name=dispatch_policy_name,
        )

        self._context = zmq.Context(io_threads=2)
        self._running = False
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

        self._pending: dict[str, bytes] = {}
        self._lock = threading.Lock()

        self._transfer_state: dict[str, _TransferRequestState] = {}

        self._group_pushes: dict[str, list[zmq.Socket]] = {}
        self._group_result_pulls: dict[str, zmq.Socket] = {}
        self._socket_to_group: dict[zmq.Socket, str] = {}

    @classmethod
    def from_classic_args(
        cls,
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
    ) -> "DiffusionServer":
        config = PlacementGroupConfig.classic_3_role()
        return cls(
            frontend_endpoint=frontend_endpoint,
            group_config=config,
            group_work_endpoints={
                "encoder": encoder_work_endpoints,
                "denoiser": denoiser_work_endpoints,
                "decoder": decoder_work_endpoints,
            },
            group_result_endpoints={
                "encoder": encoder_result_endpoint,
                "denoiser": denoiser_result_endpoint,
                "decoder": decoder_result_endpoint,
            },
            dispatch_policy_name=dispatch_policy_name,
            timeout_s=timeout_s,
            group_capacities={
                "encoder": encoder_capacity,
                "denoiser": denoiser_capacity,
                "decoder": decoder_capacity,
            },
            p2p_mode=p2p_mode,
        )

    @property
    def tracker(self) -> RequestTracker:
        return self._tracker

    @property
    def dispatcher(self) -> GroupDispatcher:
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
        group_summary = ", ".join(
            f"{name}({len(self._group_work_endpoints[name])})"
            for name in self._group_chain
        )
        logger.info(
            "DiffusionServer started: frontend=%s, groups=[%s]",
            self._frontend_endpoint,
            group_summary,
        )

    def wait_ready(self, timeout: float = 30.0) -> bool:
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

        all_sockets: list[zmq.Socket] = [frontend]

        for name in self._group_chain:
            pushes: list[zmq.Socket] = []
            for ep in self._group_work_endpoints[name]:
                sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
                pushes.append(sock)
                all_sockets.append(sock)
            self._group_pushes[name] = pushes

            result_ep = self._group_result_endpoints[name]
            if result_ep:
                result_pull, _ = get_zmq_socket(
                    self._context, zmq.PULL, result_ep, bind=True
                )
                self._group_result_pulls[name] = result_pull
                self._socket_to_group[result_pull] = name
                all_sockets.append(result_pull)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        for result_pull in self._group_result_pulls.values():
            poller.register(result_pull, zmq.POLLIN)

        self._frontend = frontend
        self._ready.set()

        try:
            while self._running:
                events = dict(poller.poll(timeout=10))

                self._handle_timeouts()

                if frontend in events:
                    self._handle_client_request(frontend)

                for socket in list(events.keys()):
                    if socket in self._socket_to_group:
                        group_name = self._socket_to_group[socket]
                        self._handle_group_result(socket, group_name)

                self._drain_all_queues()

        except Exception:
            logger.exception("DiffusionServer event loop error")
        finally:
            for sock in all_sockets:
                sock.close()
            self._context.destroy(linger=0)

    def _handle_group_result(self, result_pull: zmq.Socket, group_name: str) -> None:
        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        if is_transfer_message(frames):
            self._handle_transfer_result(frames, group_name)
            return

        if group_name == self._last_group:
            self._handle_last_group_result_frames(frames, group_name)
        else:
            self._handle_group_error_frames(frames, group_name)

    def _handle_group_error_frames(self, frames: list, group_name: str) -> None:
        try:
            tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")
        except Exception as e:
            logger.warning(
                "DiffusionServer: failed to unpack non-transfer frames from %s: %s",
                group_name,
                e,
            )
            return

        request_id = scalar_fields.get("request_id")
        disagg_error = scalar_fields.get("_disagg_error")

        if request_id and disagg_error:
            logger.error(
                "DiffusionServer: %s error for %s: %s",
                group_name,
                request_id,
                disagg_error,
            )
            self._complete_with_error(request_id, f"{group_name} error: {disagg_error}")
        elif request_id:
            logger.warning(
                "DiffusionServer: non-transfer frames from %s for %s without error",
                group_name,
                request_id,
            )
        else:
            logger.warning(
                "DiffusionServer: non-transfer frames from %s without request_id",
                group_name,
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
            self._tracker.transition(
                request_id,
                RequestState.GROUP_WAITING,
                current_group=self._first_group,
            )
        except ValueError:
            pass
        self._group_first_tta.append(
            _FirstGroupTTAEntry(
                request_id=request_id,
                client_identity=client_identity,
                payload=payload,
            )
        )
        logger.debug(
            "DiffusionServer: queued %s to %s",
            request_id,
            self._first_group,
        )

    def _handle_last_group_result_frames(self, frames: list, group_name: str) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: %s result missing request_id", group_name)
            return

        logger.debug("DiffusionServer: %s result %s", group_name, request_id)
        record = self._tracker.get(request_id)
        if record and record.current_group:
            instance = record.group_instances.get(record.current_group)
            if instance is not None:
                self._group_free_slots[record.current_group][instance] += 1

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
                "DiffusionServer: no pending client for %s result %s",
                group_name,
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

    def _dispatch_to_first_group(
        self, request_id: str, payload: bytes, instance_idx: int
    ) -> None:
        first = self._first_group
        self._group_free_slots[first][instance_idx] -= 1

        try:
            self._tracker.transition(
                request_id,
                RequestState.GROUP_RUNNING,
                current_group=first,
                group_instance=(first, instance_idx),
            )
        except ValueError:
            pass

        self._group_pushes[first][instance_idx].send_multipart(
            [request_id.encode("utf-8"), payload]
        )
        logger.debug(
            "DiffusionServer: dispatched %s to %s[%d] (free=%d)",
            request_id,
            first,
            instance_idx,
            self._group_free_slots[first][instance_idx],
        )

    def _drain_all_queues(self) -> None:
        self._drain_first_group_tta()
        for name in self._group_chain:
            if name == self._first_group:
                continue
            self._drain_group_tta(name)

    def _drain_first_group_tta(self) -> None:
        first = self._first_group
        while self._group_first_tta:
            idx = self._dispatcher.select_with_capacity(
                first, self._group_free_slots[first]
            )
            if idx is None:
                break
            entry = self._group_first_tta.popleft()
            self._dispatch_to_first_group(entry.request_id, entry.payload, idx)

    def _drain_group_tta(self, group_name: str) -> None:
        tta = self._group_tta[group_name]
        while tta:
            idx = self._dispatcher.select_with_capacity(
                group_name, self._group_free_slots[group_name]
            )
            if idx is None:
                break
            entry = tta.popleft()
            self._transfer_dispatch_to_group(
                group_name, entry.request_id, entry.transfer_state, idx
            )

    def _transfer_dispatch_to_group(
        self,
        group_name: str,
        request_id: str,
        p2p: _TransferRequestState,
        instance_idx: int,
    ) -> None:
        self._group_free_slots[group_name][instance_idx] -= 1
        p2p.receiver_instance = instance_idx
        p2p.receiver_group = group_name

        try:
            self._tracker.transition(
                request_id,
                RequestState.GROUP_RUNNING,
                current_group=group_name,
                group_instance=(group_name, instance_idx),
            )
        except ValueError:
            pass

        peer_info = self._group_peers[group_name].get(instance_idx, {})
        sender_group = p2p.sender_group
        if not self._try_fast_path_push(
            request_id=request_id,
            p2p=p2p,
            receiver_peer_info=peer_info,
            sender_pushes=self._group_pushes.get(sender_group, []),
            receiver_role_label=group_name,
            receiver_idx=instance_idx,
        ):
            self._send_slow_path_alloc(
                request_id=request_id,
                p2p=p2p,
                receiver_pushes=self._group_pushes.get(group_name, []),
                receiver_idx=instance_idx,
                source_role=sender_group,
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
            self._group_first_tta = deque(
                e for e in self._group_first_tta if e.request_id not in timed_set
            )
            for name in self._group_chain:
                tta = self._group_tta[name]
                self._group_tta[name] = deque(
                    e for e in tta if e.request_id not in timed_set
                )

    def _free_slot_for_record(self, record) -> None:
        if record.current_group and record.state in (
            RequestState.GROUP_RUNNING,
            RequestState.GROUP_DONE,
        ):
            instance = record.group_instances.get(record.current_group)
            if instance is not None:
                self._group_free_slots[record.current_group][instance] += 1

    def _handle_transfer_result(self, frames: list, group_name: str) -> None:
        try:
            msg = decode_transfer_msg(frames)
        except (ValueError, Exception) as e:
            logger.error("DiffusionServer: failed to decode transfer message: %s", e)
            return

        msg_type = msg.get("msg_type")

        if msg_type == TransferMsgType.REGISTER:
            self._handle_transfer_register(msg, group_name)
        elif msg_type == TransferMsgType.STAGED:
            self._handle_transfer_staged(msg, group_name)
        elif msg_type == TransferMsgType.ALLOCATED:
            self._handle_transfer_allocated(msg)
        elif msg_type == TransferMsgType.PUSHED:
            self._handle_transfer_pushed(msg)
        elif msg_type == TransferMsgType.DONE:
            self._handle_transfer_done(msg, group_name)
        else:
            logger.warning("DiffusionServer: unknown transfer msg_type=%s", msg_type)

    def _handle_transfer_register(self, msg: dict, group_name: str) -> None:
        work_endpoint = msg.get("work_endpoint", "")
        endpoint_to_idx = self._group_endpoint_to_idx.get(group_name, {})
        peers = self._group_peers.get(group_name, {})

        role_in_msg = msg.get("role", "")
        resolved_group = group_name
        if role_in_msg and role_in_msg != group_name:
            for gname, ep_map in self._group_endpoint_to_idx.items():
                if work_endpoint in ep_map:
                    resolved_group = gname
                    endpoint_to_idx = ep_map
                    peers = self._group_peers.get(gname, {})
                    break

        idx = endpoint_to_idx.get(work_endpoint)
        if idx is None:
            logger.error(
                "DiffusionServer transfer: register for group=%s with unknown "
                "work_endpoint=%r (known=%s); dropping registration",
                resolved_group,
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
            resolved_group,
            idx,
            work_endpoint,
            info["session_id"],
            info["pool_ptr"],
            len(prealloc),
        )

    def _handle_transfer_staged(self, msg: dict, sender_group: str) -> None:
        request_id = msg["request_id"]
        logger.debug("DiffusionServer transfer: %s staged %s", sender_group, request_id)
        record = self._tracker.get(request_id)
        sender_instance = record.group_instances.get(sender_group, 0) if record else 0

        p2p = _TransferRequestState(
            sender_session_id=msg.get("session_id", ""),
            sender_pool_ptr=msg.get("pool_ptr", 0),
            sender_slot_offset=msg.get("slot_offset", 0),
            data_size=msg.get("data_size", 0),
            manifest=msg.get("manifest", {}),
            scalar_fields=msg.get("scalar_fields", {}),
            sender_instance=sender_instance,
            sender_group=sender_group,
        )
        self._transfer_state[request_id] = p2p

        try:
            self._tracker.transition(
                request_id,
                RequestState.GROUP_DONE,
                current_group=sender_group,
            )
        except ValueError:
            pass

        next_group = self._next_group.get(sender_group)
        if next_group is not None:
            try:
                self._tracker.transition(
                    request_id,
                    RequestState.GROUP_WAITING,
                    current_group=next_group,
                )
            except ValueError:
                pass
            self._group_tta[next_group].append(
                _GroupTTAEntry(request_id=request_id, transfer_state=p2p)
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
        alloc_msg = TransferAllocMsg(
            request_id=request_id,
            data_size=p2p.data_size,
            source_role=source_role,
        )
        receiver_pushes[receiver_idx].send_multipart(encode_transfer_msg(alloc_msg))

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

        sender_group = p2p.sender_group
        sender_pushes = self._group_pushes.get(sender_group, [])
        if p2p.sender_instance < len(sender_pushes):
            sender_pushes[p2p.sender_instance].send_multipart(
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

        sender_group = p2p.sender_group
        record = self._tracker.get(request_id)
        if record:
            sender_instance = record.group_instances.get(sender_group)
            if sender_instance is not None:
                self._group_free_slots[sender_group][sender_instance] += 1

        scalar_fields = dict(p2p.scalar_fields) if p2p.scalar_fields else {}
        if p2p.prealloc_slot_id is not None:
            scalar_fields["_prealloc_slot_id"] = p2p.prealloc_slot_id
        ready_msg = TransferReadyMsg(
            request_id=request_id,
            manifest=p2p.manifest,
            slot_offset=p2p.receiver_slot_offset,
            scalar_fields=scalar_fields,
        )

        receiver_group = p2p.receiver_group
        receiver_pushes = self._group_pushes.get(receiver_group, [])
        receiver_idx = p2p.receiver_instance
        if receiver_idx < len(receiver_pushes):
            receiver_pushes[receiver_idx].send_multipart(encode_transfer_msg(ready_msg))

        logger.debug(
            "DiffusionServer transfer: notified receiver for %s (data ready)",
            request_id,
        )

    def _recycle_prealloc_slot(
        self, p2p: _TransferRequestState, group_name: str
    ) -> None:
        if p2p is None or p2p.prealloc_slot_id is None:
            return
        receiver_idx = p2p.receiver_instance
        peer_info = self._group_peers.get(group_name, {}).get(receiver_idx, {})
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

    def _handle_transfer_done(self, msg: dict, group_name: str) -> None:
        request_id = msg.get("request_id", "")
        logger.debug(
            "DiffusionServer transfer: done %s group=%s",
            request_id,
            group_name,
        )
        error = msg.get("error")
        p2p = self._transfer_state.get(request_id)
        record = self._tracker.get(request_id)

        if p2p is not None:
            self._recycle_prealloc_slot(p2p, group_name)

        if error:
            if record:
                instance = record.group_instances.get(group_name)
                if instance is not None:
                    self._group_free_slots[group_name][instance] += 1
            self._complete_with_error(request_id, f"{group_name} error: {error}")
            return

        next_group = self._next_group.get(group_name)

        if next_group is not None and msg.get(
            "staged_for_next", msg.get("staged_for_decoder")
        ):
            try:
                self._tracker.transition(
                    request_id,
                    RequestState.GROUP_DONE,
                    current_group=group_name,
                )
            except ValueError:
                pass

            if p2p is not None:
                p2p.sender_session_id = msg.get("session_id", "")
                p2p.sender_pool_ptr = msg.get("pool_ptr", 0)
                p2p.sender_slot_offset = msg.get("slot_offset", 0)
                p2p.data_size = msg.get("data_size", 0)
                p2p.manifest = msg.get("manifest", {})
                p2p.scalar_fields = msg.get("scalar_fields", {})
                p2p.sender_instance = (
                    record.group_instances.get(group_name, 0) if record else 0
                )
                p2p.sender_group = group_name

            try:
                self._tracker.transition(
                    request_id,
                    RequestState.GROUP_WAITING,
                    current_group=next_group,
                )
            except ValueError:
                pass
            self._group_tta[next_group].append(
                _GroupTTAEntry(request_id=request_id, transfer_state=p2p)
            )
        elif next_group is None:
            if record:
                instance = record.group_instances.get(group_name)
                if instance is not None:
                    self._group_free_slots[group_name][instance] += 1

            try:
                self._tracker.transition(request_id, RequestState.DONE)
            except ValueError:
                pass

            self._transfer_return_to_client_from_msg(request_id, msg)
            self._transfer_state.pop(request_id, None)
        else:
            if record:
                instance = record.group_instances.get(group_name)
                if instance is not None:
                    self._group_free_slots[group_name][instance] += 1

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

        group_stats = {}
        for name in self._group_chain:
            group_stats[name] = {
                "num_instances": len(self._group_work_endpoints[name]),
                "free_slots": list(self._group_free_slots[name]),
                "tta_depth": len(self._group_tta[name]),
                "peers": len(self._group_peers[name]),
            }

        return {
            "role": "diffusion_server",
            "transfer_mode": self._transfer_mode,
            "group_chain": self._group_chain,
            "groups": group_stats,
            "pending_requests": pending_count,
            "first_group_tta_depth": len(self._group_first_tta),
            "transfer_active_transfers": len(self._transfer_state),
            "tracker": self._tracker.snapshot(),
        }
