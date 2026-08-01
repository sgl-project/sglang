# SPDX-License-Identifier: Apache-2.0
"""Central request router for disaggregated diffusion pipelines.

``DiffusionServer`` owns the ZMQ plumbing and the point-to-point transfer
handshake; every routing decision -- what runs next, on which instance, and
whether a branch runs at all -- belongs to
:class:`~sglang.multimodal_gen.runtime.disaggregation.dag.scheduler.DagRequestScheduler`.
Keeping the two apart means a new topology is a config change rather than an
edit to this event loop.

The classic encoder/denoiser/decoder deployment is a three-node linear DAG;
:meth:`DiffusionServer.from_classic_args` builds it from the legacy arguments.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.dag.plan import ExecutionPlan
from sglang.multimodal_gen.runtime.disaggregation.dag.scheduler import (
    Action,
    CompleteRequest,
    DagRequestScheduler,
    EdgeTransfer,
    FailRequest,
    SourceDispatch,
)
from sglang.multimodal_gen.runtime.disaggregation.dag.state import TransferHandle
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import unpack_tensors
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

# Scalars worth exposing to route predicates but not part of SamplingParams.
_EXTRA_ROUTE_FIELDS = ("height", "width", "num_frames", "generate_audio")


@dataclass
class _EdgeTransferState:
    """In-flight point-to-point transfer for one DAG edge.

    A fan-out produces several of these against a single staged sender buffer,
    so receiver-side fields differ per edge while the sender-side fields are
    shared.
    """

    request_id: str
    edge_id: str
    src_node: str
    src_instance: int
    dst_node: str
    dst_instance: int
    input_index: int
    expected_inputs: int
    fanout_total: int
    data_size: int
    manifest: dict = field(default_factory=dict)
    scalar_fields: dict = field(default_factory=dict)
    sender_session_id: str = ""
    sender_pool_ptr: int = 0
    sender_slot_offset: int = 0
    receiver_session_id: str = ""
    receiver_pool_ptr: int = 0
    receiver_slot_offset: int = 0
    prealloc_slot_id: int | None = None


class DiffusionServer:
    """Global pipeline orchestrator for DAG-disaggregated diffusion."""

    def __init__(
        self,
        frontend_endpoint: str,
        plan: ExecutionPlan,
        *,
        timeout_s: float = 600.0,
        max_inflight: int | None = None,
    ):
        self._frontend_endpoint = frontend_endpoint
        self._plan = plan
        self._timeout_s = timeout_s
        self._scheduler = DagRequestScheduler(
            plan, max_inflight=max_inflight, timeout_s=timeout_s
        )

        self._context = zmq.Context(io_threads=2)
        self._running = False
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._frontend: zmq.Socket | None = None
        self._node_pushes: dict[str, list[zmq.Socket]] = {}
        self._result_pulls: dict[str, zmq.Socket] = {}

        # instance index -> {session_id, pool_ptr, free_preallocated_slots, ...}
        self._peers: dict[str, dict[int, dict]] = {n: {} for n in plan.node_names}
        self._endpoint_to_idx: dict[str, dict[str, int]] = {
            n: {url: i for i, url in enumerate(plan.node(n).pool.urls)}
            for n in plan.node_names
        }

        self._transfers: dict[tuple[str, str], _EdgeTransferState] = {}
        self._pending_clients: dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
    ) -> DiffusionServer:
        """Build the legacy three-role topology as a linear DAG."""
        del p2p_mode  # transfers are always point-to-point now
        plan = ExecutionPlan.from_classic_roles(
            encoder_work_endpoints,
            denoiser_work_endpoints,
            decoder_work_endpoints,
            encoder_capacity=encoder_capacity,
            denoiser_capacity=denoiser_capacity,
            decoder_capacity=decoder_capacity,
            dispatch_policy=dispatch_policy_name,
            encoder_result_endpoint=encoder_result_endpoint,
            denoiser_result_endpoint=denoiser_result_endpoint,
            decoder_result_endpoint=decoder_result_endpoint,
        )
        return cls(frontend_endpoint, plan, timeout_s=timeout_s)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop, name="DiffusionServer", daemon=True
        )
        self._thread.start()
        logger.info(
            "DiffusionServer started: frontend=%s, nodes=[%s]",
            self._frontend_endpoint,
            ", ".join(
                f"{n}({self._plan.node(n).num_instances})"
                for n in self._plan.node_names
            ),
        )

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until the event loop has bound all sockets, or *timeout* elapses."""
        return self._ready.wait(timeout=timeout)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_stats(self) -> dict:
        stats = self._scheduler.stats()
        stats["role"] = "diffusion_server"
        stats["plan"] = self._plan.to_dict()
        stats["active_transfers"] = len(self._transfers)
        stats["peers"] = {node: len(peers) for node, peers in self._peers.items()}
        return stats

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _event_loop(self) -> None:
        frontend, _ = get_zmq_socket(
            self._context, zmq.ROUTER, self._frontend_endpoint, bind=True
        )
        self._frontend = frontend

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        all_sockets: list[zmq.Socket] = [frontend]
        pull_to_node: dict[zmq.Socket, str] = {}

        for name in self._plan.node_names:
            node = self._plan.node(name)
            pushes = []
            for url in node.pool.urls:
                sock, _ = get_zmq_socket(self._context, zmq.PUSH, url, bind=False)
                pushes.append(sock)
                all_sockets.append(sock)
            self._node_pushes[name] = pushes

            result_endpoint = node.pool.result_endpoint
            if not result_endpoint:
                raise ValueError(
                    f"Node '{name}' has no result endpoint; the orchestrator "
                    f"cannot receive its completions"
                )
            pull, _ = get_zmq_socket(
                self._context, zmq.PULL, result_endpoint, bind=True
            )
            self._result_pulls[name] = pull
            pull_to_node[pull] = name
            poller.register(pull, zmq.POLLIN)
            all_sockets.append(pull)

        self._ready.set()

        try:
            while self._running:
                events = dict(poller.poll(timeout=10))

                self._execute(self._scheduler.check_timeouts())

                if frontend in events:
                    self._handle_client_request(frontend)

                for pull, node in pull_to_node.items():
                    if pull in events:
                        self._handle_node_result(pull, node)

                self._execute(self._scheduler.drain())
        except Exception:
            logger.exception("DiffusionServer event loop error")
        finally:
            for sock in all_sockets:
                sock.close()
            self._context.destroy(linger=0)

    # ------------------------------------------------------------------
    # Ingress
    # ------------------------------------------------------------------

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
            # Reply so a REQ socket does not hang waiting on us.
            try:
                frontend.send_multipart(
                    [client_identity, b"", pickle.dumps({"status": "ignored"})],
                    zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            return

        request_id = getattr(req, "request_id", None) or f"ds-{id(req):x}"

        with self._lock:
            self._pending_clients[request_id] = client_identity

        try:
            actions = self._scheduler.submit(
                request_id,
                client_identity,
                payload,
                _build_route_context(req),
            )
        except ValueError:
            logger.warning("DiffusionServer: duplicate request_id %s", request_id)
            return

        self._execute(actions)

    def _handle_node_result(self, pull: zmq.Socket, node: str) -> None:
        try:
            frames = pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        if is_transfer_message(frames):
            self._handle_transfer_message(frames, node)
            return

        # Non-transfer frames carry either a terminal node's final output or
        # an error raised anywhere in the pipeline.
        try:
            tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")
        except Exception as e:
            logger.warning(
                "DiffusionServer: failed to unpack frames from %s: %s", node, e
            )
            return

        request_id = scalar_fields.get("request_id") or _request_id_from_frames(frames)
        if not request_id:
            logger.warning("DiffusionServer: frames from %s without request_id", node)
            return

        error = scalar_fields.get("_disagg_error") or scalar_fields.get("error")
        if error:
            self._execute(self._scheduler.on_node_error(request_id, node, str(error)))
            return

        if not self._plan.node(node).terminal:
            logger.warning(
                "DiffusionServer: non-terminal node %s returned output frames "
                "for %s without an error",
                node,
                request_id,
            )
            return

        self._recycle_inbound_slots(request_id, node)
        fields: dict[str, Any] = {
            k: v for k, v in tensor_fields.items() if v is not None
        }
        for key in ("audio_sample_rate",):
            if scalar_fields.get(key) is not None:
                fields[key] = scalar_fields[key]

        self._execute(self._scheduler.on_terminal_result(request_id, node, fields))

    # ------------------------------------------------------------------
    # Transfer handshake
    # ------------------------------------------------------------------

    def _handle_transfer_message(self, frames: list, node: str) -> None:
        try:
            msg = decode_transfer_msg(frames)
        except Exception as e:
            logger.error("DiffusionServer: failed to decode transfer message: %s", e)
            return

        msg_type = msg.get("msg_type")
        if msg_type == TransferMsgType.REGISTER:
            self._handle_register(msg, node)
        elif msg_type == TransferMsgType.STAGED:
            self._handle_staged(msg, node)
        elif msg_type == TransferMsgType.ALLOCATED:
            self._handle_allocated(msg)
        elif msg_type == TransferMsgType.PUSHED:
            self._handle_pushed(msg)
        elif msg_type == TransferMsgType.DONE:
            self._handle_done(msg, node)
        else:
            logger.warning("DiffusionServer: unknown transfer msg_type=%s", msg_type)

    def _handle_register(self, msg: dict, node: str) -> None:
        work_endpoint = msg.get("work_endpoint", "")
        endpoint_to_idx = self._endpoint_to_idx.get(node, {})
        idx = endpoint_to_idx.get(work_endpoint)
        if idx is None:
            # Fail loudly: without a URL match the control plane (work PUSH)
            # and the data plane (RDMA destination) would silently drift.
            logger.error(
                "DiffusionServer: register for node=%s with unknown "
                "work_endpoint=%r (known=%s); dropping registration",
                node,
                work_endpoint,
                list(endpoint_to_idx),
            )
            return

        prealloc = list(msg.get("preallocated_slots", []))
        self._peers[node][idx] = {
            "session_id": msg.get("session_id", ""),
            "pool_ptr": msg.get("pool_ptr", 0),
            "pool_size": msg.get("pool_size", 0),
            "work_endpoint": work_endpoint,
            "free_preallocated_slots": prealloc,
        }
        logger.info(
            "DiffusionServer: registered %s[%d] endpoint=%s session=%s prealloc=%d",
            node,
            idx,
            work_endpoint,
            msg.get("session_id", ""),
            len(prealloc),
        )

    def _handle_staged(self, msg: dict, node: str) -> None:
        """A node finished compute and parked its output for downstream."""
        request_id = msg["request_id"]
        self._recycle_inbound_slots(request_id, node)

        handle = TransferHandle(
            src_node=node,
            session_id=msg.get("session_id", ""),
            pool_ptr=msg.get("pool_ptr", 0),
            slot_offset=msg.get("slot_offset", 0),
            data_size=msg.get("data_size", 0),
            manifest=msg.get("manifest", {}),
            scalar_fields=msg.get("scalar_fields", {}),
        )
        self._execute(self._scheduler.on_node_staged(request_id, node, handle))

    def _handle_done(self, msg: dict, node: str) -> None:
        """Completion notice; only meaningful today as an error channel."""
        request_id = msg.get("request_id", "")
        error = msg.get("error")
        if error:
            self._execute(self._scheduler.on_node_error(request_id, node, str(error)))

    def _start_edge_transfer(self, action: EdgeTransfer) -> None:
        transfer = _EdgeTransferState(
            request_id=action.request_id,
            edge_id=action.edge_id,
            src_node=action.src_node,
            src_instance=action.src_instance,
            dst_node=action.dst_node,
            dst_instance=action.dst_instance,
            input_index=action.input_index,
            expected_inputs=action.expected_inputs,
            fanout_total=action.fanout_total,
            data_size=action.transfer.data_size,
            manifest=action.transfer.manifest,
            scalar_fields=action.transfer.scalar_fields,
            sender_session_id=action.transfer.session_id,
            sender_pool_ptr=action.transfer.pool_ptr,
            sender_slot_offset=action.transfer.slot_offset,
        )
        self._transfers[(action.request_id, action.edge_id)] = transfer

        if not self._try_fast_path_push(transfer):
            self._send_slow_path_alloc(transfer)

    def _try_fast_path_push(self, transfer: _EdgeTransferState) -> bool:
        """Claim a pre-registered receive slot and start RDMA immediately.

        Returns False when the receiver has no free pre-allocated slot big
        enough, in which case the caller falls back to the alloc round trip.
        """
        peer = self._peers.get(transfer.dst_node, {}).get(transfer.dst_instance, {})
        free_slots = peer.get("free_preallocated_slots", [])
        if not (free_slots and free_slots[0].get("size", 0) >= transfer.data_size):
            return False

        slot = free_slots.pop(0)
        transfer.receiver_session_id = peer.get("session_id", "")
        transfer.receiver_pool_ptr = peer.get("pool_ptr", 0)
        transfer.receiver_slot_offset = slot["offset"]
        transfer.prealloc_slot_id = slot.get("slot_id")

        self._send_push(transfer, slot["addr"])
        logger.debug(
            "DiffusionServer: fast-path push %s to %s[%d] (slot %s, %d bytes)",
            transfer.edge_id,
            transfer.dst_node,
            transfer.dst_instance,
            transfer.prealloc_slot_id,
            transfer.data_size,
        )
        return True

    def _send_slow_path_alloc(self, transfer: _EdgeTransferState) -> None:
        alloc = TransferAllocMsg(
            request_id=transfer.request_id,
            data_size=transfer.data_size,
            source_role=transfer.src_node,
            edge_id=transfer.edge_id,
        )
        self._send_to(transfer.dst_node, transfer.dst_instance, alloc)

    def _handle_allocated(self, msg: dict) -> None:
        transfer = self._lookup_transfer(msg)
        if transfer is None:
            return
        transfer.receiver_session_id = msg.get("session_id", "")
        transfer.receiver_pool_ptr = msg.get("pool_ptr", 0)
        transfer.receiver_slot_offset = msg.get("slot_offset", 0)
        self._send_push(
            transfer, transfer.receiver_pool_ptr + transfer.receiver_slot_offset
        )

    def _send_push(self, transfer: _EdgeTransferState, dest_addr: int) -> None:
        push = TransferPushMsg(
            request_id=transfer.request_id,
            dest_session_id=transfer.receiver_session_id,
            dest_addr=dest_addr,
            transfer_size=transfer.data_size,
            edge_id=transfer.edge_id,
            fanout_total=transfer.fanout_total,
        )
        self._send_to(transfer.src_node, transfer.src_instance, push)

    def _handle_pushed(self, msg: dict) -> None:
        transfer = self._lookup_transfer(msg)
        if transfer is None:
            return

        scalar_fields = dict(transfer.scalar_fields)
        if transfer.prealloc_slot_id is not None:
            scalar_fields["_prealloc_slot_id"] = transfer.prealloc_slot_id

        ready = TransferReadyMsg(
            request_id=transfer.request_id,
            manifest=transfer.manifest,
            slot_offset=transfer.receiver_slot_offset,
            scalar_fields=scalar_fields,
            edge_id=transfer.edge_id,
            input_index=transfer.input_index,
            expected_inputs=transfer.expected_inputs,
        )
        self._send_to(transfer.dst_node, transfer.dst_instance, ready)
        self._execute(
            self._scheduler.on_edge_pushed(transfer.request_id, transfer.edge_id)
        )

    def _lookup_transfer(self, msg: dict) -> _EdgeTransferState | None:
        request_id = msg.get("request_id", "")
        edge_id = msg.get("edge_id", "")
        transfer = self._transfers.get((request_id, edge_id))
        if transfer is None:
            logger.warning(
                "DiffusionServer: no transfer state for %s on edge %r",
                request_id,
                edge_id,
            )
        return transfer

    def _recycle_inbound_slots(self, request_id: str, node: str) -> None:
        """Return the pre-allocated receive slots a node just finished reading."""
        for edge in self._plan.node(node).in_edges:
            transfer = self._transfers.pop((request_id, edge.edge_id), None)
            if transfer is None or transfer.prealloc_slot_id is None:
                continue
            peer = self._peers.get(node, {}).get(transfer.dst_instance)
            if peer is None:
                continue
            peer.setdefault("free_preallocated_slots", []).append(
                {
                    "offset": transfer.receiver_slot_offset,
                    "size": transfer.data_size,
                    "slot_id": transfer.prealloc_slot_id,
                    "addr": transfer.receiver_pool_ptr + transfer.receiver_slot_offset,
                }
            )

    def _send_to(self, node: str, instance: int, msg: Any) -> None:
        pushes = self._node_pushes.get(node, [])
        if not 0 <= instance < len(pushes):
            logger.error(
                "DiffusionServer: cannot reach %s[%s]; %d instance(s) connected",
                node,
                instance,
                len(pushes),
            )
            return
        pushes[instance].send_multipart(encode_transfer_msg(msg))

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute(self, actions: list[Action]) -> None:
        for action in actions:
            try:
                if isinstance(action, SourceDispatch):
                    self._node_pushes[action.node][action.instance].send_multipart(
                        [action.request_id.encode("utf-8"), action.payload]
                    )
                elif isinstance(action, EdgeTransfer):
                    self._start_edge_transfer(action)
                elif isinstance(action, CompleteRequest):
                    self._return_output(action)
                elif isinstance(action, FailRequest):
                    self._return_error(action)
            except Exception:
                logger.exception(
                    "DiffusionServer: failed to execute %s", type(action).__name__
                )

    def _return_output(self, action: CompleteRequest) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        self._drop_request_transfers(action.request_id)
        fields = _decode_terminal_fields(action.fields)
        batch = OutputBatch(
            output=fields.get("output"),
            audio=fields.get("audio"),
            audio_sample_rate=fields.get("audio_sample_rate"),
        )
        self._reply(action.request_id, action.client_identity, batch)

    def _return_error(self, action: FailRequest) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        logger.error("DiffusionServer: %s failed — %s", action.request_id, action.error)
        self._drop_request_transfers(action.request_id)
        self._reply(
            action.request_id,
            action.client_identity,
            OutputBatch(error=action.error),
        )

    def _drop_request_transfers(self, request_id: str) -> None:
        for key in [k for k in self._transfers if k[0] == request_id]:
            self._transfers.pop(key, None)

    def _reply(
        self, request_id: str, client_identity: bytes | None, batch: Any
    ) -> None:
        with self._lock:
            identity = self._pending_clients.pop(request_id, None)
        identity = identity or client_identity
        if identity is None:
            logger.warning("DiffusionServer: no pending client for %s", request_id)
            return
        try:
            self._frontend.send_multipart([identity, b"", pickle.dumps(batch)])
        except zmq.ZMQError as e:
            logger.error("DiffusionServer: failed to reply for %s: %s", request_id, e)


def _decode_terminal_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Restore terminal payloads after ZMQ tensor transport.

    Video decoders emit numpy frames; the codec moves them as tensors. Convert
    ``output`` back to numpy so downstream save/mux logic keeps the layout it
    expects (THWC), while leaving ``audio`` as a tensor.
    """
    if "output" not in fields or fields["output"] is None:
        return fields

    decoded = dict(fields)
    decoded["output"] = _transport_output_to_numpy(fields["output"])
    return decoded


def _transport_output_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        # Video decoders emit batched THWC (1, T, H, W, C). Downstream save
        # logic uses len(output) as the batch size, so keep the leading dim.
        if t.dim() == 5 and t.shape[-1] in (1, 3, 4):
            return t.numpy()
        if t.dim() == 4:
            if t.shape[-1] in (1, 3, 4):
                return t.numpy()[None, ...]
            if t.shape[1] in (1, 3, 4):
                return t.permute(0, 2, 3, 1).contiguous().numpy()[None, ...]
        # Decoded images are CHW float tensors; keep torch for materialize path.
        if t.dim() == 3:
            if t.shape[-1] in (1, 3, 4):
                return t.numpy()
            if t.shape[0] in (1, 3, 4):
                return t
        return t.numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return [_transport_output_to_numpy(v) for v in value]
    return value


def _build_route_context(req: Any) -> dict[str, Any]:
    """Extract the scalar metadata that route predicates evaluate against.

    Only JSON-ish scalars are exposed, so a predicate can never hold a
    reference to a tensor or a live object.
    """
    ctx: dict[str, Any] = {}

    sampling_params = getattr(req, "sampling_params", None)
    if sampling_params is not None and dataclasses.is_dataclass(sampling_params):
        for f in dataclasses.fields(sampling_params):
            value = getattr(sampling_params, f.name, None)
            if isinstance(value, (bool, int, float, str)) or value is None:
                ctx[f.name] = value

    for name in _EXTRA_ROUTE_FIELDS:
        value = getattr(req, name, None)
        if isinstance(value, (bool, int, float, str)):
            ctx[name] = value

    extra = getattr(req, "extra", None)
    if isinstance(extra, dict):
        ctx["extra"] = {
            k: v
            for k, v in extra.items()
            if isinstance(v, (bool, int, float, str)) or v is None
        }

    return ctx


def _request_id_from_frames(frames: list) -> str | None:
    try:
        metadata = json.loads(frames[0])
        return metadata.get("scalar_fields", {}).get("request_id")
    except (json.JSONDecodeError, IndexError, TypeError):
        return None
