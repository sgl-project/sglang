"""Lane-paired NCCL transport with graph-external streams and events."""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from typing import Any, Callable

from .config import AFDConfig
from .contracts import (
    AFDError,
    AFDRole,
    AFDStepDescriptor,
    AFDTopology,
    contract_digest,
)


def _shared_step(descriptor: AFDStepDescriptor) -> tuple[Any, ...]:
    """The part of a descriptor that every lane in a group must agree on.

    Two fields are excluded because they describe the sending lane rather than
    the step: eligibility, which is that lane's own forward mode and rows, and
    close usage, which is that lane's own graph statistics.
    """

    descriptor.validate_mode()
    return (
        descriptor.kind,
        descriptor.step_id,
        descriptor.lane_stage_rows,
        descriptor.hidden_size,
        descriptor.dtype,
        descriptor.num_layers,
        descriptor.is_extend_in_batch,
        descriptor.tokens_per_request,
    )


class _AFDControlChannels:
    """Isolate native metadata queues by directed AFD edge.

    StatelessProcessGroup's point-to-point keys contain destination and sequence,
    but not source. AFD fan-in needs a separate namespace for each sender; retain
    native counters, expiration and store deadlines inside each edge.
    """

    def __init__(self, group: Any) -> None:
        self._group = group
        self.store = group.store
        self._channels: dict[tuple[int, int], Any] = {}

    def _channel(self, src: int, dst: int) -> Any:
        import torch

        from sglang.srt.distributed.utils import StatelessProcessGroup

        edge = (src, dst)
        if edge not in self._channels:
            self._channels[edge] = StatelessProcessGroup(
                rank=self._group.rank,
                world_size=self._group.world_size,
                store=torch.distributed.PrefixStore(
                    f"afd-control/{src}/{dst}", self.store
                ),
                data_expiration_seconds=self._group.data_expiration_seconds,
            )
        return self._channels[edge]

    def send_obj(self, obj: Any, dst: int) -> None:
        self._channel(self._group.rank, dst).send_obj(obj, dst=dst)

    def recv_obj(self, src: int) -> Any:
        return self._channel(src, self._group.rank).recv_obj(src=src)


def _store_host(rendezvous_host: str) -> str:
    """Where this rank would bind a store server its peers can reach."""

    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # A bind succeeds only for an address this machine actually owns, so this
        # is also the test for "the rendezvous host is me".
        probe.bind((rendezvous_host, 0))
        return rendezvous_host
    except OSError:
        pass
    finally:
        probe.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((rendezvous_host, 1))
        return str(probe.getsockname()[0])
    except OSError as exc:
        raise AFDError(
            "AFD_TRANSPORT_STORE_HOST_UNRESOLVED",
            f"rendezvous_host={rendezvous_host!r}",
        ) from exc
    finally:
        probe.close()


def _canonical_cuda_device(*, torch: Any, device: Any) -> Any:
    """Bind a CUDA device identity to one explicit process-local index."""

    try:
        if isinstance(device, bool):
            raise TypeError("boolean is not a CUDA device index")
        candidate = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AFDError(
            "AFD_TRANSPORT_CUDA_DEVICE_INVALID",
            f"device={device!r}",
        ) from exc
    if candidate.type != "cuda":
        raise AFDError(
            "AFD_TRANSPORT_CUDA_DEVICE_REQUIRED",
            f"device={candidate!r}",
        )
    try:
        index = (
            int(torch.cuda.current_device())
            if candidate.index is None
            else int(candidate.index)
        )
        device_count = int(torch.cuda.device_count())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AFDError("AFD_TRANSPORT_CUDA_DEVICE_INVALID") from exc
    if index < 0 or device_count < 1 or index >= device_count:
        raise AFDError(
            "AFD_TRANSPORT_CUDA_DEVICE_UNAVAILABLE",
            f"index={index} device_count={device_count}",
        )
    canonical = torch.device("cuda", index)
    if canonical.type != "cuda" or canonical.index != index:
        raise AFDError(
            "AFD_TRANSPORT_CUDA_DEVICE_IDENTITY_MISMATCH",
            f"expected=cuda:{index} actual={canonical!r}",
        )
    return canonical


class _BoundNCCL:
    """Fail-closed communicator binding with scoped enable restoration."""

    def __init__(self, *, comm: Any, stream: Any, device: Any) -> None:
        if not comm.available:
            raise AFDError("AFD_TRANSPORT_NCCL_UNAVAILABLE")
        if comm.device != device or getattr(stream, "device", device) != device:
            raise AFDError("AFD_TRANSPORT_NCCL_DEVICE_MISMATCH")
        self._comm = comm
        self._stream = stream
        self._identity = (
            comm.group,
            comm.comm,
            comm.device,
            comm.rank,
            comm.world_size,
            int(stream.cuda_stream),
            device,
        )

    def _validate(self) -> None:
        expected = self._identity
        actual = (
            self._comm.group,
            self._comm.comm,
            self._comm.device,
            self._comm.rank,
            self._comm.world_size,
            int(self._stream.cuda_stream),
            getattr(self._stream, "device", expected[-1]),
        )
        if (
            not self._comm.available
            or actual[0] is not expected[0]
            or actual[1] != expected[1]
            or actual[2:] != expected[2:]
        ):
            raise AFDError("AFD_TRANSPORT_NCCL_BINDING_DRIFT")

    def invoke(self, *, torch: Any, operation: Callable[[Any], None]) -> None:
        self._validate()
        disabled_before = self._comm.disabled
        try:
            with torch.cuda.stream(self._stream):
                with self._comm.change_state(enable=True):
                    if self._comm.disabled:
                        raise AFDError("AFD_TRANSPORT_NCCL_ENABLE_FAILED")
                    operation(self._comm)
        finally:
            if self._comm.disabled != disabled_before:
                raise AFDError("AFD_TRANSPORT_NCCL_SENTINEL_NOT_RESTORED")


class AFDPairedP2PTransport:
    """Point-to-point transport for one lane, owning one stream per direction."""

    _control_followers: tuple[int, ...] = ()
    _control_upstream: int | None = None

    def __init__(
        self,
        *,
        role: AFDRole,
        topology: AFDTopology,
        config: AFDConfig,
        device: Any,
        model_descriptor: dict[str, Any],
        lane: int = 0,
    ) -> None:
        import torch

        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from sglang.srt.distributed.utils import StatelessProcessGroup

        topology.validate()
        self.role = role
        self.lane = lane
        self._torch = torch
        self._topology = topology
        self._device = _canonical_cuda_device(torch=torch, device=device)
        local = topology.local(role=role, ordinal=lane)
        peers = topology.peers(role=role, ordinal=lane)
        self._rank = local.transport_rank
        self._peer_ranks = tuple(peer.transport_rank for peer in peers)
        self._peer_coordination_ranks = tuple(peer.coordination_rank for peer in peers)
        self._control_upstream = 0 if role == AFDRole.FFN and not peers else None
        self._control_followers = (
            tuple(
                topology.local(role=AFDRole.FFN, ordinal=j).coordination_rank
                for j in range(topology.lanes)
                if not topology.peers(role=AFDRole.FFN, ordinal=j)
            )
            if role == AFDRole.FFN and lane == 0
            else ()
        )
        # Receives and sends are posted in edge order, so that order has to be
        # the peers' own wire order or the two sides pair up different tensors.
        if (
            (not self._peer_ranks and role != AFDRole.FFN)
            or self._rank in self._peer_ranks
            or len(set(self._peer_ranks)) != len(self._peer_ranks)
            or list(self._peer_ranks) != sorted(self._peer_ranks)
        ):
            raise AFDError(
                "AFD_TRANSPORT_EDGE_LAYOUT_INVALID",
                f"rank={self._rank} peers={self._peer_ranks!r}",
            )
        self._coordination_rank = local.coordination_rank
        # One world spanning every lane, so a misconfigured lane is caught while
        # descriptors are exchanged rather than at its first collective.
        self._control = StatelessProcessGroup.create(
            host=config.rendezvous_host,
            port=config.rendezvous_port,
            rank=self._coordination_rank,
            world_size=topology.coordination_world_size,
            store_timeout_seconds=config.rendezvous_timeout_seconds,
        )
        descriptors = self._control.all_gather_obj(model_descriptor)
        self._validate_descriptors(
            descriptors=descriptors,
            local_descriptor=model_descriptor,
        )
        # A wire group's store server is bound by its rank 0 -- the group's FFN
        # rank -- on that rank's own machine, which stops being the rendezvous
        # host as soon as N outgrows one host's GPUs. Dial the owner where it
        # actually listens. FFN ordinal j is coordination rank j, so the group
        # ordinal indexes the owner directly.
        group_ordinal = topology.group_ordinal(role=role, ordinal=lane)
        store_hosts = self._control.all_gather_obj(_store_host(config.rendezvous_host))
        # Startup consensus keeps the native broadcast namespace. Subsequent
        # descriptors and acknowledgements must distinguish every source.
        self._control = _AFDControlChannels(self._control)
        comm_group = StatelessProcessGroup.create(
            host=store_hosts[group_ordinal] or config.rendezvous_host,
            port=config.rendezvous_port + 1 + group_ordinal,
            rank=self._rank,
            world_size=topology.pair_world_size,
            store_timeout_seconds=config.rendezvous_timeout_seconds,
        )
        # max_ctas caps only these two communicators. Sizing it matters because the
        # transport shares SMs with the model's own kernels on the same device.
        self._a2e = PyNcclCommunicator(
            group=comm_group,
            device=self._device,
            max_ctas=config.nccl_num_channels,
        )
        self._e2a = PyNcclCommunicator(
            group=comm_group,
            device=self._device,
            max_ctas=config.nccl_num_channels,
        )
        if not self._a2e.available or not self._e2a.available:
            raise AFDError("AFD_TRANSPORT_NCCL_UNAVAILABLE")
        self._a2e_stream = torch.cuda.Stream(device=self._device)
        self._e2a_stream = torch.cuda.Stream(device=self._device)
        self._forked_streams: list[Any] = []
        self.validate_invariants()
        self._a2e_binding = _BoundNCCL(
            comm=self._a2e,
            stream=self._a2e_stream,
            device=self._device,
        )
        self._e2a_binding = _BoundNCCL(
            comm=self._e2a,
            stream=self._e2a_stream,
            device=self._device,
        )
        self._config = config
        self._buffer_registry: dict[str, tuple[Any, ...]] = {}
        self._buffer_identities: dict[str, tuple[Any, ...]] = {}
        self._buffer_hbm: dict[str, int] = {}
        self._peer_close_usage: dict[str, Any] | None = None
        self._close_exchange: dict[str, Any] | None = None
        self._close_exchange_error: AFDError | None = None
        # Keep startup capture/READY bounded by the rendezvous timeout. Only
        # after the READY handshake may descriptor waits use the longer serving
        # idle timeout; a missing startup peer must never look like an idle lane.
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._retime_control_store(config.rendezvous_timeout_seconds)
        self._close_timeout_seconds = config.close_timeout_seconds
        self._closed = False
        self._stats = {
            "a2e_send": 0,
            "a2e_recv": 0,
            "e2a_send": 0,
            "e2a_recv": 0,
            "steps": 0,
        }
        self._warmup()

    def _validate_descriptors(
        self,
        *,
        descriptors: list[dict[str, Any]],
        local_descriptor: dict[str, Any],
    ) -> None:
        if len(descriptors) != self._topology.coordination_world_size:
            raise AFDError(
                "AFD_TRANSPORT_DESCRIPTOR_COUNT_INVALID",
                f"count={len(descriptors)}",
            )
        required = {
            "role",
            "execution_mode",
            "model_family",
            "num_layers",
            "hidden_size",
            "dtype",
            "runtime_contract",
            "runtime_contract_digest",
            "capability_digest",
        }
        if any(set(item) != required for item in descriptors):
            raise AFDError("AFD_TRANSPORT_DESCRIPTOR_SCHEMA_INVALID")
        roles = {item["role"] for item in descriptors}
        for item in descriptors:
            claimed = item["capability_digest"]
            body = {
                key: value for key, value in item.items() if key != "capability_digest"
            }
            actual = contract_digest(body)
            runtime_actual = contract_digest(item["runtime_contract"])
            if (
                claimed != actual
                or item["runtime_contract_digest"] != runtime_actual
                or item["execution_mode"] != item["role"]
            ):
                raise AFDError("AFD_TRANSPORT_CAPABILITY_DIGEST_INVALID")
        invariant_fields = (
            "model_family",
            "num_layers",
            "hidden_size",
            "dtype",
            "runtime_contract",
            "runtime_contract_digest",
        )
        drift = {
            field: tuple(item[field] for item in descriptors)
            for field in invariant_fields
            if any(item[field] != local_descriptor[field] for item in descriptors)
        }
        if (
            roles != {AFDRole.ATTENTION.value, AFDRole.FFN.value}
            or local_descriptor not in descriptors
            or drift
        ):
            raise AFDError(
                "AFD_TRANSPORT_STARTUP_IDENTITY_MISMATCH",
                f"roles={roles!r} drift={drift!r}",
            )

    def _retime_control_store(self, seconds: float) -> None:
        """Repoint the coordination store's wait deadline at the current phase.

        One store serves three phases with incompatible deadlines -- rendezvous
        waits for a peer that is still loading weights, serving waits for a
        request that may never come, close must not outlive its own budget -- so
        the deadline is a property of the phase, not of the group.
        """

        store = getattr(self._control, "store", None)
        if store is None:
            raise AFDError(
                "AFD_TRANSPORT_CONTROL_STORE_MISSING",
                f"control={type(self._control).__name__}",
            )
        store.set_timeout(timedelta(seconds=seconds))

    def validate_invariants(self) -> None:
        self._topology.validate()
        handles = (
            int(self._a2e_stream.cuda_stream),
            int(self._e2a_stream.cuda_stream),
        )
        if 0 in handles or len(set(handles)) != 2:
            raise AFDError(
                "AFD_TRANSPORT_DISTINCT_STREAMS_REQUIRED",
                f"handles={handles!r}",
            )

    def _warmup(self) -> None:
        tx = self._torch.zeros(1, dtype=self._torch.float32, device=self._device)
        if self.role == AFDRole.ATTENTION:
            rx = self._torch.empty_like(tx)
            self._send(
                tensor=tx,
                peer=self._peer_ranks[0],
                stream=self._a2e_stream,
                binding=self._a2e_binding,
            )
            self._recv(
                buffer=rx,
                peer=self._peer_ranks[0],
                stream=self._e2a_stream,
                binding=self._e2a_binding,
            )
        else:
            # Every edge, not just the first: an unexercised connection would
            # first fault inside the step loop instead of at startup.
            for peer in self._peer_ranks:
                rx = self._torch.empty_like(tx)
                received = self._recv(
                    buffer=rx,
                    peer=peer,
                    stream=self._a2e_stream,
                    binding=self._a2e_binding,
                )
                self._torch.cuda.current_stream(self._device).wait_event(received)
                self._send(
                    tensor=rx,
                    peer=peer,
                    stream=self._e2a_stream,
                    binding=self._e2a_binding,
                )
        self._a2e_stream.synchronize()
        self._e2a_stream.synchronize()

    def begin_step(
        self,
        descriptor: AFDStepDescriptor | None,
    ) -> AFDStepDescriptor:
        if self._closed:
            raise AFDError("AFD_TRANSPORT_CLOSED")
        # Per step, not per capture: eager steps fork side streams too and never
        # call rejoin_streams, so without this the list would grow forever.
        self._forked_streams.clear()
        if self.role == AFDRole.ATTENTION:
            if descriptor is None or descriptor.kind not in (
                "STEP",
                "CAPTURE",
                "READY",
            ):
                raise AFDError("AFD_TRANSPORT_STEP_DESCRIPTOR_REQUIRED")
            self._control.send_obj(
                descriptor,
                dst=self._peer_coordination_ranks[0],
            )
            result = descriptor
        else:
            if descriptor is not None:
                raise AFDError("AFD_TRANSPORT_FFN_DESCRIPTOR_MUST_BE_RECEIVED")
            # One per edge. Every lane derives the same matrix from the same DP
            # gather, so a divergence means the lanes left lockstep and the rows
            # this rank is about to receive no longer match the plan it was told.
            received = []
            sources = (
                (self._control_upstream,)
                if self._control_upstream is not None
                else self._peer_coordination_ranks
            )
            for source in sources:
                item = self._control.recv_obj(src=source)
                if not isinstance(item, AFDStepDescriptor):
                    raise AFDError(
                        "AFD_TRANSPORT_STEP_DESCRIPTOR_INVALID",
                        f"type={type(item).__name__}",
                    )
                received.append(item)
            # Every field except eligibility describes the step the group shares,
            # so those must agree. Eligibility does not: it is
            # `is_decode() and all(rows)` on the sending lane, and forward_mode is
            # not globally consistent under DP attention -- one lane can prefill a
            # freshly admitted request while its group peers decode. Reduce with
            # AND so this rank arms its graph only when every lane it serves is in
            # clean decode; over one element that is the value itself, which is
            # what the symmetric path always used.
            if any(
                _shared_step(item) != _shared_step(received[0]) for item in received[1:]
            ):
                raise AFDError(
                    "AFD_TRANSPORT_LANE_GROUP_DESCRIPTOR_DRIFT",
                    f"descriptors={received!r}",
                )
            result = AFDStepDescriptor(
                kind=received[0].kind,
                step_id=received[0].step_id,
                lane_stage_rows=received[0].lane_stage_rows,
                hidden_size=received[0].hidden_size,
                dtype=received[0].dtype,
                num_layers=received[0].num_layers,
                is_extend_in_batch=received[0].is_extend_in_batch,
                tokens_per_request=received[0].tokens_per_request,
                graph_eligible=all(item.graph_eligible for item in received),
                # The leader lane's, because only one can be carried and the
                # peers' copies differ by design.
                close_usage=received[0].close_usage,
            )
            for follower in self._control_followers:
                self._control.send_obj(result, dst=follower)
            if result.kind == "CLOSE":
                if result.close_usage is None:
                    raise AFDError("AFD_TRANSPORT_CLOSE_USAGE_MISSING")
                self._peer_close_usage = result.close_usage
        self._stats["steps"] += 1
        return result

    def capture_ready(self) -> None:
        """Complete the startup handshake after every planned graph is installed."""
        if self.role == AFDRole.ATTENTION:
            self.begin_step(
                AFDStepDescriptor(
                    kind="READY",
                    step_id=-1,
                    lane_stage_rows=(),
                    hidden_size=0,
                    dtype="",
                    num_layers=0,
                    graph_eligible=False,
                )
            )
            self._receive_capture_ready(self._peer_coordination_ranks[0])
        else:
            acknowledgement = {"event": "AFD_CAPTURE_READY"}
            if self._control_upstream is not None:
                self._control.send_obj(acknowledgement, dst=self._control_upstream)
            for follower in self._control_followers:
                self._receive_capture_ready(follower)
            for destination in self._peer_coordination_ranks:
                self._control.send_obj(acknowledgement, dst=destination)
        self._retime_control_store(self._idle_timeout_seconds)

    def _receive_capture_ready(self, source: int) -> None:
        reply = self._control.recv_obj(src=source)
        if reply != {"event": "AFD_CAPTURE_READY"}:
            detail = (
                f"event={reply.get('event')!r}"
                if isinstance(reply, dict)
                else f"kind={getattr(reply, 'kind', None)!r}"
            )
            raise AFDError(
                "AFD_CAPTURE_READY_ACK_INVALID",
                f"source={source} type={type(reply).__name__} {detail}",
            )

    def exchange_close(self, *, usage: dict[str, Any]) -> dict[str, Any]:
        if self._close_exchange is not None:
            return self._close_exchange
        if self._close_exchange_error is not None:
            raise self._close_exchange_error
        try:
            if self.role == AFDRole.ATTENTION:
                self._control.send_obj(
                    AFDStepDescriptor(
                        kind="CLOSE",
                        step_id=-1,
                        lane_stage_rows=(),
                        hidden_size=0,
                        dtype="",
                        num_layers=0,
                        graph_eligible=False,
                        close_usage=usage,
                    ),
                    dst=self._peer_coordination_ranks[0],
                )
                self._retime_control_store(self._close_timeout_seconds)
                try:
                    acknowledgement = self._control.recv_obj(
                        src=self._peer_coordination_ranks[0]
                    )
                except Exception as exc:
                    raise AFDError("AFD_TRANSPORT_CLOSE_ACK_TIMEOUT") from exc
                if (
                    not isinstance(acknowledgement, dict)
                    or acknowledgement.get("event") != "AFD_CLOSE_ACK"
                    or not isinstance(acknowledgement.get("usage"), dict)
                ):
                    raise AFDError("AFD_TRANSPORT_CLOSE_ACK_INVALID")
                self._close_exchange = acknowledgement["usage"]
            else:
                if self._peer_close_usage is None:
                    raise AFDError("AFD_TRANSPORT_CLOSE_NOT_RECEIVED")
                if self._control_upstream is not None or self._control_followers:
                    self._retime_control_store(self._close_timeout_seconds)
                acknowledgement = {"event": "AFD_CLOSE_ACK", "usage": usage}
                if self._control_upstream is not None:
                    self._control.send_obj(acknowledgement, dst=self._control_upstream)
                follower_usage = {}
                for follower in self._control_followers:
                    reply = self._control.recv_obj(src=follower)
                    if (
                        not isinstance(reply, dict)
                        or reply.get("event") != "AFD_CLOSE_ACK"
                        or not isinstance(reply.get("usage"), dict)
                    ):
                        raise AFDError("AFD_TRANSPORT_CLOSE_ACK_INVALID")
                    follower_usage[str(follower)] = reply["usage"]
                if follower_usage:
                    usage = {**usage, "control_followers": follower_usage}
                # Every lane in the group, or the ones left unanswered sit out
                # their whole close timeout before the service can exit.
                for destination in self._peer_coordination_ranks:
                    self._control.send_obj(
                        {
                            "event": "AFD_CLOSE_ACK",
                            "usage": usage,
                        },
                        dst=destination,
                    )
                self._close_exchange = self._peer_close_usage
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, AFDError)
                else AFDError("AFD_TRANSPORT_CLOSE_EXCHANGE_FAILED", repr(exc))
            )
            self._close_exchange_error = error
            raise error
        return self._close_exchange

    def dispatch(self, tensor: Any) -> None:
        self._require_role(AFDRole.ATTENTION, operation="dispatch")
        self._send(
            tensor=tensor,
            peer=self._peer_ranks[0],
            stream=self._a2e_stream,
            binding=self._a2e_binding,
        )
        self._stats["a2e_send"] += 1

    def receive_dispatch(self, buffers: tuple[Any, ...]) -> Any:
        """Receive hidden states in peer order; the event covers every peer."""
        self._require_role(AFDRole.FFN, operation="receive_dispatch")
        self._require_edge_count(buffers, operation="receive_dispatch")
        event = None
        for peer, buffer in zip(self._peer_ranks, buffers):
            event = self._recv(
                buffer=buffer,
                peer=peer,
                stream=self._a2e_stream,
                binding=self._a2e_binding,
            )
        self._stats["a2e_recv"] += len(buffers)
        return event

    def return_result(self, tensors: tuple[Any, ...]) -> None:
        self._require_role(AFDRole.FFN, operation="return_result")
        self._require_edge_count(tensors, operation="return_result")
        for peer, tensor in zip(self._peer_ranks, tensors):
            self._send(
                tensor=tensor,
                peer=peer,
                stream=self._e2a_stream,
                binding=self._e2a_binding,
            )
            self._stats["e2a_send"] += 1

    def receive_return(self, buffer: Any) -> Any:
        self._require_role(AFDRole.ATTENTION, operation="receive_return")
        event = self._recv(
            buffer=buffer,
            peer=self._peer_ranks[0],
            stream=self._e2a_stream,
            binding=self._e2a_binding,
        )
        self._stats["e2a_recv"] += 1
        return event

    def _require_edge_count(self, values: Any, *, operation: str) -> None:
        if len(values) != len(self._peer_ranks):
            raise AFDError(
                "AFD_TRANSPORT_EDGE_COUNT_INVALID",
                f"operation={operation} given={len(values)} "
                f"edges={len(self._peer_ranks)}",
            )

    def _capturing(self) -> bool:
        return self.capturing

    @property
    def capturing(self) -> bool:
        return bool(self._torch.cuda.is_current_stream_capturing())

    def _fork(self, *, stream: Any) -> None:
        """Order a side stream after the current one, and enrol it in a capture.

        CUDA only captures work on a stream that has waited on an event recorded
        by the capturing stream. Without that wait the side stream's work stays
        outside the graph, and the later join fails the capture with "dependency
        created on uncaptured work in another stream".
        """

        current = self._torch.cuda.current_stream(self._device)
        ready = self._torch.cuda.Event(enable_timing=False)
        ready.record(current)
        stream.wait_event(ready)
        if all(existing is not stream for existing in self._forked_streams):
            self._forked_streams.append(stream)

    def _send(
        self,
        *,
        tensor: Any,
        peer: int,
        stream: Any,
        binding: _BoundNCCL,
    ) -> None:
        self._fork(stream=stream)
        binding.invoke(
            torch=self._torch,
            operation=lambda comm: comm.send(tensor, dst=peer),
        )
        tensor.record_stream(stream)

    def _recv(
        self,
        *,
        buffer: Any,
        peer: int,
        stream: Any,
        binding: _BoundNCCL,
    ) -> Any:
        if self._capturing():
            # Only under capture. Eagerly this fork would order the receive
            # behind every kernel already queued on the compute stream, so the
            # peer's send would block until local compute drained and the two
            # side streams would stop overlapping transfer with compute at all.
            self._fork(stream=stream)
        buffer.record_stream(stream)
        binding.invoke(
            torch=self._torch,
            operation=lambda comm: comm.recv(buffer, src=peer),
        )
        ready = self._torch.cuda.Event(enable_timing=False)
        ready.record(stream)
        return ready

    def wait(self, event: Any) -> None:
        if event is not None:
            self._torch.cuda.current_stream(self._device).wait_event(event)

    def rejoin_streams(self) -> None:
        """Order the current stream after every side stream this step forked.

        Multi-stream capture requires every auxiliary stream to fork from and
        rejoin the capturing stream. A role only sends on one communicator and
        only receives on the other, so neither direction closes the loop by
        itself; this is the join for both. Only forked streams are joined --
        recording an event on a stream that never entered the capture is itself
        uncaptured work, and would fail the capture this call exists to complete.
        """

        current = self._torch.cuda.current_stream(self._device)
        for stream in self._forked_streams:
            landed = self._torch.cuda.Event(enable_timing=False)
            landed.record(stream)
            current.wait_event(landed)
        self._forked_streams.clear()

    def acquire_buffers(
        self,
        *,
        key: str,
        capacities: tuple[int, ...],
        hidden_size: int,
        dtype: Any,
        retain: bool,
    ) -> tuple[Any, ...]:
        identity = (capacities, hidden_size, dtype, self._device)

        def allocate():
            return tuple(
                self._torch.empty((rows, hidden_size), dtype=dtype, device=self._device)
                for rows in capacities
            )

        if self._closed:
            raise AFDError("AFD_TRANSPORT_CLOSED")
        if not retain:
            return self._allocate_receive_backing(allocate)
        existing = self._buffer_registry.get(key)
        if existing is not None:
            if self._buffer_identities[key] != identity:
                raise AFDError("AFD_TRANSPORT_BUFFER_IDENTITY_DRIFT")
            return existing
        allocated_before = self._torch.cuda.memory_allocated(self._device)
        reserved_before = self._torch.cuda.memory_reserved(self._device)
        buffers = self._allocate_receive_backing(allocate)
        self._buffer_registry[key] = buffers
        self._buffer_identities[key] = identity
        self._buffer_hbm[key] = max(
            sum(buffer.numel() * buffer.element_size() for buffer in buffers),
            self._torch.cuda.memory_allocated(self._device) - allocated_before,
            self._torch.cuda.memory_reserved(self._device) - reserved_before,
        )
        return buffers

    def _allocate_receive_backing(
        self, allocate: Callable[[], tuple[Any, ...]]
    ) -> tuple[Any, ...]:
        buffers = allocate()
        # A cached allocation can have prior work queued on the allocating
        # stream. record_stream protects its later release, not its first write
        # on the receiving stream. Order that write once per new backing.
        # Retained lookups and eager per-layer receives add no dependency here.
        stream = self._a2e_stream if self.role == AFDRole.FFN else self._e2a_stream
        if any(buffer.numel() for buffer in buffers):
            self._fork(stream=stream)
        return buffers

    def release_buffers(self, *, key: str) -> None:
        self._buffer_registry.pop(key, None)
        self._buffer_identities.pop(key, None)
        self._buffer_hbm.pop(key, None)
        self._torch.cuda.empty_cache()

    def buffer_hbm_bytes(self, *, key: str) -> int:
        return self._buffer_hbm.get(key, 0)

    def _require_role(self, role: AFDRole, *, operation: str) -> None:
        if self.role != role:
            raise AFDError(
                "AFD_TRANSPORT_DIRECTION_INVALID",
                f"role={self.role.value} operation={operation}",
            )

    def close(self) -> dict[str, Any]:
        if self._closed:
            return dict(self._stats)
        outcomes: dict[str, BaseException | None] = {}
        threads = []
        communicators = (("a2e", self._a2e), ("e2a", self._e2a))

        # ncclCommAbort can block indefinitely when the peer is already gone, and a
        # synchronous C call cannot be interrupted. Running it on a joinable worker
        # is what makes the deadline below enforceable; moving it back to the main
        # thread turns a bounded close into an unbounded hang.
        def abort(name: str, communicator: Any) -> None:
            try:
                communicator.nccl.ncclCommAbort(communicator.comm)
                outcomes[name] = None
            except BaseException as exc:
                outcomes[name] = exc

        deadline = time.monotonic() + self._close_timeout_seconds
        for name, communicator in communicators:
            if communicator is None or not communicator.available:
                continue
            thread = threading.Thread(
                target=abort,
                args=(name, communicator),
                daemon=True,
                name=f"afd-nccl-abort-{name}",
            )
            thread.start()
            threads.append((name, thread))
        for _, thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        timed_out = tuple(name for name, thread in threads if thread.is_alive())
        failed = tuple(
            name
            for name, _ in threads
            if name in outcomes and outcomes[name] is not None
        )
        for _, communicator in communicators:
            if communicator is not None:
                communicator.available = False
                communicator.disabled = True
        self._buffer_registry.clear()
        self._buffer_identities.clear()
        self._buffer_hbm.clear()
        self._a2e_binding = None
        self._e2a_binding = None
        self._a2e = None
        self._e2a = None
        self._control = None
        self._closed = True
        if timed_out:
            raise AFDError(
                "AFD_TRANSPORT_NATIVE_CLOSE_TIMEOUT",
                f"communicators={timed_out!r}",
            )
        if failed:
            raise AFDError(
                "AFD_TRANSPORT_NATIVE_CLOSE_FAILED",
                f"communicators={failed!r}",
            ) from outcomes[failed[0]]
        return dict(self._stats)
