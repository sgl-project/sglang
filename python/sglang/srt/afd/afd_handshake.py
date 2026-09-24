"""One-shot typed startup handshake between attention and FFN instances.

The attention instance connects to the FFN API address with a fixed port
offset. The handshake checks model and topology before creating the joint
NCCL communicator, so incompatible peers fail before the first forward.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import msgspec
import zmq

from sglang.srt.afd.dispatcher import build_decode_token_buckets, split_afd_batch_size
from sglang.srt.environ import envs
from sglang.srt.utils.network import get_zmq_socket

logger = logging.getLogger(__name__)

AFD_HANDSHAKE_PORT_DELTA = 300
SYNC_PORT_OFFSET = 1
DIST_PORT_OFFSET = 2
SESSION_PORT_STRIDE = 2


@dataclass
class AFDHandshakeInfo:
    role: str
    model_identity: str
    num_ranks: int
    afd_backend: str
    afd_ffn_moe_backend: str
    wire_dtype: str
    moe_layers: list[dict] = field(repr=False)
    # A wire capacities: global tokens for FULL, per-rank tokens for SCATTERED.
    decode_token_buckets: list[int] = field(default_factory=list, repr=False)
    # Attn reports wire capacities; FFN reports expert-input capacities.
    decode_graph_token_buckets: list[int] = field(default_factory=list, repr=False)
    disable_cuda_graph_padding: bool = False
    decode_num_microbatches: int = 1
    afd_comm_sm_budget: int | None = None


class _HandshakeRequest(msgspec.Struct, tag="request", forbid_unknown_fields=True):
    info: AFDHandshakeInfo
    dist_host: str


class _HandshakeReply(msgspec.Struct, tag="reply", forbid_unknown_fields=True):
    info: AFDHandshakeInfo
    dist_host: str
    sync_port: int
    dist_port: int
    session_id: int = 0


class _HandshakeError(msgspec.Struct, tag="error", forbid_unknown_fields=True):
    error: str


_handshake_request_decoder = msgspec.msgpack.Decoder(_HandshakeRequest)
_handshake_reply_decoder = msgspec.msgpack.Decoder(_HandshakeReply | _HandshakeError)


@dataclass
class AFDHandshakeResult:
    local: AFDHandshakeInfo
    peer: AFDHandshakeInfo
    dist_host: str
    sync_port: int
    dist_port: int
    session_id: int = 0


@dataclass(frozen=True)
class AFDGraphCaptureVariant:
    attn_batch_size: int
    attn_wire_capacity: int
    child_wire_capacities: tuple[int, ...]
    child_compute_capacities: tuple[int | None, ...]


@dataclass(frozen=True)
class AFDGraphCapturePlan:
    # Match the attention runner's descending capture order.
    attn_graph_variants: tuple[AFDGraphCaptureVariant, ...]
    ffn_graph_variants: tuple[tuple[int, int], ...]

    @property
    def enabled(self) -> bool:
        return bool(self.attn_graph_variants or self.ffn_graph_variants)


@dataclass(frozen=True)
class AFDFFNCapacityPlan:
    graph_token_buckets: tuple[int, ...]
    deepep_dispatch_limit: int | None


def _select_ffn_graph_capacity(
    ffn_buckets: list[int], wire_capacity: int, disable_padding: bool
) -> int | None:
    index = bisect.bisect_left(ffn_buckets, wire_capacity)
    if index == len(ffn_buckets):
        return None
    compute_capacity = ffn_buckets[index]
    if disable_padding and compute_capacity != wire_capacity:
        return None
    return compute_capacity


def _build_attn_graph_shapes(
    attn: AFDHandshakeInfo,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    attn_graph_capacities = tuple(
        sorted(set(attn.decode_graph_token_buckets), reverse=True)
    )
    transport_buckets = sorted(set(attn.decode_token_buckets))
    transport_capacities = tuple(reversed(transport_buckets))
    if not set(attn_graph_capacities).issubset(transport_capacities):
        raise RuntimeError(
            "AFD attn graph capacities must be transport capacities: "
            f"graph={attn_graph_capacities}, "
            f"transport={transport_capacities}"
        )

    wire_token_multiplier = (
        1 if attn.afd_ffn_moe_backend == "deepep" else attn.num_ranks
    )
    shapes = []
    for attn_wire_capacity in attn_graph_capacities:
        if attn_wire_capacity % wire_token_multiplier != 0:
            raise RuntimeError(
                "AFD attn graph capacity is not divisible by the fixed wire "
                "rows per request: "
                f"capacity={attn_wire_capacity}, "
                f"rows_per_request={wire_token_multiplier}"
            )
        attn_batch_size = attn_wire_capacity // wire_token_multiplier
        num_microbatches = attn.decode_num_microbatches
        if num_microbatches > 1 and attn_batch_size >= num_microbatches:
            child_batch_sizes = split_afd_batch_size(attn_batch_size, num_microbatches)
            child_wire_capacities = tuple(
                transport_buckets[
                    bisect.bisect_left(
                        transport_buckets,
                        child_batch_size * wire_token_multiplier,
                    )
                ]
                for child_batch_size in child_batch_sizes
            )
        else:
            child_wire_capacities = (attn_wire_capacity,)
        shapes.append((attn_batch_size, attn_wire_capacity, child_wire_capacities))
    return tuple(shapes)


def build_afd_ffn_capacity_plan(
    ffn: AFDHandshakeInfo,
    attn_members: Sequence[AFDHandshakeInfo],
    use_deepep_low_latency: bool,
) -> AFDFFNCapacityPlan:
    """Derive staged F graph and DeepEP capacities from A admission limits."""
    graph_enabled = bool(ffn.decode_graph_token_buckets)
    scattered = ffn.afd_ffn_moe_backend == "deepep"
    graph_capacities: set[int] = set()
    graph_maxima: list[int] = []
    dispatch_maxima: list[int] = []

    for attn in attn_members:
        fan_in = (
            (attn.num_ranks + ffn.num_ranks - 1) // ffn.num_ranks if scattered else 1
        )
        session_graph_capacities = {
            child_wire_capacity * fan_in
            for _, _, child_wire_capacities in _build_attn_graph_shapes(attn)
            for child_wire_capacity in child_wire_capacities
        }
        graph_capacities.update(session_graph_capacities)
        graph_maxima.append(max(session_graph_capacities, default=0))
        if scattered and use_deepep_low_latency and attn.decode_token_buckets:
            dispatch_maxima.append(max(attn.decode_token_buckets) * fan_in)

    graph_max_capacity = max(graph_maxima, default=0)
    deepep_dispatch_limit = None
    if scattered and dispatch_maxima:
        max_single_session = max(dispatch_maxima)
        if max_single_session > 1024:
            raise RuntimeError(
                "AFD SCATTERED requires at most 1024 DeepEP dispatch tokens "
                "per rank for one attn session; "
                f"required={max_single_session}"
            )
        deepep_dispatch_limit = max_single_session
        # Preserve native DeepEP's default capacity for small admission limits;
        # shrinking its expert buffers can produce incorrect masked-GEMM results.
        deepep_dispatch_limit = max(
            deepep_dispatch_limit,
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.default,
        )
        graph_max_capacity = min(graph_max_capacity, deepep_dispatch_limit)
        graph_capacities = {
            capacity
            for capacity in graph_capacities
            if capacity <= deepep_dispatch_limit
        }

    if not graph_enabled or not graph_max_capacity:
        graph_buckets: tuple[int, ...] = ()
    elif ffn.disable_cuda_graph_padding:
        exact_capacities = set(graph_capacities)
        graph_buckets = tuple(
            sorted(
                capacity
                for capacity in exact_capacities
                if 0 < capacity <= graph_max_capacity
            )
        )
    else:
        graph_buckets = tuple(
            build_decode_token_buckets(
                local_batch_sizes=sorted(graph_capacities),
                max_local_requests=graph_max_capacity,
                wire_token_multiplier=1,
            )
        )

    return AFDFFNCapacityPlan(
        graph_token_buckets=graph_buckets,
        deepep_dispatch_limit=deepep_dispatch_limit,
    )


def build_afd_graph_capture_plan(
    handshake: AFDHandshakeResult,
) -> AFDGraphCapturePlan:

    attn = handshake.local if handshake.local.role == "attn" else handshake.peer
    ffn = handshake.local if handshake.local.role == "ffn" else handshake.peer
    attn_graph_shapes = _build_attn_graph_shapes(attn)
    attn_graph_capacities = tuple(shape[1] for shape in attn_graph_shapes)

    scattered = attn.afd_ffn_moe_backend == "deepep"
    wire_token_multiplier = (
        1 if attn.afd_ffn_moe_backend == "deepep" else attn.num_ranks
    )
    ffn_max_fan_in = (
        (attn.num_ranks + ffn.num_ranks - 1) // ffn.num_ranks if scattered else 1
    )
    ffn_graph_compatible = not (
        scattered
        and ffn.disable_cuda_graph_padding
        and attn.num_ranks % ffn.num_ranks != 0
    )
    ffn_buckets = sorted(set(ffn.decode_graph_token_buckets))
    if attn.decode_num_microbatches > 1:
        # Eligible buckets become multi-child parent variants below; only
        # undersized batches keep a single-child pairing.
        candidate_capacities = tuple(
            capacity
            for capacity in attn_graph_capacities
            if capacity // wire_token_multiplier < attn.decode_num_microbatches
        )
    else:
        candidate_capacities = attn_graph_capacities
    missing_capacities: list[tuple[int, int]] = []
    ffn_capacity_by_wire: dict[int, int] = {}
    for wire_capacity in candidate_capacities:
        compute_capacity = (
            _select_ffn_graph_capacity(
                ffn_buckets,
                wire_capacity * ffn_max_fan_in,
                ffn.disable_cuda_graph_padding,
            )
            if ffn_graph_compatible
            else None
        )
        if compute_capacity is None:
            missing_capacities.append((wire_capacity, wire_capacity * ffn_max_fan_in))
            continue
        ffn_capacity_by_wire[wire_capacity] = compute_capacity

    if attn.afd_backend == "nccl" and candidate_capacities and not ffn_graph_compatible:
        raise RuntimeError(
            "AFD NCCL communication graph with non-uniform SCATTERED "
            "fan-in requires F-side CUDA graph padding; remove "
            "--disable-cuda-graph-padding"
        )
    if attn.afd_backend == "nccl" and missing_capacities:
        raise RuntimeError(
            "AFD NCCL communication graph requires an ffn graph bucket for "
            "every attn graph capacity: "
            f"missing_edge_to_ffn={missing_capacities}, "
            f"ffn_buckets={ffn_buckets}"
        )
    attn_variants: list[AFDGraphCaptureVariant] = []
    for (
        attn_batch_size,
        attn_wire_capacity,
        child_wire_capacities,
    ) in attn_graph_shapes:
        if len(child_wire_capacities) > 1:
            child_compute_capacities = tuple(
                (
                    _select_ffn_graph_capacity(
                        ffn_buckets,
                        child_wire_capacity * ffn_max_fan_in,
                        ffn.disable_cuda_graph_padding,
                    )
                    if ffn_graph_compatible
                    else None
                )
                for child_wire_capacity in child_wire_capacities
            )
            if any(capacity is None for capacity in child_compute_capacities):
                raise RuntimeError(
                    "AFD microbatch parent communication graph requires an ffn "
                    "graph bucket for every child: "
                    f"attn_batch_size={attn_batch_size}, "
                    f"child_wire_capacities={child_wire_capacities}, "
                    f"ffn_buckets={ffn_buckets}"
                )
        else:
            child_wire_capacities = (attn_wire_capacity,)
            child_compute_capacities = (ffn_capacity_by_wire.get(attn_wire_capacity),)

        attn_variants.append(
            AFDGraphCaptureVariant(
                attn_batch_size=attn_batch_size,
                attn_wire_capacity=attn_wire_capacity,
                child_wire_capacities=child_wire_capacities,
                child_compute_capacities=child_compute_capacities,
            )
        )

    if attn.decode_num_microbatches > 1:
        # All overlap F graphs are captured in the paired A/F phase via the attn
        # variants above.
        standalone_ffn_variants: tuple[tuple[int, int], ...] = ()
    else:
        standalone_ffn_variants = tuple(ffn_capacity_by_wire.items())

    return AFDGraphCapturePlan(
        attn_graph_variants=tuple(attn_variants),
        ffn_graph_variants=standalone_ffn_variants,
    )


def afd_handshake_port(ffn_api_port: int) -> int:
    port = ffn_api_port + AFD_HANDSHAKE_PORT_DELTA
    if port + DIST_PORT_OFFSET > 65535:
        port = ffn_api_port - AFD_HANDSHAKE_PORT_DELTA
    return port


def parse_ffn_addr(ffn_addr: str) -> Tuple[str, int]:
    host, _, port = ffn_addr.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"Invalid --afd-ffn-addr {ffn_addr!r}, expected host:port")
    return host, int(port)


def compute_model_identity(model_path: str) -> str:
    """Identity for the A/F model-match check.

    Based on config.json content, NOT on loaded weights: the two sides load
    disjoint weight subsets, so a weight hash could never match.
    """
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
    else:
        # Remote (HF hub) paths are resolved lazily; fall back to the path
        # string until the checkpoint metadata is available locally.
        digest = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    return f"{os.path.basename(os.path.normpath(model_path))}:{digest}"


def _check_compatible(local: AFDHandshakeInfo, peer: AFDHandshakeInfo) -> None:
    if local.role == peer.role:
        raise RuntimeError(
            "AFD handshake: expected one attn role and one ffn role, "
            f"both sides are {local.role!r}"
        )

    attn = local if local.role == "attn" else peer
    ffn = local if local.role == "ffn" else peer

    if attn.model_identity != ffn.model_identity:
        raise RuntimeError(
            f"AFD handshake: model mismatch, attn={attn.model_identity!r} "
            f"ffn={ffn.model_identity!r}"
        )
    if attn.afd_backend != ffn.afd_backend:
        raise RuntimeError(
            f"AFD handshake: --afd-backend mismatch, attn={attn.afd_backend!r} "
            f"ffn={ffn.afd_backend!r}"
        )
    if attn.afd_ffn_moe_backend != ffn.afd_ffn_moe_backend:
        raise RuntimeError(
            "AFD handshake: --moe-a2a-backend mismatch, "
            f"attn_expected_ffn={attn.afd_ffn_moe_backend!r} "
            f"ffn_actual={ffn.afd_ffn_moe_backend!r}"
        )
    if attn.wire_dtype != ffn.wire_dtype:
        raise RuntimeError(
            f"AFD handshake: wire dtype mismatch, attn={attn.wire_dtype!r} "
            f"ffn={ffn.wire_dtype!r}"
        )
    if attn.num_ranks <= 0 or ffn.num_ranks <= 0:
        raise RuntimeError(
            "AFD handshake: positive rank counts are required, "
            f"attn={attn.num_ranks} ffn={ffn.num_ranks}"
        )
    _check_moe_layers_compatible(attn.moe_layers, ffn.moe_layers)


def _check_moe_layers_compatible(attn: list[dict], ffn: list[dict]) -> None:
    if len(attn) != len(ffn):
        raise RuntimeError(
            f"AFD handshake: MoE layer count mismatch, attn={len(attn)} ffn={len(ffn)}"
        )

    for attn_layer, ffn_layer in zip(attn, ffn):
        if attn_layer != ffn_layer:
            raise RuntimeError(
                "AFD handshake: MoE layer mismatch, "
                f"attn={attn_layer!r} ffn={ffn_layer!r}"
            )


def _check_attn_members_compatible(
    attn_members: tuple[AFDHandshakeInfo, ...],
) -> None:
    baseline = attn_members[0]
    fields = (
        "num_ranks",
        "decode_token_buckets",
        "decode_graph_token_buckets",
        "disable_cuda_graph_padding",
        "decode_num_microbatches",
    )
    for session_id, member in enumerate(attn_members[1:], start=1):
        for name in fields:
            if getattr(member, name) != getattr(baseline, name):
                raise RuntimeError(
                    "AFD requires homogeneous attn instances: "
                    f"session={session_id}, field={name}"
                )


def run_handshake_attn(ffn_addr: str, local: AFDHandshakeInfo) -> AFDHandshakeResult:
    """Attn side (global sender rank only): connect and exchange info."""
    host, api_port = parse_ffn_addr(ffn_addr)
    port = afd_handshake_port(api_port)
    timeout_ms = envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.get() * 1000
    socket = get_zmq_socket(
        zmq.Context.instance(), zmq.REQ, f"tcp://{host}:{port}", bind=False
    )
    try:
        logger.info("AFD handshake: connecting to ffn peer at %s:%d ...", host, port)
        socket.send(msgspec.msgpack.encode(_HandshakeRequest(local, host)))
        if not socket.poll(timeout_ms):
            raise TimeoutError(
                f"AFD handshake: no reply from ffn peer at {ffn_addr} "
                f"(handshake port {port}) within {timeout_ms // 1000}s "
                f"(is the ffn instance up?)"
            )
        reply = _handshake_reply_decoder.decode(socket.recv())
        if isinstance(reply, _HandshakeError):
            raise RuntimeError(f"AFD handshake rejected by ffn peer: {reply.error}")
        peer = reply.info
        _check_compatible(local, peer)
        result = AFDHandshakeResult(
            local=local,
            peer=peer,
            dist_host=reply.dist_host,
            sync_port=reply.sync_port,
            dist_port=reply.dist_port,
            session_id=reply.session_id,
        )
        logger.info(
            "AFD handshake done: peer_role=%s peer_ranks=%d model=%s session=%d",
            peer.role,
            peer.num_ranks,
            local.model_identity,
            result.session_id,
        )
        return result
    finally:
        socket.close(linger=0)


class AFDHandshakeServer:
    """Ffn side (rank 0 only): bind once, accept all sessions in one round.

    Uses a ROUTER socket so every attn peer's request can be collected before
    any reply is sent. Sessions are numbered in handshake-arrival order.
    Each session uses dedicated control and data-plane rendezvous ports.
    """

    def __init__(self, port: int):
        self._port = port
        self._socket = get_zmq_socket(
            zmq.Context.instance(), zmq.ROUTER, f"tcp://*:{port}", bind=True
        )

    def accept_all(
        self,
        local: AFDHandshakeInfo,
        num_sessions: int = 1,
        use_deepep_low_latency: bool = False,
    ) -> list[AFDHandshakeResult]:
        pending: list[tuple[bytes, AFDHandshakeInfo, str]] = []
        while len(pending) < num_sessions:
            logger.info(
                "AFD handshake: waiting for attn peer %d/%d on port %d ...",
                len(pending) + 1,
                num_sessions,
                self._port,
            )
            identity, _, payload = self._socket.recv_multipart()
            try:
                request = _handshake_request_decoder.decode(payload)
                peer = request.info
                _check_compatible(local, peer)
            except (msgspec.DecodeError, RuntimeError) as e:
                self._reply_error_all(
                    [identity] + [entry[0] for entry in pending], str(e)
                )
                raise
            pending.append((identity, peer, request.dist_host))

        attn_infos = tuple(peer for _, peer, _ in pending)
        try:
            _check_attn_members_compatible(attn_infos)
            capacity_plan = build_afd_ffn_capacity_plan(
                local,
                attn_infos,
                use_deepep_low_latency,
            )
            local.decode_graph_token_buckets = list(capacity_plan.graph_token_buckets)
        except (RuntimeError, TypeError, ValueError) as e:
            self._reply_error_all([entry[0] for entry in pending], str(e))
            raise

        results: list[AFDHandshakeResult] = []
        for session_id, (identity, peer, dist_host) in enumerate(pending):
            sync_port = self._port + SYNC_PORT_OFFSET + SESSION_PORT_STRIDE * session_id
            dist_port = self._port + DIST_PORT_OFFSET
            dist_port += SESSION_PORT_STRIDE * session_id
            self._socket.send_multipart(
                [
                    identity,
                    b"",
                    msgspec.msgpack.encode(
                        _HandshakeReply(
                            info=local,
                            dist_host=dist_host,
                            sync_port=sync_port,
                            dist_port=dist_port,
                            session_id=session_id,
                        )
                    ),
                ]
            )
            logger.info(
                "AFD handshake done: peer_role=%s peer_ranks=%d model=%s session=%d",
                peer.role,
                peer.num_ranks,
                local.model_identity,
                session_id,
            )
            results.append(
                AFDHandshakeResult(
                    local=local,
                    peer=peer,
                    dist_host=dist_host,
                    sync_port=sync_port,
                    dist_port=dist_port,
                    session_id=session_id,
                )
            )
        return results

    def _reply_error_all(self, identities: list[bytes], error: str) -> None:
        """Fail every collected attn peer fast instead of leaving it to
        time out waiting for a reply that will never come."""
        for identity in identities:
            self._socket.send_multipart(
                [identity, b"", msgspec.msgpack.encode(_HandshakeError(error))]
            )

    def close(self) -> None:
        self._socket.close(linger=0)
