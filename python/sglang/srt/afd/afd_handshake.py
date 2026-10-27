"""One-shot typed startup handshake between attention and FFN instances.

The attention instance connects to the FFN API address with a fixed port
offset. The handshake checks model and topology before creating the joint
NCCL communicator, so incompatible peers fail before the first forward.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Tuple

import msgspec
import zmq

from sglang.srt.environ import envs
from sglang.srt.utils.network import get_zmq_socket

logger = logging.getLogger(__name__)

AFD_HANDSHAKE_PORT_DELTA = 300
SYNC_PORT_OFFSET = 1
DIST_PORT_OFFSET = 2


@dataclass
class AFDHandshakeInfo:
    role: str
    model_identity: str
    num_ranks: int
    afd_backend: str
    afd_ffn_moe_backend: str
    wire_dtype: str
    moe_layers: list[dict] = field(repr=False)
    # A admission bound: global tokens for FULL, per-rank tokens for SCATTERED.
    decode_token_buckets: list[int] = field(default_factory=list, repr=False)


class _HandshakeRequest(msgspec.Struct, tag="request", forbid_unknown_fields=True):
    info: AFDHandshakeInfo
    dist_host: str


class _HandshakeReply(msgspec.Struct, tag="reply", forbid_unknown_fields=True):
    info: AFDHandshakeInfo
    dist_host: str
    sync_port: int
    dist_port: int


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


@dataclass(frozen=True)
class AFDFFNCapacityPlan:
    deepep_dispatch_limit: int | None


def build_afd_ffn_capacity_plan(
    ffn: AFDHandshakeInfo,
    attn: AFDHandshakeInfo,
    use_deepep_low_latency: bool,
) -> AFDFFNCapacityPlan:
    """Derive the staged DeepEP capacity from A admission and rank fan-in."""
    deepep_dispatch_limit = None
    if (
        ffn.afd_ffn_moe_backend == "deepep"
        and use_deepep_low_latency
        and attn.decode_token_buckets
    ):
        fan_in = (attn.num_ranks + ffn.num_ranks - 1) // ffn.num_ranks
        deepep_dispatch_limit = max(attn.decode_token_buckets) * fan_in
        if deepep_dispatch_limit > 1024:
            raise RuntimeError(
                "AFD SCATTERED requires at most 1024 DeepEP dispatch tokens "
                "per rank for one attn session; "
                f"required={deepep_dispatch_limit}"
            )
        # Preserve native DeepEP's default capacity for small admission limits.
        deepep_dispatch_limit = max(
            deepep_dispatch_limit,
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.default,
        )
    return AFDFFNCapacityPlan(deepep_dispatch_limit=deepep_dispatch_limit)


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
        )
        logger.info(
            "AFD handshake done: peer_role=%s peer_ranks=%d model=%s",
            peer.role,
            peer.num_ranks,
            local.model_identity,
        )
        return result
    finally:
        socket.close(linger=0)


class AFDHandshakeServer:
    """FFN rank zero accepts one attention instance before serving steps."""

    def __init__(self, port: int):
        self._port = port
        self._socket = get_zmq_socket(
            zmq.Context.instance(), zmq.ROUTER, f"tcp://*:{port}", bind=True
        )

    def accept(self, local: AFDHandshakeInfo) -> AFDHandshakeResult:
        logger.info("AFD handshake: waiting for attn peer on port %d", self._port)
        identity, _, payload = self._socket.recv_multipart()
        try:
            request = _handshake_request_decoder.decode(payload)
            peer = request.info
            _check_compatible(local, peer)
        except (msgspec.DecodeError, RuntimeError) as exc:
            self._socket.send_multipart(
                [identity, b"", msgspec.msgpack.encode(_HandshakeError(str(exc)))]
            )
            raise

        result = AFDHandshakeResult(
            local=local,
            peer=peer,
            dist_host=request.dist_host,
            sync_port=self._port + SYNC_PORT_OFFSET,
            dist_port=self._port + DIST_PORT_OFFSET,
        )
        self._socket.send_multipart(
            [
                identity,
                b"",
                msgspec.msgpack.encode(
                    _HandshakeReply(
                        info=local,
                        dist_host=result.dist_host,
                        sync_port=result.sync_port,
                        dist_port=result.dist_port,
                    )
                ),
            ]
        )
        logger.info(
            "AFD handshake done: peer_role=%s peer_ranks=%d model=%s",
            peer.role,
            peer.num_ranks,
            local.model_identity,
        )
        return result

    def close(self) -> None:
        self._socket.close(linger=0)
