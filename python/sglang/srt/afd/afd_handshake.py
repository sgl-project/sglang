"""One-shot startup handshake between the attn and ffn instances.

Runs over ZMQ REQ (attn) / ROUTER (ffn). Users only deal with server API
addresses: the attn
side points ``--afd-ffn-addr`` at the ffn instance's ``--host:--port``, and
both sides derive the internal handshake port from the ffn API port with a
fixed delta (same pattern as ``ZMQ_TCP_PORT_DELTA``). Startup order is
arbitrary (the attn request is queued until the ffn side binds).
Exchanges identity/topology so that every later mismatch (different model,
different backend, incompatible world sizes) fails loudly at startup
instead of hanging mid-forward.
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

from sglang.srt.afd.afd_config import is_direct_afd_backend
from sglang.srt.afd.dispatcher import (
    build_decode_token_buckets,
    split_afd_batch_size,
)
from sglang.srt.environ import envs
from sglang.srt.utils.network import get_zmq_socket

logger = logging.getLogger(__name__)

# The handshake port is derived from the ffn instance's server API port so
# users never configure an extra port (cf. ZMQ_TCP_PORT_DELTA in server_args).
AFD_HANDSHAKE_PORT_DELTA = 300
# The sync (control plane) channel binds next to the handshake port.
SYNC_PORT_OFFSET = 1
# Rendezvous store of the joint attn∪ffn data-plane group (ffn rank 0 hosts).
DIST_PORT_OFFSET = 2
# Each additional attn session gets its own control port. Session-local NCCL
# communicators also get a distinct rendezvous port; Direct shares one.
SESSION_PORT_STRIDE = 2
_NVSHMEM_REMOTE_TRANSPORTS = frozenset({"none", "ibgda"})


def afd_handshake_port(ffn_api_port: int) -> int:
    port = ffn_api_port + AFD_HANDSHAKE_PORT_DELTA
    if port + DIST_PORT_OFFSET > 65535:
        port = ffn_api_port - AFD_HANDSHAKE_PORT_DELTA
    return port


@dataclass
class AFDHandshakeInfo:
    role: str  # AFDRole value
    model_identity: str
    # Number of data-plane ranks (attn: DP ranks, one per GPU since
    # attn_tp_size == 1 is enforced; ffn: local TP ranks).
    num_ranks: int
    # Node identity for each instance rank, in rank order. NVSHMEM uses this
    # to assign node-major PEs independently of the logical A/F rank space.
    rank_node_ids: list[str] = field(repr=False)
    afd_backend: str
    # Logical F-internal expert layout. The attn process advertises the same
    # layout after replacing its process-local backend with AFDAttnMoE.
    afd_ffn_moe_backend: str
    wire_dtype: str
    moe_layers: list[dict] = field(repr=False)
    direct_backend_version: str | None = field(default=None, repr=False)
    # F-authoritative per-layer activation quantization scales for native
    # Direct NVFP4 dispatch. An empty list selects the FP8 activation path.
    direct_nvfp4_input_scales: list[float] = field(default_factory=list, repr=False)
    # F selects the FP8 scale encoding for every A/F rank, including empty inputs.
    direct_deepep_fp8_scale_ue8m0: bool | None = None
    # F owns the GEMM layout; every dispatcher must use matching expert offsets.
    direct_deepep_expert_alignment: int = 128
    # F-authoritative communication budget. Its meaning is data-plane specific;
    # Direct DeepEP also uses it to size A-side Elastic kernels.
    afd_comm_sm_budget: int | None = field(default=None, repr=False)
    # A-authoritative wire capacities. FULL uses global-token capacities;
    # SCATTERED uses per-rank capacities.
    decode_token_buckets: list[int] = field(default_factory=list, repr=False)
    # Stable first dimensions captured by this role's local decode graphs.
    # Attn reports transport wire capacities; ffn reports expert capacities.
    decode_graph_token_buckets: list[int] = field(default_factory=list, repr=False)
    disable_cuda_graph_padding: bool = False
    # A-authoritative count used by the ffn side to allocate decode slots.
    decode_num_microbatches: int = 1
    # F-authoritative graph policy negotiated before A captures decode graphs.
    requires_eager_decode_communication: bool = False
    # Per-rank NVML fabric identity: (cluster UUID, clique ID). Missing or
    # incomplete information conservatively requires a remote transport.
    rank_nvshmem_fabric_keys: list[tuple[bytes, int] | None] = field(
        default_factory=list, repr=False
    )
    # F-authoritative transport selected from the joint NVSHMEM topology.
    nvshmem_remote_transport: str | None = field(default=None, repr=False)
    # A-authoritative fixed target-forward rows per request. Ordinary decode
    # uses one; static EAGLE target verification uses its draft-token count.
    decode_num_tokens_per_bs: int = 1
    # F selects one transport mode for every rank and lane.
    direct_deepep_allow_hybrid: bool = True


class _HandshakeRequest(msgspec.Struct, tag="request", forbid_unknown_fields=True):
    info: AFDHandshakeInfo
    dist_host: str


class _HandshakeReply(msgspec.Struct, tag="reply", forbid_unknown_fields=True):
    info: AFDHandshakeInfo
    dist_host: str
    sync_port: int
    dist_port: int
    nvshmem_uid: bytes | None
    session_id: int
    attn_members: tuple[AFDHandshakeInfo, ...] | None


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
    nvshmem_uid: bytes | None = field(default=None, repr=False)
    # Index assigned by the ffn side in handshake-arrival order. Each attn
    # instance forms its own session against a (possibly shared) ffn instance.
    session_id: int = 0
    # All attn instances sharing the ffn instance, in session order. Populated
    # when the data plane forms one static world across every member.
    attn_members: tuple[AFDHandshakeInfo, ...] | None = field(default=None, repr=False)

    @property
    def requires_eager_decode_communication(self) -> bool:
        ffn = self.local if self.local.role == "ffn" else self.peer
        return ffn.requires_eager_decode_communication

    @property
    def total_attn_ranks(self) -> int:
        if self.attn_members is None:
            attn = self.local if self.local.role == "attn" else self.peer
            return attn.num_ranks
        return sum(member.num_ranks for member in self.attn_members)

    @property
    def attn_instance_rank_offset(self) -> int:
        if self.attn_members is None:
            return 0
        if not 0 <= self.session_id < len(self.attn_members):
            raise RuntimeError(
                "AFD handshake session is outside the attn member roster: "
                f"session={self.session_id}, members={len(self.attn_members)}"
            )
        return sum(member.num_ranks for member in self.attn_members[: self.session_id])


def select_nvshmem_remote_transport(
    ffn: AFDHandshakeInfo,
    attn_members: tuple[AFDHandshakeInfo, ...],
) -> str:
    members = (ffn, *attn_members)
    node_ids = {node_id for info in members for node_id in info.rank_node_ids}
    if len(node_ids) <= 1:
        return "none"

    fabric_keys = [
        fabric_key for info in members for fabric_key in info.rank_nvshmem_fabric_keys
    ]
    has_complete_fabric_topology = all(
        len(info.rank_nvshmem_fabric_keys) == info.num_ranks for info in members
    )
    common_fabric_key = fabric_keys[0] if fabric_keys else None
    if (
        has_complete_fabric_topology
        and common_fabric_key is not None
        and any(common_fabric_key[0])
        and all(fabric_key == common_fabric_key for fabric_key in fabric_keys)
    ):
        return "none"
    return "ibgda"


@dataclass(frozen=True)
class AFDGraphCaptureVariant:
    attn_batch_size: int
    attn_wire_capacity: int
    child_wire_capacities: tuple[int, ...]
    child_compute_capacities: tuple[int | None, ...]


@dataclass(frozen=True)
class AFDGraphCapturePlan:
    # Kept in the attn runner's descending capture order. Different parent
    # buckets may intentionally share the same child shape tuple.
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
    ffn_buckets: list[int],
    wire_capacity: int,
    disable_padding: bool,
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

    wire_token_multiplier = _attn_wire_tokens_per_bs(attn)
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
    graph_capacities_by_session: list[set[int]] = []
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
        graph_capacities_by_session.append(session_graph_capacities)
        graph_maxima.append(max(session_graph_capacities, default=0))
        if scattered and use_deepep_low_latency and attn.decode_token_buckets:
            dispatch_maxima.append(max(attn.decode_token_buckets) * fan_in)

    graph_max_capacity = sum(graph_maxima)
    deepep_dispatch_limit = None
    if scattered and dispatch_maxima:
        max_single_session = max(dispatch_maxima)
        if max_single_session > 1024:
            raise RuntimeError(
                "AFD SCATTERED requires at most 1024 DeepEP dispatch tokens "
                "per rank for one attn session; "
                f"required={max_single_session}"
            )
        deepep_dispatch_limit = min(sum(dispatch_maxima), 1024)
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
        exact_capacities = {0}
        for session_capacities in graph_capacities_by_session:
            exact_capacities |= {
                base + capacity
                for base in tuple(exact_capacities)
                for capacity in session_capacities
                if base + capacity <= graph_max_capacity
            }
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


def _attn_wire_tokens_per_bs(attn: AFDHandshakeInfo) -> int:
    num_tokens_per_bs = attn.decode_num_tokens_per_bs
    if not isinstance(num_tokens_per_bs, int) or num_tokens_per_bs <= 0:
        raise RuntimeError(
            "AFD decode_num_tokens_per_bs must be a positive integer; "
            f"got {num_tokens_per_bs!r}"
        )
    rank_multiplier = 1 if attn.afd_ffn_moe_backend == "deepep" else attn.num_ranks
    return num_tokens_per_bs * rank_multiplier


def build_afd_attn_local_graph_variants(
    handshake: AFDHandshakeResult,
) -> tuple[AFDGraphCaptureVariant, ...]:
    if not handshake.requires_eager_decode_communication:
        return ()
    attn = handshake.local if handshake.local.role == "attn" else handshake.peer
    if attn.decode_num_microbatches <= 1:
        return ()
    return tuple(
        AFDGraphCaptureVariant(
            attn_batch_size=attn_batch_size,
            attn_wire_capacity=attn_wire_capacity,
            child_wire_capacities=child_wire_capacities,
            child_compute_capacities=(None,) * len(child_wire_capacities),
        )
        for attn_batch_size, attn_wire_capacity, child_wire_capacities in (
            _build_attn_graph_shapes(attn)
        )
        if len(child_wire_capacities) == attn.decode_num_microbatches
    )


def build_afd_graph_capture_plan(
    handshake: AFDHandshakeResult,
) -> AFDGraphCapturePlan:
    if handshake.requires_eager_decode_communication:
        # F chooses singleton/cohort membership only after receiving A's
        # control message. Keep NCCL outside the A model graph so that this
        # late decision does not require paired multi-session graph replay.
        return AFDGraphCapturePlan(
            attn_graph_variants=(),
            ffn_graph_variants=(),
        )

    attn = handshake.local if handshake.local.role == "attn" else handshake.peer
    ffn = handshake.local if handshake.local.role == "ffn" else handshake.peer
    attn_graph_shapes = _build_attn_graph_shapes(attn)
    attn_graph_capacities = tuple(shape[1] for shape in attn_graph_shapes)
    if is_direct_afd_backend(attn.afd_backend):
        if attn_graph_capacities and not ffn.decode_graph_token_buckets:
            raise RuntimeError(
                "AFD direct paired CUDA graph requires FULL graphs on the ffn instance"
            )
        return AFDGraphCapturePlan(
            attn_graph_variants=tuple(
                AFDGraphCaptureVariant(
                    attn_batch_size=attn_batch_size,
                    attn_wire_capacity=attn_wire_capacity,
                    child_wire_capacities=child_wire_capacities,
                    child_compute_capacities=(None,) * len(child_wire_capacities),
                )
                for (
                    attn_batch_size,
                    attn_wire_capacity,
                    child_wire_capacities,
                ) in attn_graph_shapes
            ),
            ffn_graph_variants=(),
        )

    transport_buckets = sorted(set(attn.decode_token_buckets))
    transport_capacities = tuple(reversed(transport_buckets))
    scattered = attn.afd_ffn_moe_backend == "deepep"
    wire_token_multiplier = _attn_wire_tokens_per_bs(attn)
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
        candidate_capacities = (
            transport_capacities
            if attn.afd_backend == "nvshmem"
            else attn_graph_capacities
        )
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
    missing_target_graph_capacities = tuple(
        missing for missing in missing_capacities if missing[0] in attn_graph_capacities
    )
    if attn.decode_num_tokens_per_bs > 1 and missing_target_graph_capacities:
        raise RuntimeError(
            "AFD fixed multi-token target forward requires an ffn graph bucket "
            "for every attn graph capacity: "
            f"missing_edge_to_ffn={missing_target_graph_capacities}, "
            f"ffn_buckets={ffn_buckets}; configure the F decode CUDA graph "
            "capacity in expert-input rows"
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
        # variants above. Unlike serial NVSHMEM, there are no standalone
        # transport-only F graph variants.
        standalone_ffn_variants: tuple[tuple[int, int], ...] = ()
    else:
        standalone_ffn_variants = tuple(ffn_capacity_by_wire.items())

    return AFDGraphCapturePlan(
        attn_graph_variants=tuple(attn_variants),
        ffn_graph_variants=standalone_ffn_variants,
    )


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
    if is_direct_afd_backend(attn.afd_backend) and (
        not attn.direct_backend_version
        or attn.direct_backend_version != ffn.direct_backend_version
    ):
        raise RuntimeError(
            "AFD handshake: direct backend component version mismatch, "
            f"backend={attn.afd_backend!r} "
            f"attn={attn.direct_backend_version!r} "
            f"ffn={ffn.direct_backend_version!r}"
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
    if (
        attn.afd_backend == "direct_deepep"
        and not ffn.direct_nvfp4_input_scales
        and not isinstance(ffn.direct_deepep_fp8_scale_ue8m0, bool)
    ):
        raise RuntimeError(
            "AFD handshake: direct_deepep requires the FFN FP8 scale encoding"
        )
    _check_fixed_multi_token_forward_compatible(attn, ffn)
    _check_moe_layers_compatible(attn.moe_layers, ffn.moe_layers)


def _check_fixed_multi_token_forward_compatible(
    attn: AFDHandshakeInfo,
    ffn: AFDHandshakeInfo,
) -> None:
    _attn_wire_tokens_per_bs(attn)
    if attn.decode_num_tokens_per_bs == 1:
        return
    if not attn.decode_graph_token_buckets:
        raise RuntimeError(
            "AFD fixed multi-token target forward requires attn FULL CUDA graphs"
        )
    if not ffn.decode_graph_token_buckets:
        raise RuntimeError(
            "AFD fixed multi-token target forward requires ffn FULL CUDA graphs"
        )


def _get_nvshmem_unique_id(remote_transport: str) -> bytes:
    from afd_nvshmem import get_unique_id

    unique_id = get_unique_id(remote_transport=remote_transport)
    if not isinstance(unique_id, bytes) or not unique_id:
        raise RuntimeError("afd-nvshmem returned an invalid unique ID")
    return unique_id


def _check_nvshmem_config(
    backend: str,
    unique_id: bytes | None,
    remote_transport: str | None,
) -> None:
    if backend == "nvshmem":
        if not isinstance(unique_id, bytes) or not unique_id:
            raise RuntimeError("AFD handshake: nvshmem backend requires a unique ID")
        if remote_transport not in _NVSHMEM_REMOTE_TRANSPORTS:
            raise RuntimeError(
                "AFD handshake: nvshmem backend requires a valid remote transport"
            )
    elif unique_id is not None or remote_transport is not None:
        raise RuntimeError(
            f"AFD handshake: backend {backend!r} returned unexpected NVSHMEM state"
        )


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
        "decode_num_tokens_per_bs",
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
        nvshmem_uid = reply.nvshmem_uid
        _check_nvshmem_config(
            local.afd_backend,
            nvshmem_uid,
            peer.nvshmem_remote_transport,
        )
        result = AFDHandshakeResult(
            local=local,
            peer=peer,
            dist_host=reply.dist_host,
            sync_port=reply.sync_port,
            dist_port=reply.dist_port,
            nvshmem_uid=nvshmem_uid,
            session_id=reply.session_id,
            attn_members=reply.attn_members,
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
    Backends with a static joint world receive the full member list; direct
    backends also share one data-plane rendezvous port across all sessions.
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
            if not is_direct_afd_backend(local.afd_backend):
                capacity_plan = build_afd_ffn_capacity_plan(
                    local,
                    attn_infos,
                    use_deepep_low_latency,
                )
                local.decode_graph_token_buckets = list(
                    capacity_plan.graph_token_buckets
                )
            if local.afd_backend == "nvshmem":
                nvshmem_remote_transport = select_nvshmem_remote_transport(
                    local, attn_infos
                )
                local.nvshmem_remote_transport = nvshmem_remote_transport
                nvshmem_uid = _get_nvshmem_unique_id(nvshmem_remote_transport)
                logger.info(
                    "AFD NVSHMEM remote transport selected: %s",
                    nvshmem_remote_transport,
                )
            else:
                nvshmem_remote_transport = None
                nvshmem_uid = None
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
            self._reply_error_all([entry[0] for entry in pending], str(e))
            raise

        attn_members = (
            attn_infos
            if local.afd_backend == "nvshmem"
            or is_direct_afd_backend(local.afd_backend)
            else None
        )
        results: list[AFDHandshakeResult] = []
        for session_id, (identity, peer, dist_host) in enumerate(pending):
            sync_port = self._port + SYNC_PORT_OFFSET + SESSION_PORT_STRIDE * session_id
            dist_port = self._port + DIST_PORT_OFFSET
            if not is_direct_afd_backend(local.afd_backend):
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
                            nvshmem_uid=nvshmem_uid,
                            session_id=session_id,
                            attn_members=attn_members,
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
                    nvshmem_uid=nvshmem_uid,
                    session_id=session_id,
                    attn_members=attn_members,
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
