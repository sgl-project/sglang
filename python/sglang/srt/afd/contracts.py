"""Narrow contracts shared by AFD orchestration components."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Callable, Protocol

import msgspec


class AFDError(RuntimeError):
    """Fail-closed AFD error with a stable machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        message = code if not detail else f"{code}: {detail}"
        super().__init__(message)


def contract_digest(value: dict[str, Any]) -> str:
    """Hash the canonical JSON shared by descriptor producers and consumers."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_non_speculative_batch(forward_batch: Any) -> None:
    if getattr(forward_batch, "spec_info", None) is not None:
        raise AFDError("AFD_MTP_SPECULATIVE_UNSUPPORTED")


class AFDRole(str, Enum):
    ATTENTION = "attention"
    FFN = "ffn"


class MetadataContract(str, Enum):
    STANDARD_FA = "standard-fa-padded-stage-refresh-v1"
    PRIVATE_DSA = "private-dsa-padded-stage-breakable-v1"


class AFDEndpoint(msgspec.Struct, frozen=True, kw_only=True):
    role: AFDRole
    ordinal: int
    transport_rank: int
    coordination_rank: int


class AFDTopology(Protocol):
    endpoints: tuple[AFDEndpoint, ...]
    lanes: int

    @property
    def coordination_world_size(self) -> int: ...

    @property
    def pair_world_size(self) -> int: ...

    def validate(self) -> None: ...

    def local(self, *, role: AFDRole, ordinal: int) -> AFDEndpoint: ...

    def peers(self, *, role: AFDRole, ordinal: int) -> tuple[AFDEndpoint, ...]: ...

    def expected_parallelism(self, *, role: AFDRole) -> tuple[int, int, int, int]: ...

    def expects_dp_attention(self, *, role: AFDRole) -> bool: ...


def attention_lane_group(
    *, attention_lanes: int, ffn_lanes: int, ffn_ordinal: int
) -> tuple[int, ...]:
    """Partition A lanes once, in F rank order, allowing empty ingress groups."""

    if ffn_ordinal < 0 or ffn_ordinal >= ffn_lanes:
        raise AFDError("AFD_TOPOLOGY_ROLE_IDENTITY_INVALID", f"ffn={ffn_ordinal}")
    width, extra = divmod(attention_lanes, ffn_lanes)
    start = ffn_ordinal * width + min(ffn_ordinal, extra)
    stop = start + width + (ffn_ordinal < extra)
    return tuple(range(start, stop))


class AFDPairedTopology(msgspec.Struct, frozen=True, kw_only=True):
    """Contiguous A ingress groups over F shards; all F ranks execute the FFN.

    Integral A/F layouts retain their original per-F wire groups. Other layouts
    use one F-first A/F wire world, including F ranks with no ingress. Empty
    ingress never removes a rank from the separate F compute collectives.
    """

    endpoints: tuple[AFDEndpoint, ...]
    lanes: int
    attention_lanes: int | None = None

    @property
    def ffn_size(self) -> int:
        return self.lanes

    @property
    def attention_size(self) -> int:
        return self.lanes if self.attention_lanes is None else self.attention_lanes

    @property
    def lanes_per_ffn(self) -> int:
        """Maximum ingress group width; uniform for legacy integral layouts."""

        return (self.attention_size + self.lanes - 1) // self.lanes

    @property
    def shared_wire(self) -> bool:
        return bool(self.attention_size % self.lanes)

    @classmethod
    def paired(
        cls,
        *,
        lanes: int = 1,
        attention_lanes: int | None = None,
    ) -> AFDPairedTopology:
        if type(lanes) is not int or lanes < 1:
            raise AFDError("AFD_TOPOLOGY_LANE_COUNT_INVALID", f"lanes={lanes!r}")
        if attention_lanes is not None and (
            type(attention_lanes) is not int or attention_lanes < 1
        ):
            raise AFDError(
                "AFD_TOPOLOGY_LANE_GROUP_RATIO_INVALID",
                f"attention_lanes={attention_lanes!r} lanes={lanes}",
            )
        attention_size = lanes if attention_lanes is None else attention_lanes
        shared = bool(attention_size % lanes)
        group = (attention_size + lanes - 1) // lanes
        return cls(
            endpoints=tuple(
                AFDEndpoint(
                    role=role,
                    ordinal=ordinal,
                    transport_rank=(
                        (ordinal if role == AFDRole.FFN else lanes + ordinal)
                        if shared
                        else (0 if role == AFDRole.FFN else 1 + ordinal % group)
                    ),
                    coordination_rank=(
                        ordinal if role == AFDRole.FFN else lanes + ordinal
                    ),
                )
                for role, size in (
                    (AFDRole.FFN, lanes),
                    (AFDRole.ATTENTION, attention_size),
                )
                for ordinal in range(size)
            ),
            lanes=lanes,
            attention_lanes=attention_lanes,
        )

    @property
    def coordination_world_size(self) -> int:
        return self.lanes + self.attention_size

    @property
    def pair_world_size(self) -> int:
        return (
            self.coordination_world_size if self.shared_wire else 1 + self.lanes_per_ffn
        )

    def validate(self) -> None:
        if (
            type(self.lanes) is not int
            or self.lanes < 1
            or type(self.attention_size) is not int
            or self.attention_size < 1
        ):
            raise AFDError("AFD_TOPOLOGY_PAIRED_LAYOUT_REQUIRED")
        expected = self.paired(
            lanes=self.lanes, attention_lanes=self.attention_lanes
        ).endpoints
        if self.endpoints != expected:
            raise AFDError(
                "AFD_TOPOLOGY_PAIRED_LAYOUT_REQUIRED",
                f"lanes={self.lanes} attention_lanes={self.attention_lanes} "
                f"endpoints={self.endpoints!r}",
            )

    def local(self, *, role: AFDRole, ordinal: int) -> AFDEndpoint:
        matches = tuple(
            item
            for item in self.endpoints
            if item.role == role and item.ordinal == ordinal
        )
        if len(matches) != 1:
            raise AFDError(
                "AFD_TOPOLOGY_ROLE_IDENTITY_INVALID",
                f"role={role.value} ordinal={ordinal} lanes={self.lanes} "
                f"attention_lanes={self.attention_size}",
            )
        return matches[0]

    def ingress_ffn(self, *, attention_ordinal: int) -> int:
        self.local(role=AFDRole.ATTENTION, ordinal=attention_ordinal)
        return next(
            rank
            for rank in range(self.lanes)
            if attention_ordinal in self.attention_lane_group(ffn_ordinal=rank)
        )

    def peers(self, *, role: AFDRole, ordinal: int) -> tuple[AFDEndpoint, ...]:
        self.local(role=role, ordinal=ordinal)
        if role == AFDRole.ATTENTION:
            return (
                self.local(
                    role=AFDRole.FFN,
                    ordinal=self.ingress_ffn(attention_ordinal=ordinal),
                ),
            )
        return tuple(
            self.local(role=AFDRole.ATTENTION, ordinal=lane)
            for lane in self.attention_lane_group(ffn_ordinal=ordinal)
        )

    def attention_lane_group(self, *, ffn_ordinal: int) -> tuple[int, ...]:
        return attention_lane_group(
            attention_lanes=self.attention_size,
            ffn_lanes=self.lanes,
            ffn_ordinal=ffn_ordinal,
        )

    def group_ordinal(self, *, role: AFDRole, ordinal: int) -> int:
        self.local(role=role, ordinal=ordinal)
        if self.shared_wire:
            return 0
        return (
            ordinal
            if role == AFDRole.FFN
            else self.ingress_ffn(attention_ordinal=ordinal)
        )

    def expected_parallelism(self, *, role: AFDRole) -> tuple[int, int, int, int]:
        if role == AFDRole.ATTENTION:
            return (self.attention_size, self.attention_size, 1, 1)
        return (self.lanes, 1, self.lanes, 1)

    def expects_dp_attention(self, *, role: AFDRole) -> bool:
        return role == AFDRole.ATTENTION and self.attention_size > 1


class AFDReason(str, Enum):
    ARMING = "AFD_TYPED_EAGER_ARMING"
    BUCKET_LIMIT = "AFD_TYPED_EAGER_BUCKET_LIMIT"
    HBM_LIMIT = "AFD_TYPED_EAGER_HBM_LIMIT"
    CAPTURE_FAILED = "AFD_TYPED_EAGER_CAPTURE_FAILED"
    REPLAY_FAILED = "AFD_TYPED_EAGER_REPLAY_FAILED"
    PARTIAL_CAPTURE = "AFD_TYPED_EAGER_PARTIAL_CAPTURE_ROLLBACK"
    METADATA_DRIFT = "AFD_TYPED_EAGER_METADATA_DRIFT"
    CACHE_GUARD = "AFD_TYPED_EAGER_CACHE_GUARD"
    FORWARD_MODE = "AFD_TYPED_EAGER_FORWARD_MODE"
    SHAPE_INVALID = "AFD_TYPED_EAGER_SHAPE_INVALID"
    CLOSED = "AFD_TYPED_EAGER_CLOSED"


class AFDExecutionKind(str, Enum):
    EAGER = "typed_eager"
    REPLAY = "graph_replay"


class AFDExecutionResult(msgspec.Struct, kw_only=True):
    value: Any
    kind: AFDExecutionKind
    reason: AFDReason | None = None
    bucket: str | None = None


class AFDShapeIdentity(Protocol):
    lane: int
    lane_stage_rows: tuple[tuple[int, ...], ...]
    lane_bucket_rows: tuple[tuple[int, ...], ...]
    hidden_size: int
    dtype: str
    tokens_per_request: int

    @property
    def stage_rows(self) -> tuple[int, ...]: ...

    @property
    def bucket_rows(self) -> tuple[int, ...]: ...

    @property
    def digest(self) -> str: ...

    def merge_plan(self, *, stage: int) -> tuple[int, ...]: ...


class AFDShapeFactory(Protocol):
    def __call__(
        self,
        *,
        lane: int,
        lane_rows: tuple[tuple[int, ...], ...],
        hidden_size: int,
        dtype: str,
        config: Any,
        capture_sizes: tuple[int, ...],
        tokens_per_request: int = 1,
    ) -> AFDShapeIdentity: ...


class AFDPairedShape(msgspec.Struct, frozen=True, kw_only=True):
    lane: int
    lane_stage_rows: tuple[tuple[int, ...], ...]
    lane_bucket_rows: tuple[tuple[int, ...], ...]
    hidden_size: int
    dtype: str
    # Grouping does not change the shared wire shape digest.
    ffn_lanes: int = 0
    # Wire field retained; this base runtime only admits one decode token/request.
    tokens_per_request: int = 1

    def __post_init__(self) -> None:
        if type(self.tokens_per_request) is not int or self.tokens_per_request != 1:
            raise AFDError("AFD_TOKENS_PER_REQUEST_INVALID")

    @property
    def attention_lanes(self) -> int:
        return len(self.lane_bucket_rows)

    @property
    def ffn_size(self) -> int:
        return self.ffn_lanes or self.attention_lanes

    @property
    def lanes_per_ffn(self) -> int:
        return (self.attention_lanes + self.ffn_size - 1) // self.ffn_size

    @property
    def stage_rows(self) -> tuple[int, ...]:
        return (
            self.lane_stage_rows[self.lane]
            if self.lane < self.attention_lanes
            else (0,) * len(self.lane_stage_rows[0])
        )

    @property
    def bucket_rows(self) -> tuple[int, ...]:
        return (
            self.lane_bucket_rows[self.lane]
            if self.lane < self.attention_lanes
            else (0,) * len(self.lane_bucket_rows[0])
        )

    @property
    def digest(self) -> str:
        fields = (
            "|".join(
                "x".join(str(value) for value in vector)
                for vector in self.lane_bucket_rows
            ),
            str(self.hidden_size),
            self.dtype,
        )
        return ":".join(fields)

    def group_lanes(self, *, ffn_ordinal: int) -> tuple[int, ...]:
        """This rank's ingress lanes, possibly empty; no A lane is replicated."""

        return attention_lane_group(
            attention_lanes=self.attention_lanes,
            ffn_lanes=self.ffn_size,
            ffn_ordinal=ffn_ordinal,
        )

    def group_bucket_rows(self, *, ffn_ordinal: int) -> tuple[int, ...]:
        """One padded width per (stage, lane) this FFN rank holds, stage-major.

        Stage-major so that at k == 1 it is this rank's own bucket vector, which
        is what the receive buffers were keyed on before fan-in existed.
        """

        lanes = self.group_lanes(ffn_ordinal=ffn_ordinal)
        if not lanes:
            # Device/dtype templates retained per stage, with no wire edge.
            return (0,) * len(self.lane_bucket_rows[0])
        return tuple(
            self.lane_bucket_rows[lane][stage]
            for stage in range(len(self.lane_bucket_rows[0]))
            for lane in lanes
        )

    def group_total_stage_rows(self, *, ffn_ordinal: int) -> tuple[int, ...]:
        """Live rows an FFN rank runs per stage: its whole lane group at once."""

        lanes = self.group_lanes(ffn_ordinal=ffn_ordinal)
        return tuple(
            sum(self.lane_stage_rows[lane][stage] for lane in lanes)
            for stage in range(len(self.lane_stage_rows[0]))
        )

    def group_total_bucket_rows(self, *, ffn_ordinal: int) -> tuple[int, ...]:
        """The padded counterpart of group_total_stage_rows."""

        lanes = self.group_lanes(ffn_ordinal=ffn_ordinal)
        return tuple(
            sum(self.lane_bucket_rows[lane][stage] for lane in lanes)
            for stage in range(len(self.lane_bucket_rows[0]))
        )

    def merge_plan(self, *, stage: int) -> tuple[int, ...]:
        """One collective width per F, preserving global A order and zero slots."""

        return tuple(
            sum(
                self.lane_bucket_rows[lane][stage]
                for lane in self.group_lanes(ffn_ordinal=ordinal)
            )
            for ordinal in range(self.ffn_size)
        )


class AFDModelAdapter(Protocol):
    """All model-family knowledge required by the shared pipeline."""

    role: AFDRole
    num_layers: int
    hidden_size: int
    metadata_contract: MetadataContract

    def validate_model(self) -> None: ...

    def attention_capability(
        self,
        *,
        configured_backend: str,
    ) -> dict[str, str]: ...

    def initialize_graph_metadata(self, *, max_rows: int) -> int: ...

    def split_step(
        self,
        *,
        hidden_states: Any,
        residual: Any,
        positions: Any,
        forward_batch: Any,
        stages: int,
    ) -> list[Any]: ...

    def local_compute(
        self,
        *,
        layer: int,
        stage: Any,
        hidden_states: Any,
        residual: Any,
        positions: Any = None,
    ) -> tuple[Any, ...]: ...

    def finish_layer(
        self,
        *,
        layer: int,
        stage: Any,
        ffn_output: Any,
        residual: Any,
    ) -> tuple[Any, Any]: ...

    def join_step(self, *, stages: list[Any]) -> tuple[Any, Any]: ...

    def prepare_stage(self, *, stage: Any) -> None: ...

    def metadata_guard(
        self,
        *,
        stage: Any,
        bucket_rows: int,
    ) -> AFDMetadataGuard | None: ...

    def lane_stage_rows(
        self,
        *,
        forward_batch: Any,
        local_rows: tuple[int, ...],
        lanes: int,
        lane: int,
    ) -> tuple[tuple[int, ...], ...]: ...

    def make_ffn_stages(
        self,
        *,
        descriptor: Any,
        buffers: tuple[Any, ...],
        lane: int,
        shape: AFDShapeIdentity,
    ) -> tuple[list[Any], list[Any]]: ...


class AFDStepDescriptor(msgspec.Struct, frozen=True, kw_only=True):
    kind: str
    step_id: int
    lane_stage_rows: tuple[tuple[int, ...], ...]
    hidden_size: int
    dtype: str
    num_layers: int
    # Shared verdict before any rank posts a captured transport round.
    graph_eligible: bool
    # Global batch mode, independent of graph eligibility (IDLE is decode).
    is_extend_in_batch: bool = False
    close_usage: dict[str, Any] | None = None
    # Wire field retained; this base runtime only admits one decode token/request.
    tokens_per_request: int = 1

    def __post_init__(self) -> None:
        self.validate_mode()

    def validate_mode(self) -> None:
        if self.kind not in ("STEP", "CAPTURE", "READY", "CLOSE"):
            raise AFDError("AFD_STEP_DESCRIPTOR_KIND_INVALID", self.kind)
        if self.kind == "CAPTURE" and not self.graph_eligible:
            raise AFDError("AFD_STARTUP_CAPTURE_INELIGIBLE")
        if type(self.tokens_per_request) is not int or self.tokens_per_request != 1:
            raise AFDError("AFD_TOKENS_PER_REQUEST_INVALID")
        if (
            type(self.is_extend_in_batch) is not bool
            or type(self.graph_eligible) is not bool
        ):
            raise AFDError("AFD_STEP_DESCRIPTOR_MODE_INVALID")
        if self.is_extend_in_batch and self.graph_eligible:
            raise AFDError(
                "AFD_STEP_DESCRIPTOR_MODE_MISMATCH", "extend cannot be graphed"
            )

    def stage_rows(self, *, lane: int) -> tuple[int, ...]:
        if lane < 0 or lane >= len(self.lane_stage_rows):
            raise AFDError(
                "AFD_STEP_DESCRIPTOR_LANE_INVALID",
                f"lane={lane} lanes={len(self.lane_stage_rows)}",
            )
        return self.lane_stage_rows[lane]


class AFDGraphStrategy(Protocol):
    role: AFDRole

    @property
    def capture_sizes(self) -> tuple[int, ...]: ...

    def finish_capture(self) -> None: ...

    @property
    def capturing(self) -> bool: ...

    @property
    def retains_backing(self) -> bool: ...

    def begin_step(
        self,
        *,
        step_id: int,
        shape: AFDShapeIdentity,
        eligible: bool,
        backing_hbm_bytes: int = 0,
        capture: bool = False,
    ) -> AFDReason | None: ...

    def ensure_backing_hbm(self, *, retained_hbm_bytes: int) -> bool: ...

    def execute_step(
        self,
        *,
        stage_args: tuple[tuple[Any, ...], ...],
        stage_rows: tuple[int, ...],
        forward_batches: tuple[Any, ...],
        compute: Callable[[tuple[tuple[Any, ...], ...]], tuple[tuple[Any, ...], ...]],
        metadata_guards: tuple[AFDMetadataGuard | None, ...],
    ) -> AFDExecutionResult:
        """Run one whole role step, capturing or replaying it as a unit.

        stage_args carries the entry tensors per stage and is empty per stage for
        the FFN role, whose inputs arrive over the wire inside the captured
        region rather than being staged by the host.
        """
        ...

    def end_step(self) -> AFDReason | None: ...

    def usage(self, *, status: str = "ACTIVE") -> dict[str, Any]: ...

    def close(self) -> dict[str, Any]: ...


class AFDTransport(Protocol):
    """Graph-external dispatch/return transport owned by the connector."""

    role: AFDRole
    lane: int

    @property
    def capturing(self) -> bool:
        """Whether this role's stream is inside a graph capture right now.

        The whole-step region runs eagerly during native warmup and again on
        any decline, so owning the step is not the same as capturing it. Only a
        capture draws the ordering edges a prefetched receive depends on, so the
        pipeline has to ask per step rather than infer it from the call site.
        """
        ...

    def validate_invariants(self) -> None: ...

    def begin_step(self, descriptor: Any | None) -> Any: ...

    def dispatch(self, tensor: Any) -> None: ...

    def receive_dispatch(
        self,
        buffers: tuple[Any, ...],
    ) -> Any: ...

    def return_result(self, tensor: Any) -> None: ...

    def receive_return(self, buffer: Any) -> Any: ...

    def wait(self, event: Any) -> None: ...

    def rejoin_streams(self) -> None:
        """Order the current stream after every side stream this role used."""
        ...

    def acquire_buffers(
        self,
        *,
        key: str,
        capacities: tuple[int, ...],
        hidden_size: int,
        dtype: Any,
        retain: bool,
    ) -> tuple[Any, ...]: ...

    def release_buffers(self, *, key: str) -> None: ...

    def buffer_hbm_bytes(self, *, key: str) -> int: ...

    def close(self) -> dict[str, Any]: ...

    def exchange_close(self, *, usage: dict[str, Any]) -> dict[str, Any]: ...


class AFDMetadataGuard(Protocol):
    @property
    def graph_forward_batch(self) -> Any: ...

    def capture(self, forward_batch: Any) -> None: ...

    def activate_in_graph(self) -> None: ...

    def prepare_replay(self, forward_batch: Any) -> None: ...

    def restore(self) -> None: ...

    def close(self) -> None: ...

    def assert_stable(self) -> None: ...


LocalCompute = Callable[..., tuple[Any, ...]]
