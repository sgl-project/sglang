"""Registered AFD capability profiles and their composition factories."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from .cache import make_shape
from .config import AFDExecutionMode
from .contracts import (
    AFDError,
    AFDGraphStrategy,
    AFDModelAdapter,
    AFDPairedTopology,
    AFDRole,
    AFDShapeFactory,
    AFDTopology,
    AFDTransport,
    MetadataContract,
    contract_digest,
)
from .model_adapters.glm5 import Glm5AFDAdapter, matches_glm_moe_dsa
from .model_adapters.qwen3_moe import Qwen3AFDAdapter, matches_qwen3_moe
from .role_graph import AFDRoleGraphService, TorchRoleGraphDriver
from .transport import AFDPairedP2PTransport

QWEN3_PAIRED_C1_ID = "qwen3-moe/paired/c1"
GLM5_PAIRED_C1_ID = "glm-moe-dsa/paired/c1"
ROLE_GRAPH_STRATEGY_ID = "whole-role-graph-with-captured-transport-v1"


@dataclass(frozen=True)
class AFDCapabilityProfile:
    """Immutable peer-visible identity for one admitted AFD composition."""

    identity: str
    model_family: str
    shape_contract: str
    metadata_contract: MetadataContract
    transport_kind: str
    stage_plan: str
    stages: tuple[int, ...]
    fa_backends: tuple[str, ...]
    lanes: tuple[int, ...]
    max_nodes: int
    cache_policy: str

    def contract(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "model_family": self.model_family,
            "topology_kind": "paired-lanes-ffn-rank0-attention-rank1-v1",
            "shape_contract": self.shape_contract,
            "metadata_contract": self.metadata_contract.value,
            "transport_kind": self.transport_kind,
            "graph_strategy": ROLE_GRAPH_STRATEGY_ID,
            "stage_plan": self.stage_plan,
            "stages": list(self.stages),
            "fa_backends": list(self.fa_backends),
            "lanes": list(self.lanes),
            "attention_lane_policy": "positive-integer",
            "max_nodes": self.max_nodes,
            "cache_policy": self.cache_policy,
        }

    @property
    def digest(self) -> str:
        return contract_digest(self.contract())

    def validate_shape(self, *, config: Any) -> None:
        if config.stages not in self.stages:
            raise AFDError(
                "AFD_GRAPH_STAGES_UNSUPPORTED",
                f"stages={config.stages}",
            )
        if config.attention_backend not in self.fa_backends:
            raise AFDError(
                "AFD_ATTENTION_BACKEND_UNSUPPORTED",
                f"backend={config.attention_backend!r}",
            )
        if config.lanes not in self.lanes:
            raise AFDError(
                "AFD_TOPOLOGY_LANE_COUNT_UNSUPPORTED",
                f"lanes={config.lanes} admitted={self.lanes}",
            )
        if (
            type(config.attention_lane_count) is not int
            or config.attention_lane_count < 1
        ):
            raise AFDError(
                "AFD_TOPOLOGY_ATTENTION_LANE_COUNT_UNSUPPORTED",
                f"attention_lanes={config.attention_lane_count}",
            )


AdapterFactory = Callable[..., AFDModelAdapter]
GraphFactory = Callable[..., AFDGraphStrategy]
TransportFactory = Callable[..., AFDTransport]
TopologyFactory = Callable[..., AFDTopology]


@dataclass(frozen=True)
class AFDCapabilityFactories:
    """The narrow construction bundle selected by a capability profile."""

    profile: AFDCapabilityProfile
    model_matcher: Callable[[Any], bool]
    topology_factory: TopologyFactory
    adapter_factory: AdapterFactory
    transport_factory: TransportFactory
    graph_factory: GraphFactory
    shape_factory: AFDShapeFactory

    def make_topology(
        self,
        *,
        lanes: int = 1,
        attention_lanes: int | None = None,
    ) -> AFDTopology:
        topology = self.topology_factory(
            lanes=lanes,
            attention_lanes=attention_lanes,
        )
        topology.validate()
        return topology

    def validate_runtime(self, *, server_args: Any, config: Any) -> None:
        self.profile.validate_shape(config=config)
        role = AFDExecutionMode(server_args.afd_execution_mode).role
        if role is None:
            raise AFDError(
                "AFD_CAPABILITY_PROFILE_ROLE_REQUIRED",
                f"mode={server_args.afd_execution_mode!r}",
            )
        topology = self.make_topology(
            lanes=config.lanes,
            attention_lanes=config.attention_lanes,
        )
        expected = topology.expected_parallelism(role=role)
        actual = (
            server_args.tp_size,
            server_args.dp_size,
            server_args.ep_size,
            getattr(server_args, "pp_size", 1),
        )
        if actual != expected:
            raise AFDError(
                "AFD_TOPOLOGY_PARALLELISM_MISMATCH",
                (
                    f"role={role.value} lanes={config.lanes} "
                    f"attention_lanes={config.attention_lane_count} "
                    f"expected={expected} actual={actual}"
                ),
            )
        # A role may span hosts, but only evenly: SGLang sizes a cross-host tp
        # group by floor division, so a rank count that does not divide leaves
        # ranks stranded rather than failing.
        role_size = expected[0]
        if (
            server_args.nnodes < 1
            or server_args.nnodes > self.profile.max_nodes
            or role_size % server_args.nnodes
        ):
            raise AFDError(
                "AFD_TOPOLOGY_ROLE_NODE_SPLIT_UNSUPPORTED",
                f"role={role.value} role_size={role_size} "
                f"nnodes={server_args.nnodes} max_nodes={self.profile.max_nodes}",
            )
        expects_dp = topology.expects_dp_attention(role=role)
        if bool(getattr(server_args, "enable_dp_attention", False)) != expects_dp:
            raise AFDError(
                "AFD_TOPOLOGY_DP_ATTENTION_MISMATCH",
                f"role={role.value} lanes={config.lanes} expected={expects_dp}",
            )

    def make_adapter(
        self,
        *,
        role: AFDRole,
        model: Any,
        attention_backend: Any | None,
    ) -> AFDModelAdapter:
        adapter = self.adapter_factory(
            role=role,
            model=model,
            attention_backend=attention_backend,
        )
        adapter.validate_model()
        if adapter.metadata_contract is not self.profile.metadata_contract:
            raise AFDError("AFD_ADAPTER_METADATA_CONTRACT_MISMATCH")
        return adapter

    def make_transport(self, **kwargs: Any) -> AFDTransport:
        return self.transport_factory(**kwargs)

    def make_graph_strategy(self, **kwargs: Any) -> AFDGraphStrategy:
        return self.graph_factory(**kwargs)


def _make_role_graph(**kwargs: Any) -> AFDGraphStrategy:
    from sglang.srt.runtime_context import get_observability

    return AFDRoleGraphService(
        log_interval=get_observability().decode_log_interval,
        driver=TorchRoleGraphDriver(),
        **kwargs,
    )


QWEN3_PAIRED_C1 = AFDCapabilityProfile(
    identity=QWEN3_PAIRED_C1_ID,
    model_family="qwen3_moe",
    shape_contract="paired-lane-padded-stage-rows-v1",
    metadata_contract=MetadataContract.STANDARD_FA,
    transport_kind=("pynccl-p2p-dual-owned-streams-control-close-ack-bounded-abort-v1"),
    stage_plan="native-tbo-s2-lane-merged-return-v2",
    stages=(2,),
    fa_backends=("fa3", "fa4"),
    # Supported FFN degrees for the current native model/quantization paths;
    # these are not restrictions on the A/F topology or its ratio.
    lanes=(1, 2, 4, 8, 16, 32),
    max_nodes=8,
    cache_policy="native-capture-sizes-no-eviction-hard-hbm-v2",
)

_QWEN3_PAIRED_C1_FACTORIES = AFDCapabilityFactories(
    profile=QWEN3_PAIRED_C1,
    model_matcher=matches_qwen3_moe,
    topology_factory=AFDPairedTopology.paired,
    adapter_factory=Qwen3AFDAdapter,
    transport_factory=AFDPairedP2PTransport,
    graph_factory=_make_role_graph,
    shape_factory=make_shape,
)

GLM5_PAIRED_C1 = AFDCapabilityProfile(
    identity=GLM5_PAIRED_C1_ID,
    model_family="glm_moe_dsa",
    shape_contract="paired-lane-padded-stage-rows-v1",
    metadata_contract=MetadataContract.PRIVATE_DSA,
    transport_kind=("pynccl-p2p-dual-owned-streams-control-close-ack-bounded-abort-v1"),
    stage_plan="native-tbo-s2-lane-merged-return-v2",
    stages=(2,),
    fa_backends=("nsa",),
    # FFN EP/TP degrees must fit expert ownership and native quantized shards.
    # A has no experts; its own projections, DP LM head and capacity checks apply.
    lanes=(1, 2, 4, 8, 16, 32),
    max_nodes=8,
    cache_policy="native-capture-sizes-no-eviction-hard-hbm-v2",
)

_GLM5_PAIRED_C1_FACTORIES = AFDCapabilityFactories(
    profile=GLM5_PAIRED_C1,
    model_matcher=matches_glm_moe_dsa,
    topology_factory=AFDPairedTopology.paired,
    adapter_factory=Glm5AFDAdapter,
    transport_factory=AFDPairedP2PTransport,
    graph_factory=_make_role_graph,
    shape_factory=make_shape,
)

AFD_PROFILE_REGISTRY = MappingProxyType(
    {
        QWEN3_PAIRED_C1.identity: _QWEN3_PAIRED_C1_FACTORIES,
        GLM5_PAIRED_C1.identity: _GLM5_PAIRED_C1_FACTORIES,
    }
)


def registered_capability_profiles() -> tuple[AFDCapabilityProfile, ...]:
    return tuple(item.profile for item in AFD_PROFILE_REGISTRY.values())


def validate_startup_capabilities(
    *,
    server_args: Any,
    config: Any,
) -> tuple[AFDCapabilityProfile, ...]:
    """Admit the runtime when any registered profile accepts it.

    The model is not constructed yet, so the family cannot be resolved here.
    ``resolve_capability_factories`` performs the exact bind once it exists.
    """

    if not AFD_PROFILE_REGISTRY:
        raise AFDError("AFD_CAPABILITY_PROFILE_REGISTRY_EMPTY")
    admitted = []
    rejections = []
    for item in AFD_PROFILE_REGISTRY.values():
        try:
            item.validate_runtime(
                server_args=server_args,
                config=config,
            )
        except AFDError as exc:
            rejections.append((item.profile.identity, exc))
            continue
        admitted.append(item.profile)
    if not admitted:
        detail = " ".join(f"{identity}={exc.code}" for identity, exc in rejections)
        raise AFDError(
            "AFD_CAPABILITY_PROFILE_STARTUP_UNADMITTED",
            detail,
        ) from rejections[0][1]
    return tuple(admitted)


def resolve_capability_factories(
    *,
    model: Any,
    config: Any,
) -> AFDCapabilityFactories:
    matches = tuple(
        item for item in AFD_PROFILE_REGISTRY.values() if item.model_matcher(model)
    )
    if len(matches) != 1:
        raise AFDError(
            "AFD_CAPABILITY_PROFILE_UNRESOLVED",
            f"model={type(model).__name__} matches={len(matches)}",
        )
    factories = matches[0]
    factories.profile.validate_shape(config=config)
    return factories
