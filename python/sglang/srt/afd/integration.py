"""Registry-backed composition roots for the attention and FFN AFD roles."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
from functools import partial
from typing import Any

from sglang.srt.model_executor.cuda_graph_config import Backend, filter_capture_sizes
from sglang.srt.runtime_context import get_exec

from .config import AFDConfig
from .connector import AFDConnector
from .contracts import AFDError, AFDModelAdapter, AFDRole, contract_digest
from .pipeline import AFDAttentionPipeline, AFDFFNPipeline
from .profiles import (
    ROLE_GRAPH_STRATEGY_ID,
    AFDCapabilityFactories,
    AFDCapabilityProfile,
    resolve_capability_factories,
)


def _capture_sizes(graph) -> tuple[int, ...]:
    if graph.prefill.backend != Backend.DISABLED or graph.decode.backend not in (
        Backend.FULL,
        Backend.DISABLED,
    ):
        raise AFDError(
            "AFD_GRAPH_BACKEND_UNSUPPORTED", "use full decode and disabled prefill"
        )
    if graph.decode.backend == Backend.DISABLED:
        return ()
    sizes = filter_capture_sizes(graph.decode.bs, max_size=graph.decode.max_bs)
    return tuple(sorted({(size + 1) // 2 for size in sizes}))


logger = logging.getLogger(__name__)

# Incompatible profile, metadata, transport, graph, or lifecycle changes bump
# this explicit protocol revision. Source-control coordinates are not protocol.
_AFD_C1_ABI_REVISION = "afd-c1-abi-native-s2-startup-capture-r7"
_AFD_C1_ABI_CONTRACT = (
    "profile-factory-v1|native-s2-operations|fixed-stage-capture-plan|"
    "per-profile-metadata-contract|paired-lane-coordination-world|"
    "lane-stage-row-matrix|padded-lane-row-merge|lane-group-fanin|"
    "whole-role-graph-with-captured-transport|close-ack-bounded-abort|"
    "ffn-router-ep-reduce-scatter|peer-major-hidden|"
    "step-is-extend-in-batch|"
    "request-aligned-stage-buckets|startup-capture-ready|runtime-shared-graph-pool"
)
_AFD_C1_ABI = {
    "revision": _AFD_C1_ABI_REVISION,
    "contract_digest": hashlib.sha256(_AFD_C1_ABI_CONTRACT.encode()).hexdigest(),
}


def _emit_usage(receipt: dict[str, Any]) -> None:
    logger.info("AFD_FINAL_USAGE %s", json.dumps(receipt, sort_keys=True))


def _model_descriptor(
    *,
    role: AFDRole,
    adapter: AFDModelAdapter,
    profile: AFDCapabilityProfile,
    config: AFDConfig,
    dtype: Any,
    capture_sizes: tuple[int, ...],
) -> dict[str, Any]:
    value = str(dtype)
    runtime_contract = {
        "afd_abi": _AFD_C1_ABI,
        "capability_profile": profile.contract(),
        "capability_profile_digest": profile.digest,
        "stage_plan": {
            "stages": config.stages,
            "identity": profile.stage_plan,
        },
        # The fixed strategy remains part of the peer-visible contract so a
        # different graph implementation fails at startup before either role
        # replays a captured transport operation.
        "graph_strategy": ROLE_GRAPH_STRATEGY_ID,
        "capture_stage_sizes": capture_sizes,
        "max_retained_hbm_bytes": config.max_hbm_bytes,
        "attention": adapter.attention_capability(
            configured_backend=config.attention_backend,
        ),
        "close_timeout_seconds": config.close_timeout_seconds,
    }
    descriptor = {
        "role": role.value,
        "execution_mode": role.value,
        "model_family": profile.model_family,
        "num_layers": adapter.num_layers,
        "hidden_size": adapter.hidden_size,
        "dtype": value[6:] if value.startswith("torch.") else value,
        "runtime_contract": runtime_contract,
        "runtime_contract_digest": contract_digest(runtime_contract),
    }
    descriptor["capability_digest"] = contract_digest(descriptor)
    return descriptor


def _build_connector(
    *,
    role: AFDRole,
    lane: int,
    adapter: AFDModelAdapter,
    factories: AFDCapabilityFactories,
    config: AFDConfig,
    device: Any,
    dtype: Any,
    base_hbm_bytes: int,
    capture_sizes: tuple[int, ...],
) -> AFDConnector:
    topology = factories.make_topology(
        lanes=config.lanes,
        attention_lanes=config.attention_lanes,
    )
    transport = factories.make_transport(
        role=role,
        lane=lane,
        topology=topology,
        config=config,
        device=device,
        model_descriptor=_model_descriptor(
            role=role,
            adapter=adapter,
            profile=factories.profile,
            config=config,
            dtype=dtype,
            capture_sizes=capture_sizes,
        ),
    )
    try:
        graph_strategy = factories.make_graph_strategy(
            role=role,
            config=config,
            capture_sizes=capture_sizes,
            num_layers=adapter.num_layers,
            base_hbm_bytes=base_hbm_bytes,
            device=device,
        )
    except BaseException:
        transport.close()
        raise
    connector = AFDConnector(
        transport=transport,
        graph_strategy=graph_strategy,
        usage_emitter=_emit_usage,
    )
    atexit.register(connector.close)
    return connector


def build_attention_pipeline(
    *,
    model: Any,
    attention_backend: Any,
    config: AFDConfig,
    device: Any,
    dtype: Any,
    max_rows: int,
    lane: int,
) -> AFDAttentionPipeline:
    config.validate()
    factories = resolve_capability_factories(model=model, config=config)
    adapter = factories.make_adapter(
        role=AFDRole.ATTENTION,
        model=model,
        attention_backend=attention_backend,
    )
    capture_sizes = _capture_sizes(get_exec().graph.cuda_graph_config)
    if capture_sizes and capture_sizes[-1] > max_rows:
        raise AFDError(
            "AFD_CAPTURE_SIZE_EXCEEDS_REQUEST_POOL",
            f"stage_max={capture_sizes[-1]} pool={max_rows}",
        )
    base_hbm_bytes = (
        adapter.initialize_graph_metadata(max_rows=capture_sizes[-1])
        if capture_sizes
        else 0
    )
    connector = _build_connector(
        role=AFDRole.ATTENTION,
        lane=lane,
        adapter=adapter,
        factories=factories,
        config=config,
        device=device,
        dtype=dtype,
        base_hbm_bytes=base_hbm_bytes,
        capture_sizes=capture_sizes,
    )
    return AFDAttentionPipeline(
        adapter=adapter,
        connector=connector,
        config=config,
        shape_factory=partial(
            factories.shape_factory,
            capture_sizes=capture_sizes,
        ),
    )


def build_ffn_pipeline(
    *,
    model: Any,
    config: AFDConfig,
    device: Any,
    dtype: Any,
    lane: int,
) -> AFDFFNPipeline:
    config.validate()
    factories = resolve_capability_factories(model=model, config=config)
    adapter = factories.make_adapter(
        role=AFDRole.FFN,
        model=model,
        attention_backend=None,
    )
    capture_sizes = _capture_sizes(get_exec().graph.cuda_graph_config)
    connector = _build_connector(
        role=AFDRole.FFN,
        lane=lane,
        adapter=adapter,
        factories=factories,
        config=config,
        device=device,
        dtype=dtype,
        base_hbm_bytes=0,
        capture_sizes=capture_sizes,
    )
    return AFDFFNPipeline(
        adapter=adapter,
        connector=connector,
        config=config,
        device=device,
        dtype=dtype,
        shape_factory=partial(
            factories.shape_factory,
            capture_sizes=capture_sizes,
        ),
    )
