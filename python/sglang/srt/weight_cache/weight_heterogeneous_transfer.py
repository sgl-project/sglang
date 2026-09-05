# SPDX-License-Identifier: Apache-2.0
"""Immutable heterogeneous weight transfer between daemons.

Source daemons collectively publish one address-bearing runtime manifest over
a lightweight TCP registry. Target daemons construct their requested parallel
layout with the dummy loader, reject overlapping physical GPUs on the same node,
then use Mooncake's manifest planner and Transfer Engine reader to pull exactly
the ranges needed by each target rank. No Engine control-plane participation is
required and both daemon-owned models remain immutable.

EP and MoE-TP are represented as hierarchical Mooncake split/ownership axes.
SGLang attention DP also reuses the global TP workers, but a partially sharded
and replicated attention layout remains ambiguous and is rejected explicitly,
as are MoE-DP and attention context parallelism.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import msgspec

from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

from .mooncake_weight_adapter import (
    build_mooncake_placement_and_bindings,
    immutable_weight_allocation_guards,
)
from .protocol import recv_msg, send_msg
from .weight_runtime_manifest import (
    WeightParallelTopology,
    create_weight_runtime_manifest_builder,
    model_identity_from_config,
)

logger = logging.getLogger(__name__)


class WeightHeterogeneousTransferError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightParallelLayout:
    """Process and weight ownership dimensions managed by weight cache."""

    tp_size: int
    dp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1

    def __post_init__(self) -> None:
        if min(self.tp_size, self.dp_size, self.pp_size, self.ep_size) < 1:
            raise ValueError("TP, DP, PP and EP sizes must be positive")
        if self.tp_size % self.dp_size != 0:
            raise ValueError(
                f"tp_size={self.tp_size} must be divisible by dp_size={self.dp_size}"
            )
        if self.tp_size % self.ep_size != 0:
            raise ValueError(
                f"tp_size={self.tp_size} must be divisible by ep_size={self.ep_size}"
            )

    @property
    def world_size(self) -> int:
        # Attention DP and EP partition/reuse the TP group; neither adds a
        # daemon process axis.
        return self.tp_size * self.pp_size

def rebuild_transferred_weight_state(model: Any) -> None:
    """Recompute non-persistent tensors derived from immutable parameters.

    Normal disk loaders update these objects while copying each checkpoint
    weight.  A manifest transfer writes parameter storage directly, so the
    same derived state must be rebuilt after the pull and before readiness.
    """
    post_load_weights = getattr(model, "post_load_weights", None)
    if callable(post_load_weights):
        # Rebuild model-owned derived tensors using the same public hook as
        # checkpoint-engine weight updates. This does not re-run generic
        # quantization transforms.
        post_load_weights()

    from sglang.srt.model_loader.loader import PreshardedModelLoader

    PreshardedModelLoader._rebind_parameter_aliases(model)


def _register_weight_fragment_allocations(
    transfer_engine: Any,
    fragments: Sequence[Any],
) -> tuple[tuple[int, int], ...]:
    """Register full CUDA allocator regions owning manifest fragments.

    Transfer Engine can report success yet mishandle tiny fragmented CUDA
    registrations.  Registering the owning cudaMalloc segment matches the
    proven sglang-manifest path and also deduplicates aliased weights.
    """
    import torch

    fragment_ranges = sorted(
        {
            (int(fragment.address), int(fragment.address + fragment.nbytes))
            for fragment in fragments
        }
    )
    allocator_snapshot = torch.cuda.memory.memory_snapshot()
    active_blocks: list[tuple[int, int, int]] = []
    for segment_index, segment in enumerate(allocator_snapshot):
        for block in segment.get("blocks", ()):
            address = block.get("address", -1)
            size = block.get("size", -1)
            if (
                type(address) is int
                and type(size) is int
                and address >= 0
                and size > 0
                and block.get("state") == "active_allocated"
            ):
                active_blocks.append((address, address + size, segment_index))
    active_blocks.sort()
    block_starts = [block[0] for block in active_blocks]
    selected_segments = set()
    for begin, end in fragment_ranges:
        index = bisect_right(block_starts, begin) - 1
        if index < 0 or end > active_blocks[index][1]:
            raise WeightHeterogeneousTransferError(
                f"manifest range {begin}:{end} is not covered by an active CUDA allocation"
            )
        selected_segments.add(active_blocks[index][2])

    registered_memory_ranges: list[tuple[int, int]] = []
    try:
        for segment_index in sorted(selected_segments):
            segment = allocator_snapshot[segment_index]
            segment_begin = segment.get("address", -1)
            segment_size = segment.get("total_size", -1)
            if (
                type(segment_begin) is int
                and type(segment_size) is int
                and segment_begin >= 0
                and segment_size > 0
                and transfer_engine.register_memory(segment_begin, segment_size) == 0
            ):
                registered_memory_ranges.append((segment_begin, segment_size))
                continue
            merged: list[tuple[int, int]] = []
            for begin, end, block_segment_index in active_blocks:
                if block_segment_index != segment_index:
                    continue
                if merged and merged[-1][1] == begin:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((begin, end))
            for begin, end in merged:
                result = transfer_engine.register_memory(begin, end - begin)
                if result != 0:
                    raise WeightHeterogeneousTransferError(
                        f"Mooncake register_memory failed for {begin}+{end - begin}: {result}"
                    )
                registered_memory_ranges.append((begin, end - begin))

        registered_intervals = sorted(
            (address, address + nbytes) for address, nbytes in registered_memory_ranges
        )
        registered_starts = [item[0] for item in registered_intervals]
        for begin, end in fragment_ranges:
            index = bisect_right(registered_starts, begin) - 1
            if index < 0 or end > registered_intervals[index][1]:
                raise WeightHeterogeneousTransferError(
                    f"manifest range {begin}:{end} is not covered by registered memory"
                )
        return tuple(registered_memory_ranges)
    except BaseException:
        for address, _ in reversed(registered_memory_ranges):
            try:
                transfer_engine.unregister_memory(address)
            except Exception:
                logger.exception("failed to undo Mooncake memory registration")
        raise


def validate_weight_heterogeneous_transfer_configuration(
    *,
    tp_size: int,
    dp_size: int,
    ep_size: int,
    pp_size: int,
    enable_dp_attention: bool,
    moe_dp_size: int = 1,
    attn_cp_size: int = 1,
    nnodes: int = 1,
    quantization: Optional[str] = None,
) -> None:
    try:
        WeightParallelLayout(
            tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, ep_size=ep_size
        )
    except ValueError as error:
        raise WeightHeterogeneousTransferError(str(error)) from error

    unsupported = []
    if nnodes != 1:
        unsupported.append(f"nnodes={nnodes}")
    if dp_size != 1 and not enable_dp_attention:
        unsupported.append(f"dp_size={dp_size} without enable_dp_attention")
    if enable_dp_attention and dp_size > 1 and tp_size // dp_size > 1:
        unsupported.append(
            "attention-DP partial replication "
            f"(tp_size={tp_size}, dp_size={dp_size}, "
            f"attention_tp_size={tp_size // dp_size})"
        )
    if moe_dp_size != 1:
        unsupported.append(f"moe_dp_size={moe_dp_size}")
    if attn_cp_size != 1:
        unsupported.append(f"attn_cp_size={attn_cp_size}")
    if quantization is not None:
        unsupported.append(f"quantization={quantization!r}")
    if unsupported:
        raise WeightHeterogeneousTransferError(
            "weight-cache heterogeneous transfer supports only single-node, "
            "TP/PP/EP, moe_dp_size=1, fully replicated attention DP, "
            "attn_cp_size=1, and unquantized weights; "
            "unsupported: "
            + ", ".join(unsupported)
        )


def _build_weight_runtime_manifest(
    *,
    model: Any,
    tp_size: int,
    tp_rank: int,
    pp_size: int,
    pp_rank: int,
    ep_size: int,
    dp_size: int,
    gpu_id: int,
    transfer_endpoint: str,
    model_id: str,
    revision: str,
    load_plan: Any,
):
    from sglang.srt.distributed.parallel_state import (
        get_moe_expert_parallel_rank,
        get_moe_expert_parallel_world_size,
        get_moe_tensor_parallel_rank,
        get_moe_tensor_parallel_world_size,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    # Mooncake's TP/EP coordinates describe the MoE factorization. Dense
    # global-TP shards are represented by hierarchical EP+TP split axes in the
    # placement descriptor; attention DP remains folded into tensor geometry.
    parallel_topology = WeightParallelTopology(
        dp_rank=0,
        dp_size=1,
        tp_rank=get_moe_tensor_parallel_rank(),
        tp_size=get_moe_tensor_parallel_world_size(),
        pp_rank=get_pipeline_model_parallel_rank(),
        pp_size=get_pipeline_model_parallel_world_size(),
        ep_rank=get_moe_expert_parallel_rank(),
        ep_size=get_moe_expert_parallel_world_size(),
    )
    expected_topology = (
        tp_size,
        tp_rank,
        pp_size,
        pp_rank,
        ep_size,
        tp_size // ep_size,
        dp_size,
        1,
    )
    actual_topology = (
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
        parallel_topology.pp_size,
        parallel_topology.pp_rank,
        parallel_topology.ep_size,
        parallel_topology.tp_size,
        parallel.attn_dp_size,
        parallel.moe_dp_size,
    )
    if actual_topology != expected_topology:
        raise WeightHeterogeneousTransferError(
            "distributed parallel topology differs from daemon arguments: "
            f"expected={expected_topology}, actual={actual_topology}"
        )
    manifest_builder = create_weight_runtime_manifest_builder(
        model=model,
        load_plan=load_plan,
        topology=parallel_topology,
    )
    worker_id = (
        f"weight-cache:{os.getpid()}:gpu{gpu_id}:pp{pp_rank}:tp{tp_rank}:"
        f"adp{parallel.attn_dp_rank}:ep{parallel_topology.ep_rank}:"
        f"mtp{parallel_topology.tp_rank}"
    )
    runtime_manifest = manifest_builder.build(
        model_id=model_id,
        revision=revision,
        instance_id=worker_id,
        worker_id=worker_id,
        endpoint=transfer_endpoint,
    )
    return runtime_manifest


def _initialize_weight_transfer_engine():
    try:
        from mooncake.engine import TransferEngine
    except ImportError as error:
        raise WeightHeterogeneousTransferError(
            "Mooncake Transfer Engine is required for heterogeneous weight "
            "transfer. Run the "
            "daemon from the manifest conda environment."
        ) from error

    transfer_engine = TransferEngine()
    transfer_host = get_local_ip_auto()
    transfer_protocol = envs.MOONCAKE_PROTOCOL.get()
    transfer_device = envs.MOONCAKE_DEVICE.get()
    result = transfer_engine.initialize(
        transfer_host,
        "P2PHANDSHAKE",
        transfer_protocol,
        transfer_device,
    )
    if result != 0:
        raise WeightHeterogeneousTransferError(
            "Mooncake TransferEngine.initialize failed: "
            f"protocol={transfer_protocol!r}, device={transfer_device!r}, "
            f"result={result}"
        )
    transfer_endpoint = NetworkAddress(
        transfer_host, transfer_engine.get_rpc_port()
    ).to_host_port_str()
    return transfer_engine, transfer_endpoint


@dataclass
class WeightTransferSession:
    """One daemon's local end of a heterogeneous transfer.

    Owns the Transfer Engine handle and every Mooncake memory registration made
    on its behalf, so it must outlive ``load()`` and be closed from
    ``shutdown()``. ``runtime_manifest`` doubles as the readiness flag: a source
    fills it during ``create``, a target only after ``prepare_target``.
    """

    transfer_engine: Any
    runtime_manifest: Any | None
    parallel_layout: WeightParallelLayout
    registered_memory_ranges: tuple[tuple[int, int], ...]
    manifest_context: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        model: Any,
        model_config: Any,
        tp_size: int,
        tp_rank: int,
        pp_size: int,
        pp_rank: int,
        ep_size: int,
        dp_size: int,
        gpu_id: int,
        revision: str,
        is_source_daemon: bool,
        load_plan: Any | None,
    ) -> WeightTransferSession:
        transfer_engine, transfer_endpoint = _initialize_weight_transfer_engine()
        registered_memory_ranges: list[tuple[int, int]] = []
        try:
            manifest_context = dict(
                tp_size=tp_size,
                tp_rank=tp_rank,
                pp_size=pp_size,
                pp_rank=pp_rank,
                ep_size=ep_size,
                dp_size=dp_size,
                gpu_id=gpu_id,
                transfer_endpoint=transfer_endpoint,
                model_id=model_identity_from_config(model_config.hf_config),
                revision=revision,
            )
            runtime_manifest = None
            if is_source_daemon:
                if load_plan is None:
                    raise WeightHeterogeneousTransferError(
                        "source daemon has no recorded native weight-load plan"
                    )
                runtime_manifest = _build_weight_runtime_manifest(
                    model=model,
                    load_plan=load_plan,
                    **manifest_context,
                )
                registered_memory_ranges.extend(
                    _register_weight_fragment_allocations(
                        transfer_engine,
                        runtime_manifest.tensors,
                    )
                )
            return cls(
                transfer_engine=transfer_engine,
                runtime_manifest=runtime_manifest,
                parallel_layout=WeightParallelLayout(
                    tp_size=tp_size,
                    dp_size=dp_size,
                    pp_size=pp_size,
                    ep_size=ep_size,
                ),
                registered_memory_ranges=tuple(registered_memory_ranges),
                manifest_context=manifest_context,
            )
        except BaseException:
            for address, _ in reversed(registered_memory_ranges):
                try:
                    transfer_engine.unregister_memory(address)
                except Exception:
                    logger.exception(
                        "Failed to unregister Mooncake memory after setup error"
                    )
            raise

    def prepare_target(
        self,
        *,
        model: Any,
        source_runtime_manifests: Sequence[Any],
    ) -> None:
        if self.runtime_manifest is not None:
            raise WeightHeterogeneousTransferError(
                "target runtime manifest was prepared more than once"
            )
        from .weight_load_recorder import (
            WeightLoadRecordingError,
            logical_weight_metadata_from_runtime_manifests,
            record_target_weight_load_plan,
        )

        try:
            logical_weights = logical_weight_metadata_from_runtime_manifests(
                source_runtime_manifests
            )
            load_plan = record_target_weight_load_plan(model, logical_weights)
            runtime_manifest = _build_weight_runtime_manifest(
                model=model,
                load_plan=load_plan,
                **self.manifest_context,
            )
            registered = _register_weight_fragment_allocations(
                self.transfer_engine,
                runtime_manifest.tensors,
            )
        except WeightLoadRecordingError as error:
            raise WeightHeterogeneousTransferError(
                f"target native weight loader is outside recorder support: {error}"
            ) from error
        self.runtime_manifest = runtime_manifest
        self.registered_memory_ranges = registered

    def runtime_manifest_for_wire(self) -> Any:
        if self.runtime_manifest is None:
            raise WeightHeterogeneousTransferError(
                "target runtime manifest has not been prepared"
            )
        return msgspec.to_builtins(self.runtime_manifest)

    def close(self) -> None:
        failures = []
        for address, _ in reversed(self.registered_memory_ranges):
            try:
                result = self.transfer_engine.unregister_memory(address)
            except Exception as error:
                failures.append((address, repr(error)))
                continue
            if result != 0:
                failures.append((address, result))
        if failures:
            logger.error("weight manifest cleanup failures: %s", failures)


@dataclass(frozen=True)
class SourceDaemonGroup:
    """The whole source daemon group as advertised through the registry.

    Holds one ``WeightRuntimeManifest`` per source rank in ``runtime_manifests``,
    plus its parallel layout and physical device identities needed before planning.
    """

    parallel_layout: WeightParallelLayout
    device_uuids: tuple[str, ...]
    runtime_manifests: tuple[Any, ...]


def parse_source_daemon_group(
    wire_manifest: dict[str, Any],
) -> SourceDaemonGroup:
    """Validate one registry payload and convert it into a typed group.

    The registry assembles plain dicts from independently registering source
    ranks, so this is the only place the wire contract is checked before a
    target plans or writes any weight bytes.
    """
    try:
        parallel_layout_data = wire_manifest["parallel_layout"]
        parallel_layout = WeightParallelLayout(
            tp_size=int(parallel_layout_data["tp_size"]),
            dp_size=int(parallel_layout_data.get("dp_size", 1)),
            pp_size=int(parallel_layout_data["pp_size"]),
            ep_size=int(parallel_layout_data["ep_size"]),
        )
        device_uuids = tuple(str(value) for value in wire_manifest["device_uuids"])
        runtime_manifests = tuple(wire_manifest["runtime_manifests"])
    except (KeyError, TypeError, ValueError) as error:
        raise WeightHeterogeneousTransferError(
            "registry returned an invalid source weight manifest"
        ) from error

    rank_counts = (
        len(device_uuids),
        len(runtime_manifests),
    )
    if rank_counts != (parallel_layout.world_size,) * 2:
        raise WeightHeterogeneousTransferError(
            "source manifest rank count mismatch: "
            f"expected={parallel_layout.world_size}, actual={rank_counts}"
        )
    if len(device_uuids) != len(set(device_uuids)):
        raise WeightHeterogeneousTransferError(
            "source manifest device UUIDs are invalid"
        )
    return SourceDaemonGroup(
        parallel_layout=parallel_layout,
        device_uuids=device_uuids,
        runtime_manifests=runtime_manifests,
    )


def _send_request(
    registry_url: str,
    request: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    prefix = "tcp://"
    if not registry_url.startswith(prefix):
        raise WeightHeterogeneousTransferError(
            f"manifest registry address must use tcp://, got {registry_url!r}"
        )
    try:
        address = NetworkAddress.parse(registry_url[len(prefix) :])
    except ValueError as error:
        raise WeightHeterogeneousTransferError(
            f"invalid manifest registry address: {registry_url!r}"
        ) from error

    with socket.create_connection(address.to_bind_tuple(), timeout=timeout) as conn:
        conn.settimeout(timeout)
        send_msg(conn, request)
        response = recv_msg(conn)
    if not isinstance(response, dict):
        raise WeightHeterogeneousTransferError(
            "manifest registry returned a non-dict response"
        )
    return response


def register_weight_manifest(
    registry_url: str,
    *,
    global_rank: int,
    device_uuid: str,
    session: WeightTransferSession,
    timeout: float = 30.0,
) -> None:
    """Register one immutable source rank with the manifest server."""
    layout = session.parallel_layout
    payload = {
        "node_id": get_local_ip_auto(fallback=socket.gethostname()),
        "global_rank": global_rank,
        "device_uuid": device_uuid,
        "parallel_layout": {
            "tp_size": layout.tp_size,
            "dp_size": layout.dp_size,
            "pp_size": layout.pp_size,
            "ep_size": layout.ep_size,
        },
        "runtime_manifest": session.runtime_manifest_for_wire(),
    }
    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            response = _send_request(
                registry_url,
                {"type": "register_weight_manifest", "manifest": payload},
                timeout=min(5.0, max(deadline - time.monotonic(), 0.1)),
            )
            if response.get("status") == "ok":
                return
            if response.get("status") == "conflict":
                raise WeightHeterogeneousTransferError(
                    "manifest server rejected source rank: "
                    f"{response.get('message', response)}"
                )
            last_error = RuntimeError(
                f"manifest server returned an error: {response}"
            )
        except WeightHeterogeneousTransferError:
            raise
        except Exception as error:
            last_error = error
        time.sleep(0.2)
    raise WeightHeterogeneousTransferError(
        f"failed to register source manifest at {registry_url}: {last_error}"
    )


def fetch_weight_manifest(
    registry_url: str,
    *,
    timeout: float = 30.0,
) -> SourceDaemonGroup:
    """Wait for and fetch one complete source group from the registry."""
    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            response = _send_request(
                registry_url,
                {"type": "get_weight_manifest"},
                timeout=min(5.0, max(deadline - time.monotonic(), 0.1)),
            )
            if response.get("status") == "ok":
                return parse_source_daemon_group(
                    response["weight_manifest"]
                )
            last_error = RuntimeError(
                f"manifest server returned an error: {response}"
            )
        except Exception as error:
            last_error = error
        time.sleep(0.2)
    raise WeightHeterogeneousTransferError(
        f"failed to fetch source manifest from {registry_url}: {last_error}"
    )


def pull_weights_from_source(
    *,
    target_session: WeightTransferSession,
    source_runtime_manifests: Sequence[Any],
    source_parallel_layout: WeightParallelLayout,
    target_runtime_manifests: Sequence[Any],
) -> dict[str, int]:
    try:
        from mooncake.reshard.weight import (
            MemoryRegistrationLease,
            MooncakeTransferEngineReader,
            bind_logical_transfer_plan,
            plan_placement_transfer_to_local_target,
        )
    except ImportError as error:
        raise WeightHeterogeneousTransferError(
            "mooncake.reshard.weight is unavailable in this environment"
        ) from error

    source_placement, source_bindings = build_mooncake_placement_and_bindings(
        source_runtime_manifests,
        placement_set_id="weight-cache-source",
        tp_size=source_parallel_layout.tp_size,
        pp_size=source_parallel_layout.pp_size,
        ep_size=source_parallel_layout.ep_size,
    )
    target_placement, target_bindings = build_mooncake_placement_and_bindings(
        target_runtime_manifests,
        placement_set_id="weight-cache-target",
        tp_size=target_session.parallel_layout.tp_size,
        pp_size=target_session.parallel_layout.pp_size,
        ep_size=target_session.parallel_layout.ep_size,
    )
    target_instance_id = target_session.runtime_manifest.instance_id
    try:
        target_binding = next(
            binding
            for binding in target_bindings
            if binding.instance_id == target_instance_id
        )
    except StopIteration as error:
        raise WeightHeterogeneousTransferError(
            f"local target binding is missing: {target_instance_id}"
        ) from error

    source_fragment_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in source_bindings
        for fragment in binding.fragments
    )
    target_fragment_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=target_binding.generation,
            runtime_lease_id=target_binding.lease_id,
        )
        for fragment in target_binding.fragments
    )
    logical_plan = plan_placement_transfer_to_local_target(
        source_placement,
        target_placement,
        target_participant_id=target_binding.participant_id,
    )
    transfer_plan = bind_logical_transfer_plan(
        logical_plan,
        (target_binding,),
        source_bindings=source_bindings,
    )
    if transfer_plan.total_bytes <= 0 or not transfer_plan.operations:
        raise WeightHeterogeneousTransferError(
            "Mooncake produced an empty heterogeneous weight transfer plan"
        )

    transfer_reader = MooncakeTransferEngineReader(
        target_session.transfer_engine,
        max_batch_operations=8192,
    )
    transfer_receipts = transfer_reader.execute(
        transfer_plan,
        source_placement,
        source_bindings,
        target_placement,
        target_binding,
        source_pre_registered=True,
        source_registrations=source_fragment_registrations,
        target_pre_registered=True,
        target_registrations=target_fragment_registrations,
        source_allocation_guards=immutable_weight_allocation_guards(source_bindings),
        target_allocation_guards=immutable_weight_allocation_guards((target_binding,)),
    )
    wire_bytes = sum(receipt.nbytes for receipt in transfer_receipts)
    wire_operations = sum(receipt.operation_count for receipt in transfer_receipts)
    if wire_bytes != transfer_plan.total_bytes:
        raise WeightHeterogeneousTransferError(
            "incomplete heterogeneous weight transfer: "
            f"planned={transfer_plan.total_bytes}, "
            f"completed={wire_bytes}"
        )
    logger.info(
        "heterogeneous weight transfer complete: logical_bytes=%d operations=%d "
        "endpoints=%d",
        transfer_plan.total_bytes,
        wire_operations,
        len(transfer_receipts),
    )
    return {
        "logical_bytes": int(transfer_plan.total_bytes),
        "wire_bytes": int(wire_bytes),
        "wire_operations": int(wire_operations),
        "source_ranks": len(source_bindings),
    }


def _all_gather_rank_local_phase(
    phase: str,
    operation: Callable[[], Any],
) -> list[Any]:
    """Run a rank-local phase and make every rank observe any local failure."""
    import torch.distributed as dist

    local_error: Optional[Exception] = None
    try:
        local_result = operation()
        local_status = {"error": None, "result": local_result}
    except Exception as error:
        local_error = error
        local_status = {
            "error": f"{type(error).__name__}: {error}",
            "result": None,
        }

    rank_statuses: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_statuses, local_status)
    failures = [
        f"rank {rank}: {status['error']}"
        for rank, status in enumerate(rank_statuses)
        if status["error"] is not None
    ]
    if failures:
        aggregated_error = WeightHeterogeneousTransferError(
            f"{phase} failed on target rank(s): " + "; ".join(failures)
        )
        if local_error is not None:
            raise aggregated_error from local_error
        raise aggregated_error
    return [status["result"] for status in rank_statuses]


def transfer_weights_from_source_daemons(
    *,
    target_session: WeightTransferSession,
    model: Any,
    device_uuid: str,
    registry_url: str,
) -> dict[str, int]:
    """Complete a target copy using only the source and target daemons."""
    import torch.distributed as dist

    # Fetch the source once, then broadcast it to every target daemon rank.
    source_group_result: list[Optional[dict[str, Any]]] = [None]
    if dist.get_rank() == 0:
        try:
            source_group_result[0] = {
                "manifest": fetch_weight_manifest(registry_url),
                "error": None,
            }
        except Exception as error:
            source_group_result[0] = {
                "manifest": None,
                "error": str(error),
            }
    dist.broadcast_object_list(source_group_result, src=0)
    result = source_group_result[0]
    if result is None:
        raise WeightHeterogeneousTransferError(
            "target rank did not receive a manifest registry result"
        )
    if result["error"] is not None:
        raise WeightHeterogeneousTransferError(result["error"])
    source_daemon_group = result["manifest"]

    # Socket templates can define independent discovery namespaces, so UUID
    # socket paths alone do not prevent two daemons from sharing one physical
    # device. Keep the transfer-level no-overlap contract in physical identity.
    target_device_uuids: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(target_device_uuids, device_uuid)
    target_device_uuids = [str(value) for value in target_device_uuids]
    if len(target_device_uuids) != len(set(target_device_uuids)):
        raise WeightHeterogeneousTransferError(
            "target device UUIDs are not unique: "
            f"{tuple(target_device_uuids)}"
        )
    overlap = sorted(
        set(source_daemon_group.device_uuids) & set(target_device_uuids)
    )
    if overlap:
        raise WeightHeterogeneousTransferError(
            "source and target daemons must not share physical devices; "
            f"overlapping device UUIDs: {overlap}"
        )

    def prepare_target() -> Any:
        target_session.prepare_target(
            model=model,
            source_runtime_manifests=source_daemon_group.runtime_manifests,
        )
        return target_session.runtime_manifest_for_wire()

    target_runtime_manifests = _all_gather_rank_local_phase(
        "prepare_target",
        prepare_target,
    )

    transfer_stats_by_rank = _all_gather_rank_local_phase(
        "pull",
        lambda: pull_weights_from_source(
            target_session=target_session,
            source_runtime_manifests=source_daemon_group.runtime_manifests,
            source_parallel_layout=source_daemon_group.parallel_layout,
            target_runtime_manifests=target_runtime_manifests,
        ),
    )
    transfer_stats = transfer_stats_by_rank[dist.get_rank()]
    _all_gather_rank_local_phase(
        "rebuild",
        lambda: rebuild_transferred_weight_state(model),
    )
    return transfer_stats
