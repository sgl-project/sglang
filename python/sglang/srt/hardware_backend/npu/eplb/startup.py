"""Prepare and agree on Ascend async EPLB resources before KV sizing."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.hardware_backend.npu.eplb.coordinator import AsyncEplbCoordinator
from sglang.srt.hardware_backend.npu.eplb.placement import PlanningSpec
from sglang.srt.hardware_backend.npu.eplb.runtime import EplbControl, NpuEplbRuntime
from sglang.srt.hardware_backend.npu.eplb.weights import (
    _allocate_migration_warmup_buffers,
    _validate_migration_topology,
)

logger = logging.getLogger(__name__)


def _make_planning_spec(
    model_config, metadata, moe, nodes, *, decode_worker, records_source_counts
) -> PlanningSpec:
    """Choose the recorder counts and placement policy for this dispatcher."""
    from sglang.srt.eplb.eplb_algorithms import compute_algorithm

    common = ExpertLocationMetadata._init_common(model_config)
    groups = common["model_config_for_expert_location"].num_groups
    algorithm = compute_algorithm(moe.eplb_algorithm, groups, nodes)
    if algorithm.name.startswith("elasticity"):
        raise ValueError("Async EPLB requires a non-elastic CPU placement algorithm")
    dispatch_mode = "unique"
    if metadata.num_physical_experts > metadata.num_logical_experts:
        dispatch_mode = "static" if moe.ep_dispatch_algorithm == "static" else "random"
    # AllGather moves the same rows regardless of expert placement. A unified
    # recorder window can include both AllGather prefill and routed decode.
    score_locality = True
    if nodes > 1 and moe.moe_a2a_backend == "ascend_ops" and not decode_worker:
        from sglang.srt.layers.moe import get_moe_ascend_ops_config

        score_locality = get_moe_ascend_ops_config().prefill != "allgather"
    return PlanningSpec(
        physical_experts=metadata.num_physical_experts,
        local_experts=metadata.num_local_physical_experts,
        groups=groups,
        nodes=nodes,
        algorithm=algorithm,
        dispatch_mode=dispatch_mode,
        score_locality=score_locality,
        collect_active_sources=dispatch_mode == "random" and records_source_counts,
    )


def _hccl_blocking_wait() -> bool:
    # Match c10d's boolean parsing and TorchNPU's preference for the newer name.
    blocking = False
    for name in ("HCCL_BLOCKING_WAIT", "TORCH_HCCL_BLOCKING_WAIT"):
        value = os.environ.get(name)
        if value is None:
            continue
        value = value.lower()
        if value in ("y", "yes", "1", "t", "true"):
            blocking = True
        elif value in ("n", "no", "0", "f", "false"):
            blocking = False
        else:
            raise ValueError(f"Invalid value for environment variable: {name}")
    return blocking


def _check_startup_agreement(
    peers: list[tuple[dict[str, object], list[int]] | str],
    local_error: Exception | None,
) -> set[int]:
    """Check the gathered setup before any rank creates the transport groups."""
    peer_settings = []
    narrowed_layers = set()
    for rank, result in enumerate(peers):
        if isinstance(result, str):
            raise RuntimeError(
                f"Async EPLB setup failed on rank {rank}: {result}"
            ) from local_error
        settings, layers = result
        peer_settings.append(settings)
        narrowed_layers.update(layers)

    reference = peer_settings[0]
    for rank, settings in enumerate(peer_settings[1:], start=1):
        if settings != reference:
            different = sorted(
                key
                for key in settings.keys() | reference.keys()
                if key not in settings
                or key not in reference
                or settings[key] != reference[key]
            )
            raise ValueError(
                f"Async EPLB rank {rank} differs from rank 0 in {', '.join(different)}"
            )
    return narrowed_layers


def create_npu_async_eplb(
    model_config, model, *, startup_weight_load
) -> AsyncEplbCoordinator:
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_parallel,
        get_schedule,
        reports_expert_balancedness,
    )

    world_size = dist.get_world_size()
    cpu_timeout = timedelta(seconds=90)
    control_group = dist.new_group(backend="gloo", timeout=cpu_timeout)
    error = None
    try:
        control = EplbControl(control_group)
        import torch_npu

        _validate_migration_topology(world_size)
        if startup_weight_load is not None:
            raise ValueError(
                "Async EPLB requires completed weight loading before migration setup"
            )
        parallel = get_parallel()
        # Let TorchNPU derive its watchdog timeout from CANN by default.
        hccl_timeout = (
            timedelta(seconds=parallel.dist_timeout)
            if parallel.dist_timeout is not None
            else None
        )
        moe = get_exec().moe
        serving_mode = get_disagg().disaggregation_mode
        decode_worker = serving_mode == "decode"
        # Resolving the model's lazy weight table can fail independently on each rank.
        weights = model.routed_experts_weights_of_layer
        metadata = get_global_expert_location_metadata()
        recorder = get_global_expert_distribution_recorder()
        spec = _make_planning_spec(
            model_config,
            metadata,
            moe,
            parallel.nnodes,
            decode_worker=decode_worker,
            records_source_counts=recorder.records_source_counts,
        )
        p2p_buffer_mib = int(os.environ.get("HCCL_BUFFSIZE") or 64)
        if p2p_buffer_mib < 1:
            raise ValueError("HCCL_BUFFSIZE must be positive")
        p2p_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        p2p_options.hccl_config = {"hccl_buffer_size": p2p_buffer_mib}
        blocking_wait = _hccl_blocking_wait()
        runtime = NpuEplbRuntime(
            weights,
            metadata,
            spec,
            parallel.tp_rank,
            world_size,
            world_size // parallel.nnodes,
            order_submissions=decode_worker,
            pack_small_fields=decode_worker and parallel.nnodes == 1,
            use_prefill_admission=(
                serving_mode == "null"
                and not get_schedule().disable_overlap_schedule
                and not envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.get()
                # With multiple DP ranks the admission vote needs the
                # synchronized extend flag, including on idle ranks.
                and not (
                    parallel.num_dp_ranks > 1
                    and envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
                )
            ),
        )
        initial_layout = metadata.physical_to_logical_map_cpu.contiguous()
        if any(t.shape[0] != spec.local_experts for t in runtime.staging):
            raise ValueError(
                "Async EPLB weight rows do not match the local expert count"
            )
        # Vote on local allocations before any rank posts a device collective.
        tp_probe = torch.zeros(1, dtype=torch.int32, device=runtime.device)
        warmup_buffers = (
            _allocate_migration_warmup_buffers(world_size) if world_size > 1 else None
        )
        signature = {
            "layers": runtime.layer_ids,
            "expert_weight_format": runtime.weight_format_digest,
            "serving_mode": serving_mode,
            "prefill_admission": runtime.use_prefill_admission,
            "p2p_batch_size": runtime.p2p_batch_size,
            "HCCL_BUFFSIZE": p2p_buffer_mib,
            "hccl_timeout": hccl_timeout,
            "rebalance_interval": moe.eplb_rebalance_num_iterations,
            "recorder_window": moe.expert_distribution_recorder_buffer_size,
            "utilization_threshold": moe.eplb_min_rebalancing_utilization_threshold
            if reports_expert_balancedness()
            else None,
            "num_logical_experts": metadata.num_logical_experts,
            "dispatch_algorithm": moe.ep_dispatch_algorithm,
            "initial_layout": hashlib.sha256(initial_layout.numpy().tobytes()).digest(),
            **{f"placement.{key}": value for key, value in asdict(spec).items()},
        }
        setup = (signature, sorted(runtime.narrowed_replica_layers))
    except Exception as exc:
        error = exc
        setup = f"{type(exc).__name__}: {exc}"
    peers = [None] * world_size
    dist.all_gather_object(peers, setup, group=control_group)
    try:
        narrowed_layers = _check_startup_agreement(peers, error)
    except (RuntimeError, ValueError):
        dist.destroy_process_group(control_group)
        raise
    runtime.narrowed_replica_layers.update(narrowed_layers)
    runtime.open_transport(
        cpu_timeout, hccl_timeout, p2p_options, tp_probe, warmup_buffers
    )
    del tp_probe, warmup_buffers
    torch.npu.empty_cache()
    if decode_worker and blocking_wait:
        logger.warning(
            "Async NPU EPLB decode waits for each layer's HCCL transfers with "
            "TORCH_HCCL_BLOCKING_WAIT enabled (including its HCCL_BLOCKING_WAIT alias)"
        )
    weight_bytes, routing_bytes, packed_fields = runtime.staging_memory()
    logger.info(
        "Async NPU EPLB ready: weight_staging_bytes=%d routing_staging_bytes=%d "
        "hccl_buffer_mib=%d hccl_blocking_wait=%s layers_per_chunk=1 packed_fields=%d",
        weight_bytes,
        routing_bytes,
        p2p_buffer_mib,
        blocking_wait,
        packed_fields,
    )
    return AsyncEplbCoordinator(control, runtime)
