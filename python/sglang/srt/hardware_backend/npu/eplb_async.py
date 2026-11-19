"""NPU implementation of the forward-boundary EPLB protocol."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.eplb.expert_location_updater import (
    DeferredExpertCopies,
    update_expert_weights_single_layer,
)
from sglang.srt.hardware_backend.npu.eplb import (
    _ExpertWeightAdapter,
    _validate_migration_topology,
    prewarm_eplb_communication,
)
from sglang.srt.hardware_backend.npu.eplb_coordinator import (
    AsyncEplbCoordinator,
    WeightWriteback,
)
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_DEVICE_ROUTING_FIELDS = (
    "physical_to_logical_map",
    "logical_to_all_physical_map",
    "logical_to_all_physical_map_num_valid",
    "logical_to_rank_dispatch_physical_map",
)


class EplbControl:
    def __init__(self, group):
        self.group = group
        self.wait_seconds = 0.0

    def vote(self, epoch, phase, layer, ready, error):
        # MAX(x), MAX(-x) detects mismatched protocol positions on every rank.
        flags = torch.tensor(
            [epoch, -epoch, phase, -phase, layer, -layer, int(not ready), int(error)],
            dtype=torch.int64,
        )
        start = time.perf_counter()
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self.group)
        self.wait_seconds += time.perf_counter() - start
        values = flags.tolist()
        if any(values[i] != -values[i + 1] for i in (0, 2, 4)):
            raise RuntimeError("EPLB ranks reached different protocol boundaries")
        return values[6] == 0, bool(values[7])

    def drained(self, success):
        value = torch.tensor([int(success)], dtype=torch.int32)
        dist.all_reduce(value, op=dist.ReduceOp.MIN, group=self.group)
        return bool(value.item())


@dataclass(frozen=True)
class PlanningSpec:
    physical_experts: int
    local_experts: int
    groups: int | None
    nodes: int
    algorithm: object

    def compute(self, counts):
        from sglang.srt.eplb.eplb_algorithms import rebalance_experts

        physical, logical, _ = rebalance_experts(
            tokens_per_expert=counts,
            num_physical_experts=self.physical_experts,
            num_local_physical_experts=self.local_experts,
            num_groups=self.groups,
            num_nodes=self.nodes,
            algorithm=self.algorithm,
        )
        return physical, logical


def _make_weight_writeback(weights, buffers, rows) -> WeightWriteback:
    # The NPU adapter exposes dense byte rows. Merge adjacent copies only
    # when they are contiguous in both the staging and live storage.
    ranges = []
    for source, destination in rows:
        if ranges:
            prev_source, prev_dest, length = ranges[-1]
            if source == prev_source + length and destination == prev_dest + length:
                ranges[-1] = (prev_source, prev_dest, length + 1)
                continue
        ranges.append((source, destination, 1))

    def writeback():
        for source, destination, length in ranges:
            for weight, buffer in zip(weights, buffers, strict=True):
                weight[destination : destination + length].copy_(
                    buffer[source : source + length]
                )

    return writeback


def _install_routing_layer(current, planned, layer: int, staged_rows) -> None:
    """Publish the staged device rows and their CPU mirrors at the same boundary."""
    for name, row in staged_rows.items():
        getattr(current, name)[layer].copy_(row)
    current.physical_to_logical_map_cpu[layer].copy_(
        planned.physical_to_logical_map_cpu[layer]
    )
    current.logical_to_all_physical_map_cpu[layer].copy_(
        planned.logical_to_all_physical_map_cpu[layer]
    )

    for algorithm, (physical, count) in current._mega_gate_maps.items():
        if algorithm == "static":
            physical[layer, ..., 0].copy_(
                current.logical_to_rank_dispatch_physical_map[layer]
            )
        else:
            physical[layer].copy_(current.logical_to_all_physical_map[layer])
            count[layer].copy_(current.logical_to_all_physical_map_num_valid[layer])


class NpuEplbRuntime:
    def __init__(
        self,
        weights,
        metadata,
        spec,
        rank,
        world_size,
        ranks_per_node,
        plan_group,
        p2p_group,
    ):
        self.metadata = metadata
        self.spec = spec
        self.rank = rank
        self.world_size = world_size
        self.ranks_per_node = ranks_per_node
        self.plan_group = plan_group
        self.p2p_group = p2p_group
        self.log_metrics = get_bool_env_var(
            "SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS"
        )
        self.layer_ids = tuple(sorted(weights))
        adapter = _ExpertWeightAdapter(weights, list(self.layer_ids), rank)
        self.weights = {
            layer: adapter.make_weight_views(weights[layer]) for layer in self.layer_ids
        }
        self.staging = adapter.create_temp_buffers()
        # Reserve one layer before KV sizing. The same owner event fences reuse
        # of both the routing rows and the weight staging buffers.
        self._routing_rows = {
            name: torch.empty_like(table[0])
            for name in _DEVICE_ROUTING_FIELDS
            if (table := getattr(metadata, name)) is not None
        }
        self.device = self.staging[0].device
        self.device_module = torch.get_device_module(self.device)
        self._p2p_stream = self.device_module.Stream(device=self.device)
        self._owner_stream = self.device_module.current_stream()
        self._owner_ready = self.device_module.Event()
        self._new_metadata = None
        self.stage_seconds = self.install_enqueue_seconds = 0.0

    def planner(self, counts):
        self.stage_seconds = self.install_enqueue_seconds = 0.0
        self._owner_stream = self.device_module.current_stream()
        # dump_record returns a fresh, globally reduced tensor. Keep it in the
        # worker closure and copy it on the migration stream only on rank 0.
        payload = [None]
        if self.rank == 0:
            try:
                self._owner_ready.record(self._owner_stream)
            except Exception as error:
                payload[0] = (str(error), None)

        def compute():
            if self.rank == 0 and payload[0] is None:
                try:
                    self.device_module.set_device(self.device)
                    with self.device_module.stream(self._p2p_stream):
                        self._p2p_stream.wait_event(self._owner_ready)
                        host = counts.cpu()
                    # Idle forwards can fill a recorder window without routing tokens.
                    placement = self.spec.compute(host) if host.any().item() else None
                    payload[0] = (None, placement)
                except Exception as error:
                    payload[0] = (str(error), None)
            dist.broadcast_object_list(payload, src=0, group=self.plan_group)
            error, result = payload[0]
            if error is not None:
                raise RuntimeError(f"EPLB CPU planner failed: {error}")
            return result

        return compute

    def prepare(self, placement):
        if placement is None:
            return ()
        physical, logical = placement
        old = self.metadata.physical_to_logical_map_cpu
        if physical.shape != old.shape:
            raise ValueError("EPLB plan changed physical capacity")
        moved = (physical != old).any(dim=1).tolist()
        changed = tuple(layer for layer in self.layer_ids if moved[layer])
        if not changed:
            return ()
        self._last_layer = changed[-1]
        # Static dispatch draws tie breaks in full layer order. Build the
        # complete plan on CPU, even when only a few layers need staging.
        self._new_metadata = ExpertLocationMetadata._init_raw(
            ep_size=self.metadata.ep_size,
            physical_to_logical_map=physical,
            logical_to_all_physical_map=logical,
            moe_ep_rank=self.rank,
        )
        return changed

    def _check_stream(self):
        if self.device_module.current_stream() != self._owner_stream:
            raise RuntimeError("EPLB target forwards changed their owning stream")

    def record_source_ready(self) -> None:
        self._check_stream()
        # Single-flight staging lets each layer reuse the event after the
        # previous transfer finishes and its install is on the owner stream.
        self._owner_ready.record(self._owner_stream)

    def stage(self, layer: int) -> WeightWriteback:
        """Run on the migration worker. The forward thread owns the writeback."""
        self.device_module.set_device(self.device)
        start = time.perf_counter()
        # Work.wait orders the current stream after HCCL. The Future can
        # advertise readiness only after that dependency has completed.
        copies = DeferredExpertCopies()
        with self.device_module.stream(self._p2p_stream):
            self._p2p_stream.wait_event(self._owner_ready)
            for name, row in self._routing_rows.items():
                row.copy_(getattr(self._new_metadata, name)[layer], non_blocking=True)
            old_layout = self.metadata.physical_to_logical_map_cpu[layer]
            new_layout = self._new_metadata.physical_to_logical_map_cpu[layer]
            update_expert_weights_single_layer(
                routed_experts_weights=self.weights[layer],
                temp_buffers=self.staging,
                old_physical_to_logical_map=old_layout.tolist(),
                new_physical_to_logical_map=new_layout.tolist(),
                num_local_physical_experts=self.spec.local_experts,
                num_gpu_per_node=self.ranks_per_node,
                rank=self.rank,
                world_size=self.world_size,
                log_metrics=self.log_metrics,
                group=self.p2p_group,
                deferred_copies=copies,
            )
            writeback = _make_weight_writeback(
                self.weights[layer], self.staging, copies.rows
            )
            self._p2p_stream.synchronize()
        self.stage_seconds += time.perf_counter() - start
        return writeback

    def install(self, layer: int, writeback: WeightWriteback) -> None:
        self._check_stream()
        start = time.perf_counter()
        writeback()
        _install_routing_layer(
            self.metadata, self._new_metadata, layer, self._routing_rows
        )
        if layer == self._last_layer:
            self._new_metadata = None
        self.install_enqueue_seconds += time.perf_counter() - start

    def synchronize(self):
        # The coordinator joins the worker before draining both streams.
        self._p2p_stream.synchronize()
        self._owner_stream.synchronize()
        self._new_metadata = None

    def close(self):
        dist.destroy_process_group(self.p2p_group)


def create_npu_async_eplb(model_config, weights, metadata):
    from sglang.srt.eplb.eplb_algorithms import compute_algorithm
    from sglang.srt.runtime_context import get_exec, get_parallel

    parallel = get_parallel()
    world_size = dist.get_world_size()
    _validate_migration_topology(world_size)
    common = ExpertLocationMetadata._init_common(model_config)
    groups = common["model_config_for_expert_location"].num_groups
    algorithm = compute_algorithm(
        get_exec().moe.eplb_algorithm, groups, parallel.nnodes
    )
    if algorithm.name.startswith("elasticity"):
        raise ValueError("Async EPLB requires a non-elastic CPU placement algorithm")
    spec = PlanningSpec(
        metadata.num_physical_experts,
        metadata.num_local_physical_experts,
        groups,
        parallel.nnodes,
        algorithm,
    )
    # Fixed PP=1 TP=EP=WORLD: group construction order is identical on all ranks.
    timeout = timedelta(seconds=90)
    control_group = dist.new_group(backend="gloo", timeout=timeout)
    plan_group = dist.new_group(backend="gloo", timeout=timeout)
    p2p_group = dist.new_group(backend="hccl", timeout=timeout)
    error = None
    signature = None
    try:
        runtime = NpuEplbRuntime(
            weights,
            metadata,
            spec,
            parallel.tp_rank,
            world_size,
            world_size // parallel.nnodes,
            plan_group,
            p2p_group,
        )
        batch_size = envs.SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE.get()
        if batch_size <= 0:
            raise ValueError("Async EPLB requires a positive P2P batch chunk size")
        initial_layout = metadata.physical_to_logical_map_cpu.contiguous()
        if any(t.shape[0] != spec.local_experts for t in runtime.staging):
            raise ValueError(
                "Async EPLB weight rows do not match the local expert count"
            )
        signature = (
            runtime.layer_ids,
            tuple(tuple(t.shape) for t in runtime.staging),
            batch_size,
            spec,
            hashlib.sha256(initial_layout.numpy().tobytes()).digest(),
        )
    except Exception as exc:
        error = exc
    signatures = [None] * world_size
    dist.all_gather_object(signatures, signature, group=control_group)
    if None in signatures:
        raise RuntimeError("Async EPLB setup failed on at least one rank") from error
    if any(s != signature for s in signatures):
        raise ValueError(
            "Async EPLB ranks disagree on layout, weight geometry or batching"
        )
    prewarm_eplb_communication(p2p_group)
    logger.info(
        "Async NPU EPLB ready: weight_staging_bytes=%d routing_staging_bytes=%d",
        sum(t.nbytes for t in runtime.staging),
        sum(row.nbytes for row in runtime._routing_rows.values()),
    )
    return AsyncEplbCoordinator(EplbControl(control_group), runtime)
