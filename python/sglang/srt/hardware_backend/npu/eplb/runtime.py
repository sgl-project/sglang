"""NPU implementation of the forward-boundary EPLB protocol."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_eplb_group,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import update_expert_weights_single_layer
from sglang.srt.hardware_backend.npu.eplb.coordinator import (
    AsyncEplbCoordinator,
    WeightWriteback,
)
from sglang.srt.hardware_backend.npu.eplb.placement import (
    MigrationPlacement,
    PlanningSpec,
    plan_migration,
)
from sglang.srt.hardware_backend.npu.eplb.weights import (
    _ExpertWeightAdapter,
    _validate_migration_topology,
    prewarm_eplb_communication,
)
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_DEVICE_ROUTING_FIELDS = (
    "physical_to_logical_map",
    "logical_to_all_physical_map",
    "logical_to_all_physical_map_num_valid",
    "logical_to_rank_dispatch_physical_map",
)
_MAX_PACKED_FIELD_BYTES = 16 * 1024


class EplbControl:
    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.wait_seconds = 0.0

    def vote(
        self, epoch: int, phase: int, layer: int, ready: bool, error: bool
    ) -> tuple[bool, bool]:
        # MAX(x), MAX(-x) detects mismatched protocol positions on every rank.
        flags = torch.tensor(
            [epoch, -epoch, phase, -phase, layer, -layer, int(not ready), int(error)],
            dtype=torch.int64,
            device="cpu",
        )
        start = time.perf_counter()
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self.group)
        self.wait_seconds += time.perf_counter() - start
        values = flags.tolist()
        if any(values[i] != -values[i + 1] for i in (0, 2, 4)):
            raise RuntimeError("EPLB ranks reached different protocol boundaries")
        return values[6] == 0, bool(values[7])

    def close(self) -> None:
        dist.destroy_process_group(self.group)


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

    # Slice on the staging worker. The forward thread only submits the copies.
    copies = [
        (
            weight[destination : destination + length],
            buffer[source : source + length],
        )
        for source, destination, length in ranges
        for weight, buffer in zip(weights, buffers, strict=True)
    ]

    # Batched copy helps fragmented writeback. Reused staging rows can be
    # faster as separate copies, especially when each row is large.
    if (
        len(ranges) >= 8
        and weights[0].device.type == "npu"
        and hasattr(torch, "_foreach_copy_")
        and len({source for source, _ in rows}) == len(rows)
    ):
        destinations, sources = zip(*copies, strict=True)

        def batch_writeback():
            torch._foreach_copy_(destinations, sources, non_blocking=True)

        return batch_writeback

    def writeback():
        for destination, source in copies:
            destination.copy_(source)

    return writeback


def _make_comm_rank_lookup(old_layout, new_layout, local_experts):
    def index_ranks(layout):
        by_expert = defaultdict(list)
        for slot, expert in enumerate(layout):
            rank = slot // local_experts
            ranks = by_expert[expert]
            # Preserve rank order for the shared source assignment.
            if not ranks or ranks[-1] != rank:
                ranks.append(rank)
        return by_expert

    old_ranks = index_ranks(old_layout)
    new_ranks = index_ranks(new_layout)

    def lookup(expert):
        sources = old_ranks[expert]
        source_ranks = set(sources)
        return sources, [rank for rank in new_ranks[expert] if rank not in source_ranks]

    return lookup


def _submit_hccl_batches(p2p_op_infos, expert_order, batch_size, pending_work):
    ops_by_expert = dict(p2p_op_infos)
    if not any(ops_by_expert.values()):
        return
    for start in range(0, len(expert_order), batch_size):
        batch_ops = [
            op
            for expert in expert_order[start : start + batch_size]
            for op in ops_by_expert.get(expert, ())
        ]
        if batch_ops:
            work = dist.batch_isend_irecv(batch_ops)
            pending_work.extend(work)
            for request in work:
                request.wait()


def _hccl_expert_order(old_layout, new_layout, num_logical, world_size, batch_size):
    """Spread busy rank pairs across HCCL batches without changing the transfers."""
    block_size = 4
    if batch_size % block_size or num_logical <= batch_size:
        return None

    num_batches = (num_logical + batch_size - 1) // batch_size
    padded = num_batches * batch_size
    ranks = np.arange(world_size)[:, None]
    owners = np.zeros((world_size, padded), dtype=bool)
    receivers = np.zeros_like(owners)
    owners[ranks, np.asarray(old_layout).reshape(world_size, -1)] = True
    receivers[ranks, np.asarray(new_layout).reshape(world_size, -1)] = True
    receivers &= ~owners

    # The updater splits each expert's destinations evenly among its old
    # owners, giving the remainder to the lowest owner ranks.
    quotient, remainder = np.divmod(
        receivers.sum(axis=0), owners.sum(axis=0).clip(min=1)
    )
    sends = owners * (quotient + (owners.cumsum(axis=0) <= remainder))
    endpoint_rows = np.concatenate((sends, receivers), axis=0)
    contiguous = endpoint_rows.reshape(-1, num_batches, batch_size).sum(axis=2)
    interleaved = endpoint_rows.reshape(
        -1, batch_size // block_size, num_batches, block_size
    ).sum(axis=(1, 3))

    # Require a substantial drop in busiest-endpoint rows to offset extra
    # submissions on lightly loaded ranks.
    if interleaved.max(axis=0).sum() * 3 > contiguous.max(axis=0).sum() * 2:
        return None
    if not contiguous.any():
        return None
    return (
        np.arange(padded)
        .reshape(-1, num_batches, block_size)
        .transpose(1, 0, 2)
        .flatten()
        .tolist()
    )


def _install_routing_layer(current, planned, layer: int, staged_rows) -> None:
    """Publish the staged device rows and their CPU mirrors at the same boundary."""
    width = staged_rows["logical_to_all_physical_map"].size(-1)
    for name, row in staged_rows.items():
        destination = getattr(current, name)[layer]
        if name == "logical_to_all_physical_map":
            destination = destination[..., :width]
        destination.copy_(row)
    current.physical_to_logical_map_cpu[layer].copy_(planned["physical_to_logical_map"])
    current.logical_to_all_physical_map_cpu[layer, :, :width].copy_(
        planned["logical_to_all_physical_map"]
    )

    for algorithm, (physical, count) in current._mega_gate_maps.items():
        if algorithm == "static":
            physical[layer, ..., 0].copy_(
                current.logical_to_rank_dispatch_physical_map[layer]
            )
        else:
            physical[layer, :, :width].copy_(
                current.logical_to_all_physical_map[layer, :, :width]
            )
            count[layer].copy_(current.logical_to_all_physical_map_num_valid[layer])


class _PlacementRows(NamedTuple):
    """One row per changed layer, in installation order on this rank."""

    layers: tuple[int, ...]
    physical_to_logical: torch.Tensor
    logical_to_physical: torch.Tensor
    local_dispatch: torch.Tensor | None


def _broadcast_placement(
    placement: MigrationPlacement | None,
    error: str | None,
    rank: int,
    group: dist.ProcessGroup,
    physical_shape: torch.Size,
    num_logical: int,
) -> _PlacementRows | None:
    """Broadcast changed rows; scatter only the static picks each rank uses."""
    # The first header value is -1 on failure, 0 for an empty plan, or the
    # number of layers to install. The second is the replica table width.
    header = torch.zeros(3, dtype=torch.int64, device="cpu")
    rank_tables = None
    if rank == 0:
        if error is None and placement is not None:
            try:
                changed, physical, logical, dispatch = placement
                width = logical.size(-1)
                if physical.shape != physical_shape or logical.shape != (
                    physical_shape[0],
                    num_logical,
                    width,
                ):
                    raise ValueError("EPLB placement changed the routing geometry")
                ids = torch.tensor(changed, dtype=torch.int64, device="cpu")
                fields = [
                    ids,
                    physical.index_select(0, ids),
                    logical.index_select(0, ids),
                ]
                if dispatch is not None:
                    if dispatch.shape != (
                        physical_shape[0],
                        num_logical,
                        dist.get_world_size(group),
                    ):
                        raise ValueError(
                            "EPLB placement changed the static routing geometry"
                        )
                    rank_tables = list(
                        dispatch.index_select(0, ids)
                        .permute(2, 0, 1)
                        .long()
                        .contiguous()
                        .unbind()
                    )
                    header[2] = 1
                payload = torch.cat([field.reshape(-1) for field in fields]).long()
                header[0] = len(changed)
                header[1] = width
            except Exception as exc:
                error = str(exc)
        if error is not None:
            header[0] = -1
    dist.broadcast(header, src=0, group=group)
    count, width, has_static_dispatch = header.tolist()
    if count < 0:
        message = [error]
        dist.broadcast_object_list(message, src=0, group=group)
        raise RuntimeError(f"EPLB CPU planner failed: {message[0]}")
    if count == 0:
        return None
    slots = physical_shape[1]
    physical_end = count + count * slots
    logical_end = physical_end + count * num_logical * width
    if rank != 0:
        payload = torch.empty(
            logical_end,
            dtype=torch.int64,
            device="cpu",
        )
    dist.broadcast(payload, src=0, group=group)
    local_dispatch = None
    if has_static_dispatch:
        local_dispatch = torch.empty(
            (count, num_logical), dtype=torch.int64, device="cpu"
        )
        dist.scatter(local_dispatch, rank_tables, src=0, group=group)
    return _PlacementRows(
        tuple(payload[:count].tolist()),
        payload[count:physical_end].view(count, slots),
        payload[physical_end:logical_end].view(count, num_logical, width),
        local_dispatch,
    )


@dataclass(frozen=True)
class _PackedExpertFields:
    """Send narrow fields together and expose their original views for writeback."""

    small: tuple[int, ...]
    direct: tuple[int, ...]
    send: torch.Tensor
    receive: list[torch.Tensor]

    @classmethod
    def create(cls, staging: list[torch.Tensor]) -> _PackedExpertFields | None:
        # The weight adapter exposes contiguous [expert, bytes] tensors.
        small = tuple(
            i
            for i, field in enumerate(staging)
            if 0 < field.size(1) <= _MAX_PACKED_FIELD_BYTES
        )
        if len(small) < 2:
            return None
        direct = tuple(i for i in range(len(staging)) if i not in small)
        widths = [staging[i].size(1) for i in small]
        packed = staging[0].new_empty((staging[0].size(0), sum(widths)))
        for i, view in zip(small, packed.split(widths, dim=1), strict=True):
            staging[i] = view
        return cls(
            small,
            direct,
            torch.empty_like(packed),
            [staging[i] for i in direct] + [packed],
        )

    def pack(self, weights: list[torch.Tensor]) -> None:
        torch.cat([weights[i] for i in self.small], dim=1, out=self.send)


@dataclass(frozen=True)
class _RoutingPlan:
    """Pinned host tables and views of the reusable one-layer device buffers."""

    host_rows: dict[int, dict[str, torch.Tensor]]
    device_rows: dict[str, torch.Tensor]
    last_layer: int


class NpuEplbRuntime:
    def __init__(
        self,
        weights: dict[int, list[torch.Tensor]],
        metadata: ExpertLocationMetadata,
        spec: PlanningSpec,
        rank: int,
        world_size: int,
        ranks_per_node: int,
        *,
        order_submissions: bool = False,
        pack_small_fields: bool = False,
        use_prefill_admission: bool = False,
    ):
        # Peer batches must use the size checked across ranks at startup.
        self._p2p_batch_size = envs.SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE.get()
        if self._p2p_batch_size <= 0:
            raise ValueError("Async EPLB requires a positive P2P batch chunk size")
        self.metadata = metadata
        self.spec = spec
        self.rank = rank
        self.world_size = world_size
        self.ranks_per_node = ranks_per_node
        # Opened together after all ranks agree on the startup configuration.
        self.plan_group: dist.ProcessGroup
        self.p2p_group: dist.ProcessGroup
        self.log_metrics = get_bool_env_var(
            "SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS"
        )
        self.layer_ids = tuple(sorted(weights.keys()))
        adapter = _ExpertWeightAdapter(weights, list(self.layer_ids), rank)
        weight_formats = hashlib.sha256()
        for layer in self.layer_ids:
            fields = adapter.transfer_signature(weights[layer])
            weight_formats.update(repr(fields).encode("ascii"))
        self.weight_format_digest = weight_formats.digest()
        self.weights = {
            layer: adapter.make_weight_views(weights[layer]) for layer in self.layer_ids
        }
        self.staging = adapter.create_temp_buffers()
        self._packed_fields = (
            _PackedExpertFields.create(self.staging) if pack_small_fields else None
        )
        # Reserve one layer before KV sizing. The owner event fences its reuse
        # after writes from the preceding layer have been enqueued.
        self._routing_buffers = {
            name: torch.empty_like(table[0], memory_format=torch.contiguous_format)
            for name in _DEVICE_ROUTING_FIELDS
            if (table := getattr(metadata, name)) is not None
        }
        # A loaded placement can have holes before the last populated slot.
        live = metadata.logical_to_all_physical_map_cpu
        occupied = (live != -1).any(dim=(0, 1)).nonzero()
        self._replica_width = int(occupied[-1, 0]) + 1 if occupied.numel() else 0
        # An initial mapping may expose only nearby replicas on this rank.
        # Startup takes the union across ranks; install clears one layer at a time.
        self.narrowed_replica_layers: set[int] = set()
        if spec.physical_count_mode == "random":
            visible_slots = (self._replica_map >= 0).sum(dim=(1, 2))
            narrowed = (visible_slots < spec.physical_experts).nonzero().flatten()
            self.narrowed_replica_layers.update(narrowed.tolist())
        self.device = self.staging[0].device
        self.device_module = torch.get_device_module(self.device)
        self._p2p_stream = self.device_module.Stream(device=self.device)
        self._owner_stream = self.device_module.current_stream()
        self._owner_ready = self.device_module.Event()
        self._order_submissions = order_submissions
        self.use_prefill_admission = use_prefill_admission
        self._routing_plan: _RoutingPlan | None = None
        self.stage_seconds = self.install_enqueue_seconds = 0.0
        self.interleaved_layers = 0

    @property
    def _replica_map(self) -> torch.Tensor:
        return self.metadata.logical_to_all_physical_map_cpu[
            :, :, : self._replica_width
        ]

    def planner(self, counts, sources=None):
        self.stage_seconds = self.install_enqueue_seconds = 0.0
        self.interleaved_layers = 0
        self._owner_stream = self.device_module.current_stream()
        # Only the root needs the reduced counts, until its D2H copy completes.
        event_error = None
        if self.rank == 0:
            try:
                self._owner_ready.record(self._owner_stream)
            except Exception as error:
                event_error = str(error)

        def compute():
            nonlocal counts, sources
            error = event_error
            placement = None
            if self.rank == 0 and error is None:
                try:
                    host_counts = self._copy_counts_to_host(counts)
                    host_sources = (
                        self._copy_counts_to_host(sources)
                        if sources is not None
                        else None
                    )
                    counts = sources = None
                    if host_counts.any().item():
                        placement = plan_migration(
                            host_counts,
                            self.metadata.physical_to_logical_map_cpu,
                            self.spec,
                            self.ranks_per_node,
                            self.layer_ids,
                            source_totals=host_sources,
                            narrowed_layers=self.narrowed_replica_layers,
                            current_replicas=(
                                self._replica_map
                                if self.spec.physical_count_mode is not None
                                else None
                            ),
                        )
                except Exception as exc:
                    error = str(exc)
            return _broadcast_placement(
                placement,
                error,
                self.rank,
                self.plan_group,
                self.metadata.physical_to_logical_map_cpu.shape,
                self.metadata.num_logical_experts,
            )

        return compute

    def _copy_counts_to_host(self, counts):
        self.device_module.set_device(self.device)
        with self.device_module.stream(self._p2p_stream):
            self._p2p_stream.wait_event(self._owner_ready)
            return counts.cpu()

    def prepare(self, placement: _PlacementRows | None) -> tuple[int, ...]:
        if placement is None:
            return ()
        changed, physical, logical, dispatch = placement
        width = logical.size(-1)
        # Include the previous populated columns so shrinking the active width
        # overwrites stale replicas. The graph still points to the full table.
        self._replica_width = max(self._replica_width, width)
        device_rows = dict(
            self._routing_buffers,
            logical_to_all_physical_map=self._routing_buffers[
                "logical_to_all_physical_map"
            ]
            .view(-1)[: logical.size(1) * self._replica_width]
            .view(logical.size(1), self._replica_width),
        )
        host_rows = {
            "physical_to_logical_map": physical,
            "logical_to_all_physical_map": torch.nn.functional.pad(
                logical, (0, self._replica_width - width), value=-1
            )
            if width < self._replica_width
            else logical,
            "logical_to_all_physical_map_num_valid": (logical != -1).sum(dim=-1),
        }
        if "logical_to_rank_dispatch_physical_map" in device_rows:
            # Without replicas the inverse table already gives the static pick.
            host_rows["logical_to_rank_dispatch_physical_map"] = (
                logical[..., 0] if dispatch is None else dispatch
            )
        # Pageable H2D blocks the worker on the current stream. Keep the plan
        # pinned until the asynchronous routing copies have drained.
        self.device_module.set_device(self.device)
        pinned = {
            name: rows.contiguous().pin_memory() for name, rows in host_rows.items()
        }
        self._routing_plan = _RoutingPlan(
            host_rows={
                layer: {name: rows[i] for name, rows in pinned.items()}
                for i, layer in enumerate(changed)
            },
            device_rows=device_rows,
            last_layer=changed[-1],
        )
        return changed

    def _check_stream(self):
        if self.device_module.current_stream() != self._owner_stream:
            raise RuntimeError("EPLB target forwards changed their owning stream")

    def record_source_ready(self) -> Future[None] | None:
        self._check_stream()
        # Reuse the event after the previous transfers finish and their writes
        # are on the owner stream.
        self._owner_ready.record(self._owner_stream)
        return Future() if self._order_submissions else None

    def stage(
        self, layer: int, submitted: Future[None] | None = None
    ) -> WeightWriteback:
        """Run on the migration worker. The forward thread owns the writeback."""
        self.device_module.set_device(self.device)
        start = time.perf_counter()
        routing = self._routing_plan
        assert routing is not None
        pending_work = []
        with self.device_module.stream(self._p2p_stream):
            self._p2p_stream.wait_event(self._owner_ready)
            for name, row in routing.device_rows.items():
                row.copy_(routing.host_rows[layer][name], non_blocking=True)
            writeback = self._stage_weights(layer, routing, pending_work)
            if submitted is not None:
                submitted.set_result(None)
            # The caller can install only after the stream and its Work handles drain.
            self._p2p_stream.synchronize()
        self.stage_seconds += time.perf_counter() - start
        return writeback

    def _stage_weights(self, layer, routing, pending_work) -> WeightWriteback:
        old_layout = self.metadata.physical_to_logical_map_cpu[layer].tolist()
        new_layout = routing.host_rows[layer]["physical_to_logical_map"].tolist()
        if old_layout == new_layout:
            # A replica-order update only publishes the staged routing rows.
            return lambda: None

        staged_rows = []
        get_comm_ranks = _make_comm_rank_lookup(
            old_layout, new_layout, self.spec.local_experts
        )
        p2p_expert_order = None
        if self.world_size == self.ranks_per_node:
            p2p_expert_order = _hccl_expert_order(
                old_layout,
                new_layout,
                self.metadata.num_logical_experts,
                self.world_size,
                self._p2p_batch_size,
            )
        self.interleaved_layers += p2p_expert_order is not None
        expert_order = p2p_expert_order or range(len(old_layout))

        def stage_transfers(p2p_op_infos, rows):
            staged_rows.extend(rows)
            if (
                packed is not None
                and not packed_for_local_copy
                and any(op.op == dist.isend for _, ops in p2p_op_infos for op in ops)
            ):
                packed.pack(weights)
            _submit_hccl_batches(
                p2p_op_infos, expert_order, self._p2p_batch_size, pending_work
            )

        weights, staging = self.weights[layer], self.staging
        sends, receives = weights, staging
        packed = self._packed_fields
        packed_for_local_copy = False
        if packed is not None:
            first = self.rank * self.spec.local_experts
            local = slice(first, first + self.spec.local_experts)
            resident = old_layout[local]
            # The updater copies local moves before calling stage_transfers.
            # Other sources only need packing if an isend was selected.
            resident_experts = set(resident)
            packed_for_local_copy = any(
                old != new and new in resident_experts
                for old, new in zip(resident, new_layout[local], strict=True)
            )
            if packed_for_local_copy:
                packed.pack(weights)
            sends = [weights[i] for i in packed.direct] + [packed.send]
            receives = packed.receive
        update_expert_weights_single_layer(
            routed_experts_weights=sends,
            temp_buffers=receives,
            old_physical_to_logical_map=old_layout,
            new_physical_to_logical_map=new_layout,
            num_local_physical_experts=self.spec.local_experts,
            num_gpu_per_node=self.ranks_per_node,
            rank=self.rank,
            world_size=self.world_size,
            log_metrics=self.log_metrics,
            group=self.p2p_group,
            get_comm_ranks=get_comm_ranks,
            stage_transfers=stage_transfers,
        )
        return _make_weight_writeback(weights, staging, staged_rows)

    def install(self, layer: int, writeback: WeightWriteback) -> None:
        self._check_stream()
        start = time.perf_counter()
        routing = self._routing_plan
        assert routing is not None
        get_global_expert_distribution_recorder().before_layer_install(layer)
        writeback()
        _install_routing_layer(
            self.metadata, routing.host_rows[layer], layer, routing.device_rows
        )
        self.narrowed_replica_layers.discard(layer)
        if layer == routing.last_layer:
            self._routing_plan = None
        self.install_enqueue_seconds += time.perf_counter() - start

    def synchronize(self):
        # The coordinator joins the worker before draining both streams.
        self._p2p_stream.synchronize()
        self._owner_stream.synchronize()
        self._routing_plan = None

    def open_transport(self, timeout, p2p_options) -> None:
        self.plan_group = dist.new_group(backend="gloo", timeout=timeout)
        # Background P2P and foreground TP can be submitted in different orders.
        self.p2p_group = dist.new_group(
            backend="hccl", timeout=timeout, pg_options=p2p_options
        )
        # Warm the recorder's TP reduction before KV sizing.
        dist.all_reduce(
            torch.zeros(1, dtype=torch.int32, device=self.device),
            group=get_eplb_group(),
        )
        prewarm_eplb_communication(self.p2p_group)

    def close(self) -> None:
        dist.destroy_process_group(self.p2p_group)
        dist.destroy_process_group(self.plan_group)


def _make_planning_spec(model_config, metadata, moe, nodes, *, decode_worker):
    """Choose the recorder counts and placement policy for this dispatcher."""
    from sglang.srt.eplb.eplb_algorithms import compute_algorithm

    common = ExpertLocationMetadata._init_common(model_config)
    groups = common["model_config_for_expert_location"].num_groups
    algorithm = compute_algorithm(moe.eplb_algorithm, groups, nodes)
    if algorithm.name.startswith("elasticity"):
        raise ValueError("Async EPLB requires a non-elastic CPU placement algorithm")
    physical_count_mode = None
    if metadata.num_physical_experts > metadata.num_logical_experts:
        if moe.ep_dispatch_algorithm == "static":
            physical_count_mode = "static"
        else:
            physical_count_mode = "row" if moe.moe_a2a_backend == "none" else "random"
    elif nodes > 1 and moe.moe_a2a_backend != "none":
        physical_count_mode = "unique"
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
        physical_count_mode=physical_count_mode,
        score_locality=score_locality,
    )


def create_npu_async_eplb(model_config, model, *, startup_weight_load):
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_parallel,
        get_schedule,
        reports_expert_balancedness,
    )

    world_size = dist.get_world_size()
    timeout = timedelta(seconds=90)
    control = EplbControl(dist.new_group(backend="gloo", timeout=timeout))
    error = None
    try:
        import torch_npu

        _validate_migration_topology(world_size)
        if startup_weight_load is not None:
            raise ValueError(
                "Async EPLB requires completed weight loading before migration setup"
            )
        parallel = get_parallel()
        moe = get_exec().moe
        serving_mode = get_disagg().disaggregation_mode
        decode_worker = serving_mode == "decode"
        # Resolving the model's lazy weight table can fail independently on each rank.
        weights = model.routed_experts_weights_of_layer
        metadata = get_global_expert_location_metadata()
        spec = _make_planning_spec(
            model_config, metadata, moe, parallel.nnodes, decode_worker=decode_worker
        )
        p2p_buffer_mib = int(os.environ.get("HCCL_BUFFSIZE") or 64)
        if p2p_buffer_mib < 1:
            raise ValueError("HCCL_BUFFSIZE must be positive")
        p2p_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        p2p_options.hccl_config = {"hccl_buffer_size": p2p_buffer_mib}
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
        signature = (
            runtime.layer_ids,
            runtime.weight_format_digest,
            serving_mode,
            runtime.use_prefill_admission,
            runtime._p2p_batch_size,
            p2p_buffer_mib,
            moe.eplb_rebalance_num_iterations,
            moe.expert_distribution_recorder_buffer_size,
            moe.eplb_min_rebalancing_utilization_threshold
            if reports_expert_balancedness()
            else None,
            spec,
            metadata.num_logical_experts,
            moe.ep_dispatch_algorithm,
            hashlib.sha256(initial_layout.numpy().tobytes()).digest(),
        )
        setup = (signature, sorted(runtime.narrowed_replica_layers))
    except Exception as exc:
        error = exc
        setup = f"{type(exc).__name__}: {exc}"
    peers = [None] * world_size
    dist.all_gather_object(peers, setup, group=control.group)
    for rank, result in enumerate(peers):
        if isinstance(result, str):
            control.close()
            raise RuntimeError(
                f"Async EPLB setup failed on rank {rank}: {result}"
            ) from error
    if any(s != signature for s, _ in peers):
        control.close()
        raise ValueError(
            "Async EPLB ranks disagree on transport, placement, expert weight format, "
            "batching, scheduling or recorder configuration"
        )
    for _, narrowed_layers in peers:
        runtime.narrowed_replica_layers.update(narrowed_layers)
    runtime.open_transport(timeout, p2p_options)
    weight_staging_bytes = sum(t.nbytes for t in runtime.staging)
    packed_fields = runtime._packed_fields
    if packed_fields is not None:
        weight_staging_bytes += packed_fields.send.nbytes
    logger.info(
        "Async NPU EPLB ready: weight_staging_bytes=%d routing_staging_bytes=%d "
        "hccl_buffer_mib=%d layers_per_chunk=1 packed_fields=%d",
        weight_staging_bytes,
        sum(row.nbytes for row in runtime._routing_buffers.values()),
        p2p_buffer_mib,
        len(packed_fields.small) if packed_fields is not None else 0,
    )
    return AsyncEplbCoordinator(control, runtime)
