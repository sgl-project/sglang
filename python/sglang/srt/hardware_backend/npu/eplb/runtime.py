"""Stage Ascend expert weights and publish routing at a forward boundary."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_eplb_group,
)
from sglang.srt.eplb.expert_location_updater import update_expert_weights_single_layer
from sglang.srt.hardware_backend.npu.eplb.coordinator import WeightWriteback
from sglang.srt.hardware_backend.npu.eplb.placement import (
    PlanningSpec,
    RecordedWindow,
    plan_migration,
)
from sglang.srt.hardware_backend.npu.eplb.plan_exchange import (
    PlacementRows,
    allocate_plan_header,
    broadcast_placement,
    populated_replica_width,
)
from sglang.srt.hardware_backend.npu.eplb.weights import (
    _ExpertWeightAdapter,
    _warm_migration_group,
)
from sglang.srt.utils import get_bool_env_var

_MAX_PACKED_FIELD_BYTES = 16 * 1024


class EplbControl:
    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.wait_seconds = 0.0
        # The forward thread owns votes and waits for each reduction to finish.
        self._vote_values = np.empty(8, dtype=np.int64)
        self._vote_flags = torch.from_numpy(self._vote_values)

    def vote(
        self, epoch: int, phase: int, layer: int, ready: bool, error: bool
    ) -> tuple[bool, bool]:
        # MAX(x), MAX(-x) detects mismatched protocol positions on every rank.
        self._vote_values[:] = (
            epoch,
            -epoch,
            phase,
            -phase,
            layer,
            -layer,
            int(not ready),
            int(error),
        )
        start = time.perf_counter()
        dist.all_reduce(self._vote_flags, op=dist.ReduceOp.MAX, group=self.group)
        self.wait_seconds += time.perf_counter() - start
        values = self._vote_flags.tolist()
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


class _RoutingTensors(NamedTuple):
    """The routing fields carried by both pinned host rows and device staging."""

    physical_to_logical: torch.Tensor
    logical_to_physical: torch.Tensor
    replica_counts: torch.Tensor
    static_dispatch: torch.Tensor | None = None

    @classmethod
    def allocate_layer(cls, metadata: ExpertLocationMetadata) -> _RoutingTensors:
        def allocate(table: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(table[0], memory_format=torch.contiguous_format)

        static_dispatch = metadata.logical_to_rank_dispatch_physical_map
        return cls(
            physical_to_logical=allocate(metadata.physical_to_logical_map),
            logical_to_physical=allocate(metadata.logical_to_all_physical_map),
            replica_counts=allocate(metadata.logical_to_all_physical_map_num_valid),
            static_dispatch=allocate(static_dispatch)
            if static_dispatch is not None
            else None,
        )

    def pin_memory(self) -> _RoutingTensors:
        return _RoutingTensors(
            *(
                table.contiguous().pin_memory() if table is not None else None
                for table in self
            )
        )

    def row_at(self, row: int) -> _RoutingTensors:
        return _RoutingTensors(
            *(table[row] if table is not None else None for table in self)
        )


def _install_routing_layer(
    current: ExpertLocationMetadata,
    layer: int,
    host_rows: _RoutingTensors,
    staged_rows: _RoutingTensors,
) -> None:
    """Publish staged device rows and their CPU mirrors in place."""
    width = staged_rows.logical_to_physical.size(-1)
    current.physical_to_logical_map[layer].copy_(staged_rows.physical_to_logical)
    replica_slots = current.logical_to_all_physical_map[layer, :, :width]
    replica_slots.copy_(staged_rows.logical_to_physical)
    replica_counts = current.logical_to_all_physical_map_num_valid[layer]
    replica_counts.copy_(staged_rows.replica_counts)
    static_dispatch = current.logical_to_rank_dispatch_physical_map
    if staged_rows.static_dispatch is not None:
        assert static_dispatch is not None
        static_dispatch[layer].copy_(staged_rows.static_dispatch)

    current.physical_to_logical_map_cpu[layer].copy_(host_rows.physical_to_logical)
    current.logical_to_all_physical_map_cpu[layer, :, :width].copy_(
        host_rows.logical_to_physical
    )
    for algorithm, (physical, count) in current._mega_gate_maps.items():
        if algorithm == "static":
            assert static_dispatch is not None
            physical[layer, ..., 0].copy_(static_dispatch[layer])
        else:
            physical[layer, :, :width].copy_(replica_slots)
            count[layer].copy_(replica_counts)


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

    host_rows: dict[int, _RoutingTensors]
    device_rows: _RoutingTensors
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
        self.p2p_batch_size = envs.SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE.get()
        if self.p2p_batch_size <= 0:
            raise ValueError("Async EPLB requires a positive P2P batch chunk size")
        self.metadata = metadata
        self.spec = spec
        self.rank = rank
        self.world_size = world_size
        self.ranks_per_node = ranks_per_node
        # Opened together after all ranks agree on the startup configuration.
        self.plan_group: dist.ProcessGroup
        self.p2p_group: dist.ProcessGroup
        # Reused only by the single planner worker. Allocate it before the startup vote.
        self._plan_header = allocate_plan_header()
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
        self._routing_buffers = _RoutingTensors.allocate_layer(metadata)
        # A loaded placement can have holes before the last populated slot.
        initial_replicas = metadata.logical_to_all_physical_map_cpu
        self._replica_width = populated_replica_width(initial_replicas)
        # An initial mapping may expose only nearby replicas on this rank.
        # Startup takes the union across ranks; install clears one layer at a time.
        self.narrowed_replica_layers: set[int] = set()
        if spec.dispatch_mode == "random":
            visible_slots = (initial_replicas[..., : self._replica_width] >= 0).sum(
                dim=(1, 2)
            )
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

    def planner(self, window: RecordedWindow):
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

        device_window: RecordedWindow | None = window

        def compute() -> PlacementRows | None:
            nonlocal device_window
            error = event_error
            placement = None
            if self.rank == 0 and error is None:
                try:
                    assert device_window is not None
                    host_window = self._copy_window_to_host(device_window)
                    device_window = None
                    slot_counts = host_window.slot_counts
                    assert slot_counts is not None
                    if slot_counts.any().item():
                        placement = plan_migration(
                            host_window,
                            self.metadata.physical_to_logical_map_cpu,
                            self.spec,
                            self.ranks_per_node,
                            self.layer_ids,
                            num_logical_experts=self.metadata.num_logical_experts,
                            narrowed_layers=self.narrowed_replica_layers,
                        )
                except Exception as exc:
                    error = str(exc)
            return broadcast_placement(
                placement,
                error,
                self.rank,
                self.plan_group,
                self.metadata.physical_to_logical_map_cpu.shape,
                self.metadata.num_logical_experts,
                header=self._plan_header,
            )

        return compute

    def _copy_window_to_host(self, window: RecordedWindow) -> RecordedWindow:
        assert window.slot_counts is not None
        self.device_module.set_device(self.device)
        with self.device_module.stream(self._p2p_stream):
            self._p2p_stream.wait_event(self._owner_ready)
            return RecordedWindow(
                slot_counts=window.slot_counts.cpu(),
                source_totals=(
                    window.source_totals.cpu()
                    if window.source_totals is not None
                    else None
                ),
                active_sources=(
                    window.active_sources.cpu()
                    if window.active_sources is not None
                    else None
                ),
            )

    def prepare(self, placement: PlacementRows | None) -> tuple[int, ...]:
        if placement is None:
            return ()
        changed, physical, logical, dispatch = placement
        incoming_width = logical.size(-1)
        # Include the previous populated columns so shrinking the active width
        # overwrites stale replicas. The graph still points to the full table.
        install_width = max(self._replica_width, incoming_width)
        self._replica_width = install_width
        num_logical = logical.size(1)
        replica_buffer = self._routing_buffers.logical_to_physical
        device_rows = self._routing_buffers._replace(
            logical_to_physical=replica_buffer.view(-1)[
                : num_logical * install_width
            ].view(num_logical, install_width)
        )
        padded_logical = logical
        if incoming_width < install_width:
            padded_logical = torch.nn.functional.pad(
                logical, (0, install_width - incoming_width), value=-1
            )
        static_dispatch = None
        if device_rows.static_dispatch is not None:
            # Without replicas the inverse table already gives the static pick.
            static_dispatch = logical[..., 0] if dispatch is None else dispatch
        # Pageable H2D blocks the worker on the current stream. Keep the plan
        # pinned until the asynchronous routing copies have drained.
        self.device_module.set_device(self.device)
        pinned = _RoutingTensors(
            physical_to_logical=physical,
            logical_to_physical=padded_logical,
            replica_counts=(logical != -1).sum(dim=-1),
            static_dispatch=static_dispatch,
        ).pin_memory()
        self._routing_plan = _RoutingPlan(
            host_rows={layer: pinned.row_at(i) for i, layer in enumerate(changed)},
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
        host_rows = routing.host_rows[layer]
        pending_work = []
        with self.device_module.stream(self._p2p_stream):
            self._p2p_stream.wait_event(self._owner_ready)
            for destination, source in zip(routing.device_rows, host_rows, strict=True):
                if destination is not None:
                    assert source is not None
                    destination.copy_(source, non_blocking=True)
            writeback = self._stage_weights(
                layer, host_rows.physical_to_logical, pending_work
            )
            if submitted is not None:
                submitted.set_result(None)
            # The caller can install only after the stream and its Work handles drain.
            self._p2p_stream.synchronize()
        self.stage_seconds += time.perf_counter() - start
        return writeback

    def _stage_weights(
        self, layer: int, planned_layout: torch.Tensor, pending_work: list[dist.Work]
    ) -> WeightWriteback:
        old_layout = self.metadata.physical_to_logical_map_cpu[layer].tolist()
        new_layout = planned_layout.tolist()
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
                self.p2p_batch_size,
            )
        self.interleaved_layers += p2p_expert_order is not None
        expert_order = p2p_expert_order or range(len(old_layout))

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

        def stage_transfers(p2p_op_infos, rows):
            staged_rows.extend(rows)
            if (
                packed is not None
                and not packed_for_local_copy
                and any(op.op == dist.isend for _, ops in p2p_op_infos for op in ops)
            ):
                packed.pack(weights)
            _submit_hccl_batches(
                p2p_op_infos, expert_order, self.p2p_batch_size, pending_work
            )

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
            self.metadata, layer, routing.host_rows[layer], routing.device_rows
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

    def staging_memory(self) -> tuple[int, int, int]:
        """Return weight bytes, routing bytes, and the number of packed fields."""
        weight_bytes = sum(tensor.nbytes for tensor in self.staging)
        routing_bytes = sum(
            tensor.nbytes for tensor in self._routing_buffers if tensor is not None
        )
        packed = self._packed_fields
        if packed is None:
            return weight_bytes, routing_bytes, 0
        return weight_bytes + packed.send.nbytes, routing_bytes, len(packed.small)

    def open_transport(
        self, cpu_timeout, hccl_timeout, p2p_options, tp_probe, warmup_buffers
    ) -> None:
        self.plan_group = dist.new_group(backend="gloo", timeout=cpu_timeout)
        # Background P2P and foreground TP can be submitted in different orders.
        self.p2p_group = dist.new_group(
            backend="hccl", timeout=hccl_timeout, pg_options=p2p_options
        )
        # Warm the recorder's TP reduction before KV sizing.
        dist.all_reduce(tp_probe, group=get_eplb_group())
        if warmup_buffers is not None:
            _warm_migration_group(self.p2p_group, *warmup_buffers)

    def close(self) -> None:
        dist.destroy_process_group(self.p2p_group)
        dist.destroy_process_group(self.plan_group)
