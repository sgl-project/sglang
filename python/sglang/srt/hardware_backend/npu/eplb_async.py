"""NPU implementation of the forward-boundary EPLB protocol."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    compute_logical_to_rank_dispatch_physical_map,
    get_eplb_group,
)
from sglang.srt.eplb.expert_location_updater import update_expert_weights_single_layer
from sglang.srt.hardware_backend.npu.distributed import (
    eplb_hccl_options,
    eplb_reuses_world_group,
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
from sglang.srt.hardware_backend.npu.eplb_placement import (
    MigrationPlacement,
    PlanningSpec,
    plan_migration,
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
    current.physical_to_logical_map_cpu[layer].copy_(
        planned["physical_to_logical_map"][layer]
    )
    current.logical_to_all_physical_map_cpu[layer, :, :width].copy_(
        planned["logical_to_all_physical_map"][layer]
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


def _broadcast_placement(
    placement: MigrationPlacement | None,
    error: str | None,
    rank: int,
    group: dist.ProcessGroup,
    physical_shape: torch.Size,
    num_logical: int,
) -> MigrationPlacement | None:
    """Use a fixed header and one CPU tensor for a successful plan."""
    # The first header value is -1 on failure, 0 for an empty plan, or the
    # number of layers to install. The second is the replica table width.
    header = torch.zeros(2, dtype=torch.int64, device="cpu")
    if rank == 0:
        if error is None and placement is not None:
            try:
                changed, physical, logical = placement
                width = logical.size(-1)
                if physical.shape != physical_shape or logical.shape != (
                    physical_shape[0],
                    num_logical,
                    width,
                ):
                    raise ValueError("EPLB placement changed the routing geometry")
                payload = torch.cat(
                    (
                        torch.tensor(changed, dtype=torch.int64, device="cpu"),
                        physical.flatten(),
                        logical.flatten(),
                    )
                )
                payload = payload.to(torch.int64)
                header[0] = len(changed)
                header[1] = width
            except Exception as exc:
                error = str(exc)
        if error is not None:
            header[0] = -1
    dist.broadcast(header, src=0, group=group)
    count, width = header.tolist()
    if count < 0:
        message = [error]
        dist.broadcast_object_list(message, src=0, group=group)
        raise RuntimeError(f"EPLB CPU planner failed: {message[0]}")
    if count == 0:
        return None
    layers, slots = physical_shape
    physical_end = count + layers * slots
    logical_shape = (layers, num_logical, width)
    if rank != 0:
        payload = torch.empty(
            physical_end + layers * num_logical * width, dtype=torch.int64, device="cpu"
        )
    dist.broadcast(payload, src=0, group=group)
    return MigrationPlacement(
        tuple(payload[:count].tolist()),
        payload[count:physical_end].view(physical_shape),
        payload[physical_end:].view(logical_shape),
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

    def pack(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        torch.cat([weights[i] for i in self.small], dim=1, out=self.send)
        return [weights[i] for i in self.direct] + [self.send]


@dataclass(frozen=True)
class _RoutingPlan:
    """Pinned host tables and views of the reusable one-layer device buffers."""

    host_rows: dict[str, torch.Tensor]
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
        plan_group: dist.ProcessGroup,
        p2p_group: dist.ProcessGroup,
        *,
        owns_p2p_group: bool = True,
        order_submissions: bool = False,
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
        self.plan_group = plan_group
        self.p2p_group = p2p_group
        self._owns_p2p_group = owns_p2p_group
        self.log_metrics = get_bool_env_var(
            "SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS"
        )
        self.layer_ids = tuple(sorted(weights))
        adapter = _ExpertWeightAdapter(weights, list(self.layer_ids), rank)
        self.weights = {
            layer: adapter.make_weight_views(weights[layer]) for layer in self.layer_ids
        }
        self.staging = adapter.create_temp_buffers()
        self._packed_fields = (
            _PackedExpertFields.create(self.staging)
            if order_submissions and world_size == ranks_per_node
            else None
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
        self.device = self.staging[0].device
        self.device_module = torch.get_device_module(self.device)
        self._p2p_stream = self.device_module.Stream(device=self.device)
        self._owner_stream = self.device_module.current_stream()
        self._owner_ready = self.device_module.Event()
        self._order_submissions = order_submissions
        self._routing_plan: _RoutingPlan | None = None
        self.stage_seconds = self.install_enqueue_seconds = 0.0
        self.interleaved_layers = 0

    def planner(self, counts):
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
        else:
            counts = None

        def compute():
            nonlocal counts
            error = event_error
            placement = None
            if self.rank == 0 and error is None:
                try:
                    host_counts = self._copy_counts_to_host(counts)
                    counts = None
                    if host_counts.any().item():
                        placement = plan_migration(
                            host_counts,
                            self.metadata.physical_to_logical_map_cpu,
                            self.spec,
                            self.ranks_per_node,
                            self.layer_ids,
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

    def prepare(self, placement: MigrationPlacement | None) -> tuple[int, ...]:
        if placement is None:
            return ()
        changed, physical, logical = placement
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
            # Preserve the shared static algorithm's draws in full layer order.
            host_rows["logical_to_rank_dispatch_physical_map"] = (
                compute_logical_to_rank_dispatch_physical_map(
                    logical, self.metadata.ep_size, physical.size(1), self.rank
                )
            )
        # Pageable H2D blocks the worker on the current stream. Keep the plan
        # pinned until the asynchronous routing copies have drained.
        self.device_module.set_device(self.device)
        self._routing_plan = _RoutingPlan(
            host_rows={name: rows.pin_memory() for name, rows in host_rows.items()},
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
        staged_rows = []
        pending_work = []
        old_layout = self.metadata.physical_to_logical_map_cpu[layer].tolist()
        new_layout = routing.host_rows["physical_to_logical_map"][layer].tolist()
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
            _submit_hccl_batches(
                p2p_op_infos, expert_order, self._p2p_batch_size, pending_work
            )

        with self.device_module.stream(self._p2p_stream):
            self._p2p_stream.wait_event(self._owner_ready)
            for name, row in routing.device_rows.items():
                row.copy_(routing.host_rows[name][layer], non_blocking=True)
            weights, staging = self.weights[layer], self.staging
            sends, receives = weights, staging
            if self._packed_fields is not None:
                sends = self._packed_fields.pack(weights)
                receives = self._packed_fields.receive
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
            writeback = _make_weight_writeback(weights, staging, staged_rows)
            if submitted is not None:
                submitted.set_result(None)
            # The caller can install only after the stream and its Work handles drain.
            self._p2p_stream.synchronize()
        self.stage_seconds += time.perf_counter() - start
        return writeback

    def install(self, layer: int, writeback: WeightWriteback) -> None:
        self._check_stream()
        start = time.perf_counter()
        routing = self._routing_plan
        assert routing is not None
        get_global_expert_distribution_recorder().before_layer_install(layer)
        writeback()
        _install_routing_layer(
            self.metadata, routing.host_rows, layer, routing.device_rows
        )
        if layer == routing.last_layer:
            self._routing_plan = None
        self.install_enqueue_seconds += time.perf_counter() - start

    def synchronize(self):
        # The coordinator joins the worker before draining both streams.
        self._p2p_stream.synchronize()
        self._owner_stream.synchronize()
        self._routing_plan = None

    def close(self):
        if self._owns_p2p_group:
            dist.destroy_process_group(self.p2p_group)


def create_npu_async_eplb(model_config, weights, metadata):
    from sglang.srt.eplb.eplb_algorithms import compute_algorithm
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel

    parallel = get_parallel()
    world_size = dist.get_world_size()
    _validate_migration_topology(world_size)
    common = ExpertLocationMetadata._init_common(model_config)
    groups = common["model_config_for_expert_location"].num_groups
    moe = get_exec().moe
    algorithm = compute_algorithm(moe.eplb_algorithm, groups, parallel.nnodes)
    order_submissions = get_disagg().disaggregation_mode == "decode"
    if algorithm.name.startswith("elasticity"):
        raise ValueError("Async EPLB requires a non-elastic CPU placement algorithm")
    spec = PlanningSpec(
        metadata.num_physical_experts,
        metadata.num_local_physical_experts,
        groups,
        parallel.nnodes,
        algorithm,
        rank_fixed_dispatch=moe.ep_dispatch_algorithm == "static",
    )
    # Fixed PP=1 TP=EP=WORLD: group construction order is identical on all ranks.
    timeout = timedelta(seconds=90)
    control_group = dist.new_group(backend="gloo", timeout=timeout)
    plan_group = dist.new_group(backend="gloo", timeout=timeout)
    p2p_options = eplb_hccl_options()
    p2p_buffer_mib = p2p_options.hccl_config["hccl_buffer_size"]
    borrow_world_group = eplb_reuses_world_group()
    if borrow_world_group:
        p2p_group = parallel.world_group.device_group
    else:
        p2p_group = dist.new_group(
            backend="hccl", timeout=timeout, pg_options=p2p_options
        )
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
            owns_p2p_group=not borrow_world_group,
            order_submissions=order_submissions,
        )
        initial_layout = metadata.physical_to_logical_map_cpu.contiguous()
        if any(t.shape[0] != spec.local_experts for t in runtime.staging):
            raise ValueError(
                "Async EPLB weight rows do not match the local expert count"
            )
        signature = (
            runtime.layer_ids,
            tuple(tuple(t.shape) for t in runtime.staging),
            order_submissions,
            runtime._p2p_batch_size,
            p2p_buffer_mib,
            spec,
            metadata.num_logical_experts,
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
    # The recorder reduces int32 counts on TP. Initialize its HCCL collective
    # before KV sizing and before prewarm_eplb_communication releases the cache.
    dist.all_reduce(
        torch.zeros(1, dtype=torch.int32, device=runtime.device),
        group=get_eplb_group(),
    )
    prewarm_eplb_communication(p2p_group)
    weight_staging_bytes = sum(t.nbytes for t in runtime.staging)
    packed_fields = runtime._packed_fields
    if packed_fields is not None:
        weight_staging_bytes += packed_fields.send.nbytes
    logger.info(
        "Async NPU EPLB ready: weight_staging_bytes=%d routing_staging_bytes=%d "
        "hccl_buffer_mib=%d reuse_world_group=%s layers_per_chunk=1 packed_fields=%d",
        weight_staging_bytes,
        sum(row.nbytes for row in runtime._routing_buffers.values()),
        p2p_buffer_mib,
        borrow_world_group,
        len(packed_fields.small) if packed_fields is not None else 0,
    )
    return AsyncEplbCoordinator(EplbControl(control_group), runtime)
