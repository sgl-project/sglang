"""NPU implementation of the forward-boundary EPLB protocol."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import timedelta

import torch
import torch.distributed as dist

from sglang.srt.eplb.async_rebalance import AsyncEplbCoordinator, plan_layer_pull
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.hardware_backend.npu.eplb import (
    _ExpertWeightAdapter,
    _validate_migration_topology,
)

logger = logging.getLogger(__name__)


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
        transport_factory,
    ):
        self.metadata, self.spec = metadata, spec
        self.rank, self.world_size, self.ranks_per_node = (
            rank,
            world_size,
            ranks_per_node,
        )
        self.plan_group = plan_group
        self.layer_ids = tuple(sorted(weights))
        adapter = _ExpertWeightAdapter(weights, list(self.layer_ids), rank)
        self.weights = {
            layer: adapter.make_weight_views(weights[layer]) for layer in self.layer_ids
        }
        self.staging = adapter.create_temp_buffers()
        self.device = self.staging[0].device
        self.device_module = torch.get_device_module(self.device)
        self.transport = transport_factory(self.weights, self.staging)
        self._prepared = self.device_module.Event()
        self._prepared.record()
        self._consumed = self.device_module.Event()
        self._consumed.record()
        self._owner_stream = self.device_module.current_stream()
        self._new_metadata = None
        self.plans = {}
        self.read_seconds = self.install_enqueue_seconds = 0.0
        self.bytes_read = 0

    def planner(self, counts):
        self.read_seconds = self.install_enqueue_seconds = 0.0
        self.bytes_read = 0
        # Snapshot on the caller's inference stream before recorder buffers reuse.
        self._owner_stream = self.device_module.current_stream()
        snapshot_error = None
        try:
            host = torch.empty(
                counts.shape, dtype=counts.dtype, device="cpu", pin_memory=True
            )
            host.copy_(counts, non_blocking=True)
            event = self.device_module.Event()
            event.record(self._owner_stream)
        except Exception as error:
            snapshot_error = error
        spec, rank, group = self.spec, self.rank, self.plan_group

        def compute():
            error = snapshot_error
            if error is None:
                try:
                    self.device_module.set_device(self.device)
                    event.synchronize()
                except Exception as exc:
                    error = exc
            # Keep counts alive until the D2H finishes, then release the capture.
            _ = counts
            failed = torch.tensor([int(error is not None)], dtype=torch.int32)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
            if failed.item():
                raise RuntimeError("EPLB statistics snapshot failed") from error
            payload = [None]
            if rank == 0:
                try:
                    payload[0] = (None, spec.compute(host))
                except Exception as error:
                    payload[0] = (str(error), None)
            dist.broadcast_object_list(payload, src=0, group=group)
            error, result = payload[0]
            if error is not None:
                raise RuntimeError(f"EPLB CPU planner failed: {error}")
            return result

        return compute

    def prepare(self, placement):
        physical, logical = placement
        if physical.shape != self.metadata.physical_to_logical_map_cpu.shape:
            raise ValueError("EPLB plan changed physical capacity")
        old = self.metadata.physical_to_logical_map_cpu
        changed = tuple(
            layer
            for layer in self.layer_ids
            if not torch.equal(old[layer], physical[layer])
        )
        self.plans = {
            layer: plan_layer_pull(
                tuple(old[layer].tolist()),
                tuple(physical[layer].tolist()),
                self.rank,
                self.world_size,
                self.ranks_per_node,
            )
            for layer in changed
        }
        # Reuse main's exact dispatch construction on CPU. Building it on NPU
        # would immediately copy it back to CPU and synchronize the forward.
        cpu_metadata = ExpertLocationMetadata._init_raw(
            ep_size=self.metadata.ep_size,
            physical_to_logical_map=physical,
            logical_to_all_physical_map=logical,
        )
        device_fields = (
            "physical_to_logical_map",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        )
        self._new_metadata = replace(
            cpu_metadata,
            **{
                name: value.to(self.device, non_blocking=True)
                for name in device_fields
                if (value := getattr(cpu_metadata, name)) is not None
            },
        )
        self._owner_stream = self.device_module.current_stream()
        self._prepared.record(self._owner_stream)
        return changed

    def preparation_ready(self):
        return self._prepared.query()

    def _check_stream(self):
        if self.device_module.current_stream() != self._owner_stream:
            raise RuntimeError("EPLB target forwards changed their owning stream")

    def stage(self, layer):
        self._check_stream()
        plan = self.plans[layer]
        for source, destination in plan.local_copies:
            for weight, buffer in zip(self.weights[layer], self.staging, strict=True):
                buffer[destination].copy_(weight[source])
        copied = self.device_module.Event()
        copied.record(self._owner_stream)
        consumed = self._consumed

        def read():
            self.device_module.set_device(self.device)
            # HIXL owns a private device stream. A torch wait_event enqueue is
            # insufficient: wait on the host before handing staging back to DMA.
            consumed.synchronize()
            copied.synchronize()
            start = time.perf_counter()
            self.transport.read(layer, plan.reads)
            self.read_seconds += time.perf_counter() - start
            self.bytes_read += len(plan.reads) * sum(t.shape[1] for t in self.staging)

        return read

    def install(self, layer):
        self._check_stream()
        start = time.perf_counter()
        for source, destination in self.plans[layer].installs:
            for weight, buffer in zip(self.weights[layer], self.staging, strict=True):
                weight[destination].copy_(buffer[source])
        self.metadata.update(self._new_metadata, update_layer_ids=[layer])
        self._consumed = self.device_module.Event()
        self._consumed.record(self._owner_stream)
        self.install_enqueue_seconds += time.perf_counter() - start

    def synchronize(self):
        self.transport.drain()
        # Also cover a partial foreground copy whose event enqueue failed.
        self._owner_stream.synchronize()

    def close(self):
        self.transport.close()


def create_npu_async_eplb(model_config, weights, metadata):
    from sglang.srt.eplb.eplb_algorithms import compute_algorithm
    from sglang.srt.hardware_backend.npu.hixl_transport import HixlExpertTransport
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang.srt.utils.network import get_local_ip_auto, get_open_port

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
    transport_group = dist.new_group(backend="gloo", timeout=timeout)
    endpoint = f"{get_local_ip_auto()}:{get_open_port()}"
    runtime = NpuEplbRuntime(
        weights,
        metadata,
        spec,
        parallel.tp_rank,
        world_size,
        world_size // parallel.nnodes,
        plan_group,
        lambda views, staging: HixlExpertTransport(
            views, staging, transport_group, endpoint
        ),
    )
    logger.info(
        "Async NPU EPLB ready: staging_bytes=%d",
        sum(t.numel() for t in runtime.staging),
    )
    return AsyncEplbCoordinator(EplbControl(control_group), runtime)
