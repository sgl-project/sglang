# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Ascend EPLB communication and expert-weight migration.

Callers select these helpers after NPU distributed initialization.
Import torch_npu only when an NPU operation actually needs it.
"""

import logging
import time
from collections import deque
from typing import Callable

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_warmed_device_group = None


def get_eplb_device_group() -> dist.ProcessGroup:
    """Reuse TP's device group and its existing communication configuration."""
    from sglang.srt.runtime_context import get_parallel

    tp_group = get_parallel().tp_group
    # Shared statistics and migration use WORLD ranks, including root rank 0.
    if tp_group.ranks != list(range(dist.get_world_size())):
        raise ValueError(
            "NPU EPLB requires a TP group covering WORLD in rank order; "
            "TP subgroups are not supported."
        )
    return tp_group.device_group


def _get_migration_group() -> dist.ProcessGroup:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    # Migration indexes each rank's expert slots; statistics only sum counts.
    if parallel.tp_size != parallel.moe_ep_size:
        raise ValueError("NPU EPLB migration requires TP size to equal EP size.")
    return get_eplb_device_group()


def warmup_eplb_communication() -> None:
    """Initialize statistics and all-peer migration channels before serving."""
    global _warmed_device_group
    group = _get_migration_group()
    if group is _warmed_device_group:
        return

    start = time.perf_counter()
    world_size = dist.get_world_size(group)
    if world_size > 1:
        rank = dist.get_rank()
        send = torch.zeros(1, dtype=torch.int32, device="npu")
        recv = torch.empty((world_size, 1), dtype=torch.int32, device="npu")
        dist.all_reduce(send, group=group)
        ops = []
        for peer in range(world_size):
            if peer != rank:
                ops.append(dist.P2POp(dist.isend, send, peer, group=group))
                ops.append(dist.P2POp(dist.irecv, recv[peer], peer, group=group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        # Retain the tiny buffers until communication has actually completed.
        torch.npu.synchronize()
    _warmed_device_group = group
    logger.info(
        "[NPU EPLB] communication warmup finished: ranks=%d time=%.3fs",
        world_size,
        time.perf_counter() - start,
    )


def update_expert_weights_npu(
    *,
    weights_by_layer: dict[int, list[torch.Tensor]],
    layer_ids: list[int],
    old_placement_by_layer: torch.Tensor,
    new_placement_by_layer: torch.Tensor,
    num_local_physical_experts: int,
    nnodes: int,
    rank: int,
    update_layer_fn: Callable[..., None],
    log_metrics: bool = False,
) -> dict[int, list[int]]:
    """Migrate a chunk of Ascend expert weights with the shared single-layer updater.

    Validate the entire chunk before allocating workspaces or moving weights.
    Reuse receive buffers across participating layers, convert each layer to ND,
    and write changed local slots back into the original model storage.
    Return missing logical experts by layer for elastic EP recovery.
    """
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

    elastic_ep_enabled = ElasticEPStateManager.instance() is not None
    if not elastic_ep_enabled:
        # A rank with unchanged local slots may still need to send to another rank.
        layer_ids = [
            layer
            for layer in layer_ids
            if not torch.equal(
                old_placement_by_layer[layer], new_placement_by_layer[layer]
            )
        ]
    if not layer_ids:
        return {}

    needs_staging = _validate_migration_weights(weights_by_layer, layer_ids, rank)
    staging = _ExpertWeightStaging(needs_staging)
    group = _get_migration_group()
    world_size = dist.get_world_size()
    num_gpu_per_node = world_size // nnodes
    start = rank * num_local_physical_experts
    end = start + num_local_physical_experts
    temp_buffers = []
    missing_experts_by_layer = {}

    for layer in layer_ids:
        old_placement = old_placement_by_layer[layer].tolist()
        new_placement = new_placement_by_layer[layer].tolist()
        local_changed = old_placement[start:end] != new_placement[start:end]
        if not local_changed and not elastic_ep_enabled:
            # Any outgoing expert must appear in a changed destination slot.
            # Keep possible senders; only skip ranks with provably no work.
            local_experts = set(old_placement[start:end])
            if not any(
                old != new and new in local_experts
                for old, new in zip(old_placement, new_placement, strict=True)
            ):
                continue
        weights = weights_by_layer[layer]
        # Allocate only when this rank participates, then reuse across layers.
        if not temp_buffers:
            temp_buffers = staging.create_temp_buffers(weights)
        migration_weights = staging.load(weights)
        missing_experts = []
        update_layer_fn(
            routed_experts_weights=migration_weights,
            temp_buffers=temp_buffers,
            old_physical_to_logical_map=old_placement,
            new_physical_to_logical_map=new_placement,
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            world_size=world_size,
            missing_logical_experts_info=missing_experts,
            log_metrics=log_metrics,
            group=group,
        )
        # Unchanged local slots may still send; only their writeback is redundant.
        if local_changed:
            staging.store(weights, migration_weights)
        # Release this layer's ND references before preparing the next layer.
        del migration_weights
        if missing_experts:
            missing_experts_by_layer[layer] = missing_experts

    return missing_experts_by_layer


def _validate_migration_weights(
    weights_by_layer: dict[int, list[torch.Tensor]], layer_ids: list[int], rank: int
) -> list[bool]:
    """Validate chunk compatibility and return its ND staging requirements."""
    import torch_npu

    def specification(weights):
        return [
            (
                tuple(t.shape),
                t.dtype,
                t.device,
                torch_npu.get_npu_format(t) not in (0, 2) or not t.is_contiguous(),
            )
            for t in weights
        ]

    sample_layer = layer_ids[0]
    expected = specification(weights_by_layer[sample_layer])
    if not expected:
        raise ValueError(
            f"[NPU EPLB] rank={rank} layer={sample_layer}: no expert weights"
        )
    for layer in layer_ids[1:]:
        if specification(weights_by_layer[layer]) != expected:
            raise ValueError(
                f"[NPU EPLB] rank={rank} layer={layer}: migration weights must match "
                f"layer {sample_layer} in tensor count, shape, dtype, device, "
                "and ND staging requirements"
            )
    return [needs_staging for _, _, _, needs_staging in expected]


class _ExpertWeightStaging:
    """Prepare ND migration weights for a validated chunk's tensor formats.

    load() prepares the tensors that the updater mutates; store() copies the
    result into model storage without changing addresses held by captured graphs.
    """

    def __init__(self, needs_staging: list[bool]):
        # Convert internal formats and pack strided ND before migrating slices.
        self._needs_staging = needs_staging
        self._completion_events = deque(maxlen=2)

    def create_temp_buffers(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
            for weight in weights
        ]

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        # Bound queued work before allocating this layer's ND tensors.
        # The event recorded here covers the preceding layer's P2P and writeback.
        if len(self._completion_events) == self._completion_events.maxlen:
            self._completion_events.popleft().synchronize()
        previous_layer_done = torch.npu.Event()
        previous_layer_done.record()
        self._completion_events.append(previous_layer_done)
        # Keep the conversion result; copy_ would add another ND-to-ND copy.
        return [
            torch.ops.npu.npu_format_cast(weight, 2).contiguous() if needed else weight
            for weight, needed in zip(weights, self._needs_staging)
        ]

    def store(
        self, weights: list[torch.Tensor], migration_weights: list[torch.Tensor]
    ) -> None:
        for weight, buffer, needed in zip(
            weights, migration_weights, self._needs_staging
        ):
            if not needed:
                continue
            restored = torch.ops.npu.npu_format_cast(
                buffer, torch.ops.npu.get_npu_format(weight)
            )
            # Keep the storage addresses referenced by captured graphs.
            weight.copy_(restored)
