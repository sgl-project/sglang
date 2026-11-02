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

"""Ascend EPLB statistics, communication reuse, and expert-weight staging.

Callers select these helpers after NPU distributed initialization.
Import torch_npu only when an NPU operation actually needs it.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist

from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
from sglang.srt.runtime_context import get_device, get_exec, get_parallel

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata


_warmed_group = None


def get_eplb_group():
    """Reuse a WORLD-equivalent TP group for NPU statistics and migration."""
    if not dist.is_initialized():
        return None
    tp_group = get_parallel().tp_group
    # Collective roots and migration peers use global ranks. Elastic groups and
    # TP subgroups keep the existing default-WORLD path.
    if (
        get_exec().moe.elastic_ep_backend
        or tp_group is None
        or tp_group.ranks != list(range(dist.get_world_size()))
        or dist.get_backend(tp_group.device_group) != dist.get_backend()
    ):
        return None
    return tp_group.device_group


def _warmup_eplb_communication(group):
    """Join all ranks before migration can skip ranks without local transfers."""
    global _warmed_group
    if group is None or group is _warmed_group:
        return
    world_size = dist.get_world_size(group)
    if world_size > 1:
        rank = dist.get_rank()
        send = torch.zeros(1, dtype=torch.int32, device=get_device().device)
        recv = torch.empty((world_size, 1), dtype=torch.int32, device=send.device)
        dist.all_reduce(send, group=group)
        ops = []
        for peer in range(world_size):
            if peer != rank:
                ops.append(dist.P2POp(dist.isend, send, peer, group=group))
                ops.append(dist.P2POp(dist.irecv, recv[peer], peer, group=group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        # Keep the startup buffers alive until NPU communication completes.
        torch.npu.synchronize()
    _warmed_group = group


class NpuExpertDistributionRecorder(_ExpertDistributionRecorderReal):
    """Keep NPU draft passes out of target statistics and reuse TP communication."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        group = get_eplb_group()
        if get_exec().moe.enable_eplb:
            _warmup_eplb_communication(group)
        self._accumulator._communication_group = group

    @contextmanager
    def with_forward_pass(self, forward_pass_id, forward_batch, *, is_draft=False):
        if self._disable_all or is_draft:
            with self.disable_this_region():
                yield {}
            return
        with super().with_forward_pass(forward_pass_id, forward_batch) as outputs:
            yield outputs


def update_expert_weights_with_staging(
    weights_by_layer: dict[int, list[torch.Tensor]],
    old_metadata: ExpertLocationMetadata,
    new_metadata: ExpertLocationMetadata,
    layer_ids: list[int],
    nnodes: int,
    rank: int,
    log_metrics: bool,
    update_layer_fn: Callable[..., None],
) -> dict[int, list[int]]:
    """Migrate each participating layer through ND storage, preserving model addresses."""
    changed_layer_ids, local_update_plan = _select_expert_weight_updates(
        old_metadata, new_metadata, layer_ids, rank
    )
    if not changed_layer_ids:
        return {}

    _validate_migration_topology()
    group = get_eplb_group()
    staging = _ExpertWeightStaging(weights_by_layer, changed_layer_ids, rank)
    world_size = dist.get_world_size()
    temp_buffers = []
    missing_experts_by_layer = {}

    for layer, old_placement, new_placement, local_changed in local_update_plan:
        weights = weights_by_layer[layer]
        if not temp_buffers:
            temp_buffers = staging.create_temp_buffers(weights)
        migration_weights = staging.load(weights)
        missing_experts = []
        update_layer_fn(
            routed_experts_weights=migration_weights,
            temp_buffers=temp_buffers,
            old_physical_to_logical_map=old_placement,
            new_physical_to_logical_map=new_placement,
            num_local_physical_experts=old_metadata.num_local_physical_experts,
            num_gpu_per_node=world_size // nnodes,
            rank=rank,
            world_size=world_size,
            missing_logical_experts_info=missing_experts,
            log_metrics=log_metrics,
            group=group,
        )
        # Unchanged local slots may still send, but need no writeback.
        if local_changed:
            staging.store(weights, migration_weights)
        # Release this layer's ND references before preparing the next layer.
        del migration_weights
        if missing_experts:
            missing_experts_by_layer[layer] = missing_experts
    return missing_experts_by_layer


def _validate_migration_topology() -> None:
    parallel = get_parallel()
    if (
        parallel.tp_size != parallel.moe_ep_size
        or parallel.moe_ep_size != dist.get_world_size()
    ):
        raise ValueError("NPU EPLB migration requires TP size = EP size = WORLD size.")


def _select_expert_weight_updates(old_metadata, new_metadata, layer_ids, rank):
    """Select globally updated layers and this rank's participating layers."""
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

    old_placements = old_metadata.physical_to_logical_map_cpu
    new_placements = new_metadata.physical_to_logical_map_cpu
    elastic_ep_enabled = ElasticEPStateManager.instance() is not None
    changed_layer_ids = layer_ids
    if not elastic_ep_enabled:
        changed_layer_ids = [
            layer
            for layer in layer_ids
            if not torch.equal(old_placements[layer], new_placements[layer])
        ]
    num_local_experts = old_metadata.num_local_physical_experts
    start = rank * num_local_experts
    local_slots = slice(start, start + num_local_experts)
    local_update_plan = []
    for layer in changed_layer_ids:
        old_placement = old_placements[layer].tolist()
        new_placement = new_placements[layer].tolist()
        local_changed = old_placement[local_slots] != new_placement[local_slots]
        if (
            not elastic_ep_enabled
            and not local_changed
            and not _rank_may_send_experts(old_placement, new_placement, local_slots)
        ):
            continue
        local_update_plan.append((layer, old_placement, new_placement, local_changed))
    # Validate the whole selected chunk, including layers this rank skips.
    return changed_layer_ids, local_update_plan


def _rank_may_send_experts(old_placement, new_placement, local_slots):
    """A possible sender holds an expert needed by a changed destination slot."""
    local_experts = set(old_placement[local_slots])
    return any(
        old != new and new in local_experts
        for old, new in zip(old_placement, new_placement, strict=True)
    )


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
    """Validate a chunk and prepare its weights for ND migration.

    load() prepares the tensors that the updater mutates; store() copies the
    result into model storage without changing addresses held by captured graphs.
    """

    def __init__(
        self,
        weights_by_layer: dict[int, list[torch.Tensor]],
        layer_ids: list[int],
        rank: int,
    ):
        # Convert internal formats and pack strided ND before migrating slices.
        self._needs_staging = _validate_migration_weights(
            weights_by_layer, layer_ids, rank
        )
        self._completion_events = deque(maxlen=2)

    def create_temp_buffers(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
            for weight in weights
        ]

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        # Bound queued work before allocating this layer's ND tensors.
        # Each event follows the preceding layer's P2P waits and writeback.
        # With two saved events, the fourth load waits for the first layer.
        # The allocator can then reuse that layer's released ND storage.
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
