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

"""Ascend expert-weight staging and migration for EPLB.

Callers select these helpers after NPU distributed initialization.
Import torch_npu only when an NPU operation actually needs it.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata


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

    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    _validate_migration_topology()
    # Both Ascend 950PR and 950DT use ARCH35. Their NZ copy/cast paths do not
    # accept the transposed expert views used by MXFP4/MXFP8 inference.
    staging_type = (
        _PhysicalExpertWeightStaging if is_npu_arch35() else _ExpertWeightStaging
    )
    staging = staging_type(weights_by_layer, changed_layer_ids, rank)
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
    from sglang.srt.runtime_context import get_parallel

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

    def _begin_layer(self) -> None:
        # Bound queued work before allocating this layer's ND tensors.
        # Each event follows the preceding layer's P2P waits and writeback.
        # With two saved events, the fourth load waits for the first layer.
        # The allocator can then reuse that layer's released ND storage.
        if len(self._completion_events) == self._completion_events.maxlen:
            self._completion_events.popleft().synchronize()
        previous_layer_done = torch.npu.Event()
        previous_layer_done.record()
        self._completion_events.append(previous_layer_done)

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        self._begin_layer()
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


class _ExpertStorageLayout(NamedTuple):
    num_experts: int
    storage_elements: int
    npu_format: int

    def wire_shape(self, weight: torch.Tensor) -> tuple[int, int]:
        return (
            weight.shape[0],
            self.storage_elements // self.num_experts * weight.element_size(),
        )


def _full_expert_weight(weight: torch.Tensor) -> torch.Tensor:
    # get_moe_weights returns parameter.data[:routed_experts]. Its base retains
    # the complete expert axis, including fused shared experts at the end.
    owner = weight if weight._base is None else weight._base
    if (
        weight.ndim == 0
        or owner.ndim != weight.ndim
        or weight.storage_offset() != 0
        or owner.storage_offset() != 0
        or weight.shape[1:] != owner.shape[1:]
        or weight.stride() != owner.stride()
        or not 0 < weight.shape[0] <= owner.shape[0]
    ):
        raise ValueError("A5 EPLB requires a full expert tensor or its expert prefix.")
    # Expert rows must occupy independent contiguous blocks. Inner transposes
    # are allowed, but selecting/stepping elements within an expert is not.
    span = 1
    for stride, size in sorted(zip(owner.stride()[1:], owner.shape[1:])):
        if size > 1:
            if stride != span:
                raise ValueError("A5 EPLB requires dense storage within each expert.")
            span *= size
    if owner.stride(0) != span:
        raise ValueError("A5 EPLB requires an outermost, contiguous expert axis.")
    return owner


def _expert_storage_layout(weight: torch.Tensor) -> _ExpertStorageLayout:
    import torch_npu

    owner = _full_expert_weight(weight)
    storage_elements = torch_npu.get_storage_size(owner)
    npu_format = torch_npu.get_npu_format(owner)
    # ND and batched NZ (including C0_16/C0_32) keep experts independent. Other
    # internal formats must not be split by simply dividing their storage size.
    if npu_format not in (0, 2, 29, 50, 51):
        raise ValueError(f"A5 EPLB does not support storage format {npu_format}.")
    if npu_format not in (0, 2) and owner.ndim != 3:
        raise ValueError("A5 EPLB requires 3D batched NZ expert weights.")
    # copy_memory_ copies destination-sized storage without checking source
    # capacity and uses signed int element counts. Validate before any transfer.
    if (
        not 0 < storage_elements <= 2**31 - 1
        or storage_elements < owner.numel()
        or storage_elements % owner.shape[0]
        or storage_elements * owner.element_size() > owner.untyped_storage().nbytes()
        or (npu_format in (0, 2) and storage_elements != owner.numel())
        or (weight._base is None and storage_elements != owner.numel())
    ):
        raise ValueError(
            "A5 EPLB requires complete expert storage; padded NZ weights must "
            "retain their full parameter base through the routed-expert slice."
        )
    return _ExpertStorageLayout(owner.shape[0], storage_elements, npu_format)


class _PhysicalExpertWeightStaging(_ExpertWeightStaging):
    """Move A5 expert storage unchanged, including NZ padding and packed bytes.

    Contiguous ND weights use byte views of the routed prefix. Other layouts
    use full-storage snapshots to preserve padding and shared-expert suffixes.
    Model tensors retain their dtype, format, strides and addresses.
    """

    def __init__(self, weights_by_layer, layer_ids, rank):
        super().__init__(weights_by_layer, layer_ids, rank)
        self._layouts = [
            _expert_storage_layout(weight) for weight in weights_by_layer[layer_ids[0]]
        ]
        for layer in layer_ids[1:]:
            if [
                _expert_storage_layout(weight) for weight in weights_by_layer[layer]
            ] != self._layouts:
                raise ValueError(
                    f"[NPU EPLB] rank={rank} layer={layer}: expert storage layouts "
                    "must match across the migration chunk"
                )
        self._storage_snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []

    def create_temp_buffers(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.empty(
                layout.wire_shape(weight), dtype=torch.uint8, device=weight.device
            )
            for weight, layout in zip(weights, self._layouts, strict=True)
        ]

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        self._begin_layer()
        self._storage_snapshots = []
        migration_weights = []
        for weight, layout, needs_staging in zip(
            weights, self._layouts, self._needs_staging, strict=True
        ):
            if not needs_staging:
                # Keep byte transport for FP8 scales, without copying ND storage.
                migration_weights.append(
                    weight.view(torch.uint8).view(layout.wire_shape(weight))
                )
                continue
            # The chunk's owners and layouts were validated during initialization.
            owner = weight if weight._base is None else weight._base
            snapshot = torch.empty(
                layout.storage_elements, dtype=weight.dtype, device=weight.device
            )
            torch.ops.npu.copy_memory_(snapshot, owner, non_blocking=True)
            self._storage_snapshots.append((owner, snapshot))
            # Reinterpret bytes; a dtype conversion would alter FP4/FP8 values.
            rows = snapshot.view(torch.uint8).view(layout.num_experts, -1)
            migration_weights.append(rows[: weight.shape[0]])
        return migration_weights

    def store(
        self, weights: list[torch.Tensor], migration_weights: list[torch.Tensor]
    ) -> None:
        # Direct ND views already updated model storage. Snapshots also retain
        # the shared-expert suffix, which is outside migration_weights.
        for owner, snapshot in self._storage_snapshots:
            torch.ops.npu.copy_memory_(owner, snapshot, non_blocking=True)
