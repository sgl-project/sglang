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

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

# copy_memory_ uses a signed 32-bit element count.
_MAX_COPY_ELEMENTS = 2**31 - 1

# ACL format IDs, kept here so importing this module does not load torch_npu.
_FORMAT_NCHW = 0
_FORMAT_ND = 2
_FORMAT_FRACTAL_NZ = 29
_BASE_FORMATS = (_FORMAT_NCHW, _FORMAT_ND)


def update_expert_weights_with_staging(
    weights_by_layer: dict[int, list[torch.Tensor]],
    old_metadata: ExpertLocationMetadata,
    new_metadata: ExpertLocationMetadata,
    layer_ids: list[int],
    num_gpu_per_node: int,
    rank: int,
    group: dist.ProcessGroup | None,
    log_metrics: bool,
    update_layer_fn: Callable[..., None],
) -> dict[int, list[int]]:
    """Prepare, migrate and write back expert weights without changing model addresses."""
    changed_layer_ids, local_update_plan = _select_expert_weight_updates(
        old_metadata, new_metadata, layer_ids, rank
    )
    if not changed_layer_ids:
        return {}

    world_size = dist.get_world_size(group)
    _validate_migration_topology(world_size)
    staging = _ExpertWeightStaging(weights_by_layer, changed_layer_ids, rank)
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
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            world_size=world_size,
            missing_logical_experts_info=missing_experts,
            log_metrics=log_metrics,
            group=group,
        )
        # Unchanged local slots may still send, but need no writeback.
        if local_changed:
            staging.store()
        # Release this layer's migration views before preparing the next layer.
        del migration_weights
        if missing_experts:
            missing_experts_by_layer[layer] = missing_experts
    return missing_experts_by_layer


def _validate_migration_topology(group_size: int) -> None:
    parallel = get_parallel()
    if parallel.tp_size != parallel.moe_ep_size or parallel.moe_ep_size != group_size:
        raise ValueError(
            "NPU EPLB migration requires TP size = EP size = EPLB group size."
        )


def _select_expert_weight_updates(old_metadata, new_metadata, layer_ids, rank):
    """Return changed layers for validation and this rank's migration plan.

    A sender-only layer participates in migration but needs no local writeback.
    Validate every changed layer, even when this rank has no work in that layer.
    """
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
    """Validate chunk compatibility and identify tensors needing storage snapshots."""
    import torch_npu

    def specification(weights):
        return [
            (
                tuple(t.shape),
                t.dtype,
                t.device,
                torch_npu.get_npu_format(t) not in _BASE_FORMATS
                or not t.is_contiguous(),
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
                "and storage snapshot requirements"
            )
    return [needs_snapshot for _, _, _, needs_snapshot in expected]


class _ExpertStorageLayout(NamedTuple):
    """Physical storage of the full tensor, including shared experts and NZ padding."""

    total_experts: int
    storage_elements: int
    npu_format: int

    def expert_byte_shape(self, weight: torch.Tensor) -> tuple[int, int]:
        """Return [routed experts, physical bytes per expert] for uint8 transport."""
        elements_per_expert = self.storage_elements // self.total_experts
        bytes_per_expert = elements_per_expert * weight.element_size()
        return weight.shape[0], bytes_per_expert


def _validated_weight_owner(weight: torch.Tensor) -> torch.Tensor:
    """Validate the routed-expert prefix and return its complete backing tensor."""
    # get_moe_weights returns parameter.data[:routed_experts]. Its base retains
    # the complete expert axis, including fused shared experts at the end.
    owner = weight if weight._base is None else weight._base
    if (
        weight.ndim == 0
        or weight.storage_offset() != 0
        or owner.storage_offset() != 0
        or weight.shape[1:] != owner.shape[1:]
        or weight.stride() != owner.stride()
        or not 0 < weight.shape[0] <= owner.shape[0]
    ):
        raise ValueError("NPU EPLB requires a full expert tensor or its expert prefix.")
    # Expert rows must occupy independent contiguous blocks. Inner transposes
    # are allowed, but selecting/stepping elements within an expert is not.
    expert_elements = 1
    for stride, size in sorted(zip(owner.stride()[1:], owner.shape[1:])):
        if size > 1:
            if stride != expert_elements:
                raise ValueError("NPU EPLB requires dense storage within each expert.")
            expert_elements *= size
    if owner.stride(0) != expert_elements:
        raise ValueError("NPU EPLB requires an outermost, contiguous expert axis.")
    return owner


def _expert_storage_layout(weight: torch.Tensor) -> _ExpertStorageLayout:
    """Check that raw storage can be split into equal, complete expert blocks."""
    import torch_npu

    owner = _validated_weight_owner(weight)
    storage_elements = torch_npu.get_storage_size(owner)
    npu_format = torch_npu.get_npu_format(owner)
    # Inspect the tensor's actual format, independently of the model name.
    # This implementation accepts base formats and 3D batched FRACTAL_NZ.
    if npu_format not in _BASE_FORMATS + (_FORMAT_FRACTAL_NZ,):
        raise ValueError(f"NPU EPLB does not support storage format {npu_format}.")
    if npu_format not in _BASE_FORMATS and owner.ndim != 3:
        raise ValueError("NPU EPLB requires 3D batched NZ expert weights.")
    # copy_memory_ copies destination-sized storage without checking source
    # capacity and uses signed int element counts. Validate before any transfer.
    storage_bytes = storage_elements * owner.element_size()
    allocated_bytes = owner.untyped_storage().nbytes()
    if (
        not 0 < storage_elements <= _MAX_COPY_ELEMENTS
        or storage_elements < owner.numel()
        or storage_bytes > allocated_bytes
    ):
        raise ValueError(
            f"NPU EPLB has invalid storage capacity: {storage_elements} elements "
            f"for shape {tuple(owner.shape)} and "
            f"{allocated_bytes} allocated bytes."
        )
    if storage_elements % owner.shape[0]:
        raise ValueError(
            f"NPU EPLB cannot divide {storage_elements} storage elements evenly "
            f"among {owner.shape[0]} experts."
        )
    if npu_format in _BASE_FORMATS and storage_elements != owner.numel():
        raise ValueError("NPU EPLB requires base-format storage without padding.")
    if weight._base is None and storage_elements != owner.numel():
        raise ValueError(
            "NPU EPLB padded NZ weights must "
            "retain their full parameter base through the routed-expert slice."
        )
    return _ExpertStorageLayout(owner.shape[0], storage_elements, npu_format)


class _ExpertWeightStaging:
    """Move NPU expert storage unchanged, including NZ padding and packed bytes.

    Contiguous base formats use byte views of the routed prefix. Other layouts
    use full-storage snapshots to preserve padding and shared-expert suffixes.
    Model tensors retain their dtype, format, strides and addresses.

    load() replaces the pending snapshots; store() writes back that layer before
    the next load(). Sender-only layers may skip store().
    """

    def __init__(
        self,
        weights_by_layer: dict[int, list[torch.Tensor]],
        layer_ids: list[int],
        rank: int,
    ):
        self._needs_snapshot = _validate_migration_weights(
            weights_by_layer, layer_ids, rank
        )
        self._completion_events = deque(maxlen=2)
        self._layouts: list[_ExpertStorageLayout] = []
        self._storage_snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, layer in enumerate(layer_ids):
            for weight_index, weight in enumerate(weights_by_layer[layer]):
                try:
                    layout = _expert_storage_layout(weight)
                except ValueError as error:
                    raise ValueError(
                        f"[NPU EPLB] rank={rank} layer={layer} "
                        f"weight_index={weight_index}: {error}"
                    ) from error
                if layer_index == 0:
                    self._layouts.append(layout)
                elif layout != self._layouts[weight_index]:
                    raise ValueError(
                        f"[NPU EPLB] rank={rank} layer={layer} "
                        f"weight_index={weight_index}: storage layout {layout} "
                        f"does not match layer {layer_ids[0]}: "
                        f"{self._layouts[weight_index]}"
                    )

    def _begin_layer(self) -> None:
        # Bound queued work before allocating this layer's staging storage.
        # Each event follows the preceding layer's P2P waits and writeback.
        # With two saved events, the fourth load waits for the first layer.
        # The allocator can then reuse that layer's released storage.
        if len(self._completion_events) == self._completion_events.maxlen:
            self._completion_events.popleft().synchronize()
        previous_layer_done = torch.npu.Event()
        previous_layer_done.record()
        self._completion_events.append(previous_layer_done)

    def create_temp_buffers(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.empty(
                layout.expert_byte_shape(weight),
                dtype=torch.uint8,
                device=weight.device,
            )
            for weight, layout in zip(weights, self._layouts, strict=True)
        ]

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        self._begin_layer()
        self._storage_snapshots.clear()
        migration_weights = []
        for weight, layout, needs_snapshot in zip(
            weights, self._layouts, self._needs_snapshot, strict=True
        ):
            storage = weight
            if needs_snapshot:
                # Preserve the full parameter, including its shared-expert suffix.
                owner = weight if weight._base is None else weight._base
                storage = torch.empty(
                    layout.storage_elements, dtype=weight.dtype, device=weight.device
                )
                torch.ops.npu.copy_memory_(storage, owner, non_blocking=True)
                self._storage_snapshots.append((owner, storage))
            # Reinterpret bytes; a dtype conversion would alter FP4/FP8 values.
            # Contiguous base formats alias model storage; only routed rows move.
            routed_experts, bytes_per_expert = layout.expert_byte_shape(weight)
            expert_bytes = storage.view(torch.uint8).view(-1, bytes_per_expert)
            migration_weights.append(expert_bytes[:routed_experts])
        return migration_weights

    def store(self) -> None:
        # Direct byte views already updated model storage. Snapshots also retain
        # the shared-expert suffix, which is outside migration_weights.
        for owner, snapshot in self._storage_snapshots:
            torch.ops.npu.copy_memory_(owner, snapshot, non_blocking=True)
