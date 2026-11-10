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

from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

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
    """Migrate expert storage in place without changing model tensor metadata."""
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

    for layer, old_placement, new_placement in local_update_plan:
        weights = weights_by_layer[layer]
        if not temp_buffers:
            temp_buffers = staging.create_temp_buffers()
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
        local_update_plan.append((layer, old_placement, new_placement))
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
) -> list[_ExpertStorageLayout]:
    """Validate the whole chunk once and return reusable per-weight layouts."""
    sample_layer = layer_ids[0]
    num_weights = len(weights_by_layer[sample_layer])
    if not num_weights:
        raise ValueError(
            f"[NPU EPLB] rank={rank} layer={sample_layer}: no expert weights"
        )
    layouts = []
    for layer_index, layer in enumerate(layer_ids):
        weights = weights_by_layer[layer]
        if len(weights) != num_weights:
            raise ValueError(
                f"[NPU EPLB] rank={rank} layer={layer}: expected {num_weights} "
                f"weight tensors as in layer {sample_layer}, got {len(weights)}"
            )
        for weight_index, weight in enumerate(weights):
            try:
                layout = _expert_storage_layout(weight)
            except ValueError as error:
                raise ValueError(
                    f"[NPU EPLB] rank={rank} layer={layer} "
                    f"weight_index={weight_index}: {error}"
                ) from error
            if layer_index == 0:
                layouts.append(layout)
            elif layout != layouts[weight_index]:
                raise ValueError(
                    f"[NPU EPLB] rank={rank} layer={layer} "
                    f"weight_index={weight_index}: storage layout {layout} "
                    f"does not match layer {sample_layer}: {layouts[weight_index]}"
                )
    return layouts


class _ExpertStorageLayout(NamedTuple):
    """Logical and physical metadata required to reuse a chunk's byte buffers."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    total_experts: int
    storage_elements: int
    npu_format: int
    needs_storage_alias: bool

    @property
    def expert_byte_shape(self) -> tuple[int, int]:
        """Return [routed experts, physical bytes per expert] for uint8 transport."""
        elements_per_expert = self.storage_elements // self.total_experts
        bytes_per_expert = elements_per_expert * self.dtype.itemsize
        return self.shape[0], bytes_per_expert


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
    # The byte view must stay within the original allocation, including NZ padding.
    storage_bytes = storage_elements * owner.element_size()
    allocated_bytes = owner.untyped_storage().nbytes()
    if (
        storage_elements <= 0
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
    return _ExpertStorageLayout(
        shape=tuple(weight.shape),
        dtype=weight.dtype,
        device=weight.device,
        total_experts=owner.shape[0],
        storage_elements=storage_elements,
        npu_format=npu_format,
        needs_storage_alias=(
            npu_format not in _BASE_FORMATS or not weight.is_contiguous()
        ),
    )


def _expert_storage_view(
    weight: torch.Tensor, layout: _ExpertStorageLayout
) -> torch.Tensor:
    """Expose physical expert bytes without changing the original NZ descriptor."""
    if not layout.needs_storage_alias:
        return weight.view(torch.uint8).view(layout.expert_byte_shape)

    import torch_npu

    owner = weight if weight._base is None else weight._base
    # A separate storage descriptor presents the same allocation as ND bytes.
    # Reusing owner.untyped_storage() would also reuse its NZ descriptor.
    storage = torch_npu._C._construct_storage_from_data_pointer(
        owner.data_ptr(), owner.device, layout.storage_elements * owner.element_size()
    )
    _, bytes_per_expert = layout.expert_byte_shape
    view = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        {
            "dtype": torch.uint8,
            "size": layout.expert_byte_shape,
            "stride": (bytes_per_expert, 1),
            "storage_offset": 0,
            "npu_format": _FORMAT_ND,
        },
        storage,
    )
    # The new storage borrows the pointer. Keep the owning tensor alive while
    # submitting work, then protect its allocation until this stream completes.
    view._eplb_weight_owner = owner
    owner.record_stream(torch.npu.current_stream())
    return view


class _ExpertWeightStaging:
    """Expose original expert bytes and allocate the updater's scratch buffers.

    Independent ND views preserve NZ padding and packed values. Only routed
    expert rows are exposed; shared experts retain their original storage.
    The updater stages sources before overwriting any destination, so no full
    weight snapshot or separate writeback is needed.
    """

    def __init__(
        self,
        weights_by_layer: dict[int, list[torch.Tensor]],
        layer_ids: list[int],
        rank: int,
    ):
        self._layouts = _validate_migration_weights(weights_by_layer, layer_ids, rank)

    def create_temp_buffers(self) -> list[torch.Tensor]:
        return [
            torch.empty(
                layout.expert_byte_shape,
                dtype=torch.uint8,
                device=layout.device,
            )
            for layout in self._layouts
        ]

    def load(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            _expert_storage_view(weight, layout)
            for weight, layout in zip(weights, self._layouts, strict=True)
        ]
