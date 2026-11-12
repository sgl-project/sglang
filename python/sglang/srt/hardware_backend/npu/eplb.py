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

"""Ascend expert-weight storage views and migration for EPLB.

Callers select these helpers after NPU distributed initialization.
Import torch_npu only when an NPU operation actually needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
import torch.distributed as dist

from sglang.srt.eplb.expert_location import get_eplb_group

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

# ACL format IDs, kept here so importing this module does not load torch_npu.
_FORMAT_NCHW = 0
_FORMAT_ND = 2
_FORMAT_FRACTAL_NZ = 29
_FORMAT_FRACTAL_NZ_C0_16 = 50
_FORMAT_FRACTAL_NZ_C0_32 = 51
_FORMAT_FRACTAL_NZ_C0_2 = 52
_FORMAT_FRACTAL_NZ_C0_4 = 53
_FORMAT_FRACTAL_NZ_C0_8 = 54
_BASE_FORMATS = (_FORMAT_NCHW, _FORMAT_ND)
_SUPPORTED_NZ_FORMATS = (
    _FORMAT_FRACTAL_NZ,
    _FORMAT_FRACTAL_NZ_C0_16,
    _FORMAT_FRACTAL_NZ_C0_32,
    _FORMAT_FRACTAL_NZ_C0_2,
    _FORMAT_FRACTAL_NZ_C0_4,
    _FORMAT_FRACTAL_NZ_C0_8,
)
# NZ expert weights retain the logical axes [experts, M, N].
_NZ_EXPERT_WEIGHT_NDIM = 3
# Per-peer warmup payload in bytes, independent of HCCL buffer capacity.
_EPLB_WARMUP_BYTES_PER_PEER = 4 << 20


def migrate_expert_weights(
    weights_by_layer: dict[int, list[torch.Tensor]],
    old_metadata: ExpertLocationMetadata,
    new_metadata: ExpertLocationMetadata,
    layer_ids: list[int],
    nnodes: int,
    rank: int,
    log_metrics: bool,
    update_layer_fn: Callable[..., None],
) -> dict[int, list[int]]:
    """Migrate expert storage in place without changing model tensor metadata."""
    changed_layer_ids, local_update_plan = _select_expert_weight_updates(
        old_metadata, new_metadata, layer_ids, rank
    )
    if not changed_layer_ids:
        return {}

    group = get_eplb_group()
    world_size = dist.get_world_size(group)
    _validate_migration_topology(world_size)
    adapter = _ExpertWeightAdapter(weights_by_layer, changed_layer_ids, rank)
    # Even an idle rank validates the changed chunk, but needs no scratch buffers.
    if not local_update_plan:
        return {}
    temp_buffers = adapter.create_temp_buffers()
    missing_experts_by_layer = {}

    for layer, old_placement, new_placement in local_update_plan:
        weights = weights_by_layer[layer]
        migration_weights = adapter.make_weight_views(weights)
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
        # Release this layer's migration views before preparing the next layer.
        del migration_weights
        if missing_experts:
            missing_experts_by_layer[layer] = missing_experts
    return missing_experts_by_layer


def _validate_migration_topology(group_size: int) -> None:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world_size = dist.get_world_size()
    # TP and WORLD must have the same members and rank numbering.
    if parallel.pp_size != 1 or not (
        parallel.tp_size == parallel.moe_ep_size == group_size == world_size
    ):
        raise ValueError(
            "NPU EPLB requires PP=1 and TP size = EP size = EPLB group size = "
            f"WORLD size; got PP={parallel.pp_size}, TP={parallel.tp_size}, "
            f"EP={parallel.moe_ep_size}, EPLB group={group_size}, WORLD={world_size}."
        )


def _select_expert_weight_updates(
    old_metadata: ExpertLocationMetadata,
    new_metadata: ExpertLocationMetadata,
    layer_ids: list[int],
    rank: int,
) -> tuple[list[int], list[tuple[int, list[int], list[int]]]]:
    """Return changed layers for validation and this rank's migration plan.

    Each plan entry contains the layer ID and its old/new expert placements.
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


def _rank_may_send_experts(
    old_placement: list[int], new_placement: list[int], local_slots: slice
) -> bool:
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
    """Metadata for byte views and scratch buffers shared by a layer chunk.

    ``shape`` describes the exported routed weights. ``total_experts`` counts
    all expert rows in the allocation, including any shared-expert suffix.
    ``storage_bytes`` includes NZ padding, so dividing it by ``total_experts``
    gives the physical size of one expert, not just its logical element count.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    total_experts: int
    storage_bytes: int
    npu_format: int
    needs_storage_alias: bool

    @property
    def expert_byte_shape(self) -> tuple[int, int]:
        """Return [routed experts, physical bytes per expert] for uint8 transport."""
        return self.shape[0], self.storage_bytes // self.total_experts


def _validated_full_storage_tensor(weight: torch.Tensor) -> torch.Tensor:
    """Validate dense expert rows and recover the backing storage's metadata."""
    import torch_npu

    if weight.numel() == 0 or weight.ndim == 0 or weight.storage_offset() != 0:
        raise ValueError("NPU EPLB requires a full expert tensor or its expert prefix.")
    # Expert rows must occupy independent contiguous blocks. Inner transposes
    # are allowed, but selecting/stepping elements within an expert is not.
    expert_elements = 1
    for stride, size in sorted(zip(weight.stride()[1:], weight.shape[1:])):
        if size > 1:
            if stride != expert_elements:
                raise ValueError("NPU EPLB requires dense storage within each expert.")
            expert_elements *= size
    if weight.stride(0) != expert_elements:
        raise ValueError("NPU EPLB requires an outermost, contiguous expert axis.")
    # Recover allocation metadata without copying data. For ND, dtype views and
    # reshapes can change the exported tensor's dtype and expert axis. NZ still
    # needs the original expert axis to account for physical padding.
    return torch_npu._C._tensor_construct_from_storage(weight.untyped_storage())


def _expert_storage_layout(weight: torch.Tensor) -> _ExpertStorageLayout:
    """Check that raw storage can be split into equal, complete expert blocks."""
    import torch_npu

    full_storage_tensor = _validated_full_storage_tensor(weight)
    storage_elements = torch_npu.get_storage_size(full_storage_tensor)
    npu_format = torch_npu.get_npu_format(full_storage_tensor)
    # Check the actual storage format: requesting NZ (29) can produce a C0
    # variant, such as 50 for packed FP4. Expert-block checks below still apply.
    if npu_format not in _BASE_FORMATS and npu_format not in _SUPPORTED_NZ_FORMATS:
        raise ValueError(f"NPU EPLB does not support storage format {npu_format}.")
    # The byte view must stay within the original allocation, including NZ padding.
    storage_bytes = storage_elements * full_storage_tensor.element_size()
    allocated_bytes = full_storage_tensor.untyped_storage().nbytes()
    if (
        storage_elements <= 0
        or storage_elements < full_storage_tensor.numel()
        or storage_bytes > allocated_bytes
    ):
        raise ValueError(
            f"NPU EPLB has invalid storage capacity: {storage_elements} elements "
            f"for shape {tuple(full_storage_tensor.shape)} and "
            f"{allocated_bytes} allocated bytes."
        )
    if npu_format in _BASE_FORMATS:
        if storage_elements != full_storage_tensor.numel():
            raise ValueError("NPU EPLB requires base-format storage without padding.")
        # ND packing/reshaping may retain a different allocation dtype or shape.
        # Derive expert boundaries from the exported tensor, in bytes.
        bytes_per_expert = weight.stride(0) * weight.element_size()
        if storage_bytes % bytes_per_expert:
            raise ValueError("NPU EPLB requires complete expert blocks in ND storage.")
        total_experts = storage_bytes // bytes_per_expert
    else:
        total_experts = _validate_nz_expert_blocks(
            weight, full_storage_tensor, storage_elements
        )
    if weight.shape[0] > total_experts:
        raise ValueError("NPU EPLB expert prefix exceeds the backing storage.")
    return _ExpertStorageLayout(
        shape=tuple(weight.shape),
        dtype=weight.dtype,
        device=weight.device,
        total_experts=total_experts,
        storage_bytes=storage_bytes,
        npu_format=npu_format,
        needs_storage_alias=(
            npu_format not in _BASE_FORMATS or not weight.is_contiguous()
        ),
    )


def _validate_nz_expert_blocks(
    weight: torch.Tensor, full_storage_tensor: torch.Tensor, storage_elements: int
) -> int:
    """Validate NZ expert boundaries and return the full storage's expert count."""
    if (
        full_storage_tensor.ndim != _NZ_EXPERT_WEIGHT_NDIM
        or weight.ndim != _NZ_EXPERT_WEIGHT_NDIM
    ):
        raise ValueError("NPU EPLB requires 3D batched NZ expert weights.")
    if (
        weight.element_size() != full_storage_tensor.element_size()
        or weight.stride(0) != full_storage_tensor.stride(0)
        or full_storage_tensor.numel()
        != full_storage_tensor.shape[0] * weight.stride(0)
    ):
        raise ValueError("NPU EPLB requires the original NZ expert blocks.")
    total_experts = full_storage_tensor.shape[0]
    if total_experts <= 0 or storage_elements % total_experts:
        raise ValueError("NPU EPLB requires equal physical NZ expert blocks.")
    return total_experts


def _expert_storage_view(
    weight: torch.Tensor, layout: _ExpertStorageLayout
) -> torch.Tensor:
    """Return a writable byte alias, preserving the original tensor's descriptor."""
    if not layout.needs_storage_alias:
        return weight.view(torch.uint8).view(layout.expert_byte_shape)

    import torch_npu

    # A separate storage descriptor presents the same allocation as ND bytes.
    # Reusing weight.untyped_storage() would also reuse its NZ descriptor.
    storage = torch_npu._C._construct_storage_from_data_pointer(
        weight.data_ptr(),
        weight.device,
        layout.storage_bytes,
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
    view._eplb_weight_owner = weight
    weight.record_stream(torch.npu.current_stream())
    return view


class _ExpertWeightAdapter:
    """Expose original expert bytes and allocate the updater's scratch buffers.

    Returned views share storage with the model: writing a view changes its
    original weight. Independent ND descriptors preserve NZ padding and packed
    values. Only routed expert rows are exposed; shared experts retain their
    original storage.
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

    def make_weight_views(self, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return writable aliases of the routed experts, without copying weights."""
        return [
            _expert_storage_view(weight, layout)
            for weight, layout in zip(weights, self._layouts, strict=True)
        ]


def prewarm_eplb_communication() -> None:
    """Warm TP transports for enabled, non-elastic NPU EPLB before KV sizing."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp = parallel.tp_group
    group = tp.device_group
    _validate_migration_topology(dist.get_world_size(group))
    if tp.world_size == 1:
        return
    send = torch.empty(_EPLB_WARMUP_BYTES_PER_PEER, dtype=torch.uint8, device="npu")
    recv = torch.empty(
        (tp.world_size - 1, send.numel()), dtype=send.dtype, device=send.device
    )
    ops = []
    for slot, peer in enumerate(p for p in tp.ranks if p != tp.rank):
        ops.extend(
            [
                dist.P2POp(dist.isend, send, peer, group=group),
                dist.P2POp(dist.irecv, recv[slot], peer, group=group),
            ]
        )
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    dist.all_reduce(send.view(torch.int32), group=group)
    torch.npu.synchronize()
    del ops, send, recv
    torch.npu.empty_cache()
