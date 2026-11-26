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

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
import torch.distributed as dist
from sglang.srt.hardware_backend.npu.eplb.transport import _validate_migration_topology

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata
    from sglang.srt.hardware_backend.npu.eplb.coordinator import WeightWriteback

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
_MAX_PACKED_FIELD_BYTES = 16 * 1024


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

    from sglang.srt.eplb.expert_location import get_eplb_group

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
    weights_by_layer: dict[int, list[torch.Tensor]],
    layer_ids: list[int],
    rank: int,
    *,
    compute_format_digest: bool,
) -> tuple[list[_ExpertStorageLayout], bytes | None]:
    """Validate the chunk and optionally digest the wire format of each layer."""
    sample_layer = layer_ids[0]
    num_weights = len(weights_by_layer[sample_layer])
    if not num_weights:
        raise ValueError(
            f"[NPU EPLB] rank={rank} layer={sample_layer}: no expert weights"
        )
    layouts = []
    format_digest = hashlib.sha256() if compute_format_digest else None
    for layer_index, layer in enumerate(layer_ids):
        weights = weights_by_layer[layer]
        if len(weights) != num_weights:
            raise ValueError(
                f"[NPU EPLB] rank={rank} layer={layer}: expected {num_weights} "
                f"weight tensors as in layer {sample_layer}, got {len(weights)}"
            )
        layer_signature = []
        for weight_index, weight in enumerate(weights):
            try:
                layout, backing = _inspect_expert_storage(weight)
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
            if format_digest is not None:
                layer_signature.append(
                    _transfer_field_signature(weight, layout, backing)
                )
        if format_digest is not None:
            format_digest.update(repr(tuple(layer_signature)).encode("ascii"))
    return layouts, format_digest.digest() if format_digest is not None else None


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


def _validate_routed_expert_view(weight: torch.Tensor) -> None:
    """Require a prefix of complete, densely stored expert rows."""
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


def _inspect_expert_storage(
    weight: torch.Tensor,
) -> tuple[_ExpertStorageLayout, torch.Tensor]:
    """Check that raw storage can be split into equal, complete expert blocks."""
    import torch_npu

    _validate_routed_expert_view(weight)
    # Recover the allocation descriptor without copying. The routed view can
    # differ from it in shape or dtype.
    backing = torch_npu._C._tensor_construct_from_storage(weight.untyped_storage())
    physical_elements = torch_npu.get_storage_size(backing)
    npu_format = torch_npu.get_npu_format(backing)
    is_base_format = npu_format in _BASE_FORMATS
    # Check the actual storage format: requesting NZ (29) can produce a C0
    # variant, such as 50 for packed FP4. Expert-block checks below still apply.
    if not is_base_format and npu_format not in _SUPPORTED_NZ_FORMATS:
        raise ValueError(f"NPU EPLB does not support storage format {npu_format}.")
    # The byte view must stay within the original allocation, including NZ padding.
    physical_bytes = physical_elements * backing.element_size()
    allocated_bytes = backing.untyped_storage().nbytes()
    logical_elements = backing.numel()
    if (
        physical_elements <= 0
        or physical_elements < logical_elements
        or physical_bytes > allocated_bytes
    ):
        raise ValueError(
            f"NPU EPLB has invalid storage capacity: {physical_elements} elements "
            f"for shape {tuple(backing.shape)} and "
            f"{allocated_bytes} allocated bytes."
        )
    if is_base_format:
        if physical_elements != logical_elements:
            raise ValueError("NPU EPLB requires base-format storage without padding.")
        # ND packing/reshaping may retain a different allocation dtype or shape.
        # Derive expert boundaries from the exported tensor, in bytes.
        bytes_per_expert = weight.stride(0) * weight.element_size()
        if physical_bytes % bytes_per_expert:
            raise ValueError("NPU EPLB requires complete expert blocks in ND storage.")
        total_experts = physical_bytes // bytes_per_expert
    else:
        total_experts = _validate_nz_expert_blocks(weight, backing, physical_elements)
    if weight.shape[0] > total_experts:
        raise ValueError("NPU EPLB expert prefix exceeds the backing storage.")
    layout = _ExpertStorageLayout(
        shape=tuple(weight.shape),
        dtype=weight.dtype,
        device=weight.device,
        total_experts=total_experts,
        storage_bytes=physical_bytes,
        npu_format=npu_format,
        needs_storage_alias=not is_base_format or not weight.is_contiguous(),
    )
    return layout, backing


def _transfer_field_signature(
    weight: torch.Tensor, layout: _ExpertStorageLayout, backing: torch.Tensor
) -> tuple:
    """Describe how a rank interprets an expert's bytes, independent of device ID."""
    nz_backing_row = None
    if layout.npu_format not in _BASE_FORMATS:
        nz_backing_row = (backing.dtype, tuple(backing.shape[1:]))
    return (
        layout.shape,
        weight.stride(),
        layout.dtype,
        layout.npu_format,
        layout.expert_byte_shape,
        nz_backing_row,
    )


def _validate_nz_expert_blocks(
    weight: torch.Tensor, backing: torch.Tensor, physical_elements: int
) -> int:
    """Validate NZ expert boundaries and return the full storage's expert count."""
    if backing.ndim != _NZ_EXPERT_WEIGHT_NDIM or weight.ndim != _NZ_EXPERT_WEIGHT_NDIM:
        raise ValueError("NPU EPLB requires 3D batched NZ expert weights.")
    elements_per_expert = weight.stride(0)
    total_experts = backing.shape[0]
    if (
        weight.element_size() != backing.element_size()
        or elements_per_expert != backing.stride(0)
        or backing.numel() != total_experts * elements_per_expert
    ):
        raise ValueError("NPU EPLB requires the original NZ expert blocks.")
    if total_experts <= 0 or physical_elements % total_experts:
        raise ValueError("NPU EPLB requires equal physical NZ expert blocks.")
    return total_experts


def _expert_storage_view(
    weight: torch.Tensor, layout: _ExpertStorageLayout
) -> torch.Tensor:
    """Return a writable byte alias, preserving the original tensor's descriptor."""
    byte_shape = layout.expert_byte_shape
    if not layout.needs_storage_alias:
        return weight.view(torch.uint8).view(byte_shape)

    import torch_npu

    # A separate storage descriptor presents the same allocation as ND bytes.
    # Reusing weight.untyped_storage() would also reuse its NZ descriptor.
    storage = torch_npu._C._construct_storage_from_data_pointer(
        weight.data_ptr(),
        weight.device,
        layout.storage_bytes,
    )
    _, bytes_per_expert = byte_shape
    view = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        {
            "dtype": torch.uint8,
            "size": byte_shape,
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
    The same one-layer scratch buffers can stage sources before writeback for
    each layer in the chunk.
    """

    def __init__(
        self,
        weights_by_layer: dict[int, list[torch.Tensor]],
        layer_ids: list[int],
        rank: int,
        *,
        compute_format_digest: bool = False,
    ):
        self._layouts, self.weight_format_digest = _validate_migration_weights(
            weights_by_layer,
            layer_ids,
            rank,
            compute_format_digest=compute_format_digest,
        )

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


@dataclass(frozen=True)
class _PackedExpertFields:
    """Send narrow fields together and expose their original views for writeback."""

    packed_indices: tuple[int, ...]
    direct_indices: tuple[int, ...]
    send_buffer: torch.Tensor
    receive_buffers: list[torch.Tensor]

    @classmethod
    def create(cls, staging: list[torch.Tensor]) -> _PackedExpertFields | None:
        # The weight adapter exposes contiguous [expert, bytes] tensors.
        packed_indices = tuple(
            i
            for i, field in enumerate(staging)
            if 0 < field.size(1) <= _MAX_PACKED_FIELD_BYTES
        )
        if len(packed_indices) < 2:
            return None
        direct_indices = tuple(
            i for i in range(len(staging)) if i not in packed_indices
        )
        widths = [staging[i].size(1) for i in packed_indices]
        receive_buffer = staging[0].new_empty((staging[0].size(0), sum(widths)))
        for i, view in zip(
            packed_indices, receive_buffer.split(widths, dim=1), strict=True
        ):
            staging[i] = view
        return cls(
            packed_indices=packed_indices,
            direct_indices=direct_indices,
            send_buffer=torch.empty_like(receive_buffer),
            receive_buffers=[staging[i] for i in direct_indices] + [receive_buffer],
        )

    def pack(self, weights: list[torch.Tensor]) -> None:
        torch.cat(
            [weights[i] for i in self.packed_indices], dim=1, out=self.send_buffer
        )

    def pack_for_local_copies(
        self,
        weights: list[torch.Tensor],
        old_local_layout: list[int],
        new_local_layout: list[int],
    ) -> bool:
        """Pack early if the updater will copy a resident expert before P2P."""
        resident_experts = set(old_local_layout)
        if any(
            old != new and new in resident_experts
            for old, new in zip(old_local_layout, new_local_layout, strict=True)
        ):
            self.pack(weights)
            return True
        return False


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
