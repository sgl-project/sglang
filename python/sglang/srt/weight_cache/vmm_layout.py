# SPDX-License-Identifier: Apache-2.0
"""Packing layout for the ``vmm_fd`` weight cache transport.

The daemon copies its final tensors into a few large CUDA VMM allocations and
the engine maps those allocations into its own address space. Every distinct
storage becomes one *slot*: a byte range at an aligned offset inside one
allocation. A tensor is described relative to its slot the same way torch
describes it relative to its storage (``storage_offset``, ``shape``,
``stride``), so tensors that shared a storage in the daemon (tied weights,
transposes, slices, dtype views) share one storage again in the engine.

Nothing here calls CUDA: planning, validation and rebuilding tensors only use
torch metadata, so this module is unit-tested on CPU.
"""

from __future__ import annotations

import bisect
import dataclasses
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

# The CUDA caching allocator hands out blocks aligned to 512 bytes. Slots use
# the same alignment, so a packed tensor keeps the alignment it was loaded with.
SLOT_ALIGNMENT = 512


@dataclasses.dataclass
class VmmSlot:
    """One packed storage: ``nbytes`` at ``offset`` in allocation ``alloc_index``."""

    alloc_index: int
    offset: int
    nbytes: int
    # (name of a tensor on the source storage, byte offset of that storage in
    # the slot, storage nbytes). More than one source only when distinct
    # storages overlapped in the daemon.
    sources: List[Tuple[str, int, int]]
    # Every tensor that views this slot.
    names: List[str]


@dataclasses.dataclass
class VmmPackPlan:
    alloc_sizes: List[int]
    slots: List[VmmSlot]
    # Wire layout: one dict per tensor name, see plan_vmm_pack().
    entries: Dict[str, Dict[str, Any]]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None) if isinstance(name, str) else None
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {name!r}")
    return dtype


def view_nbytes(
    shape: Sequence[int], stride: Sequence[int], storage_offset: int, itemsize: int
) -> int:
    """Bytes a view needs from the start of its storage (0 for an empty view,
    matching torch's own bounds check)."""
    if any(size == 0 for size in shape):
        return 0
    extent = 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    return (storage_offset + extent) * itemsize


def plan_vmm_pack(
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
    *,
    device: torch.device,
    granularity: int,
    max_alloc_bytes: int,
) -> VmmPackPlan:
    """Assign the storage behind every tensor in ``state_tensors`` to a slot.

    Storages whose byte ranges overlap share one slot, so aliasing between
    distinct storages survives too. Slots are placed in first-use order at
    ``SLOT_ALIGNMENT``-aligned offsets. An allocation is closed when the next
    slot would push it past ``max_alloc_bytes``; a larger storage gets an
    allocation of its own. Allocation sizes are rounded up to ``granularity``.

    Each entry holds ``alloc_index``, ``offset`` (slot start, bytes),
    ``storage_nbytes`` (slot size), ``storage_offset`` (elements, relative to
    the slot), ``shape``, ``stride``, ``dtype`` and ``is_param``. The plan
    keeps tensor names, not tensors, so it does not keep any original alive.
    """
    if granularity <= 0 or granularity % SLOT_ALIGNMENT:
        raise ValueError(
            f"granularity {granularity} must be a positive multiple of {SLOT_ALIGNMENT}"
        )
    if max_alloc_bytes <= 0:
        raise ValueError(f"max_alloc_bytes must be positive, got {max_alloc_bytes}")

    # Storage span (start pointer, nbytes) behind every tensor. Empty storages
    # get a zero-byte slot each.
    tensor_spans: Dict[str, Tuple[int, int]] = {}
    span_first_use: Dict[int, int] = {}  # start -> index of the first tensor
    # start -> (nbytes, name): the largest storage at that start is the one
    # copied, as it covers every smaller storage that starts there too.
    span_source: Dict[int, Tuple[int, str]] = {}
    empty_names: List[Tuple[int, str]] = []
    for index, (name, (tensor, _)) in enumerate(state_tensors.items()):
        if tensor.device != device:
            raise ValueError(
                f"tensor {name!r} is on {tensor.device}, expected {device}"
            )
        storage = tensor.untyped_storage()
        start, nbytes = storage.data_ptr(), storage.nbytes()
        tensor_spans[name] = (start, nbytes)
        if nbytes == 0:
            empty_names.append((index, name))
            continue
        span_first_use.setdefault(start, index)
        if nbytes > span_source.get(start, (0, ""))[0]:
            span_source[start] = (nbytes, name)

    # Merge spans that overlap (spans that only touch stay separate).
    span_starts = sorted(span_source)
    groups: List[List[int]] = []  # [start, end, first_use]
    for start in span_starts:
        end = start + span_source[start][0]
        if groups and start < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2] = min(groups[-1][2], span_first_use[start])
        else:
            groups.append([start, end, span_first_use[start]])
    group_starts = [group[0] for group in groups]

    group_slots = [
        VmmSlot(alloc_index=-1, offset=-1, nbytes=end - start, sources=[], names=[])
        for start, end, _ in groups
    ]
    for start in span_starts:
        group = bisect.bisect_right(group_starts, start) - 1
        nbytes, name = span_source[start]
        group_slots[group].sources.append((name, start - group_starts[group], nbytes))

    tensor_slot: Dict[str, Tuple[VmmSlot, int]] = {}  # name -> (slot, byte offset)
    for name, (start, nbytes) in tensor_spans.items():
        if nbytes == 0:
            continue
        group = bisect.bisect_right(group_starts, start) - 1
        group_slots[group].names.append(name)
        tensor_slot[name] = (group_slots[group], start - group_starts[group])

    ordered: List[Tuple[int, VmmSlot]] = [
        (groups[i][2], slot) for i, slot in enumerate(group_slots)
    ]
    for index, name in empty_names:
        slot = VmmSlot(alloc_index=-1, offset=-1, nbytes=0, sources=[], names=[name])
        ordered.append((index, slot))
        tensor_slot[name] = (slot, 0)
    ordered.sort(key=lambda item: item[0])
    slots = [slot for _, slot in ordered]

    # Lay the slots out back to back, opening a new allocation when needed.
    alloc_used: List[int] = []
    for slot in slots:
        offset = _align_up(alloc_used[-1], SLOT_ALIGNMENT) if alloc_used else 0
        if not alloc_used or (
            alloc_used[-1] > 0 and offset + slot.nbytes > max_alloc_bytes
        ):
            alloc_used.append(0)
            offset = 0
        slot.alloc_index = len(alloc_used) - 1
        slot.offset = offset
        alloc_used[-1] = offset + slot.nbytes
    alloc_sizes = [_align_up(max(used, 1), granularity) for used in alloc_used]

    entries: Dict[str, Dict[str, Any]] = {}
    for name, (tensor, is_param) in state_tensors.items():
        slot, byte_offset = tensor_slot[name]
        itemsize = tensor.element_size()
        if byte_offset % itemsize:
            raise ValueError(
                f"tensor {name!r} starts {byte_offset} bytes into a storage it "
                f"overlaps with, which is not a multiple of its {itemsize}-byte "
                f"element size"
            )
        entries[name] = {
            "alloc_index": slot.alloc_index,
            "offset": slot.offset,
            "storage_nbytes": slot.nbytes,
            "storage_offset": byte_offset // itemsize + tensor.storage_offset(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "is_param": bool(is_param),
        }
    return VmmPackPlan(alloc_sizes=alloc_sizes, slots=slots, entries=entries)


def validate_vmm_layout(
    entries: Mapping[str, Mapping[str, Any]], alloc_sizes: Any
) -> None:
    """Check a received layout before any memory is mapped.

    Every slot must lie inside its allocation, every tensor inside its slot,
    and two different slots must not overlap. Raises ValueError otherwise.
    """
    if not isinstance(alloc_sizes, (list, tuple)) or not all(
        _is_int(size) and size > 0 for size in alloc_sizes
    ):
        raise ValueError(f"invalid vmm_fd allocation sizes: {alloc_sizes!r}")

    slots = set()
    for name, entry in entries.items():
        try:
            alloc_index = entry["alloc_index"]
            offset = entry["offset"]
            nbytes = entry["storage_nbytes"]
            storage_offset = entry["storage_offset"]
            shape, stride = entry["shape"], entry["stride"]
            dtype = parse_dtype(entry["dtype"])
            is_param = entry["is_param"]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed vmm_fd entry {name!r}: {e!r}") from e

        if not (
            all(_is_int(v) for v in (alloc_index, offset, nbytes, storage_offset))
            and 0 <= alloc_index < len(alloc_sizes)
            and offset >= 0
            and nbytes >= 0
            and storage_offset >= 0
            and offset % SLOT_ALIGNMENT == 0
            and offset + nbytes <= alloc_sizes[alloc_index]
        ):
            raise ValueError(
                f"vmm_fd entry {name!r} does not fit its allocation: "
                f"alloc_index={alloc_index!r} offset={offset!r} "
                f"storage_nbytes={nbytes!r} storage_offset={storage_offset!r} "
                f"allocation sizes={list(alloc_sizes)}"
            )
        if not (
            isinstance(shape, (list, tuple))
            and isinstance(stride, (list, tuple))
            and len(shape) == len(stride)
            and all(_is_int(v) and v >= 0 for v in (*shape, *stride))
            and isinstance(is_param, bool)
        ):
            raise ValueError(
                f"vmm_fd entry {name!r} has a malformed view: shape={shape!r} "
                f"stride={stride!r} is_param={is_param!r}"
            )
        needed = view_nbytes(shape, stride, storage_offset, dtype.itemsize)
        if needed > nbytes:
            raise ValueError(
                f"vmm_fd entry {name!r} needs {needed} bytes but its storage "
                f"has {nbytes}"
            )
        slots.add((alloc_index, offset, nbytes))

    by_alloc: Dict[int, List[Tuple[int, int]]] = {}
    for alloc_index, offset, nbytes in slots:
        if nbytes:
            by_alloc.setdefault(alloc_index, []).append((offset, nbytes))
    for alloc_index, spans in by_alloc.items():
        spans.sort()
        for (offset, nbytes), (next_offset, _) in zip(spans, spans[1:]):
            if offset + nbytes > next_offset:
                raise ValueError(
                    f"vmm_fd slots overlap in allocation {alloc_index}: "
                    f"[{offset}, {offset + nbytes}) and one starting at "
                    f"{next_offset}"
                )


def view_from_entry(
    storage: torch.UntypedStorage, entry: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    """Build the tensor ``entry`` describes on top of its slot's ``storage``."""
    return torch.empty(0, dtype=parse_dtype(entry["dtype"]), device=device).set_(
        storage, entry["storage_offset"], entry["shape"], entry["stride"]
    )


def pack_slot(
    slot: VmmSlot,
    entries: Mapping[str, Mapping[str, Any]],
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
    dst: torch.Tensor,
) -> None:
    """Copy ``slot``'s source storages into ``dst`` and repoint its tensors.

    ``dst`` is a uint8 tensor whose storage is exactly the slot. Once the
    bytes are copied, every tensor of the slot is repointed in place with
    ``tensor.data = ...``; when nothing else references the original storage
    any more, torch frees it.
    """
    if (
        dst.dtype != torch.uint8
        or dst.storage_offset() != 0
        or dst.untyped_storage().nbytes() != slot.nbytes
    ):
        raise ValueError("dst must be a uint8 tensor covering exactly the slot")
    for name, rel, nbytes in slot.sources:
        source = state_tensors[name][0].untyped_storage()
        src = torch.empty(0, dtype=torch.uint8, device=source.device).set_(source)
        dst[rel : rel + nbytes].copy_(src)
    storage = dst.untyped_storage()
    for name in slot.names:
        state_tensors[name][0].data = view_from_entry(
            storage, entries[name], dst.device
        )
