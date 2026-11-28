# SPDX-License-Identifier: Apache-2.0
"""Packing layout for the ``vmm_fd`` weight cache transport.

The daemon copies its final tensors into a few large VMM allocations -- CUDA
``cuMemCreate`` or Level Zero ``zePhysicalMemCreate`` -- and the engine maps
those allocations into its own address space. Every distinct storage becomes one
*slot*: a byte range at an aligned offset inside one allocation. A tensor is
described relative to its slot the same way torch describes it relative to its
storage (``storage_offset``, ``shape``, ``stride``), so tensors that shared a
storage in the daemon (tied weights, transposes, slices, dtype views) share one
storage again in the engine.

Planning and validation use torch metadata, so this module is device-neutral and
shared by every platform's ``vmm_fd`` backend. Slots retain their source
storages until packing copies and releases them. These paths are also tested on
CPU.
"""

from __future__ import annotations

import bisect
import dataclasses
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

# The CUDA caching allocator hands out blocks aligned to 512 bytes. Slots use
# the same alignment, so a packed tensor keeps the alignment it was loaded with.
# Level Zero's device page size is larger (64 KiB or 2 MiB on Arc Pro B60), but
# that bounds whole allocations, not the slots inside one.
SLOT_ALIGNMENT = 512

# Wire format of the layout a daemon sends with a vmm_fd fetch_state response.
# Bump when the meaning of an entry field or of ``allocation_sizes`` changes, so
# an engine refuses a layout it would otherwise misread rather than mapping it.
# Version 1 is the format #46953 shipped; a daemon that predates the field sends
# nothing and is read as version 1.
LAYOUT_VERSION = 1
SUPPORTED_LAYOUT_VERSIONS = (1,)

# Entry and allocation sizes end up in a ``size_t`` driver argument, where
# ctypes truncates an out-of-range value modulo 2**64 without complaining. Bound
# every wire integer well below that, at the signed 64-bit range torch itself
# uses for sizes and strides.
MAX_WIRE_INT = (1 << 63) - 1


@dataclasses.dataclass
class VmmSlot:
    """One packed storage: ``nbytes`` at ``offset`` in allocation ``alloc_index``."""

    alloc_index: int
    offset: int
    nbytes: int
    # (source storage, byte offset in the slot). Retain each source until this
    # slot is copied; multiple sources are needed only for overlapping storages.
    sources: List[Tuple[torch.UntypedStorage, int]]


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
    owns the source storages until pack_slot() copies and releases each slot.
    It does not retain model tensors or need to rebind them.
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
        VmmSlot(alloc_index=-1, offset=-1, nbytes=end - start, sources=[])
        for start, end, _ in groups
    ]
    for start in span_starts:
        group = bisect.bisect_right(group_starts, start) - 1
        _, name = span_source[start]
        group_slots[group].sources.append(
            (state_tensors[name][0].untyped_storage(), start - group_starts[group])
        )

    tensor_slot: Dict[str, Tuple[VmmSlot, int]] = {}  # name -> (slot, byte offset)
    for name, (start, nbytes) in tensor_spans.items():
        if nbytes == 0:
            continue
        group = bisect.bisect_right(group_starts, start) - 1
        tensor_slot[name] = (group_slots[group], start - group_starts[group])

    ordered: List[Tuple[int, VmmSlot]] = [
        (groups[i][2], slot) for i, slot in enumerate(group_slots)
    ]
    for index, name in empty_names:
        slot = VmmSlot(alloc_index=-1, offset=-1, nbytes=0, sources=[])
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


def check_layout_version(version: Any) -> int:
    """Validate the layout version a daemon reported; returns it.

    ``None`` means the daemon predates the field, which is version 1.
    """
    if version is None:
        return 1
    # _is_int first: 1.0 and True both compare equal to version 1.
    if not _is_int(version) or version not in SUPPORTED_LAYOUT_VERSIONS:
        raise ValueError(
            f"vmm_fd layout version {version!r} is not one this engine reads "
            f"({', '.join(str(v) for v in SUPPORTED_LAYOUT_VERSIONS)}). The "
            f"daemon and the engine run different sglang versions; restart the "
            f"daemon from this build."
        )
    return int(version)


def check_layout_device(device_type: Any, local_device_type: str) -> None:
    """Refuse a layout packed for another device type.

    ``vmm_fd`` is one wire contract with a driver-specific implementation on each
    side, and a handle from one driver means nothing to another. ``None`` means
    the daemon predates the field, where CUDA was the only implementation.
    """
    if device_type is None:
        return
    if device_type != local_device_type:
        raise ValueError(
            f"vmm_fd layout was packed on a {device_type!r} device but this "
            f"engine runs on {local_device_type!r}; its allocation handles "
            f"cannot be imported here."
        )


def _validate_entry_names(entries: Any) -> None:
    """The entries must be a mapping of distinct tensor names.

    A pickled payload can carry any object here, including a sequence of pairs
    whose names repeat, which would silently drop a tensor when read as a dict.
    """
    if not isinstance(entries, Mapping):
        raise ValueError(
            f"vmm_fd entries must be a mapping of tensor name to layout, got "
            f"{type(entries).__name__}"
        )
    names = list(entries)
    if not all(isinstance(name, str) for name in names):
        raise ValueError("vmm_fd entry names must all be strings")
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"vmm_fd entries repeat the tensor name(s) {duplicates}")


def validate_vmm_layout(
    entries: Mapping[str, Mapping[str, Any]], alloc_sizes: Any
) -> None:
    """Check a received layout before any memory is mapped.

    Every slot must lie inside its allocation, every tensor inside its slot,
    and two different slots must not overlap. Page alignment of the allocation
    sizes is a driver property, so each platform's backend checks that itself.
    Raises ValueError otherwise.
    """
    _validate_entry_names(entries)
    if not isinstance(alloc_sizes, (list, tuple)) or not all(
        _is_int(size) and 0 < size <= MAX_WIRE_INT for size in alloc_sizes
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
            # An empty view needs no bytes, so nothing below would bound a
            # storage_offset that overflows the driver's size_t.
            and 0 <= storage_offset <= MAX_WIRE_INT
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
            and all(_is_int(v) and 0 <= v <= MAX_WIRE_INT for v in (*shape, *stride))
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
    dst: torch.Tensor,
) -> None:
    """Copy ``slot``'s retained source storages into ``dst``, then release them.

    ``dst`` is a uint8 tensor whose storage is exactly the slot. Once the
    bytes are copied, discard the source storage references for this slot. No
    daemon model needs rebinding; engines rebuild views from the wire metadata.
    """
    if (
        dst.dtype != torch.uint8
        or dst.storage_offset() != 0
        or dst.untyped_storage().nbytes() != slot.nbytes
    ):
        raise ValueError("dst must be a uint8 tensor covering exactly the slot")
    for source, rel in slot.sources:
        src = torch.empty(0, dtype=torch.uint8, device=source.device).set_(source)
        dst[rel : rel + source.nbytes()].copy_(src)
    slot.sources.clear()
