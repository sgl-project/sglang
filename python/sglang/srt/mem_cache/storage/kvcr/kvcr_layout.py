# SPDX-License-Identifier: Apache-2.0
"""Map SGLang physical device pools onto KVCR objects, subpools, and regions.

One KVCR object holds one page of one physical pool. Its ordered descriptor
list covers that page's component/layer spans, each span living in a named
KVCR subpool sized to exactly that span. The mapping is derived from
``DevicePoolEntry.buffer_meta`` so the linker never reconstructs pool geometry
from tensor shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.storage.kvcr.router_hint import compatibility_digest

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
        DevicePoolEntry,
        DevicePoolGroup,
    )

# NIXL segment type per torch device type.
_MEM_TYPE_BY_DEVICE = {"cuda": "VRAM", "cpu": "DRAM"}


def nixl_mem_type(device: torch.device) -> str:
    mem_type = _MEM_TYPE_BY_DEVICE.get(device.type)
    if mem_type is None:
        raise ValueError(
            f"KVCR linker cannot address {device.type!r} memory; supported "
            f"device types: {sorted(_MEM_TYPE_BY_DEVICE)}."
        )
    return mem_type


def nixl_device_id(device: torch.device) -> int:
    return device.index if device.index is not None else 0


class PoolObjectLayout(msgspec.Struct, frozen=True, kw_only=True):
    """How one physical pool's page maps to a KVCR object."""

    pool: str
    mem_type: str
    device_id: int
    # Subpool name per span, in descriptor order.
    subpools: tuple[str, ...]
    # (base_ptr, row_stride_bytes, span_bytes) per span.
    spans: tuple[tuple[int, int, int], ...]

    @property
    def span_sizes(self) -> tuple[int, ...]:
        return tuple(size for _, _, size in self.spans)

    @property
    def object_bytes(self) -> int:
        return sum(self.span_sizes)

    @property
    def expected_layout(self) -> list[str]:
        return list(self.subpools)


class CapacityPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved per-rank KVCR DRAM allocation."""

    budget_bytes: int
    page_capacity: int
    # (subpool name, span bytes) in pool_layouts order.
    pool_layouts: tuple[tuple[str, int], ...]
    # Subpool name -> byte length of its region.
    subpool_bytes: dict[str, int]
    total_bytes: int

    @property
    def unused_bytes(self) -> int:
        return self.budget_bytes - self.total_bytes


def build_pool_object_layouts(
    pool_group: DevicePoolGroup,
) -> dict[str, PoolObjectLayout]:
    """One object layout per physical pool entry, in group order."""
    layouts: dict[str, PoolObjectLayout] = {}
    for entry in pool_group.entries:
        layouts[str(entry.name)] = _layout_for_entry(entry)
    return layouts


def _layout_for_entry(entry: DevicePoolEntry) -> PoolObjectLayout:
    if not entry.packed:
        raise ValueError(
            f"KVCR linker requires packed device pools; pool {entry.name} "
            "stores components as separate objects."
        )
    devices = {buffer.device for buffer in entry.kv_buffer}
    if len(devices) != 1:
        raise ValueError(
            f"KVCR linker pool {entry.name} spans several devices: {devices}."
        )
    device = devices.pop()
    spans: list[tuple[int, int, int]] = []
    subpools: list[str] = []
    for component_index, component in enumerate(entry.buffer_meta):
        for buffer_index, (base_ptr, row_stride, size) in enumerate(component):
            if size <= 0:
                raise ValueError(
                    f"KVCR linker pool {entry.name} has an empty span at "
                    f"component {component_index}, buffer {buffer_index}."
                )
            spans.append((int(base_ptr), int(row_stride), int(size)))
            subpools.append(f"{entry.name}/{component_index}.{buffer_index}")
    return PoolObjectLayout(
        pool=str(entry.name),
        mem_type=nixl_mem_type(device),
        device_id=nixl_device_id(device),
        subpools=tuple(subpools),
        spans=tuple(spans),
    )


def plan_capacity(
    layouts: Mapping[str, PoolObjectLayout], budget_bytes: int
) -> CapacityPlan:
    """Give every physical pool the same page capacity within the budget.

    Every restorable boundary needs a page in every required pool, so a
    smaller pool would cap all objects; sizing each subpool as
    ``page_capacity * span_bytes`` keeps capacities compatible. The remainder
    below one more page per pool is the reported unused tail.
    """
    if budget_bytes <= 0:
        raise ValueError("KVCR linker DRAM budget must be positive.")
    bytes_per_page = sum(layout.object_bytes for layout in layouts.values())
    if bytes_per_page <= 0:
        raise ValueError("KVCR linker pool layouts describe no bytes.")
    page_capacity = budget_bytes // bytes_per_page
    if page_capacity <= 0:
        raise ValueError(
            f"KVCR linker DRAM budget of {budget_bytes} bytes per rank holds no "
            f"complete page ({bytes_per_page} bytes across all pools). Raise "
            "local_dram_bytes_per_worker."
        )
    pool_layouts: list[tuple[str, int]] = []
    subpool_bytes: dict[str, int] = {}
    for layout in layouts.values():
        for subpool, size in zip(layout.subpools, layout.span_sizes):
            pool_layouts.append((subpool, size))
            subpool_bytes[subpool] = size * page_capacity
    return CapacityPlan(
        budget_bytes=budget_bytes,
        page_capacity=page_capacity,
        pool_layouts=tuple(pool_layouts),
        subpool_bytes=subpool_bytes,
        total_bytes=sum(subpool_bytes.values()),
    )


def carve_local_dram(
    plan: CapacityPlan, buffer: torch.Tensor
) -> list[tuple[str, int, int]]:
    """``(name, address, length)`` per subpool carved from one host buffer."""
    if buffer.numel() * buffer.element_size() < plan.total_bytes:
        raise ValueError("KVCR linker local DRAM buffer is smaller than the plan.")
    base = buffer.data_ptr()
    offset = 0
    regions = []
    for name, _ in plan.pool_layouts:
        length = plan.subpool_bytes[name]
        regions.append((name, base + offset, length))
        offset += length
    return regions


def unique_allocations(
    pool_group: DevicePoolGroup,
) -> list[tuple[int, int, str, int, torch.Tensor]]:
    """Distinct underlying allocations behind every pool buffer.

    Returns ``(address, length, mem_type, device_id, owner)`` per allocation,
    deduplicated by storage so several views register once.
    """
    seen: set[tuple[int, int]] = set()
    allocations = []
    for entry in pool_group.entries:
        for buffer in entry.get_hybrid_pool_buffer():
            storage = buffer.untyped_storage()
            identity = (int(storage.data_ptr()), int(storage.nbytes()))
            if identity in seen or identity[1] == 0:
                continue
            seen.add(identity)
            allocations.append(
                (
                    identity[0],
                    identity[1],
                    nixl_mem_type(buffer.device),
                    nixl_device_id(buffer.device),
                    buffer,
                )
            )
    return allocations


def page_descriptors(
    layout: PoolObjectLayout, row: int, agent_name: str, descriptor_type: Any
) -> list:
    """Descriptor list for one page row of a pool, in subpool order."""
    return [
        descriptor_type(
            end_point_name=agent_name,
            mem_type=layout.mem_type,
            addr=base_ptr + row * row_stride,
            size=size,
            device_Id=layout.device_id,
            info=subpool,
        )
        for (base_ptr, row_stride, size), subpool in zip(layout.spans, layout.subpools)
    ]


def compatibility_identity(
    *,
    model_path: str,
    revision: str | None,
    dtype: str,
    kv_cache_dtype: str,
    quantization: str | None,
    page_size: int,
    is_eagle: bool,
    layouts: Mapping[str, PoolObjectLayout],
    shard: Mapping[str, int],
    speculative: Mapping[str, Any],
) -> dict[str, Any]:
    """Identity whose digest namespaces every stored key.

    Covers what changes the bytes of a page: checkpoint, KV representation,
    page geometry and physical span layout, target/draft identity, and the
    physical shard. Runtime addresses, worker identity, and pure DP replica
    identity are excluded so byte-compatible replicas share keys.
    """
    return {
        "namespace": "kvcr-linker-v1",
        "model_path": model_path,
        "revision": revision,
        "dtype": dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "quantization": quantization,
        "page_size": page_size,
        "is_eagle": is_eagle,
        "pools": {
            name: {
                "mem_type": layout.mem_type,
                "subpools": list(layout.subpools),
                "span_sizes": list(layout.span_sizes),
            }
            for name, layout in layouts.items()
        },
        "shard": dict(shard),
        "speculative": dict(speculative),
    }


def compatibility_digest_for(identity: Mapping[str, Any]) -> str:
    return compatibility_digest(identity)


def restorable_boundaries(
    present: Mapping[str, Sequence[bool]],
    policies: Mapping[str, tuple[str, int]],
    num_pages: int,
) -> list[int]:
    """Prefix lengths (in pages) every pool can restore at that boundary.

    ``present[pool][i]`` says whether page ``i`` of ``pool`` is confirmed;
    ``policies[pool]`` is ``("all_pages", _)`` or ``("trailing_pages", window)``.
    Mirrors the UMBP hit policy so the tree's cross-rank intersection sees the
    same sparse set semantics.
    """
    valid = list(range(1, num_pages + 1))
    for pool, (policy, window) in policies.items():
        flags = list(present.get(pool, ()))
        flags.extend([False] * (num_pages - len(flags)))
        prefix = [0]
        for flag in flags[:num_pages]:
            prefix.append(prefix[-1] + int(flag))
        if policy == "all_pages":
            valid = [end for end in valid if prefix[end] == end]
        elif policy == "trailing_pages":
            span = max(1, window)
            valid = [
                end
                for end in valid
                if prefix[end] - prefix[max(0, end - span)] == end - max(0, end - span)
            ]
        else:
            raise ValueError(f"Unsupported pool hit policy: {policy}")
        if not valid:
            break
    return valid
