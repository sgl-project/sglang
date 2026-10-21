# SPDX-License-Identifier: Apache-2.0
"""Translate stable host-page layouts into indexed KVCR registrations."""

from __future__ import annotations

import hashlib
import json

import msgspec
import torch


class PageComponent(msgspec.Struct, frozen=True):
    name: str
    address: int
    size: int
    stride: int
    count: int


class PoolLayout(msgspec.Struct, frozen=True):
    name: str
    page_size: int
    components: tuple[PageComponent, ...]
    format: str

    @property
    def bytes_per_page(self) -> int:
        return sum(component.size for component in self.components)


def describe_pool(name: str, pool) -> PoolLayout:
    """Derive the layout from the same accessor that performs storage IO.

    Each component may have a different element size (e.g. temporal and conv
    state), so it gets its own KVCR allocation pool. A logical page groups all
    its components under one block key; a successful read must carry them all.
    """
    count = pool.size // pool.page_size
    if pool.layout not in ("page_first", "page_first_direct") or count < 2:
        raise ValueError("KVCR requires at least two host slots in a page-first layout")
    pointers, sizes = pool.get_page_buffer_meta(torch.arange(2 * pool.page_size))
    if not pointers or len(pointers) % 2 or len(pointers) != len(sizes):
        raise ValueError(f"Invalid host-page metadata for {name}")
    parts = len(pointers) // 2
    components = []
    for part in range(parts):
        size = sizes[part]
        stride = pointers[parts + part] - pointers[part]
        if size <= 0 or sizes[parts + part] != size or stride < size:
            raise ValueError(f"Nonuniform host-page component in {name}")
        components.append(
            PageComponent(f"{name}_{part}", pointers[part], size, stride, count)
        )
    return PoolLayout(
        name, pool.page_size, tuple(components), f"{pool.layout}:{pool.dtype}"
    )


def page_slots(layout: PoolLayout, keys: list[str], indices: torch.Tensor) -> list[int]:
    """Validate whole, contiguous physical pages before exposing memory to NIXL."""
    if indices.device.type != "cpu" or indices.ndim != 1:
        raise ValueError("KVCR storage indices must be a one-dimensional CPU tensor")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("KVCR storage indices must be integers")
    if len(indices) != len(keys) * layout.page_size:
        raise ValueError("Keys and host-page indices have different lengths")
    pages = indices.reshape(-1, layout.page_size)
    starts = pages[:, 0]
    if len(keys) and (
        torch.any(starts < 0)
        or torch.any(starts % layout.page_size != 0)
        or torch.any(starts // layout.page_size >= layout.components[0].count)
        or not torch.equal(
            pages, starts[:, None] + torch.arange(layout.page_size)[None, :]
        )
    ):
        raise ValueError("KVCR received unaligned, noncontiguous or out-of-range pages")
    return (starts // layout.page_size).tolist()


def storage_namespace(
    config, layouts: dict[str, PoolLayout], *, shared_mla=False
) -> str:
    """Model, byte format and shard identity, but not replica/DP identity."""
    identity = [
        "sglang-kvcr-shared-mla-v1" if shared_mla else "sglang-kvcr-v1",
        config.model_name,
        config.tp_size,
        0 if shared_mla else config.tp_rank,
        config.pp_size,
        config.pp_rank,
        config.attn_cp_size,
        config.attn_cp_rank,
        [
            [
                name,
                layout.page_size,
                layout.format,
                [part.size for part in layout.components],
            ]
            for name, layout in sorted(layouts.items())
        ],
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def block_key(page_key: str, namespace: str, pool: str) -> bytes:
    # Keep the complete hash for storage; only hint membership uses its u64 prefix.
    if len(page_key) != 64:
        raise ValueError("KVCR expects a full SHA256 storage-page hash")
    bytes.fromhex(page_key)
    return f"{page_key}#{namespace}#{pool}".encode()


class StorageKeyAdapter:
    def encode(self, framework_key: object) -> bytes:
        if not isinstance(framework_key, bytes):
            raise TypeError("KVCR storage keys must be bytes")
        return framework_key

    def decode(self, key: bytes) -> int:
        # Inverts SGLang's hash_str_to_int64, as the unsigned router hash.
        return int(key[:16], 16)


def resume_boundaries(page_exists: list[bool], *, trailing: int | None) -> list[int]:
    """A missing checkpoint creates holes, not a monotonically shrinking prefix."""
    if trailing is None:
        boundary = next(
            (i for i, hit in enumerate(page_exists) if not hit), len(page_exists)
        )
        return list(range(1, boundary + 1))
    if trailing < 1:
        raise ValueError("Trailing windows must contain at least one page")
    return [
        end
        for end in range(1, len(page_exists) + 1)
        if all(page_exists[max(0, end - trailing) : end])
    ]
