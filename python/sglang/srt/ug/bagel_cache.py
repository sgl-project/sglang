# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from sglang.srt.ug.context import UGSRTKVTokenBinding


class BAGELSRTKVCacheError(RuntimeError):
    """Raised when BAGEL KV cache cannot be mapped to SRT-owned storage."""


@dataclass(frozen=True, slots=True)
class BAGELSRTKVCacheHandle:
    """Opaque BAGEL KV cache handle owned by the SRT UG runtime."""

    session_id: str
    role: str
    cache_id: str
    num_layers: int


class BAGELSRTKVCacheBackingProtocol(Protocol):
    def create_cache(
        self, *, session_id: str, role: str, num_layers: int
    ) -> BAGELSRTKVCacheHandle: ...

    def clone_cache(
        self, source: BAGELSRTKVCacheHandle, *, role: str
    ) -> BAGELSRTKVCacheHandle: ...

    def get_layer_tensor(
        self, handle: BAGELSRTKVCacheHandle, *, kind: str, layer_id: int
    ) -> torch.Tensor | None: ...

    def set_layer_tensor(
        self,
        handle: BAGELSRTKVCacheHandle,
        *,
        kind: str,
        layer_id: int,
        tensor: torch.Tensor | None,
    ) -> None: ...

    def release_session(self, session_id: str) -> None: ...


class BAGELSRTKVCacheFactory:
    """Builds BAGEL NaiveCache-compatible views over SRT-owned KV storage."""

    def __init__(self, backing: BAGELSRTKVCacheBackingProtocol) -> None:
        self.backing = backing

    def create_cache(
        self,
        *,
        session_id: str,
        role: str,
        template_cache: Any,
    ) -> "BAGELSRTKVCache":
        num_layers = _infer_num_layers(template_cache)
        handle = self.backing.create_cache(
            session_id=session_id,
            role=role,
            num_layers=num_layers,
        )
        cache = BAGELSRTKVCache(handle=handle, backing=self.backing)
        _copy_bagel_cache(source=template_cache, target=cache)
        return cache

    def clone_cache(
        self,
        source_cache: Any,
        *,
        session_id: str,
        role: str,
    ) -> "BAGELSRTKVCache":
        if isinstance(source_cache, BAGELSRTKVCache):
            handle = self.backing.clone_cache(source_cache.handle, role=role)
            return BAGELSRTKVCache(handle=handle, backing=self.backing)
        return self.create_cache(
            session_id=session_id,
            role=role,
            template_cache=source_cache,
        )

    def release_session(self, session_id: str) -> None:
        self.backing.release_session(session_id)

    def bind_request_tokens(self, binding: UGSRTKVTokenBinding) -> None:
        bind = getattr(self.backing, "bind_request_tokens", None)
        if callable(bind):
            bind(binding)


class BAGELSRTKVCache:
    """Official BAGEL `NaiveCache` shape backed by an SRT cache owner."""

    def __init__(
        self,
        *,
        handle: BAGELSRTKVCacheHandle,
        backing: BAGELSRTKVCacheBackingProtocol,
    ) -> None:
        self.handle = handle
        self._backing = backing
        self.key_cache = _BAGELCacheTensorMapping(self, kind="key")
        self.value_cache = _BAGELCacheTensorMapping(self, kind="value")

    @property
    def num_layers(self) -> int:
        return self.handle.num_layers

    @property
    def seq_lens(self) -> int:
        key_tensor = self.key_cache[0]
        if key_tensor is None:
            return 0
        return int(key_tensor.shape[0])

    def clone(self, *, role: str | None = None) -> "BAGELSRTKVCache":
        handle = self._backing.clone_cache(
            self.handle,
            role=role or self.handle.role,
        )
        return BAGELSRTKVCache(handle=handle, backing=self._backing)

    def __deepcopy__(self, memo):
        cloned = self.clone()
        memo[id(self)] = cloned
        return cloned

    def _get(self, *, kind: str, layer_id: int) -> torch.Tensor | None:
        return self._backing.get_layer_tensor(
            self.handle,
            kind=kind,
            layer_id=layer_id,
        )

    def _set(
        self,
        *,
        kind: str,
        layer_id: int,
        tensor: torch.Tensor | None,
    ) -> None:
        self._backing.set_layer_tensor(
            self.handle,
            kind=kind,
            layer_id=layer_id,
            tensor=tensor,
        )


class _BAGELCacheTensorMapping:
    def __init__(self, cache: BAGELSRTKVCache, *, kind: str) -> None:
        self._cache = cache
        self._kind = kind

    def __len__(self) -> int:
        return self._cache.num_layers

    def __getitem__(self, layer_id: int) -> torch.Tensor | None:
        self._validate_layer_id(layer_id)
        return self._cache._get(kind=self._kind, layer_id=int(layer_id))

    def __setitem__(self, layer_id: int, tensor: torch.Tensor | None) -> None:
        self._validate_layer_id(layer_id)
        self._cache._set(kind=self._kind, layer_id=int(layer_id), tensor=tensor)

    def get(
        self,
        layer_id: int,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        value = self[layer_id]
        return default if value is None else value

    def keys(self):
        return range(self._cache.num_layers)

    def items(self):
        for layer_id in self.keys():
            yield layer_id, self[layer_id]

    def _validate_layer_id(self, layer_id: int) -> None:
        if not isinstance(layer_id, int):
            raise TypeError(f"BAGEL cache layer id must be int, got {type(layer_id)}")
        if layer_id < 0 or layer_id >= self._cache.num_layers:
            raise IndexError(
                f"BAGEL cache layer id {layer_id} out of range "
                f"for {self._cache.num_layers} layers"
            )


@dataclass(slots=True)
class _InMemoryCacheRecord:
    handle: BAGELSRTKVCacheHandle
    key_cache: dict[int, torch.Tensor | None] = field(default_factory=dict)
    value_cache: dict[int, torch.Tensor | None] = field(default_factory=dict)

    def tensor_map(self, kind: str) -> dict[int, torch.Tensor | None]:
        if kind == "key":
            return self.key_cache
        if kind == "value":
            return self.value_cache
        raise BAGELSRTKVCacheError(f"Unknown BAGEL cache tensor kind: {kind}")


class InMemoryBAGELSRTKVCacheBacking:
    """Small SRT-owned backing used by unit tests and CPU-only spikes."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._records: dict[str, _InMemoryCacheRecord] = {}
        self._session_cache_ids: dict[str, list[str]] = defaultdict(list)
        self.released_sessions: list[str] = []
        self.request_bindings: list[UGSRTKVTokenBinding] = []

    def bind_request_tokens(self, binding: UGSRTKVTokenBinding) -> None:
        self.request_bindings.append(binding)

    def create_cache(
        self, *, session_id: str, role: str, num_layers: int
    ) -> BAGELSRTKVCacheHandle:
        handle = BAGELSRTKVCacheHandle(
            session_id=session_id,
            role=role,
            cache_id=f"{session_id}:{role}:{next(self._counter)}",
            num_layers=num_layers,
        )
        self._records[handle.cache_id] = _InMemoryCacheRecord(
            handle=handle,
            key_cache={layer_id: None for layer_id in range(num_layers)},
            value_cache={layer_id: None for layer_id in range(num_layers)},
        )
        self._session_cache_ids[session_id].append(handle.cache_id)
        return handle

    def clone_cache(
        self, source: BAGELSRTKVCacheHandle, *, role: str
    ) -> BAGELSRTKVCacheHandle:
        source_record = self._record_for(source)
        cloned = self.create_cache(
            session_id=source.session_id,
            role=role,
            num_layers=source.num_layers,
        )
        cloned_record = self._record_for(cloned)
        for layer_id in range(source.num_layers):
            key = source_record.key_cache[layer_id]
            value = source_record.value_cache[layer_id]
            cloned_record.key_cache[layer_id] = None if key is None else key.clone()
            cloned_record.value_cache[layer_id] = (
                None if value is None else value.clone()
            )
        return cloned

    def get_layer_tensor(
        self, handle: BAGELSRTKVCacheHandle, *, kind: str, layer_id: int
    ) -> torch.Tensor | None:
        tensor = self._record_for(handle).tensor_map(kind)[layer_id]
        return None if tensor is None else tensor.clone()

    def set_layer_tensor(
        self,
        handle: BAGELSRTKVCacheHandle,
        *,
        kind: str,
        layer_id: int,
        tensor: torch.Tensor | None,
    ) -> None:
        self._record_for(handle).tensor_map(kind)[layer_id] = (
            None if tensor is None else tensor.clone()
        )

    def release_session(self, session_id: str) -> None:
        self.released_sessions.append(session_id)
        for cache_id in self._session_cache_ids.pop(session_id, []):
            self._records.pop(cache_id, None)

    def _record_for(self, handle: BAGELSRTKVCacheHandle) -> _InMemoryCacheRecord:
        record = self._records.get(handle.cache_id)
        if record is None:
            raise BAGELSRTKVCacheError(
                f"Unknown BAGEL SRT KV cache handle: {handle.cache_id}"
            )
        return record


@dataclass(slots=True)
class _PagedLayerEntry:
    active_indices: torch.Tensor
    allocated_indices: torch.Tensor
    owns_indices: bool = True


@dataclass(slots=True)
class _PagedCacheRecord:
    handle: BAGELSRTKVCacheHandle
    key_layers: dict[int, _PagedLayerEntry] = field(default_factory=dict)
    value_layers: dict[int, _PagedLayerEntry] = field(default_factory=dict)

    def layer_entries(self, kind: str) -> dict[int, _PagedLayerEntry]:
        if kind == "key":
            return self.key_layers
        if kind == "value":
            return self.value_layers
        raise BAGELSRTKVCacheError(f"Unknown BAGEL cache tensor kind: {kind}")


class BAGELPagedKVCacheBacking:
    """Stores BAGEL dense layer KV tensors in an SRT TokenToKV pool."""

    def __init__(self, token_to_kv_pool_allocator: Any) -> None:
        self.allocator = token_to_kv_pool_allocator
        if not hasattr(token_to_kv_pool_allocator, "get_kvcache"):
            raise BAGELSRTKVCacheError(
                "BAGELPagedKVCacheBacking requires an SRT token_to_kv "
                "allocator with get_kvcache()"
            )
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.page_size = int(getattr(token_to_kv_pool_allocator, "page_size", 1))
        self._counter = itertools.count(1)
        self._records: dict[str, _PagedCacheRecord] = {}
        self._session_cache_ids: dict[str, list[str]] = defaultdict(list)
        self._active_request_bindings: dict[str, UGSRTKVTokenBinding] = {}

    def bind_request_tokens(self, binding: UGSRTKVTokenBinding) -> None:
        token_indices = binding.token_indices
        if not isinstance(token_indices, torch.Tensor):
            token_indices = torch.as_tensor(token_indices, dtype=torch.long)
            binding = UGSRTKVTokenBinding(
                session_id=binding.session_id,
                request_id=binding.request_id,
                token_count=binding.token_count,
                token_indices=token_indices,
            )
        if int(binding.token_count) != int(token_indices.numel()):
            raise BAGELSRTKVCacheError(
                "BAGEL SRT KV token binding token_count does not match "
                f"token_indices length: {binding.token_count} vs "
                f"{token_indices.numel()}"
            )
        self._active_request_bindings[binding.session_id] = binding

    def create_cache(
        self, *, session_id: str, role: str, num_layers: int
    ) -> BAGELSRTKVCacheHandle:
        handle = BAGELSRTKVCacheHandle(
            session_id=session_id,
            role=role,
            cache_id=f"{session_id}:{role}:{next(self._counter)}",
            num_layers=num_layers,
        )
        self._records[handle.cache_id] = _PagedCacheRecord(handle=handle)
        self._session_cache_ids[session_id].append(handle.cache_id)
        return handle

    def clone_cache(
        self, source: BAGELSRTKVCacheHandle, *, role: str
    ) -> BAGELSRTKVCacheHandle:
        cloned = self.create_cache(
            session_id=source.session_id,
            role=role,
            num_layers=source.num_layers,
        )
        for layer_id in range(source.num_layers):
            for kind in ("key", "value"):
                tensor = self.get_layer_tensor(source, kind=kind, layer_id=layer_id)
                if tensor is not None:
                    self.set_layer_tensor(
                        cloned,
                        kind=kind,
                        layer_id=layer_id,
                        tensor=tensor,
                    )
        return cloned

    def get_layer_tensor(
        self, handle: BAGELSRTKVCacheHandle, *, kind: str, layer_id: int
    ) -> torch.Tensor | None:
        record = self._record_for(handle)
        entry = record.layer_entries(kind).get(layer_id)
        if entry is None:
            return None
        buffer = self._buffer_for(kind=kind, layer_id=layer_id)
        return buffer[entry.active_indices].clone()

    def set_layer_tensor(
        self,
        handle: BAGELSRTKVCacheHandle,
        *,
        kind: str,
        layer_id: int,
        tensor: torch.Tensor | None,
    ) -> None:
        record = self._record_for(handle)
        entries = record.layer_entries(kind)
        old_entry = entries.pop(layer_id, None)
        if tensor is None:
            self._free_entry(old_entry)
            return

        buffer = self._buffer_for(kind=kind, layer_id=layer_id)
        bound_indices = self._bound_request_indices(
            handle,
            tensor=tensor,
            device=buffer.device,
        )
        if bound_indices is not None:
            entries[layer_id] = _PagedLayerEntry(
                active_indices=bound_indices,
                allocated_indices=bound_indices,
                owns_indices=False,
            )
            buffer[bound_indices] = tensor.to(device=buffer.device, dtype=buffer.dtype)
            self._free_entry(old_entry)
            return

        counterpart = self._counterpart_entry(record, kind=kind, layer_id=layer_id)
        if counterpart is not None and counterpart.active_indices.numel() == int(
            tensor.shape[0]
        ):
            allocated_indices = counterpart.allocated_indices
            active_indices = counterpart.active_indices
            owns_indices = (
                old_entry.owns_indices
                if self._same_allocation(old_entry, counterpart)
                else False
            )
        else:
            allocated_indices = self._alloc_indices(int(tensor.shape[0]))
            active_indices = allocated_indices[: tensor.shape[0]]
            owns_indices = True
        buffer[active_indices] = tensor.to(device=buffer.device, dtype=buffer.dtype)
        entries[layer_id] = _PagedLayerEntry(
            active_indices=active_indices,
            allocated_indices=allocated_indices,
            owns_indices=owns_indices,
        )
        if not self._same_allocation(old_entry, entries[layer_id]):
            self._free_entry(old_entry)

    def release_session(self, session_id: str) -> None:
        for cache_id in self._session_cache_ids.pop(session_id, []):
            record = self._records.pop(cache_id, None)
            if record is None:
                continue
            for entries in (record.key_layers, record.value_layers):
                for entry in entries.values():
                    self._free_entry(entry)
        self._active_request_bindings.pop(session_id, None)

    def _alloc_indices(self, num_tokens: int) -> torch.Tensor:
        padded_tokens = _ceil_to_page(max(num_tokens, 1), self.page_size)
        indices = self.allocator.alloc(padded_tokens)
        if indices is None:
            raise BAGELSRTKVCacheError(
                f"Failed to allocate {padded_tokens} SRT KV slots for BAGEL cache"
            )
        return indices

    def _free_entry(self, entry: _PagedLayerEntry | None) -> None:
        if entry is not None and entry.owns_indices:
            self.allocator.free(entry.allocated_indices)

    def _bound_request_indices(
        self,
        handle: BAGELSRTKVCacheHandle,
        *,
        tensor: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        binding = self._active_request_bindings.get(handle.session_id)
        if binding is None:
            return None
        token_count = int(tensor.shape[0])
        if token_count <= 0 or token_count > int(binding.token_count):
            return None
        token_indices = binding.token_indices[-token_count:].to(
            device=device,
            dtype=torch.int64,
        )
        return token_indices.clone()

    @staticmethod
    def _same_allocation(
        left: _PagedLayerEntry | None,
        right: _PagedLayerEntry | None,
    ) -> bool:
        if left is None or right is None:
            return False
        return left.allocated_indices.data_ptr() == right.allocated_indices.data_ptr()

    @staticmethod
    def _counterpart_entry(
        record: _PagedCacheRecord,
        *,
        kind: str,
        layer_id: int,
    ) -> _PagedLayerEntry | None:
        if kind == "key":
            return record.value_layers.get(layer_id)
        if kind == "value":
            return record.key_layers.get(layer_id)
        raise BAGELSRTKVCacheError(f"Unknown BAGEL cache tensor kind: {kind}")

    def _buffer_for(self, *, kind: str, layer_id: int) -> torch.Tensor:
        if kind == "key":
            return self.kv_cache.get_key_buffer(layer_id)
        if kind == "value":
            return self.kv_cache.get_value_buffer(layer_id)
        raise BAGELSRTKVCacheError(f"Unknown BAGEL cache tensor kind: {kind}")

    def _record_for(self, handle: BAGELSRTKVCacheHandle) -> _PagedCacheRecord:
        record = self._records.get(handle.cache_id)
        if record is None:
            raise BAGELSRTKVCacheError(
                f"Unknown BAGEL SRT paged KV cache handle: {handle.cache_id}"
            )
        return record


def _copy_bagel_cache(*, source: Any, target: BAGELSRTKVCache) -> None:
    key_cache = getattr(source, "key_cache", None)
    value_cache = getattr(source, "value_cache", None)
    if key_cache is None or value_cache is None:
        return
    for layer_id in range(target.num_layers):
        target.key_cache[layer_id] = key_cache[layer_id]
        target.value_cache[layer_id] = value_cache[layer_id]


def _infer_num_layers(cache: Any) -> int:
    num_layers = getattr(cache, "num_layers", None)
    if num_layers is not None:
        return int(num_layers)
    key_cache = getattr(cache, "key_cache", None)
    if key_cache is not None:
        return len(key_cache)
    raise BAGELSRTKVCacheError(
        "Cannot infer BAGEL cache layer count from reference cache template"
    )


def _ceil_to_page(value: int, page_size: int) -> int:
    if page_size <= 1:
        return value
    return ((value + page_size - 1) // page_size) * page_size
