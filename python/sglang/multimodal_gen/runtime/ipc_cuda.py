# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC for local ComfyUI / DiffGenerator ↔ rank-0 scheduler hops.

Standard ``pickle.dumps`` copies CUDA tensors through host memory. This module
replaces those tensors with a handle before pickle, and rebuilds them on the
other process with ``UntypedStorage._new_shared_cuda`` so the bytes stay on GPU.
"""

from __future__ import annotations

import copy
import dataclasses
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

# Producer-side retain: CUDA IPC cannot open a handle in the same process,
# and the allocation must stay alive until the consumer maps it.
_PRODUCER_TENSORS: OrderedDict[tuple[bytes, int], torch.Tensor] = OrderedDict()
_MAX_PRODUCER_TENSORS = 64


def _producer_key(handle: bytes, storage_offset_bytes: int) -> tuple[bytes, int]:
    return (handle, storage_offset_bytes)


def _retain_producer_tensor(
    handle: bytes, storage_offset_bytes: int, tensor: torch.Tensor
) -> None:
    key = _producer_key(handle, storage_offset_bytes)
    _PRODUCER_TENSORS[key] = tensor
    _PRODUCER_TENSORS.move_to_end(key)
    while len(_PRODUCER_TENSORS) > _MAX_PRODUCER_TENSORS:
        _PRODUCER_TENSORS.popitem(last=False)


@dataclass
class CudaIpcRef:
    """Picklable CUDA IPC handle plus enough metadata to rebuild the tensor."""

    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    device_index: int
    handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "CudaIpcRef":
        if not tensor.is_cuda:
            raise TypeError("CudaIpcRef only shares CUDA tensors")
        tensor = tensor.detach().contiguous()
        (
            device_index,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = tensor.untyped_storage()._share_cuda_()
        ref = cls(
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            device_index=int(device_index),
            handle=handle,
            storage_size_bytes=int(storage_size_bytes),
            storage_offset_bytes=int(storage_offset_bytes),
            ref_counter_handle=ref_counter_handle,
            ref_counter_offset=int(ref_counter_offset),
            event_handle=event_handle,
            event_sync_required=bool(event_sync_required),
        )
        _retain_producer_tensor(handle, int(storage_offset_bytes), tensor)
        return ref

    def materialize(self) -> torch.Tensor:
        dtype = getattr(torch, self.dtype)
        if self.handle is None or self.storage_size_bytes == 0:
            return torch.empty(self.shape, dtype=dtype, device=f"cuda:{self.device_index}")

        local = _PRODUCER_TENSORS.get(
            _producer_key(self.handle, self.storage_offset_bytes)
        )
        if local is not None:
            return local.detach().clone()

        torch.cuda._lazy_init()
        storage = torch.UntypedStorage._new_shared_cuda(
            self.device_index,
            self.handle,
            self.storage_size_bytes,
            self.storage_offset_bytes,
            self.ref_counter_handle,
            self.ref_counter_offset,
            self.event_handle,
            self.event_sync_required,
        )
        typed = torch.storage.TypedStorage(
            wrap_storage=storage, dtype=dtype, _internal=True
        )
        mapped = torch._utils._rebuild_tensor(
            typed, self.storage_offset, self.shape, self.stride
        )
        # Own a private copy so the IPC mapping can be released immediately.
        return mapped.clone()


def spill_cuda_tensors(value: Any, *, in_place: bool = False) -> Any:
    """Replace CUDA tensors with ``CudaIpcRef`` handles.

    By default dataclasses are shallow-copied so the caller's tensors stay
    intact (needed on the client, which still holds the original ``Req``).
    Scheduler replies can pass ``in_place=True``.
    """
    return _map_tree(value, _spill_one, copy_dataclasses=not in_place)


def materialize_cuda_refs(value: Any) -> Any:
    """Rebuild CUDA tensors from ``CudaIpcRef`` handles, in place on dataclasses."""
    return _map_tree(value, _materialize_one, copy_dataclasses=False)


def _spill_one(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_cuda:
        try:
            return CudaIpcRef.from_tensor(value)
        except RuntimeError as exc:
            # cudaMallocAsync (ComfyUI default) cannot export IPC handles.
            # Leave the tensor in place so pickle falls back to a host copy.
            if "shareIpcHandle" not in str(exc) and "cudaMallocAsync" not in str(exc):
                raise
    return value


def _materialize_one(value: Any) -> Any:
    if isinstance(value, CudaIpcRef):
        return value.materialize()
    return value


def _map_tree(value: Any, fn, copy_dataclasses: bool) -> Any:
    replaced = fn(value)
    if replaced is not value:
        return replaced
    if isinstance(value, list):
        return [_map_tree(item, fn, copy_dataclasses) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tree(item, fn, copy_dataclasses) for item in value)
    if isinstance(value, dict):
        return {
            key: _map_tree(item, fn, copy_dataclasses) for key, item in value.items()
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if copy_dataclasses:
            value = copy.copy(value)
        for field in dataclasses.fields(value):
            current = getattr(value, field.name, None)
            updated = _map_tree(current, fn, copy_dataclasses)
            if updated is not current:
                setattr(value, field.name, updated)
        return value
    return value
