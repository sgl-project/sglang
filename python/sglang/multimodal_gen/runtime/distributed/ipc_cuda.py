# SPDX-License-Identifier: Apache-2.0
"""Keep CUDA tensors off the pickle path for local scheduler hops.

Two hops share the same tree walk:

- ComfyUI / DiffGenerator ↔ rank-0: replace tensors with ``CudaIpcRef``
  handles before pickle, rebuild with ``UntypedStorage._new_shared_cuda``
- ComfyUI multi-rank recv: detach tensors for NCCL, pickle the skeleton,
  attach them again (not the disagg extract path)
"""

from __future__ import annotations

import copy
import dataclasses
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Producer-side retain: CUDA IPC cannot open a handle in the same process,
# and the allocation must stay alive until the consumer maps it.
# Do not LRU-evict: a live handle that is dropped before the peer maps it
# silently corrupts the hop.
_PRODUCER_TENSORS: OrderedDict[tuple[bytes, int], torch.Tensor] = OrderedDict()
_WARNED_ASYNC_ALLOC = False


def _producer_key(handle: bytes, storage_offset_bytes: int) -> tuple[bytes, int]:
    return (handle, storage_offset_bytes)


def _retain_producer_tensor(
    handle: bytes, storage_offset_bytes: int, tensor: torch.Tensor
) -> None:
    key = _producer_key(handle, storage_offset_bytes)
    _PRODUCER_TENSORS[key] = tensor
    _PRODUCER_TENSORS.move_to_end(key)


def release_retained_producer_tensors() -> None:
    """Drop producer-side IPC retains after the peer has mapped them."""
    _PRODUCER_TENSORS.clear()


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
    def from_tensor(cls, tensor: torch.Tensor) -> CudaIpcRef:
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
            return torch.empty(
                self.shape, dtype=dtype, device=f"cuda:{self.device_index}"
            )

        local = _PRODUCER_TENSORS.pop(
            _producer_key(self.handle, self.storage_offset_bytes), None
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
            msg = str(exc)
            if "shareIpcHandle" not in msg and "cudaMallocAsync" not in msg:
                raise
            global _WARNED_ASYNC_ALLOC
            if not _WARNED_ASYNC_ALLOC:
                _WARNED_ASYNC_ALLOC = True
                logger.warning(
                    "CUDA IPC export failed (%s); falling back to a host pickle "
                    "copy. ComfyUI's default cudaMallocAsync pool cannot export "
                    "IPC handles.",
                    exc,
                )
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


_SEP = "\x1f"


def detach_cuda_tensors(value: Any) -> tuple[Any, dict[str, torch.Tensor]]:
    """Copy the tree, replace CUDA tensors with ``None``, collect them by path."""
    tensors: dict[str, torch.Tensor] = {}
    skeleton = _detach(value, "", tensors)
    return skeleton, tensors


def attach_cuda_tensors(
    value: Any,
    tensors: dict[str, torch.Tensor],
    device: torch.device | None = None,
) -> Any:
    """Put detached CUDA tensors back onto the skeleton."""
    if device is not None:
        moved: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            moved[key] = (
                tensor
                if tensor.device == device
                else tensor.to(device, non_blocking=True)
            )
        tensors = moved
    return _attach(value, "", tensors)


def _join(prefix: str, part: str) -> str:
    return part if not prefix else f"{prefix}{_SEP}{part}"


def _detach(value: Any, prefix: str, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor) and value.is_cuda:
        tensors[prefix] = value
        return None
    if isinstance(value, list):
        return [
            _detach(item, _join(prefix, str(i)), tensors)
            for i, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _detach(item, _join(prefix, str(i)), tensors)
            for i, item in enumerate(value)
        )
    if isinstance(value, dict):
        return {
            key: _detach(item, _join(prefix, str(key)), tensors)
            for key, item in value.items()
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = copy.copy(value)
        for field in dataclasses.fields(value):
            current = getattr(value, field.name, None)
            updated = _detach(current, _join(prefix, field.name), tensors)
            if updated is not current:
                setattr(value, field.name, updated)
        return value
    return value


def _attach(value: Any, prefix: str, tensors: dict[str, torch.Tensor]) -> Any:
    if prefix in tensors:
        return tensors[prefix]
    if isinstance(value, list):
        return [
            _attach(item, _join(prefix, str(i)), tensors)
            for i, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _attach(item, _join(prefix, str(i)), tensors)
            for i, item in enumerate(value)
        )
    if isinstance(value, dict):
        return {
            key: _attach(item, _join(prefix, str(key)), tensors)
            for key, item in value.items()
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            current = getattr(value, field.name, None)
            updated = _attach(current, _join(prefix, field.name), tensors)
            if updated is not current:
                setattr(value, field.name, updated)
        return value
    return value
