# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import atexit
import json
import logging
import mmap
import os
import threading
from contextlib import suppress
from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Protocol

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool_host import HostTensorAllocator
from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHostAllocatorConfig,
    tensorcast_host_allocator_config_from_extra_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorcastHostRegionBinding:
    region_id: str
    capacity_bytes: int
    base_ptr: int
    handle: Any
    region_name: str


@dataclass
class _AllocatorRegionState:
    tensor: torch.Tensor
    mapping: mmap.mmap
    array: np.ndarray
    handle: Any
    binding: TensorcastHostRegionBinding
    cuda_host_registered: bool = False


class _CudaHostRegistrationOps(Protocol):
    def is_available(self) -> bool: ...

    def register(self, ptr: int, size_bytes: int) -> None: ...

    def unregister(self, ptr: int) -> None: ...


class _TorchCudaHostRegistrationOps:
    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def register(self, ptr: int, size_bytes: int) -> None:
        import cuda.bindings.runtime as cuda_rt

        (err,) = cuda_rt.cudaHostRegister(
            ptr,
            size_bytes,
            cuda_rt.cudaHostRegisterPortable,
        )
        if err != cuda_rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostRegister failed with {err}")

    def unregister(self, ptr: int) -> None:
        import cuda.bindings.runtime as cuda_rt

        (err,) = cuda_rt.cudaHostUnregister(ptr)
        if err != cuda_rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostUnregister failed with {err}")


class TensorcastHostTensorAllocator(HostTensorAllocator):
    def __init__(
        self,
        config: TensorcastHostAllocatorConfig,
        *,
        store_factory: Callable[[str], Any] | None = None,
        host_registration_ops: _CudaHostRegistrationOps | None = None,
    ) -> None:
        super().__init__()
        if store_factory is None:
            try:
                import tensorcast as tc
            except ImportError as e:
                raise ImportError(
                    "Please install the `tensorcast` Python package "
                    "(see https://github.com/tensorcast-ai/tensorcast) "
                    "to use the Tensorcast HiCache storage backend."
                ) from e

            store_factory = tc.Store
        self._config = config
        self._store = store_factory(config.daemon_address)
        self._host_registration_ops = (
            host_registration_ops or _TorchCudaHostRegistrationOps()
        )
        self._state: _AllocatorRegionState | None = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    @property
    def binding(self) -> TensorcastHostRegionBinding | None:
        state = self._state
        if state is None:
            return None
        return state.binding

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        if device != "cpu":
            raise ValueError(
                "TensorcastHostTensorAllocator only supports CPU host tensors"
            )
        element_count = int(prod(dims))
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_bytes = element_count * element_size
        if size_bytes <= 0:
            raise ValueError(
                "TensorcastHostTensorAllocator requires a positive allocation size"
            )
        with self._lock:
            if self._state is not None:
                existing = self._state.tensor
                if existing.shape == dims and existing.dtype == dtype:
                    return existing
                raise RuntimeError(
                    "TensorcastHostTensorAllocator supports only one live host-slab allocation per allocator instance"
                )

            from tensorcast.types import HostSharedRegionClass, RegionMemoryKind

            handle = self._store.register_region(
                memory_kind=RegionMemoryKind.HOST_SHARED,
                size_bytes=size_bytes,
                ttl_ms=int(self._config.region_ttl_ms),
                daemon_managed=True,
                host_shared_region_class=HostSharedRegionClass.ALLOCATOR,
                name=self._config.region_name,
            )
            attachment = None
            mapping = None
            try:
                attachment = self._store.attach_host_shared_region(handle)
                fd = int(attachment.fd)
                try:
                    mapping = mmap.mmap(
                        fd,
                        int(attachment.size_bytes),
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(fd)
                array = np.ndarray(
                    (int(attachment.size_bytes),), dtype=np.uint8, buffer=mapping
                )
                tensor = torch.from_numpy(array).view(dtype).view(dims)
            except Exception:
                with suppress(Exception):
                    if mapping is not None:
                        mapping.close()
                with suppress(Exception):
                    if attachment is not None:
                        self._store.release_host_shared_region(handle)
                with suppress(Exception):
                    self._store.unregister_region(str(handle.region_id), force=True)
                raise

            binding = TensorcastHostRegionBinding(
                region_id=str(handle.region_id),
                capacity_bytes=int(attachment.size_bytes),
                base_ptr=int(tensor.data_ptr()),
                handle=handle,
                region_name=self._config.region_name,
            )
            self.dtype = dtype
            self.dims = dims
            self._state = _AllocatorRegionState(
                tensor=tensor,
                mapping=mapping,
                array=array,
                handle=handle,
                binding=binding,
            )
            return tensor

    def ensure_host_registration(self, tensor: torch.Tensor, pin_memory: bool) -> None:
        if not pin_memory:
            return
        with self._lock:
            state = self._state
            if state is None:
                raise RuntimeError(
                    "TensorcastHostTensorAllocator.ensure_host_registration called before allocate"
                )
            if state.cuda_host_registered:
                return
            if state.binding.base_ptr != int(tensor.data_ptr()):
                raise RuntimeError(
                    "TensorcastHostTensorAllocator host registration target does not match active slab binding"
                )
            if not self._host_registration_ops.is_available():
                logger.info(
                    "Tensorcast host slab cudaHostRegister skipped: CUDA runtime unavailable region=%s bytes=%d",
                    state.binding.region_id,
                    state.binding.capacity_bytes,
                )
                return
            try:
                self._host_registration_ops.register(
                    state.binding.base_ptr,
                    state.binding.capacity_bytes,
                )
            except Exception:
                logger.warning(
                    "Tensorcast host slab cudaHostRegister failed; falling back to unregistered mapping region=%s bytes=%d",
                    state.binding.region_id,
                    state.binding.capacity_bytes,
                    exc_info=True,
                )
                return
            state.cuda_host_registered = True
            logger.info(
                "Tensorcast host slab cudaHostRegister succeeded region=%s bytes=%d",
                state.binding.region_id,
                state.binding.capacity_bytes,
            )

    def close(self) -> None:
        with self._lock:
            state = self._state
            self._state = None
            if state is None:
                return
            if state.cuda_host_registered:
                try:
                    self._host_registration_ops.unregister(state.binding.base_ptr)
                except Exception:
                    logger.warning(
                        "Tensorcast host slab cudaHostUnregister failed region=%s",
                        state.binding.region_id,
                        exc_info=True,
                    )
            state.tensor = torch.empty(0, dtype=torch.uint8)
            state.array = np.empty((0,), dtype=np.uint8)
            with suppress(Exception):
                state.mapping.close()
            with suppress(Exception):
                self._store.release_host_shared_region(state.handle)
            with suppress(Exception):
                self._store.unregister_region(state.binding.region_id, force=True)


def build_tensorcast_host_allocator_from_extra_config(
    allocator_config: dict[str, Any] | str | None,
    *,
    layout: str | None,
) -> HostTensorAllocator:
    raw_payload = allocator_config
    if isinstance(raw_payload, str):
        raw_payload = json.loads(raw_payload)

    resolved_config = tensorcast_host_allocator_config_from_extra_config(
        raw_payload if isinstance(raw_payload, dict) else None
    )
    if resolved_config is None:
        return HostTensorAllocator()
    if layout != "page_blob_direct":
        raise ValueError(
            "TensorCast allocator-backed host residency requires --hicache-mem-layout=page_blob_direct"
        )
    return TensorcastHostTensorAllocator(resolved_config)
