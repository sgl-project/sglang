"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import logging
import threading
from enum import IntEnum
from functools import wraps
from typing import List, Optional, Tuple, Union

import numpy as np
import psutil
import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import debug_timing, get_compiler_backend

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region():
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data(self, indices):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter


class TokenToKVPoolAllocator:
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = 1

        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self._kvcache = kvcache

    def available_size(self):
        return len(self.free_slots)

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.free_slots = torch.cat((self.free_slots, free_index))
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []


class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self._create_buffers()

        self.layer_transfer_counter = None
        self.capture_mode = False
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream()

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr() for i in range(self.layer_num)
        ] + [self.get_value_buffer(i).data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes for i in range(self.layer_num)
        ] + [self.get_value_buffer(i).nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes for i in range(self.layer_num)
        ] + [self.get_value_buffer(i)[0].nbytes for i in range(self.layer_num)]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        flatten = torch.stack(
            [
                torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        self.k_buffer[layer_id][indices] = k_data
        self.v_buffer[layer_id][indices] = v_data

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)

        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.capture_mode and cache_k.shape[0] < 4:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with self.device_module.stream(self.alt_stream):
                self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v


@torch.compile
def fused_downcast(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dtype: torch.dtype,
    store_dtype: torch.dtype,
    max_fp8: float,
    min_fp8: float,
):
    cache_k = cache_k / k_scale
    cache_k = torch.clamp(cache_k, min_fp8, max_fp8)
    cache_v = cache_v / v_scale
    cache_v = torch.clamp(cache_v, min_fp8, max_fp8)
    cache_k = cache_k.to(dtype)
    cache_v = cache_v.to(dtype)
    cache_k = cache_k.view(store_dtype)
    cache_v = cache_v.view(store_dtype)
    return cache_k, cache_v


# This compiled version is slower in the unit test
# python3 -m unittest test_bench_serving.TestBenchServing.test_offline_throughput_non_stream_small_batch_size
@torch.compile(dynamic=True, backend=get_compiler_backend())
def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):
    dst_1[loc] = src_1.to(dtype).view(store_dtype)
    dst_2[loc] = src_2.to(dtype).view(store_dtype)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        with memory_saver_adapter.region():
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.kv_buffer = [
                torch.zeros(
                    (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                    dtype=self.store_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]

        self.layer_transfer_counter = None

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id].view(self.dtype)
        return self.kv_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        return self.kv_buffer[layer_id][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
        else:
            self.kv_buffer[layer_id][loc] = cache_k

    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        return torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        for i in range(self.layer_num):
            self.kv_buffer[i][indices] = flat_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        self.kv_buffer[layer_id][indices] = flat_data


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        with memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

            # [size, head_num, heavy_channel_num] for each layer
            self.label_buffer = [
                torch.zeros(
                    (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v
        self.label_buffer[layer_id][loc] = cache_label

    def get_flat_data(self, indices):
        pass

    def transfer(self, indices, flat_data):
        pass

    def transfer_per_layer(self, indices, flat_data, layer_id):
        pass


class MemoryStateInt(IntEnum):
    IDLE = 0
    RESERVED = 1
    PROTECTED = 2
    SYNCED = 3
    BACKUP = 4


def synchronized(debug_only=False):
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if (not debug_only) or self.debug:
                return func(self, *args, **kwargs)
                with self.lock:
                    return func(self, *args, **kwargs)
            else:
                return True

        return wrapper

    return _decorator


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        pin_memory: bool = False,  # no need to use pin memory with the double buffering
        device: str = "cpu",
    ):
        assert (
            host_to_device_ratio >= 1
        ), "The host memory should be larger than the device memory with the current protocol"
        # todo, other ways of configuring the size

        self.device_pool = device_pool
        self.host_to_device_ratio = host_to_device_ratio
        self.pin_memory = pin_memory
        self.device = device

        self.size = int(device_pool.size * host_to_device_ratio)
        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        if requested_bytes > host_mem.available - ten_gb:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{host_mem.available / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data(self, indices):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data_by_layer(self, indices, layer_id):
        raise NotImplementedError()

    @abc.abstractmethod
    def assign_flat_data(self, indices, flat_data):
        raise NotImplementedError()

    @synchronized()
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized()
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        if self.debug:
            self.mem_state[select_index] = MemoryStateInt.RESERVED

        return select_index

    @synchronized()
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        if self.debug:
            self.mem_state[indices] = MemoryStateInt.IDLE
        return len(indices)

    @synchronized(debug_only=True)
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (
            states == states[0]
        ).all(), "The memory slots should have the same state {}".format(states)
        return MemoryStateInt(states[0].item())

    @synchronized(debug_only=True)
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized(debug_only=True)
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized(debug_only=True)
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized(debug_only=True)
    def update_backup(self, indices: torch.Tensor):
        if not self.is_synced(indices):
            raise ValueError(
                f"The host memory slots should be in SYNCED state before turning into BACKUP. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized(debug_only=True)
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized(debug_only=True)
    def protect_write(self, indices: torch.Tensor):
        if not self.is_reserved(indices):
            raise ValueError(
                f"The host memory slots should be RESERVED before write operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def protect_load(self, indices: torch.Tensor):
        if not self.is_backup(indices):
            raise ValueError(
                f"The host memory slots should be in BACKUP state before load operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def complete_io(self, indices: torch.Tensor):
        if not self.is_protected(indices):
            raise ValueError(
                f"The host memory slots should be PROTECTED during I/O operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.SYNCED


class MHATokenToKVPoolHost(HostKVCache):
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        pin_memory: bool = False,  # no need to use pin memory with the double buffering
        device: str = "cpu",
    ):
        super().__init__(device_pool, host_to_device_ratio, pin_memory, device)

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        return torch.empty(
            (2, self.layer_num, self.size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        self.kv_buffer[:, :, indices] = flat_data.to(
            device=self.device, non_blocking=False
        )

    def get_flat_data(self, indices):
        return self.kv_buffer[:, :, indices]

    def get_flat_data_by_layer(self, indices, layer_id):
        return self.kv_buffer[:, layer_id, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, :, indices] = flat_data


class MLATokenToKVPoolHost(HostKVCache):
    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        pin_memory: bool = False,  # no need to use pin memory with the double buffering
        device: str = "cpu",
    ):
        super().__init__(device_pool, host_to_device_ratio, pin_memory, device)

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num

        return (self.kv_lora_rank + self.qk_rope_head_dim) * 1 * self.dtype.itemsize

    def init_kv_buffer(self):
        return torch.empty(
            (
                self.layer_num,
                self.size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        self.kv_buffer[:, indices] = flat_data.to(
            device=self.device, non_blocking=False
        )

    def get_flat_data(self, indices):
        return self.kv_buffer[:, indices]

    def get_flat_data_by_layer(self, indices, layer_id):
        return self.kv_buffer[layer_id, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, indices] = flat_data
