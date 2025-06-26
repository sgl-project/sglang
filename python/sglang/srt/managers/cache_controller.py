from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
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

import concurrent.futures
import logging
import math
import threading
from queue import Empty, Full, PriorityQueue, Queue
from typing import TYPE_CHECKING, List, Optional, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache, MLATokenToKVPoolHost

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    get_world_group
)
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.mem_cache.mooncake_store import MooncakeStore

logger = logging.getLogger(__name__)


class RLockCounter:
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.RLock()

    def increment(self, amount=1):
        with self._lock:
            self.value += amount

    def get_value(self):
        with self._lock:
            return self.value

    def reset(self):
        with self._lock:
            self.value = 0


class LayerDoneCounter:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.counters = [num_layers] * self.num_counters
        self.conditions = [threading.Condition() for _ in range(self.num_counters)]
        self.producer_index = 0
        self.consumer_index = 0

    def next_producer(self):
        return (self.producer_index + 1) % self.num_counters

    def update_producer(self):
        self.producer_index = self.next_producer()
        return self.producer_index

    def set_consumer(self, index):
        self.consumer_index = index

    def increment(self):
        with self.conditions[self.producer_index]:
            self.counters[self.producer_index] += 1
            self.conditions[self.producer_index].notify_all()

    def compare_increment(self, value):
        with self.conditions[self.producer_index]:
            if value > self.counters[self.producer_index]:
                self.counters[self.producer_index] = value
                self.conditions[self.producer_index].notify_all()

    def wait_until(self, threshold):
        with self.conditions[self.consumer_index]:
            while self.counters[self.consumer_index] <= threshold:
                self.conditions[self.consumer_index].wait()

    def reset(self):
        with self.conditions[self.producer_index]:
            self.counters[self.producer_index] = 0


class L3LoadCacheOperation:
    counter = 0

    def __init__(
        self,
        device_indices: torch.Tensor,
        data: torch.Tensor,
        node_id: Union[int, List[int]],
        priority: Optional[int] = None,
    ):
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = data

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    def merge(self, other: "L3LoadCacheOperation", cat_dim: int = 1) -> None:
        # multiple operations can be merged into a single operation for batch processing
        self.device_indices = torch.cat([self.device_indices, other.device_indices])
        self.priority = min(self.priority, other.priority)
        self.node_ids.extend(other.node_ids)
        self.data = torch.cat([self.data, other.data], dim=cat_dim)

    def __lt__(self, other: "L3LoadCacheOperation"):
        return self.priority < other.priority


class MooncakeStoreCacheOperation:
    counter = 0

    def __init__(
        self,
        mooncake_keys: List,
        host_indices: torch.Tensor,
        node_id: Union[int, List[int]],
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.node_ids = [node_id]
        self.mooncake_keys = mooncake_keys

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    def merge(self, other: "MooncakeStoreCacheOperation") -> None:
        # multiple operations can be merged into a single operation for batch processing
        self.host_indices = torch.cat([self.host_indices, other.host_indices])
        self.priority = min(self.priority, other.priority)
        self.mooncake_keys.extend(other.mooncake_keys)
        self.node_ids.extend(other.node_ids)

    def __lt__(self, other: "MooncakeStoreCacheOperation"):
        return self.priority < other.priority


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        l3_keys: Optional[List[str]] = None,
        page_size: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None
        self.l3_keys = l3_keys
        self.page_size = page_size

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    def merge(self, other: "CacheOperation") -> None:
        # multiple operations can be merged into a single operation for batch processing
        self.host_indices = torch.cat([self.host_indices, other.host_indices])
        self.device_indices = torch.cat([self.device_indices, other.device_indices])
        self.priority = min(self.priority, other.priority)
        self.node_ids.extend(other.node_ids)
        if self.l3_keys and other.l3_keys:
            self.l3_keys.extend(other.l3_keys)

    def split(self, factor) -> List["CacheOperation"]:
        # split an operation into smaller operations to reduce the size of intermediate buffers
        if factor <= 1:
            return [self]

        chunk_size = math.ceil(len(self.host_indices) / factor)
        split_ops = []
        for i in range(0, len(self.host_indices), chunk_size):
            split_ops.append(
                CacheOperation(
                    host_indices=self.host_indices[i : i + chunk_size],
                    device_indices=self.device_indices[i : i + chunk_size],
                    node_id=0,
                    l3_keys=(
                        self.l3_keys[
                            i / self.page_size : (i + chunk_size) / self.page_size
                        ]
                        if self.l3_keys
                        else None
                    ),
                    page_size=self.page_size,
                )
            )
        # Inherit the node_ids on the final chunk
        if split_ops:
            split_ops[-1].node_ids = self.node_ids

        return split_ops

    def __lt__(self, other: "CacheOperation"):
        return self.priority < other.priority


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(
        self, stop_event, buffer_count: int = 3, max_buffer_size: int = 1024
    ) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)
        # todo: adjust the buffer size based on throughput profile of the system
        self.max_buffer_size = max_buffer_size

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        self.buffers.queue.clear()


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        enable_mooncake_store_l3_cache: bool,
        load_cache_event: threading.Event = None,
        write_policy: str = "write_through_selective",
        mooncake_l3_kv_pool: MooncakeStore = None,
        mooncake_l3_load_cache_event: threading.Event = None,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size

        self.load_cache_event = load_cache_event
        self.layer_done_counter = LayerDoneCounter(self.mem_pool_device.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.write_queue = PriorityQueue()
        self.load_queue = PriorityQueue()

        self.ack_write_queue = Queue()
        self.ack_load_queue = Queue()

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.write_thread = threading.Thread(
            target=(
                self.write_thread_func_buffer
                if self.page_size == 1
                else self.write_thread_func_direct
            ),
            daemon=True,
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_layer_by_layer, daemon=True
        )
        self.write_thread.start()
        self.load_thread.start()

        self.enable_mooncake_store_l3_cache = enable_mooncake_store_l3_cache
        if self.enable_mooncake_store_l3_cache:

            self.mooncake_l3_kv_pool = mooncake_l3_kv_pool

            self.mooncake_l3_write_queue = PriorityQueue()
            self.mooncake_load_queue = PriorityQueue()
            self.l3_load_queue = PriorityQueue()

            self.mooncake_l3_stop_event = threading.Event()

            self.mooncake_l3_load_cache_event = mooncake_l3_load_cache_event

            self.mooncake_l3_load_stream = torch.cuda.Stream()

            self.mooncake_l3_ack_load_queue = Queue()

            self.l2_layer_counter = RLockCounter()
            self.l3_layer_counter = RLockCounter()

            self.l3_fragment_load = False
            if isinstance(mem_pool_host, MLATokenToKVPoolHost):
                self.l3_fragment_load = True

            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()

            # {"last_node_id" : cache(tensor)}
            self.l3_cache_pool = {}

            # L2 -> L3
            self.mooncake_l3_write_thread = threading.Thread(
                target=self.mooncake_l3_write_thread_func_direct,
                daemon=True,
            )
            # L3 -> tensor(cpu)
            self.mooncake_load_thread = threading.Thread(
                target=self.mooncake_load_thread_func, daemon=True
            )
            # tensor(cpu) -> L1
            self.l3_load_thread = threading.Thread(
                target=self.l3_load_thread_func_layer_by_layer, daemon=True
            )

            self.mooncake_l3_write_thread.start()
            self.mooncake_load_thread.start()
            self.l3_load_thread.start()

    def reset(self):
        self.stop_event.set()
        self.write_thread.join()
        self.load_thread.join()

        self.write_queue.queue.clear()
        self.load_queue.queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.queue.clear()
        self.ack_load_queue.queue.clear()

        self.write_thread = threading.Thread(
            target=(
                self.write_thread_func_buffer
                if self.page_size == 1
                else self.write_thread_func_direct
            ),
            daemon=True,
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_layer_by_layer, daemon=True
        )
        self.stop_event.clear()
        self.write_thread.start()
        self.load_thread.start()

        if self.enable_mooncake_store_l3_cache:
            self.mooncake_l3_stop_event.set()
            self.mooncake_l3_write_thread.join()
            self.mooncake_load_thread.join()
            self.l3_load_thread.join()

            self.mooncake_l3_write_queue.queue.clear()
            self.mooncake_load_queue.queue.clear()
            self.l3_load_queue.queue.clear()

            self.mooncake_l3_ack_load_queue.queue.clear()

            self.mooncake_l3_write_thread = threading.Thread(
                target=self.mooncake_l3_write_thread_func_direct,
                daemon=True,
            )
            self.mooncake_load_thread = threading.Thread(
                target=self.mooncake_load_thread_func, daemon=True
            )
            self.l3_load_thread = threading.Thread(
                target=self.l3_load_thread_func_layer_by_layer, daemon=True
            )

            self.mooncake_l3_stop_event.clear()

            self.mooncake_l3_write_thread.start()
            self.mooncake_load_thread.start()
            self.l3_load_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        l3_keys: Optional[List] = None,
        node_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.mem_pool_host.protect_write(host_indices)
        self.write_queue.put(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                l3_keys=l3_keys,
                page_size=self.page_size,
            )
        )
        return host_indices

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
        load_back_size: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(load_back_size)
        if device_indices is None:
            return None

        if len(host_indices):
            self.mem_pool_host.protect_load(host_indices)
            # to ensure the device indices are ready before accessed by another CUDA stream
            torch.cuda.current_stream().synchronize()
            self.load_queue.put(
                CacheOperation(
                    host_indices, device_indices[: len(host_indices)], node_id, priority
                )
            )
        if self.enable_mooncake_store_l3_cache and load_back_size > len(host_indices):
            l3_cache, l3_cache_len = self.l3_cache_pool.pop(node_id)
            assert load_back_size - len(host_indices) == l3_cache_len
            self.l3_load_queue.put(
                L3LoadCacheOperation(
                    device_indices[len(host_indices) :], l3_cache, node_id
                )
            )

        return device_indices

    def mooncake_load(
        self,
        l3_keys: List[str],
        priority: Optional[int] = None,
        node_id: int = 0,
    ) -> Optional[torch.Tensor]:
        self.mooncake_load_queue.put(
            MooncakeStoreCacheOperation(l3_keys, None, node_id, priority=priority)
        )

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        torch.cuda.set_stream(self.write_stream)
        while not self.stop_event.is_set():
            try:
                operation = self.write_queue.get(block=True, timeout=1)
                self.mem_pool_host.write_page_all_layers(
                    operation.host_indices,
                    operation.device_indices,
                    self.mem_pool_device,
                )
                self.write_stream.synchronize()
                self.mem_pool_host.complete_io(operation.host_indices)

                # write L3 cache
                if self.enable_mooncake_store_l3_cache:
                    mooncake_operation = MooncakeStoreCacheOperation(
                        operation.l3_keys,
                        operation.host_indices,
                        operation.node_ids,
                        priority=operation.priority,
                    )

                    self.mooncake_l3_write_queue.put(mooncake_operation)

                for node_id in operation.node_ids:
                    if node_id != 0:
                        self.ack_write_queue.put(node_id)
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    def mooncake_l3_write_thread_func_direct(self):
        while not self.mooncake_l3_stop_event.is_set():
            try:
                operation = self.mooncake_l3_write_queue.get(block=True, timeout=0.001)

                keys = operation.mooncake_keys
                key_len = len(keys)
                fragment_keys = keys
                start_index = 0

                if self.l3_fragment_load:
                    if key_len % self.tp_size == 0:
                        start_index = key_len // self.tp_size * self.tp_rank
                        end_index = key_len // self.tp_size * (self.tp_rank + 1)
                    else:
                        start_index = (key_len // self.tp_size + 1) * self.tp_rank
                        end_index = min(
                            (key_len // self.tp_size + 1) * (self.tp_rank + 1), key_len
                        )

                    fragment_keys = keys[start_index:end_index]

                mooncake_exist_keys = self.mooncake_l3_kv_pool.is_batch_exist(
                    fragment_keys
                )

                non_exist_keys = []
                non_exist_value = []
                for i in range(len(fragment_keys)):
                    # Other sglang instances may have already written to it,
                    # so only the cache that does not exist is written.
                    if not mooncake_exist_keys[fragment_keys[i]]:
                        non_exist_keys.append(fragment_keys[i])
                        non_exist_value.append(
                            self.mem_pool_host.get_flat_data(
                                operation.host_indices[
                                    (start_index + i)
                                    * self.page_size : (start_index + i + 1)
                                    * self.page_size
                                ]
                            )
                            .contiguous()
                            .pin_memory()
                        )
                if len(non_exist_keys) > 0:
                    self.mooncake_l3_kv_pool.batch_put(non_exist_keys, non_exist_value)
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    def load_thread_func_direct(self):
        """
        Directly load KV caches from host memory to device memory without buffering.
        """
        torch.cuda.set_stream(self.load_stream)
        while not self.stop_event.is_set():
            try:
                operation = self.load_queue.get(block=True, timeout=1)
                operation.data = self.mem_pool_host.get_flat_data(
                    operation.host_indices
                )
                self.mem_pool_device.transfer(operation.device_indices, operation.data)
                self.mem_pool_host.complete_io(operation.host_indices)
                for node_id in operation.node_ids:
                    if node_id != 0:
                        self.ack_load_queue.put(node_id)
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    def _fragment_cache_all_gather(self, fragment_tensor, unpadded_len):
        fragment_tensor = fragment_tensor.to(f"cuda:{get_world_group().local_rank}", non_blocking=True)
        if not self.l3_fragment_load:
            return fragment_tensor
        gathered_tensor = tensor_model_parallel_all_gather(fragment_tensor, dim=1)
        return gathered_tensor[:, :unpadded_len]

    def mooncake_load_thread_func(self):
        while not self.mooncake_l3_stop_event.is_set():
            try:
                operation = self.mooncake_load_queue.get(block=True, timeout=0.001)
                keys = operation.mooncake_keys
                key_len = len(keys)
                fragment_keys = keys
                if self.l3_fragment_load:
                    # The PCIe bandwidth is much lower than that of NVLink.
                    # At the same time, for the MLA model, the KV cache of each TP worker is the same,
                    # so the TP rank loads the mooncake cache in segments and then all gather
                    if key_len % self.tp_size == 0:
                        start_index = key_len // self.tp_size * self.tp_rank
                        end_index = key_len // self.tp_size * (self.tp_rank + 1)
                    else:
                        start_index = min(
                            (key_len // self.tp_size + 1) * self.tp_rank, key_len
                        )
                        end_index = min(
                            (key_len // self.tp_size + 1) * (self.tp_rank + 1), key_len
                        )

                    fragment_keys = keys[start_index:end_index]

                fragment_key_len = len(fragment_keys)
                batch_data = None
                if fragment_key_len > 0:
                    batch_data = self.mooncake_l3_kv_pool.batch_get(fragment_keys)

                # last few rank need to pad
                if (
                    self.l3_fragment_load
                    and key_len % self.tp_size != 0
                ):
                    fragment_keys_pad_size = key_len // self.tp_size + 1
                    if fragment_key_len < fragment_keys_pad_size:
                        token_tensor_shape = self.mooncake_l3_kv_pool.page_tensor_shape
                        token_tensor_shape_pad = torch.Size(
                            [
                                token_tensor_shape[0],
                                fragment_keys_pad_size * self.page_size,
                                token_tensor_shape[2],
                                token_tensor_shape[3],
                            ]
                        )
                        batch_data_pad = torch.zeros(
                            token_tensor_shape_pad, dtype=self.mooncake_l3_kv_pool.dtype
                        )
                        if fragment_key_len > 0:
                            batch_data_pad[:, : fragment_key_len * self.page_size] = (
                                batch_data
                            )
                        batch_data = batch_data_pad

                batch_data = self._fragment_cache_all_gather(
                    batch_data, key_len * self.page_size
                )
                self.l3_cache_pool[operation.node_ids[0]] = (
                    batch_data,
                    key_len * self.page_size,
                )

                for node_id in operation.node_ids:
                    if node_id != 0:
                        self.mooncake_l3_ack_load_queue.put(node_id)
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    def l3_load_thread_func_layer_by_layer(self):
        torch.cuda.set_stream(self.mooncake_l3_load_stream)
        while not self.mooncake_l3_stop_event.is_set():
            self.mooncake_l3_load_cache_event.wait(timeout=1)
            if not self.mooncake_l3_load_cache_event.is_set():
                continue
            self.mooncake_l3_load_cache_event.clear()

            batch_operation = None
            cat_dim = 1 if isinstance(self.mem_pool_host, MLATokenToKVPoolHost) else 2
            while self.l3_load_queue.qsize() > 0:
                op = self.l3_load_queue.get(block=True)
                if batch_operation is None:
                    batch_operation = op
                else:
                    batch_operation.merge(op, cat_dim=cat_dim)
            if batch_operation is None:
                continue

            self.l3_layer_counter.reset()

            # TODO:(huangtingwei9988): Can be optimized to layer_by_layer
            self.mem_pool_device.transfer(
                batch_operation.device_indices, batch_operation.data
            )
            del batch_operation.data

            self.l3_layer_counter.increment(self.mem_pool_host.layer_num)
            self.mooncake_l3_load_stream.synchronize()

            self.layer_done_counter.compare_increment(
                min(
                    self.l3_layer_counter.get_value(), self.l2_layer_counter.get_value()
                )
            )

            for node_id in batch_operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put(node_id)

    def load_thread_func_layer_by_layer(self):
        """
        Load KV caches from host memory to device memory layer by layer.
        """
        torch.cuda.set_stream(self.load_stream)
        while not self.stop_event.is_set():
            self.load_cache_event.wait(timeout=1)
            if not self.load_cache_event.is_set():
                continue
            self.load_cache_event.clear()
            self.layer_done_counter.update_producer()

            batch_operation = None
            while self.load_queue.qsize() > 0:
                op = self.load_queue.get(block=True)
                if batch_operation is None:
                    batch_operation = op
                else:
                    batch_operation.merge(op)
            if batch_operation is None:
                continue

            # start layer-wise KV cache transfer from CPU to GPU
            self.layer_done_counter.reset()
            if self.enable_mooncake_store_l3_cache:
                self.l2_layer_counter.reset()

            for i in range(self.mem_pool_host.layer_num):
                if self.page_size == 1:
                    flat_data = self.mem_pool_host.get_flat_data_by_layer(
                        batch_operation.host_indices, i
                    )
                    self.mem_pool_device.transfer_per_layer(
                        batch_operation.device_indices, flat_data, i
                    )
                else:
                    self.mem_pool_host.load_page_per_layer(
                        batch_operation.host_indices,
                        batch_operation.device_indices,
                        self.mem_pool_device,
                        i,
                    )
                    self.load_stream.synchronize()

                if self.enable_mooncake_store_l3_cache:
                    self.layer_done_counter.compare_increment(
                        min(
                            self.l3_layer_counter.get_value(),
                            self.l2_layer_counter.get_value(),
                        )
                    )
                else:
                    self.layer_done_counter.increment()

            self.mem_pool_host.complete_io(batch_operation.host_indices)
            for node_id in batch_operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put(node_id)

    def write_aux_func(self, no_wait=False):
        """
        Auxiliary function to prepare the buffer for write operations.
        """
        torch.cuda.set_stream(self.write_stream)

        def _to_op(op_):
            assert op_.device_indices.is_cuda, "Device indices should be on GPU"
            op_.data = self.mem_pool_device.get_flat_data(op_.device_indices).to(
                self.mem_pool_host.device
            )
            self.write_buffer.put(op_)
            return op_

        buffer = None
        while not self.stop_event.is_set():
            try:
                operation = self.write_queue.get(block=True, timeout=1)
                factor = (
                    len(operation.device_indices) // self.write_buffer.max_buffer_size
                )

                if factor >= 1:
                    if buffer is not None:
                        _to_op(buffer)
                        buffer = None

                    if factor < 2:
                        _to_op(operation)
                    else:
                        split_ops = operation.split(factor)
                        for op_ in split_ops:
                            _to_op(op_)
                    continue

                if buffer is None:
                    buffer = operation
                else:
                    buffer.merge(operation)
                if (
                    no_wait
                    or len(buffer.host_indices) >= self.write_buffer.max_buffer_size
                    or self.write_queue.empty()
                    or self.write_buffer.empty()
                ):
                    _to_op(buffer)
                    buffer = None
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    def load_aux_func(self):
        """
        Auxiliary function to prepare the buffer for load operations.
        """

        def _pin_op(op_, put=True):
            op_.data = (
                self.mem_pool_host.get_flat_data(op_.host_indices)
                .contiguous()
                .pin_memory()
            )
            if put:
                self.load_buffer.put(op_)
            return op_

        buffer = None
        while not self.stop_event.is_set():
            try:
                operation = self.load_queue.get(block=True, timeout=1)
                factor = len(operation.host_indices) // self.load_buffer.max_buffer_size

                if factor >= 1:
                    if buffer is not None:
                        _pin_op(buffer)
                        buffer = None

                    if factor < 2:
                        _pin_op(operation)
                    else:
                        split_ops = operation.split(factor)
                        split_args = [(op_, True) for op_ in split_ops[:-1]]
                        split_args.append((split_ops[-1], False))
                        # Spawn threads to pin each op concurrently
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            pinned_ops = list(
                                executor.map(
                                    lambda x: _pin_op(x[0], put=x[1]), split_args
                                )
                            )
                        # preserve the order of last op to ensure correct ack
                        self.load_buffer.put(pinned_ops[-1])
                    continue

                if buffer is None:
                    buffer = operation
                else:
                    buffer.merge(operation)
                if (
                    len(buffer.host_indices) >= self.load_buffer.max_buffer_size
                    or self.load_queue.empty()
                    or self.load_buffer.empty()
                ):
                    _pin_op(buffer)
                    buffer = None
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    # todo (zhiqiang): double buffering to be deprecated
    def write_thread_func_buffer(self):
        aux_thread = threading.Thread(target=self.write_aux_func, daemon=True)
        aux_thread.start()

        while not self.stop_event.is_set():
            operation = self.write_buffer.get()
            if operation is None:
                continue

            # write L2 cache
            self.mem_pool_host.assign_flat_data(operation.host_indices, operation.data)
            self.mem_pool_host.complete_io(operation.host_indices)

            # write L3 cache
            if self.enable_mooncake_store_l3_cache:
                mooncake_operation = MooncakeStoreCacheOperation(
                    operation.l3_keys,
                    operation.host_indices,
                    operation.node_ids,
                    priority=operation.priority,
                )

                self.mooncake_l3_write_queue.put(mooncake_operation)

            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_write_queue.put(node_id)
        aux_thread.join()

    def load_thread_func_buffer(self):
        torch.cuda.set_stream(self.load_stream)
        aux_thread = threading.Thread(target=self.load_aux_func, daemon=True)
        aux_thread.start()
        while not self.stop_event.is_set():
            operation = self.load_buffer.get()
            if operation is None:
                continue
            self.mem_pool_device.transfer(operation.device_indices, operation.data)
            self.mem_pool_host.complete_io(operation.host_indices)
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put(node_id)
        aux_thread.join()

    def evict_device(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> int:
        if self.mem_pool_host.is_synced(host_indices):
            self.mem_pool_device_allocator.free(device_indices)
            self.mem_pool_host.update_backup(host_indices)
            return len(device_indices)
        else:
            raise ValueError(
                f"Inconsistent states: {self.mem_pool_host.get_state(host_indices)}"
            )

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        if self.mem_pool_host.is_backup(host_indices):
            self.mem_pool_host.free(host_indices)
            return len(host_indices)
        else:
            raise ValueError(
                f"Inconsistent states: {self.mem_pool_host.get_state(host_indices)}"
            )
