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

import logging
import threading
from queue import PriorityQueue, Queue
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPoolHost

logger = logging.getLogger(__name__)


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

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

    def __lt__(self, other: "CacheOperation"):
        return self.priority < other.priority


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(self, buffer_count: int = 3, max_buffer_size: int = 1000) -> None:
        self.buffers = Queue(maxsize=buffer_count)
        # todo: adjust the buffer size based on throughput profile of the system
        self.max_buffer_size = max_buffer_size

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True) -> None:
        self.buffers.put(item, block=block)

    def get(self, block=True) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block)
        except Exception as e:
            logger.error(e)


class HiCacheController:

    def __init__(
        self,
        mem_pool_device: MHATokenToKVPool,
        mem_pool_host: MLATokenToKVPoolHost,
        write_policy: str = "write_through_selective",
    ):

        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy

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

        self.write_buffer = TransferBuffer()
        self.load_buffer = TransferBuffer()

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.write_thread = threading.Thread(
            target=self.write_thread_func_buffer, daemon=True
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_buffer, daemon=True
        )
        self.write_thread.start()
        self.load_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.put(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.mem_pool_host.protect_write(host_indices)
        return host_indices

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.put(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.mem_pool_host.protect_load(host_indices)
        return device_indices

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        with torch.cuda.stream(self.write_stream):
            while True:
                try:
                    operation = self.write_queue.get(block=True)
                    operation.data = self.mem_pool_device.get_flat_data(
                        operation.device_indices
                    )
                    self.mem_pool_host.transfer(operation.host_indices, operation.data)
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_ids:
                        self.ack_write_queue.put(node_id)
                except Exception as e:
                    logger.error(e)

    def load_thread_func_direct(self):
        """
        Directly load KV caches from host memory to device memory without buffering.
        """
        with torch.cuda.stream(self.load_stream):
            while True:
                try:
                    operation = self.load_queue.get(block=True)
                    operation.data = self.mem_pool_host.get_flat_data(
                        operation.host_indices
                    )
                    self.mem_pool_device.transfer(
                        operation.device_indices, operation.data
                    )
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_ids:
                        self.ack_load_queue.put(node_id)
                except Exception as e:
                    logger.error(e)

    def write_aux_func(self, no_wait=False):
        """
        Auxiliary function to prepare the buffer for write operations.
        """
        buffer = None
        while True:
            try:
                operation = self.write_queue.get(block=True)
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
                    assert (
                        buffer.device_indices.is_cuda
                    ), "Device indices should be on GPU"
                    buffer.data = self.mem_pool_device.get_flat_data(
                        buffer.device_indices
                    ).contiguous()
                    self.write_buffer.put(buffer, block=True)
                    buffer = None
            except Exception as e:
                logger.error(e)

    def load_aux_func(self):
        """
        Auxiliary function to prepare the buffer for load operations.
        """
        buffer = None
        while True:
            try:
                operation = self.load_queue.get(block=True)
                if buffer is None:
                    buffer = operation
                else:
                    buffer.merge(operation)
                if (
                    len(buffer.host_indices) >= self.load_buffer.max_buffer_size
                    or self.load_queue.empty()
                    or self.load_buffer.empty()
                ):
                    buffer.data = (
                        self.mem_pool_host.get_flat_data(buffer.host_indices)
                        .contiguous()
                        .pin_memory()
                    )
                    self.load_buffer.put(buffer, block=True)
                    buffer = None
            except Exception as e:
                logger.error(e)

    def write_thread_func_buffer(self):
        aux_thread = threading.Thread(target=self.write_aux_func, daemon=True)
        aux_thread.start()
        with torch.cuda.stream(self.write_stream):
            while True:
                operation = self.write_buffer.get()
                if operation is None:
                    continue
                self.mem_pool_host.transfer(operation.host_indices, operation.data)
                self.mem_pool_host.complete_io(operation.host_indices)
                for node_id in operation.node_ids:
                    self.ack_write_queue.put(node_id)

    def load_thread_func_buffer(self):
        aux_thread = threading.Thread(target=self.load_aux_func, daemon=True)
        aux_thread.start()
        with torch.cuda.stream(self.load_stream):
            while True:
                operation = self.load_buffer.get()
                if operation is None:
                    continue
                self.mem_pool_device.transfer(operation.device_indices, operation.data)
                self.mem_pool_host.complete_io(operation.host_indices)
                for node_id in operation.node_ids:
                    self.ack_load_queue.put(node_id)

    def evict_device(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> int:
        if self.mem_pool_host.is_synced(host_indices):
            self.mem_pool_device.free(device_indices)
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
