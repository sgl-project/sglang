import threading
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import torch
from sgl_kernel.radix_tree import RadixTreeCpp, TreeNodeCpp

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolHost,
    MLATokenToKVPool,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

import logging
import threading
from queue import Empty, PriorityQueue, Queue
from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import HostKVCache, TokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class LayerDoneCounter:
    def __init__(self, num_layers):
        self.counter = num_layers
        self.condition = threading.Condition()

    def increment(self):
        with self.condition:
            self.counter += 1
            self.condition.notify_all()

    def wait_until(self, threshold):
        with self.condition:
            while self.counter <= threshold:
                self.condition.wait()

    def reset(self):
        with self.condition:
            self.counter = 0


class CacheOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: TreeNodeCpp,
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


class HiCacheController_v2:
    def __init__(
        self,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tree: RadixTreeCpp,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        oracle: bool = False,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.oracle = oracle
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

        self.write_queue = PriorityQueue[TreeNodeCpp]()
        self.load_queue = PriorityQueue[CacheOperation]()

        self.ack_write_queue = Queue[TreeNodeCpp]()
        self.ack_load_queue = Queue[TreeNodeCpp]()

        self.stop_event = threading.Event()

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.write_thread = threading.Thread(
            target=self.write_thread_func_direct, daemon=True
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_layer_by_layer, daemon=True
        )
        self.write_thread.start()
        self.load_thread.start()
        self.tree = tree

    def reset(self):
        self.stop_event.set()
        self.write_thread.join()
        self.load_thread.join()

        self.write_queue.queue.clear()
        self.load_queue.queue.clear()
        self.ack_write_queue.queue.clear()
        self.ack_load_queue.queue.clear()

        self.write_thread = threading.Thread(
            target=self.write_thread_func_direct, daemon=True
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_layer_by_layer, daemon=True
        )
        self.stop_event.clear()
        self.write_thread.start()
        self.load_thread.start()

    def write(self, node_id: TreeNodeCpp):
        self.write_queue.put(node_id)

    def load(
        self,
        device_indices: torch.Tensor,
        host_indices: torch.Tensor,
        node_id: TreeNodeCpp,
    ):
        """
        Load KV caches from host memory to device memory.
        """
        self.load_queue.put(CacheOperation(host_indices, device_indices, node_id))

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        torch.cuda.set_stream(self.write_stream)  # type: ignore
        while not self.stop_event.is_set():
            try:
                node_id = self.write_queue.get(block=True, timeout=1)
                device_indices, host_indices = self.tree.start_write_through(node_id)
                operation = CacheOperation(host_indices, device_indices, node_id)
                if not self.oracle:
                    if self.page_size == 1:
                        self.mem_pool_host.transfer_all_layer_kernel(
                            self.mem_pool_device,
                            operation.device_indices,
                            operation.host_indices.to(self.mem_pool_device.device),
                        )
                        self.write_stream.synchronize()
                    else:
                        self.mem_pool_host.write_page_all_layers(
                            operation.host_indices,
                            operation.device_indices,
                            self.mem_pool_device,
                        )
                        self.write_stream.synchronize()

                self.mem_pool_host.complete_io(operation.host_indices)
                for node_id in operation.node_ids:
                    if node_id != 0:
                        self.ack_write_queue.put(node_id)
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    def load_thread_func_layer_by_layer(self):
        """
        Load KV caches from host memory to device memory layer by layer.
        """
        torch.cuda.set_stream(self.load_stream)  # type: ignore
        while not self.stop_event.is_set():
            self.load_cache_event.wait(timeout=1)
            if not self.load_cache_event.is_set():
                continue
            self.load_cache_event.clear()

            # TODO(dark): optimize the batch merge operations
            # reduce the number of torch.cat calls
            batch_operation = None
            while self.load_queue.qsize() > 0:
                op = self.load_queue.get(block=True)
                if batch_operation is None:
                    batch_operation = op
                else:
                    batch_operation.merge(op)
            if batch_operation is None:
                continue

            self.layer_done_counter.reset()
            host_indices_device = batch_operation.host_indices.to(
                self.mem_pool_device.device
            )
            for i in range(self.mem_pool_host.layer_num):
                if not self.oracle:
                    if self.page_size == 1:
                        self.mem_pool_device.transfer_per_layer_kernel(
                            self.mem_pool_host,
                            host_indices_device,
                            batch_operation.device_indices,
                            i,
                        )
                        self.load_stream.synchronize()
                    else:
                        self.mem_pool_host.load_page_per_layer(
                            batch_operation.host_indices,
                            batch_operation.device_indices,
                            self.mem_pool_device,
                            i,
                        )
                        self.load_stream.synchronize()
                self.layer_done_counter.increment()

            self.mem_pool_host.complete_io(batch_operation.host_indices)
            for node_id in batch_operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put(node_id)

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


class HiRadixCacheCpp(BasePrefixCache):
    def merge_tensor(self, l: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge a list of tensors into a single tensor.
        Args:
            l (List[torch.Tensor]): List of tensors to merge.
        Returns:
            torch.Tensor: Merged tensor.
        """
        if len(l) == 0:
            return self.empty_indices(self.device)
        elif len(l) == 1:
            return l[0]
        else:
            return torch.cat(l)

    def __init__(
        self,
        disable: bool,
        use_hicache: bool,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: TokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        enable_kv_cache_events: bool,
    ):
        assert (
            enable_kv_cache_events is False
        ), "HiRadixCache does not support kv cache events yet"
        self.kv_cache = token_to_kv_pool.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                page_size,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                page_size,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = tp_cache_group

        self.load_cache_event = threading.Event()
        self.tree = RadixTreeCpp(
            disable,
            use_hicache,
            page_size,
            self.token_to_kv_pool_host.size,
            self.write_through_threshold,
        )
        self.cache_controller = HiCacheController_v2(
            token_to_kv_pool,
            self.token_to_kv_pool_host,
            page_size,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
            tree=self.tree,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through: Set[TreeNodeCpp] = set()
        # record the node segments with ongoing load back
        self.ongoing_load_back: Dict[TreeNodeCpp, Tuple[TreeNodeCpp, TreeNodeCpp]] = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.device = token_to_kv_pool.device
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.page_size = page_size

    def reset(self):
        self.tree.reset()

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        device_indices_vec, host_indices_length, node_gpu, node_cpu = (
            self.tree.match_prefix(key)
        )
        return MatchResult(
            device_indices=self.merge_tensor(device_indices_vec),
            last_device_node=node_gpu,
            last_host_node=node_cpu,
            host_indices_length=host_indices_length,
        )

    def insert(self, key: List[int], value: torch.Tensor) -> int:
        """
        Insert a key-value pair into the radix tree.
        Args:
            key (List[int]): The key to insert, represented as a list of integers.
            value (torch.Tensor): The value to associate with the key.
        Returns:
            int: Number of device nodes that were already present in the tree before the insertion.
        """
        ongoing_write_node, length = self.tree.insert(key, value)
        self.ongoing_write_through.update(ongoing_write_node)
        for node_id in ongoing_write_node:
            self.cache_controller.write(node_id)
        return length

    def dec_lock_ref(self, node: TreeNodeCpp):
        """
        Decrement the reference count of a node to root of the radix tree.
        Args:
            node (TreeNodeCpp): The handle of the node to decrement the reference count for.
        """
        self.tree.lock_ref(node, False)  # do not increment

    def inc_lock_ref(self, node: TreeNodeCpp):
        """
        Increment the reference count of from a node to root of the radix tree.
        Args:
            node (TreeNodeCpp): The handle of the node to increment the reference count for.
        """
        self.tree.lock_ref(node, True)

    def evict(self, num_tokens: int):
        evicted_device_indices = self.tree.evict(num_tokens)
        for indice in evicted_device_indices:
            self.token_to_kv_pool.free(indice)

    def evictable_size(self):
        return self.tree.evictable_size()

    def protected_size(self):
        return self.tree.protected_size()

    def total_size(self):
        return self.tree.total_size()

    def cache_finished_req(self, req: Req):
        """Cache request when it finishes."""
        assert req.req_pool_idx is not None
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        self.token_to_kv_pool.free(kv_indices)
        self.req_to_token_pool.free(req.req_pool_idx)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # NOTE: our C++ implementation don't need token_ids and kv_indices to be page-aligned
        # it will automatically align them, but the length of them should be equal
        old_prefix_len = len(req.prefix_indices)
        new_prefix_len = self.insert(token_ids, kv_indices)

        # these 2 prefix_len should be page-aligned
        if old_prefix_len < new_prefix_len:
            self.token_to_kv_pool.free(kv_indices[old_prefix_len:new_prefix_len])

        # need to free the unaligned part, since it cannot be inserted into the radix tree
        # NOTE: sglang PagedAllocator support unaligned free (which will automatically align it)
        if (
            self.page_size != 1  # unaligned tail only exists when page_size > 1
            and (unaligned_len := len(token_ids) % self.page_size) > 0
        ):
            self.token_to_kv_pool.free(kv_indices[len(token_ids) - unaligned_len :])

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)
        self.req_to_token_pool.free(req.req_pool_idx)

    def cache_unfinished_req(self, req: Req):
        """Cache request when it is unfinished."""
        assert req.req_pool_idx is not None

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # these should be page-aligned
        old_prefix_len = len(req.prefix_indices)
        new_prefix_len = self.insert(token_ids, kv_indices)

        # TODO(dark): optimize the insert and match (e.g. merge into 1 function)

        # KVCache between old & new is newly generated, but already exists in the pool
        # we need to free the newly generated kv indices and reuse the indices in the pool
        if old_prefix_len < new_prefix_len:
            self.token_to_kv_pool.free(kv_indices[old_prefix_len:new_prefix_len])

            # The prefix indices need to updated to reuse the kv indices in the pool
            new_indices_vec, _, new_last_node, _ = self.tree.match_prefix(token_ids)
            new_indices = self.merge_tensor(new_indices_vec)
            assert new_prefix_len <= len(new_indices)
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, old_prefix_len:new_prefix_len
            ] = new_indices[old_prefix_len:new_prefix_len]

            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)

            req.prefix_indices = new_indices
            req.last_node = new_last_node

    def init_load_host(
        self,
        device_node: TreeNodeCpp,
        host_node: TreeNodeCpp,
        new_device_indices: torch.Tensor,
    ):
        host_indices_vec = self.tree.load_onboard(
            device_node, host_node, new_device_indices
        )
        host_indices = self.merge_tensor(host_indices_vec)
        assert len(host_indices) == len(new_device_indices)
        self.cache_controller.load(
            device_indices=new_device_indices,
            host_indices=host_indices,
            node_id=host_node,  # NOTE: node is actually an int id
        )

    def ready_to_load_host(self):
        self.load_cache_event.set()

    def check_host_cache(self):
        self.writing_check()
        self.loading_check()

    def writing_check(self):
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(int(queue_size.item())):
            ack_id = self.cache_controller.ack_write_queue.get()
            self.tree.commit_write_through(ack_id, True)
            self.ongoing_write_through.remove(ack_id)

    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
            except Exception:
                break
            start_node, end_node = self.ongoing_load_back[ack_id]
            self.tree.commit_load_onboard(start_node, end_node, True)
            del self.ongoing_load_back[ack_id]
