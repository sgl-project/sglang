import os
import threading
from typing import TYPE_CHECKING, List, Set

import torch
from sgl_kernel.radix_tree import IOHandle, RadixTreeCpp, TreeNodeCpp

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
else:
    Req = object

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


class DebugTree:
    def __init__(self, tree: RadixTreeCpp):
        self.tree = tree

    # override getattr to proxy all attributes to the tree
    def __getattr__(self, name):
        if hasattr(self.tree, name):
            func = getattr(self.tree, name)
            if name.endswith("_size"):
                return func

            def wrapper(*args):
                result = func(*args)
                print("=" * 100)
                logger.info(f"[DEBUG Tree]: Calling {name} method, {args=}")
                logger.info(f"[DEBUG Tree]: {result=}")
                import traceback

                # print out recent 5 functions
                stack = traceback.extract_stack()[-5:]
                for frame in stack:
                    logger.info(
                        f"[DEBUG Tree]: {frame.filename}:{frame.lineno} in function <{frame.name}>"
                    )
                self.tree.debug_print()
                print("=" * 100)
                return result

            return wrapper
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")


class CacheOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        handle: IOHandle,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.handles = [handle]
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
        self.handles.extend(other.handles)

    def __lt__(self, other: "CacheOperation"):
        return self.priority < other.priority


class HiCacheController_v2:
    def __init__(
        self,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        oracle: bool = False,
    ):
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
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue = PriorityQueue[CacheOperation]()

        self.ack_write_queue = Queue[IOHandle]()
        self.ack_load_queue = Queue[IOHandle]()

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

    def write(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor, handle: IOHandle
    ):
        self.write_queue.put(CacheOperation(host_indices, device_indices, handle))

    def load(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor, handle: IOHandle
    ):
        """
        Load KV caches from host memory to device memory.
        """
        self.load_queue.put(CacheOperation(host_indices, device_indices, handle))

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        torch.cuda.set_stream(self.write_stream)  # type: ignore
        while not self.stop_event.is_set():
            try:
                operation = self.write_queue.get(block=True, timeout=1)
                if not self.oracle:
                    self.mem_pool_host.write_page_all_layers(
                        operation.host_indices,
                        operation.device_indices,
                        self.mem_pool_device,
                    )
                    self.write_stream.synchronize()
                self.mem_pool_host.complete_io(operation.host_indices)
                for handle in operation.handles:
                    self.ack_write_queue.put(handle)
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
            for i in range(self.mem_pool_host.layer_num):
                if not self.oracle:
                    self.mem_pool_host.load_page_per_layer(
                        batch_operation.host_indices,
                        batch_operation.device_indices,
                        self.mem_pool_device,
                        i,
                    )
                    self.load_stream.synchronize()
                self.layer_done_counter.increment()

            self.mem_pool_host.complete_io(batch_operation.host_indices)
            for handle in batch_operation.handles:
                self.ack_load_queue.put(handle)


def make_tree(
    disabled: bool,
    use_hicache: bool,
    page_size: int,
    host_size: int,
    write_through_threshold: int,
):
    tree = RadixTreeCpp(
        disabled=disabled,
        host_size=host_size if use_hicache else None,
        page_size=page_size,
        write_through_threshold=write_through_threshold,
    )
    if os.environ.get("SGLANG_DEBUG_HIRADIX_CPP") == "1":
        return DebugTree(tree)  # for debugging purposes
    else:
        return tree


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
        self.disable = disable

        assert (
            enable_kv_cache_events is False
        ), "HiRadixCache does not support kv cache events yet"
        self.kv_cache = token_to_kv_pool.get_kvcache()

        # record the nodes with ongoing write through
        self.ongoing_write_through: Set[IOHandle] = set()
        # record the node segments with ongoing load back
        self.ongoing_load_back: Set[IOHandle] = set()
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 2
        )
        self.device = token_to_kv_pool.device
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.page_size = page_size

        self.tp_group = tp_cache_group
        self.load_cache_event = threading.Event()

        if not use_hicache:
            # TODO(dark): pass the second argument as `std::optional`
            self.tree = make_tree(
                disabled=disable,
                use_hicache=use_hicache,
                page_size=page_size,
                host_size=0,  # no host cache, this should be removed in the future
                write_through_threshold=self.write_through_threshold,
            )
            self.cache_controller = None
            return  # early return if hicache is not used

        if isinstance(self.kv_cache, MHATokenToKVPool):
            token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                page_size,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                page_size,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.cache_controller = HiCacheController_v2(
            token_to_kv_pool,
            token_to_kv_pool_host,
            page_size,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
        )
        self.tree = make_tree(
            disabled=disable,
            use_hicache=use_hicache,
            page_size=page_size,
            host_size=token_to_kv_pool_host.size,
            write_through_threshold=self.write_through_threshold,
        )

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
            int: Number of device indices that were already present in the tree before the insertion.
        """
        ongoing_write, length = self.tree.writing_through(key, value)
        if self.cache_controller is None:
            assert len(ongoing_write) == 0, "Implementation error"
            return length

        for io_handle, device_indices, host_indices in ongoing_write:
            self.cache_controller.write(device_indices, host_indices, io_handle)
            self.ongoing_write_through.add(io_handle)
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
        overall_len = len(token_ids)  # prefill + decode
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :overall_len]

        # NOTE: our C++ implementation don't need `token_ids` and `kv_indices` to be page-aligned
        # it will automatically align them, but length of them should be equal
        old_prefix_len = len(req.prefix_indices) // self.page_size * self.page_size
        new_prefix_len = self.insert(token_ids, kv_indices)

        # NOTE: kv_indices[:old_prefix_len] == req.prefix_indices
        assert old_prefix_len <= new_prefix_len, "Wrong prefix indices"

        # KVCache between old & new is newly generated, but already exists in the pool
        # we need to free this newly generated kv indices
        if old_prefix_len < new_prefix_len:
            self.token_to_kv_pool.free(kv_indices[old_prefix_len:new_prefix_len])

        # need to free the unaligned part, since it cannot be inserted into the radix tree
        if self.page_size != 1 and (  # unaligned tail only exists when page_size > 1
            (unaligned_len := overall_len % self.page_size) > 0
        ):
            # NOTE: sglang PagedAllocator support unaligned free (which will automatically align it)
            self.token_to_kv_pool.free(kv_indices[overall_len - unaligned_len :])

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)
        self.req_to_token_pool.free(req.req_pool_idx)

    def cache_unfinished_req(self, req: Req):
        """Cache request when it is unfinished."""
        assert req.req_pool_idx is not None
        token_ids = req.fill_ids
        prefill_len = len(token_ids)  # prefill only (maybe chunked)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :prefill_len]

        # NOTE: our C++ implementation don't need `token_ids` and `kv_indices` to be page-aligned
        # it will automatically align them, but length of them should be equal
        old_prefix_len = len(req.prefix_indices) // self.page_size * self.page_size
        new_prefix_len = self.insert(token_ids, kv_indices)

        # NOTE: kv_indices[:old_prefix_len] == req.prefix_indices
        assert old_prefix_len <= new_prefix_len, "Wrong prefix indices"

        # TODO(dark): optimize the `insert` and `match` (e.g. merge into 1 function)
        # The prefix indices need to updated to reuse the kv indices in the pool
        new_indices_vec, _, new_last_node, _ = self.tree.match_prefix(token_ids)
        new_indices = self.merge_tensor(new_indices_vec)
        assert new_prefix_len <= len(new_indices)

        # KVCache between old & new is newly generated, but already exists in the pool
        # we need to free this newly generated kv indices and reuse the indices in the pool
        if old_prefix_len < new_prefix_len:
            self.token_to_kv_pool.free(kv_indices[old_prefix_len:new_prefix_len])
            reused_indices = new_indices[old_prefix_len:new_prefix_len]
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, old_prefix_len:new_prefix_len
            ] = reused_indices

        if req.last_node != new_last_node:
            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)

        # NOTE: there might be unaligned tail, so we may need to append it
        assert len(new_indices) <= prefill_len < len(new_indices) + self.page_size
        if self.page_size != 1 and len(new_indices) < prefill_len:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def init_load_host(
        self,
        device_node: TreeNodeCpp,
        host_node: TreeNodeCpp,
        new_device_indices: torch.Tensor,
    ):
        if self.cache_controller is None:
            assert device_node == host_node, "Implementation error"
            return  # no host cache, nothing to load

        io_handle, host_indices_vec = self.tree.loading_onboard(
            device_node, host_node, new_device_indices
        )
        host_indices = self.merge_tensor(host_indices_vec)
        assert len(host_indices) == len(new_device_indices)
        self.cache_controller.load(
            device_indices=new_device_indices,
            host_indices=host_indices,
            handle=io_handle,  # NOTE: node is actually an int id
        )
        self.ongoing_load_back.add(io_handle)

    def ready_to_load_host(self):
        self.load_cache_event.set()

    def check_host_cache(self):
        self.writing_check()
        self.loading_check()

    def writing_check(self):
        if self.cache_controller is None:
            return  # no host cache, nothing to check
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
            self.tree.commit_writing_through(ack_id, True)
            self.ongoing_write_through.remove(ack_id)

    def loading_check(self):
        if self.cache_controller is None:
            return  # no host cache, nothing to check
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
            except Exception:
                break
            self.tree.commit_loading_onboard(ack_id, True)
            self.ongoing_load_back.remove(ack_id)

    def pretty_print(self):
        return self.tree.debug_print()
