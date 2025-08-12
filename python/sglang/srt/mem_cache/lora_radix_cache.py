"""Radix cache for LoRA. It's modified based on RadixCache with lora_id added to the key of nodes."""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
else:
    Req = Any  # Placeholder for Req type when not type checking


class LoRAKey:

    def __init__(self, lora_id: str, token_ids: List[int]):
        self.lora_id = (
            lora_id  # lora_id of adaptor, should be hash value of adaptor path
        )
        self.token_ids = token_ids  # token_ids of the key

    def __len__(self):
        return len(self.token_ids)


def get_child_key(key: LoRAKey):
    # Here the key of children dict is the hash of lora_id + str(token_ids[0])
    # So the child key can be matched only when lora_id and token_ids[0] are the same
    if key.lora_id is None:
        return hash(str(key.token_ids[0]))
    else:
        return hash(key.lora_id + str(key.token_ids[0]))


class LoRATreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(LoRATreeNode)
        self.parent: LoRATreeNode = None
        self.key: LoRAKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.id = LoRATreeNode.counter if id is None else id
        LoRATreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    def __lt__(self, other: "LoRATreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: LoRAKey, key1: LoRAKey):
    if key0.lora_id != key1.lora_id:
        raise ValueError(
            f"_key_match should be run on the same lora_id, but got key0.lora_id={key0.lora_id} != key1.lora_id={key1.lora_id}"
        )
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


class LoRARadixCache(BasePrefixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
    ):
        if page_size > 1:
            raise ValueError("LoRARadixCache currently only supports page_size = 1")

        if token_to_kv_pool_allocator is None:
            raise ValueError(
                "token_to_kv_pool_allocator is required to run LoraRadixCache"
            )

        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.device = self.token_to_kv_pool_allocator.device

        self.key_match_fn = _key_match
        self.get_child_key_fn = get_child_key
        self.reset()

    def reset(self):
        self.root_node = LoRATreeNode()
        self.root_node.key = LoRAKey(lora_id="", token_ids=[])
        self.root_node.value = None
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        raise ValueError(
            "LoRARadixCache needs both token ids and lora id as inputs for matching. Please use match_prefix_with_lora_id instead."
        )

    def match_prefix_with_lora_id(self, key: LoRAKey, **kwargs) -> MatchResult:
        """Find the matching prefix from the lora radix tree.
        Args:
            key: A LoRAKey to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, key: LoRAKey, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key.token_ids]
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req):
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        page_aligned_len = len(kv_indices)
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        lora_key = LoRAKey(lora_id=req.lora_id, token_ids=token_ids[:page_aligned_len])
        new_prefix_len = self.insert(lora_key, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        page_aligned_len = len(kv_indices)
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Radix Cache takes one ref in memory pool
        inserted_key = LoRAKey(lora_id=req.lora_id, token_ids=page_aligned_token_ids)
        new_prefix_len = self.insert(inserted_key, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node, _, _ = self.match_prefix_with_lora_id(inserted_key)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: LoRATreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: LoRATreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: LoRATreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: LoRATreeNode, key: LoRAKey):
        node.last_access_time = time.monotonic()

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = LoRAKey(lora_id=key.lora_id, token_ids=key.token_ids[prefix_len:])

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: LoRAKey, child: LoRATreeNode, split_len: int):
        # new_node -> child
        new_node = LoRATreeNode()
        key_split_1 = LoRAKey(lora_id=key.lora_id, token_ids=key.token_ids[:split_len])
        key_split_2 = LoRAKey(lora_id=key.lora_id, token_ids=key.token_ids[split_len:])
        new_node.children = {self.get_child_key_fn(key_split_2): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = key_split_1
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = key_split_2
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def _insert_helper(self, node: LoRATreeNode, key: LoRAKey, value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = LoRAKey(lora_id=key.lora_id, token_ids=key.token_ids[prefix_len:])
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = LoRATreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
        return total_prefix_length

    def _print_helper(self, node: LoRATreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list
