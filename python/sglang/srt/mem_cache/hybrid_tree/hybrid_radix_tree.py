"""
Unified Hybrid Radix Tree for managing multiple cache types.
"""

import logging
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
)
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE

from .cache_component import CacheComponent, FullComponent, MambaComponent
from .tree_node import TreeNode, get_last_access_time

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams


class LockHandle:
    """Handle for lock/unlock operations."""

    def __init__(self):
        self.locked: Dict[str, List[TreeNode]] = {}


class HybridRadixTree(BasePrefixCache):
    """Unified Radix Tree supporting multiple cache components."""

    def __init__(self, params: "CacheInitParams", page_size: Optional[int] = None):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer

        if page_size is None:
            page_size = params.page_size
        self.page_size = page_size

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        self.components: Dict[str, CacheComponent] = {}
        self.primary: Optional[CacheComponent] = None

        if page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=page_size)

        self.root = None
        self.reset()
        self.register_component(FullComponent())
        self.register_component(MambaComponent())

    def reset(self) -> None:
        """Reset the radix tree to initial state."""
        self.root = TreeNode()
        self.root.key = RadixKey([], None)

        # Reset all components
        for component in self.components.values():
            root_data = component.get_component_data(self.root, create=True)
            root_data.lock_ref = 1

            if component.is_primary:
                root_data.value = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            else:
                root_data.value = None

            component.evictable_size = 0
            component.protected_size = 0
            component.lru_list = type(component.lru_list)(component.name)

    def register_component(self, component: CacheComponent):
        """Register a cache component and set up its allocator.
        
        If a component with the same name is already registered, skip registration.
        """
        # Skip if already registered
        if component.name in self.components:
            return

        self.components[component.name] = component

        # Set appropriate allocator for each component type
        if component.is_primary:
            allocator = self.token_to_kv_pool_allocator
            assert self.primary is None, "Only one primary component allowed"
            self.primary = component
        else:
            allocator = (
                self.req_to_token_pool.mamba_pool
                if self.req_to_token_pool
                and hasattr(self.req_to_token_pool, "mamba_pool")
                else None
            )

        component.setup(self, allocator)

        # Initialize root data for this component
        if self.root:
            root_data = component.get_component_data(self.root, create=True)
            root_data.lock_ref = 1

            if component.is_primary:
                root_data.value = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            else:
                root_data.value = None

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Find the longest prefix that satisfies all component constraints."""
        if len(key) == 0:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root,
                last_host_node=self.root,
            )

        def _update_match_context(node: TreeNode, match_context: dict, init: bool = False):
            for comp in self.components.values():
                comp.update_prefix_match_context(node, match_context, init=init)

        node = self.root
        primary_values = []
        best_match_len = 0
        best_match_node = node
        child_key = self.get_child_key_fn(key)
        match_context = {}
        _update_match_context(node, match_context, init=True)

        # Traverse tree to find longest matching prefix
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # Check if current node satisfies all constraints
            if self._check_all_constraints(node, match_context, child):
                best_match_len = len(primary_values)
                best_match_node = node
                _update_match_context(child, match_context)

            # Match child's key with remaining search key
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                # Partial match: split node and stop
                new_node = self._split_node(child, prefix_len)
                primary_values.append(self._get_primary_value(new_node))
                _update_match_context(new_node, match_context)

                node = new_node
                break
            else:
                # Full match: continue traversing
                primary_values.append(self._get_primary_value(child))
                _update_match_context(child, match_context)

                node = child
                key = key[prefix_len:]
                if len(key):
                    child_key = self.get_child_key_fn(key)

        # Final check at last traversed node
        if self._check_all_constraints(node, match_context, None):
            best_match_len = len(primary_values)
            best_match_node = node

        # Update LRU for matched path
        self._update_lru_for_match(best_match_node)

        # Allow components to perform post-match operations
        req = kwargs.get("req", None)
        cow_mamba = kwargs.get("cow_mamba", False)

        if req is not None and (cow_mamba or "cow_mamba" not in kwargs):
            for comp in self.components.values():
                comp.on_match_complete(best_match_node, req)

        # Build result indices
        if best_match_len > 0:
            result_indices = torch.cat(primary_values[:best_match_len])
        else:
            result_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

        # Calculate mamba branching point (for mamba state management)
        mamba_branching_seqlen = None
        if len(primary_values) > best_match_len:
            total_len = sum(len(v) for v in primary_values)
            fla_chunk_aligned = (total_len // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
            mamba_branching_seqlen = (
                fla_chunk_aligned if fla_chunk_aligned > 0 else None
            )

        logger.info(
            f"Match prefix result for req {req.rid if req else 'None'}: match prefix: {len(result_indices.tolist())}"
        )
        return MatchResult(
            device_indices=result_indices,
            last_device_node=best_match_node,
            last_host_node=best_match_node,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def insert(
        self, key: RadixKey, values: Dict[str, torch.Tensor]
    ) -> Tuple[int, Dict[str, bool]]:
        """Insert component values into radix tree.

        Returns:
            (matched_prefix_len, component_existed) where component_existed indicates
            if each component's value already existed in the tree
        """

        node = self.root
        node.last_access_time = get_last_access_time()

        if len(key) == 0:
            return 0, {}

        child_key = self.get_child_key_fn(key)
        matched_prefix_len = 0

        # Traverse tree along key
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()

            # Update LRU for each component
            for comp in self.components.values():
                if comp.lru_list.in_list(node):
                    comp.lru_list.reset_node_mru(node)

            prefix_len = self.key_match_fn(node.key, key)

            # Split node if partial match
            if prefix_len < len(node.key):
                node = self._split_node(node, prefix_len)

            # Advance key and values
            matched_prefix_len += prefix_len
            key = key[prefix_len:]

            # TODO: Handle swa case
            values = {
                name: val[prefix_len:] if self.components[name].is_primary else val
                for name, val in values.items()
            }

            if len(key) > 0:
                child_key = self.get_child_key_fn(key)

        # Insert values at final node
        component_existed = {}
        if len(key) > 0:
            # Create new leaf
            new_leaf = TreeNode()
            new_leaf.parent = node
            new_leaf.key = key
            node.children[child_key] = new_leaf

            for comp_name, comp in self.components.items():
                comp.on_insert_leaf(new_leaf, values[comp_name])
                component_existed[comp_name] = False
        else:
            # Update existing node
            node.last_access_time = get_last_access_time()
            for comp_name, comp in self.components.items():
                data = comp.get_component_data(node)
                existed = data is not None and not data.is_tombstone()

                if not existed:
                    comp.on_insert_leaf(node, values[comp_name])
                elif comp.lru_list.in_list(node):
                    comp.lru_list.reset_node_mru(node)

                component_existed[comp_name] = existed

        return matched_prefix_len, component_existed

    def evict_component(self, component_name: str, num_tokens: int) -> int:
        """Evict from specified component with cascading."""
        if self.disable or num_tokens <= 0:
            return 0

        component = self.components[component_name]
        return component.evict_cascade(num_tokens)

    def evict(self, num_tokens: int) -> None:
        """Evict tokens from full (primary) component."""
        return self.evict_component(self.primary.name, num_tokens)

    def evict_mamba(self, num_tokens: int) -> None:
        """Evict tokens from mamba component (for compatibility)."""
        if "mamba" in self.components:
            self.evict_component("mamba", num_tokens)

    def cache_finished_req(self, req: "Req", is_insert: bool = True) -> None:
        """Cache finished request into radix tree."""
        # Cleanup aborted request
        if req.req_pool_idx is None:
            if req.mamba_pool_idx is not None:
                self.components["mamba"].free_value(
                    req.mamba_pool_idx.unsqueeze(-1), update_accounting=False
                )
                req.mamba_pool_idx = None
            return

        kv_committed_len = req.pop_committed_kv_cache()

        # Free cache if tree is disabled
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Insert into tree or just free
        component_existed = {}
        if is_insert:
            cache_len = (
                req.mamba_last_track_seqlen
                if self.enable_mamba_extra_buffer
                else len(token_ids)
            )
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            req._cache_len = cache_len
            component_values = {}
            for comp_name, comp in self.components.items():
                component_values[comp_name] = comp.prepare_component_value(
                    req, is_finished=True
                )

            page_aligned_len = len(component_values[self.primary.name])
            page_aligned_token_ids = token_ids[:page_aligned_len]
            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            radix_key = RadixKey(page_aligned_token_ids, req.extra_key)
            new_prefix_len, component_existed = self.insert(radix_key, component_values)

            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])
            component_existed = {
                name: True
                for name in self.components
                if not self.components[name].is_primary
            }

        # TODO: Optimize the mamba-related logic here
        mamba_exist = component_existed.get("mamba", False)
        mamba_ping_pong_track_buffer_to_keep = None
        if self.enable_mamba_extra_buffer and not mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
        free_mamba_cache = self.enable_mamba_extra_buffer or mamba_exist

        self.req_to_token_pool.free(
            req.req_pool_idx,
            free_mamba_cache=free_mamba_cache,
            mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
        )
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: "Req", chunked=False) -> None:
        """Cache unfinished request (chunked prefill scenario)."""
        logger.info(f"Cache unfinished request, {req.rid}, chunked: {chunked}")
        token_ids = req.fill_ids
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )

        if self.disable or cache_len is None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        kv_indices_full = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        kv_indices_to_cache = kv_indices_full[:cache_len]
        req._cache_len = cache_len

        # Prepare and insert component values
        component_values = {}
        for comp_name, comp in self.components.items():
            component_values[comp_name] = comp.prepare_component_value(
                req, is_finished=False
            )

        page_aligned_kv_indices = component_values[self.primary.name]
        page_aligned_len = len(page_aligned_kv_indices)
        page_aligned_token_ids = token_ids[:page_aligned_len]

        assert page_aligned_len == len(
            kv_indices_to_cache
        ), f"Page alignment mismatch: {page_aligned_len=}, {len(kv_indices_to_cache)=}, {cache_len=}, {self.page_size=}"

        radix_key = RadixKey(page_aligned_token_ids, req.extra_key)
        new_prefix_len, component_existed = self.insert(radix_key, component_values)

        # Free duplicate KV cache already in tree
        self.token_to_kv_pool_allocator.free(
            kv_indices_to_cache[req.cache_protected_len : new_prefix_len]
        )

        # Cleanup secondary components' values if already existed (e.g., forked mamba state)
        for comp_name, comp in self.components.items():
            if not comp.is_primary:
                existed = component_existed.get(comp_name, False)
                if existed:
                    comp.free_value(
                        component_values[comp_name], update_accounting=False
                    )

        # Re-match to get updated cached indices
        match_result = self.match_prefix(radix_key)
        cached_indices = match_result.device_indices
        new_last_node = match_result.last_device_node

        assert (
            req.cache_protected_len <= len(cached_indices) + self.page_size - 1
        ), f"Protected length out of bounds: {req.cache_protected_len=}, {len(cached_indices)=}"
        assert new_prefix_len <= len(
            cached_indices
        ), f"Prefix length mismatch: {new_prefix_len=}, {len(cached_indices)=}"

        # Update req_to_token_pool with cached indices
        if req.cache_protected_len < len(cached_indices):
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(req.cache_protected_len, len(cached_indices))),
                cached_indices[req.cache_protected_len :],
            )

        # Update locks and request state
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        req.prefix_indices = torch.cat(
            [cached_indices, kv_indices_full[len(cached_indices) :]]
        )
        req.cache_protected_len = len(cached_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

    # ============================================================================
    # Internal Helper Methods
    # ============================================================================

    def inc_lock_ref(self, node: TreeNode):
        """Increment lock references for all components.

        Returns:
            LockHandle containing all locked nodes for later dec_lock_ref call
        """
        if self.disable:
            return None

        handle = LockHandle()

        for comp in self.components.values():
            nodes_to_lock = comp.get_lock_range(node, self.root)
            for n in nodes_to_lock:
                data = comp.get_component_data(n, create=True)
                # Move from evictable to protected
                # Note: tombstone nodes have value_len=0, which doesn't affect evictable_size.
                if data.lock_ref == 0:
                    value_len = len(data.value) if not data.is_tombstone() else 0
                    comp.evictable_size -= value_len
                    comp.protected_size += value_len
                data.lock_ref += 1
            handle.locked[comp.name] = nodes_to_lock

        return handle

    def dec_lock_ref(self, handle_or_node, swa_uuid_for_lock: Optional[str] = None):
        """Decrement lock references.

        Args:
            handle_or_node: Either LockHandle (preferred) or TreeNode (for backward compatibility)
            swa_uuid_for_lock: Optional SWA UUID (reserved for future use)
        """
        if self.disable or handle_or_node is None:
            return

        if isinstance(handle_or_node, LockHandle):
            # Use LockHandle: unlock exactly the nodes that were locked
            for comp_name, locked_nodes in handle_or_node.locked.items():
                comp = self.components[comp_name]
                for n in locked_nodes:
                    data = comp.get_component_data(n)
                    if data and data.lock_ref > 0:
                        data.lock_ref -= 1
                        # Move from protected to evictable
                        # Use the same logic as inc_lock_ref for symmetry
                        if data.lock_ref == 0:
                            value_len = len(data.value) if not data.is_tombstone() else 0
                            comp.evictable_size += value_len
                            comp.protected_size -= value_len
        else:
            # Backward compatibility: TreeNode passed directly (recompute lock range)
            node = handle_or_node
            for comp in self.components.values():
                nodes_to_unlock = comp.get_lock_range(node, self.root)
                for n in nodes_to_unlock:
                    data = comp.get_component_data(n)
                    if data and data.lock_ref > 0:
                        data.lock_ref -= 1
                        # Use the same logic as inc_lock_ref for symmetry
                        if data.lock_ref == 0:
                            value_len = len(data.value) if not data.is_tombstone() else 0
                            comp.evictable_size += value_len
                            comp.protected_size -= value_len

    def _check_all_constraints(self, node: TreeNode, context: dict, child: Optional[TreeNode] = None) -> bool:
        """Check if node satisfies all component constraints."""
        return all(
            comp.check_prefix_match_constraints(node, context, child)
            for comp in self.components.values()
        )

    def _get_primary_value(self, node: TreeNode) -> torch.Tensor:
        """Get primary component's value."""
        data = self.primary.get_component_data(node)
        return (
            data.value
            if data
            else torch.empty((0,), dtype=torch.int64, device=self.device)
        )

    def _update_lru_for_match(self, node: TreeNode):
        """Update LRU lists and access times after match."""
        for comp in self.components.values():
            comp.lru_list.reset_node_and_parents_mru(node, self.root, component=comp)

        cur_time = get_last_access_time()
        while node:
            node.last_access_time = cur_time
            cur_time -= 0.00001
            node = node.parent

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        """Split a node at the specified position.

        Creates a new parent node and adjusts the tree structure.
        Component-specific logic (data splitting, LRU updates) is handled
        by on_insert_split hook called from insert method.
        """
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(child.key[split_len:]): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]

        child.parent = new_node
        child.key = child.key[split_len:]
        child.last_access_time = get_last_access_time()

        new_node.parent.children[self.get_child_key_fn(new_node.key)] = new_node

        for comp in self.components.values():
            comp.on_insert_split_node(child, new_node, split_len)

        logger.info(f"Split node: {id(child)} -> {id(new_node)}")
        return new_node

    def _delete_node_cascade(self, node: TreeNode):
        """Delete leaf node and recursively cleanup tombstone parents."""
        for comp in self.components.values():
            if comp.lru_list.in_list(node):
                comp.lru_list.remove_node(node)

        key = self.get_child_key_fn(node.key)
        del node.parent.children[key]

        self._cleanup_tombstone_parents(node.parent)

    def _cleanup_tombstone_parents(self, node: TreeNode):
        """Recursively cleanup childless parent nodes where all secondary components are tombstones.

        Cleanup conditions:
        1. Node has no children (became childless after deleting a child)
        2. All secondary components have no value (primary can still have value)
        3. Node is not locked by any component

        This maintains the invariant: leaf nodes must have values for all components.
        """
        while node != self.root and len(node.children) == 0:
            # Check if all secondary components are tombstones
            all_secondary_tombstones = all(
                comp.get_component_data(node) is None
                or comp.get_component_data(node).value is None
                for comp in self.components.values()
                if not comp.is_primary
            )
            if not all_secondary_tombstones:
                break

            # Check if node is locked
            any_locked = any(comp.is_locked(node) for comp in self.components.values())
            if any_locked:
                break

            # Delete node and continue to parent
            parent = node.parent
            for comp in self.components.values():
                comp.free_value(node)
                if comp.lru_list.in_list(node):
                    comp.lru_list.remove_node(node)

            key = self.get_child_key_fn(node.key)
            del parent.children[key]
            node = parent

    # ============================================================================
    # BasePrefixCache Interface Implementation
    # ============================================================================

    def evictable_size(self) -> Tuple[int, int]:
        """Return (full_evictable, mamba_evictable) for compatibility."""
        raise NotImplementedError(
            "Use full_evictable_size() and mamba_evictable_size() instead"
        )

    def full_evictable_size(self) -> int:
        """Return evictable size of primary (full) component."""
        return self.primary.evictable_size if self.primary else 0

    def mamba_evictable_size(self) -> int:
        """Return evictable size of mamba component (for compatibility)."""
        return (
            self.components["mamba"].evictable_size if "mamba" in self.components else 0
        )

    def full_lru_list_evictable_size(self) -> int:
        """Return evictable size by iterating full LRU list (expensive, for debug)."""
        if not self.primary:
            return 0
        return self.primary.lru_list.sanity_check_evictable_size(self.primary)

    def mamba_lru_list_evictable_size(self) -> int:
        """Return evictable size by iterating mamba LRU list (expensive, for debug)."""
        if "mamba" not in self.components:
            return 0
        return self.components["mamba"].lru_list.sanity_check_evictable_size(
            self.components["mamba"]
        )

    def protected_size(self) -> Tuple[int, int]:
        """Return (full_protected, mamba_protected) for compatibility."""
        raise NotImplementedError(
            "Use full_protected_size() and mamba_protected_size() instead"
        )

    def full_protected_size(self) -> int:
        """Return protected size of primary (full) component."""
        return self.primary.protected_size if self.primary else 0

    def mamba_protected_size(self) -> int:
        """Return protected size of mamba component (for compatibility)."""
        return (
            self.components["mamba"].protected_size if "mamba" in self.components else 0
        )

    def all_values_flatten(self) -> torch.Tensor:
        """Return all full (primary) cache values flattened (for compatibility)."""
        values = []

        def _dfs_helper(node: TreeNode):
            for child in node.children.values():
                if self.primary:
                    data = self.primary.get_component_data(child)
                    if data and not data.is_tombstone():
                        values.append(data.value)
                _dfs_helper(child)

        _dfs_helper(self.root)
        return (
            torch.cat(values)
            if len(values) > 0
            else torch.tensor([], dtype=torch.int64, device=self.device)
        )

    def all_mamba_values_flatten(self) -> torch.Tensor:
        """Return all mamba cache values flattened (for compatibility)."""
        values = []

        def _dfs_helper(node: TreeNode):
            if "mamba" in self.components:
                data = self.components["mamba"].get_component_data(node)
                if data and not data.is_tombstone():
                    values.append(data.value)
            for child in node.children.values():
                _dfs_helper(child)

        _dfs_helper(self.root)
        return (
            torch.cat(values)
            if len(values) > 0
            else torch.tensor([], dtype=torch.int64, device=self.device)
        )

    def total_size(self) -> Tuple[int, int]:
        """Return (full_size, mamba_size)."""
        full_size = 0
        mamba_size = 0

        def _dfs_helper(node: TreeNode):
            nonlocal full_size, mamba_size

            if self.primary:
                full_data = self.primary.get_component_data(node)
                if full_data and not full_data.is_tombstone():
                    full_size += len(full_data.value)

            if "mamba" in self.components:
                mamba_data = self.components["mamba"].get_component_data(node)
                if mamba_data and not mamba_data.is_tombstone():
                    mamba_size += len(mamba_data.value)

            for child in node.children.values():
                _dfs_helper(child)

        _dfs_helper(self.root)
        return full_size, mamba_size

    def sanity_check(self) -> None:
        """Sanity check the tree."""
        if self.disable:
            return

        for comp in self.components.values():
            try:
                comp.sanity_check_lru(self)
            except Exception as e:
                from sglang.srt.distributed import get_tensor_model_parallel_rank

                if get_tensor_model_parallel_rank() == 0:
                    msg = f"Hybrid Radix tree sanity check failed for {comp.name}: {e}"
                    logger.error(msg)
                    self.pretty_print()
                    raise Exception(msg)

    def pretty_print(self) -> None:
        """Print the tree structure."""

        def _print_helper(node: TreeNode, indent: int):
            info = f"[{id(node)}] key_len={len(node.key) if node.key else 0}"
            for comp_name, comp in self.components.items():
                data = comp.get_component_data(node)
                if data:
                    val_len = len(data.value) if not data.is_tombstone() else 0
                    info += f" {comp_name}={val_len}(lock={data.lock_ref})"
            print(" " * indent + info)

            for child in node.children.values():
                _print_helper(child, indent + 2)

        _print_helper(self.root, 0)
        full_size, mamba_size = self.total_size()
        print(f"#full_tokens: {full_size}, #mamba_num: {mamba_size}")
