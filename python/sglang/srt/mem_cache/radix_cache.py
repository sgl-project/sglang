from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

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

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.mem_cache.utils import get_hash_str, hash_str_to_int64

# Fuzzy matching imports
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


def maybe_bigram_convert(
    is_eagle: bool,
    key: RadixKey,
    value: Optional[torch.Tensor] = None,
) -> Tuple[RadixKey, Optional[torch.Tensor]]:
    if is_eagle and not key.is_bigram:
        key.token_ids = convert_to_bigram_key(key.token_ids)
        key.is_bigram = True
        if value is not None:
            value = value[: len(key)]
    return key, value


def page_align_keys(key: list, page_size) -> list:
    if page_size == 1:
        return key
    page_aligned_len = len(key) // page_size * page_size
    return key[:page_aligned_len]


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy: EvictionStrategy = SLRUStrategy()

        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru'."
            )

        self.evictable_leaves = set()
        
        # Node registry: maps node_id -> TreeNode for non_prefix_store to resolve pool indices
        self._node_registry: Dict[int, TreeNode] = {}
        
        # Fuzzy matching support
        self.fuzzy_config: Optional[FuzzyMatchConfig] = None
        self.fuzzy_match_provider: Optional[FuzzyMatchProvider] = None
        self._fuzzy_cache_enabled: bool = False
        
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._record_all_cleared_event()
    
    def init_fuzzy_match(self, config: FuzzyMatchConfig, provider: FuzzyMatchProvider):
        """Initialize fuzzy matching support.
        
        Args:
            config: Fuzzy matching configuration.
            provider: Semantic provider instance.
        """
        self.fuzzy_config = config
        self.fuzzy_match_provider = provider
        self._fuzzy_cache_enabled = config.cache_fuzzy_results
        
        # Pass node registry to non_prefix_store for resolving pool indices
        if hasattr(self.fuzzy_match_provider, 'non_prefix_store'):
            self.fuzzy_match_provider.non_prefix_store.set_node_registry(self._node_registry)
        
        logger.info(
            f"Fuzzy matching initialized with provider={config.fuzzy_match_provider}, "
            f"min_match_length={config.fuzzy_min_match_length}, "
            f"cache_results={config.cache_fuzzy_results}"
        )

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        return maybe_bigram_convert(self.is_eagle, key, value)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                fuzzy_matched_len=0,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        # Step 1: Exact prefix matching
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        
        exact_matched_len = len(value)
        total_len = len(key.token_ids) if hasattr(key, 'token_ids') else len(key)
        
        # Step 2: If exact match is incomplete, try fuzzy matching
        fuzzy_matched_len = 0
        if exact_matched_len < total_len:
            logger.info(
                f"[FUZZY RADIX] match_prefix: exact={exact_matched_len}, fuzzy=0, "
                f"miss={total_len - exact_matched_len}, total={total_len}, attempting fuzzy..."
            )
            
            fuzzy_result = self.match_prefix_fuzzy(
                params=MatchPrefixParams(key=key),
                exact_matched_len=exact_matched_len,
            )
            
            if fuzzy_result is not None:
                fuzzy_matched_len = fuzzy_result.cached_token_count

                # Reserve realization slots up front. _correct_fuzzy_kv_rope
                # consumes from req.fuzzy_realized_locs; if we let it
                # allocate inside the forward pass, an alloc failure would
                # leave req_to_token_pool referencing donor slots after
                # state has already been committed here. Allocating now
                # makes the capacity check the first mutating step.
                # Skip the alloc when donor positions already match the
                # recipient's target (no copy needed); scattered N:M
                # results always need fresh slots.
                needs_realization = fuzzy_matched_len > 0 and (
                    getattr(fuzzy_result, "segments", None) is not None
                    or fuzzy_result.cached_start_pos != exact_matched_len
                )
                realized_locs = None
                if params.req is not None and needs_realization:
                    realized_locs = self.token_to_kv_pool_allocator.alloc(
                        fuzzy_matched_len
                    )
                    if realized_locs is None:
                        # No state mutated yet; safe to return exact-only.
                        logger.info(
                            f"[FUZZY RADIX] no pool capacity for "
                            f"{fuzzy_matched_len} fuzzy tokens; "
                            f"falling back to exact-only match"
                        )
                        return MatchResult(
                            device_indices=value,
                            last_device_node=last_node,
                            last_host_node=last_node,
                        )
                    # Free any leftover from a prior chunked/retracted
                    # match before stashing the new block.
                    prev_locs = getattr(params.req, "fuzzy_realized_locs", None)
                    if prev_locs is not None:
                        try:
                            self.token_to_kv_pool_allocator.free(prev_locs)
                        except Exception:
                            pass
                    params.req.fuzzy_realized_locs = realized_locs

                miss_len = total_len - exact_matched_len - fuzzy_matched_len
                logger.info(
                    f"[FUZZY RADIX] match_prefix: exact={exact_matched_len}, "
                    f"fuzzy={fuzzy_matched_len}, miss={miss_len}, total={total_len}, "
                    f"cached_start_pos={fuzzy_result.cached_start_pos}, "
                    f"realized_locs={'pre-allocated' if realized_locs is not None else 'none'}"
                )

                # The merged value still references the donor's slots;
                # the copy + RoPE correction into realized_locs runs in
                # model_runner._correct_fuzzy_kv_rope.
                fuzzy_kv_indices = torch.tensor(
                    fuzzy_result.kv_cache_indices,
                    device=value.device,
                    dtype=value.dtype,
                )

                # Provider contract: kv_cache_indices length must equal
                # cached_token_count. If it is shorter (e.g. an empty
                # tensor returned for a multi-segment match), the merged
                # device_indices below would mis-report the cached prefix
                # length to the scheduler, silently disabling KV reuse on
                # the fuzzy region.
                if len(fuzzy_kv_indices) != fuzzy_matched_len:
                    logger.warning(
                        f"[FUZZY RADIX] provider returned "
                        f"kv_cache_indices of length {len(fuzzy_kv_indices)} "
                        f"but cached_token_count={fuzzy_matched_len}; "
                        f"falling back to exact-only to avoid silent "
                        f"prefill of fuzzy region"
                    )
                    if realized_locs is not None:
                        try:
                            self.token_to_kv_pool_allocator.free(realized_locs)
                        except Exception:
                            pass
                        params.req.fuzzy_realized_locs = None
                    return MatchResult(
                        device_indices=value,
                        last_device_node=last_node,
                        last_host_node=last_node,
                    )

                if params.req is not None:
                    params.req.fuzzy_match_result = fuzzy_result

                # Lock the donor TreeNode so LRU eviction can't free the
                # slots in fuzzy_kv_indices before _correct_fuzzy_kv_rope
                # copies them. Released in cache_finished_req.
                if (
                    params.req is not None
                    and getattr(fuzzy_result, "donor_last_node_id", None) is not None
                ):
                    donor_node = self._node_registry.get(
                        fuzzy_result.donor_last_node_id
                    )
                    if donor_node is not None:
                        # Release any prior donor lock before acquiring
                        # a new one (chunked-prefill / resume case).
                        prev_donor = getattr(params.req, "fuzzy_donor_node", None)
                        if prev_donor is not None and prev_donor is not donor_node:
                            self.dec_lock_ref(prev_donor)
                        self.inc_lock_ref(donor_node)
                        params.req.fuzzy_donor_node = donor_node
                    else:
                        logger.warning(
                            f"[FUZZY RADIX] donor_last_node_id="
                            f"{fuzzy_result.donor_last_node_id} not in "
                            f"_node_registry; donor KV may be evicted "
                            f"mid-request"
                        )

                merged_value = torch.cat([value, fuzzy_kv_indices])
                return MatchResult(
                    device_indices=merged_value,
                    last_device_node=last_node,
                    last_host_node=last_node,
                    fuzzy_matched_len=fuzzy_result.cached_token_count,
                    cache_protected_len=exact_matched_len + fuzzy_result.cached_token_count,
                )
            else:
                logger.info(
                    f"[FUZZY RADIX] match_prefix: exact={exact_matched_len}, fuzzy=0, "
                    f"miss={total_len - exact_matched_len}, total={total_len}, fuzzy match failed"
                )
        else:
            logger.info(
                f"[FUZZY RADIX] match_prefix: exact={exact_matched_len}, fuzzy=0, "
                f"miss=0, total={total_len}"
            )
        # Return exact match result (with or without fuzzy matching)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            fuzzy_matched_len=0,
        )
    
    def match_prefix_fuzzy(
        self,
        params: MatchPrefixParams,
        exact_matched_len: int,
    ) -> Optional[FuzzyMatchResult]:
        """Perform fuzzy prefix matching when exact matching falls short.
        
        Called from match_prefix when exact_matched_len < total_len.
        """
        # Check minimum match length: need enough exact match to anchor fuzzy search
        if self.fuzzy_match_provider is None:
            return None
        if exact_matched_len > 0 and exact_matched_len < self.fuzzy_config.fuzzy_min_match_length:
            logger.info(
                f"[FUZZY RADIX] Skipping fuzzy match: exact_matched_len({exact_matched_len}) "
                f"< fuzzy_min_match_length({self.fuzzy_config.fuzzy_min_match_length})"
            )
            return None
        
        try:
            result = self.fuzzy_match_provider.match_on_prefix_miss(
                prompt_token_ids=params.key.token_ids,
                already_matched_len=exact_matched_len,
            )
            
            # Note: non_prefix_store entries don't need additional locking here.
            # Node references are resolved from the radix tree's node registry,
            # and the radix tree manages node lifecycle independently.
            
            if result is not None:
                logger.info(
                    f"[FUZZY RADIX] Fuzzy match success: cached={result.cached_token_count}, "
                    f"prompt={result.prompt_token_count}, offset={result.position_offset}"
                )
            else:
                logger.info(
                    f"[FUZZY RADIX] Fuzzy match failed: no suitable match found"
                )
            
            return result
            
        except Exception as e:
            logger.error(
                f"[FUZZY RADIX] Exception during fuzzy matching: {e}"
            )
            import traceback
            logger.error(traceback.format_exc())
            return None

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        prefix_len = self._insert_helper(
            self.root_node, key, value, priority, chunked
        )

        # Register the root node (always registered)
        self._register_node(self.root_node)

        # Resolve the deepest TreeNode for the just-inserted ``key`` via a
        # side-effect-free tree walk. The id is surfaced as
        # ``last_node_id`` so RadixCache.match_prefix's fuzzy path can
        # ``inc_lock_ref`` the donor TreeNode at match time and prevent
        # LRU eviction while a recipient request is consuming its KV.
        # Computing this here (rather than threading it back through
        # ``_insert_helper``'s return) preserves the helper's original
        # int-return ABI so subclasses overriding ``_insert_helper`` are
        # not forced to update their signature.
        last_node = self._find_leaf_for_key(key)

        return InsertResult(
            prefix_len=prefix_len,
            last_node_id=last_node.id if last_node is not None else None,
        )

    def _find_leaf_for_key(self, key: RadixKey) -> Optional[TreeNode]:
        """Walk the radix tree from root following ``key`` and return the leaf.

        Side-effect free: does not update access times, does not split
        nodes, does not allocate. Assumes ``key`` was just successfully
        inserted via ``_insert_helper`` so the path is expected to exist;
        returns the deepest reachable node if the trie shape is
        unexpected (e.g. a downstream subclass diverged from the base's
        insert semantics).

        Returns ``self.root_node`` when ``key`` is empty.
        """
        node = self.root_node
        if len(key) == 0:
            return node
        child_key = self.get_child_key_fn(key)
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            node = child
            key = key[prefix_len:]
            if prefix_len < len(child.key):
                # Partial match within child; insert would have split here
                # and the leaf we want is this node.
                break
            if len(key) > 0:
                child_key = self.get_child_key_fn(key)
        return node
    
    def _register_node(self, node: TreeNode):
        """Register a TreeNode in the node registry for reference resolution by non_prefix_store."""
        self._node_registry[node.id] = node

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        # Reclaim any pre-allocated realization slots that the forward
        # pass did not consume (aborted request, partial segments).
        leftover = getattr(req, "fuzzy_realized_locs", None)
        if leftover is not None:
            try:
                self.token_to_kv_pool_allocator.free(leftover)
            except Exception:
                pass
            req.fuzzy_realized_locs = None

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            donor_node = getattr(req, "fuzzy_donor_node", None)
            if donor_node is not None:
                self.dec_lock_ref(donor_node)
                req.fuzzy_donor_node = None
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Cache to non_prefix_store BEFORE freeing indices, since non_prefix_store
        # saves pool indices that must still be valid.
        if self._fuzzy_cache_enabled:
            try:
                cache_start = getattr(req, 'cache_start_pos', None)
                cache_end = getattr(req, 'cache_end_pos', None)

                if cache_start is None:
                    cache_start = 0
                if cache_end is None or cache_end == -1:
                    cache_end = len(token_ids)

                self.fuzzy_match_provider.cache_on_request_finished(
                    request=req,
                    token_ids=token_ids,
                    kv_cache=kv_indices,
                    cache_start_pos=cache_start,
                    cache_end_pos=cache_end,
                    radix_tree=self,
                )
            except Exception as e:
                logger.warning(f"[FUZZY RADIX] cache_on_request_finished failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            new_prefix_len = result.prefix_len
            # Hand the inserted node id to the fuzzy provider so
            # subsequent donor lookups can resolve a stable NodeRef.
            if (
                self._fuzzy_cache_enabled
                and result.last_node_id is not None
                and self.fuzzy_match_provider is not None
            ):
                try:
                    self.fuzzy_match_provider.on_donor_inserted(
                        request=req,
                        donor_last_node_id=result.last_node_id,
                    )
                except Exception as e:
                    logger.debug(f"[FUZZY RADIX] on_donor_inserted failed: {e}")
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Release the donor lock_ref acquired in match_prefix.
        donor_node = getattr(req, "fuzzy_donor_node", None)
        if donor_node is not None:
            self.dec_lock_ref(donor_node)
            req.fuzzy_donor_node = None

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        self._register_node(new_node)
        self._register_node(child)  # child's value was modified, re-register
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Update NodeRefs in non_prefix_store to reflect the split.
        # child.id is the old node (now holds suffix), new_node.id is the new node (holds prefix).
        if self.fuzzy_match_provider is not None and hasattr(self.fuzzy_match_provider, 'non_prefix_store'):
            self.fuzzy_match_provider.non_prefix_store.update_node_refs_on_split(
                old_node_id=child.id,
                new_node_id=new_node.id,
                split_len=split_len,
            )

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ) -> int:
        """Insert ``key``/``value`` into the radix tree rooted at ``node``.

        Returns the total prefix length matched against existing nodes.
        The deepest inserted/touched node is reachable via
        ``_find_leaf_for_key(key)`` from ``insert()``; we deliberately do
        NOT thread it back through the return value here so the helper's
        original int-return ABI stays stable for subclasses that override
        ``_insert_helper``.
        """
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._register_node(new_node)
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
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
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        # Mirror _register_node: drop on eviction.
        self._node_registry.pop(node.id, None)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

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

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
