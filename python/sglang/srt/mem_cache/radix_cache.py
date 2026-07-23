from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

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
import math
import sys
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

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
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.session_radix_cache import SessionRadixCacheMixin
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
from sglang.srt.mem_cache.utils import (
    get_eviction_strategy,
    get_hash_str,
    split_node_hash_value,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
    ):
        # token ids sequence (raw ints in both modes)
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram
        # Optional cap on raw tokens: behave as if token_ids were sliced to
        # token_ids[:limit], without the O(n) copy. None = use all tokens.
        self.limit = limit

    def _raw_len(self) -> int:
        n = len(self.token_ids)
        if self.limit is not None and self.limit < n:
            return self.limit
        return n

    def raw_token_ids(self) -> array:
        """token_ids honoring `limit` (copies only when capped)."""
        n = self._raw_len()
        t = self.token_ids
        return t if n == len(t) else t[:n]

    def __len__(self) -> int:
        n = self._raw_len()
        if self.is_bigram:
            return n - 1 if n > 0 else 0
        return n

    # TODO(Jialin): vectorize with numpy without PyLong boxing
    def __iter__(self) -> Iterator:
        t = self.token_ids
        n = self._raw_len()
        if self.is_bigram:
            for i in range(n - 1 if n > 0 else 0):
                yield (t[i], t[i + 1])
        elif n == len(t):
            yield from t
        else:
            for i in range(n):
                yield t[i]

    def __getitem__(self, idx: Union[int, slice]) -> RadixKey:
        # Normalize int -> 1-element slice so the rest handles one shape.
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigrams [start, stop) span raw tokens [start, stop + 1);
            # empty slice -> empty raw tokens (not a dangling boundary token).
            raw = self.token_ids[start : stop + 1] if stop > start else array("q")
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> RadixKey:
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: RadixKey) -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )

    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(len(t0), len(t1))

        # Exponential search for the first diverging token: gallop in doubling
        # windows (one C-level slice compare each), then binary-search the window
        # holding the divergence -- no per-token Python loop on long shared prefixes.
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2

        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        matched_tokens = min(matched_tokens, len(self), len(other))
        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size

    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str] = None) -> str:
        """SHA256 for logical units [start, end); bigram mode feeds overlapping (t_i, t_{i+1}) byte pairs."""
        hash_value = get_hash_str(self[start:end], prior_hash)
        assert isinstance(hash_value, str)
        return hash_value


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
        self.write_through_pending_id: Optional[int] = None
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

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


class RadixCache(SessionRadixCacheMixin, KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.enable_session_radix_cache = params.enable_session_radix_cache
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        self.evictable_leaves = set()

        # Mixed-KV: tree caches the shared HP-prefix pool + quant slots;
        # per-request HP-recent slots are excluded by ``_mixed_kv_tail_to_drop``
        # (correctness — they alias across requests) and ``match_prefix`` is
        # capped via ``_mixed_kv_match_cap_overhead``.
        self._mixed_kv_enabled = False
        self._mixed_kv_hp_prefix_tokens = 0
        self._mixed_kv_match_cap_overhead = 0
        if self.token_to_kv_pool_allocator is not None:
            kvc_getter = getattr(self.token_to_kv_pool_allocator, "get_kvcache", None)
            kvc = kvc_getter() if kvc_getter is not None else None
            if isinstance(kvc, UnifiedInt2HPKVPool) and kvc.mixed_kv_enabled():
                self._mixed_kv_enabled = True
                self._mixed_kv_hp_prefix_tokens = int(kvc.hp_prefix_tokens)
                flush_overflow = max(0, int(kvc.flush_interval) - 1)
                self._mixed_kv_match_cap_overhead = (
                    int(kvc.hp_recent_tokens) + flush_overflow
                )

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
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._reset_session_radix_state()
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

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
            cached prefix (may be length 0).
            ``last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects
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
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        if self.disable or len(key) == 0:
            return self._empty_match_result

        key = key.page_aligned(self.page_size)

        # Mixed-KV: cap the match so positions in the request's HP-recent
        # window are NEVER returned from cache. A full match would cover
        # those positions with quant slots from a deeper inserter,
        # breaking the HP-precision invariant for the recent window.
        # Cacheable positions = [0, max(hp_prefix, len - hp_recent_overflow))
        # — for very short requests where the post-prefix region is
        # entirely HP-recent, only the HP-prefix portion is cacheable;
        # for long requests, everything before the trailing HP-recent
        # band is cacheable. Page-aligned to keep the radix tree happy.
        #
        # Internal callers (``cache_unfinished_req``'s post-insert
        # sibling-coverage match) pass ``bypass_mixed_kv_cap=True``: that
        # call needs the FULL match length to stay consistent with the
        # ``cache_protected_len`` admission set. Capping it desyncs the
        # ``prefix_indices`` reconstruction at line ~725 (Python slicing
        # silently truncates ``new_indices[:cache_protected_len]`` when
        # ``len(new_indices) < cache_protected_len``), which silently
        # drops slot ids and leaks them (the leak detector flags it as
        # ~32-slot per-request residue under stress).
        if self._mixed_kv_enabled and len(key) > 0 and not params.bypass_mixed_kv_cap:
            n = len(key)
            cap = min(
                n,
                max(
                    self._mixed_kv_hp_prefix_tokens,
                    n - self._mixed_kv_match_cap_overhead,
                ),
            )
            if self.page_size > 1:
                cap = cap // self.page_size * self.page_size
            if cap < n:
                key = key[:cap]

        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len, last_node = self._insert_helper(
            self.root_node, key, value, priority, chunked
        )
        return InsertResult(prefix_len=prefix_len, last_device_node=last_node)

    def _with_mixed_quant_slack(self, req: Req, indices: torch.Tensor) -> torch.Tensor:
        """Fold request-owned quant-page slack slots (mixed HP+int2 KV) into
        the indices being freed so the atomic quant page is returned whole."""
        slack = req.mixed_kv_quant_slack_indices
        if slack.numel() == 0:
            return indices

        req.mixed_kv_quant_slack_indices = torch.empty((0,), dtype=torch.int64)
        req.mixed_kv_quant_slack_cutoff_len = None
        return torch.cat([indices.to(torch.int64), slack.to(indices.device)])

    def _mixed_kv_slack_insert_limit(self, req: Req, key_len: int) -> int:
        """Keep radix ownership below any request-owned partial quant page."""
        cutoff_len = req.mixed_kv_quant_slack_cutoff_len
        if cutoff_len is None:
            return key_len
        return max(0, min(key_len, int(cutoff_len)))

    def _committed_fill_ids(self, req: Req) -> list[int]:
        committed_len = min(int(req.kv_committed_len), len(req.fill_ids))
        return req.fill_ids[:committed_len]

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        if self.disable:
            # The protected prefix is not this req's to free.
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.cache_protected_len : kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free(
                self._with_mixed_quant_slack(req, kv_indices)
            )
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        protected_len = min(req.cache_protected_len, len(kv_indices))
        if protected_len != req.cache_protected_len:
            req.cache_protected_len = protected_len

        if self._mixed_kv_enabled:
            # Mixed-KV: the radix tree is populated ONLY by
            # ``cache_unfinished_req`` (called once after each request's
            # prefill via the scheduler output-processor mixin).
            # ``cache_finished_req`` deliberately returns early for both
            # is_insert=True (natural finish) and is_insert=False (retract):
            # finished/retracted requests just free their tail slots and
            # ``dec_lock_ref`` — they do NOT extend the tree.
            #
            # The naive "insert + bypass-cap re-match + inc_lock_ref(new)
            # → dec_lock_ref(old)" pattern is unsafe under mixed-KV +
            # retract: the new leaf has ``lock_ref=0`` immediately and
            # gets evicted under concurrent retract memory pressure,
            # freeing slot ids that other live requests' ``req_to_token``
            # mappings still reference → corrupted reads → gibberish
            # (~0.5% rate at this batch size, confirmed against the
            # mixed-pool reference implementation which has the same
            # early-return).
            self.token_to_kv_pool_allocator.free(
                self._with_mixed_quant_slack(req, kv_indices[protected_len:])
            )
            self.dec_lock_ref(req.last_node)
            return

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            session_leaf = result.last_device_node
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len]
            )
        else:
            session_leaf = None
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        self._tag_session_leaf(req, radix_key, node=session_leaf)

        # Remove req slot release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if self._mixed_kv_enabled:
            return self._cache_unfinished_req_mixed_kv(req, chunked=chunked)

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

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
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

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

        self._tag_session_leaf(req, radix_key, node=new_last_node)

    def _cache_unfinished_req_mixed_kv(self, req: Req, chunked: bool = False):
        """Mixed HP+int2 KV variant of :meth:`cache_unfinished_req`.

        Differences from the plain path, all required for correctness of the
        two-tier pool:

        * Only *committed* fill ids enter the tree (``kv_committed_len``).
        * The per-request HP-recent tail never enters the tree
          (``_mixed_kv_tail_to_drop``) — those slot ids alias across requests.
        * Every free/insert boundary is clamped at the request-owned partial
          quant page (``_mixed_kv_slack_insert_limit``); crossing it would
          hand a page to ``free_pages`` while this request still owns part of
          it via ``mixed_kv_quant_slack_indices`` (double-issue → double-free).
        * ``cache_protected_len`` only ever advances (monotonic); regressing
          it causes a cross-request double-free at ``cache_finished_req``.
        """
        token_ids = self._committed_fill_ids(req)
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        protected_len = min(req.cache_protected_len, len(kv_indices))
        if protected_len != req.cache_protected_len:
            req.cache_protected_len = protected_len

        full_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(full_key)].to(dtype=torch.int64, copy=True)

        # Drop the HP-recent tail before insert; the live request still owns
        # those HP slots (they stay addressable via the ``torch.cat`` branch
        # below that rebuilds ``req.prefix_indices``), so they are not freed.
        insert_len = len(full_key) - self._mixed_kv_tail_to_drop(len(full_key))
        insert_len = self._mixed_kv_slack_insert_limit(req, insert_len)
        insert_key = full_key[:insert_len]
        insert_values = values[:insert_len]

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=insert_key,
                value=insert_values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        # Free our own slot ids that duplicated the tree's pre-existing nodes
        # at insert time. Clamped by the slack boundary (see docstring).
        free_end = self._mixed_kv_slack_insert_limit(req, new_prefix_len)
        if free_end > protected_len:
            self.token_to_kv_pool_allocator.free(kv_indices[protected_len:free_end])

        # Match as deeply as it is safe to share: the FULL key (so
        # ``cache_protected_len`` never regresses), clamped at the slack
        # boundary. ``bypass_mixed_kv_cap`` because this internal call needs
        # the full coverage to stay consistent with the admission set.
        match_len = self._mixed_kv_slack_insert_limit(req, len(full_key))
        match_result = self.match_prefix(
            MatchPrefixParams(key=full_key[:match_len], bypass_mixed_kv_cap=True)
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        full_match_len = len(new_indices)
        # The tree must contain at least what we just inserted.
        assert full_match_len >= new_prefix_len, (
            f"{full_match_len=} regressed below {new_prefix_len=}; tree "
            f"must cover at least what we inserted"
        )
        # `match_prefix` on the full key can extend past the trimmed insert
        # via tree coverage from sibling requests; upper bound is match_len.
        assert full_match_len <= match_len, f"{full_match_len=}, {match_len=}"

        # Only advance the protected region; never regress.
        if full_match_len > req.cache_protected_len:
            # Free our-own slots at positions that became tree-covered AFTER
            # our insert (via a sibling-added node past
            # ``max(new_prefix_len, insert_len)``): we are about to overwrite
            # req_to_token with the tree's ids there, so our extend-allocated
            # originals must be reclaimed. Clamped at the slack boundary.
            extra_free_start = max(protected_len, new_prefix_len, insert_len)
            extra_free_end = self._mixed_kv_slack_insert_limit(req, full_match_len)
            if extra_free_start < extra_free_end:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[extra_free_start:extra_free_end]
                )

            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(protected_len, full_match_len)),
                new_indices[protected_len:],
            )
            req.cache_protected_len = full_match_len
            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)
            req.last_node = new_last_node

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req`
        # later; keep the (request-owned) tail addressable past the protected
        # region.
        protected_len = req.cache_protected_len
        if protected_len <= len(new_indices):
            protected_indices = new_indices[:protected_len]
        else:
            protected_indices = kv_indices[:protected_len].to(dtype=torch.int64)

        if protected_len < len(kv_indices):
            req.prefix_indices = torch.cat(
                [protected_indices, kv_indices[protected_len:]]
            )
        else:
            req.prefix_indices = protected_indices

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
                ), "This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def recoverable_size(self):
        """Capacity recoverable via eviction.

        Under the unified mixed KV pool the radix tree only stores quant
        slots (HP slots are per-request and never enter the tree), so this
        is identical to ``evictable_size_`` — the standard SGLang shape.
        """
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

    def _mixed_kv_tail_to_drop(self, committed_len: int) -> int:
        # HP-recent slot ids are per-request and must not enter the tree.
        # Trim a fixed ``hp_recent + flush_overflow`` window from the
        # tail (page-aligned, ceil'd), which fully covers the worst-case
        # HP-recent span at any time.
        if not self._mixed_kv_enabled:
            return 0
        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        hp_prefix = int(kvcache.hp_prefix_tokens)
        hp_recent = int(kvcache.hp_recent_tokens)
        flush_overflow = max(1, int(kvcache.flush_interval)) - 1
        if hp_recent <= 0 or committed_len <= hp_prefix:
            return 0
        trim = min(hp_recent + flush_overflow, committed_len - hp_prefix)
        if self.page_size > 1:
            trim = math.ceil(trim / self.page_size) * self.page_size
        # Clip back if ceil pushed past the available range.
        trim = min(trim, committed_len - hp_prefix)
        return trim

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = key.child_key(self.page_size)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size)
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
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref

        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

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
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0, node

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
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
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
            node = new_node
        return total_prefix_length, node

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

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _delete_leaf(self, node):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self._discard_session_leaf(node)
        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
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


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5, 6, 7]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [8, 9, 10, 11, 12]))))
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 13, 14])))
        )
    )
