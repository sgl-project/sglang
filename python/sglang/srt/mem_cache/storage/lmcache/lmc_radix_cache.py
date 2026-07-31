from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import create_device_stream, device_stream_context

try:
    from lmcache.integration.sglang.multi_process_adapter import LMCacheMPConnector
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )
    from lmcache.integration.sglang.utils import lmcache_get_config
except ImportError as e:
    raise RuntimeError(
        "LMCache is not installed. Please install it by running `pip install lmcache`"
    ) from e


if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


def _extract_kv_pools(kvcache) -> list[list[torch.Tensor]]:
    """Classify a sglang KV pool and return the per-group tensor lists.

    sglang has three per-layer KV layouts we care about:

    * MHA / GQA (``MHATokenToKVPool``): two disjoint per-layer lists
      → ``[k_buffer, v_buffer]``
    * MLA (``MLATokenToKVPool``): a single fused per-layer list
      → ``[kv_buffer]``
    * DSA (``DSATokenToKVPool``): fused ``kv_buffer`` + an additional
      per-layer ``index_k_with_scale_buffer`` for sparse attention
      → ``[kv_buffer, index_k_with_scale_buffer]``
    """
    if isinstance(kvcache, DSATokenToKVPool):
        return [
            kvcache.kv_buffer,
            kvcache.index_k_with_scale_buffer,
        ]
    if isinstance(kvcache, MLATokenToKVPool):
        return [kvcache.kv_buffer]
    if isinstance(kvcache, MHATokenToKVPool):
        return [kvcache.k_buffer, kvcache.v_buffer]

    raise RuntimeError(
        f"Unsupported KV pool type {type(kvcache).__name__}"
    )


@dataclass
class _LMCacheLoadBackMarker:
    """Carries the data ``init_load_back`` needs from the
    ``match_prefix`` call in MP mode.
    """

    key: RadixKey  # page-aligned key the scheduler matched on
    value_numel: int  # number of tokens already in radix at match time


class LMCacheMode(enum.Enum):
    MP = enum.auto()  # multi-process mode
    IP = enum.auto()  # in-process mode


class LayerTransferCounter:
    """Minimal adapter that lets the memory pool notify LMCache per-layer.

    The KV pool calls `wait_until(layer_id)` after finishing a layer, which we
    translate into a `load_kv_layerwise(layer_id)` call on the LMCache connector
    within the provided device stream.
    """

    def __init__(
        self,
        num_layers: int,
        load_stream: torch.Stream,
        lmc_connector: LMCacheLayerwiseConnector,
        printable: bool = False,
    ):
        self.num_layers = num_layers
        self.load_stream = load_stream
        self.lmc_connector = lmc_connector

    def wait_until(self, layer_id: int):
        # Ensure ordering of the async loads wrt compute stream(s).
        self.load_stream.synchronize()
        with device_stream_context(self.load_stream):
            self.lmc_connector.load_kv_layerwise(layer_id)


class LMCRadixCache(RadixCache):
    """RadixCache + LMCache IO.

    IP mode keeps the existing layerwise connector and
    its per-layer transfer hook: ``match_prefix`` kicks off the load via
    ``start_load_kv`` and SGLang's per-layer KV-pool hook drives subsequent
    layers during forward.

    MP mode uses ``LMCacheMPConnector`` with a two-phase
    load: ``match_prefix`` fires LOOKUP only (``connector.lookup_kv``) and
    returns ``host_hit_length`` on the ``MatchResult``; the SGLang
    scheduler then calls `init_load_back` at dispatch time,
    which fires the actual RETRIEVE (``connector.retrieve_kv``) into
    pre-allocated GPU slots.
    """

    def __init__(
        self,
        params: CacheInitParams,
        model_config: Optional["ModelConfig"] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(params)

        cli_lmc_cfg = get_global_server_args().lmcache_config_file or ""

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        kv_cache_pools = _extract_kv_pools(kvcache)
        connector_kwargs = dict(
            sgl_config=model_config,
            tp_size=tp_size,
            rank=rank,
            kv_cache_pools=kv_cache_pools,
            tp_group=tp_group.device_group if tp_group is not None else None,
        )

        self.load_stream = create_device_stream(self.device)
        self.store_stream = create_device_stream(self.device)

        # MP (multi-process) is the default. XPU defaults to IP (in-process
        # layerwise) because the MP connector shares the KV cache via CUDA IPC
        # (``Tensor._share_cuda_``), which is unavailable on XPU.
        self._mode = LMCacheMode.IP if self.device.type == "xpu" else LMCacheMode.MP
        if self._mode is LMCacheMode.MP:
            if not cli_lmc_cfg:
                raise ValueError(
                    "MP mode requires --lmcache-config-file (the YAML "
                    "supplies mp_host / mp_port)."
                )
            lm_cfg = lmcache_get_config(cli_lmc_cfg)
            self.lmcache_connector = LMCacheMPConnector(
                page_size=params.page_size,
                host=lm_cfg.mp_host,
                port=lm_cfg.mp_port,
                **connector_kwargs,
            )
        elif self._mode is LMCacheMode.IP:
            self.lmcache_connector = LMCacheLayerwiseConnector(
                config_file=cli_lmc_cfg, **connector_kwargs
            )
            # Per-layer hook
            self.layer_done_executor = LayerTransferCounter(
                num_layers=(
                    model_config.num_hidden_layers if model_config is not None else 0
                ),
                load_stream=self.load_stream,
                lmc_connector=self.lmcache_connector,
            )
            kvcache.register_layer_transfer_counter(self.layer_done_executor)

        self._in_flight_nodes: list[TreeNode] = []
        self._node_lock = threading.Lock()
        self._mp_load_back_markers: dict[str, _LMCacheLoadBackMarker] = {}
        # Track LMCache hit count per request so cache_finished_req can
        # skip storing tokens that LMCache already has.
        self._mp_lmcache_hit_tokens: dict[str, int] = {}

    def reset(self):
        super().reset()
        if hasattr(self, "_in_flight_nodes"):
            with self._node_lock:
                self._in_flight_nodes.clear()
        if hasattr(self, "_mp_load_back_markers"):
            self._mp_load_back_markers.clear()
        if hasattr(self, "_mp_lmcache_hit_tokens"):
            self._mp_lmcache_hit_tokens.clear()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Dispatch to the mode-specific match_prefix.

        MP mode → ``_mp_match_prefix`` (fires LOOKUP only).
        IP mode → ``_ip_match_prefix`` (single-shot ``start_load_kv``
        plus per-layer hook).
        """
        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(params)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        if self._mode is LMCacheMode.MP:
            if params.req is None:
                return base_res
            return self._mp_match_prefix(key, base_res, value, last_node, params.req)
        elif self._mode is LMCacheMode.IP:
            return self._ip_match_prefix(key, base_res, value, last_node)
        return base_res

    def _mp_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:
        """MP LOOKUP

        Returns a ``MatchResult`` with ``host_hit_length`` set when
        LMCache has tokens beyond radix. Otherwise releases
        the held read locks and returns the radix-only result.

        Cache the LOOKUP result per request to avoid firing redundant
        LOOKUPs.  The scheduler re-evaluates waiting-queue requests
        every cycle (``calc_priority`` → ``_compute_prefix_matches``
        → ``match_prefix_for_req``), and each call would otherwise
        fire a new LOOKUP to the daemon, creating duplicate prefetch
        jobs and wasting daemon resources.
        """
        cached = self._mp_lmcache_hit_tokens.get(req.rid)
        if cached is not None:
            if cached <= value.numel():
                return base_res
            return MatchResult(
                device_indices=value,
                last_device_node=last_node,
                last_host_node=last_node,
                best_match_node=last_node,
                host_hit_length=cached - int(value.numel()),
            )

        matched = self.lmcache_connector.lookup_kv(key.token_ids, req.rid)

        # Record LMCache hit count so cache_finished_req can skip
        # storing tokens that LMCache already has.
        self._mp_lmcache_hit_tokens[req.rid] = matched

        if matched <= value.numel():
            # Release the read locks; keep the pending session for end_session.
            self.lmcache_connector.release_pending(req.rid)
            return base_res

        self._mp_load_back_markers[req.rid] = _LMCacheLoadBackMarker(
            key=key,
            value_numel=int(value.numel()),
        )
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=matched - int(value.numel()),
        )

    def _ip_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        value: torch.Tensor,
        last_node: TreeNode,
    ) -> MatchResult:
        """IP mode: ``start_load_kv`` + per-layer hook.

        Allocates slots for the page-aligned uncached tail and kicks off
        the layerwise load. Returns ``base_res`` if there's nothing to
        fetch or alloc/load fails.
        """
        if value.numel() == len(key):
            return base_res

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        result = self._load_back(
            key=key,
            value_numel=int(value.numel()),
            uncached_len=uncached_len,
            last_node=last_node,
            load_fn=lambda sm, pp: self._ip_load_back(
                token_ids=key.token_ids,
                value_numel=int(value.numel()),
                slot_mapping=sm,
                prefix_pad=pp,
            ),
        )
        if result is None:
            return base_res
        new_slots, new_node = result
        return MatchResult(
            device_indices=torch.cat([value, new_slots]),
            last_device_node=new_node,
            last_host_node=new_node,
            best_match_node=new_node,
        )

    def init_load_back(
        self, params: InitLoadBackParams
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        """MP RETRIEVE.

        Called by the scheduler when ``match_prefix`` returned
        ``host_hit_length > 0``. Uses the cached LOOKUP result to
        allocate slots and fire RETRIEVE, inserts the resulting
        TreeNode into the radix tree, and returns
        ``(new_indices, new_last_node)``.
        """
        req = params.req
        marker = self._mp_load_back_markers.pop(req.rid)
        last_node: TreeNode = params.best_match_node

        result = self._load_back(
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
            load_fn=lambda sm, pp: self._mp_load_back(
                marker=marker,
                request_id=req.rid,
                slot_mapping=sm,
                prefix_pad=pp,
            ),
        )
        if result is None:
            # Either alloc failed (locks still held by lookup_kv) or
            # retrieve returned nothing (locks already released by
            # retrieve_kv). release_pending is idempotent on locks_held.
            self.lmcache_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        return result

    def _insert_from_node(
        self,
        start_node: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
    ) -> TreeNode:
        """Insert ``key``/``value`` pairs starting from ``start_node``.

        Mirrors the bottom half of ``_insert_helper`` but begins walking
        the tree at ``start_node`` instead of the root.  Returns the leaf
        ``TreeNode`` that was created or reached.

        Used by ``_load_back`` to attach LMCache-loaded tokens under the
        radix-cached parent node so that ``cache_finished_req`` does not
        create a duplicate set of nodes and double-count ``evictable_size_``.
        """
        node = start_node
        access_time = time.monotonic()
        node.last_access_time = access_time

        while len(key) > 0:
            child_key = key.child_key(self.page_size)
            if child_key in node.children:
                child = node.children[child_key]
                child.last_access_time = access_time
                prefix_len = child.key.match(key, page_size=self.page_size)
                if prefix_len < len(child.key):
                    child = self._split_node(child.key, child, prefix_len)
                child.priority = max(child.priority, priority)
                self._inc_hit_count(child)
                node = child
                key = key[prefix_len:]
                value = value[prefix_len:]
            else:
                new_node = TreeNode(priority=priority)
                new_node.parent = node
                new_node.key = key
                new_node.value = value.clone()
                node.children[child_key] = new_node
                self.evictable_size_ += len(key)
                self._update_leaf_status(node)
                self._update_leaf_status(new_node)
                self._record_store_event(new_node)
                return new_node

        return node

    def _load_back(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
        load_fn,  # Callable[[torch.Tensor, int], int] — (slot_mapping, prefix_pad) -> num_retrieved
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Alloc slots, run ``load_fn``, attach a TreeNode for what was loaded.

        Returns ``(slots, new_node)`` on success, ``None`` if alloc fails
        or the load returned zero (slots are freed in either case).
        """
        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = value_numel % chunk_size

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None

        slot_mapping = torch.empty(
            value_numel + token_slots.numel(),
            dtype=torch.int64,
            device=self.device,
        )
        slot_mapping[:value_numel].fill_(-1)
        slot_mapping[value_numel:].copy_(token_slots)

        # Dispatch to the mode-specific loader (IP: start_load_kv, MP:
        # retrieve_kv). Each loader manages its own load_stream context.
        num_retrieved = load_fn(slot_mapping, prefix_pad)
        logger.debug("num_retrieved_tokens: %s", num_retrieved)

        if num_retrieved > 0:
            self.token_to_kv_pool_allocator.free(
                token_slots[(num_retrieved - prefix_pad) :]
            )
        else:
            self.token_to_kv_pool_allocator.free(token_slots)

        if num_retrieved > 0:
            fetched = num_retrieved - prefix_pad

            start = value_numel
            end = start + fetched

            insert_key = key[start:end]
            insert_value = token_slots[:fetched]
            # Match base RadixCache.insert() semantics exactly so that the
            # radix subtree we build here is discoverable by later
            # super().insert() calls (which walk from root using the same
            # (bigram, page_aligned) key transform):
            #
            #   1) When is_eagle is enabled, radix keys are stored in bigram
            #      view. If we insert here in raw view, the child_key hash
            #      differs and later super().insert() cannot find this node
            #      — it will attach a parallel subtree covering the SAME kv
            #      slots, double-counting evictable_size_ and leaking pool
            #      accounting (the extra evictable is never balanced by an
            #      alloc, so available+evictable+protected > total).
            #   2) Base insert page-aligns the key. If we skip alignment, a
            #      later match() against a bigram key rounds down to the
            #      page, so a page-straddling suffix ends up covered by both
            #      the LMCache leaf and the newly created bigram node —
            #      another form of double-counting.
            insert_key, insert_value = insert_key.maybe_to_bigram_view(
                self.is_eagle, insert_value
            )
            # Page-align to base insert() semantics.
            aligned_len = (len(insert_key) // self.page_size) * self.page_size
            insert_key = insert_key[:aligned_len]
            if insert_value is not None:
                insert_value = insert_value[:aligned_len]

            if aligned_len == 0:
                # Nothing survives page alignment — release the slots we
                # retrieved into and return None so init_load_back falls
                # back to the empty result path.
                self.token_to_kv_pool_allocator.free(token_slots[:fetched])
                return None

            # Walk from last_node (the radix-cached node) so the new nodes
            # are attached as children of the correct parent in the tree.
            # Using self.insert() would start from root and incorrectly
            # attach the partial-key nodes at root level, causing
            # cache_finished_req to create a duplicate set of nodes under
            # last_node and double-counting evictable_size_.
            new_node = self._insert_from_node(
                last_node,
                insert_key,
                insert_value,
                priority=last_node.priority or 0,
            )

            # Return exactly the slots that are now referenced by the radix
            # subtree (aligned_len of them). Any tail slots dropped by
            # page-alignment (or by the bigram len-1 truncation) must be
            # released so they don't leak from the allocator's perspective.
            kept_slots = token_slots[:aligned_len]
            if aligned_len < fetched:
                self.token_to_kv_pool_allocator.free(token_slots[aligned_len:fetched])

            return kept_slots, new_node

        return None

    def _mp_load_back(
        self,
        *,
        marker: _LMCacheLoadBackMarker,
        request_id: str,
        slot_mapping: torch.Tensor,
        prefix_pad: int,
    ) -> int:
        """MP non-layerwise loader: fire ``retrieve_kv`` and wait for the
        load_stream so the compute stream observes the writes.
        """
        current_stream = torch.get_device_module(self.device).current_stream()
        self.load_stream.wait_stream(current_stream)
        with device_stream_context(self.load_stream):
            n = self.lmcache_connector.retrieve_kv(
                LoadMetadata(
                    token_ids=marker.key.token_ids,
                    slot_mapping=slot_mapping,
                    offset=marker.value_numel - prefix_pad,
                    prefix_pad=prefix_pad,
                    request_id=request_id,
                )
            )
        current_stream.wait_stream(self.load_stream)
        return n

    def _ip_load_back(
        self,
        *,
        token_ids: list[int],
        value_numel: int,
        slot_mapping: torch.Tensor,
        prefix_pad: int,
    ) -> int:
        """IP layerwise loader: kick off ``start_load_kv`` on ``self.load_stream``.

        ``start_load_kv`` enqueues the first layer's transfer; the
        ``LayerTransferCounter`` hook drives the rest during forward.
        """
        with device_stream_context(self.load_stream):
            return self.lmcache_connector.start_load_kv(
                LoadMetadata(
                    token_ids=token_ids,
                    slot_mapping=slot_mapping,
                    offset=value_numel - prefix_pad,
                )
            )

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:
        """On request completion, insert device KV into radix and store to LMCache."""

        # Ensure cache_protected_len accounts for all tokens that are already
        # in the radix tree (including those loaded by LMCache via _load_back).
        # The base class uses cache_protected_len to compute the range of pool
        # slots to free — if it undercounts, LMCache-allocated slots get
        # double-freed, causing pool <-> tree accounting drift.
        orig_protected_len = req.cache_protected_len
        prefix_len = len(req.prefix_indices)

        if prefix_len > req.cache_protected_len:
            req.cache_protected_len = prefix_len

        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )

        req.cache_protected_len = orig_protected_len

        if not is_insert:
            if self._mode is LMCacheMode.MP:
                self._mp_load_back_markers.pop(req.rid, None)
                self._mp_lmcache_hit_tokens.pop(req.rid, None)
                self.lmcache_connector.end_session(req.rid)
            return

        global_server_args = get_global_server_args()
        topk = global_server_args.speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Skip storing tokens that LMCache already has.
        # _mp_lmcache_hit_tokens is populated by _mp_match_prefix during LOOKUP.
        lmcache_hit = self._mp_lmcache_hit_tokens.pop(req.rid, 0)
        chunk_size = self.lmcache_connector.chunk_size()
        store_offset = (lmcache_hit // chunk_size) * chunk_size
        if store_offset >= kv_committed_len:
            # Entire request is already in LMCache — nothing to store.
            if self._mode is LMCacheMode.MP:
                self._mp_load_back_markers.pop(req.rid, None)
                self.lmcache_connector.end_session(req.rid)
            return

        uncached_kv_indices = kv_indices[store_offset:]

        # Use super() to avoid a redundant LOOKUP — we only need new_last_node from radix.
        match_result = super().match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
        )
        new_last_node = match_result.last_device_node
        assert new_last_node is not None

        self.inc_lock_ref(new_last_node)
        store_md = StoreMetadata(
            last_node=new_last_node,
            token_ids=token_ids,  # FULL token_ids (not truncated) — the daemon
            kv_indices=uncached_kv_indices,  # uses a rolling hash that requires
            offset=store_offset,  # computing from position 0 for correct chaining.
            request_id=req.rid,
        )
        if self._mode is LMCacheMode.MP:
            self.lmcache_connector.store_kv(store_md)
            # MP store_kv blocks until the daemon's signal event fires, so the slots are safe to evict immediately.
            self._mp_load_back_markers.pop(req.rid, None)
            self.dec_lock_ref(new_last_node)
            self.lmcache_connector.end_session(req.rid)
        elif self._mode is LMCacheMode.IP:
            with device_stream_context(self.store_stream):
                self.lmcache_connector.store_kv(store_md)
            # Layerwise store is async on store_stream; defer the unlock to evict()'s store_stream.synchronize().
            with self._node_lock:
                self._in_flight_nodes.append(new_last_node)

    def evict(self, params: EvictParams) -> EvictResult:
        """Before base eviction, wait for any outstanding stores and release locks."""
        if self.disable:
            return EvictResult()

        self.store_stream.synchronize()
        with self._node_lock:
            for node in self._in_flight_nodes:
                self.dec_lock_ref(node)
            self._in_flight_nodes.clear()

        return super().evict(params)

    def pretty_print(self):
        super().pretty_print()
        try:
            logger.debug(
                "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
            )
        except Exception:  # pragma: no cover
            pass

