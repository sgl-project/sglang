from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheFinishedReqResult,
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.runtime_context import get_server_args

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


@dataclass
class _LMCacheLoadBackMarker:
    """Carries the data ``init_load_back`` needs from the
    ``match_prefix`` call in MP mode.
    """

    key: RadixKey  # detached snapshot of the matched key (the live query key
    # aliases the req's growing fill_ids and must not be retained)
    value_numel: int  # number of tokens already in radix at match time


class LMCacheMode(enum.Enum):
    MP = enum.auto()  # multi-process mode
    IP = enum.auto()  # in-process mode


class LayerTransferCounter:
    """Minimal adapter that lets the memory pool notify LMCache per-layer.

    The KV pool calls `wait_until(layer_id)` after finishing a layer, which we
    translate into a `load_kv_layerwise(layer_id)` call on the LMCache connector
    within the provided CUDA stream.
    """

    def __init__(
        self,
        num_layers: int,
        load_stream: torch.cuda.Stream,
        lmc_connector: LMCacheLayerwiseConnector,
        printable: bool = False,
    ):
        self.num_layers = num_layers
        self.load_stream = load_stream
        self.lmc_connector = lmc_connector

    def wait_until(self, layer_id: int):
        # Ensure ordering of the async loads wrt compute stream(s).
        self.load_stream.synchronize()
        with self.load_stream:
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
        model_config: Optional[ModelConfig] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(params)

        cli_lmc_cfg = get_server_args().lmcache_config_file or ""

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        connector_kwargs = dict(
            sgl_config=model_config,
            tp_size=tp_size,
            rank=rank,
            # NOTE: The original implementation accessed private buffers via
            # `_kvcache.k_buffer` / `.v_buffer`. We prefer public accessors when
            # available; fall back to private fields if needed.
            k_pool=getattr(
                kvcache,
                "k_buffer",
                getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),
            ),
            v_pool=getattr(
                kvcache,
                "v_buffer",
                getattr(self.token_to_kv_pool_allocator._kvcache, "v_buffer"),
            ),
            tp_group=tp_group.device_group if tp_group is not None else None,
        )

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        # MP is the default. To use the in-process layerwise connector,
        # set ``self._mode = LMCacheMode.IP`` here.
        self._mode = LMCacheMode.MP
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

    def reset(self):
        super().reset()
        if hasattr(self, "_in_flight_nodes"):
            with self._node_lock:
                self._in_flight_nodes.clear()
        if hasattr(self, "_mp_load_back_markers"):
            self._mp_load_back_markers.clear()

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
        """
        token_ids = key.raw_token_ids()
        matched = self.lmcache_connector.lookup_kv(token_ids, req.rid)
        if matched <= value.numel():
            # Release the read locks; keep the pending session for end_session.
            self.lmcache_connector.release_pending(req.rid)
            return base_res

        if token_ids is key.token_ids:
            token_ids = token_ids[:]
        self._mp_load_back_markers[req.rid] = _LMCacheLoadBackMarker(
            key=RadixKey(token_ids, key.extra_key, key.is_bigram),
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

        token_ids = key.raw_token_ids()
        result = self._load_back(
            key=key,
            value_numel=int(value.numel()),
            uncached_len=uncached_len,
            last_node=last_node,
            load_fn=lambda sm, pp: self._ip_load_back(
                token_ids=token_ids,
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
            new_node = TreeNode(priority=last_node.priority)
            start = value_numel
            end = start + fetched
            new_node.key = key[start:end]
            new_node.value = token_slots[:fetched]
            new_node.parent = last_node
            last_node.children[new_node.key.child_key(self.page_size)] = new_node
            self.evictable_size_ += fetched
            self._update_leaf_status(last_node)
            self._update_leaf_status(new_node)

            self._record_store_event(new_node.parent)
            self._record_store_event(new_node)

            return token_slots[:fetched], new_node

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
        self.load_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.load_stream):
            n = self.lmcache_connector.retrieve_kv(
                LoadMetadata(
                    token_ids=marker.key.token_ids,
                    slot_mapping=slot_mapping,
                    offset=marker.value_numel - prefix_pad,
                    prefix_pad=prefix_pad,
                    request_id=request_id,
                )
            )
        torch.cuda.current_stream().wait_stream(self.load_stream)
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
        with torch.cuda.stream(self.load_stream):
            return self.lmcache_connector.start_load_kv(
                LoadMetadata(
                    token_ids=token_ids,
                    slot_mapping=slot_mapping,
                    offset=value_numel - prefix_pad,
                )
            )

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> CacheFinishedReqResult:
        """On request completion, insert device KV into radix and store to LMCache."""

        result = super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if not is_insert:
            if self._mode is LMCacheMode.MP:
                self._mp_load_back_markers.pop(req.rid, None)
                self.lmcache_connector.end_session(req.rid)
            return result

        global_server_args = get_server_args()
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

        # Use super() to avoid a redundant LOOKUP — we only need new_last_node from radix.
        match_result = super().match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
        )
        new_last_node = match_result.last_device_node
        assert new_last_node is not None

        self.inc_lock_ref(new_last_node)
        store_md = StoreMetadata(
            last_node=new_last_node,
            token_ids=token_ids,
            kv_indices=kv_indices,
            offset=0,
            request_id=req.rid,
        )
        with torch.cuda.stream(self.store_stream):
            self.lmcache_connector.store_kv(store_md)
        if self._mode is LMCacheMode.MP:
            # MP store_kv blocks until the daemon's signal event fires, so the slots are safe to evict immediately.
            self._mp_load_back_markers.pop(req.rid, None)
            self.dec_lock_ref(new_last_node)
            self.lmcache_connector.end_session(req.rid)
        elif self._mode is LMCacheMode.IP:
            # Layerwise store is async on store_stream; defer the unlock to evict()'s store_stream.synchronize().
            with self._node_lock:
                self._in_flight_nodes.append(new_last_node)

        return result

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
