from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.runtime_context import get_memory, get_spec
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


@dataclass
class _LMCacheLoadBackMarker:
    """Carries the data ``init_load_back`` needs from the
    ``match_prefix`` call in MP mode.
    """

    key: RadixKey  # detached snapshot of the matched key (the live query key
    # aliases the req's growing fill_ids and must not be retained)
    value_numel: int  # number of tokens already in radix at match time


def _get_authoritative_page_values(
    req_to_token_pool,
    request,
    positions: torch.Tensor,
    target_device: torch.device,
) -> Optional[torch.Tensor]:
    """Snapshot the request's real page mapping, not an active overlay."""
    req_kv = getattr(request, "kv", None)
    req_pool_idx = getattr(req_kv, "req_pool_idx", None)
    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
    if req_pool_idx is None or req_to_token is None:
        return None
    if req_pool_idx < 0 or req_pool_idx >= req_to_token.shape[0]:
        return None
    source_positions = positions.to(device=req_to_token.device)
    return req_to_token[req_pool_idx, source_positions].clone().to(device=target_device)


def _get_authoritative_row_generation(req_to_token_pool, request) -> Optional[int]:
    """Return the allocator generation for a request row when available."""
    req_kv = getattr(request, "kv", None)
    req_pool_idx = getattr(req_kv, "req_pool_idx", None)
    generations = getattr(req_to_token_pool, "req_generation", None)
    if req_pool_idx is None or generations is None:
        return None
    if req_pool_idx < 0 or req_pool_idx >= len(generations):
        return None
    value = generations[req_pool_idx]
    return int(value.item()) if hasattr(value, "item") else int(value)


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
        model_config: Optional[ModelConfig] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(params)

        cli_lmc_cfg = get_memory().lmcache_config_file or ""

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
        self.sparda_prefetcher = None
        self._sparda_host_resident_enabled = bool(
            getattr(get_memory(), "enable_sparda", False)
        )
        self._sparda_compressed_indices: dict[tuple, dict[int, tuple]] = {}
        self._sparda_compressed_index_order: list[tuple] = []
        self._sparda_index_max_records = 2
        self._sparda_metrics: dict[str, int] = {}
        logger.debug(
            "SparDA LMCache cache init: host_resident=%s mode=%s page_size=%s",
            self._sparda_host_resident_enabled,
            self._mode.name,
            self.page_size,
        )

    def register_sparda_prefetcher(self, prefetcher) -> None:
        """Attach the optional request-scoped SparDA coordinator."""
        self.sparda_prefetcher = prefetcher

    def check_hicache_events(self) -> None:
        """Keep the scheduler hook harmless when LMCache owns the host tier.

        ``--enable-hierarchical-cache`` makes the scheduler poll this hook on
        every step.  LMCache completes its remote requests through the
        connector futures, so there is no SGLang HiCache event queue to drain
        here.
        """

        return None

    def sparda_prefetch_available(self) -> bool:
        """Return whether the LMCache connector can stage one sparse layer."""
        return (
            self._mode is LMCacheMode.MP
            and self.page_size == 1
            and callable(getattr(self.lmcache_connector, "sparse_prefetch", None))
            and callable(getattr(self.lmcache_connector, "sparse_retrieve", None))
            and callable(
                getattr(self.lmcache_connector, "create_sparse_object_keys", None)
            )
        )

    def _sparda_index_available(
        self, marker: _LMCacheLoadBackMarker, request: Req
    ) -> bool:
        """Return whether a host-side compressed index is ready for ``request``.

        The base LMCache adapter does not manufacture a compressed index.  A
        MiniCPM backend may publish one after it has built the index from a
        fully materialized request.  Keeping this conservative default makes
        an index miss an explicit full-load fallback instead of exposing
        uninitialized request pages to attention.
        """
        key = self._sparda_index_key(
            marker.key.raw_token_ids(),
            cache_salt=marker.key.cache_salt,
        )
        available = bool(getattr(self, "_sparda_compressed_indices", {}).get(key))
        logger.debug(
            "SparDA index lookup: prefix_tokens=%d salt=%s available=%s records=%s",
            len(marker.key),
            marker.key.cache_salt,
            available,
            sorted(
                (len(record_key[1]), record_key[0])
                for record_key in getattr(self, "_sparda_compressed_indices", {})
            ),
        )
        return available

    @staticmethod
    def _sparda_index_key(token_ids, *, cache_salt) -> tuple:
        try:
            hash(cache_salt)
            salt = cache_salt
        except TypeError:
            salt = repr(cache_salt)
        return salt, tuple(int(token_id) for token_id in token_ids)

    @staticmethod
    def _sparda_index_token_ids(request) -> list[int] | None:
        get_fill_ids = getattr(request, "get_fill_ids", None)
        if not callable(get_fill_ids):
            return None
        token_ids = list(get_fill_ids())
        full_ids = getattr(request, "full_untruncated_fill_ids", None)
        if full_ids is not None and len(token_ids) == len(full_ids) and token_ids:
            # The radix/LMCache logical prefix excludes the current token.
            # The request view includes it during prefill, so keep the index
            # under the same key that match_prefix later uses.
            token_ids.pop()
        return token_ids

    def publish_sparda_compressed_index(
        self, request, layer_id: int, levels: Sequence[torch.Tensor]
    ) -> None:
        """Keep a bounded CPU copy of the MiniCPM compressed-key index.

        This is intentionally separate from LMCache's logical KV objects.  It
        is an SGLang-side admission hint and is used only to decide whether a
        host hit can safely skip full-prefix restoration.  A missing record is
        a normal fallback, never a reason to read uninitialized KV pages.
        """
        token_ids = self._sparda_index_token_ids(request)
        if token_ids is None:
            return
        key = self._sparda_index_key(
            token_ids,
            cache_salt=getattr(request, "cache_salt", None),
        )
        copied_levels = tuple(
            level.detach().to(device="cpu", copy=True) for level in levels
        )
        records = getattr(self, "_sparda_compressed_indices", None)
        if records is None:
            self._sparda_compressed_indices = {}
            records = self._sparda_compressed_indices
        order = getattr(self, "_sparda_compressed_index_order", None)
        if order is None:
            self._sparda_compressed_index_order = []
            order = self._sparda_compressed_index_order
        if key not in records:
            order.append(key)
        records.setdefault(key, {})[int(layer_id)] = copied_levels
        max_records = int(getattr(self, "_sparda_index_max_records", 2))
        while len(order) > max_records:
            stale_key = order.pop(0)
            records.pop(stale_key, None)

    def get_sparda_compressed_index(self, request, layer_id: int):
        token_ids = self._sparda_index_token_ids(request)
        if token_ids is None:
            return None
        key = self._sparda_index_key(
            token_ids,
            cache_salt=getattr(request, "cache_salt", None),
        )
        record = getattr(self, "_sparda_compressed_indices", {}).get(key)
        if record is None:
            return None
        return record.get(int(layer_id))

    def sparda_metrics(self) -> dict[str, int]:
        return dict(getattr(self, "_sparda_metrics", {}))

    def _sparda_can_admit_host_resident(
        self, marker: _LMCacheLoadBackMarker, request: Req
    ) -> bool:
        full_ids = getattr(request, "full_untruncated_fill_ids", None)
        if not getattr(self, "_sparda_host_resident_enabled", False):
            return False
        if full_ids is not None and len(full_ids) not in (
            len(marker.key),
            len(marker.key) + 1,
        ):
            # The radix key normally excludes the current token.  A larger
            # gap is a genuine partial host hit and still needs boundary
            # compression from dense K values.
            return False
        available = bool(self._sparda_index_available(marker, request))
        metrics = getattr(self, "_sparda_metrics", None)
        if metrics is not None:
            metrics["index_hit" if available else "index_miss"] = (
                metrics.get("index_hit" if available else "index_miss", 0) + 1
            )
        return available

    def _allocate_sparda_host_resident(
        self,
        *,
        marker: _LMCacheLoadBackMarker,
        request: Req,
        uncached_len: int,
    ) -> Optional[torch.Tensor]:
        """Reserve request-owned pages without restoring the whole prefix.

        The pages are deliberately returned through the normal scheduler
        prefix path.  Sparse retrieval then writes selected logical chunks
        directly into this authoritative row; no temporary overlay pages are
        created for this mode.
        """
        if uncached_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))
        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None

        try:
            # LOOKUP read locks cover the complete logical prefix.  Once the
            # request is admitted into sparse mode, the per-layer sparse
            # lease becomes the owner of the selected chunks.  Re-LOOKUP is
            # used if a later fallback needs the full prefix.
            self.lmcache_connector.release_pending(request.rid)
        except BaseException:
            self.token_to_kv_pool_allocator.free(token_slots)
            raise

        request._sparda_host_resident = True
        request._sparda_host_marker = marker
        request._sparda_host_prefix_len = int(uncached_len)
        request._sparda_host_prefix_start = int(marker.value_numel)
        request._sparda_host_lookup_released = True
        metrics = getattr(self, "_sparda_metrics", None)
        if metrics is not None:
            metrics["host_resident_admission"] = (
                metrics.get("host_resident_admission", 0) + 1
            )
        logger.debug(
            "SparDA host-resident admission: request=%s tokens=%d",
            request.rid,
            uncached_len,
        )
        return token_slots

    def _materialize_sparda_host_request(self, request: Req) -> bool:
        """Restore a host-resident prefix before radix/cache mutation.

        This is used only on an explicit fallback or at request completion.
        It is the safety valve that prevents a request with partially staged
        pages from entering the radix cache or being recycled as valid KV.
        """
        if not getattr(request, "_sparda_host_resident", False):
            return True
        marker = getattr(request, "_sparda_host_marker", None)
        if marker is None or self._mode is not LMCacheMode.MP:
            return False

        host_len = int(getattr(request, "_sparda_host_prefix_len", 0))
        start = int(getattr(request, "_sparda_host_prefix_start", marker.value_numel))
        if host_len <= 0:
            request._sparda_host_resident = False
            return True

        token_ids = marker.key.raw_token_ids()
        matched = self.lmcache_connector.lookup_kv(token_ids, request.rid)
        if matched < start + host_len:
            logger.warning(
                "SparDA host-resident fallback lookup is incomplete: "
                "request=%s matched=%d required=%d",
                request.rid,
                matched,
                start + host_len,
            )
            self.lmcache_connector.release_pending(request.rid)
            return False

        req_pool_idx = getattr(getattr(request, "kv", None), "req_pool_idx", None)
        req_to_token = getattr(self.req_to_token_pool, "req_to_token", None)
        if req_pool_idx is None or req_to_token is None:
            self.lmcache_connector.release_pending(request.rid)
            return False
        if start + host_len > req_to_token.shape[1]:
            self.lmcache_connector.release_pending(request.rid)
            return False

        token_slots = req_to_token[req_pool_idx, start : start + host_len].to(
            device=self.device, dtype=torch.int64
        )
        slot_mapping = torch.empty(
            start + host_len, dtype=torch.int64, device=self.device
        )
        slot_mapping[:start].fill_(-1)
        slot_mapping[start:].copy_(token_slots)
        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = start % chunk_size
        current_stream = torch.get_device_module(self.device).current_stream()
        self.load_stream.wait_stream(current_stream)
        try:
            with device_stream_context(self.load_stream):
                num_retrieved = self.lmcache_connector.retrieve_kv(
                    LoadMetadata(
                        token_ids=marker.key.token_ids,
                        slot_mapping=slot_mapping,
                        offset=start - prefix_pad,
                        prefix_pad=prefix_pad,
                        request_id=request.rid,
                    )
                )
            current_stream.wait_stream(self.load_stream)
        except BaseException:
            logger.warning(
                "SparDA host-resident fallback retrieve failed: request=%s",
                request.rid,
                exc_info=True,
            )
            return False

        if int(num_retrieved) - prefix_pad < host_len:
            logger.warning(
                "SparDA host-resident fallback retrieve is partial: "
                "request=%s retrieved=%d required=%d",
                request.rid,
                int(num_retrieved) - prefix_pad,
                host_len,
            )
            return False

        request._sparda_host_resident = False
        request._sparda_host_marker = None
        request._sparda_host_prefix_len = 0
        request._sparda_host_prefix_start = 0
        request._sparda_host_lookup_released = False
        logger.debug(
            "SparDA host-resident prefix materialized: request=%s tokens=%d",
            request.rid,
            host_len,
        )
        return True

    def resolve_sparda_prefetch(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        context=None,
    ):
        """Turn predicted MiniCPM blocks into a one-layer LMCache transfer.

        LMCache stores complete logical chunks.  MiniCPM's selector returns
        sparse-attention block numbers, so multiple selected blocks can map to
        one LMCache chunk.  The remote side returns the hit bitmap and an IPC
        completion event; the page-table overlay is installed only after that
        event and is restored before the staging pages are freed.
        """
        from sglang.srt.mem_cache.sparda_prefetch import (
            CallbackPageLease,
            RemoteTransferCompletion,
            ResolvedPrefetch,
        )

        request = getattr(context, "request", None)
        forward_batch = getattr(context, "forward_batch", None)
        backend = getattr(context, "selector_backend", None)
        request_index = getattr(context, "request_index", None)
        if request is None or forward_batch is None or backend is None:
            return None
        if request_index is None or not forward_batch.forward_mode.is_decode_or_idle():
            return None
        if self._mode is not LMCacheMode.MP or self.page_size != 1:
            return None

        metadata = getattr(backend, "forward_metadata", None)
        base_metadata = getattr(metadata, "base", None)
        page_table = getattr(base_metadata, "page_table", None)
        if page_table is None or request_index >= page_table.shape[0]:
            return None
        if request_index >= len(forward_batch.seq_lens_cpu):
            return None

        get_fill_ids = getattr(request, "get_fill_ids", None)
        if not callable(get_fill_ids):
            return None
        token_ids = list(get_fill_ids())
        history_len = max(0, int(forward_batch.seq_lens_cpu[request_index]) - 1)
        history_len = min(history_len, len(token_ids))
        sparse_block_size = int(getattr(backend, "block_size", 0) or 0)
        chunk_size = self.lmcache_connector.chunk_size()
        if sparse_block_size <= 0 or chunk_size <= 0:
            return None

        chunk_indices = sorted(
            {
                (int(block_id) * sparse_block_size) // chunk_size
                for block_id in predicted_block_ids
                if int(block_id) >= 0
                and int(block_id) * sparse_block_size + chunk_size <= history_len
            }
        )
        if not chunk_indices:
            return None
        keys = self.lmcache_connector.create_sparse_object_keys(
            token_ids,
            chunk_indices,
            cache_salt=getattr(request, "cache_salt", None),
            request_id=request_id,
            generation=generation,
            layer_id=layer_id,
        )
        if len(keys) != len(chunk_indices):
            return None

        positions_list = [
            position
            for chunk_index in chunk_indices
            for position in range(
                chunk_index * chunk_size, (chunk_index + 1) * chunk_size
            )
        ]
        if not positions_list or positions_list[-1] >= page_table.shape[1]:
            return None
        positions = torch.tensor(
            positions_list, dtype=torch.long, device=page_table.device
        )
        request_row_idx = getattr(getattr(request, "kv", None), "req_pool_idx", None)
        request_row_generation = _get_authoritative_row_generation(
            self.req_to_token_pool, request
        )
        request_rid = getattr(request, "rid", request_id)
        authoritative_indices = _get_authoritative_page_values(
            self.req_to_token_pool,
            request,
            positions,
            page_table.device,
        )
        if authoritative_indices is None:
            logger.warning(
                "Cannot resolve authoritative page mapping for SparDA overlay"
            )
            return None
        use_request_pages = bool(getattr(request, "_sparda_host_resident", False))
        if use_request_pages:
            # Host-resident admission already reserved these request-owned
            # pages.  Sparse H2D must target the canonical row mapping so the
            # attention backend consumes exactly the pages it was given.
            device_indices = authoritative_indices
        else:
            device_indices = self.token_to_kv_pool_allocator.alloc(len(positions_list))
            if device_indices is None:
                return None
        block_ids = [
            device_indices.detach().to(dtype=torch.int64, device="cpu").tolist()
        ]
        state = {
            "active": False,
            "use_request_pages": use_request_pages,
            "local_freed": False,
            "remote_submitted": False,
            "remote_retrieved": False,
            "remote_released": False,
            "consumer_event": None,
            "found_indices": None,
        }
        timeout = float(getattr(self.lmcache_connector, "_mq_timeout", 30.0))

        def submit_remote():
            state["remote_submitted"] = True
            try:
                accepted = self.lmcache_connector.sparse_prefetch(
                    request_rid, generation, layer_id, keys
                ).result(timeout=timeout)
                if not accepted:
                    raise RuntimeError("LMCache sparse prefetch was rejected")
                state["remote_retrieved"] = True
                remote_future = self.lmcache_connector.sparse_retrieve(
                    request_rid, generation, layer_id, keys, block_ids
                )
                return RemoteTransferCompletion(
                    remote_future,
                    lambda found: state.__setitem__("found_indices", found),
                )
            except BaseException:
                if state["remote_submitted"] and not state["remote_released"]:
                    try:
                        released = self.lmcache_connector.sparse_cancel_prefetch(
                            request_rid, generation, layer_id
                        ).result(timeout=timeout)
                        if released:
                            state["remote_released"] = True
                    except BaseException:
                        logger.warning(
                            "Failed to cancel LMCache sparse prefetch after "
                            "submission failure",
                            exc_info=True,
                        )
                raise

        def install_overlay() -> Optional[bool]:
            found_indices = state["found_indices"]
            expected = tuple(range(len(keys)))
            if found_indices != expected:
                logger.debug(
                    "SparDA LMCache prefetch incomplete: found=%s expected=%d",
                    found_indices,
                    len(keys),
                )
                return False
            if state["use_request_pages"]:
                current_indices = _get_authoritative_page_values(
                    self.req_to_token_pool,
                    request,
                    positions,
                    page_table.device,
                )
                if current_indices is None or not torch.equal(
                    current_indices, device_indices
                ):
                    logger.warning(
                        "SparDA host-resident page mapping changed before install"
                    )
                    return False
                state["active"] = True
                return True
            page_table[request_index, positions] = device_indices.to(
                dtype=page_table.dtype
            )
            state["active"] = True
            return True

        def mark_consumed() -> None:
            if not state["active"] or not page_table.is_cuda:
                return
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=page_table.device))
            state["consumer_event"] = event

        def release_staging() -> None:
            consumer_event = state["consumer_event"]
            if consumer_event is not None:
                consumer_event.synchronize()
            elif state["active"] and page_table.is_cuda:
                torch.cuda.synchronize(device=page_table.device)
            if state["active"] and not state["use_request_pages"]:
                overlay_values = device_indices.to(
                    dtype=page_table.dtype, device=page_table.device
                )
                current_values = page_table[request_index, positions]
                if torch.equal(current_values, overlay_values):
                    current_row_idx = getattr(
                        getattr(request, "kv", None), "req_pool_idx", None
                    )
                    current_row_generation = _get_authoritative_row_generation(
                        self.req_to_token_pool, request
                    )
                    owner_changed = (
                        current_row_idx != request_row_idx
                        or (
                            request_row_generation is not None
                            and current_row_generation != request_row_generation
                        )
                        or getattr(request, "rid", request_rid) != request_rid
                    )
                    if owner_changed:
                        raise RuntimeError(
                            "SparDA overlay owner changed before page restoration"
                        )
                    restored_values = _get_authoritative_page_values(
                        self.req_to_token_pool,
                        request,
                        positions,
                        page_table.device,
                    )
                    if restored_values is None:
                        raise RuntimeError(
                            "Cannot restore authoritative page mapping for "
                            "SparDA overlay"
                        )
                    page_table[request_index, positions] = restored_values.to(
                        dtype=page_table.dtype
                    )
                state["active"] = False
            elif state["use_request_pages"]:
                # The request row owns these pages.  We only synchronize the
                # consumer before releasing the remote lease; row cleanup is
                # handled by the normal request allocator and generation.
                state["active"] = False
            if not state["local_freed"] and not state["use_request_pages"]:
                self.token_to_kv_pool_allocator.free(device_indices)
                state["local_freed"] = True

            if state["remote_submitted"] and not state["remote_released"]:
                if state["remote_retrieved"]:
                    released = self.lmcache_connector.sparse_release_prefetch(
                        request_rid, generation, layer_id
                    ).result(timeout=timeout)
                else:
                    released = self.lmcache_connector.sparse_cancel_prefetch(
                        request_rid, generation, layer_id
                    ).result(timeout=timeout)
                if not released:
                    raise RuntimeError("LMCache sparse lease release failed")
                state["remote_released"] = True

        lease = CallbackPageLease(
            release_callback=release_staging,
            consumed_callback=mark_consumed,
        )
        return ResolvedPrefetch(
            transfers=(),
            layer_num=self.lmcache_connector.num_layers,
            lease=lease,
            on_ready=install_overlay,
            submit_callback=submit_remote,
        )

    def restore_sparda_request(self, request) -> bool:
        """Release request tickets before the request row is recycled."""
        if self.sparda_prefetcher is not None:
            if not self.sparda_prefetcher.cleanup_request(request.rid):
                return False
        if getattr(request, "_sparda_host_resident", False):
            return self._materialize_sparda_host_request(request)
        return True

    def _discard_sparda_host_request(self, request) -> bool:
        """Drop a host-resident request without exposing partial KV to radix.

        A host-resident request owns only the pages populated by sparse
        transfers.  Materializing the complete logical prefix here would
        defeat host residency just before the request row is recycled.  The
        request therefore takes the non-inserting cleanup path; the host
        objects and compressed index remain the authoritative cache state.
        """
        if self.sparda_prefetcher is not None:
            if not self.sparda_prefetcher.cleanup_request(request.rid):
                return False
        request._sparda_host_resident = False
        request._sparda_host_marker = None
        request._sparda_host_prefix_len = 0
        request._sparda_host_prefix_start = 0
        request._sparda_host_lookup_released = False
        return True

    def reset(self):
        if getattr(self, "sparda_prefetcher", None) is not None:
            if not self.sparda_prefetcher.cleanup_all():
                raise RuntimeError("SparDA prefetch cleanup failed before reset")
        super().reset()
        if hasattr(self, "_in_flight_nodes"):
            with self._node_lock:
                self._in_flight_nodes.clear()
        if hasattr(self, "_mp_load_back_markers"):
            self._mp_load_back_markers.clear()
        # ``reset`` only drops GPU radix/page state.  The corresponding
        # logical KV objects and their compressed-key indices live in the
        # LMCache host pool and remain valid across a GPU cache flush.  Keep
        # the bounded index registry so the next host hit can enter the
        # sparse path instead of restoring the complete prefix.
        if hasattr(self, "_sparda_metrics"):
            self._sparda_metrics.clear()

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
            key=RadixKey(
                token_ids,
                key.extra_key,
                key.is_bigram,
                cache_salt=key.cache_salt,
            ),
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

        if self._sparda_can_admit_host_resident(marker, req):
            token_slots = self._allocate_sparda_host_resident(
                marker=marker,
                request=req,
                uncached_len=params.host_hit_length,
            )
            if token_slots is not None:
                return token_slots, last_node
            logger.debug(
                "SparDA host-resident admission fell back to full load: "
                "request=%s reason=allocation",
                req.rid,
            )

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

            self.kv_events.record_store(new_node.parent)
            self.kv_events.record_store(new_node)

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

        if getattr(req, "_sparda_host_resident", False):
            if not self._discard_sparda_host_request(req):
                raise RuntimeError(
                    "SparDA prefetch cleanup failed before host-resident "
                    "request finalization"
                )

            # Sparse selection populated only selected request pages.  Do not
            # insert or store those partial pages as a complete prefix.  The
            # logical host objects and compressed index stay authoritative,
            # while the normal non-inserting path frees the request-owned row.
            super().cache_finished_req(
                req, is_insert=False, kv_len_to_handle=kv_len_to_handle
            )
            if self._mode is LMCacheMode.MP:
                self._mp_load_back_markers.pop(req.rid, None)
                self.lmcache_connector.end_session(req.rid)
            return

        if not self.restore_sparda_request(req):
            raise RuntimeError(
                "SparDA prefetch cleanup failed before request cache mutation"
            )

        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if not is_insert:
            if self._mode is LMCacheMode.MP:
                self._mp_load_back_markers.pop(req.rid, None)
                self.lmcache_connector.end_session(req.rid)
            return

        topk = get_spec().speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]

        # Use super() to avoid a redundant LOOKUP — we only need new_last_node from radix.
        match_result = super().match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    req.extra_key,
                    cache_salt=req.cache_salt,
                )
            )
        )
        new_last_node = match_result.last_device_node
        assert new_last_node is not None

        # ``super().cache_finished_req`` may have freed or reused the request
        # row before this method runs.  The radix match is the authoritative
        # mapping that keeps those pages alive; never read the mutable request
        # row for an asynchronous LMCache store.
        store_len = min(kv_committed_len, match_result.device_indices.numel())
        store_md = StoreMetadata(
            last_node=new_last_node,
            token_ids=token_ids[:store_len],
            kv_indices=match_result.device_indices[:store_len].clone(),
            offset=0,
            request_id=req.rid,
        )

        self.inc_lock_ref(new_last_node)
        if self._mode is LMCacheMode.MP:
            try:
                self.lmcache_connector.store_kv(store_md)
            finally:
                # MP store_kv blocks until the daemon's signal event fires, so
                # the slots are safe to evict immediately.  Also release the
                # lock when the daemon reports a failure; otherwise one bad
                # store permanently pins the radix node.
                self._mp_load_back_markers.pop(req.rid, None)
                self.dec_lock_ref(new_last_node)
                self.lmcache_connector.end_session(req.rid)
        elif self._mode is LMCacheMode.IP:
            try:
                with device_stream_context(self.store_stream):
                    self.lmcache_connector.store_kv(store_md)
            except BaseException:
                self.dec_lock_ref(new_last_node)
                raise
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
