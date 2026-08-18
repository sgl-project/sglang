"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. Self-contained: this module owns both halves of the contract.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``_inc_hit_count``    -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InsertParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``prefix_key`` covers the device-cached prefix plus the restorable tail, so
    it is what gets inserted once the tail lands. ``tail_hashes`` are the
    per-page storage hashes of that tail alone, starting at ``device_hit_len``.
    """

    prefix_key: RadixKey
    tail_hashes: list[str]
    device_hit_len: int


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        server_args: ServerArgs,
        params: CacheInitParams,
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            MooncakeDirectLinker,
        )

        self.cache = cache
        self.cache_linker: UnifiedCacheLinker = MooncakeDirectLinker(
            server_args, params
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[tuple[NodeId, DecLockRefParams]] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        page = cache.page_size
        device_hit_len = int(result.device_indices.numel())
        if device_hit_len >= len(key):
            return result

        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        if not tail_hashes:
            return result

        lookup_transfers = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, tail_hashes
            )
            if transfer is None:
                return result
            lookup_transfers.append(transfer)
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        hit_pages = self._sync_restorable_prefix(
            self.cache_linker.lookup(req.rid, lookup_transfers),
            num_pages=len(tail_hashes),
            device_hit_pages=0,
        )
        if hit_pages == 0:
            return result
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            prefix_key=key[: device_hit_len + hit_tokens],
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in restorable:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> device, then insert ----

    def load_back(self, req: Req) -> tuple[torch.Tensor, NodeId]:
        cache = self.cache
        empty_indices = cache.tree_core.empty_match_result.device_indices
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return empty_indices, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, tail_hashes
            )
            if transfer is None:
                self._finish_load(req, component_transfers, prefix_len, False)
                return empty_indices, req.last_node
            component_transfers.append((component, transfer))

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        # The reservation is committed here; the data itself arrives later, when
        # the scheduler starts the layer-wise load for this batch.
        self._finish_load(req, component_transfers, prefix_len, True)

        # Insert the newly loaded tail into the tree.
        prefix_indices = torch.cat(
            [req.prefix_indices.to(torch.int64), full_transfer.device_indices]
        )
        mamba_transfer = next(
            (
                transfer
                for _, transfer in component_transfers
                if transfer.name == PoolName.MAMBA
            ),
            None,
        )
        insert_result = cache.insert(
            InsertParams(
                key=hit.prefix_key,
                value=prefix_indices,
                mamba_value=(
                    mamba_transfer.device_indices[:1]
                    if mamba_transfer is not None
                    else None
                ),
                prev_prefix_len=device_hit_len,
                swa_evicted_seqlen=(
                    req.kv.swa_evicted_seqlen if req.kv is not None else 0
                ),
                chunked=True,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        if mamba_transfer is not None and insert_result.mamba_exist:
            cache.req_to_token_pool.mamba_allocator.free(
                mamba_transfer.device_indices[:1]
            )

        # Insert already resolved every overlap. Read the committed Full path
        # directly (without matching the key again), then keep only provisional
        # component pages whose slots survived insert.
        canonical_tail = cache.tree_core.collect_full_device_indices(
            insert_result.last_device_node, req.last_node
        )
        assert canonical_tail.numel() == len(tail_hashes) * cache.page_size
        load_transfers = self._filter_load_pages_after_insert(
            component_transfers, canonical_tail
        )

        if load_transfers and not self.cache_linker.load(req.rid, load_transfers):
            raise RuntimeError(f"Failed to queue the linker load for {req.rid=}.")

        node = cache.resolve_node_handle(insert_result.last_device_node)
        while node.id != req.last_node:
            node.external_cache_stored = True
            node = node.parent
        return canonical_tail, insert_result.last_device_node

    def _finish_load(
        self,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        success: bool,
    ) -> None:
        """Let every component that reserved room commit it, or release it."""
        if not component_transfers:
            return
        full = component_transfers[0][1]
        for component, transfer in component_transfers:
            component.finish_external_linker_load(
                req, full, transfer, prefix_len, success
            )

    def _filter_load_pages_after_insert(
        self,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        canonical_full_tail: torch.Tensor,
    ) -> list[PoolTransfer]:
        """Drop provisional pages that insert replaced with resident L1 slots."""
        result = []
        page = self.cache.page_size
        for _, transfer in component_transfers:
            if transfer.name == PoolName.KV:
                canonical = canonical_full_tail
            else:
                # Mamba's direct-linker hooks currently reject this mode, so the
                # only supported auxiliary component here is SWA.
                assert transfer.name == PoolName.SWA
                swa_len = transfer.device_indices.numel()
                canonical = self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    canonical_full_tail[-swa_len:]
                ).to(
                    transfer.device_indices
                )

            incoming_pages = transfer.device_indices.reshape(-1, page)
            canonical_pages = canonical.to(transfer.device_indices).reshape(-1, page)
            assert incoming_pages.shape == canonical_pages.shape
            assert incoming_pages.shape[0] == len(transfer.keys)

            # ``insert`` resolves each provisional page against the canonical
            # page currently stored in the tree:
            #
            # 1. canonical == provisional: ``insert`` kept the newly allocated,
            #    still-empty page. Keep it in this transfer so remote storage
            #    can populate it.
            # 2. canonical != provisional: ``insert`` deduplicated against an
            #    existing L1 page and already released the provisional slots.
            #    Exclude it from this transfer; the canonical page has the data.
            slot_matches = incoming_pages == canonical_pages
            pages_to_load = slot_matches.all(dim=1)
            pages_deduplicated = (~slot_matches).all(dim=1)
            assert bool(
                (pages_to_load | pages_deduplicated).all().item()
            ), "insert must keep or replace a whole page"

            page_ids_to_load = pages_to_load.nonzero(as_tuple=True)[0].tolist()
            if page_ids_to_load:
                transfer.keys = [transfer.keys[i] for i in page_ids_to_load]
                transfer.device_indices = incoming_pages[pages_to_load].reshape(-1)
                result.append(transfer)
        return result

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            if not self.cache.resolve_node_handle(node_id).external_cache_stored:
                self._offload_node(node_id)

    def _offload_node(self, node_id: NodeId) -> None:
        cache = self.cache
        node = cache.resolve_node_handle(node_id)
        transfers = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.OFFLOAD, node, None
            )
            if transfer is not None:
                transfers.append(transfer)

        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        node.external_cache_stored = True
        self.pending_offloads.append((node_id, lock_params))

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def drain_offloads(self, finish_count: int) -> None:
        assert finish_count <= len(self.pending_offloads)
        for _ in range(finish_count):
            node_id, lock_params = self.pending_offloads.pop(0)
            node = self.cache.resolve_node_handle(node_id)
            node.external_cache_stored = self.cache_linker.pop_completed_offload()
            self.cache.dec_lock_ref(node_id, lock_params)

    def start_layer_wise_loading(self) -> int:
        return self.cache_linker.start_layer_wise_loading()

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self.hit_markers.clear()
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)

    def close(self) -> None:
        self.cache_linker.close()
