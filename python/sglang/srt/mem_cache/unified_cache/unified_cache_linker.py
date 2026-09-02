"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, NamedTuple, Optional, Protocol

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InsertParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


@dataclass(frozen=True)
class LinkerKeyDomain:
    """Inputs that namespace KV content beyond its token sequence."""

    cache_salt: Optional[str]
    extra_key: Optional[str]


class LinkerKeyCodec(Protocol):
    """Backend-owned incremental codec for stable external page keys."""

    codec_id: str

    def extend_pages(
        self,
        *,
        parent_key: Optional[bytes],
        page_tokens: Sequence[int],
        page_size: int,
        key_domain: LinkerKeyDomain,
    ) -> list[bytes]: ...


class LinkerCancelOutcome(str, Enum):
    """Terminal state observed while releasing one request's linker work."""

    LOOKUP_RELEASED = "lookup_released"
    QUEUED_LOAD_CANCELLED = "queued_load_cancelled"
    SUBMITTED_LOAD_RETAINED = "submitted_load_retained"
    NOT_FOUND = "not_found"


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object
    key_codec: Optional[LinkerKeyCodec] = None

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
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    def cancel_request(self, rid: str) -> LinkerCancelOutcome:
        """Release lookup/queued state while retaining submitted DMA guards.

        The default preserves compatibility with linkers that only implement
        ``cancel_queued_load``.  New backends should override this method so a
        lookup ticket and a submitted load are distinguishable.
        """
        if self.cancel_queued_load(rid):
            return LinkerCancelOutcome.QUEUED_LOAD_CANCELLED
        return LinkerCancelOutcome.NOT_FOUND

    def has_pending_operations(self) -> bool:
        """Return backend work not otherwise represented by wrapper state."""
        return False

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> list[str]:
        """Consume the oldest completed load batch and return its request IDs."""

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``prefix_key`` covers the device-cached prefix plus the restorable tail, so
    it is what gets inserted once the tail lands. ``tail_hashes`` are the
    per-page storage hashes of that tail alone, starting at ``device_hit_len``.
    """

    prefix_key: RadixKey
    tail_hashes: list[str]
    tail_linker_keys: Optional[list[bytes]]
    device_hit_len: int


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


@dataclass
class PendingExternalLoad:
    """Tree transaction retained until an external H2D reaches completion."""

    rid: str
    inserted_node: NodeId
    anchor_node: NodeId
    adopted_ranges: dict
    allocated_component_slots: dict[PoolName, torch.Tensor]
    lock_params: DecLockRefParams
    phase: Literal["queued", "submitted"] = "queued"


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        self.cache = cache
        self.cache_linker = cache_linker
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Loads in flight, each pinning its inserted endpoint until DMA completes.
        self.pending_loads: dict[str, PendingExternalLoad] = {}
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1
        if cache_linker.key_codec is not None:
            set_codec = getattr(cache.tree_core, "set_linker_key_codec", None)
            if set_codec is None:
                raise RuntimeError(
                    "This tree core does not support linker-owned key sidecars."
                )
            set_codec(cache_linker.key_codec)

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    def has_pending_io(self) -> bool:
        return bool(
            self.hit_markers
            or self.pending_loads
            or self.pending_offloads
            or self.cache_linker.has_pending_operations()
        )

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        # Cache-aware scheduling can probe a waiting request once for priority
        # and again for admission.  A lookup ticket pins external blocks, so
        # replace any unconsumed probe instead of leaking it or asking the
        # backend to accept a duplicate request ID.
        if req.rid in self.hit_markers:
            self.release_request(req.rid)

        cache = self.cache
        page = cache.page_size
        device_hit_len = int(result.device_indices.numel())
        if device_hit_len >= len(key):
            return result

        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        if not tail_hashes:
            return result
        tail_linker_keys = self._tail_linker_keys(key, result, device_hit_len)
        if self.cache_linker.key_codec is not None and tail_linker_keys is None:
            return result

        lookup_transfers = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, tail_hashes
            )
            if transfer is None:
                return result
            self._set_transfer_linker_keys(transfer, tail_linker_keys)
            lookup_transfers.append(transfer)
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        local_restorable = self.cache_linker.lookup(req.rid, lookup_transfers)
        try:
            hit_pages = self._sync_restorable_prefix(
                local_restorable,
                num_pages=len(tail_hashes),
                device_hit_pages=0,
            )
        except BaseException:
            self.cache_linker.cancel_request(req.rid)
            raise
        if hit_pages == 0:
            # A rank can hold a local lookup ticket even when the TP/CP
            # intersection is empty.  No hit marker will own that ticket.
            self.cache_linker.cancel_request(req.rid)
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
            tail_linker_keys=(
                list(tail_linker_keys[:hit_pages])
                if tail_linker_keys is not None
                else None
            ),
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

    def _tail_linker_keys(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> Optional[list[bytes]]:
        codec = self.cache_linker.key_codec
        if codec is None:
            return None

        parent_key = None
        if device_hit_len > 0:
            parent_key = self.cache.tree_core.get_last_linker_key_value(
                result.last_device_node
            )
            if parent_key is None:
                return None

        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        linker_keys = codec.extend_pages(
            parent_key=parent_key,
            page_tokens=list(key[device_hit_len : device_hit_len + tail_len]),
            page_size=page,
            key_domain=LinkerKeyDomain(
                cache_salt=key.cache_salt,
                extra_key=key.extra_key,
            ),
        )
        expected = tail_len // page
        if len(linker_keys) != expected or any(
            not isinstance(linker_key, bytes) for linker_key in linker_keys
        ):
            raise RuntimeError(
                f"Linker key codec {codec.codec_id!r} returned "
                f"{len(linker_keys)} keys for {expected} pages."
            )
        return linker_keys

    @staticmethod
    def _set_transfer_linker_keys(
        transfer: PoolTransfer, linker_keys: Optional[Sequence[bytes]]
    ) -> None:
        if linker_keys is None:
            return
        if transfer.keys is None or len(transfer.keys) != len(linker_keys):
            raise RuntimeError(
                f"External linker key count does not match {transfer.name} transfer."
            )
        transfer.linker_keys = list(linker_keys)

    # ---- init_load_back: remote -> device, then insert ----

    def load_back(self, req: Req) -> tuple[torch.Tensor, NodeId]:
        cache = self.cache
        empty_indices = cache.tree_core.empty_match_result.device_indices
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return empty_indices, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        tail_linker_keys = hit.tail_linker_keys
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in cache._components_tuple:
            try:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, tail_hashes
                )
            except BaseException:
                self._abort_prepared_load(req, component_transfers, prefix_len)
                raise
            if transfer is None:
                self._abort_prepared_load(req, component_transfers, prefix_len)
                return empty_indices, req.last_node
            component_transfers.append((component, transfer))
            try:
                self._set_transfer_linker_keys(transfer, tail_linker_keys)
            except BaseException:
                self._abort_prepared_load(req, component_transfers, prefix_len)
                raise

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

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
                track_adopted_ranges=True,
            )
        )
        if mamba_transfer is not None and insert_result.mamba_exist:
            cache.req_to_token_pool.mamba_allocator.free(
                mamba_transfer.device_indices[:1]
            )

        canonical_tail = cache.tree_core.collect_full_device_indices(
            insert_result.last_device_node, req.last_node
        )
        assert canonical_tail.numel() == len(tail_hashes) * cache.page_size
        load_transfers = self._update_load(
            ExternalLinkerLoadPhase.COMMIT,
            req,
            component_transfers,
            prefix_len,
            insert_result=insert_result,
            canonical_full=canonical_tail,
        )

        self._queue_load(
            req.rid,
            insert_result.last_device_node,
            load_transfers,
            anchor_node=req.last_node,
            adopted_ranges=insert_result.adopted_ranges or {},
        )

        node = cache.resolve_node_handle(insert_result.last_device_node)
        while node.id != req.last_node:
            node.external_cache_stored = True
            node = node.parent
        return canonical_tail, insert_result.last_device_node

    def _abort_prepared_load(
        self,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
    ) -> None:
        try:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT,
                req,
                component_transfers,
                prefix_len,
            )
        finally:
            # ``hit`` has already been popped, so no tree marker remains to
            # own the backend's lookup ticket after preparation fails.
            self.cache_linker.cancel_request(req.rid)

    def _queue_load(
        self,
        rid: str,
        node_id: NodeId,
        transfers: list[PoolTransfer],
        *,
        anchor_node: Optional[NodeId] = None,
        adopted_ranges: Optional[dict] = None,
    ) -> None:
        if not transfers:
            self.cache_linker.cancel_request(rid)
            return
        assert rid not in self.pending_loads
        lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
        pending = PendingExternalLoad(
            rid=rid,
            inserted_node=node_id,
            anchor_node=node_id if anchor_node is None else anchor_node,
            adopted_ranges={} if adopted_ranges is None else adopted_ranges,
            allocated_component_slots={
                transfer.name: transfer.device_indices
                for transfer in transfers
                if getattr(transfer, "device_indices", None) is not None
            },
            lock_params=lock_params,
        )
        try:
            queued = self.cache_linker.load(rid, transfers)
        except BaseException:
            try:
                self.cache_linker.cancel_request(rid)
            finally:
                self._rollback_pending_load(pending)
            raise
        if not queued:
            self.cache_linker.cancel_request(rid)
            self._rollback_pending_load(pending)
            raise RuntimeError(f"Failed to queue the linker load for rid={rid!r}.")
        self.pending_loads[rid] = pending

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        insert_result=None,
        canonical_full: torch.Tensor | None = None,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            component_canonical = canonical_full
            if phase == ExternalLinkerLoadPhase.COMMIT:
                assert insert_result.adopted_ranges is not None
                coverage_start = prefix_len - len(transfer.device_indices)
                ranges = [
                    (max(start, coverage_start), min(end, prefix_len))
                    for start, end in insert_result.adopted_ranges.get(
                        component.component_type, ()
                    )
                    if max(start, coverage_start) < min(end, prefix_len)
                ]
                indices, keys = self._select_adopted_pages(
                    transfer.device_indices,
                    ranges,
                    prefix_len,
                    transfer.keys,
                )
                if not keys:
                    continue
                linker_keys = None
                if transfer.linker_keys is not None:
                    _, linker_keys = self._select_adopted_pages(
                        transfer.device_indices,
                        ranges,
                        prefix_len,
                        transfer.linker_keys,
                    )
                transfer.device_indices = indices
                transfer.keys = keys
                transfer.linker_keys = linker_keys
                component_canonical, _ = self._select_adopted_pages(
                    canonical_full, ranges, prefix_len
                )
            transfer = component.update_external_linker_load(
                phase,
                req,
                full,
                transfer,
                prefix_len,
                insert_result=insert_result,
                canonical_full=component_canonical,
            )
            if transfer is not None:
                result.append(transfer)
        return result

    def _select_adopted_pages(
        self,
        indices: torch.Tensor,
        ranges: Sequence[tuple[int, int]],
        prefix_len: int,
        keys: Sequence | None = None,
    ) -> tuple[torch.Tensor, list]:
        page = self.cache.page_size
        coverage_start = prefix_len - len(indices)
        pages = indices.reshape(-1, page)
        if keys is not None:
            assert len(keys) == len(pages)

        chunks = []
        selected_keys = []
        for start, end in ranges:
            start = max(start, coverage_start)
            end = min(end, prefix_len)
            if start >= end:
                continue
            assert (start - coverage_start) % page == 0
            assert (end - coverage_start) % page == 0
            first = (start - coverage_start) // page
            last = (end - coverage_start) // page
            chunks.append(pages[first:last].reshape(-1))
            if keys is not None:
                selected_keys.extend(keys[first:last])

        if not chunks:
            return indices[:0], selected_keys
        selected = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return selected, selected_keys

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
                if self.cache_linker.key_codec is not None:
                    linker_keys = cache.tree_core.get_linker_key_values(node_id)
                    self._set_transfer_linker_keys(transfer, linker_keys)
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

        cache.tree_core.mark_write_through_pending(node_id)
        node.external_cache_stored = True
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def num_completed_loads(self) -> int:
        return self.cache_linker.num_completed_loads()

    def drain_loads(self, finish_count: int) -> None:
        for _ in range(finish_count):
            for rid in self.cache_linker.pop_completed_load():
                pending = self.pending_loads.pop(rid)
                self.cache.dec_lock_ref(pending.inserted_node, pending.lock_params)

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            for node_id in pending.publish_node_ids:
                node = self.cache.resolve_node_handle(node_id)
                if node.write_through_pending_id == pending.lock_node_id:
                    node.write_through_pending_id = None
                node.external_cache_stored = success
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        consumer_index = self.cache_linker.start_layer_wise_loading()
        if consumer_index >= 0:
            for pending in self.pending_loads.values():
                if pending.phase == "queued":
                    pending.phase = "submitted"
        return consumer_index

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self.hit_markers.clear()
        self._release_pending_locks()

    def _release_pending_locks(self) -> None:
        for pending in self.pending_loads.values():
            self.cache.dec_lock_ref(pending.inserted_node, pending.lock_params)
        self.pending_loads.clear()
        for pending in self.pending_offloads:
            for node_id in pending.publish_node_ids:
                node = self.cache.resolve_node_handle(node_id)
                if node.write_through_pending_id == pending.lock_node_id:
                    node.write_through_pending_id = None
                node.external_cache_stored = False
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)
        outcome = self.cache_linker.cancel_request(rid)
        if outcome is LinkerCancelOutcome.QUEUED_LOAD_CANCELLED:
            pending = self.pending_loads.pop(rid)
            if pending.phase != "queued":
                raise RuntimeError(f"Linker cancelled submitted load for rid={rid!r}.")
            self._rollback_pending_load(pending)
        elif outcome is LinkerCancelOutcome.SUBMITTED_LOAD_RETAINED:
            pending = self.pending_loads.get(rid)
            if pending is None or pending.phase != "submitted":
                raise RuntimeError(
                    f"Linker retained unknown submitted load for rid={rid!r}."
                )
        elif (
            outcome is LinkerCancelOutcome.LOOKUP_RELEASED and rid in self.pending_loads
        ):
            raise RuntimeError(f"Linker lost queued load state for rid={rid!r}.")
        elif outcome is LinkerCancelOutcome.NOT_FOUND and rid in self.pending_loads:
            raise RuntimeError(f"Linker lost pending load state for rid={rid!r}.")

    def _rollback_pending_load(self, pending: PendingExternalLoad) -> None:
        """Undo tree adoption before releasing the transaction's node lock."""
        try:
            if pending.adopted_ranges:
                self.cache.rollback_external_load(
                    anchor_node=pending.anchor_node,
                    inserted_node=pending.inserted_node,
                    adopted_ranges=pending.adopted_ranges,
                    allocated_component_slots=pending.allocated_component_slots,
                    lock_params=pending.lock_params,
                )
        finally:
            self.cache.dec_lock_ref(pending.inserted_node, pending.lock_params)

    def close(self) -> None:
        self.cache_linker.close()
        self._release_pending_locks()
