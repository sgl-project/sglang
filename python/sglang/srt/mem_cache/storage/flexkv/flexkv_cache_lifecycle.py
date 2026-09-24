"""Shared request identity, restore ownership, and prefetch lifecycle for FlexKV.

Tree insertion, SWA allocation, and store node locking stay in the adapters.
This mixin owns only the protocol shared by ordinary and hybrid caches.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def _request_key(handle: CacheRequestHandle) -> str:
    """Keep attempt identity in FlexKV's string tracking keys."""
    return json.dumps([handle.rid, handle.attempt_id], separators=(",", ":"))


def _namespace_kwargs(connector, extra_key, cache_salt) -> Optional[dict]:
    """Return scoped connector arguments, or None when reuse is unsupported.

    A single JSON component distinguishes null, empty strings and embedded
    separators without depending on FlexKV's namespace component delimiter.
    Unscoped requests keep their historical hash and connector call signature.
    """
    if extra_key is None and cache_salt is None:
        return {}
    if getattr(connector, "supports_cache_namespace", False) is not True:
        return None
    return {
        "namespace": [
            json.dumps(
                ["sglang-cache-v1", extra_key, cache_salt], separators=(",", ":")
            )
        ]
    }


@dataclass
class _RestoreLease:
    """Fresh destination slots retained until cache commit or a fenced reset."""

    generation: int
    rid: str
    req: Req
    device_indices: torch.Tensor


class FlexKVCacheLifecycleMixin:
    def _init_restore_state(self) -> None:
        self._restore_leases: dict[str, _RestoreLease] = {}
        self._aborted_restore_leases: dict[int, _RestoreLease] = {}
        self._restore_generation = 0

    def _register_restore_lease(self, req: Req, slots: torch.Tensor) -> _RestoreLease:
        rid = _request_key(req.cache_request_handle)
        if rid in self._restore_leases:
            raise RuntimeError(f"FlexKV duplicate load-back: rid={rid}")
        lease = _RestoreLease(
            generation=self._restore_generation, rid=rid, req=req, device_indices=slots
        )
        self._restore_generation += 1
        self._restore_leases[rid] = lease
        req.pending_restore_generation = lease.generation
        req.pending_restore_slots = slots
        return lease

    def has_uncommitted_restore(self, req: Req) -> bool:
        return _request_key(req.cache_request_handle) in self._restore_leases

    @staticmethod
    def _restore_lease_matches_req(req: Req, lease: _RestoreLease) -> bool:
        return (
            lease.req is req
            and getattr(req, "pending_restore_generation", None) == lease.generation
            and getattr(req, "pending_restore_slots", None) is lease.device_indices
        )

    def _validate_restore_lease(self, req: Req) -> Optional[_RestoreLease]:
        lease = self._restore_leases.get(_request_key(req.cache_request_handle))
        if lease is None or lease.req is not req:
            # An older aborted Req may finish after a new Req reused its rid.
            # Find by object identity, never commit the successor's lease.
            lease = next(
                (
                    item
                    for item in self._aborted_restore_leases.values()
                    if item.req is req
                ),
                lease,
            )
        if lease is not None and not self._restore_lease_matches_req(req, lease):
            # Ordinary completion mutates/frees KV. Continuing on a mismatch
            # could free a different owner's slots and free them again at reset.
            raise RuntimeError(
                f"FlexKV restore lease mismatch: rid={_request_key(req.cache_request_handle)}"
            )
        return lease

    def _forget_restore_lease(self, lease: _RestoreLease) -> None:
        if self._restore_leases.get(lease.rid) is lease:
            self._restore_leases.pop(lease.rid)
        if self._aborted_restore_leases.get(lease.generation) is lease:
            self._aborted_restore_leases.pop(lease.generation)
        # Reset trusts the allocation ledger, not mutable request fields. Do
        # not overwrite fields belonging to another generation of the Req.
        if self._restore_lease_matches_req(lease.req, lease):
            lease.req.pending_restore_generation = None
            lease.req.pending_restore_slots = None
            lease.req._flexkv_uncached_restore = False

    def _commit_restore(self, req: Req) -> None:
        lease = self._validate_restore_lease(req)
        if lease is not None:
            self._forget_restore_lease(lease)
        else:
            req._flexkv_uncached_restore = False

    def _free_uncommitted_restores(self) -> None:
        # Only call after connector.reset has fenced all DMA. Request metadata
        # may be stale; each ledger entry still identifies the allocation to free.
        failed = []
        leases = list(self._restore_leases.values()) + list(
            self._aborted_restore_leases.values()
        )
        for lease in leases:
            try:
                self.token_to_kv_pool_allocator.free(lease.device_indices)
            except Exception:
                logger.exception(
                    "FlexKV failed to free restore slots rid=%s", lease.rid
                )
                failed.append(lease.rid)
                continue
            self._forget_restore_lease(lease)
        if failed:
            # Attempt every allocation, retain failures for diagnosis/retry, and
            # do not report a successful reset or discard the remaining ledger.
            raise RuntimeError(f"FlexKV failed to free restore allocations: {failed}")

    def prefetch_request(self, req: Req) -> None:
        """Start queued prefetch without a foreground lookup or H2D allocation."""
        # Foreground lookup runs after stop-and-drain, otherwise it could fetch
        # the whole remote prefix before the prefetch policy gets a chance to stop.
        req.init_next_round_input(tree_cache=None, cow_mamba=False)
        fill_ids = req.full_untruncated_fill_ids
        if not fill_ids:
            return
        match_end = req._compute_max_prefix_len(len(fill_ids))
        self.prefetch_from_storage(
            req.cache_request_handle,
            None,
            fill_ids[:match_end],
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
        )

    def prefetch_from_storage(
        self,
        handle: CacheRequestHandle,
        last_host_node=None,
        token_ids=None,
        last_hash=None,
        prefix_keys=None,
        *,
        matched_prefix_tokens=None,
        extra_key=None,
        cache_salt=None,
    ) -> None:
        """Pass the complete token hash chain and the candidate's absolute offset."""
        rid = _request_key(handle)
        del last_host_node, last_hash, prefix_keys
        namespace_kwargs = _namespace_kwargs(
            self.flexkv_connector, extra_key, cache_salt
        )
        if namespace_kwargs is None or not token_ids:
            return
        prefix = [] if matched_prefix_tokens is None else list(matched_prefix_tokens)
        ids = prefix + list(token_ids)
        ids = ids[: len(ids) // self.page_size * self.page_size]
        if len(ids) <= len(prefix):
            return
        if getattr(self.flexkv_connector, "_chunked_prefetch", False):
            self.flexkv_connector.prefetch_async(
                rid,
                ids,
                sglang_req_id=handle.rid,
                candidate_start_token=len(prefix),
                **namespace_kwargs,
            )
        else:
            self.flexkv_connector.prefetch_async(
                rid, ids, sglang_req_id=handle.rid, **namespace_kwargs
            )

    def check_prefetch_progress(self, handle: CacheRequestHandle) -> bool:
        rid = _request_key(handle)
        return self.flexkv_connector.check_prefetch_progress(rid)

    def terminate_prefetch(self, handle: CacheRequestHandle) -> None:
        rid = _request_key(handle)
        self.flexkv_connector.cancel_prefetch(rid)

    def pop_prefetch_loaded_span(
        self, handle: CacheRequestHandle
    ) -> tuple[int, Optional[int]]:
        rid = _request_key(handle)
        if getattr(self.flexkv_connector, "_chunked_prefetch", False):
            return self.flexkv_connector.pop_prefetch_loaded_span(rid)
        return self.pop_prefetch_loaded_tokens(handle), None

    def pop_prefetch_loaded_tokens(self, handle: CacheRequestHandle) -> int:
        rid = _request_key(handle)
        pop = getattr(self.flexkv_connector, "pop_prefetch_loaded_tokens", None)
        if callable(pop):
            return int(pop(rid))
        # Older FlexKV builds do not track the materialized REMOTE2H prefix.
        # Reporting 0 attributes the whole hit to the host tier, which only
        # skews the #cached-host / #cached-storage split in the logs.
        del rid
        return 0
