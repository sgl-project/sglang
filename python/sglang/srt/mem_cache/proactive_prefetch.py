"""Bounded requestless L3-to-resident-L2 restoration; scheduler-thread only."""

import time
import uuid
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType


@dataclass
class _Restore:
    operation_id: str
    tokens: tuple[int, ...]
    cache_salt: Optional[str]
    ttl_ms: int
    handle: CacheRequestHandle
    started: float
    deadline: float
    operation: object = None
    state: str = "RUNNING"
    finished: Optional[float] = None
    restored_tokens: int = 0


class ProactivePrefetch:
    """One active restore, with bounded recent outcomes and ordinary ownership.

    TTL limits in-flight work only. Published host pages are normally evictable;
    cancellation after publication does not evict or pin shared cache state.
    """

    def __init__(self, cache):
        if (
            not cache.enable_storage
            or cache.host_memory_mode != "cache"
            or set(cache.tree_components) != {ComponentType.FULL}
            or cache.sidecar_pool_specs
            or cache.tree_core.is_eagle
            or cache.cache_controller.storage_backend.__class__.__name__
            != "HiCacheFile"
        ):
            raise ValueError("Requires resident FULL KV-only HiCache with file storage")
        self.cache = cache
        self._backend = cache.cache_controller.storage_backend
        self.active: Optional[_Restore] = None
        self._waiting_req = None
        self.records: OrderedDict[str, _Restore] = OrderedDict()

    def tick(self):
        record = self.active
        if record is None:
            return
        if record.state == "RUNNING" and time.monotonic() >= record.deadline:
            self.cancel(record.operation_id, state="EXPIRED")
        if record.handle in self.cache.ongoing_prefetch:
            return
        if record.state == "RUNNING":
            record.restored_tokens, _ = self.cache.pop_prefetch_loaded_span(
                record.handle
            )
            record.state = record.operation.terminal_outcome or "MISS"
            record.finished = time.monotonic()
        self.cache.discard_storage_prefetch_accounting(record.handle)
        self.cache.storage_prefetch_retries.cancel(record.handle.rid)
        # An allocated cancelled read still owns its tail until terminal ACK.
        # Keep admission of another control operation bounded until it drains.
        if (
            record.operation.host_indices is None
            or record.operation.terminal_ack_consumed
        ):
            self.active = None
            self._waiting_req = None

    def submit(self, operation_id, input_ids, cache_salt=None, ttl_ms=10000):
        if (
            not self.cache.enable_storage
            or self.cache.cache_controller.storage_backend is not self._backend
        ):
            raise ValueError("Storage backend changed; restart the fixed-model server")
        self.tick()
        if not operation_id or len(operation_id) > 128:
            raise ValueError("operation_id must contain 1..128 characters")
        if cache_salt is not None and len(cache_salt) > 256:
            raise ValueError("cache_salt must contain at most 256 characters")
        if not 1 <= ttl_ms <= 60000:
            raise ValueError("ttl_ms must be in 1..60000")
        key = RadixKey(array("q", input_ids), cache_salt=cache_salt).page_aligned(
            self.cache.page_size
        )
        if len(key) < self.cache.prefetch_threshold:
            raise ValueError("Prefix is shorter than the storage prefetch threshold")
        tokens = tuple(key.token_ids)
        if operation_id in self.records:
            record = self.records[operation_id]
            if (record.tokens, record.cache_salt, record.ttl_ms) != (
                tokens,
                key.cache_salt,
                ttl_ms,
            ):
                raise ValueError("operation_id already identifies a different restore")
            return self.status(operation_id)
        if self.active is not None:
            raise ValueError("One proactive restore is already active")
        match = self.cache.match_prefix(MatchPrefixParams(key=key))
        matched = len(match.device_indices) + match.host_hit_length
        anchor = match.last_host_node
        if matched < len(key) and not (
            self.cache.is_root(anchor) or self.cache.is_backuped(anchor)
        ):
            raise ValueError("The matched anchor is not backed by host KV")
        now = time.monotonic()
        record = _Restore(
            operation_id,
            tokens,
            key.cache_salt,
            ttl_ms,
            CacheRequestHandle("__proactive__" + uuid.uuid4().hex, 0),
            now,
            now + ttl_ms / 1000,
        )
        if matched >= len(key):
            record.state, record.finished = "CACHED", now
        else:
            if len(key) - matched < self.cache.prefetch_threshold:
                raise ValueError("Uncached suffix is below the prefetch threshold")
            self.cache.prefetch_from_storage(
                record.handle,
                anchor,
                array("q", tokens[matched:]),
                self.cache.get_last_hash_value(anchor),
                (
                    self.cache.get_prefix_hash_values(anchor)
                    if self.cache.hicache_storage_pass_prefix_keys
                    else None
                ),
                matched_prefix_tokens=array("q", tokens[:matched]),
                cache_salt=key.cache_salt,
            )
            info = self.cache.ongoing_prefetch.get(record.handle)
            if info is None:
                self.cache.storage_prefetch_retries.cancel(record.handle.rid)
                record.state, record.finished = "DECLINED", now
            else:
                record.operation = info.operation
                self.active = record
        self.records[operation_id] = record
        while len(self.records) > 32:
            self.records.popitem(last=False)
        return self.status(operation_id)

    def cancel(self, operation_id, state="CANCELLED"):
        record = self.records[operation_id]
        if record.state == "RUNNING":
            # Ownership/ACK cleanup is entirely the existing controller contract.
            self._waiting_req = None
            self.cache.release_aborted_request(record.handle)
            record.state, record.finished = state, time.monotonic()
        return self.status(operation_id)

    def blocks(self, req):
        # A queued Req keeps its original token prefix. Compare it once, not on
        # every idle scheduling poll (which otherwise contends with file I/O).
        if self.active is None or self.active.state != "RUNNING":
            return False
        if req is self._waiting_req:
            return True
        if self.waits_for(req):
            self._waiting_req = req
            return True
        return False

    def waits_for(self, req):
        record = self.active
        if record is None or record.state != "RUNNING":
            return False
        return (
            req.extra_key is None
            and (req.cache_salt or None) == record.cache_salt
            and tuple(req.origin_input_ids[: len(record.tokens)]) == record.tokens
        )

    def status(self, operation_id):
        record = self.records[operation_id]
        pool = self.cache.cache_controller.mem_pool_host
        end = record.finished or time.monotonic()
        return {
            "operation_id": operation_id,
            "state": record.state,
            "requested_tokens": len(record.tokens),
            "restored_tokens": record.restored_tokens,
            "restored_bytes": record.restored_tokens
            * pool.anchor_entry.host_pool.get_size_per_token(),
            "elapsed_ms": (end - record.started) * 1000,
            "cleanup_pending": record is self.active and record.state != "RUNNING",
            "host_available_tokens": pool.available_size(),
            "inflight_tokens": self.cache.cache_controller.prefetch_tokens_occupied,
        }
