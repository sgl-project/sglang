from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""
import itertools
import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.connector import BaseConnector
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.mem_cache.session_cache.base_meta import (
    BaseSessionCacheMetaManager,
    SessionCacheEntry,
    SessionCacheMeta,
)
from sglang.srt.mem_cache.session_cache.lru_cache import (
    LRUSessionCache,
    LRUSessionCacheEntry,
    LRUSessionCacheStatus,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class SessionCache(BasePrefixCache):
    def __init__(
        self,
        tp: int,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        metadata_manager: BaseSessionCacheMetaManager,
        connectors: List[BaseConnector],
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.tp = tp
        self.dtype = self.token_to_kv_pool.dtype
        self.device = "cuda:" + str(tp)
        self.disable = False

        self.evictable_size_ = 0
        self.protected_size_ = 0

        self.lru_cache = LRUSessionCache()
        self.metadata_manager = metadata_manager

        self.connectors = connectors
        self.connector_map = {}
        for connector in self.connectors:
            self.connector_map[connector.get_url()] = connector
        self.connector_key_cycle = itertools.cycle(self.connector_map.keys())

        self.enable_hierarchical_cache = len(self.connectors) > 0

    def _get_prefix_path(self, sid):
        return sid + "/" + str(self.tp)

    def reset(self):
        self.evictable_size_ = 0
        self.protected_size_ = 0

        self.lru_cache.reset()
        self.metadata_manager.reset()

    def match_prefix(self, sid: Optional[str], key: List[int], **kwargs):
        if sid is None:
            if self.enable_hierarchical_cache:
                return [], None, None
            else:
                return [], None

        lru_entry = self.lru_cache.get(sid)
        meta = self.metadata_manager.load(sid)
        if lru_entry == None:
            if self.enable_hierarchical_cache:
                return [], lru_entry, meta
            else:
                return [], lru_entry

        kv_indices = lru_entry.get_kv_indices()

        # TODO(luchangqi): add content consistency verification
        # Currently, it is assumed that within a session,
        # the prompt for a request consists of the concatenation
        # of all previous n-1 rounds of prompts and their
        # corresponding answers, combined with the n round of questioning.

        assert len(key) >= len(
            kv_indices
        ), "content of the prompt in this session is incorrect."

        if self.enable_hierarchical_cache:
            # Assign the metadata information inside the GPU
            # to `req.last_node`, and assign the external
            # metadata information to `req.last_mode_global`.
            return kv_indices, lru_entry, meta
        else:
            return kv_indices, lru_entry

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]

        if req.session_id is None:
            # When the request does not include a `session_id`,
            # the request is invalid, and the memory should be released.
            self.req_to_token_pool.free(req.req_pool_idx)
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        old_lru_entry = self.lru_cache.get(req.session_id)
        if old_lru_entry is not None:
            old_token_length = old_lru_entry.get_length()
            assert (
                len(kv_indices) >= old_token_length
            ), "content of the prompt in this session is incorrect."

            # To ensure the correctness of `evictable_size_`,
            # whenever the KV cache is updated, subtract the
            # length of the previous `kv_indices` and add the
            # length of the latest `kv_indices`.
            self.evictable_size_ -= old_token_length

        # kv_indices includes all the KV caches of the n rounds
        # of the current session. Use the new session cache to
        # overwrite the previous session cache.
        self.lru_cache.set(req.session_id, kv_indices, LRUSessionCacheStatus.FINISHED)

        self.evictable_size_ += len(kv_indices)

        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if req.session_id is None:
            return

        if token_ids is None:
            token_id_len = len(req.fill_ids)
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]

        old_lru_entry = self.lru_cache.get(req.session_id)
        if old_lru_entry is None:
            old_token_length = 0
        else:
            old_token_length = old_lru_entry.get_length()

        self.lru_cache.set(req.session_id, kv_indices, LRUSessionCacheStatus.UNFINISHED)
        assert (
            len(kv_indices) >= old_token_length
        ), "content of the prompt in this session is incorrect."
        self.evictable_size_ -= old_token_length
        self.evictable_size_ += len(kv_indices)

        assert len(kv_indices) >= len(req.prefix_indices)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(kv_indices))),
            kv_indices[len(req.prefix_indices) :],
        )

        new_lru_entry = self.lru_cache.get(req.session_id)
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_lru_entry)
        req.prefix_indices = kv_indices
        req.last_node = new_lru_entry

    def write_backup(self, lru_entry: LRUSessionCacheEntry):
        meta = self.metadata_manager.load(lru_entry.sid)
        if meta is None:
            meta = SessionCacheMeta(lru_entry.sid)

        length = meta.get_length()
        kv_indices = lru_entry.get_kv_indices()
        lru_length = len(kv_indices)
        assert lru_length >= length

        if length == 0:
            connector_uri = next(self.connector_key_cycle)
            connector = self.connector_map[connector_uri]
        else:
            connector_uri = meta.get_entries()[0].connector
            connector = self.connector_map[connector_uri]

        uri, start, inc_length = meta.get_next_entry_info(
            self._get_prefix_path(meta.sid), lru_length
        )

        kv_caches = self.token_to_kv_pool.get_flat_data(
            lru_entry.get_kv_indices()[length:]
        )

        ret = connector.set(uri, kv_caches)
        if ret == 0:
            meta.append(SessionCacheEntry(connector_uri, uri, start, inc_length))
            self.metadata_manager.save(
                meta.sid,
                meta,
            )
        else:
            logging.error("failed to save kv cache.")

        self.token_to_kv_pool_allocator.free(kv_indices)
        self.evictable_size_ -= lru_length

    def init_load_back(
        self,
        meta: SessionCacheMeta,
        prefix_indices,
    ):
        assert meta is not None
        lru_entry = self.lru_cache.get(meta.sid)
        assert lru_entry is None

        return self.load_back(meta)

    def load_back(
        self,
        meta: SessionCacheMeta,
    ):
        token_length = meta.get_length()
        device_indices = self.token_to_kv_pool_allocator.alloc(token_length)

        if device_indices is None:
            return None, []

        self.lru_cache.set(meta.sid, device_indices, LRUSessionCacheStatus.LOADING)
        lru_entry = self.lru_cache.get(meta.sid)
        self.inc_lock_ref(lru_entry)

        length = meta.get_length()
        assert length == len(device_indices)

        meta_entries = meta.get_entries()
        for meta_entry in meta_entries:
            meta_entry_len = meta_entry.length
            offset = meta_entry.offset
            shape = self.token_to_kv_pool.get_kv_cache_shape(meta_entry_len)
            kv_caches = torch.zeros(*shape, dtype=self.dtype, device=self.device)

            connector_uri = meta_entry.connector
            connector = self.connector_map[connector_uri]
            connector.get(meta_entry.uri, kv_caches)
            self.token_to_kv_pool.set_flat_data(
                kv_caches, device_indices[offset : offset + meta_entry_len]
            )

        self.dec_lock_ref(lru_entry)
        self.evictable_size_ += len(lru_entry.kv_indices)
        assert lru_entry.is_loading()
        lru_entry.set_status(LRUSessionCacheStatus.LOADED)
        return lru_entry, device_indices

    def evict(self, num_tokens: int):
        num_evicted = 0
        while num_evicted < num_tokens:
            lru_entry = self.lru_cache.evict_by_cond(lambda entry: entry.lock_ref == 0)
            if lru_entry is None:
                logger.warn(f"session cache evict failed.")
                return

            logger.debug(f"start ot evict session cache {lru_entry.sid}")
            length = lru_entry.get_length()
            if self.enable_hierarchical_cache:
                self.write_backup(lru_entry)
            else:
                self.token_to_kv_pool_allocator.free(lru_entry.get_kv_indices())
                self.evictable_size_ -= length

            num_evicted += length

    def inc_lock_ref(self, lru_entry: Optional[LRUSessionCacheEntry]):
        if lru_entry is None:
            return

        length = lru_entry.get_length()
        locked_length = lru_entry.lock_size

        # always lock the latest size and release the previous size
        self.evictable_size_ += locked_length
        self.evictable_size_ -= length
        lru_entry.lock_size = length
        self.protected_size_ -= locked_length
        self.protected_size_ += length

        lru_entry.lock_ref += 1

    def dec_lock_ref(self, lru_entry: Optional[LRUSessionCacheEntry]):
        if lru_entry is None:
            return

        # release only once
        if lru_entry.lock_ref == 1:
            locked_length = lru_entry.lock_size
            self.evictable_size_ += locked_length
            self.protected_size_ -= locked_length

            # clear lock_size to zero
            lru_entry.lock_size = 0

        lru_entry.lock_ref -= 1
        assert lru_entry.lock_ref >= 0

    def evictable_size(self):
        if self.enable_hierarchical_cache:
            self.writing_check()
            self.loading_check()
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def load_from_extended(self, req) -> bool:
        lru_cache = req.last_node
        meta = req.last_node_global

        if lru_cache is None and meta is not None:
            return True

        return False

    def insert(self):
        # only for radix_cache
        raise NotImplementedError()

    def writing_check(self):
        pass

    def loading_check(self):
        pass

    def read_to_load_cache(self):
        pass
