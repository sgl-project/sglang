import abc
from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import KVCache


class BaseSWAKVPool(KVCache):
    """ABC for SWA-like KV pools.

    Subclasses expose a `swa_kv_pool` sub-pool plus a full -> swa index
    mapping. Used by `SWATokenToKVPoolAllocator` and the disagg paths to
    handle SWA state separately from the full KV state.
    """

    swa_kv_pool: KVCache
    # Set when SWA KV is a per-request ring of this many tokens (addressed by
    # req_pool_idx) rather than a paged token pool; SWA is then not budgeted per token.
    swa_req_ring_size: Optional[int] = None

    @abc.abstractmethod
    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        raise NotImplementedError()
