import abc
from typing import List, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import KVCache


class BaseSWAKVPool(KVCache):
    """ABC for SWA-like KV pools.

    Subclasses expose a `swa_kv_pool` sub-pool plus a full -> swa index
    mapping. Used by `SWATokenToKVPoolAllocator` and the disagg paths to
    handle SWA state separately from the full KV state.
    """

    # Whether PD prefill should cap prompt length by the SWA pool size.
    # Generic SWA pools allocate full-prompt SWA KV; specialized pools may not.
    pd_prefill_swa_pool_holds_full_prompt: bool = True

    swa_kv_pool: KVCache

    def invalidate_loc_cache(self) -> None:
        pass

    @abc.abstractmethod
    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        raise NotImplementedError()
