import logging
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)


class HiCacheShm(HiCacheStorage):
    """
    Dummy storage backend for shared memory allocator.
    Since shm is a local allocator, there's no actual storage transfer needed.
    """

    def __init__(
        self, storage_config: HiCacheStorageConfig, mem_pool_host: Optional[Any] = None
    ):
        pass

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [None] * len(keys)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return True

    def exists(self, key: str) -> bool:
        return False

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        return PoolTransferResult(0, {})

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results = {}
        for transfer in transfers:
            keys = transfer.keys or []
            results[transfer.name] = [False] * len(keys)
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results = {}
        for transfer in transfers:
            keys = transfer.keys or []
            results[transfer.name] = [True] * len(keys)
        return results

    def clear(self) -> bool:
        return True
