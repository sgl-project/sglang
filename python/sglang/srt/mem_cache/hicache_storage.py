from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Literal, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def _pool_name_key(pool_name: PoolName | str | None) -> Optional[str]:
    if pool_name is None:
        return None
    return pool_name.value if isinstance(pool_name, Enum) else str(pool_name)


def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    uint64_val = int(hash_str[:16], 16)
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    is_mla_model: bool
    enable_storage_metrics: bool
    layout: Literal["layer_first", "page_first", "page_first_direct", "page_head"]
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


class PoolName(str, Enum):
    KV = "kv"
    MAMBA = "mamba"
    NSA = "nsa"


class PoolHitPolicy(str, Enum):
    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    use_anchor_host_indices: bool = False
    use_anchor_device_indices: bool = False


@dataclass
class PoolTransferResult:
    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]

    @classmethod
    def empty(cls) -> "PoolTransferResult":
        return cls(0, {})

    @staticmethod
    def _count_consecutive_true(results: List[bool]) -> int:
        for i, ok in enumerate(results):
            if not ok:
                return i
        return len(results)

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

    def update_extra_pool_hit_pages(self, results: dict[str, List[bool]]) -> None:
        for name, rs in results.items():
            self.extra_pool_hit_pages[name] = self.extra_pool_hit_pages.get(name, 0) + (
                self._count_consecutive_true(rs)
            )


class HiCacheStorage(ABC):
    _NSA_INDEXER_SUFFIX = "__nsa_idx"

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        raise NotImplementedError()

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        pass

    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info("Created HiCacheFile storage directory at %s", self.file_path)

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _component_key(self, key: str, pool_name: PoolName | str | None = None) -> str:
        pool_name_key = _pool_name_key(pool_name)
        if pool_name_key in (None, "__default__", PoolName.KV.value):
            return self._get_suffixed_key(key)
        if pool_name_key == PoolName.NSA.value:
            return self._get_suffixed_key(f"{key}{self._NSA_INDEXER_SUFFIX}")
        return self._get_suffixed_key(f"{key}.{pool_name_key}")

    def _component_path(self, key: str, pool_name: PoolName | str | None = None) -> str:
        return os.path.join(
            self.file_path, f"{self._component_key(key, pool_name)}.bin"
        )

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        tensor_path = self._component_path(key)
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            return target_location
        except FileNotFoundError:
            logger.warning("Failed to fetch %s from HiCacheFile storage.", key)
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug("Key %s already exists. Skipped.", key)
            return True

        tensor_path = self._component_path(key)
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True
        except Exception as e:
            logger.error("Failed to save tensor %s: %s", key, e)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        return os.path.exists(self._component_path(key))

    def _has_component(self, key: str, pool_name: PoolName | str | None = None) -> bool:
        return os.path.exists(self._component_path(key, pool_name))

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = next(
            (
                i
                for i, key in enumerate(keys)
                if not self._has_component(key, PoolName.KV)
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV.value: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (
                        i
                        for i in range(kv_pages)
                        if not self._has_component(keys[i], name)
                    ),
                    kv_pages,
                )
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        self._has_component(keys[i], name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[_pool_name_key(name)] = boundary
            final_pages = min(final_pages, boundary)

        if pool_transfers:
            logger.info(
                "HiCacheFile batch_exists_v2: kv_pages=%s final_pages=%s hit_count=%s first_key=%s last_key=%s",
                kv_pages,
                final_pages,
                hit_count,
                keys[0] if keys else None,
                keys[final_pages - 1] if final_pages > 0 else None,
            )

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: PoolName | str, key: str) -> str:
        pool_name_key = _pool_name_key(pool_name)
        if pool_name_key == PoolName.KV.value:
            return key
        if pool_name_key == PoolName.NSA.value:
            return f"{key}{self._NSA_INDEXER_SUFFIX}"
        return f"{key}.{pool_name_key}"

    def _read_page(self, pool_name, key: str, host_pool, page_offset: int) -> bool:
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(self, pool_name, key: str, host_pool, page_offset: int) -> bool:
        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            transfer_name = _pool_name_key(transfer.name)
            host_pool = self.registered_pools[transfer_name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer_name] = [False] * len(keys)
                continue

            results[transfer_name] = [
                op_fn(
                    transfer.name,
                    key,
                    host_pool,
                    host_indices[i * page_size].item(),
                )
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error("Failed to clear HiCacheFile storage: %s", e)
            return False
