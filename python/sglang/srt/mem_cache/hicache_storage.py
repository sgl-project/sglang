from __future__ import annotations

import argparse
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import human_readable_int

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Max pages per batched storage IO call.
STORAGE_BATCH_SIZE = 128


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    is_mla_model: bool
    enable_storage_metrics: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = None
    extra_info: Optional[dict] = None


@dataclass(frozen=True)
class PrefetchTimeoutConfig:
    """Knobs for the linear prefetch-timeout policy used by HiCache."""

    base: float = 2.0  # seconds, fixed overhead unrelated to token count
    per_ki_token: float = 0.1  # seconds per 1024 tokens
    max: float = 30.0  # seconds, upper bound for the linear timeout


class PoolName(str, Enum):
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers."""

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    INDEXER = "indexer"
    # TODO(hzh0425): Current DeepSeek V4 pool naming is verbose; will be normalized to
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE' in the next PR.
    DEEPSEEK_V4_C4 = "deepseek_v4_c4"
    DEEPSEEK_V4_C4_INDEXER = "deepseek_v4_c4_indexer"
    DEEPSEEK_V4_C128 = "deepseek_v4_c128"
    DEEPSEEK_V4_C4_STATE = "deepseek_v4_c4_state"
    DEEPSEEK_V4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"
    DEEPSEEK_V4_C128_STATE = "deepseek_v4_c128_state"

    # Draft KV pool
    DRAFT = "draft"

    def __str__(self) -> str:
        return self.value


class PoolHitPolicy(str, Enum):
    """Hit policy for batch_exists_v2 per-pool prefix matching.

    ALL_PAGES      : every page in [0, kv_hit) must exist (e.g. DSA).
    TRAILING_PAGES : only the last N pages must exist (e.g. Mamba/SWA states).
    """

    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    """Unified per-pool transfer descriptor for batch v2 interface.

    device<->host path : host_indices + device_indices
    host<->storage path: host_indices + keys
    nodes_to_load      : evicted nodes this transfer covers
    """

    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    nodes_to_load: Optional[List[Any]] = None
    indices_from_pool: Optional[PoolName] = None


@dataclass(frozen=True)
class SidecarPoolSpec:
    """Pool whose transfer indices are reused from one real source pool."""

    pool_name: PoolName
    indices_from_pool: PoolName
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES


@dataclass
class PoolTransferResult:
    """Tracks how many pages were successfully processed per pool."""

    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]

    @classmethod
    def empty(cls) -> "PoolTransferResult":
        return cls(0, {})

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
        """Accumulate kv_hit_pages across batches (max = last successful batch)."""
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

    def update_extra_pool_hit_pages(self, results: dict[str, List[bool]]) -> None:
        """Record actual load/write success counts per extra pool."""
        self.extra_pool_hit_pages.update(
            {name: sum(rs) for name, rs in results.items()}
        )


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
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
        """Check which cache pages exist in storage, respecting per-pool hit policies.

        Longest-prefix semantics
        Extra-pool hit policies (``PoolTransfer.hit_policy``)
        ------------------------------------------------------
        Each ``PoolTransfer`` in ``pool_transfers`` describes a secondary
        cache pool (e.g. Mamba SSM states) that must be co-present with the
        KV pages.  The final ``final_pages`` is the minimum across all pools,
        so a missing auxiliary page shrinks the usable prefix.

        - ``"all_pages"`` (default):  every page in [0, kv_hit) must exist
          for this pool.  Used for pools that are required for every token
          in the prefix (e.g. DeepSeek DSA pool).

        - ``"trailing_pages"``:  only the *last* ``len(transfer.keys)`` pages
          of the KV prefix need to exist.  Used for pools whose data covers
          only the tail of a prefix (e.g. Mamba/SWA Pool).

        Returns
        -------
        PoolTransferResult
            ``kv_hit_pages`` = length of the usable KV prefix.
            ``extra_pool_hit_pages`` maps each pool name to the number of pages
            that were found.
        """
        raise NotImplementedError()

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        """Read data from storage into host memory for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        """Write data from host memory to storage for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


def _parse_size_to_bytes(value: Any) -> int:
    """Parse a size to bytes via human_readable_int (e.g. '200G', '1Gi', '1048576').
    None / empty / '0' disables; an invalid value also disables (with a warning)."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    s = str(value).strip()
    if not s or s == "0":
        return 0
    try:
        return max(0, human_readable_int(s))
    except (argparse.ArgumentTypeError, ValueError):
        logger.warning(f"Invalid size {value!r} for HiCacheFile; disabling.")
        return 0


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        model_name = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = pp_size > 1
        self.config_suffix = f"_{model_name}"
        if not is_mla_model:
            self.config_suffix += f"_{tp_rank}_{tp_size}"
        if enable_pp:
            self.config_suffix += f"_{pp_size}_{pp_rank}"
        # Under NSA context parallel each CP rank holds a disjoint slice of every
        # page, so give each rank its own file key to avoid a cross-rank write race.
        if attn_cp_size > 1:
            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"

        # MLA ranks share the same physical files, so centralize LRU bookkeeping
        # on rank 0; non-MLA ranks each own their own files via the suffix.
        self.tp_rank = tp_rank
        self._is_storage_owner = (not is_mla_model) or (tp_rank == 0)

        if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        self._init_eviction(storage_config)

    def _init_eviction(self, storage_config: HiCacheStorageConfig) -> None:
        # extra_config (per-backend) takes precedence over env vars.
        extra = storage_config.extra_config or {}

        def _cfg(key, env):
            val = extra.get(key)
            return env.get() if val is None else val

        self.max_size_bytes = _parse_size_to_bytes(
            _cfg("max_size", envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE)
        )
        self.min_free_bytes = _parse_size_to_bytes(
            _cfg("min_free_space", envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE)
        )

        ratio_raw = _cfg(
            "eviction_ratio", envs.SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO
        )
        try:
            self.eviction_ratio = float(ratio_raw)
        except (TypeError, ValueError):
            self.eviction_ratio = 0.9
        if not (0.0 < self.eviction_ratio <= 1.0):
            self.eviction_ratio = 0.9

        self._eviction_configured = self.max_size_bytes > 0 or self.min_free_bytes > 0
        self._eviction_enabled = self._eviction_configured and self._is_storage_owner
        if self._eviction_configured and not self._is_storage_owner:
            logger.info(
                f"HiCacheFile rank {self.tp_rank} (MLA): eviction handled by rank 0; "
                f"this rank skips LRU bookkeeping and will not create new files."
            )

        # suffixed_key -> file size in bytes; oldest at front.
        self._lru: "OrderedDict[str, int]" = OrderedDict()
        self._pending_writes: Set[str] = set()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

        if not self._eviction_enabled:
            return

        # Clamp max_size to the filesystem capacity so a too-large cap can't OOM tmpfs.
        fs = self._fs_stats()
        if fs is not None and self.max_size_bytes > 0:
            safe_max = max(0, fs[0] - self.min_free_bytes)
            if self.max_size_bytes > safe_max:
                logger.warning(
                    f"HiCacheFile max_size exceeds filesystem capacity; "
                    f"clamping to {safe_max} B."
                )
                self.max_size_bytes = safe_max

        self._scan_existing_files()
        with self._lock:
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:
                self._evict_locked(0)
            if self.min_free_bytes > 0:
                self._enforce_free_space_locked(0)
        logger.info(
            f"HiCacheFile eviction enabled: cap={self.max_size_bytes} B, "
            f"watermark={self.eviction_ratio:.2f}, min_free={self.min_free_bytes} B, "
            f"existing={self._total_bytes} B ({len(self._lru)} entries)"
        )

    def _fs_stats(self) -> Optional[tuple]:
        """(total, available) bytes for the filesystem; None if unavailable."""
        try:
            st = os.statvfs(self.file_path)
        except (OSError, AttributeError):
            return None
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        return total, free

    def _enforce_free_space_locked(self, value_bytes: int) -> bool:
        """Evict until writing value_bytes still leaves min_free_bytes free.
        Caller holds _lock. Returns False if the write can't be satisfied."""
        if self.min_free_bytes <= 0:
            return True
        fs = self._fs_stats()
        if fs is None:
            return True  # cannot probe -> permissive, fall back to OS errors
        # tmpfs frees space on unlink, so credit reclaimed bytes back to the
        # estimate rather than re-probing statvfs on every eviction.
        free = fs[1]
        self._evict_while(
            lambda reclaimed: (free + reclaimed) - value_bytes < self.min_free_bytes
        )
        # Re-probe: external writers may have changed free space meanwhile.
        fs = self._fs_stats()
        if fs is None:
            return True
        return fs[1] - value_bytes >= self.min_free_bytes

    def _scan_existing_files(self) -> None:
        """Seed LRU index from disk on startup (oldest mtime first)."""
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        entries = []
        for fn in names:
            if not fn.endswith(".bin"):
                continue
            stem = fn[:-4]
            # Only files belonging to this rank/model.
            if not stem.endswith(self.config_suffix):
                continue
            fp = os.path.join(self.file_path, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            entries.append((st.st_mtime, stem, st.st_size))
        entries.sort(key=lambda e: e[0])  # oldest first
        for _, stem, size in entries:
            self._lru[stem] = size
            self._total_bytes += size

    def _evict_one_lru_locked(self) -> Tuple[str, int]:
        """Evict the single oldest evictable LRU entry. Caller holds _lock.

        The shared pop / skip-pending / unlink / ``_total_bytes`` step driven by
        `_evict_while`. Returns ``(outcome, freed_bytes)``:

        - ``("evicted", n)``: oldest entry dropped from the index; ``n`` disk
          bytes reclaimed (0 if the file was already gone).
        - ``("skipped", 0)``: oldest entry is an in-flight write; re-pinned at MRU
          so the writer is not evicted out from under itself.
        - ``("stop", 0)``: nothing evictable (empty index) or the unlink failed
          (entry re-pinned at LRU); the caller should stop its eviction loop.
        """
        if not self._lru:
            return "stop", 0
        evict_stem, evict_size = self._lru.popitem(last=False)  # oldest
        if evict_stem in self._pending_writes:
            # Keep in-flight reservations; their file isn't committed yet.
            self._lru[evict_stem] = evict_size
            return "skipped", 0
        tensor_path = os.path.join(self.file_path, f"{evict_stem}.bin")
        try:
            os.remove(tensor_path)
            freed = evict_size
        except FileNotFoundError:
            freed = 0  # file already gone; still drop the stale index entry
        except OSError as e:
            logger.warning(f"HiCacheFile eviction failed for {evict_stem}: {e}")
            self._lru[evict_stem] = evict_size
            self._lru.move_to_end(evict_stem, last=False)
            return "stop", 0
        self._total_bytes -= evict_size
        return "evicted", freed

    def _evict_while(self, should_continue) -> int:
        """Evict oldest non-pending entries while ``should_continue(reclaimed)``.

        ``should_continue`` is passed the disk bytes reclaimed so far and returns
        whether to keep evicting. In-flight writes are skipped; the loop is bounded
        so it can't spin once every remaining entry is pending. Caller holds _lock.
        Returns the total disk bytes reclaimed.
        """
        reclaimed = 0
        attempts_left = len(self._lru)
        while self._lru and attempts_left > 0 and should_continue(reclaimed):
            outcome, freed = self._evict_one_lru_locked()
            if outcome == "stop":
                break
            if outcome == "skipped":
                attempts_left -= 1
                continue
            # An entry left the index; reset the skip budget and bank the bytes.
            reclaimed += freed
            attempts_left = len(self._lru)
        return reclaimed

    def _evict_locked(self, needed_bytes: int) -> None:
        """Evict LRU entries until total + needed <= cap*ratio. Caller holds _lock."""
        if self.max_size_bytes <= 0:
            return
        target = max(0, int(self.max_size_bytes * self.eviction_ratio) - needed_bytes)
        reclaimed = self._evict_while(lambda _: self._total_bytes > target)
        if reclaimed:
            logger.debug(
                f"HiCacheFile reclaimed {reclaimed} bytes; "
                f"now {self._total_bytes} bytes used"
            )

    def _track_or_touch(self, suffixed_key: str, tensor_path: str) -> None:
        """Mark key as MRU, adopting an untracked on-disk file if needed."""
        if not self._eviction_enabled:
            return
        with self._lock:
            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
                return
        # Untracked file: stat without holding the lock.
        try:
            size = os.path.getsize(tensor_path)
        except OSError:
            return
        with self._lock:
            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
            else:
                self._lru[suffixed_key] = size
                self._total_bytes += size

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"
        )

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {suffixed}")
            self._track_or_touch(suffixed, tensor_path)
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
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
        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")

        # Fast path: same key already on disk. Refresh LRU and skip rewrite.
        if os.path.exists(tensor_path):
            logger.debug(f"Key {key} already exists. Skipped.")
            self._track_or_touch(suffixed, tensor_path)
            return True

        if self._eviction_configured and not self._is_storage_owner:
            logger.warning(
                f"HiCacheFile rank {self.tp_rank} is not the MLA storage owner; "
                f"not caching new key {key} because file eviction is enabled."
            )
            return False

        # Bytes provisionally added to _total_bytes so concurrent writers see
        # this allocation before the file is committed.
        reserved = 0
        tmp_path = None
        try:
            value_bytes = value.numel() * value.element_size()

            if self.max_size_bytes > 0 and value_bytes > self.max_size_bytes:
                logger.warning(
                    f"HiCacheFile: value {value_bytes} B exceeds cap "
                    f"{self.max_size_bytes} B; not caching {key}"
                )
                return False

            if self._eviction_enabled:
                with self._lock:
                    # Cap-based eviction: evict, then bail if still over cap.
                    if (
                        self.max_size_bytes > 0
                        and (self._total_bytes + value_bytes) > self.max_size_bytes
                    ):
                        self._evict_locked(value_bytes)
                        if (self._total_bytes + value_bytes) > self.max_size_bytes:
                            logger.warning(
                                f"HiCacheFile: no evictable space for {value_bytes} B "
                                f"under cap {self.max_size_bytes} B; not caching {key}"
                            )
                            return False
                    # Free-space watermark.
                    if self.min_free_bytes > 0 and not self._enforce_free_space_locked(
                        value_bytes
                    ):
                        logger.warning(
                            f"HiCacheFile: filesystem hosting {self.file_path!r} "
                            f"would fall below min_free={self.min_free_bytes} B "
                            f"after writing {value_bytes} B; refusing {key} "
                            f"to avoid OOM/ENOSPC."
                        )
                        return False
                    # Pre-reserve at MRU so a concurrent evict won't grab this slot.
                    prev = self._lru.pop(suffixed, None)
                    if prev is not None:
                        self._total_bytes -= prev
                    self._lru[suffixed] = value_bytes
                    self._pending_writes.add(suffixed)
                    self._total_bytes += value_bytes
                    reserved = value_bytes

            tmp_path = (
                f"{tensor_path}.tmp."
                f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
            )
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tmp_path)
            os.replace(tmp_path, tensor_path)
            if reserved:
                with self._lock:
                    self._pending_writes.discard(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            # Roll back the reservation and clean up any half-written file.
            if reserved:
                with self._lock:
                    cur = self._lru.pop(suffixed, None)
                    self._pending_writes.discard(suffixed)
                    if cur is not None:
                        self._total_bytes -= cur
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
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
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_files.add(f"{self._get_component_key(key, transfer.name)}.bin")

        existing_files = set()
        with os.scandir(self.file_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name in target_files:
                    existing_files.add(entry.name)
        return existing_files

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files
            )

        # Longest contiguous KV prefix present in storage.
        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)), kv_pages
                )
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: str, key: str) -> str:
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        """Read one page from storage into host_pool at page_offset."""
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        """Write one page from host_pool at page_offset to storage as raw bytes."""
        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
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
                results[transfer.name] = [False] * len(keys)
                continue

            results[transfer.name] = [
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())
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
            with self._lock:
                self._lru.clear()
                self._pending_writes.clear()
                self._total_bytes = 0
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
