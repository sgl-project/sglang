# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""``fast_file``: a portable, performance-oriented local-file HiCache backend.

It keeps the reference ``file`` backend's on-disk format (one raw
``<key><suffix>.bin`` page per file, same key suffix) and adds what a local
NVMe or filesystem tier needs to keep up with serving:

- vectored ``readv``/``writev`` straight between page-first host pool buffers
  and the page file, so pages skip the staging tensor copy;
- a bounded worker pool for parallel page reads and existence checks;
- atomic publication (temporary file + rename), so a reader never observes a
  partially written page;
- background LRU eviction between two watermarks plus an optional positive
  metadata cache.

Pages live in an exact per-model/per-layout namespace directory under the
storage root, so the startup scan, eviction and ``clear()`` cannot touch a
different deployment's files. The storage root comes from the ``storage_dir``
extra-config key, then the ``file`` backend's
``SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR``, then ``/tmp/hicache``; the
namespace directory keeps ``fast_file`` pages apart from ``file`` pages that
share the root.

Extra-config keys (``--hicache-storage-backend-extra-config``):

  storage_dir            root directory (default: see above)
  read_workers           parallel read threads (default 1)
  enable_metadata_cache  cache positive existence checks (default false)
  metadata_ttl           positive-cache lifetime in seconds, -1 = forever (5.0)
  max_size               byte cap per namespace, SI/IEC suffixes (unbounded)
  min_free_space         free-space floor for the filesystem (0 = off)
  evict_high_watermark   start background eviction above this cap fraction (0.95)
  evict_low_watermark    stop background eviction at this cap fraction (0.85)
  preevict_interval_ms   background eviction check interval (10)
  evict_batch_size       files removed per background batch (128)
  stale_temp_age_s       age before abandoned temp files are removed at
                         startup, 0 = keep them (3600)

Locking, eviction accounting and the metadata cache are process-local, and
buffered write completion means close plus rename rather than ``fsync``.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    MetadataCache,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.fast_file.lru_file_evictor import (
    LRUFileEvictor,
    setting,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE_DIR = "/tmp/hicache"
_DEFAULT_READ_WORKERS = 1
_DEFAULT_METADATA_TTL_S = 5.0


class _CorruptPageError(OSError):
    """An on-disk page whose size does not match the requested transfer."""


class HiCacheFastFile(HiCacheFile):
    """See the module docstring for the design and configuration keys."""

    _NAMESPACE_FORMAT_VERSION = 1
    _WRITE_LOCK_STRIPES = 256
    _DIRECT_IO_LAYOUTS = ("page_first", "page_first_direct")

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Optional[HostKVCache] = None,
        file_path: str = _DEFAULT_STORAGE_DIR,
    ):
        # HiCacheFile.__init__ is intentionally not called: it wires the
        # reference backend's evictor and metadata cache. Only its key helpers
        # and storage-dir env var are shared.
        extra_config = storage_config.extra_config or {}
        self.config_suffix = self._build_config_suffix(storage_config)
        self.storage_root = (
            extra_config.get("storage_dir")
            or envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get()
            or file_path
        )
        self.file_path = os.path.join(
            self.storage_root, self._namespace_name(storage_config, mem_pool_host)
        )
        os.makedirs(self.file_path, exist_ok=True)

        self.read_workers = int(
            setting(extra_config, "read_workers", _DEFAULT_READ_WORKERS)
        )
        if self.read_workers < 1:
            raise ValueError(
                f"HiCacheFastFile read_workers must be at least 1, got {self.read_workers}."
            )
        self.enable_metadata_cache = bool(
            setting(extra_config, "enable_metadata_cache", False)
        )
        self.metadata_cache: Optional[MetadataCache] = None
        if self.enable_metadata_cache:
            self.metadata_cache = MetadataCache(
                float(setting(extra_config, "metadata_ttl", _DEFAULT_METADATA_TTL_S))
            )
        self.enable_storage_metrics = storage_config.enable_storage_metrics
        self._is_rank_replicated = storage_config.is_mla_model
        self._tp_size = storage_config.tp_size
        self._vector_io_supported = hasattr(os, "readv") and hasattr(os, "writev")
        self._iov_max = self._get_iov_max() if self._vector_io_supported else 1
        self._write_locks = tuple(
            threading.Lock() for _ in range(self._WRITE_LOCK_STRIPES)
        )
        self._pool_direct_io: dict[str, bool] = {}
        self._warned_partial_eviction = False
        self._metrics_lock = threading.Lock()
        self._prefetch_pgs: list[int] = []
        self._backup_pgs: list[int] = []
        self._prefetch_bandwidth: list[float] = []
        self._backup_bandwidth: list[float] = []

        self._evictor = LRUFileEvictor(
            self.file_path,
            self.config_suffix,
            tp_rank=storage_config.tp_rank,
            is_mla_model=storage_config.is_mla_model,
            extra_config=extra_config,
            on_evict=(
                self.metadata_cache.remove if self.metadata_cache is not None else None
            ),
        )
        if self.metadata_cache is not None:
            self._scan_existing_files_to_metadata_cache()
        self._read_executor: Optional[ThreadPoolExecutor] = None
        if self.read_workers > 1:
            self._read_executor = ThreadPoolExecutor(
                max_workers=self.read_workers,
                thread_name_prefix=f"HiCacheFastFileRead-{storage_config.tp_rank}",
            )
        if not self._vector_io_supported:
            logger.warning(
                "HiCacheFastFile vectored I/O is unavailable; using staged copies."
            )
        logger.info(
            "HiCacheFastFile namespace=%s read_workers=%d",
            self.file_path,
            self.read_workers,
        )

    @classmethod
    def _namespace_name(
        cls,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Optional[HostKVCache],
    ) -> str:
        """Exact, filesystem-safe namespace for one model and cache layout."""
        storage_rank = 0 if storage_config.is_mla_model else storage_config.tp_rank
        fields: list[Any] = [
            cls._NAMESPACE_FORMAT_VERSION,
            storage_config.model_name or "",
            storage_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.attn_cp_rank,
            storage_config.attn_cp_size,
            storage_config.is_mla_model,
            storage_config.tp_lcm_size,
            storage_config.should_split_heads,
        ]
        if mem_pool_host is not None:
            # Same key hashes with another layout or dtype are not the same bytes.
            fields.extend(
                (
                    mem_pool_host.layout,
                    mem_pool_host.page_size,
                    str(mem_pool_host.dtype),
                    mem_pool_host.size_per_token,
                )
            )
        identity = "\0".join(str(field) for field in fields)
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        return f"namespace-{digest}"

    @staticmethod
    def _get_iov_max() -> int:
        try:
            return max(1, int(os.sysconf("SC_IOV_MAX")))
        except (AttributeError, OSError, ValueError):
            return 1024

    # ----- host pool registration -------------------------------------------

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        super().register_mem_pool_host(mem_pool_host)
        super().register_mem_host_pool_v2(mem_pool_host, PoolName.KV)
        self._pool_direct_io[PoolName.KV] = self._supports_direct_io(mem_pool_host)
        logger.info(
            "HiCacheFastFile registered KV host pool layout=%s direct_io=%s",
            mem_pool_host.layout,
            self._pool_direct_io[PoolName.KV],
        )

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name) -> None:
        self._check_side_pool(host_pool, host_pool_name)
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._pool_direct_io[host_pool_name] = self._supports_direct_io(host_pool)
        if host_pool_name == PoolName.KV:
            self.mem_pool_host = host_pool
        logger.info(
            "HiCacheFastFile registered host pool=%s layout=%s direct_io=%s",
            host_pool_name,
            host_pool.layout,
            self._pool_direct_io[host_pool_name],
        )

    def _check_side_pool(self, host_pool: HostKVCache, host_pool_name) -> None:
        if host_pool_name == PoolName.KV:
            return
        if self._evictor.configured and not self._warned_partial_eviction:
            self._warned_partial_eviction = True
            logger.warning(
                "HiCacheFastFile evicts KV pages and %s sidecar files independently; "
                "a partially evicted page only shrinks the usable prefix.",
                host_pool_name,
            )
        rank_sharded = host_pool_name == PoolName.MAMBA
        if not rank_sharded:
            from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

            rank_sharded = isinstance(host_pool, MHATokenToKVPoolHost)
        # MLA ranks share one namespace without a TP rank in the key suffix, so
        # rank-sharded sidecars from different ranks would overwrite each other.
        if self._is_rank_replicated and self._tp_size > 1 and rank_sharded:
            raise ValueError(
                "HiCacheFastFile does not support rank-sharded Mamba or MHA side "
                "pools with a rank-replicated KV namespace at TP > 1; use a backend "
                "with rank-aware side-pool keys such as mooncake."
            )

    def _supports_direct_io(self, host_pool: HostKVCache) -> bool:
        if (
            not self._vector_io_supported
            or host_pool.layout not in self._DIRECT_IO_LAYOUTS
        ):
            return False
        probe = torch.arange(host_pool.page_size, dtype=torch.int64)
        return host_pool.get_page_buffer_meta(probe) is not None

    # ----- buffers and vectored I/O ------------------------------------------

    @staticmethod
    def _address_buffer(address: int, num_bytes: int) -> memoryview:
        return memoryview((ctypes.c_ubyte * num_bytes).from_address(address)).cast("B")

    @staticmethod
    def _tensor_buffer(tensor: torch.Tensor, *, writable: bool) -> memoryview:
        if tensor.device.type != "cpu":
            raise ValueError("HiCacheFastFile tensors must reside in CPU memory")
        if not tensor.is_contiguous():
            if writable:
                raise ValueError("HiCacheFastFile read targets must be contiguous")
            tensor = tensor.contiguous()
        return memoryview(tensor.view(torch.uint8).numpy()).cast("B")

    def _pool_page_buffers(
        self, host_pool: HostKVCache, host_indices: torch.Tensor, page_count: int
    ) -> Optional[list[list[memoryview]]]:
        """One iovec list per page over the pool's own memory, or None."""
        if page_count == 0:
            return []
        ptrs, sizes = host_pool.get_page_buffer_meta(host_indices)
        if not ptrs or len(ptrs) != len(sizes) or len(ptrs) % page_count:
            logger.error(
                "HiCacheFastFile host pool returned invalid page buffer metadata "
                "(%d pointers for %d pages)",
                len(ptrs) if ptrs else 0,
                page_count,
            )
            return None
        per_page = len(ptrs) // page_count
        return [
            [
                self._address_buffer(int(ptrs[i + j]), int(sizes[i + j]))
                for j in range(per_page)
            ]
            for i in range(0, len(ptrs), per_page)
        ]

    @staticmethod
    def _consume_iovecs(buffers: list[memoryview], num_bytes: int) -> None:
        while buffers and num_bytes >= len(buffers[0]):
            num_bytes -= len(buffers[0])
            buffers.pop(0)
        if num_bytes:
            buffers[0] = buffers[0][num_bytes:]

    def _readv_exact(self, fd: int, buffers: list[memoryview]) -> int:
        pending = [buffer for buffer in buffers if len(buffer)]
        expected = sum(len(buffer) for buffer in pending)
        completed = 0
        while pending:
            if self._vector_io_supported:
                transferred = os.readv(fd, pending[: self._iov_max])
            else:
                chunk = os.read(fd, len(pending[0]))
                transferred = len(chunk)
                pending[0][:transferred] = chunk
            if transferred == 0:
                break
            completed += transferred
            self._consume_iovecs(pending, transferred)
        if completed != expected:
            raise _CorruptPageError(
                f"Short read: expected {expected} bytes, got {completed}"
            )
        return completed

    def _writev_exact(self, fd: int, buffers: list[memoryview]) -> int:
        pending = [buffer for buffer in buffers if len(buffer)]
        expected = sum(len(buffer) for buffer in pending)
        completed = 0
        while pending:
            if self._vector_io_supported:
                transferred = os.writev(fd, pending[: self._iov_max])
            else:
                transferred = os.write(fd, pending[0])
            if transferred == 0:
                raise IOError(
                    f"Short write: expected {expected} bytes, got {completed}"
                )
            completed += transferred
            self._consume_iovecs(pending, transferred)
        return completed

    def _parallel_map(self, fn: Callable, *iterables) -> list:
        if self._read_executor is None:
            return [fn(*args) for args in zip(*iterables)]
        futures = [self._read_executor.submit(fn, *args) for args in zip(*iterables)]
        results = [None] * len(futures)
        first_error = None
        for index, future in enumerate(futures):
            try:
                results[index] = future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return results

    # ----- single page read / write ------------------------------------------

    def _page_path(self, suffixed_key: str) -> str:
        return os.path.join(self.file_path, f"{suffixed_key}.bin")

    def _write_lock_for(self, suffixed_key: str) -> threading.Lock:
        return self._write_locks[hash(suffixed_key) % len(self._write_locks)]

    def _read_page_buffers(self, key: str, buffers: list[memoryview]) -> bool:
        suffixed = self._get_suffixed_key(key)
        tensor_path = self._page_path(suffixed)
        expected = sum(len(buffer) for buffer in buffers)
        try:
            fd = os.open(tensor_path, os.O_RDONLY)
            try:
                actual = os.fstat(fd).st_size
                if actual != expected:
                    raise _CorruptPageError(
                        f"page size mismatch: expected {expected} bytes, found {actual}"
                    )
                self._readv_exact(fd, buffers)
            finally:
                os.close(fd)
        except FileNotFoundError:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            self._evictor.forget(suffixed)
            logger.warning("Failed to fetch %s from HiCacheFastFile storage.", key)
            return False
        except _CorruptPageError as exc:
            self._remove_corrupt_page(suffixed, tensor_path, expected)
            logger.warning("Discarded corrupt HiCacheFastFile page %s: %s", key, exc)
            return False
        except OSError as exc:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            logger.warning("Failed to read HiCacheFastFile page %s: %s", key, exc)
            return False
        self._evictor.touch(suffixed, tensor_path)
        if self.metadata_cache is not None:
            self.metadata_cache.add(suffixed)
        return True

    def _remove_corrupt_page(
        self, suffixed: str, tensor_path: str, expected_bytes: int
    ) -> None:
        if self.metadata_cache is not None:
            self.metadata_cache.remove(suffixed)
        with self._write_lock_for(suffixed):
            try:
                if os.path.getsize(tensor_path) == expected_bytes:
                    # A writer replaced the page after the reader opened it.
                    self._evictor.touch(suffixed, tensor_path, size=expected_bytes)
                    if self.metadata_cache is not None:
                        self.metadata_cache.add(suffixed)
                    return
                os.remove(tensor_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "Failed to remove corrupt HiCacheFastFile page %s: %s",
                    tensor_path,
                    exc,
                )
                return
            self._evictor.forget(suffixed)

    def _write_page_buffers(
        self, key: str, buffers: list[memoryview]
    ) -> tuple[bool, int]:
        """Publish a page; returns (ok, bytes written). A duplicate writes 0."""
        suffixed = self._get_suffixed_key(key)
        tensor_path = self._page_path(suffixed)
        value_bytes = sum(len(buffer) for buffer in buffers)
        with self._write_lock_for(suffixed):
            # Check the filesystem, not the positive metadata cache: an entry
            # can outlive an external deletion.
            try:
                actual_bytes = os.path.getsize(tensor_path)
            except FileNotFoundError:
                actual_bytes = None
            except OSError as exc:
                logger.error("Failed to stat HiCacheFastFile page %s: %s", key, exc)
                return False, 0
            if actual_bytes == value_bytes:
                logger.debug("Key %s already exists. Skipped.", key)
                self._evictor.touch(suffixed, tensor_path, size=value_bytes)
                if self.metadata_cache is not None:
                    self.metadata_cache.add(suffixed)
                return True, 0
            if actual_bytes is not None:
                logger.warning(
                    "Replacing wrong-sized HiCacheFastFile page %s: expected %d bytes, "
                    "found %d",
                    key,
                    value_bytes,
                    actual_bytes,
                )
            return self._publish_page_locked(key, suffixed, tensor_path, buffers)

    def _publish_page_locked(
        self, key: str, suffixed: str, tensor_path: str, buffers: list[memoryview]
    ) -> tuple[bool, int]:
        value_bytes = sum(len(buffer) for buffer in buffers)
        tmp_path = f"{tensor_path}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        reserved = False
        fd = -1
        try:
            if not self._evictor.reserve(suffixed, value_bytes, key=key):
                return False, 0
            reserved = True
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                self._writev_exact(fd, buffers)
            finally:
                os.close(fd)
                fd = -1
            os.replace(tmp_path, tensor_path)
            # Publish the metadata before commit: once committed, background
            # eviction may remove both the page and its metadata entry.
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            self._evictor.commit(suffixed)
            return True, value_bytes
        except Exception as exc:
            logger.error("Failed to save tensor %s: %s", key, exc)
            if reserved:
                self._evictor.abort(suffixed)
            if fd != -1:
                os.close(fd)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False, 0

    # ----- staged (tensor copy) transfers for pools without direct I/O ------

    def _staged_read_page(
        self, pool_name: str, key: str, host_pool: HostKVCache, page_offset: int
    ) -> tuple[bool, int]:
        target = host_pool.get_dummy_flat_data_page()
        storage_key = self._log_key(pool_name, key)
        if not self._read_page_buffers(
            storage_key, [self._tensor_buffer(target, writable=True)]
        ):
            return False, 0
        host_pool.set_from_flat_data_page(page_offset, target)
        return True, target.numel() * target.element_size()

    def _staged_write_page(
        self, pool_name: str, key: str, host_pool: HostKVCache, page_offset: int
    ) -> tuple[bool, int]:
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self._write_page_buffers(
            self._log_key(pool_name, key),
            [self._tensor_buffer(data_page, writable=False)],
        )

    def _transfer_pages(
        self,
        pool_name: str,
        host_pool: HostKVCache,
        keys: List[str],
        host_indices: Optional[torch.Tensor],
        *,
        read: bool,
    ) -> tuple[List[bool], int, int]:
        """Move ``keys`` between ``host_pool`` and storage.

        Returns (per-key success, pages moved, bytes moved); duplicate writes
        that were skipped count as success but not as moved.
        """
        page_size = host_pool.page_size
        expected = len(keys) * page_size
        if host_indices is None or host_indices.numel() != expected:
            logger.error(
                "HiCacheFastFile %s indices length mismatch for %s: expected %s, got %s",
                "read" if read else "write",
                pool_name,
                expected,
                host_indices.numel() if host_indices is not None else 0,
            )
            return [False] * len(keys), 0, 0
        storage_keys = [self._log_key(pool_name, key) for key in keys]
        if self._pool_direct_io[pool_name]:
            page_buffers = self._pool_page_buffers(host_pool, host_indices, len(keys))
            if page_buffers is None:
                return [False] * len(keys), 0, 0
            if read:
                results = self._parallel_map(
                    self._read_page_buffers, storage_keys, page_buffers
                )
                num_bytes = sum(
                    sum(len(buffer) for buffer in buffers)
                    for ok, buffers in zip(results, page_buffers)
                    if ok
                )
                return results, sum(results), num_bytes
            sized = [
                self._write_page_buffers(storage_key, buffers)
                for storage_key, buffers in zip(storage_keys, page_buffers)
            ]
        else:
            op = self._staged_read_page if read else self._staged_write_page
            offsets = [host_indices[i * page_size].item() for i in range(len(keys))]
            if read:
                sized = self._parallel_map(
                    op, [pool_name] * len(keys), keys, [host_pool] * len(keys), offsets
                )
            else:
                sized = [
                    op(pool_name, key, host_pool, o) for key, o in zip(keys, offsets)
                ]
        results = [ok for ok, _ in sized]
        pages = sum(1 for _, num_bytes in sized if num_bytes)
        return results, pages, sum(num_bytes for _, num_bytes in sized)

    def _record_io_metrics(
        self, *, prefetch: bool, pages: int, num_bytes: int, elapsed_s: float
    ) -> None:
        if not self.enable_storage_metrics or pages == 0:
            return
        # Buffered writes report acceptance bandwidth, not durable media bandwidth.
        bandwidth = num_bytes / 1e9 / max(elapsed_s, 1e-9)
        with self._metrics_lock:
            if prefetch:
                self._prefetch_pgs.append(pages)
                self._prefetch_bandwidth.append(bandwidth)
            else:
                self._backup_pgs.append(pages)
                self._backup_bandwidth.append(bandwidth)

    # ----- HiCacheStorage interface ------------------------------------------

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        buffer = self._tensor_buffer(target_location, writable=True)
        return target_location if self._read_page_buffers(key, [buffer]) else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        start = time.perf_counter()
        results = self._parallel_map(self.get, keys, target_locations)
        hits = [target for target in results if target is not None]
        self._record_io_metrics(
            prefetch=True,
            pages=len(hits),
            num_bytes=sum(t.numel() * t.element_size() for t in hits),
            elapsed_s=time.perf_counter() - start,
        )
        return results

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        buffer = self._tensor_buffer(value, writable=False)
        return self._write_page_buffers(key, [buffer])[0]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        start = time.perf_counter()
        pages = 0
        num_bytes = 0
        success = True
        for key, value in zip(keys, values):
            ok, written = self._write_page_buffers(
                key, [self._tensor_buffer(value, writable=False)]
            )
            if not ok:
                success = False
                break
            if written:
                pages += 1
                num_bytes += written
        self._record_io_metrics(
            prefetch=False,
            pages=pages,
            num_bytes=num_bytes,
            elapsed_s=time.perf_counter() - start,
        )
        return success

    def exists(self, key: str) -> bool:
        suffixed = self._get_suffixed_key(key)
        if self.metadata_cache is not None and self.metadata_cache.contains(suffixed):
            return True
        if not os.path.exists(self._page_path(suffixed)):
            return False
        if self.metadata_cache is not None:
            self.metadata_cache.add(suffixed)
        return True

    def _present_filename(self, filename: str) -> Optional[str]:
        stem = filename[:-4]
        if self.metadata_cache is not None and self.metadata_cache.contains(stem):
            return filename
        if not os.path.exists(os.path.join(self.file_path, filename)):
            return None
        if self.metadata_cache is not None:
            self.metadata_cache.add(stem)
        return filename

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> set[str]:
        # Targeted stats instead of the reference backend's directory scan,
        # which is linear in the number of cached pages.
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for transfer in pool_transfers or []:
            target_files.update(
                f"{self._get_component_key(key, transfer.name)}.bin" for key in keys
            )
        present = self._parallel_map(self._present_filename, sorted(target_files))
        return {filename for filename in present if filename is not None}

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        start = time.perf_counter()
        results, pages, num_bytes = self._transfer_pages(
            PoolName.KV, self.mem_pool_host, keys, host_indices, read=True
        )
        self._record_io_metrics(
            prefetch=True,
            pages=pages,
            num_bytes=num_bytes,
            elapsed_s=time.perf_counter() - start,
        )
        return results

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        start = time.perf_counter()
        results, pages, num_bytes = self._transfer_pages(
            PoolName.KV, self.mem_pool_host, keys, host_indices, read=False
        )
        self._record_io_metrics(
            prefetch=False,
            pages=pages,
            num_bytes=num_bytes,
            elapsed_s=time.perf_counter() - start,
        )
        return results

    def _batch_io_v2(
        self, transfers: List[PoolTransfer], *, read: bool
    ) -> dict[str, List[bool]]:
        start = time.perf_counter()
        results: dict[str, List[bool]] = {}
        pages = 0
        num_bytes = 0
        for transfer in transfers:
            transfer_results, transfer_pages, transfer_bytes = self._transfer_pages(
                transfer.name,
                self.registered_pools[transfer.name],
                transfer.keys or [],
                transfer.host_indices,
                read=read,
            )
            results[transfer.name] = transfer_results
            pages = max(pages, transfer_pages)
            num_bytes += transfer_bytes
        self._record_io_metrics(
            prefetch=read,
            pages=pages,
            num_bytes=num_bytes,
            elapsed_s=time.perf_counter() - start,
        )
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, read=True)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, read=False)

    def clear(self) -> bool:
        success = self._evictor.clear_storage()
        if self.metadata_cache is not None:
            self.metadata_cache.clear()
        if success:
            logger.info("Cleared HiCacheFastFile namespace %s.", self.file_path)
        else:
            logger.error(
                "Failed to fully clear HiCacheFastFile namespace %s.", self.file_path
            )
        return success

    def get_stats(self):
        if not self.enable_storage_metrics:
            return None
        from sglang.srt.observability.metrics_collector import StorageMetrics

        with self._metrics_lock:
            storage_metrics = StorageMetrics(
                prefetch_pgs=self._prefetch_pgs,
                backup_pgs=self._backup_pgs,
                prefetch_bandwidth=self._prefetch_bandwidth,
                backup_bandwidth=self._backup_bandwidth,
            )
            self._prefetch_pgs = []
            self._backup_pgs = []
            self._prefetch_bandwidth = []
            self._backup_bandwidth = []
        return storage_metrics

    def close(self) -> None:
        executor = self._read_executor
        self._read_executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        self._evictor.close()
