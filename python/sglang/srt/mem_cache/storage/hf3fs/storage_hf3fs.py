import atexit
import concurrent.futures
import json
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient
from sglang.srt.observability.metrics_collector import StorageMetrics

logger = logging.getLogger(__name__)


# Default fraction of the KV file size used for an auxiliary pool
# (e.g. Mamba/DSA) when no per-pool override is supplied.
_DEFAULT_AUX_FILE_SIZE_FRACTION = 0.1


class Hf3fsMetadataInterface(ABC):
    """Interface for HF3FS metadata operations."""

    @abstractmethod
    def initialize(self, rank: int, num_pages: int) -> None:
        """Initialize the metadata service with specified number of pages."""
        pass

    @abstractmethod
    def reserve_and_allocate_page_indices(
        self,
        rank: int,
        keys: List[Tuple[str, str]],
    ) -> List[Tuple[bool, int]]:
        """
        Reserve and allocate page indices for the specified keys.
        Args:
            rank: The rank of the process.
            keys: The keys to reserve and allocate page indices for. Each tuple contains a key and the key of its prefix block.
        Returns:
            List[Tuple[bool, int]]: A list of tuples, where each tuple contains a boolean indicating whether the key has existed and an integer indicating the allocated page index.
        """
        pass

    @abstractmethod
    def confirm_write(
        self,
        rank: int,
        written_keys_to_confirm: List[Tuple[str, int]],
        pages_to_release: List[int],
    ) -> None:
        """
        Confirm that key-value pairs have been successfully written to storage.
        Args:
            rank: The rank of the process.
            written_keys_to_confirm: A list of tuples, where each tuple contains a key and its corresponding page index.
            pages_to_release: A list of page indices to be released.
        """
        pass

    @abstractmethod
    def get_page_indices(self, rank: int, keys: List[str]) -> List[Optional[int]]:
        """
        Get page indices for the specified keys.
        Args:
            rank: The rank of the process.
            keys: A list of keys.
        Returns:
            List[Optional[int]]: A list of integers representing the page indices for the specified keys.
                                 If a key is not found, the corresponding index will be None.
        """
        pass

    @abstractmethod
    def delete_keys(self, rank: int, keys: List[str]) -> None:
        """Delete specified keys and their associated pages."""
        pass

    @abstractmethod
    def exists(self, rank: int, keys: List[str]) -> List[bool]:
        """Check if the specified keys exist."""
        pass

    @abstractmethod
    def clear(self, rank: int) -> None:
        """Clear all key-value pairs and page allocations for the specified rank."""
        pass


class AtomicCounter:
    def __init__(self, n: int):
        assert n > 0
        self.n = n
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            current = self._value
            self._value = (current + 1) % self.n
            return current


def synchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.lock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


@dataclass
class _Hf3fsPoolEngine:
    """Per-pool 3FS state.

    Each registered host pool (KV, MAMBA, ...) gets its own preallocated 3FS
    file, client list, executor, metadata client and namespace. The KV engine
    is created in ``HiCacheHF3FS.__init__`` so v1 deployments keep working
    without any extra wiring; auxiliary engines are added lazily by
    ``register_mem_host_pool_v2``.
    """

    pool_name: str
    file_path: str
    file_size: int
    bytes_per_page: int
    num_pages: int
    clients: List[Hf3fsClient]
    ac: AtomicCounter
    executor: concurrent.futures.ThreadPoolExecutor
    metadata_client: Hf3fsMetadataInterface
    metadata_rank: int
    host_pool: Optional[HostKVCache] = None
    is_zero_copy: bool = False
    skip_backup: bool = False
    gb_per_page: float = 0.0


def create_hf3fs_client(
    path: str,
    size: int,
    bytes_per_page: int,
    entries: int,
    client_timeout: int,
    use_mock: bool = False,
) -> Hf3fsClient:
    """Factory function to create appropriate HF3FS client.

    Args:
        path: File path for storage
        size: Total size of storage file
        bytes_per_page: Bytes per page
        entries: Number of entries for batch operations
        use_mock: Whether to use mock client instead of real usrbio client

    Returns:
    """
    if use_mock:
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsMockClient

        logger.info(f"[Rank Using Hf3fsMockClient for testing")
        return Hf3fsMockClient(path, size, bytes_per_page, entries)
    else:
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_usrbio_client import (
            Hf3fsUsrBioClient,
        )

        return Hf3fsUsrBioClient(path, size, bytes_per_page, entries, client_timeout)


class HiCacheHF3FS(HiCacheStorage):
    """HiCache backend that stores KV cache pages in HF3FS files."""

    default_env_var: str = "SGLANG_HICACHE_HF3FS_CONFIG_PATH"

    def __init__(
        self,
        rank: int,
        file_path: str,
        file_size: int,
        numjobs: int,
        bytes_per_page: int,
        entries: int,
        client_timeout: int,
        dtype: torch.dtype,
        metadata_client: Hf3fsMetadataInterface,
        is_mla_model: bool = False,
        is_page_first_layout: bool = False,
        use_mock_client: bool = False,
        enable_storage_metrics: bool = False,
        metadata_server_url: Optional[str] = None,
        pools_extra_config: Optional[dict] = None,
    ):
        self._original_rank = rank
        self.rank = rank
        self.file_path = file_path
        self.file_size = file_size
        self.numjobs = numjobs
        self.bytes_per_page = bytes_per_page
        self.gb_per_page = bytes_per_page / (1 << 30)
        self.entries = entries
        self.client_timeout = client_timeout
        self.dtype = dtype
        self.metadata_client = metadata_client
        self.is_mla_model = is_mla_model
        self.is_page_first_layout = is_page_first_layout
        self.enable_storage_metrics = enable_storage_metrics
        self.use_mock_client = use_mock_client
        self._metadata_server_url = metadata_server_url
        self._pools_extra_config = pools_extra_config or {}
        self.numel = self.bytes_per_page // self.dtype.itemsize
        self.num_pages = self.file_size // self.bytes_per_page
        self.skip_backup = False
        if self.is_mla_model and self.rank != 0:
            self.skip_backup = True
            self.rank = 0

        self.is_zero_copy = False

        logger.info(
            f"[Rank {self.rank}] HiCacheHF3FS Client Initializing: "
            f"file_path={self.file_path}, "
            f"file_size={self.file_size / (2 ** 30):.2f} GB, "
            f"num_pages={self.num_pages}, "
            f"is_mla_model={self.is_mla_model}"
        )

        # Per-pool engine registry. Populated for KV here so v1 callers keep
        # working unchanged; auxiliary pools are added by
        # ``register_mem_host_pool_v2``. Note: ``_engines`` must be set before
        # the SIGTERM handler is installed below — otherwise an early signal
        # could find ``self._engines`` undefined when ``close()`` runs.
        self._engines: Dict[str, _Hf3fsPoolEngine] = {}
        # Tracks the next stable pool index used to derive a unique metadata
        # rank namespace for auxiliary pools (KV is always 0).
        self._next_pool_idx = 1
        self._pool_idx_map: Dict[str, int] = {PoolName.KV: 0}

        self.ac = AtomicCounter(self.numjobs)
        self.clients = [
            create_hf3fs_client(
                self.file_path,
                self.file_size,
                self.bytes_per_page,
                self.entries,
                self.client_timeout,
                use_mock_client,
            )
            for _ in range(numjobs)
        ]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.numjobs, thread_name_prefix=f"HiCacheHF3FS-Rank{self.rank}"
        )

        self.metadata_client.initialize(self.rank, self.num_pages)
        self.lock = threading.RLock()

        # Build the KV engine from the parameters above. v1 deployments only
        # ever interact with this single engine.
        kv_engine = _Hf3fsPoolEngine(
            pool_name=PoolName.KV,
            file_path=self.file_path,
            file_size=self.file_size,
            bytes_per_page=self.bytes_per_page,
            num_pages=self.num_pages,
            clients=self.clients,
            ac=self.ac,
            executor=self.executor,
            metadata_client=self.metadata_client,
            metadata_rank=self.rank,
            host_pool=None,
            is_zero_copy=False,
            skip_backup=self.skip_backup,
            gb_per_page=self.gb_per_page,
        )
        self._engines[PoolName.KV] = kv_engine

        atexit.register(self.close)

        signal.signal(signal.SIGINT, lambda sig, frame: self.close())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.close())
        signal.signal(signal.SIGQUIT, lambda sig, frame: self.close())

        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

    @staticmethod
    def from_env_config(
        bytes_per_page: int,
        dtype: Optional[torch.dtype] = None,
        storage_config: Any = None,
        *,
        rank: Optional[int] = None,
        is_mla_model: Optional[bool] = None,
        is_page_first_layout: Optional[bool] = None,
    ) -> "HiCacheHF3FS":
        """Create a HiCacheHF3FS instance from environment configuration.

        Accepted ``storage_config`` shapes:
            * ``HiCacheStorageConfig`` instance — the production path used by
              ``StorageBackendFactory``.
            * ``dict`` — an inline JSON-style config (file_path_prefix /
              file_size / numjobs / entries / ...). Used by unit tests so they
              don't need to set the ``SGLANG_HICACHE_HF3FS_CONFIG_PATH`` env
              var. The ``"pools"`` key is forwarded as
              ``pools_extra_config`` for per-pool overrides.
            * ``None`` — falls back to env-var lookup, then to a local
              single-machine default.

        Keyword overrides ``rank``/``is_mla_model``/``is_page_first_layout``
        take precedence over fields read from ``storage_config``.

        Environment:
            - Uses env var stored in ``HiCacheHF3FS.default_env_var`` to
              locate a JSON config when ``storage_config`` is None or a
              ``HiCacheStorageConfig``.

        Raises:
            ValueError: If required config keys are missing.
        """
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
            Hf3fsGlobalMetadataClient,
            Hf3fsLocalMetadataClient,
        )

        if dtype is None:
            dtype = torch.uint8

        use_mock_client = False
        pools_extra_config: Optional[dict] = None
        enable_storage_metrics = False
        inline_config: Optional[dict] = None

        if storage_config is None:
            cfg_rank, cfg_is_mla, cfg_is_pfl = 0, False, False
        elif isinstance(storage_config, dict):
            # Inline dict config (used by unit tests). Treat as the JSON
            # config and bypass env-var lookup entirely.
            inline_config = dict(storage_config)
            use_mock_client = bool(inline_config.get("use_mock_hf3fs_client", False))
            pools_extra_config = inline_config.get("pools")
            cfg_rank, cfg_is_mla, cfg_is_pfl = 0, False, False
        else:
            cfg_rank, cfg_is_mla, cfg_is_pfl = (
                storage_config.tp_rank,
                storage_config.is_mla_model,
                storage_config.is_page_first_layout,
            )
            enable_storage_metrics = storage_config.enable_storage_metrics
            if storage_config.extra_config is not None:
                use_mock_client = storage_config.extra_config.get(
                    "use_mock_hf3fs_client", False
                )
                pools_extra_config = storage_config.extra_config.get("pools")

        final_rank = rank if rank is not None else cfg_rank
        final_is_mla = is_mla_model if is_mla_model is not None else cfg_is_mla
        final_is_pfl = (
            is_page_first_layout if is_page_first_layout is not None else cfg_is_pfl
        )

        mla_unsupported_msg = f"MLA model is not supported without global metadata server, please refer to https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/hf3fs/docs/deploy_sglang_3fs_multinode.md"

        config: Optional[dict] = inline_config
        if config is None:
            config_path = os.getenv(HiCacheHF3FS.default_env_var)
            if not config_path:
                if final_is_mla:
                    raise ValueError(mla_unsupported_msg)

                return HiCacheHF3FS(
                    rank=final_rank,
                    file_path=f"/data/hicache.{final_rank}.bin",
                    file_size=1 << 40,
                    numjobs=16,
                    bytes_per_page=bytes_per_page,
                    entries=8,
                    client_timeout=5,
                    dtype=dtype,
                    metadata_client=Hf3fsLocalMetadataClient(),
                    is_page_first_layout=final_is_pfl,
                    use_mock_client=use_mock_client,
                    pools_extra_config=pools_extra_config,
                )

            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load config from {config_path}: {str(e)}"
                )

        # Check required keys (metadata_server_url is now optional)
        required_keys = {
            "file_path_prefix",
            "file_size",
            "numjobs",
            "entries",
        }
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {missing_keys}")

        # Inline dicts may also pass through use_mock / pools fields directly.
        if inline_config is not None:
            use_mock_client = bool(
                inline_config.get("use_mock_hf3fs_client", use_mock_client)
            )
            if pools_extra_config is None and isinstance(
                inline_config.get("pools"), dict
            ):
                pools_extra_config = inline_config.get("pools")

        # Choose metadata client based on configuration
        metadata_server_url: Optional[str] = None
        if config.get("metadata_server_url"):
            # Use global metadata client to connect to metadata server
            metadata_server_url = config["metadata_server_url"]
            metadata_client = Hf3fsGlobalMetadataClient(metadata_server_url)

            logger.info(
                f"Using global metadata client with server url: {metadata_server_url}"
            )
        else:
            # Enable MLA optimization only when using the global metadata
            # client. Inline dict configs (test mode) are exempt from this
            # check — they use a fresh local metadata client per process and
            # don't actually share state between ranks.
            if final_is_mla and inline_config is None:
                raise ValueError(mla_unsupported_msg)

            # Use local metadata client for single-machine deployment
            metadata_client = Hf3fsLocalMetadataClient()

        rank_for_path = 0 if final_is_mla else final_rank
        # Allow per-pool overrides via extra_config["pools"], e.g.:
        #   {"pools": {"mamba": {"file_size_fraction": 0.1}}}
        # Defaults to ``_DEFAULT_AUX_FILE_SIZE_FRACTION`` of the KV file size
        # when omitted. See HiCacheHF3FS.register_mem_host_pool_v2.
        if pools_extra_config is None and isinstance(config.get("pools"), dict):
            pools_extra_config = config.get("pools")
        return HiCacheHF3FS(
            rank=final_rank,
            # Let all ranks use the same file path for MLA model
            file_path=f"{config['file_path_prefix']}.{rank_for_path}.bin",
            file_size=int(config["file_size"]),
            numjobs=int(config["numjobs"]),
            bytes_per_page=bytes_per_page,
            entries=int(config["entries"]),
            client_timeout=config.get("client_timeout", 5),
            dtype=dtype,
            metadata_client=metadata_client,
            is_mla_model=final_is_mla,
            is_page_first_layout=final_is_pfl,
            use_mock_client=use_mock_client,
            enable_storage_metrics=enable_storage_metrics,
            metadata_server_url=metadata_server_url,
            pools_extra_config=pools_extra_config,
        )

    def _batch_get(
        self,
        engine: _Hf3fsPoolEngine,
        keys: List[str],
        values: List[torch.Tensor],
    ) -> List[bool]:
        page_indices = engine.metadata_client.get_page_indices(
            engine.metadata_rank, keys
        )
        if len(page_indices) != len(keys):
            logger.error(
                f"[Rank {engine.metadata_rank}] HiCacheHF3FS get ({engine.pool_name}): "
                f"page_indices length {len(page_indices)} mismatch keys length {len(keys)}."
            )
            return [False] * len(keys)
        batch_indices, file_offsets = [], []
        for i, page_index in enumerate(page_indices):
            if page_index is not None:
                batch_indices.append(i)
                file_offsets.append(page_index * engine.bytes_per_page)

        for target_location in values:
            assert target_location.is_contiguous()
        file_results = values

        start_time = time.perf_counter()

        futures = [
            engine.executor.submit(
                engine.clients[engine.ac.next()].batch_read,
                file_offsets[i : i + self.entries],
                file_results[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        read_results = [result for future in futures for result in future.result()]

        end_time = time.perf_counter()
        ionum = len(batch_indices)

        if self.enable_storage_metrics and engine.pool_name == PoolName.KV:
            self.prefetch_pgs.append(ionum)
            self.prefetch_bandwidth.append(
                ionum / (end_time - start_time) * engine.gb_per_page
            )

        results = [False] * len(keys)
        for batch_index, read_result in zip(batch_indices, read_results):
            if read_result == engine.bytes_per_page:
                results[batch_index] = True
            else:
                logger.error(
                    f"[Rank {engine.metadata_rank}] HiCacheHF3FS get "
                    f"({engine.pool_name}) {keys[batch_index]} failed"
                )

        return results

    def _batch_set(
        self,
        engine: _Hf3fsPoolEngine,
        keys: List[str],
        values: Optional[Any] = None,
    ) -> List[bool]:
        # In MLA backend, only one rank needs to backup the KV cache. The
        # contract is a per-key list (not a scalar) — return a list of True
        # so v2 callers can faithfully report per-page status.
        if engine.skip_backup:
            return [True] * len(keys)

        # Todo: Add prefix block's hash key
        key_with_prefix = [(key, "") for key in keys]
        try:
            indices = engine.metadata_client.reserve_and_allocate_page_indices(
                engine.metadata_rank, key_with_prefix
            )
        except Exception as e:
            # The mini metadata server raises when its free-page list and
            # key-to-index map are both empty (over-allocation past the
            # configured num_pages). Surface that to the caller as a per-key
            # failure list rather than letting the exception escape — the
            # v2 contract guarantees a List[bool] result so the controller
            # can attribute the loss to this specific pool. PLAN.md §4 #5.
            logger.warning(
                "[Rank %s] HiCacheHF3FS batch_set (%s) capacity exhausted: %s",
                engine.metadata_rank,
                engine.pool_name,
                e,
            )
            return [False] * len(keys)
        if len(indices) != len(keys):
            logger.error(
                f"[Rank {engine.metadata_rank}] HiCacheHF3FS batch_set "
                f"({engine.pool_name}): mismatched lengths {len(indices)} != {len(keys)}"
            )
            # free allocated pages
            if indices:
                engine.metadata_client.confirm_write(
                    engine.metadata_rank, [], [index[1] for index in indices]
                )
            return [False] * len(keys)
        batch_indices, file_offsets, file_values = [], [], []
        pages_to_release = []

        for i, (value, (is_written, page_index)) in enumerate(zip(values, indices)):
            if is_written or page_index == -1:
                continue

            batch_indices.append(i)
            file_offsets.append(page_index * engine.bytes_per_page)
            assert value.is_contiguous()
            file_values.append(value)

        start_time = time.perf_counter()

        futures = [
            engine.executor.submit(
                engine.clients[engine.ac.next()].batch_write,
                file_offsets[i : i + self.entries],
                file_values[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        write_results = [
            result == engine.bytes_per_page
            for future in futures
            for result in future.result()
        ]

        end_time = time.perf_counter()
        ionum = len(batch_indices)

        if self.enable_storage_metrics and engine.pool_name == PoolName.KV:
            self.backup_pgs.append(ionum)
            self.backup_bandwidth.append(
                ionum / (end_time - start_time) * engine.gb_per_page
            )

        written_keys_to_confirm = []
        results = [index[0] for index in indices]
        for batch_index, write_result in zip(batch_indices, write_results):
            key = keys[batch_index]
            page_index = indices[batch_index][1]
            if write_result:
                written_keys_to_confirm.append((key, page_index))
            else:
                logger.error(
                    f"[Rank {engine.metadata_rank}] HiCacheHF3FS set "
                    f"({engine.pool_name}) {key} failed"
                )
                pages_to_release.append(page_index)
            results[batch_index] = write_result

        if len(written_keys_to_confirm) > 0 or len(pages_to_release) > 0:
            engine.metadata_client.confirm_write(
                engine.metadata_rank, written_keys_to_confirm, pages_to_release
            )

        return results

    def delete(self, key: str) -> None:
        self.metadata_client.delete_keys(self.rank, [key])

    def exists(self, key: str) -> bool:
        result = self.metadata_client.exists(self.rank, [key])
        return result[0] if result else False

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        factor = 1
        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            factor = 2

        results = self.metadata_client.exists(self.rank, keys)

        i = 0
        while i < len(keys) and results[i]:
            i += 1

        return i // factor

    def clear(self) -> None:
        for pool_name, engine in self._engines.items():
            try:
                engine.metadata_client.clear(engine.metadata_rank)
                logger.info(
                    f"Cleared HiCacheHF3FS pool={pool_name} rank={engine.metadata_rank}"
                )
            except Exception as e:
                logger.error(f"Failed to clear HiCacheHF3FS ({pool_name}): {e}")

    def close(self) -> None:
        # Iterate over every engine (KV plus any registered auxiliary pools)
        # and shut down its clients/executor. ``_engines`` is populated before
        # the SIGTERM/atexit hook is installed in __init__, so this is safe.
        for pool_name, engine in list(self._engines.items()):
            try:
                for c in engine.clients:
                    c.close()
                engine.executor.shutdown(wait=True)
            except Exception as e:
                logger.error(f"close HiCacheHF3FS ({pool_name}): {e}")
        logger.info("close HiCacheHF3FS")

    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        self.is_zero_copy = self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ]
        # Keep the KV engine in sync — v2 callers route through the engine.
        kv_engine = self._engines[PoolName.KV]
        kv_engine.host_pool = mem_pool_host
        kv_engine.is_zero_copy = self.is_zero_copy

        logger.info(f"{self.is_zero_copy=}, layout={self.mem_pool_host.layout}")

    # ------------------------------------------------------------------
    # v2 / per-pool plumbing
    # ------------------------------------------------------------------
    def _pool_metadata_rank(self, pool_name: str) -> int:
        """Return a metadata-server rank that is unique per (rank, pool).

        KV uses ``self.rank`` so existing layouts and persisted state stay
        compatible. Auxiliary pools are offset by a stable per-process pool
        index (allocated lazily) into a high-bit namespace so they cannot
        collide with KV state on a shared global metadata server.
        """
        if pool_name == PoolName.KV:
            return self.rank
        if pool_name not in self._pool_idx_map:
            self._pool_idx_map[pool_name] = self._next_pool_idx
            self._next_pool_idx += 1
        # Use the original (per-rank) rank as the base. Mamba/DSA state is
        # per-rank — even in MLA — so the metadata namespace must be too.
        return self._original_rank + (self._pool_idx_map[pool_name] << 24)

    def _make_pool_metadata_client(self) -> "Hf3fsMetadataInterface":
        """Create a fresh metadata client of the same kind as the KV one."""
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
            Hf3fsGlobalMetadataClient,
            Hf3fsLocalMetadataClient,
        )

        if self._metadata_server_url is not None:
            return Hf3fsGlobalMetadataClient(self._metadata_server_url)
        return Hf3fsLocalMetadataClient()

    def _aux_pool_file_size(self, pool_name: str) -> int:
        cfg = (self._pools_extra_config or {}).get(pool_name, {}) or {}
        fraction = float(
            cfg.get("file_size_fraction", _DEFAULT_AUX_FILE_SIZE_FRACTION)
        )
        if "file_size" in cfg:
            return int(cfg["file_size"])
        return max(int(self.file_size * fraction), 1)

    def _aux_pool_file_path(self, pool_name: str) -> str:
        # Mamba/DSA state is per-rank, even in MLA mode. Insert the
        # original rank into the path so per-rank backends never collide.
        base, ext = os.path.splitext(self.file_path)
        # Strip the trailing rank suffix from the KV path (".<rank>") so we
        # can substitute the *original* (per-rank) rank for aux pools.
        kv_rank_suffix = f".{self.rank}"
        if base.endswith(kv_rank_suffix):
            base = base[: -len(kv_rank_suffix)] + f".{self._original_rank}"
        return f"{base}.{pool_name}{ext or '.bin'}"

    def register_mem_host_pool_v2(
        self, host_pool: HostKVCache, host_pool_name
    ) -> None:
        """Register a host pool for the v2 multi-pool interface.

        Called by ``HybridCacheController.attach_storage_backend`` once per
        registered pool. The KV engine already exists from ``__init__``; for
        any auxiliary pool name we lazily allocate a dedicated 3FS file,
        client list, executor and metadata namespace. Idempotent and
        order-agnostic — re-registering the same pool just updates the
        bound host pool / layout flags.
        """
        pool_name = (
            host_pool_name.value
            if isinstance(host_pool_name, PoolName)
            else str(host_pool_name)
        )
        # PLAN.md §3 "PoolName.DSA" decision: start with KV + MAMBA only.
        # Any other pool name is rejected with a clear error so callers do
        # not silently end up writing to an unintended namespace.
        try:
            PoolName(pool_name)
        except ValueError as e:
            raise ValueError(
                f"HiCacheHF3FS: unknown pool name {host_pool_name!r}. "
                f"Supported pools: {[p.value for p in PoolName]}."
            ) from e
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

        with self.lock:
            if pool_name == PoolName.KV:
                kv_engine = self._engines[PoolName.KV]
                kv_engine.host_pool = host_pool
                # Mirror register_mem_pool_host (v1) so v2-only callers still
                # get the right zero-copy layout flag.
                kv_engine.is_zero_copy = host_pool.layout in [
                    "page_first",
                    "page_first_direct",
                ]
                # The host pool is the source of truth for the per-page byte
                # size. The factory only passes a tentative bytes_per_page in
                # __init__ (it doesn't yet know which actual host pool will
                # be bound), so we re-derive it here and update num_pages /
                # gb_per_page accordingly. The on-disk file is the same size
                # — only the carve-up changes.
                if host_pool.layout in ["page_first", "page_first_direct"]:
                    new_bpp = (
                        host_pool.get_ksize_per_token() * host_pool.page_size
                    )
                else:
                    new_bpp = (
                        host_pool.get_size_per_token() * host_pool.page_size
                    )
                if new_bpp != kv_engine.bytes_per_page:
                    kv_engine.bytes_per_page = new_bpp
                    kv_engine.num_pages = max(
                        kv_engine.file_size // new_bpp, 1
                    )
                    kv_engine.gb_per_page = new_bpp / (1 << 30)
                    # Re-initialize the metadata client so its free-page
                    # accounting matches the new num_pages.
                    kv_engine.metadata_client.initialize(
                        kv_engine.metadata_rank, kv_engine.num_pages
                    )
                    # Keep the legacy ``self.bytes_per_page`` / ``self.num_pages``
                    # in sync so v1 paths still see consistent values.
                    self.bytes_per_page = new_bpp
                    self.num_pages = kv_engine.num_pages
                    self.gb_per_page = kv_engine.gb_per_page
                self.mem_pool_host = host_pool
                self.is_zero_copy = kv_engine.is_zero_copy
                return

            existing = self._engines.get(pool_name)
            if existing is not None:
                # Idempotent re-registration: just rebind the host pool.
                existing.host_pool = host_pool
                return

            # Derive bytes_per_page from the auxiliary host pool's per-token
            # size, mirroring backend_factory.py's KV computation.
            if host_pool.layout in ["page_first", "page_first_direct"]:
                bytes_per_page = (
                    host_pool.get_ksize_per_token() * host_pool.page_size
                )
            else:
                bytes_per_page = (
                    host_pool.get_size_per_token() * host_pool.page_size
                )

            file_size = self._aux_pool_file_size(pool_name)
            # Round file_size up to a multiple of bytes_per_page so the
            # whole file can be carved into integral pages.
            num_pages = max(file_size // bytes_per_page, 1)
            file_size = num_pages * bytes_per_page

            file_path = self._aux_pool_file_path(pool_name)
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

            clients = [
                create_hf3fs_client(
                    file_path,
                    file_size,
                    bytes_per_page,
                    self.entries,
                    self.client_timeout,
                    self.use_mock_client,
                )
                for _ in range(self.numjobs)
            ]
            ac = AtomicCounter(self.numjobs)
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.numjobs,
                thread_name_prefix=(
                    f"HiCacheHF3FS-Rank{self._original_rank}-{pool_name}"
                ),
            )
            metadata_client = self._make_pool_metadata_client()
            metadata_rank = self._pool_metadata_rank(pool_name)
            metadata_client.initialize(metadata_rank, num_pages)

            engine = _Hf3fsPoolEngine(
                pool_name=pool_name,
                file_path=file_path,
                file_size=file_size,
                bytes_per_page=bytes_per_page,
                num_pages=num_pages,
                clients=clients,
                ac=ac,
                executor=executor,
                metadata_client=metadata_client,
                metadata_rank=metadata_rank,
                host_pool=host_pool,
                # Auxiliary pools never split heads — KV-only optimization.
                is_zero_copy=False,
                # Mamba/DSA state must be backed up on every rank.
                skip_backup=False,
                gb_per_page=bytes_per_page / (1 << 30),
            )
            self._engines[pool_name] = engine
            logger.info(
                "[Rank %s] HiCacheHF3FS registered aux pool=%s "
                "file_path=%s file_size=%.2f GB num_pages=%s bytes_per_page=%s",
                self._original_rank,
                pool_name,
                file_path,
                file_size / (1 << 30),
                num_pages,
                bytes_per_page,
            )

    def _pool_log_key(self, pool_name: str, key: str) -> str:
        """Apply the per-pool key namespace.

        KV uses the bare key for backwards compatibility with persisted
        state and v1 deployments. Auxiliary pools append a ``.<pool>``
        suffix, mirroring ``HiCacheFile`` semantics.
        """
        if pool_name == PoolName.KV:
            return key
        return f"{key}.{pool_name}"

    @staticmethod
    def _longest_prefix_true(values: List[bool]) -> int:
        i = 0
        while i < len(values) and values[i]:
            i += 1
        return i

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_engine = self._engines.get(PoolName.KV)
        if kv_engine is None:
            raise RuntimeError(
                "HiCacheHF3FS.batch_exists_v2 called before any host pool was "
                "registered. Call register_mem_host_pool_v2 (KV) first."
            )

        # KV existence (apply zero-copy MHA-split-head doubling at the KV
        # engine boundary, just like the v1 batch_exists path).
        kv_keys = list(keys)
        kv_factor = 1
        if kv_engine.is_zero_copy and not self.is_mla_model:
            kv_keys = self._get_mha_zero_copy_keys(kv_keys)
            kv_factor = 2

        kv_exists = kv_engine.metadata_client.exists(
            kv_engine.metadata_rank, kv_keys
        )
        kv_pages = self._longest_prefix_true(kv_exists) // kv_factor

        hit_count: Dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = (
                transfer.name.value
                if isinstance(transfer.name, PoolName)
                else str(transfer.name)
            )
            engine = self._engines.get(name)
            if engine is None:
                logger.error(
                    "HiCacheHF3FS.batch_exists_v2: pool %s is not registered", name
                )
                final_pages = 0
                hit_count[name] = 0
                break

            # Aux pools never apply MHA head splitting; pass keys through
            # with the pool suffix appended.
            pool_keys = [
                self._pool_log_key(name, k) for k in keys[:kv_pages]
            ]
            pool_exists = engine.metadata_client.exists(
                engine.metadata_rank, pool_keys
            )

            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                # Longest contiguous [0, b) where every page exists.
                boundary = next(
                    (i for i in range(len(pool_exists)) if not pool_exists[i]),
                    len(pool_exists),
                )
            else:
                # TRAILING_PAGES: only the last `trailing` pages of [0, kv_pages)
                # need to exist. Mirrors HiCacheFile.batch_exists_v2 lines
                # 434-443 — keep the semantics in sync.
                trailing = max(
                    1, len(transfer.keys) if transfer.keys else 1
                )
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        pool_exists[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        if not hit_count and final_pages == 0:
            return PoolTransferResult.empty()
        return PoolTransferResult(final_pages, hit_count)

    @staticmethod
    def _normalize_v2_host_indices(
        host_indices: Any,
        num_pages: int,
        page_size: int,
    ) -> Optional[List[int]]:
        """Return one slot offset per page from a transfer's host_indices.

        ``host_indices`` may be either:
            * a 1D tensor (or list) of length ``num_pages * page_size`` —
              the canonical slot-array passed by HybridCacheController. Each
              page-i slot start is at index ``i * page_size``.
            * a 1D tensor (or list) of length ``num_pages`` — a compact
              page-indexed array. Each entry is multiplied by ``page_size``
              to recover the slot offset.

        Returns ``None`` when the length matches neither convention.
        """
        if host_indices is None:
            return None
        if hasattr(host_indices, "tolist") and not isinstance(host_indices, list):
            try:
                flat = host_indices.tolist()
            except Exception:
                flat = list(host_indices)
        else:
            flat = list(host_indices)
        n = len(flat)
        if n == num_pages:
            return [int(p) * page_size for p in flat]
        if n == num_pages * page_size:
            return [int(flat[i * page_size]) for i in range(num_pages)]
        return None

    def _resolve_v2_transfer(
        self,
        transfer: PoolTransfer,
    ) -> Tuple[
        Optional[_Hf3fsPoolEngine],
        Optional[List[str]],
        Optional[List[int]],
        int,
    ]:
        """Build the per-page (engine, keys, slot offsets) for a transfer.

        Returns ``(engine, keys, slot_offsets, page_count)``. ``page_count``
        is the number of host pages this transfer asks us to move; the
        per-page result list returned to the controller must always have
        this length.
        """
        name = (
            transfer.name.value
            if isinstance(transfer.name, PoolName)
            else str(transfer.name)
        )
        engine = self._engines.get(name)
        keys = transfer.keys or []
        page_count = len(keys)
        if engine is None:
            raise RuntimeError(
                f"HiCacheHF3FS: pool {name!r} is not registered. "
                f"Call register_mem_host_pool_v2({name!r}, host_pool) before "
                "issuing v2 transfers."
            )

        host_pool = engine.host_pool
        if host_pool is None:
            raise RuntimeError(
                f"HiCacheHF3FS: pool {name!r} is registered but no host pool "
                "is bound. Call register_mem_host_pool_v2 with a HostKVCache."
            )

        page_size = getattr(host_pool, "page_size", 1) or 1
        slot_offsets = self._normalize_v2_host_indices(
            transfer.host_indices, page_count, page_size
        )
        if slot_offsets is None and page_count > 0:
            n = (
                transfer.host_indices.numel()
                if transfer.host_indices is not None
                and hasattr(transfer.host_indices, "numel")
                else (
                    len(transfer.host_indices)
                    if transfer.host_indices is not None
                    else 0
                )
            )
            logger.error(
                "HiCacheHF3FS v2 transfer for pool %s: indices length "
                "mismatch (got %s for %s pages, page_size=%s)",
                name,
                n,
                page_count,
                page_size,
            )
            return None, None, None, page_count

        return engine, keys, slot_offsets, page_count

    def _v2_read_dest(
        self,
        host_pool: Any,
        slot_offset: int,
        bytes_per_page: int,
        is_zero_copy: bool,
    ) -> torch.Tensor:
        """Return a tensor that the storage backend will write into.

        For zero-copy layouts the host pool exposes a write-through view
        via ``get_data_page(slot, flat=False)``. For non-zero-copy layouts
        we hand out a scratch buffer (``get_dummy_flat_data_page``) and the
        result is later copied back via ``set_from_flat_data_page``.

        When the host pool is a minimal stub that lacks these methods we
        fall back to slicing ``host_pool.kv_buffer`` directly so the unit
        tests' ``_FakeHostKVCache`` can still round-trip bytes.
        """
        if is_zero_copy and hasattr(host_pool, "get_data_page"):
            return host_pool.get_data_page(slot_offset, flat=False)
        if hasattr(host_pool, "get_dummy_flat_data_page"):
            return host_pool.get_dummy_flat_data_page()
        # Fallback: directly slice the host pool's flat buffer.
        flat = host_pool.kv_buffer.view(torch.uint8).reshape(-1)
        page_byte_size = bytes_per_page
        # Convert slot offset → byte offset using the per-token byte size
        # implied by the engine's bytes_per_page and the host's page_size.
        page_size = getattr(host_pool, "page_size", 1) or 1
        bytes_per_token = page_byte_size // page_size
        start = slot_offset * bytes_per_token
        return flat[start : start + page_byte_size]

    def _v2_finalize_read(
        self,
        host_pool: Any,
        slot_offset: int,
        scratch: torch.Tensor,
        is_zero_copy: bool,
    ) -> None:
        if is_zero_copy:
            return
        if hasattr(host_pool, "set_from_flat_data_page"):
            host_pool.set_from_flat_data_page(slot_offset, scratch)
            return
        # Fallback path: copy scratch into the kv_buffer slice. Skip when
        # the scratch already aliases the buffer (the slice fallback
        # returns a view into kv_buffer, which makes copy a no-op).
        flat = host_pool.kv_buffer.view(torch.uint8).reshape(-1)
        page_size = getattr(host_pool, "page_size", 1) or 1
        bytes_per_token = scratch.numel() // page_size if page_size else scratch.numel()
        start = slot_offset * bytes_per_token
        target = flat[start : start + scratch.numel()]
        if target.data_ptr() != scratch.data_ptr():
            target.copy_(scratch.contiguous().view(torch.uint8).reshape(-1))

    def _v2_write_source(
        self,
        host_pool: Any,
        slot_offset: int,
        bytes_per_page: int,
        is_zero_copy: bool,
    ) -> torch.Tensor:
        if hasattr(host_pool, "get_data_page"):
            return host_pool.get_data_page(slot_offset, flat=not is_zero_copy)
        # Fallback: slice kv_buffer directly.
        flat = host_pool.kv_buffer.view(torch.uint8).reshape(-1)
        page_size = getattr(host_pool, "page_size", 1) or 1
        bytes_per_token = bytes_per_page // page_size
        start = slot_offset * bytes_per_token
        return flat[start : start + bytes_per_page]

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        results: Dict[str, List[bool]] = {}
        for transfer in transfers:
            engine, keys, slot_offsets, page_count = self._resolve_v2_transfer(
                transfer
            )
            name = (
                transfer.name.value
                if isinstance(transfer.name, PoolName)
                else str(transfer.name)
            )
            if engine is None or not keys or slot_offsets is None:
                results[name] = [False] * page_count
                continue

            host_pool = engine.host_pool
            apply_mha = (
                engine.pool_name == PoolName.KV
                and engine.is_zero_copy
                and not self.is_mla_model
            )

            values = [
                self._v2_read_dest(
                    host_pool,
                    slot_offsets[i],
                    engine.bytes_per_page,
                    engine.is_zero_copy,
                )
                for i in range(page_count)
            ]

            log_keys = [self._pool_log_key(engine.pool_name, k) for k in keys]
            io_values = values
            if apply_mha:
                log_keys = self._get_mha_zero_copy_keys(log_keys)
                io_values = self._get_mha_zero_copy_values(values)

            raw_results = self._batch_get(engine, log_keys, io_values)

            if apply_mha:
                page_results = [
                    bool(raw_results[2 * i] and raw_results[2 * i + 1])
                    for i in range(page_count)
                ]
            else:
                page_results = [bool(r) for r in raw_results[:page_count]]

            if not engine.is_zero_copy:
                # Copy each scratch buffer back into the host pool slot.
                for i in range(page_count):
                    if not page_results[i]:
                        break
                    self._v2_finalize_read(
                        host_pool, slot_offsets[i], values[i], engine.is_zero_copy
                    )

            results[name] = page_results
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        results: Dict[str, List[bool]] = {}
        for transfer in transfers:
            engine, keys, slot_offsets, page_count = self._resolve_v2_transfer(
                transfer
            )
            name = (
                transfer.name.value
                if isinstance(transfer.name, PoolName)
                else str(transfer.name)
            )
            if engine is None or not keys or slot_offsets is None:
                results[name] = [False] * page_count
                continue

            host_pool = engine.host_pool
            apply_mha = (
                engine.pool_name == PoolName.KV
                and engine.is_zero_copy
                and not self.is_mla_model
            )

            values = [
                self._v2_write_source(
                    host_pool,
                    slot_offsets[i],
                    engine.bytes_per_page,
                    engine.is_zero_copy,
                )
                for i in range(page_count)
            ]

            log_keys = [self._pool_log_key(engine.pool_name, k) for k in keys]
            if apply_mha:
                log_keys = self._get_mha_zero_copy_keys(log_keys)
                values = self._get_mha_zero_copy_values(values)

            raw_results = self._batch_set(engine, log_keys, values)

            if apply_mha:
                page_results = [
                    bool(raw_results[2 * i] and raw_results[2 * i + 1])
                    for i in range(page_count)
                ]
            else:
                page_results = [bool(r) for r in raw_results[:page_count]]

            results[name] = page_results
        return results

    def _get_mha_zero_copy_keys(self, keys: List[str]) -> List[str]:
        _keys = []
        for k in keys:
            _keys.append(f"{k}-k")
            _keys.append(f"{k}-v")
        return _keys

    def _get_mha_zero_copy_values(
        self, values: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        _values = []
        for value in values:
            _values.append(value[0])
            _values.append(value[1])
        return _values

    def _batch_get_preprocess(self, keys, host_indices):
        page_num = len(host_indices) // self.mem_pool_host.page_size
        # host_indices to kv_buffer
        flat = not self.is_zero_copy
        values = (
            [
                self.mem_pool_host.get_data_page(
                    host_indices[i * self.mem_pool_host.page_size], flat=flat
                )
                for i in range(page_num)
            ]
            if self.is_zero_copy
            else [
                self.mem_pool_host.get_dummy_flat_data_page() for _ in range(page_num)
            ]
        )

        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def _batch_get_postprocess(self, host_indices, values, results):
        page_num = len(host_indices) // self.mem_pool_host.page_size

        if self.is_zero_copy:
            if not self.is_mla_model:
                results = [
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)
                ]
                results = results[:page_num]
            return results

        for i in range(page_num):
            if not results[i]:
                break
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.mem_pool_host.page_size], values[i]
            )

        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        keys, values = self._batch_get_preprocess(keys, host_indices)
        results = self._batch_get(self._engines[PoolName.KV], keys, values)
        return self._batch_get_postprocess(host_indices, values, results)

    def _batch_set_preprocess(self, keys, host_indices):
        page_num = len(host_indices) // self.mem_pool_host.page_size
        # host_indices to kv_buffer
        flat = not self.is_zero_copy
        values = [
            self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size], flat=flat
            )
            for i in range(page_num)
        ]

        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        len_keys = len(keys)
        keys, values = self._batch_set_preprocess(keys, host_indices)
        results = self._batch_set(self._engines[PoolName.KV], keys, values)
        return results

    # Deprecated
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        pass

    # Deprecated
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        pass

    # Deprecated
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    # Deprecated
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass
