"""Google Cloud Storage backend for HiCache L3 storage.

Each KV page is one object in a GCS bucket. A page's object name is its cache
key plus the suffix from :func:`resolve_storage_key_scheme`: with the unified
key scheme one page becomes one object per layer/head chunk, named by the
topology-free coordinate, so TP/PP/CP/DCP deployments that tile the same grid
share objects. Otherwise the model/rank suffix from the storage config is
appended. This backend never builds rank coordinates itself.

A read whose object length does not match the target page is treated as a
miss, so a truncated or foreign object is never copied into the host pool.
Credential and configuration errors surface at attach time instead of turning
every read into a silent miss.

Configuration keys in ``--hicache-storage-backend-extra-config``:

  bucket        bucket name (required; the bucket is never created here)
  prefix        object-name prefix shared by this deployment (default: "")
  project       GCP project for billing and quota (default: credential's)
  credentials   path to a service-account JSON key (default: ADC)
  max_workers   parallel upload/download threads (default 16)

``google-cloud-storage`` is imported lazily, so the backend can be registered
without the SDK installed; attaching it without the SDK raises an ImportError
that names the package to install.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    l3_page_span,
    resolve_storage_key_scheme,
)
from sglang.srt.mem_cache.storage.gcs.config import GCSConfig

logger = logging.getLogger(__name__)

# GCS object names are capped at 1024 bytes; longer names are hashed down
# below this limit, leaving room for the digest.
_MAX_OBJECT_NAME_BYTES = 900


def _make_gcs_client(config: GCSConfig):
    """Build the client, with an actionable error when the SDK is absent."""
    try:
        from google.cloud import storage as gcs
    except ImportError as exc:
        raise ImportError(
            "The GCS HiCache backend requires google-cloud-storage. Install it "
            "with `pip install google-cloud-storage`, or choose another "
            "--hicache-storage-backend."
        ) from exc
    if config.credentials:
        return gcs.Client.from_service_account_json(
            config.credentials, project=config.project
        )
    return gcs.Client(project=config.project)


def _byte_view(tensor: torch.Tensor) -> memoryview:
    # Through uint8 so bf16/fp8 pages work; numpy has no such dtypes.
    return memoryview(tensor.reshape(-1).view(torch.uint8).numpy())


class GcsStore(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig, mem_pool_host: Any = None
    ) -> None:
        self.config = GCSConfig.from_extra_config(storage_config.extra_config)
        scheme = resolve_storage_key_scheme(storage_config)
        self.config_suffix = scheme.suffix
        self.tp_config_suffix = scheme.tp_suffix
        self._unified_suffixes = scheme.unified_suffixes
        self._unified_chunk_spans = scheme.unified_chunk_spans
        self.key_prefix = "/".join(
            part for part in (self.config.prefix.strip("/"),) if part
        )
        self._client = _make_gcs_client(self.config)
        self._bucket = self._client.bucket(self.config.bucket)
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix=f"hicache-gcs-{storage_config.tp_rank}",
        )
        logger.info(
            "GCS HiCache backend: bucket=%s prefix=%s",
            self.config.bucket,
            self.key_prefix or "<none>",
        )

    # ---- object naming -----------------------------------------------------

    def _object_name(self, key: str, suffix: str) -> str:
        name = f"{key}{suffix}"
        if self.key_prefix:
            name = f"{self.key_prefix}/{name}"
        if len(name.encode("utf-8")) <= _MAX_OBJECT_NAME_BYTES:
            return name
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        head = name[: _MAX_OBJECT_NAME_BYTES - 65].rstrip("/")
        return f"{head}/{digest}"

    def _kv_page_parts(self, key: str) -> List[tuple]:
        """``(object name, offset, length)`` of each object holding this KV page.

        ``length`` is None when the object holds the whole page.
        """
        if self._unified_suffixes is None:
            return [(self._object_name(key, self.config_suffix), 0, None)]
        spans = self._unified_chunk_spans
        if spans is None:
            return [(self._object_name(key, self._unified_suffixes[0]), 0, None)]
        return [
            (self._object_name(key, suffix), offset, length)
            for suffix, (offset, length) in zip(self._unified_suffixes, spans)
        ]

    def _component_parts(self, key: str, pool_name) -> List[tuple]:
        name = getattr(pool_name, "value", pool_name)
        if self._unified_suffixes is not None:
            if name not in (None, "__default__", PoolName.KV.value):
                raise NotImplementedError(
                    "Unified GCS storage does not support sidecar pools."
                )
            return self._kv_page_parts(key)
        if name in (None, "__default__", PoolName.KV.value):
            return [(self._object_name(key, self.config_suffix), 0, None)]
        log_key = f"{key}.{name}"
        # Mamba state is TP-sharded even when the model's MLA KV is replicated.
        suffix = (
            self.tp_config_suffix
            if name == PoolName.MAMBA.value
            else self.config_suffix
        )
        return [(self._object_name(log_key, suffix), 0, None)]

    def _component_key(self, key: str, pool_name) -> tuple:
        return tuple(part[0] for part in self._component_parts(key, pool_name))

    # ---- single-object IO --------------------------------------------------

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        # google.api_core.exceptions.NotFound carries HTTP 404.
        return getattr(exc, "code", None) == 404

    def _download(self, object_name: str) -> Optional[bytes]:
        try:
            return self._bucket.blob(object_name).download_as_bytes()
        except Exception as exc:
            if not self._is_missing(exc):
                logger.warning("GCS get %s failed: %s", object_name, exc)
            return None

    def _upload(self, object_name: str, data: bytes) -> bool:
        try:
            self._bucket.blob(object_name).upload_from_string(
                data, content_type="application/octet-stream"
            )
            return True
        except Exception as exc:
            logger.error("GCS set %s failed: %s", object_name, exc)
            return False

    def _exists_name(self, object_name: str) -> bool:
        try:
            return bool(self._bucket.blob(object_name).exists())
        except Exception as exc:
            if not self._is_missing(exc):
                logger.warning("GCS exists %s failed: %s", object_name, exc)
            return False

    # ---- page assembly -----------------------------------------------------

    def _read_component(self, key: str, pool_name) -> Optional[bytes]:
        parts = self._component_parts(key, pool_name)
        if len(parts) == 1 and parts[0][2] is None:
            return self._download(parts[0][0])
        page = bytearray(sum(length for _, _, length in parts))
        for object_name, offset, length in parts:
            chunk = self._download(object_name)
            if chunk is None or len(chunk) != length:
                logger.warning(
                    "GCS object %s is %s bytes, expected %d",
                    object_name,
                    len(chunk) if chunk is not None else "missing",
                    length,
                )
                return None
            page[offset : offset + length] = chunk
        return bytes(page)

    def _write_component(self, key: str, pool_name, data: bytes) -> bool:
        parts = self._component_parts(key, pool_name)
        if len(parts) == 1 and parts[0][2] is None:
            return self._upload(parts[0][0], data)
        expected = sum(length for _, _, length in parts)
        if len(data) != expected:
            raise ValueError(
                f"Unified chunk spans cover {expected} bytes, but the page has {len(data)}."
            )
        for object_name, offset, length in parts:
            if not self._upload(object_name, data[offset : offset + length]):
                return False
        return True

    # ---- HiCacheStorage interface -----------------------------------------

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        data = self._read_component(key, PoolName.KV)
        if data is None:
            return None
        if target_location is None:
            return torch.frombuffer(bytearray(data), dtype=torch.uint8)
        nbytes = target_location.numel() * target_location.element_size()
        # A size mismatch is a stale or foreign object; copying it would
        # overrun the page or silently restore truncated KV.
        if len(data) != nbytes or not target_location.is_contiguous():
            logger.warning(
                "GCS object for %s is %d bytes, page is %d", key, len(data), nbytes
            )
            return None
        _byte_view(target_location)[:] = data
        return target_location

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        locations = target_locations or [None] * len(keys)
        futures = [
            self._executor.submit(self.get, key, location)
            for key, location in zip(keys, locations)
        ]
        return [future.result() for future in futures]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        page = value if value is not None else target_location
        if page is None:
            return False
        data = bytes(_byte_view(page.contiguous()))
        return self._write_component(key, PoolName.KV, data)

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pages = values if values is not None else target_locations
        futures = [
            self._executor.submit(self.set, key, page) for key, page in zip(keys, pages)
        ]
        return all(future.result() for future in futures)

    def exists(self, key: str) -> bool:
        return all(
            self._exists_name(object_name)
            for object_name in self._component_key(key, PoolName.KV)
        )

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        futures = [self._executor.submit(self.exists, key) for key in keys]
        for i, future in enumerate(futures):
            if not future.result():
                return i
        return len(keys)

    # ---- v2 interface: KV plus hybrid side pools --------------------------

    def _exists_component(self, key: str, pool_name) -> bool:
        futures = [
            self._executor.submit(self._exists_name, object_name)
            for object_name in self._component_key(key, pool_name)
        ]
        return all(future.result() for future in futures)

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        hit_count: Dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))

        for transfer in pool_transfers or []:
            if not restorable:
                break
            name = transfer.name
            present = [self._exists_component(keys[i], name) for i in range(kv_pages)]
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next((i for i, ok in enumerate(present) if not ok), kv_pages)
                pool_restorable = set(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                pool_restorable = {
                    prefix_len
                    for prefix_len in range(1, kv_pages + 1)
                    if all(present[max(0, prefix_len - trailing) : prefix_len])
                }
                boundary = max(pool_restorable, default=0)
            else:
                raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")
            if boundary:
                hit_count[name] = boundary
            restorable = [page for page in restorable if page in pool_restorable]

        final_pages = restorable[-1] if restorable else 0
        return PoolTransferResult(final_pages, hit_count, restorable)

    def _read_page(self, pool_name, key: str, host_pool, offset: int) -> bool:
        data = self._read_component(key, pool_name)
        if data is None:
            return False
        page = host_pool.get_dummy_flat_data_page()
        nbytes = page.numel() * page.element_size()
        if len(data) != nbytes:
            logger.warning(
                "GCS object for %s is %d bytes, page is %d", key, len(data), nbytes
            )
            return False
        _byte_view(page)[:] = data
        host_pool.set_from_flat_data_page(offset, page)
        return True

    def _write_page(self, pool_name, key: str, host_pool, offset: int) -> bool:
        page = host_pool.get_data_page(offset, flat=True)
        return self._write_component(
            key, pool_name, bytes(_byte_view(page.contiguous()))
        )

    def _batch_io_v2(self, transfers: List[PoolTransfer], page_fn):
        results: Dict[str, List[bool]] = {}
        for transfer in transfers:
            keys = transfer.keys or []
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            if host_pool is None:
                logger.error("GCS: host pool %s is not registered", transfer.name)
                results[transfer.name] = [False] * len(keys)
                continue
            page_size = l3_page_span(host_pool)
            indices = transfer.host_indices
            if indices is None or indices.numel() != len(keys) * page_size:
                logger.error(
                    "GCS: %s has %d keys but %s host indices (page span %d)",
                    transfer.name,
                    len(keys),
                    indices.numel() if indices is not None else 0,
                    page_size,
                )
                results[transfer.name] = [False] * len(keys)
                continue
            futures = [
                self._executor.submit(
                    page_fn,
                    transfer.name,
                    key,
                    host_pool,
                    indices[i * page_size].item(),
                )
                for i, key in enumerate(keys)
            ]
            results[transfer.name] = [future.result() for future in futures]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> None:
        # Only this deployment's prefix; other models and ranks may share the
        # bucket.
        prefix = f"{self.key_prefix}/" if self.key_prefix else ""
        try:
            for blob in self._bucket.list_blobs(prefix=prefix):
                blob.delete()
        except Exception as exc:
            logger.error("GCS clear failed: %s", exc)
