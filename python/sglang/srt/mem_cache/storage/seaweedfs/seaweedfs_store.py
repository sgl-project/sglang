"""SeaweedFS backend for HiCache L3 storage.

Each KV page is one object in a SeaweedFS bucket, reached through SeaweedFS's S3 gateway. The
cache controller drives the zero-copy interface: ``batch_get_v1`` / ``batch_set_v1`` move a
page between its object and the host pool's own buffers (from ``get_page_buffer_meta``),
returning one success flag per page. An object holds the page's buffers back to back, which
for the page-first layouts is byte-identical to the flat page of the generic ``batch_get`` /
``batch_set`` interface, so the two interfaces read each other's objects.
``batch_exists`` returns the length of the leading run of existing keys.

Configuration comes from ``--hicache-storage-backend-extra-config``; only ``endpoint`` is
required. When ``access_key``/``secret_key`` are absent the standard AWS credential chain is
used, which keeps secrets off the command line.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    mla_tp_shard_tag,
)

logger = logging.getLogger(__name__)

# S3 keys are capped at 1024 bytes; longer page keys are hashed down below this.
_MAX_KEY_LEN = 900
_MISSING_CODES = ("404", "NoSuchKey", "NoSuchBucket", "NotFound")


class SeaweedFSConfig(msgspec.Struct, frozen=True, kw_only=True):
    endpoint: str
    bucket: str = "sglang-hicache"
    prefix: str = ""
    region: str = "us-east-1"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    max_workers: int = 16

    @classmethod
    def from_extra_config(cls, extra_config: Optional[dict]) -> SeaweedFSConfig:
        # extra_config also carries factory keys (backend_name, module_path, ...).
        fields = set(cls.__struct_fields__)
        return msgspec.convert(
            {k: v for k, v in (extra_config or {}).items() if k in fields}, type=cls
        )


def _make_s3_client(config: SeaweedFSConfig):
    import boto3
    from botocore.config import Config as BotoConfig

    credentials = {}
    if config.access_key and config.secret_key:
        credentials = {
            "aws_access_key_id": config.access_key,
            "aws_secret_access_key": config.secret_key,
        }
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint,
        region_name=config.region,
        # Path-style: virtual-hosted addressing needs per-bucket DNS a cluster gateway lacks.
        config=BotoConfig(
            s3={"addressing_style": "path"},
            max_pool_connections=max(config.max_workers, 10),
            retries={"max_attempts": 3, "mode": "standard"},
        ),
        **credentials,
    )


def key_scope(storage_config: HiCacheStorageConfig) -> List[str]:
    """Model and rank components of an object key; mirrors HiCacheFile's suffix rules."""
    parts = []
    if storage_config.model_name:
        parts.append("-".join(storage_config.model_name.split("/")))
    # MLA ranks hold identical KV and only one of them backs up, so they share keys;
    # MHA ranks hold different heads and must not.
    if not storage_config.is_mla_model:
        parts.append(f"tp{storage_config.tp_rank}of{storage_config.tp_size}")
    if storage_config.pp_size > 1:
        parts.append(f"pp{storage_config.pp_rank}of{storage_config.pp_size}")
    if storage_config.attn_cp_size > 1:
        parts.append(f"cp{storage_config.attn_cp_rank}of{storage_config.attn_cp_size}")
    return parts


def _byte_view(tensor: torch.Tensor) -> memoryview:
    # Through uint8 so bf16/fp8 pages work; numpy has no such dtypes.
    return memoryview(tensor.reshape(-1).view(torch.uint8).numpy())


class SeaweedFSStore(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig, mem_pool_host: Any = None
    ) -> None:
        self.config = SeaweedFSConfig.from_extra_config(storage_config.extra_config)
        scope = [self.config.prefix.strip("/")] if self.config.prefix else []
        self.key_prefix = "/".join(scope + key_scope(storage_config))
        self._storage_config = storage_config
        self._s3 = _make_s3_client(self.config)
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._ensure_bucket()
        logger.info(
            "SeaweedFS HiCache backend: endpoint=%s bucket=%s prefix=%s",
            self.config.endpoint,
            self.config.bucket,
            self.key_prefix,
        )

    def _is_missing(self, exc: Exception) -> bool:
        return (
            isinstance(exc, self._s3.exceptions.ClientError)
            and str(exc.response["Error"]["Code"]) in _MISSING_CODES
        )

    def _ensure_bucket(self) -> None:
        try:
            self._s3.head_bucket(Bucket=self.config.bucket)
        except Exception as exc:
            if not self._is_missing(exc):
                raise
            self._s3.create_bucket(Bucket=self.config.bucket)

    def _object_key(self, key: str) -> str:
        full = f"{self.key_prefix}/{key}" if self.key_prefix else key
        if len(full) <= _MAX_KEY_LEN:
            return full
        return f"{full[:800]}/{hashlib.sha256(full.encode()).hexdigest()}"

    def _fetch(self, key: str) -> Optional[bytes]:
        try:
            resp = self._s3.get_object(
                Bucket=self.config.bucket, Key=self._object_key(key)
            )
            return resp["Body"].read()
        except Exception as exc:
            if not self._is_missing(exc):
                logger.warning("SeaweedFS get %s failed: %s", key, exc)
            return None

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        data = self._fetch(key)
        if data is None:
            return None
        if target_location is None:
            return torch.frombuffer(bytearray(data), dtype=torch.uint8)
        nbytes = target_location.numel() * target_location.element_size()
        # A size mismatch is a stale or foreign object; copying it would overrun the page.
        if len(data) != nbytes or not target_location.is_contiguous():
            logger.warning(
                "SeaweedFS object %s is %d bytes, page is %d", key, len(data), nbytes
            )
            return None
        _byte_view(target_location)[:] = data
        return target_location

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        locations = target_locations or [None] * len(keys)
        futures = [
            self._executor.submit(self.get, key, location)
            for key, location in zip(keys, locations)
        ]
        return [f.result() for f in futures]

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        page = value if value is not None else target_location
        try:
            body = bytes(_byte_view(page.contiguous()))
            self._s3.put_object(
                Bucket=self.config.bucket, Key=self._object_key(key), Body=body
            )
            return True
        except Exception as exc:
            logger.error("SeaweedFS set %s failed: %s", key, exc)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pages = values if values is not None else target_locations
        futures = [
            self._executor.submit(self.set, key, page) for key, page in zip(keys, pages)
        ]
        return all([f.result() for f in futures])

    def _page_buffers(self, host_indices: torch.Tensor) -> List[List[tuple]]:
        """(address, nbytes) of each host-pool buffer of every page, in page order."""
        addresses, sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        num_pages = len(host_indices) // self.mem_pool_host.page_size
        per_page = len(addresses) // num_pages
        if isinstance(sizes, int):
            sizes = [sizes] * len(addresses)
        return [
            list(
                zip(
                    addresses[i * per_page : (i + 1) * per_page],
                    sizes[i * per_page : (i + 1) * per_page],
                )
            )
            for i in range(num_pages)
        ]

    def _get_into(self, key: str, buffers: List[tuple]) -> bool:
        data = self._fetch(key)
        if data is None:
            return False
        expected = sum(nbytes for _, nbytes in buffers)
        # A size mismatch is a stale or foreign object; copying it would overrun the page.
        if len(data) != expected:
            logger.warning(
                "SeaweedFS object %s is %d bytes, page is %d", key, len(data), expected
            )
            return False
        src = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value
        offset = 0
        for address, nbytes in buffers:
            ctypes.memmove(address, src + offset, nbytes)
            offset += nbytes
        return True

    def _put_from(self, key: str, buffers: List[tuple]) -> bool:
        try:
            body = bytearray(sum(nbytes for _, nbytes in buffers))
            dst = ctypes.addressof(ctypes.c_char.from_buffer(body))
            offset = 0
            for address, nbytes in buffers:
                ctypes.memmove(dst + offset, address, nbytes)
                offset += nbytes
            self._s3.put_object(
                Bucket=self.config.bucket, Key=self._object_key(key), Body=body
            )
            return True
        except Exception as exc:
            logger.error("SeaweedFS set %s failed: %s", key, exc)
            return False

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        pages = self._page_buffers(host_indices)
        futures = [
            self._executor.submit(self._get_into, key, buffers)
            for key, buffers in zip(keys, pages)
        ]
        return [f.result() for f in futures]

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        pages = self._page_buffers(host_indices)
        futures = [
            self._executor.submit(self._put_from, key, buffers)
            for key, buffers in zip(keys, pages)
        ]
        return [f.result() for f in futures]

    # Hybrid models (Mamba, SWA, DeepSeek V4 indexer and other side pools) go
    # through the v2 interface. Side-pool pages are stored next to their KV page
    # as "<key>.<pool>"; KV pages keep the same object names as v1, so a cache
    # written by either path is readable by the other.

    def _component_key(self, key: str, pool_name) -> str:
        name = getattr(pool_name, "value", pool_name)
        if name == PoolName.KV.value:
            return key
        cfg = self._storage_config
        tag = mla_tp_shard_tag(cfg.is_mla_model, cfg.tp_rank, cfg.tp_size, name)
        return f"{key}.{name}.{tag}" if tag else f"{key}.{name}"

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        hit_count: Dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages
        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            futures = [
                self._executor.submit(
                    self.exists, self._component_key(keys[i], transfer.name)
                )
                for i in range(kv_pages)
            ]
            present = [f.result() for f in futures]
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next((i for i, ok in enumerate(present) if not ok), kv_pages)
            else:  # TRAILING_PAGES: only the window ending at the prefix must exist
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = next(
                    (
                        n
                        for n in range(kv_pages, 0, -1)
                        if all(present[max(0, n - trailing) : n])
                    ),
                    0,
                )
            if boundary:
                hit_count[transfer.name] = boundary
            final_pages = min(final_pages, boundary)
        return PoolTransferResult(final_pages, hit_count)

    def _read_page_v2(self, pool_name, key: str, host_pool, offset: int) -> bool:
        page = host_pool.get_dummy_flat_data_page()
        if self.get(self._component_key(key, pool_name), page) is None:
            return False
        host_pool.set_from_flat_data_page(offset, page)
        return True

    def _write_page_v2(self, pool_name, key: str, host_pool, offset: int) -> bool:
        page = host_pool.get_data_page(offset, flat=True)
        return self.set(self._component_key(key, pool_name), page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], page_fn):
        results: Dict[str, List[bool]] = {}
        pending = []
        for transfer in transfers:
            keys = transfer.keys or []
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            if host_pool is None:
                logger.error("SeaweedFS: host pool %s is not registered", transfer.name)
                results[transfer.name] = [False] * len(keys)
                continue
            page_size = getattr(host_pool, "page_size", 1) or 1
            indices = transfer.host_indices
            if indices is None or indices.numel() != len(keys) * page_size:
                logger.error(
                    "SeaweedFS: %s has %d keys but %s host indices (page size %d)",
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
            pending.append((transfer.name, futures))
        for name, futures in pending:
            results[name] = [f.result() for f in futures]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page_v2)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page_v2)

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.config.bucket, Key=self._object_key(key))
            return True
        except Exception as exc:
            if not self._is_missing(exc):
                logger.warning("SeaweedFS exists %s failed: %s", key, exc)
            return False

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        futures = [self._executor.submit(self.exists, key) for key in keys]
        for i, f in enumerate(futures):
            if not f.result():
                return i
        return len(keys)

    def clear(self) -> None:
        # Only this rank's prefix; other models and ranks may share the bucket.
        prefix = f"{self.key_prefix}/" if self.key_prefix else ""
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objects:
                self._s3.delete_objects(
                    Bucket=self.config.bucket, Delete={"Objects": objects}
                )
