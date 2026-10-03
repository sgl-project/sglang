"""SeaweedFS backend for HiCache L3 storage.

Each KV page is one object in a SeaweedFS bucket, reached through SeaweedFS's S3 gateway. The
backend implements the generic tensor-page interface driven by the cache controller: pages are
flat host tensors, ``batch_get`` returns one entry per key (``None`` on a miss), and
``batch_exists`` returns the length of the leading run of existing keys.

Configuration comes from ``--hicache-storage-backend-extra-config``; only ``endpoint`` is
required. When ``access_key``/``secret_key`` are absent the standard AWS credential chain is
used, which keeps secrets off the command line.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
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
