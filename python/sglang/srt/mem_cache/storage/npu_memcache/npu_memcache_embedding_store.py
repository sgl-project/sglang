# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Global multimodal embedding cache backend for Ascend MemCache.

Implements the ``EmbeddingStore`` contract on top of
``memcache_hybrid.DistributedObjectStore`` (the same object store that the
HiCache ``npu_memcache`` KV backend uses), so that ``--enable-mm-global-cache``
with ``--mm-global-cache-backend npu_memcache`` works on NPU/Ascend.

Unlike the Mooncake backend this store does not rely on RDMA/mooncake master;
it connects to Ascend MemCache (MetaService/LocalService). Connection settings
are read from the ``SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH`` JSON file.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

from sglang.srt.environ import envs
from sglang.srt.mem_cache.embedding_store import EmbeddingStore
from sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store import (
    _MEMCACHE_CTRL_KEYS,
    _default_memcache_device_id,
)

logger = logging.getLogger(__name__)


def _resolve_device_id(ctrl_device_id: Any) -> int:
    """Resolve the memcache ``init(device_id)`` for the encoder process.

    The embedding store is constructed per encoder process and has no explicit
    rank/tp_rank, so a per-rank dict map cannot be resolved here. We fall back to
    ``torch.npu.current_device()`` for such cases.
    """
    if ctrl_device_id is None:
        return _default_memcache_device_id(None)
    if isinstance(ctrl_device_id, dict):
        logger.warning(
            "device_id is a per-rank dict but the rank is unknown in the encoder "
            "process; falling back to torch.npu.current_device()"
        )
        return _default_memcache_device_id(None)
    if isinstance(ctrl_device_id, str) and ctrl_device_id.strip().startswith("{"):
        logger.warning(
            "device_id is a JSON-string map; falling back to torch.npu.current_device()"
        )
        return _default_memcache_device_id(None)
    return int(ctrl_device_id)


class NpuMemcacheEmbeddingStore(EmbeddingStore):
    """``EmbeddingStore`` backed by Ascend MemCache object store."""

    def __init__(self, storage_config: Any = None):
        self.store = None
        try:
            from memcache_hybrid import DistributedObjectStore, LocalConfig
        except ImportError as e:
            raise ImportError(
                "NpuMemcacheEmbeddingStore requires `memcache_hybrid`. Install it "
                "with `pip install memcache_hybrid` and deploy MetaService/LocalService "
                "according to https://gitcode.com/Ascend/memcache"
            ) from e

        try:
            merged = self._load_config(storage_config)
            local_cfg = LocalConfig()
            unknown_fields = []
            for key, value in merged["local"].items():
                if hasattr(local_cfg, key):
                    setattr(local_cfg, key, value)
                else:
                    unknown_fields.append(key)
            if unknown_fields:
                logger.warning(
                    "Ignoring unknown Memcache LocalConfig keys for mm embedding "
                    "store: %s",
                    unknown_fields,
                )

            self.store = DistributedObjectStore()
            if self.store.setup(local_cfg) != 0:
                raise RuntimeError(
                    "memcache_hybrid.DistributedObjectStore.setup failed for mm "
                    "embedding store"
                )

            ctrl = merged["ctrl"]
            device_id = _resolve_device_id(ctrl.get("device_id"))
            init_bm = bool(ctrl.get("init_bm", True))
            if self.store.init(device_id, init_bm) != 0:
                raise RuntimeError(
                    "memcache_hybrid.DistributedObjectStore.init failed for mm "
                    "embedding store"
                )
            logger.info(
                "Ascend MemCache embedding store initialized (device_id=%s, init_bm=%s)",
                device_id,
                init_bm,
            )
        except Exception as exc:
            logger.error("Ascend MemCache embedding store init failed: %s", exc)
            raise

    def _load_config(self, storage_config: Any = None) -> dict:
        merged: dict = {}
        path = envs.SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH.get()
        if path:
            try:
                with open(path, encoding="utf-8") as fin:
                    merged.update(json.load(fin))
                logger.info("mm embedding store config loaded from %s", path)
            except Exception as exc:
                logger.warning(
                    "Failed to load mm embedding store memcache config from %s: %s",
                    path,
                    exc,
                )
        extra = getattr(storage_config, "extra_config", None) or {}
        merged.update(extra)
        local = {k: v for k, v in merged.items() if k not in _MEMCACHE_CTRL_KEYS}
        ctrl = {k: merged[k] for k in _MEMCACHE_CTRL_KEYS if k in merged}
        return {"local": local, "ctrl": ctrl}

    def get_key(self, mm_hash: str) -> str:
        return f"emb_{mm_hash}"

    def register_buffer(self, tensor: "torch.Tensor") -> None:
        if self.store is None:
            raise RuntimeError("Ascend MemCache embedding store is not initialized.")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        ret_code = self.store.register_buffer(ptr, size)
        if ret_code != 0:
            raise RuntimeError(
                f"Failed to register buffer to Ascend MemCache embedding store, "
                f"error code: {ret_code}"
            )

    def close(self) -> None:
        if self.store is None:
            return
        try:
            self.store.close()
        except Exception as e:
            logger.warning("Ascend MemCache embedding store.close failed: %s", e)
        self.store = None

    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]
        results = self.store.batch_get_into(keys, ptrs, sizes)
        # memcache_hybrid reports 0 on success.
        return [code == 0 for code in results]

    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]
        exists = self.store.batch_is_exist(keys)

        put_keys, put_ptrs, put_sizes, indices = [], [], [], []
        success_map = [True] * len(hashes)

        for i, status in enumerate(exists):
            if status != 1:
                put_keys.append(keys[i])
                put_ptrs.append(ptrs[i])
                put_sizes.append(sizes[i])
                indices.append(i)

        if put_keys:
            results = self.store.batch_put_from(put_keys, put_ptrs, put_sizes)
            for res, idx in zip(results, indices):
                success_map[idx] = res == 0
        return success_map

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]
        results = self.store.batch_is_exist(keys)
        return [code == 1 for code in results]

    def batch_get_into_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        """Flatten per-hash buffer runs into one single-buffer batch GET.

        memcache_hybrid only exposes a single-buffer API. Each hash may span
        multiple page-run buffers, so we flatten (key, ptr, size) triples, do one
        ``batch_get_into`` call, then regroup per hash: an entry is a hit only if
        every one of its buffers was fetched successfully.
        """
        flat_keys: List[str] = []
        flat_ptrs: List[int] = []
        flat_sizes: List[int] = []
        offsets: List[int] = []
        for h, ps, ss in zip(hashes, ptrs, sizes):
            key = self.get_key(h)
            offsets.append(len(flat_keys))
            for p, s in zip(ps, ss):
                flat_keys.append(key)
                flat_ptrs.append(p)
                flat_sizes.append(s)

        if not flat_keys:
            return [False] * len(hashes)

        results = self.store.batch_get_into(flat_keys, flat_ptrs, flat_sizes)
        out: List[bool] = []
        for i in range(len(hashes)):
            end = offsets[i + 1] if i + 1 < len(offsets) else len(results)
            chunk = results[offsets[i]:end]
            out.append(all(code == 0 for code in chunk) if chunk else False)
        return out

    def batch_put_from_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]

        # Skip hashes that already exist.
        exists = self.store.batch_is_exist(keys)
        flat_keys: List[str] = []
        flat_ptrs: List[int] = []
        flat_sizes: List[int] = []
        offsets: List[int] = []
        put_indices: List[int] = []
        for i, status in enumerate(exists):
            if status == 1:
                continue
            puts = ptrs[i]
            sizes_i = sizes[i]
            if not puts:
                continue
            offsets.append(len(flat_keys))
            put_indices.append(i)
            for p, s in zip(puts, sizes_i):
                flat_keys.append(keys[i])
                flat_ptrs.append(p)
                flat_sizes.append(s)

        success_map = [True] * len(hashes)
        if not flat_keys:
            return success_map

        results = self.store.batch_put_from(flat_keys, flat_ptrs, flat_sizes)
        for gi, hidx in enumerate(put_indices):
            end = offsets[gi + 1] if gi + 1 < len(offsets) else len(results)
            chunk = results[offsets[gi]:end]
            success_map[hidx] = all(code == 0 for code in chunk) if chunk else True
        return success_map