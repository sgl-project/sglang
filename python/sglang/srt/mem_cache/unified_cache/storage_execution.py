# SPDX-License-Identifier: Apache-2.0
"""Declared native storage execution profiles; I/O stays in HiCache workers.

Adapters bind a concrete backend instance, physical layout and protection domain.
Descriptions contain keys, never pointers or inferred residency. No network calls
run on the cache owner thread.
"""

from __future__ import annotations

import hashlib
import os
import socket
from typing import Protocol

import msgspec
import torch

from sglang.srt.managers.kv_control.contracts import ControlError, Location
from sglang.srt.mem_cache.hicache_storage import HiCacheFile


class StorageExecutionAdapter(Protocol):
    backend: str
    profile: str
    backend_namespace: str
    store: object
    copy_paths: tuple[tuple[str, str], ...]
    move_paths: tuple[tuple[str, str], ...]

    def describe_pages(self, hashes: list[str]) -> list[dict]: ...

    def location(self) -> Location: ...


class _Adapter:
    copy_paths = (
        ("gpu", "storage"),
        ("host", "storage"),
        ("storage", "host"),
    )
    move_paths = (("gpu", "storage"), ("host", "storage"))

    def __init__(self, store, pool, domain):
        self.store, self.pool = store, pool
        # Pool capacity and worker epoch are intentionally absent. Byte layout
        # and key suffixes must agree across independent compatible engines.
        layout = (
            type(pool).__name__,
            pool.layout,
            str(pool.dtype),
            pool.page_size,
            pool.layer_num,
            pool.start_layer,
            pool.end_layer,
            pool.size_per_token,
            getattr(pool, "head_num", None),
            getattr(pool, "head_dim", None),
        )
        self.backend_namespace = (
            "sglang.storage.v1:"
            + hashlib.sha256(
                msgspec.json.encode((self.backend, domain, layout))
            ).hexdigest()
        )

    def location(self):
        return Location(
            tier="storage",
            backend=self.backend,
            backend_namespace=self.backend_namespace,
        )


class FileExecutionAdapter(_Adapter):
    backend = "file"
    profile = "native_full_file_v1"

    def __init__(self, store, pool):
        super().__init__(
            store,
            pool,
            (
                socket.gethostname(),
                os.path.realpath(store.file_path),
                store._get_component_keys(""),
            ),
        )
        # Legacy positive metadata can outlive another writer's eviction.
        # Destructive host moves require fresh backend existence checks;
        # unified file chunks already bypass that cache and validate sizes.
        if store.metadata_cache is not None and store._unified_suffixes is None:
            self.move_paths = (("gpu", "storage"),)

    def describe_pages(self, hashes):
        return [
            {
                "keys": [
                    os.path.join(os.path.realpath(self.store.file_path), key + ".bin")
                    for key in self.store._get_component_keys(h)
                ]
            }
            for h in hashes
        ]


class MooncakeExecutionAdapter(_Adapter):
    backend = "mooncake"
    profile = "tp1_full_mha_page_first_direct_v1"

    def __init__(self, store, pool):
        config = store.storage_config
        if (
            config is None
            or any(
                v != 1
                for v in (
                    config.tp_size,
                    config.pp_size,
                    config.attn_cp_size,
                    config.dcp_size,
                )
            )
            or store.is_mla_backend
            or store.should_split_heads
            or config.unified_suffixes is not None
            or pool.layout != "page_first_direct"
        ):
            raise ControlError(
                "unsupported",
                "Mooncake control requires TP1 FULL MHA page_first_direct",
            )
        # Use exactly the writer's tag and K/V/rank expansion. Probe only buffer
        # metadata, without allocating, reading or writing the first host page.
        keys, _, sizes = store._batch_preprocess(
            store._tag_keys([""]), torch.arange(pool.page_size, dtype=torch.int64)
        )
        if len(keys) != 2 or any(type(n) is not int or n <= 0 for n in sizes):
            raise ControlError(
                "unsupported", "Unsupported Mooncake physical page layout"
            )
        self.object_sizes = sizes
        c = store.config
        super().__init__(
            store,
            pool,
            (c.master_server_address, c.metadata_server, c.tenant_id, keys, sizes),
        )

    def describe_pages(self, hashes):
        # Reuse a metadata-only page for each key; no host allocation/residency
        # is asserted and the returned description never exposes buffer addresses.
        indices = torch.arange(self.pool.page_size, dtype=torch.int64)
        return [
            {
                "keys": self.store._batch_preprocess(
                    self.store._tag_keys([h]), indices
                )[0],
                "object_bytes": list(self.object_sizes),
            }
            for h in hashes
        ]


# Explicit registration: an arbitrary HiCache backend does not acquire native
# control support merely by implementing generic read/write methods.
_ADAPTERS = {"file": FileExecutionAdapter, "mooncake": MooncakeExecutionAdapter}


def storage_execution_adapter(cache) -> tuple[StorageExecutionAdapter | None, str]:
    controller = cache.cache_controller
    backend = controller.storage_backend_type
    factory = _ADAPTERS.get(backend)
    if factory is None:
        return None, "Storage backend has no registered control adapter"
    store = controller.storage_backend
    if backend == "file":
        valid = isinstance(store, HiCacheFile)
    else:
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        valid = isinstance(store, MooncakeStore)
    if not valid:
        return None, "Storage backend instance does not match its adapter"
    try:
        return factory(store, cache.host_pool_group.anchor_entry.host_pool), ""
    except ControlError as error:
        return None, str(error)
