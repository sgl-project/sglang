# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.tensorcast_store.client import (
    DefaultTensorcastPageClient,
    TensorcastBatchExistsResult,
    TensorcastBatchTransferResult,
    TensorcastPageClient,
)
from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)
from sglang.srt.mem_cache.storage.tensorcast_store.host_shared_slot_state import (
    HostSharedPageSlotManager,
    HostSharedPageSlotToken,
)

logger = logging.getLogger(__name__)


@dataclass
class _TensorcastPublicationStats:
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    batch_calls: int = 0
    page_calls: int = 0
    duplicate_pages: int = 0
    failed_pages: int = 0
    batch_total_s: float = 0.0
    pack_total_s: float = 0.0
    stage_copy_total_s: float = 0.0
    rpc_total_s: float = 0.0

    def record_batch(
        self,
        *,
        pages: int,
        duplicate_pages: int,
        failed_pages: int,
        batch_elapsed_s: float,
        pack_elapsed_s: float,
        stage_copy_elapsed_s: float,
        rpc_elapsed_s: float,
    ) -> dict[str, float | int]:
        with self.lock:
            self.batch_calls += 1
            self.page_calls += pages
            self.duplicate_pages += duplicate_pages
            self.failed_pages += failed_pages
            self.batch_total_s += batch_elapsed_s
            self.pack_total_s += pack_elapsed_s
            self.stage_copy_total_s += stage_copy_elapsed_s
            self.rpc_total_s += rpc_elapsed_s
            return {
                "batch_calls": self.batch_calls,
                "page_calls": self.page_calls,
                "duplicate_pages": self.duplicate_pages,
                "failed_pages": self.failed_pages,
                "batch_total_ms": self.batch_total_s * 1000.0,
                "pack_total_ms": self.pack_total_s * 1000.0,
                "stage_copy_total_ms": self.stage_copy_total_s * 1000.0,
                "rpc_total_ms": self.rpc_total_s * 1000.0,
            }


def _sanitize_component(value: str) -> str:
    sanitized = str(value).strip().replace("/", "_")
    sanitized = sanitized.replace(":", "_").replace(" ", "_")
    return sanitized or "default"


class TensorcastStore(HiCacheStorage):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: HostKVCache,
        page_client: TensorcastPageClient | None = None,
    ) -> None:
        self._storage_config = storage_config
        self._tensorcast_config = TensorcastHiCacheConfig.from_storage_config(
            storage_config
        )
        self.is_mla_backend = storage_config.is_mla_model
        self.local_rank = storage_config.tp_rank
        self.pp_rank = storage_config.pp_rank
        self.pp_size = storage_config.pp_size
        self.layout = mem_pool_host.layout
        self.page_size = mem_pool_host.page_size
        self.dtype = mem_pool_host.dtype
        self.tp_size = storage_config.tp_size
        self._rank_suffix = self._build_rank_suffix()
        self._layout_id = self._build_layout_id()
        self._page_client = page_client or DefaultTensorcastPageClient(
            self._tensorcast_config,
            layout_id=self._layout_id,
            engine_key_prefix=self._rank_suffix,
        )
        self._host_slot_manager: HostSharedPageSlotManager | None = None
        self._publication_stats = _TensorcastPublicationStats()
        self.register_mem_pool_host(mem_pool_host)
        logger.info(
            "Initialized Tensorcast HiCache backend: daemon=%s namespace=%s rank_suffix=%s layout_id=%s",
            self._tensorcast_config.daemon_address,
            self._tensorcast_config.namespace,
            self._rank_suffix,
            self._layout_id,
        )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self._ensure_host_slot_manager(mem_pool_host)
        host_region_binding = mem_pool_host.host_region_binding
        if self._tensorcast_config.host_allocator_enabled:
            if mem_pool_host.layout != "page_blob_direct":
                raise ValueError(
                    "TensorCast allocator-backed host residency requires mem_pool_host.layout=page_blob_direct"
                )
            if host_region_binding is None:
                raise ValueError(
                    "TensorCast allocator-backed host residency requires a live host_region_binding"
                )
            slot_bytes = int(mem_pool_host.size_per_token) * int(
                mem_pool_host.page_size
            )
            if slot_bytes <= 0:
                raise ValueError(
                    "TensorCast allocator-backed host residency requires positive slot_bytes"
                )
            self._page_client.activate_stable_local_backing(
                str(host_region_binding.region_id),
                slot_bytes=slot_bytes,
            )
            logger.info(
                "Activated Tensorcast stable local backing region=%s slot_bytes=%d",
                host_region_binding.region_id,
                slot_bytes,
            )
        super().register_mem_pool_host(mem_pool_host)

    def _ensure_host_slot_manager(
        self,
        mem_pool_host: HostKVCache,
    ) -> HostSharedPageSlotManager:
        manager = self._host_slot_manager
        if manager is None:
            manager = HostSharedPageSlotManager(
                page_size=mem_pool_host.page_size,
                page_num=mem_pool_host.page_num,
            )
            self._host_slot_manager = manager
        elif (
            manager.page_size != mem_pool_host.page_size
            or manager.page_num != mem_pool_host.page_num
        ):
            raise ValueError(
                "Tensorcast host slot manager already attached to incompatible host pool"
            )
        mem_pool_host.attach_host_page_slot_lifecycle_manager(manager)
        return manager

    def _require_host_slot_manager(self) -> HostSharedPageSlotManager:
        if self._host_slot_manager is None:
            raise RuntimeError("Tensorcast host slot manager is not attached")
        return self._host_slot_manager

    def _build_rank_suffix(self) -> str:
        if self.is_mla_backend:
            return f"pp{self.pp_rank}of{self.pp_size}"
        if self.pp_size > 1:
            return (
                f"tp{self.local_rank}of{self.tp_size}_"
                f"pp{self.pp_rank}of{self.pp_size}"
            )
        return f"tp{self.local_rank}of{self.tp_size}"

    def _build_layout_id(self) -> str:
        attention_family = "mla" if self.is_mla_backend else "mha"
        return "_".join(
            [
                "sglang_kv_page",
                _sanitize_component(self._tensorcast_config.page_layout_version),
                _sanitize_component(self.layout),
                _sanitize_component(str(self.dtype)),
                f"ps{self.page_size}",
                attention_family,
            ]
        )

    def _page_start_indices(
        self,
        host_indices: torch.Tensor,
        expected_pages: int,
    ) -> list[int]:
        if host_indices.numel() != expected_pages * self.page_size:
            raise ValueError(
                "host_indices length must equal number of pages multiplied by page_size"
            )
        return [
            int(host_indices[i * self.page_size].item()) for i in range(expected_pages)
        ]

    def _host_page_views(self, page_starts: list[int]) -> list[torch.Tensor]:
        return [
            self.mem_pool_host.get_data_page(index, flat=True) for index in page_starts
        ]

    def _slot_tokens_enabled(self, extra_info: HiCacheStorageExtraInfo | None) -> bool:
        if extra_info is None or extra_info.extra_info is None:
            return False
        enabled = extra_info.extra_info.get("tensorcast_use_host_slot_tokens", False)
        return bool(enabled)

    def _slot_tokens_for_page_starts_if_enabled(
        self,
        page_starts: list[int],
        extra_info: HiCacheStorageExtraInfo | None,
    ) -> list[HostSharedPageSlotToken] | None:
        if not self._slot_tokens_enabled(extra_info):
            return None
        return list(
            self._require_host_slot_manager().slot_tokens_for_page_starts(page_starts)
        )

    def _allocator_backed_direct_put_enabled(self) -> bool:
        return (
            self._tensorcast_config.host_allocator_enabled
            and self.mem_pool_host.host_region_binding is not None
        )

    def _allocator_backed_direct_get_enabled(self) -> bool:
        return self._allocator_backed_direct_put_enabled()

    def _success_prefix_count(self, success_mask: tuple[bool, ...]) -> int:
        prefix_success = 0
        for success in success_mask:
            if not success:
                break
            prefix_success += 1
        return prefix_success

    def batch_exists(
        self,
        keys: list[str],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> int:
        _ = extra_info
        result: TensorcastBatchExistsResult = self._page_client.batch_exists(keys)
        prefix_success = 0
        for exists in result.existence_mask:
            if not exists:
                break
            prefix_success += 1
        first_key = keys[0] if keys else ""
        first_artifact_id = (
            self._page_client.artifact_id_for(first_key) if first_key else ""
        )
        logger.debug(
            "Tensorcast batch_exists pages=%d prefix_success=%d rpc_elapsed_ms=%.2f first_key=%s first_artifact_id=%s",
            len(keys),
            prefix_success,
            result.rpc_elapsed_s * 1000.0,
            first_key,
            first_artifact_id,
        )
        return prefix_success

    def batch_get_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        page_starts = self._page_start_indices(host_indices, len(keys))
        targets = self._host_page_views(page_starts)
        direct_get_enabled = self._allocator_backed_direct_get_enabled()
        host_slot_manager = self._require_host_slot_manager()
        slot_tokens = (
            list(host_slot_manager.reserve_page_slots(page_starts, keys))
            if direct_get_enabled
            else self._slot_tokens_for_page_starts_if_enabled(page_starts, extra_info)
        )
        target_region_binding = (
            self.mem_pool_host.host_region_binding if direct_get_enabled else None
        )
        if direct_get_enabled:
            host_slot_manager.mark_page_get_inflight(slot_tokens)
        try:
            result = self._page_client.batch_get_into(
                keys,
                targets,
                slot_tokens=slot_tokens,
                target_region_binding=target_region_binding,
            )
        except Exception:
            if direct_get_enabled:
                with suppress(Exception):
                    host_slot_manager.fail_page_get(slot_tokens)
            raise
        if direct_get_enabled:
            prefix_success = self._success_prefix_count(result.success_mask)
            if prefix_success > 0:
                host_slot_manager.commit_page_get_success(
                    slot_tokens[:prefix_success],
                    keys[:prefix_success],
                )
            if prefix_success < len(slot_tokens):
                host_slot_manager.fail_page_get(slot_tokens[prefix_success:])
        first_key = keys[0] if keys else ""
        first_artifact_id = (
            self._page_client.artifact_id_for(first_key) if first_key else ""
        )
        logger.debug(
            "Tensorcast batch_get_v1 pages=%d succeeded=%d pack_elapsed_ms=%.2f rpc_elapsed_ms=%.2f host_fill_ms=%.2f operation_id=%s first_key=%s first_artifact_id=%s",
            len(keys),
            sum(1 for item in result.success_mask if item),
            result.pack_elapsed_s * 1000.0,
            result.rpc_elapsed_s * 1000.0,
            result.host_fill_elapsed_s * 1000.0,
            result.operation_id,
            first_key,
            first_artifact_id,
        )
        return list(result.success_mask)

    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        page_starts = self._page_start_indices(host_indices, len(keys))
        pages = self._host_page_views(page_starts)
        direct_put_enabled = self._allocator_backed_direct_put_enabled()
        slot_tokens = (
            list(
                self._require_host_slot_manager().slot_tokens_for_page_starts(
                    page_starts
                )
            )
            if direct_put_enabled
            else self._slot_tokens_for_page_starts_if_enabled(page_starts, extra_info)
        )
        source_region_binding = (
            self.mem_pool_host.host_region_binding if direct_put_enabled else None
        )
        batch_started_at = time.perf_counter()
        result: TensorcastBatchTransferResult = self._page_client.batch_put(
            keys,
            pages,
            slot_tokens=slot_tokens,
            source_region_binding=source_region_binding,
        )
        batch_elapsed_s = time.perf_counter() - batch_started_at
        succeeded = sum(1 for item in result.success_mask if item)
        failed_pages = len(keys) - succeeded
        first_key = keys[0] if keys else ""
        first_artifact_id = (
            self._page_client.artifact_id_for(first_key) if first_key else ""
        )
        cumulative = self._publication_stats.record_batch(
            pages=len(keys),
            duplicate_pages=result.adopted_duplicate_count,
            failed_pages=failed_pages,
            batch_elapsed_s=batch_elapsed_s,
            pack_elapsed_s=result.pack_elapsed_s,
            stage_copy_elapsed_s=result.stage_copy_elapsed_s,
            rpc_elapsed_s=result.rpc_elapsed_s,
        )
        logger.debug(
            "Tensorcast batch_set_v1 pages=%d succeeded=%d duplicates=%d failed=%d batch_elapsed_ms=%.2f pack_elapsed_ms=%.2f stage_copy_ms=%.2f rpc_elapsed_ms=%.2f first_key=%s first_artifact_id=%s cumulative_pages=%d cumulative_duplicates=%d cumulative_failed=%d cumulative_batch_ms=%.2f cumulative_pack_ms=%.2f cumulative_stage_copy_ms=%.2f cumulative_rpc_ms=%.2f",
            len(keys),
            succeeded,
            result.adopted_duplicate_count,
            failed_pages,
            batch_elapsed_s * 1000.0,
            result.pack_elapsed_s * 1000.0,
            result.stage_copy_elapsed_s * 1000.0,
            result.rpc_elapsed_s * 1000.0,
            first_key,
            first_artifact_id,
            cumulative["page_calls"],
            cumulative["duplicate_pages"],
            cumulative["failed_pages"],
            cumulative["batch_total_ms"],
            cumulative["pack_total_ms"],
            cumulative["stage_copy_total_ms"],
            cumulative["rpc_total_ms"],
        )
        return list(result.success_mask)

    def get(
        self,
        key: str,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> torch.Tensor | None:
        _ = target_sizes
        cpu_target = (
            torch.empty_like(target_location, device="cpu")
            if isinstance(target_location, torch.Tensor)
            else self.mem_pool_host.get_dummy_flat_data_page()
        )
        result = self._page_client.batch_get_into([key], [cpu_target])
        if not result.success_mask or not result.success_mask[0]:
            return None
        if isinstance(target_location, torch.Tensor):
            target_location.copy_(cpu_target.reshape_as(target_location))
            return target_location
        return cpu_target

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> list[torch.Tensor | None] | int:
        _ = target_sizes
        if target_locations is None:
            outputs: list[torch.Tensor | None] = []
            for key in keys:
                outputs.append(self.get(key))
            return outputs
        outputs = []
        for key, target in zip(keys, target_locations, strict=False):
            outputs.append(self.get(key, target_location=target))
        return outputs

    def set(
        self,
        key: str,
        value: Any | None = None,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        _ = target_sizes
        tensor = value if isinstance(value, torch.Tensor) else target_location
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("TensorcastStore.set requires a tensor value")
        cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.cpu()
        result = self._page_client.batch_put([key], [cpu_tensor.reshape(-1)])
        return bool(result.success_mask and result.success_mask[0])

    def batch_set(
        self,
        keys: list[str],
        values: Any | None = None,
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        _ = target_sizes
        tensors = values if values is not None else target_locations
        if tensors is None:
            raise ValueError("TensorcastStore.batch_set requires tensors")
        results = [
            self.set(key, value=tensor)
            for key, tensor in zip(keys, tensors, strict=False)
        ]
        return all(results)

    def exists(self, key: str) -> bool:
        result = self._page_client.batch_exists([key])
        return bool(result.existence_mask and result.existence_mask[0])
