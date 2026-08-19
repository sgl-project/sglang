from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import replace
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.storage_hit_count = 0
        self.start_time = time.monotonic()
        super().__init__(
            None,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
        host_memory_mode: str = "cache",
    ):
        startup_storage_backend = storage_backend
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=host_memory_mode,
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    def register_host_pool_entry(self, entry: PoolEntry) -> None:
        if not isinstance(self.mem_pool_host, HostPoolGroup):
            raise TypeError("Dynamic HiCache sidecars require HostPoolGroup.")
        self.mem_pool_host.add_entry(entry)
        if not entry.is_primary_index_anchor:
            self.extra_host_mem_release_queues.setdefault(entry.name, Queue())
        if self.enable_storage and self.storage_backend is not None:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def _move_op_indices(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        return self.move_hybrid_indices(op)

    def _move_write_operation(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_group = self.mem_pool_host
        if self.io_backend != "kernel" or host_group.layout != "page_first":
            return self.move_hybrid_indices(op)
        if not getattr(host_group, "supports_per_pool_backup_indices", False):
            if not getattr(host_group, "can_use_write_back_jit", False):
                return self.move_hybrid_indices(op)
            return op.host_indices, op.device_indices, op.pool_transfers

        def move_for_pool(host_pool, host_indices, device_indices):
            if getattr(host_pool, "can_use_write_back_jit", False):
                if host_indices.is_cuda:
                    host_indices = host_indices.cpu()
                return host_indices, device_indices
            return self.move_indices(host_indices, device_indices)

        host_indices, device_indices = move_for_pool(
            host_group.anchor_entry.host_pool,
            op.host_indices,
            op.device_indices,
        )
        pool_transfers = []
        for transfer in op.pool_transfers or []:
            entry = host_group.entry_map[transfer.name]
            transfer_host_indices, transfer_device_indices = move_for_pool(
                entry.host_pool,
                transfer.host_indices,
                transfer.device_indices,
            )
            pool_transfers.append(
                replace(
                    transfer,
                    host_indices=transfer_host_indices,
                    device_indices=transfer_device_indices,
                )
            )
        return host_indices, device_indices, pool_transfers or None

    def _l2_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        anchor = self.mem_pool_host.anchor_entry
        transfers = []
        if host_indices.numel() > 0:
            transfers.append(
                L2Transfer(
                    host_pool=anchor.host_pool,
                    device_pool=anchor.device_pool,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_mapper=anchor.layer_mapper,
                )
            )
        for pool_transfer in pool_transfers or []:
            if (
                pool_transfer.host_indices is None
                or pool_transfer.device_indices is None
            ):
                raise ValueError(f"Unresolved L2 transfer for {pool_transfer.name}.")
            entry = self.mem_pool_host.entry_map[pool_transfer.name]
            transfers.append(
                L2Transfer(
                    host_pool=entry.host_pool,
                    device_pool=entry.device_pool,
                    host_indices=pool_transfer.host_indices,
                    device_indices=pool_transfer.device_indices,
                    layer_mapper=entry.layer_mapper,
                )
            )
        if self.has_draft and host_indices.numel() > 0:
            transfers.append(
                L2Transfer(
                    host_pool=self.mem_pool_host_draft,
                    device_pool=self.mem_pool_device_draft,
                    host_indices=host_indices,
                    device_indices=device_indices,
                )
            )
        return transfers

    def _l2_load_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        transfers = self._l2_transfers(host_indices, device_indices, pool_transfers)
        if getattr(self, "has_mtp_draft", False):
            target_transfers = list(transfers)
            for depth, draft_device_pool in enumerate(self.mtp_draft_device_pools):
                for transfer in target_transfers:
                    if transfer.layer_mapper is None:
                        continue
                    draft_host_layer = transfer.layer_mapper(self.layer_num + depth)
                    if draft_host_layer is None:
                        continue

                    def draft_layer_mapper(
                        layer_id: int,
                        *,
                        expected_layer_id: int = depth,
                        host_layer_id: int = draft_host_layer,
                    ) -> Optional[int]:
                        if layer_id == expected_layer_id:
                            return host_layer_id
                        return None

                    transfers.append(
                        L2Transfer(
                            host_pool=transfer.host_pool,
                            device_pool=draft_device_pool,
                            host_indices=transfer.host_indices,
                            device_indices=transfer.device_indices,
                            layer_mapper=draft_layer_mapper,
                            is_draft=True,
                        )
                    )
        return transfers

    def _num_tokens_by_pool(self, op: CacheOperation) -> dict[str, int]:
        """Per-pool token counts for a merged transfer op (anchor + extra
        pools), shared by D->H write and H->D load acks; sidecar transfers
        reusing another pool's indices are excluded."""
        counts = {self.mem_pool_host.anchor_entry.name.value: len(op.device_indices)}
        for transfer in op.pool_transfers or []:
            if transfer.indices_from_pool is not None or transfer.host_indices is None:
                continue
            name = transfer.name.value
            counts[name] = counts.get(name, 0) + len(transfer.host_indices)
        return counts

    def _transfer_num_bytes(self, op: CacheOperation) -> int:
        """Total bytes moved by a merged transfer op across all pools,
        including draft piggyback and sidecar transfers riding another
        pool's indices (both excluded from the per-pool token counts)."""
        kv_tokens = len(op.device_indices)
        num_bytes = kv_tokens * self.mem_pool_host.anchor_entry.host_pool.size_per_token
        if self.has_draft:
            num_bytes += kv_tokens * self.mem_pool_host_draft.size_per_token
        # Slot counts of the pools sidecars can ride on.
        source_len = {self.mem_pool_host.anchor_entry.name: kv_tokens}
        for t in op.pool_transfers or []:
            if t.indices_from_pool is None and t.host_indices is not None:
                source_len[t.name] = len(t.host_indices)
        for t in op.pool_transfers or []:
            entry = self.mem_pool_host.entry_map.get(t.name)
            if entry is None:
                continue
            if t.indices_from_pool is not None:
                num_slots = source_len.get(t.indices_from_pool, 0)
            else:
                num_slots = len(t.host_indices) if t.host_indices is not None else 0
            num_bytes += num_slots * entry.host_pool.size_per_token
        return num_bytes

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None

        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def prefetch(
        self,
        request_id: str,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        hash_value = self.get_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.page_size
        )
        operation.all_hash_values = hash_value

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        # KV pools first — determines actual completed page count
        super()._page_transfer(operation)

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        kv_completed_pages = operation.completed_tokens // self.page_size
        if (
            operation.pool_transfers
            and not operation.is_terminated()
            and kv_completed_pages == len(operation.hash_value)
        ):
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        # MLA KV is replicated across TP ranks and should still be written only
        # by TP0. Rank-sharded sidecars still need every TP rank.
        backup_transfers = [
            transfer
            for transfer in operation.pool_transfers or []
            if self.should_backup(transfer)
        ]

        if backup_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(backup_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        if not self.backup_skip:
            super()._page_backup(operation)
        else:
            sidecar_ok = bool(backup_transfers)
            if sidecar_ok:
                for transfer in backup_transfers:
                    result = results.get(transfer.name)
                    if result is None:
                        result = results.get(transfer.name.value)
                    expected = len(transfer.keys or [])
                    if expected == 0 and transfer.host_indices is not None:
                        expected = int(transfer.host_indices.numel())
                    if (
                        not isinstance(result, (list, tuple))
                        or len(result) != expected
                        or not all(bool(ok) for ok in result)
                    ):
                        sidecar_ok = False
                        break
            operation.completed_tokens = (
                len(operation.hash_value) * self.page_size if sidecar_ok else 0
            )

    def should_backup(self, transfer: PoolTransfer) -> bool:
        if not self.backup_skip:
            return True

        # Kimi-K3 Mamba/KDA state is TP-sharded even when the primary MLA KV
        # pool is replicated.
        if transfer.name == PoolName.MAMBA:
            return True

        # Mooncake gives MHA draft and draft-SWA objects rank-specific keys.
        # MLA/DeepSeek-V4 draft pools remain TP0-only.
        if self.storage_backend_type == "mooncake" and transfer.name in (
            PoolName.DRAFT,
            PoolName.DRAFT_SWA,
        ):
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            return entry is not None and isinstance(
                entry.host_pool, MHATokenToKVPoolHost
            )

        return False

    def backup_thread_func(self):
        """Back up rank-sharded sidecars on every TP rank.

        The base implementation skips the entire operation on non-zero MLA TP
        ranks. That optimization is valid for replicated MLA KV, but not for
        hybrid rank-sharded pools such as Kimi-K3 Mamba state.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_backup(operation)
                self.ack_backup_queue.put(operation)
            except Empty:
                continue

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]
            if transfer.host_indices is None:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            pool_page_size = (
                entry.host_pool.page_size if entry is not None else self.page_size
            )
            needed = len(transfer.keys) * pool_page_size
            if transfer.host_indices.numel() > needed:
                # The hit undershot the pre-allocated window buffer. Backends
                # fetch keys zipped against the buffer head, so shrink the
                # transfer to match and release the tail now — otherwise the
                # length mismatch makes batch_get_v2 fetch nothing and the
                # whole window is silently lost downstream.
                self.append_host_mem_release(
                    extra_pools=[
                        PoolTransfer(
                            name=transfer.name,
                            host_indices=transfer.host_indices[needed:],
                        )
                    ]
                )
                transfer.host_indices = transfer.host_indices[:needed]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Auto-alloc host or device indices for PoolTransfers where they are None."""
        if not extra_pools:
            return None
        # (pool, free_fn, indices) for atomic rollback on failure.
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if alloc_host:
                if pool.host_indices is not None or pool.device_indices is None:
                    continue
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                if pool.device_indices is not None or pool.host_indices is None:
                    continue
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                evict_fn(size)
                indices = alloc_fn(size)
            if indices is None:
                # Atomic rollback: free everything we successfully allocated.
                rollback_allocated()
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools
