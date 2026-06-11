from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.utils import compute_node_hash_values

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


class HiSparseUnifiedRadixCache(UnifiedRadixCache):
    """Unified radix variant that indexes HiSparse host KV entries."""

    def __init__(self, params: CacheInitParams):
        self._cache_init_params = params
        self._hisparse_req_to_host_pool = None
        self._hisparse_req_to_token = None
        self._req_radix_node: dict[int, Optional[UnifiedTreeNode]] = {}
        self._req_radix_prefix_len: dict[int, int] = {}
        super().__init__(params)

    def _reset_full(self) -> None:
        super()._reset_full()
        self._req_radix_node.clear()
        self._req_radix_prefix_len.clear()

    def init_hisparse_radix_cache(
        self, host_pool, server_args: Optional[ServerArgs] = None
    ) -> bool:
        """Attach the HiSparse coordinator host pool to the unified tree."""
        self.components[BASE_COMPONENT_TYPE]._full_kv_pool_host = host_pool

        allocator = self.token_to_kv_pool_allocator
        kvcache = allocator.get_kvcache()
        compress_ratio = getattr(allocator, "compress_ratio", 1)
        can_load = compress_ratio == 1 and hasattr(
            host_pool, "load_to_device_per_layer"
        )
        if not can_load:
            logger.warning(
                "HiSparse unified radix cache is disabled for this pool "
                "(compress_ratio=%s, host_pool=%s).",
                compress_ratio,
                type(host_pool).__name__,
            )
            return False

        if (
            server_args is None
            or not server_args.enable_hierarchical_cache
            or not hasattr(kvcache, "index_k_with_scale_buffer")
        ):
            logger.warning("HiSparse unified radix cache requires DSA HiCache sidecar.")
            return False

        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            build_anchor_sidecar_stack,
        )
        from sglang.srt.mem_cache.memory_pool_host import DSAIndexerPoolHost

        storage_backend = server_args.hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = server_args.extra_metric_labels
        indexer_layout = (
            "page_first_direct"
            if server_args.hicache_io_backend == "direct"
            else "page_first"
        )
        self.host_pool_group, self.cache_controller = build_anchor_sidecar_stack(
            params=self._cache_init_params,
            server_args=server_args,
            kv_pool=kvcache,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping={i: i for i in range(kvcache.layer_num)},
            page_size=self.page_size,
            tp_group=self.tp_group,
            load_cache_event=self.load_cache_event,
            attn_cp_group=self.attn_cp_group,
            attn_tp_group=self.attn_tp_group,
            pp_group=self.pp_group,
            storage_backend=storage_backend,
            use_mla=True,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                kvcache,
                kv_host_pool,
                indexer_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            kv_host_pool=host_pool,
            prefetch_threshold=storage_prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=storage_extra_config,
            enable_storage_metrics=self._enable_metrics_flag,
        )
        self.full_kv_pool_host = host_pool
        self.register_sidecar_pool(
            SidecarPoolSpec(pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
        )
        kvcache.register_layer_transfer_counter(self.cache_controller.layer_done_counter)
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 0
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy
        if storage_backend is not None:
            self._apply_storage_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )
        return True

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        empty = self._empty_match_result.device_indices
        if (
            self.cache_controller is None
            or self.disable
            or len(params.key) == 0
        ):
            return self._empty_match_result

        host_indices, node, raw_match_len = self.host_match_prefix(
            params.key.token_ids, params.key.extra_key
        )
        host_hit_len = (raw_match_len // self.page_size) * self.page_size
        if host_hit_len <= 0 or node is None:
            return self._empty_match_result

        return MatchResult(
            device_indices=empty,
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=host_hit_len,
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if self.disable or node is None:
            return IncLockRefResult(delta=0)
        while node is not self.root_node:
            cd = node.component_data[BASE_COMPONENT_TYPE]
            if cd.host_value is not None:
                cd.host_lock_ref += 1
                self._update_evictable_leaf_sets(node)
            node = node.parent
        return IncLockRefResult(delta=0)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable or node is None:
            return DecLockRefResult(delta=0)
        while node is not self.root_node:
            cd = node.component_data[BASE_COMPONENT_TYPE]
            if cd.host_value is not None:
                assert cd.host_lock_ref > 0
                cd.host_lock_ref -= 1
                self._update_evictable_leaf_sets(node)
            node = node.parent
        return DecLockRefResult(delta=0)

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs) -> None:
        kv_committed_len = req.pop_committed_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def host_match_prefix(
        self, token_ids: list[int], extra_key: Optional[str] = None
    ) -> tuple[torch.Tensor, UnifiedTreeNode, int]:
        empty = torch.empty((0,), dtype=torch.int64, device="cpu")
        if len(token_ids) == 0:
            return empty, self.root_node, 0
        key = RadixKey(token_ids=token_ids, extra_key=extra_key).page_aligned(
            self.page_size
        )
        if len(key) == 0:
            return empty, self.root_node, 0

        _, best_match_node, _, _ = self._match_prefix_helper(key)
        host_indices = self._collect_host_indices(best_match_node)
        return host_indices, best_match_node, len(host_indices)

    def bind_hisparse_req_pools(self, req_to_host_pool, req_to_token) -> None:
        self._hisparse_req_to_host_pool = req_to_host_pool
        self._hisparse_req_to_token = req_to_token

    def match_and_lock_req_prefix(
        self,
        req_idx: int,
        token_ids: list[int],
        extra_key: Optional[str],
        max_len: int,
    ) -> tuple[torch.Tensor, int]:
        host_prefix, node, raw_match_len = self.host_match_prefix(token_ids, extra_key)
        prefix_len = min(raw_match_len, max_len)
        if prefix_len > 0:
            self.inc_lock_ref(node)
        self._req_radix_node[req_idx] = node
        self._req_radix_prefix_len[req_idx] = prefix_len
        return host_prefix, prefix_len

    def req_prefix_len(self, req_idx: int) -> int:
        return self._req_radix_prefix_len.get(req_idx, 0)

    def _indexer_pool_transfer(
        self,
        req_idx: int,
        start_pos: int,
        end_pos: int,
    ) -> Optional[PoolTransfer]:
        if (
            self._hisparse_req_to_host_pool is None
            or self._hisparse_req_to_token is None
        ):
            return None

        aligned_start = (
            (start_pos + self.page_size - 1) // self.page_size
        ) * self.page_size
        aligned_end = (end_pos // self.page_size) * self.page_size
        if aligned_end <= aligned_start:
            return None

        host_indices = self._hisparse_req_to_host_pool[
            req_idx, aligned_start:aligned_end
        ]
        if host_indices.numel() == 0:
            return None
        return PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=host_indices,
            device_indices=self._hisparse_req_to_token[
                req_idx, aligned_start:aligned_end
            ].to(dtype=torch.int64),
        )

    def backup_from_device_all_layer(
        self,
        mem_pool_device,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
        non_sparse_pool_offload_ranges=None,
        non_sparse_pool_offload_lens=None,
    ) -> None:
        if non_sparse_pool_offload_ranges or non_sparse_pool_offload_lens:
            pool_transfers = list(pool_transfers or [])
            ranges = list(non_sparse_pool_offload_ranges or [])
            ranges.extend(
                (req_idx, written_end - self.page_size, written_end)
                for req_idx, written_end in non_sparse_pool_offload_lens or []
                if written_end > 0 and written_end % self.page_size == 0
            )
            for req_idx, start_pos, end_pos in ranges:
                transfer = self._indexer_pool_transfer(req_idx, start_pos, end_pos)
                if transfer is not None:
                    pool_transfers.append(transfer)
            if len(pool_transfers) == 0:
                pool_transfers = None

        self.host_pool_group.backup_from_device_all_layer(
            mem_pool_device,
            host_indices,
            device_indices,
            io_backend=getattr(self.cache_controller, "io_backend", "kernel"),
            pool_transfers=pool_transfers,
        )

    def insert_req_host_indices(
        self,
        req_idx: int,
        token_ids: list[int],
        host_indices: torch.Tensor,
        extra_key: Optional[str],
        new_protected_len: Optional[int] = None,
        lock_new_node: bool = False,
        return_canonical_indices: bool = False,
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        empty = host_indices[:0]
        if len(token_ids) == 0:
            return 0, empty, None

        key = RadixKey(token_ids=token_ids, extra_key=extra_key).page_aligned(
            self.page_size
        )
        if len(key) == 0:
            return 0, empty, None

        value = host_indices[: len(key)].to(dtype=torch.int64, device="cpu")
        hash_value = []
        if self.enable_storage:
            hash_node = UnifiedTreeNode(self.tree_components)
            hash_node.parent = self.root_node
            hash_node.key = key
            hash_value = compute_node_hash_values(hash_node, self.page_size)

        insert_result = self._insert_helper_host(
            self.root_node, key, value, hash_value
        )
        prefix_len = insert_result.prefix_len
        if (
            prefix_len < insert_result.total_len
            and insert_result.inserted_host_node is not None
        ):
            self._inc_hit_count(insert_result.inserted_host_node)
            self.write_backup_storage(insert_result.inserted_host_node)

        protected_len = self.req_prefix_len(req_idx)
        duplicate_indices = (
            host_indices[protected_len:prefix_len]
            if prefix_len > protected_len
            else empty
        )
        old_node = (
            self._req_radix_node.get(req_idx)
            if new_protected_len is not None
            else None
        )
        canonical_indices = None
        new_node = None
        if (
            old_node is not None
            or lock_new_node
            or (return_canonical_indices and duplicate_indices.numel() > 0)
        ):
            canonical_indices_all, new_node, _ = self.host_match_prefix(
                token_ids, extra_key
            )
            if return_canonical_indices and duplicate_indices.numel() > 0:
                canonical_indices = canonical_indices_all[protected_len:prefix_len]

        if old_node is not None and old_node is not self.root_node:
            self.dec_lock_ref(old_node)
        if lock_new_node and new_node is not None and new_node is not self.root_node:
            self.inc_lock_ref(new_node)

        if new_protected_len is not None:
            if new_node is None:
                self._req_radix_node.pop(req_idx, None)
            else:
                self._req_radix_node[req_idx] = new_node
            self._req_radix_prefix_len[req_idx] = new_protected_len
        return prefix_len, duplicate_indices, canonical_indices

    def release_req_node(self, req_idx: int) -> None:
        node = self._req_radix_node.pop(req_idx, None)
        if node is not None and node is not self.root_node:
            self.dec_lock_ref(node)
        self._req_radix_prefix_len.pop(req_idx, None)

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        best_match_node = params.best_match_node
        host_hit_length = params.host_hit_length
        empty = self._empty_match_result.device_indices

        if (
            self.cache_controller is None
            or best_match_node is None
            or host_hit_length <= 0
        ):
            return empty, best_match_node

        host_indices = self._collect_host_indices(best_match_node)
        page_aligned_len = (host_hit_length // self.page_size) * self.page_size
        if page_aligned_len <= 0:
            return empty, best_match_node
        host_indices = host_indices[:page_aligned_len]

        allocator = self.token_to_kv_pool_allocator
        logical_alloc = getattr(allocator, "logical_attn_allocator", None)
        hisparse_alloc = getattr(allocator, "hisparse_attn_allocator", None)
        mapping = getattr(allocator, "full_to_hisparse_device_index_mapping", None)

        if logical_alloc is None or hisparse_alloc is None or mapping is None:
            return empty, best_match_node

        logical_indices = logical_alloc.alloc(page_aligned_len)
        if logical_indices is None:
            logger.warning(
                "HiSparse unified radix load-back failed to allocate %d logical slots",
                page_aligned_len,
            )
            return empty, best_match_node

        hisparse_indices = hisparse_alloc.alloc(page_aligned_len)
        if hisparse_indices is None:
            logical_alloc.free(logical_indices)
            logger.warning(
                "HiSparse unified radix load-back failed to allocate %d device slots",
                page_aligned_len,
            )
            return empty, best_match_node

        mapping[logical_indices] = hisparse_indices

        self.cache_controller.load_queue.append(
            CacheOperation(
                host_indices=host_indices,
                device_indices=hisparse_indices,
                node_id=best_match_node.id,
                pool_transfers=[
                    PoolTransfer(
                        name=PoolName.INDEXER,
                        host_indices=host_indices,
                        device_indices=logical_indices,
                    )
                ],
            )
        )
        self.ongoing_load_back[best_match_node.id] = (
            best_match_node,
            self.inc_lock_ref(best_match_node).to_dec_params(),
        )
        return logical_indices, best_match_node

    def _collect_host_indices(self, node: UnifiedTreeNode) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        cur = node
        while cur is not self.root_node:
            host_value = cur.component_data[BASE_COMPONENT_TYPE].host_value
            if host_value is None:
                break
            chunks.append(host_value)
            cur = cur.parent
        if not chunks:
            return torch.empty((0,), dtype=torch.int64, device="cpu")
        chunks.reverse()
        return torch.cat(chunks)
