from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

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


class HiSparseHostInsertResult(NamedTuple):
    prefix_len: int
    node: Optional[UnifiedTreeNode]
    duplicate_host_indices: torch.Tensor
    canonical_host_indices: Optional[torch.Tensor]


class HiSparseUnifiedRadixCache(UnifiedRadixCache):
    """Unified radix variant that indexes HiSparse host KV entries."""

    def __init__(self, params: CacheInitParams):
        self.hisparse_mode = True
        self.hisparse_host_pool = None
        self._hisparse_load_back_supported = False
        self._last_match_host_indices: dict[int, torch.Tensor] = {}
        super().__init__(params)

    def _reset_full(self) -> None:
        super()._reset_full()
        self._last_match_host_indices.clear()

    def enable_hisparse_mode(self) -> None:
        self.hisparse_mode = True

    def init_hisparse_radix_cache(
        self, host_pool, server_args: Optional[ServerArgs] = None
    ) -> bool:
        """Attach the HiSparse coordinator host pool to the unified tree."""
        self.enable_hisparse_mode()
        self.hisparse_host_pool = host_pool
        self.components[BASE_COMPONENT_TYPE]._full_kv_pool_host = host_pool

        allocator = self.token_to_kv_pool_allocator
        kvcache = allocator.get_kvcache()
        compress_ratio = getattr(allocator, "compress_ratio", 1)
        can_load = compress_ratio == 1 and hasattr(
            host_pool, "load_to_device_per_layer"
        )
        self._hisparse_load_back_supported = can_load
        if not can_load:
            logger.warning(
                "HiSparse unified radix cache is disabled for this pool "
                "(compress_ratio=%s, host_pool=%s).",
                compress_ratio,
                type(host_pool).__name__,
            )
            return False

        if (
            server_args is not None
            and server_args.enable_hierarchical_cache
            and hasattr(kvcache, "index_k_with_scale_buffer")
        ):
            from sglang.srt.mem_cache.memory_pool_host import (
                DSAIndexerPoolHost,
                HostPoolGroup,
                PoolEntry,
            )

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
            layer_num = kvcache.layer_num

            def layer_mapper(layer_id):
                return layer_id

            indexer_layout = (
                "page_first_direct"
                if server_args.hicache_io_backend == "direct"
                else "page_first"
            )
            indexer_host_pool = DSAIndexerPoolHost(
                kvcache,
                host_pool,
                indexer_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
            host_pool_group = HostPoolGroup(
                [
                    PoolEntry(
                        name=PoolName.KV,
                        host_pool=host_pool,
                        device_pool=kvcache,
                        layer_mapper=layer_mapper,
                        is_primary_index_anchor=True,
                    ),
                    PoolEntry(
                        name=PoolName.INDEXER,
                        host_pool=indexer_host_pool,
                        device_pool=kvcache,
                        layer_mapper=layer_mapper,
                    ),
                ]
            )
            self.host_pool_group = host_pool_group
            self.cache_controller = HybridCacheController(
                self.token_to_kv_pool_allocator,
                host_pool_group,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                pp_group=self.pp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=storage_extra_config,
                transfer_layer_num=layer_num,
                enable_storage_metrics=self._enable_metrics_flag,
            )
            self.full_kv_pool_host = host_pool
            self.register_sidecar_pool(
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                )
            )
            kvcache.register_layer_transfer_counter(
                self.cache_controller.layer_done_counter
            )
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
            logger.info(
                "HiSparse unified radix cache enabled with DSA HiCache sidecar "
                "(host_pool=%s).",
                type(host_pool).__name__,
            )
            return True

        from sglang.srt.managers.cache_controller import HiCacheController

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=host_pool,
            page_size=self.page_size,
            tp_group=self.tp_group,
            load_cache_event=self.load_cache_event,
            io_backend="kernel",
        )
        self.load_back_threshold = 0
        logger.info(
            "HiSparse unified radix cache enabled (compress_ratio=%s, host_pool=%s).",
            compress_ratio,
            type(host_pool).__name__,
        )
        return True

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        empty = self._empty_match_result.device_indices
        if (
            not self._hisparse_load_back_supported
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

        self._last_match_host_indices[node.id] = host_indices[:host_hit_len]
        logger.debug(
            "HiSparse unified radix host prefix match: node=%d raw=%d aligned=%d host_indices=%d",
            node.id,
            raw_match_len,
            host_hit_len,
            host_indices.numel(),
        )
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
        if len(token_ids) == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device="cpu"),
                self.root_node,
                0,
            )

        key = RadixKey(token_ids=token_ids, extra_key=extra_key).page_aligned(
            self.page_size
        )
        if len(key) == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device="cpu"),
                self.root_node,
                0,
            )

        _, best_match_node, _, _ = self._match_prefix_helper(key)
        host_indices = self._collect_host_indices(best_match_node)
        return host_indices, best_match_node, len(host_indices)

    def host_insert(
        self,
        token_ids: list[int],
        host_indices: torch.Tensor,
        extra_key: Optional[str] = None,
    ) -> int:
        if len(token_ids) == 0:
            return 0
        key = RadixKey(token_ids=token_ids, extra_key=extra_key).page_aligned(
            self.page_size
        )
        if len(key) == 0:
            return 0
        value = host_indices[: len(key)].to(dtype=torch.int64, device="cpu")
        return self._host_insert_helper(self.root_node, key, value)

    def host_inc_lock_ref(self, node: UnifiedTreeNode) -> IncLockRefResult:
        return self.inc_lock_ref(node)

    def host_dec_lock_ref(self, node: UnifiedTreeNode) -> DecLockRefResult:
        return self.dec_lock_ref(node)

    def host_evictable_size(self) -> int:
        return sum(
            len(node.component_data[BASE_COMPONENT_TYPE].host_value)
            for node in self.evictable_host_leaves
            if node.component_data[BASE_COMPONENT_TYPE].host_value is not None
        )

    def host_evict(self, num_tokens: int) -> int:
        return self.evict_host(num_tokens)

    def evict_host_if_needed(self, host_pool, need: int) -> None:
        if need <= 0 or host_pool.available_size() >= need:
            return

        deficit = need - host_pool.available_size()
        if self.host_evictable_size() >= deficit:
            self.host_evict(deficit)

    def indexer_pool_transfer_for_range(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        start_pos: int,
        end_pos: int,
    ) -> Optional[PoolTransfer]:
        if self._hisparse_host_pool_group() is None:
            return None

        aligned_start = (
            (start_pos + self.page_size - 1) // self.page_size
        ) * self.page_size
        aligned_end = (end_pos // self.page_size) * self.page_size
        if aligned_end <= aligned_start:
            return None

        host_indices = host_indices[aligned_start:aligned_end]
        if host_indices.numel() == 0:
            return None
        return PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=host_indices,
            device_indices=device_indices[aligned_start:aligned_end].to(
                dtype=torch.int64
            ),
        )

    def indexer_pool_transfer_for_completed_page(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        written_end: int,
    ) -> Optional[PoolTransfer]:
        if written_end <= 0 or written_end % self.page_size != 0:
            return None
        return self.indexer_pool_transfer_for_range(
            host_indices,
            device_indices,
            written_end - self.page_size,
            written_end,
        )

    def backup_from_device_all_layer(
        self,
        default_host_pool,
        mem_pool_device,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        host_pool_group = self._hisparse_host_pool_group()
        if host_pool_group is None:
            default_host_pool.backup_from_device_all_layer(
                mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            return

        host_pool_group.backup_from_device_all_layer(
            mem_pool_device,
            host_indices,
            device_indices,
            io_backend=getattr(self.cache_controller, "io_backend", "kernel"),
            pool_transfers=pool_transfers,
        )

    def match_and_lock_host_prefix(
        self,
        token_ids: list[int],
        extra_key: Optional[str],
        max_len: int,
    ) -> tuple[torch.Tensor, UnifiedTreeNode, int]:
        host_prefix, node, raw_match_len = self.host_match_prefix(token_ids, extra_key)
        prefix_len = min(raw_match_len, max_len)
        if prefix_len > 0 and node is not None:
            self.host_inc_lock_ref(node)
        return host_prefix, node, prefix_len

    def insert_host_indices(
        self,
        token_ids: list[int],
        host_indices: torch.Tensor,
        extra_key: Optional[str],
        protected_len: int,
        old_node: Optional[UnifiedTreeNode] = None,
        lock_new_node: bool = False,
        return_canonical_indices: bool = False,
    ) -> HiSparseHostInsertResult:
        prefix_len = self.host_insert(token_ids, host_indices, extra_key)
        duplicate_host_indices = host_indices.new_empty((0,))
        canonical_host_indices = None

        need_match = old_node is not None or lock_new_node
        if prefix_len > protected_len:
            duplicate_host_indices = host_indices[protected_len:prefix_len]
            need_match |= (
                return_canonical_indices and duplicate_host_indices.numel() > 0
            )

        new_node = None
        if need_match:
            canonical_host_indices_all, new_node, _ = self.host_match_prefix(
                token_ids, extra_key
            )
            if return_canonical_indices and duplicate_host_indices.numel() > 0:
                canonical_host_indices = canonical_host_indices_all[
                    protected_len:prefix_len
                ]

        if old_node is not None and old_node is not self.root_node:
            self.host_dec_lock_ref(old_node)
        if lock_new_node and new_node is not None and new_node is not self.root_node:
            self.host_inc_lock_ref(new_node)

        return HiSparseHostInsertResult(
            prefix_len=prefix_len,
            node=new_node,
            duplicate_host_indices=duplicate_host_indices,
            canonical_host_indices=canonical_host_indices,
        )

    def release_host_node(self, node: Optional[UnifiedTreeNode]) -> None:
        if node is not None and node is not self.root_node:
            self.host_dec_lock_ref(node)

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        best_match_node = params.best_match_node
        host_hit_length = params.host_hit_length
        empty = self._empty_match_result.device_indices

        if (
            not self._hisparse_load_back_supported
            or self.cache_controller is None
            or best_match_node is None
            or host_hit_length <= 0
        ):
            return empty, best_match_node

        host_indices = self._last_match_host_indices.pop(best_match_node.id, None)
        if host_indices is None or host_indices.numel() == 0:
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

        pool_transfers = None
        if self._hisparse_host_pool_group() is not None:
            from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
                CacheOperation,
            )

            pool_transfers = [
                PoolTransfer(
                    name=PoolName.INDEXER,
                    host_indices=host_indices,
                    device_indices=logical_indices,
                )
            ]
        else:
            from sglang.srt.managers.cache_controller import CacheOperation

        kwargs = {"pool_transfers": pool_transfers} if pool_transfers else {}
        self.cache_controller.load_queue.append(
            CacheOperation(
                host_indices=host_indices,
                device_indices=hisparse_indices,
                node_id=best_match_node.id,
                **kwargs,
            )
        )
        self.ongoing_load_back[best_match_node.id] = (
            best_match_node,
            self.inc_lock_ref(best_match_node).to_dec_params(),
        )
        logger.debug(
            "HiSparse unified radix load-back queued: node=%d tokens=%d host_indices=%d",
            best_match_node.id,
            page_aligned_len,
            host_indices.numel(),
        )
        return logical_indices, best_match_node

    def _hisparse_host_pool_group(self):
        cache_controller = getattr(self, "cache_controller", None)
        host_pool_group = getattr(cache_controller, "mem_pool_host", None)
        entry_map = getattr(host_pool_group, "entry_map", {})
        anchor_entry = getattr(host_pool_group, "anchor_entry", None)
        if (
            PoolName.INDEXER in entry_map
            and getattr(anchor_entry, "host_pool", None) is self.hisparse_host_pool
        ):
            return host_pool_group
        return None

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

    def _add_new_host_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        host_value: torch.Tensor,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
        parent.children[key.child_key(self.page_size)] = new_node
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        return new_node

    def _host_insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        host_value: torch.Tensor,
    ) -> int:
        self._touch_node(node)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            if node.component_data[BASE_COMPONENT_TYPE].host_value is None:
                raise RuntimeError(
                    "HiSparse unified radix cache found a matching node without "
                    f"host KV indices (node_id={node.id})."
                )

            self._inc_hit_count(node)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = self._add_new_host_node(node, key, host_value)
            self._inc_hit_count(new_node)
            self.write_backup_storage(new_node)
        return total_prefix_length
