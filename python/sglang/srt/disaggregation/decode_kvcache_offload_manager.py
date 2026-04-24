from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_kv_only_stack,
    build_shared_anchor_stack,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import NSAIndexerPoolHost
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeOffloadStackSpec:
    use_mla: bool
    override_kv_cache_dim: int | None = None
    shared_pool_name: PoolName | None = None
    shared_host_pool_factory: Callable[[Any], Any] | None = None
    extra_pool_specs: tuple[tuple[PoolName, PoolHitPolicy], ...] = ()


def _parse_storage_backend_extra_config(server_args: ServerArgs) -> dict:
    hicache_storage_backend_extra_config = {}
    if server_args.hicache_storage_backend_extra_config:
        try:
            hicache_storage_backend_extra_config = json.loads(
                server_args.hicache_storage_backend_extra_config
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid hicache storage backend extra config JSON: {e}"
            ) from e
    return hicache_storage_backend_extra_config


def _has_decode_offload_attrs(kv_cache: Any, attrs: tuple[str, ...]) -> bool:
    return all(hasattr(kv_cache, attr) for attr in attrs)


def _supports_mha_decode_offload(kv_cache: Any) -> bool:
    return _has_decode_offload_attrs(
        kv_cache,
        ("head_num", "head_dim", "layer_num"),
    )


def _supports_mla_decode_offload(kv_cache: Any) -> bool:
    return _has_decode_offload_attrs(
        kv_cache,
        ("kv_lora_rank", "qk_rope_head_dim", "kv_cache_dim", "layer_num"),
    )


def _supports_indexer_sidecar(kv_cache: Any) -> bool:
    return _supports_mla_decode_offload(kv_cache) and _has_decode_offload_attrs(
        kv_cache,
        ("index_k_with_scale_buffer", "index_head_dim", "quant_block_size"),
    )


def _get_decode_offload_stack_spec(
    kv_cache: Any,
    server_args: ServerArgs,
) -> DecodeOffloadStackSpec | None:
    if _supports_indexer_sidecar(kv_cache):
        return DecodeOffloadStackSpec(
            use_mla=True,
            override_kv_cache_dim=kv_cache.kv_cache_dim,
            shared_pool_name=PoolName.INDEXER,
            shared_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                kv_cache,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            extra_pool_specs=((PoolName.INDEXER, PoolHitPolicy.ALL_PAGES),),
        )
    if _supports_mla_decode_offload(kv_cache):
        return DecodeOffloadStackSpec(use_mla=True)
    if _supports_mha_decode_offload(kv_cache):
        return DecodeOffloadStackSpec(use_mla=False)
    return None


def _build_decode_offload_stack(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    kv_cache: Any,
    tp_group: torch.distributed.ProcessGroup,
    server_args: ServerArgs,
    storage_backend_extra_config: dict,
):
    spec = _get_decode_offload_stack_spec(kv_cache, server_args)
    if spec is None:
        raise ValueError(
            "Unsupported KV cache for decode offload: missing a supported host-pool assembly capability."
        )

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=server_args.page_size,
    )
    layer_mapping = {layer_id: layer_id for layer_id in range(kv_cache.layer_num)}
    common_kwargs = dict(
        params=params,
        server_args=server_args,
        kv_pool=kv_cache,
        full_layer_mapping=layer_mapping,
        page_size=server_args.page_size,
        tp_group=tp_group,
        load_cache_event=threading.Event(),
        storage_backend=server_args.hicache_storage_backend,
        use_mla=spec.use_mla,
        override_kv_cache_dim=spec.override_kv_cache_dim,
        model_name=server_args.served_model_name,
        storage_backend_extra_config=storage_backend_extra_config,
    )
    if spec.shared_pool_name is None:
        return (
            *build_kv_only_stack(**common_kwargs),
            spec.extra_pool_specs,
        )
    return (
        *build_shared_anchor_stack(
            shared_pool_name=spec.shared_pool_name,
            shared_host_pool_factory=spec.shared_host_pool_factory,
            **common_kwargs,
        ),
        spec.extra_pool_specs,
    )


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        hicache_storage_backend_extra_config = _parse_storage_backend_extra_config(
            server_args
        )
        (
            self.decode_host_mem_pool,
            self.cache_controller,
            self._extra_pool_specs,
        ) = _build_decode_offload_stack(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            kv_cache=kv_cache,
            tp_group=tp_group,
            server_args=server_args,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        logger.info("Enable offload kv cache for decode side")

    def _build_extra_pools(self) -> list[PoolTransfer] | None:
        if not self._extra_pool_specs:
            return None
        return [
            PoolTransfer(
                name=name,
                hit_policy=hit_policy,
            )
            for name, hit_policy in self._extra_pool_specs
        ]

    def _write_device_to_host(
        self,
        *,
        device_indices: torch.Tensor,
        node_id: int,
    ) -> torch.Tensor | None:
        extra_pools = self._build_extra_pools()
        if extra_pools is None:
            return self.cache_controller.write(
                device_indices=device_indices,
                node_id=node_id,
            )
        return self.cache_controller.write(
            device_indices=device_indices,
            node_id=node_id,
            extra_pools=extra_pools,
        )

    def _write_host_to_storage(
        self,
        *,
        host_indices: torch.Tensor,
        token_ids: list[int],
        hash_value: list[str],
    ) -> int:
        extra_pools = self._build_extra_pools()
        if extra_pools is None:
            return self.cache_controller.write_storage(
                host_indices,
                token_ids,
                hash_value=hash_value,
            )
        return self.cache_controller.write_storage(
            host_indices,
            token_ids,
            hash_value=hash_value,
            extra_pools=extra_pools,
        )

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Early free prefill-offloaded GPU memory
        if state.prefill_len > 0 and state.inc_len == 0:
            self.token_to_kv_pool_allocator.free(token_indices[: state.prefill_len])

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self._write_device_to_host(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
        )
        state.inc_len += incremental_aligned_len
        return True

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                ) = self.ongoing_offload.pop(ack_id)

                if req.finished():
                    self._release_finished_req(req, start)
                else:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, start:end
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)

                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req, host_indices, incremental_tokens, start_time, prior_hash
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        kv_committed_len = req.pop_committed_kv_cache()
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (NSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.req_to_token_pool.free(req)
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time = self.ongoing_backup.pop(ack_id)

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, start_time, prior_hash
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        ack_id = self._write_host_to_storage(
            host_indices=host_indices,
            token_ids=incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        if req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # If no incremental offload ever happened, the prefill-aligned part was never freed.
        # Free the prefill portion on request finish to avoid leaks.
        if prefill_len > 0 and inc_len == 0:
            token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            self.token_to_kv_pool_allocator.free(token_indices[:prefill_len])
            logger.info(
                f"Finalize release: freed prefill-aligned KV for req {req.rid}, len:{prefill_len}"
            )
        start_offset = prefill_len + inc_len
        self._release_finished_req(req, start_offset)
