from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_hybrid_mamba_stack,
    build_kv_host_pool,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.pool_host.common import get_allocator_type
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


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
        parallel = get_parallel()
        self.page_size = (
            token_to_kv_pool_allocator.page_size
            if parallel.dcp_enabled
            else server_args.page_size
        )
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
        hicache_storage_backend_extra_config = (
            server_args.hicache_storage_backend_extra_config_dict
        )

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        allocator_type = get_allocator_type(server_args)
        self.is_hybrid_mamba = isinstance(kv_cache, HybridLinearKVPool)

        if self.is_hybrid_mamba:
            params = CacheInitParams(
                disable=True,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                page_size=self.page_size,
                tp_cache_group=tp_group,
            )
            (
                self.decode_host_mem_pool,
                self.cache_controller,
            ) = build_hybrid_mamba_stack(
                params=params,
                server_args=server_args,
                kv_pool=kv_cache.full_kv_pool,
                mamba_pool=req_to_token_pool.mamba_pool,
                full_layer_mapping=dict(kv_cache.full_attention_layer_id_mapping),
                mamba_layer_mapping=dict(req_to_token_pool.mamba_map),
                load_cache_event=threading.Event(),
                storage_backend=server_args.hicache_storage_backend,
                use_mla=kv_cache.use_mla,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )
            self.decode_mamba_host_pool = self.decode_host_mem_pool.get_pool(
                PoolName.MAMBA
            )
            req_to_token_pool.register_layer_transfer_counter(
                self.cache_controller.layer_done_counter
            )
            kv_cache.register_layer_transfer_counter(
                self.cache_controller.layer_done_counter
            )
        elif isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = get_mha_host_pool_cls(kv_cache)(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=allocator_type,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = build_kv_host_pool(
                kv_pool=kv_cache,
                page_size=self.page_size,
                server_args=server_args,
                use_mla=True,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        if not self.is_hybrid_mamba:
            self.cache_controller = HiCacheController(
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                mem_pool_host=self.decode_host_mem_pool,
                page_size=self.page_size,
                tp_group=tp_group,
                io_backend=server_args.hicache_io_backend,
                load_cache_event=threading.Event(),
                storage_backend=server_args.hicache_storage_backend,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )
        self.canonical_storage_layout = bool(
            hicache_storage_backend_extra_config.get("canonical_dcp_size", 0)
        )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        self.offload_inflight = {}
        logger.info("Enable offload kv cache for decode side")

    def release_host_resources(self) -> None:
        self.decode_host_mem_pool.destroy()

    def _mark_offload_started(self, rid):
        self.offload_inflight[rid] = self.offload_inflight.get(rid, 0) + 1

    def _mark_offload_finished(self, rid):
        count = self.offload_inflight.get(rid, 0)
        if count <= 1:
            self.offload_inflight.pop(rid, None)
        else:
            self.offload_inflight[rid] = count - 1

    def _has_inflight_offload(self, rid):
        return self.offload_inflight.get(rid, 0) > 0

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # In the ordinary layout, Prefill owns the page-aligned prompt and
        # Decode only appends incremental pages. A canonical cross-DCP chain
        # cannot make that assumption: PD Prefill inserts intermediate chunks
        # with chunked=True, which deliberately suppresses HiCache write-
        # through. Decode therefore publishes the complete chain from offset
        # zero, beginning with the transferred prompt checkpoint.
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        offload_start = (
            0 if self.canonical_storage_layout else prefill_offloaded_len
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_tokens = req.origin_input_ids[:offload_start]
            prefill_hashes = self._compute_prefix_hash(prefill_tokens)
            storage_prefill_hashes = self._compute_prefix_hash(
                prefill_tokens,
                page_size=self.cache_controller.storage_page_size,
            )
            last_prefill_hash = (
                prefill_hashes[-1] if offload_start > 0 else None
            )
            storage_last_prefill_hash = (
                storage_prefill_hashes[-1] if offload_start > 0 else None
            )
            state = OffloadedState(
                prefill_len=offload_start,
                inc_len=0,
                last_hash=last_prefill_hash,
                storage_last_hash=storage_last_prefill_hash,
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

        # Prefill-aligned GPU slots are freed at request finish in
        # _release_finished_req, NOT here. The decoding request
        # continues to attend to those slots via req_to_token; freeing
        # them mid-decode races with concurrent admission, which can
        # reuse the slots and produce cross-pollinated KV reads.

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        extra_pools = self._build_extra_pool_transfers(req, checkpoint_len=end)
        if self.is_hybrid_mamba and extra_pools is None:
            logger.error("Missing KDA state for request %s", req.rid)
            return False

        write_kwargs = {}
        if extra_pools is not None:
            write_kwargs["extra_pools"] = extra_pools
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
            **write_kwargs,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self._mark_offload_started(req.rid)
        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
            extra_pools,
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
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                    extra_pools,
                ) = self.ongoing_offload.pop(ack_id)

                self._mark_offload_finished(req.rid)
                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                storage_prior_hash = (
                    self.offloaded_state[req.rid].storage_last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                last_hash, storage_last_hash = self._trigger_backup(
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    prior_hash,
                    storage_prior_hash,
                    extra_pools,
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash
                    self.offloaded_state[req.rid].storage_last_hash = (
                        storage_last_hash
                    )

                if req.finished() and not self._has_inflight_offload(req.rid):
                    state = self.offloaded_state.get(req.rid)
                    start_offset = state.prefill_len if state is not None else start
                    self._release_finished_req(req, start_offset)
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (e.g. tree_cache.protected_size_
        # double-decrement, host pool double-free).
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return

        kv_committed_len = req.effective_kv_committed_len()

        # Free the prefill-aligned slots. Previously this was done
        # eagerly in offload_kv_cache (mid-decode), which raced with
        # concurrent admission. Now consolidated here at request
        # finish, where the request is guaranteed to no longer attend
        # to those slots.
        state = self.offloaded_state.get(req.rid)
        if state is not None and state.prefill_len > 0:
            prefill_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : state.prefill_len
            ]
            self.token_to_kv_pool_allocator.free(prefill_indices)
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (DSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = kv_committed_len, req.kv.kv_allocated_len
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        # Hybrid requests own a separate Mamba/KDA state slot. The regular
        # release_kv_cache path frees it before releasing req_to_token, but
        # decode offload bypasses that helper and must mirror the lifecycle
        # here after the device-to-host copy has completed.
        if self.is_hybrid_mamba and req.mamba_pool_idx is not None:
            self.req_to_token_pool.free_mamba_cache(req)
        self.req_to_token_pool.free(req)
        req.kv = None
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, extra_pools, start_time = self.ongoing_backup.pop(
                ack_id
            )

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)
            self._free_extra_host_indices(extra_pools)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self,
        req,
        host_indices,
        incremental_tokens,
        start_time,
        prior_hash,
        storage_prior_hash,
        extra_pools=None,
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        storage_page_hashes = self._compute_prefix_hash(
            incremental_tokens,
            storage_prior_hash,
            page_size=self.cache_controller.storage_page_size,
        )
        for transfer in extra_pools or []:
            if transfer.name == PoolName.MAMBA:
                transfer.keys = [storage_page_hashes[-1]]
                transfer.hit_policy = PoolHitPolicy.TRAILING_PAGES
        storage_kwargs = {}
        if self.is_hybrid_mamba:
            storage_kwargs["storage_hash_value"] = storage_page_hashes
            if extra_pools is not None:
                storage_kwargs["extra_pools"] = extra_pools
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
            **storage_kwargs,
        )
        self.ongoing_backup[ack_id] = (
            req.rid,
            host_indices,
            extra_pools,
            start_time,
        )
        return (
            page_hashes[-1] if page_hashes else prior_hash,
            storage_page_hashes[-1]
            if storage_page_hashes
            else storage_prior_hash,
        )

    def _build_extra_pool_transfers(self, req, checkpoint_len=None):
        if not self.is_hybrid_mamba:
            return None
        mamba_pool_idx = getattr(req, "mamba_pool_idx", None)
        if self.canonical_storage_layout:
            tracked_len = getattr(req, "mamba_last_track_seqlen", None)
            if tracked_len != checkpoint_len:
                logger.warning(
                    "Skip Decode-origin L3 publication for request %s: "
                    "KDA checkpoint is at %s, storage boundary is %s.",
                    req.rid,
                    tracked_len,
                    checkpoint_len,
                )
                return None
            ping_pong = getattr(req, "mamba_ping_pong_track_buffer", None)
            if ping_pong is None:
                return None
            keep_idx = self.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
            mamba_pool_idx = ping_pong[keep_idx]
        if mamba_pool_idx is None:
            return None
        if not isinstance(mamba_pool_idx, torch.Tensor):
            mamba_pool_idx = torch.tensor(
                mamba_pool_idx,
                dtype=torch.int64,
                device=self.token_to_kv_pool_allocator.device,
            )
        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                device_indices=mamba_pool_idx.reshape(1),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def _free_extra_host_indices(self, extra_pools) -> None:
        if not extra_pools or not self.is_hybrid_mamba:
            return
        for transfer in extra_pools:
            if transfer.host_indices is None:
                continue
            entry = self.decode_host_mem_pool.entry_map.get(transfer.name)
            if entry is not None:
                entry.host_pool.free(transfer.host_indices)

    def _compute_prefix_hash(self, tokens, prior_hash="", page_size=None):
        page_size = page_size or self.page_size
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), page_size):
            page_tokens = tokens[offset : offset + page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        # ReqToTokenPool.free sets req_pool_idx to None on release, so
        # guard against both sentinels here.
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # Prefill-aligned slots are freed by _release_finished_req. Make
        # sure state exists so it can find prefill_len.
        if state is None:
            self.offloaded_state[req.rid] = OffloadedState(
                prefill_len=prefill_len, inc_len=0, last_hash=None
            )
        if self._has_inflight_offload(req.rid):
            return
        start_offset = prefill_len
        self._release_finished_req(req, start_offset)
