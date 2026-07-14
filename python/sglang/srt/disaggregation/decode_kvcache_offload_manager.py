from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import is_npu

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
        if is_npu():
            raise ValueError("Decode KV cache offload is not supported on NPU.")

        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.storage_page_size = server_args.page_size
        self.allocator_page_size = token_to_kv_pool_allocator.page_size
        if self.allocator_page_size % self.storage_page_size != 0:
            raise ValueError(
                "Decode KV cache offload requires the storage page size to divide "
                "the allocator page size"
            )
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.storage_page_size
        else:
            self.offload_stride = max(
                self.storage_page_size,
                (env_stride // self.storage_page_size) * self.storage_page_size,
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = get_mha_host_pool_cls(kv_cache)(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.storage_page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.storage_page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.storage_page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        self.offload_inflight = {}
        logger.info("Enable offload kv cache for decode side")

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

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.storage_page_size * self.storage_page_size
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

        # Prefill-aligned GPU slots are freed at request finish in
        # _release_finished_req, NOT here. The decoding request
        # continues to attend to those slots via req_to_token; freeing
        # them mid-decode races with concurrent admission, which can
        # reuse the slots and produce cross-pollinated KV reads.

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
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
                    _start,
                    _end,
                ) = self.ongoing_offload.pop(ack_id)

                self._mark_offload_finished(req.rid)
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

                if req.finished() and not self._has_inflight_offload(req.rid):
                    self._release_finished_req(req)
            finish_count -= 1

    def _release_finished_req(self, req: Req) -> None:
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (e.g. tree_cache.protected_size_
        # double-decrement, host pool double-free).
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return

        # Free the prefill-aligned slots. Previously this was done
        # eagerly in offload_kv_cache (mid-decode), which raced with
        # concurrent admission. Now consolidated here at request
        # finish, where the request is guaranteed to no longer attend
        # to those slots.
        start_offset = (
            len(req.origin_input_ids)
            // self.allocator_page_size
            * self.allocator_page_size
        )
        kv_committed_len = req.effective_kv_committed_len()
        kv_allocated_len = req.kv.kv_allocated_len
        if (
            start_offset > kv_committed_len
            or kv_committed_len > kv_allocated_len
            or start_offset % self.allocator_page_size != 0
            or kv_allocated_len % self.allocator_page_size != 0
        ):
            raise RuntimeError(
                "Invalid decode KV offload ownership bounds: "
                f"start={start_offset}, committed={kv_committed_len}, "
                f"allocated={kv_allocated_len}, "
                f"page_size={self.allocator_page_size}"
            )

        if start_offset > 0:
            prefill_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :start_offset
            ]
            self.token_to_kv_pool_allocator.free(prefill_indices)

        if start_offset < kv_allocated_len:
            remaining_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_offset:kv_allocated_len
            ]
            self.token_to_kv_pool_allocator.free(remaining_indices)

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
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.storage_page_size):
            page_tokens = tokens[offset : offset + self.storage_page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req) -> None:
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        # ReqToTokenPool.free sets req_pool_idx to None on release, so
        # guard against both sentinels here.
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = (
                len(req.origin_input_ids)
                // self.storage_page_size
                * self.storage_page_size
            )
        # Prefill-aligned slots are freed by _release_finished_req. Make
        # sure state exists so it can find prefill_len.
        if state is None:
            self.offloaded_state[req.rid] = OffloadedState(
                prefill_len=prefill_len, inc_len=0, last_hash=None
            )
        if self._has_inflight_offload(req.rid):
            return
        self._release_finished_req(req)
