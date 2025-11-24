from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostKVCache,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@triton.jit
def sparse_diff_triton_kernel(
    prev_top_k_result_ptr,
    curr_top_k_result_ptr,
    prev_device_indices_ptr,
    curr_device_indices_ptr,
    bitmap_ptr,
    curr_host_indices_ptr,
    full_host_indices_ptr,
    host_start_indices_ptr,
    prev_top_k_result_stride: tl.constexpr,
    curr_top_k_result_stride: tl.constexpr,
    prev_device_indices_stride: tl.constexpr,
    curr_device_indices_stride: tl.constexpr,
    bitmap_stride: tl.constexpr,
    curr_host_indices_stride: tl.constexpr,
    TOPK: tl.constexpr,
):
    bid = tl.program_id(0)
    offset = tl.arange(0, TOPK)
    prev_top_k_result = tl.load(
        prev_top_k_result_ptr + prev_top_k_result_stride * bid + offset
    )
    max_val = tl.max(prev_top_k_result)
    if max_val == -1:
        # After prefilling the first round, the entire cache needs to be loaded.
        no_exist_top_k_result = tl.load(
            curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
        )
        start_index = tl.load(host_start_indices_ptr + bid)
        no_exist_host_indices = tl.load(
            full_host_indices_ptr + start_index + no_exist_top_k_result
        )
        tl.store(
            curr_host_indices_ptr + curr_host_indices_stride * bid + offset,
            no_exist_host_indices,
        )
        return
    tl.store(bitmap_ptr + bitmap_stride * bid + prev_top_k_result, offset)

    curr_top_k_result = tl.load(
        curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
    )
    exist_indices = tl.load(bitmap_ptr + bitmap_stride * bid + curr_top_k_result)

    mask = exist_indices >= 0
    exist_prev_device_indices = tl.load(
        prev_device_indices_ptr + prev_device_indices_stride * bid + exist_indices,
        mask=mask,
    )
    tl.store(
        curr_device_indices_ptr + curr_device_indices_stride * bid + offset,
        exist_prev_device_indices,
        mask=mask,
    )

    tl.store(
        prev_device_indices_ptr + prev_device_indices_stride * bid + exist_indices,
        -1,
        mask=mask,
    )
    tl.store(
        curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset, -1, mask=mask
    )
    tl.store(bitmap_ptr + bitmap_stride * bid + prev_top_k_result, -1)

    no_exist_top_k_result = tl.load(
        curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
    )
    start_index = tl.load(host_start_indices_ptr + bid)
    no_exist_host_indices = tl.load(
        full_host_indices_ptr + start_index + no_exist_top_k_result, mask=~mask
    )
    tl.store(
        curr_host_indices_ptr + curr_host_indices_stride * bid + offset,
        no_exist_host_indices,
        mask=~mask,
    )


class SparseTopKIndicesHelper:
    """Sparse top_k indices helper handles the differences between the top_k indices and extracts the indices that need to be transformed."""

    def __init__(self, server_args: ServerArgs, host_mem_pool: HostKVCache):
        self.device = server_args.device
        self.max_model_len = server_args.model_config.context_len
        self.max_req_bs = server_args.max_num_reqs
        self.host_mem_pool = host_mem_pool

        # init bitmap
        self.bitmap = torch.full(
            (self.max_req_bs, self.max_model_len),
            -1,
            dtype=torch.int16,
            device=self.device,
        )

    def load_top_k_cache(
        self,
        prev_top_k_result: torch.Tensor,
        curr_top_k_result: torch.Tensor,
        prev_device_indices: torch.Tensor,
        full_host_indices: torch.Tensor,
        host_start_indices: torch.Tensor,
        layer_id: int,
    ):
        bs = prev_top_k_result.shape[0]
        top_k = prev_top_k_result.shape[1]
        curr_device_indices = torch.full(
            (bs, top_k), -1, dtype=torch.int64, device=self.device
        )
        curr_host_indices = torch.full(
            (bs, top_k), -1, dtype=torch.int64, device=self.device
        )

        grid = (bs,)
        sparse_diff_triton_kernel[grid](
            prev_top_k_result,
            curr_top_k_result,
            prev_device_indices,
            curr_device_indices,
            self.bitmap,
            curr_host_indices,
            full_host_indices,
            host_start_indices,
            prev_top_k_result.stride(0),
            curr_top_k_result.stride(0),
            prev_device_indices.stride(0),
            curr_device_indices.stride(0),
            self.bitmap.stride(0),
            curr_host_indices.stride(0),
            top_k,
        )

        # TODO(huangtingwei9988)：Further optimization is needed.
        should_load_device_indices = [
            prev_device_indices[i][prev_device_indices[i] != -1] for i in range(bs)
        ]
        should_load_host_indices = [
            curr_host_indices[i][curr_host_indices[i] != -1] for i in range(bs)
        ]
        for i in range(bs):
            mask = curr_device_indices[i] == -1
            curr_device_indices[i][mask] = should_load_device_indices[i]
        should_load_device_indices = torch.cat(should_load_device_indices)
        should_load_host_indices = torch.cat(should_load_host_indices)

        # load cache from cpu
        self.host_mem_pool.load_to_device_per_layer(
            self.host_mem_pool.device_pool,
            should_load_host_indices,
            should_load_device_indices,
            layer_id,
            "kernel",
        )

        return curr_device_indices


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
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=server_args.hicache_storage_backend_extra_config,
        )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        if server_args.enable_sparse_attn:
            self.sparse_decode_ongoing_offload = {}
            self.sparse_prefill_ongoing_offload = {}
            max_pool_size = self.req_to_token_pool.req_to_token.shape[0]
            self.sparse_indices_helper = SparseTopKIndicesHelper(
                server_args, self.decode_host_mem_pool, max_pool_size
            )
            self.req_states = None
        logger.info("Enable offload kv cache for decode side")

    def transform_sparse_top_k_cache(
        self,
        req_pool_indices,
        top_k_result,
        layer_id,
        valid_lengths,
    ):

        # get prev data from req_states
        prev_top_k_result = self.req_states.prev_top_k_result[
            req_pool_indices, layer_id
        ]
        prev_device_indices = self.req_states.prev_device_indices[
            req_pool_indices, layer_id
        ]
        full_host_indices = torch.cat(
            [self.req_states.full_host_indices[idx] for idx in req_pool_indices], dim=0
        )

        host_indices_lens = [0] + [
            len(self.req_states.full_host_indices[idx]) for idx in req_pool_indices
        ][:-1]
        host_indices_lens = torch.tensor(
            host_indices_lens,
            dtype=torch.int64,
            device=full_host_indices.device,
        )
        host_start_indices = torch.cumsum(host_indices_lens, dim=-1)

        curr_device_indices = self.sparse_indices_helper.load_top_k_cache(
            prev_top_k_result=prev_top_k_result,
            curr_top_k_result=top_k_result,
            prev_device_indices=prev_device_indices,
            full_host_indices=full_host_indices,
            host_start_indices=host_start_indices,
            layer_id=layer_id,
        )

        # update indices
        self.req_states.prev_top_k_result[req_pool_indices, layer_id] = top_k_result
        self.req_states.prev_device_indices[req_pool_indices, layer_id] = (
            curr_device_indices
        )

        return curr_device_indices

    def offload_sparse_decode_req_tokens(self, req_pool_indices, out_alloc_len):
        """Offload incremental token KV cache for sparse attention."""

        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=out_alloc_len,
            node_id=ack_id,
        )
        assert host_indices is not None, "Host out of memory"
        self.sparse_decode_ongoing_offload[ack_id] = (host_indices, req_pool_indices)
        return ack_id

    def check_sparse_offload_progress(self):
        """Check the progress of offload from device to host for sparse schedule every step"""
        if len(self.sparse_decode_ongoing_offload) == 0:
            return

        cc = self.cache_controller
        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        finish_count = qsizes.tolist()[0]
        assert finish_count == 1

        _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
        finish_event.synchronize()

        # update full_host_indices
        host_indices, req_pool_indices = self.sparse_decode_ongoing_offload.pop(
            ack_list[0]
        )
        for i in range(len(req_pool_indices)):
            full_host_indices = self.req_states.full_host_indices[req_pool_indices[i]]
            self.req_states.full_host_indices[req_pool_indices[i]] = torch.cat(
                [full_host_indices, host_indices[i].to(self.req_states.device)]
            )

    def offload_prefill_full_kv_cache(self, req):
        token_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.seqlen - 1
        ]

        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=token_indices,
            node_id=ack_id,
        )
        assert host_indices is not None, "Host out of memory"
        self.sparse_prefill_ongoing_offload[ack_id] = (host_indices, req)
        return ack_id

    def check_prefill_offload_progress(self):
        if len(self.sparse_prefill_ongoing_offload) == 0:
            return

        cc = self.cache_controller
        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        finish_count = qsizes.tolist()[0]
        assert finish_count == 1

        _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
        finish_event.synchronize()

        (host_indices, req) = self.sparse_prefill_ongoing_offload.pop(ack_list[0])
        self.req_states.full_host_indices[req.req_pool_idx] = host_indices.to(
            self.req_states.device
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
        incremental_len = len(all_tokens) - prefill_offloaded_len
        incremental_aligned_len = incremental_len // self.page_size * self.page_size

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices
        start, end = (
            prefill_offloaded_len,
            prefill_offloaded_len + incremental_aligned_len,
        )
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Early free prefill-offloaded GPU memory
        if prefill_offloaded_len > 0:
            self.token_to_kv_pool_allocator.free(token_indices[:prefill_offloaded_len])

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

        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            prefill_offloaded_len,
        )
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
                    prefill_offloaded_len,
                ) = self.ongoing_offload.pop(ack_id)

                self._release_finished_req(req, prefill_offloaded_len)
                self._trigger_backup(
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    prefill_offloaded_len,
                )
            finish_count -= 1

    def _release_finished_req(self, req: Req, prefill_offloaded_len: int):
        # FIXME: not sure which length to use here: kv_allocated_len or kv_committed_len
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, prefill_offloaded_len : req.kv_allocated_len
        ]

        # Free the incremental part of the request
        self.token_to_kv_pool_allocator.free(kv_indices)
        self.req_to_token_pool.free(req.req_pool_idx)

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time = self.ongoing_backup.pop(ack_id)

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)

            logger.info(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, start_time, prefill_offloaded_len
    ):
        """Trigger async backup from host to storage."""
        prefill_hashes = self._compute_prefix_hash(
            req.origin_input_ids[:prefill_offloaded_len]
        )
        last_prefill_hash = prefill_hashes[-1] if prefill_offloaded_len > 0 else ""

        page_hashes = self._compute_prefix_hash(incremental_tokens, last_prefill_hash)
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes
