from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.sparsity.kernel.diff_kernel import invoke_sparse_diff_kernel
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SparseKVCacheManager:
    """
    Manages KV cache offloading between device (GPU) and host (CPU) memory
    for hierarchical sparse attention.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.next_offload_id = 0

        # Initialize host memory pool based on KV cache type
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for sparse attention offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # Initialize cache controller for device-host transfers
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=server_args.hicache_storage_backend_extra_config,
            enable_hierarchical_sparse_attention=True,
        )

        # Track pending offload operations
        self.pending_sparse_decode_offloads = {}
        self.pending_sparse_prompt_offloads = {}

        # Initialize bitmap for tracking KV cache locations
        max_pool_size = self.req_to_token_pool.req_to_token.shape[0]
        self.bitmap = torch.full(
            (max_pool_size, server_args.model_config.context_len),
            -1,
            dtype=torch.int16,
            device=server_args.device,
        )
        self.req_states = None

    def swap_in_selected_pages(
        self,
        req_pool_indices,
        top_k_result,
        seq_lens,
        sparse_mask,
        page_table,
        layer_id,
        page_size,
        out_cache_loc,
    ):
        """
        Swap in selected top-k pages/tokens from host to device memory.
        First step: Using diff kernel to identify the top-k pages/tokens that need to be swapped in.
        Second step: Using the io kernel to load the pages/tokens from host to device.

        Returns:
            Device indices of the selected pages/tokens
        """
        batch_size = sparse_mask.shape[0]

        invoke_sparse_diff_kernel(
            self.req_states.last_top_k_result,
            top_k_result,
            self.req_states.last_device_indices,
            self.req_states.curr_device_indices,
            self.bitmap,
            self.req_states.req_to_tokens_host,
            self.req_states.should_load_device_indices,
            self.req_states.should_load_host_indices,
            seq_lens,
            req_pool_indices,
            sparse_mask,
            page_table,
            layer_id,
            self.req_states.topk_tokens_cnt,
            self.req_states.device_buffer_cnt,
            page_size,
        )
        swap_target_device_slots = self.req_states.should_load_device_indices[
            :batch_size, : self.req_states.topk_tokens_cnt
        ]
        swap_source_host_slots = self.req_states.should_load_host_indices[
            :batch_size, : self.req_states.topk_tokens_cnt
        ]
        swap_target_device_slots = swap_target_device_slots[
            swap_target_device_slots != -1
        ]
        swap_source_host_slots = swap_source_host_slots[swap_source_host_slots != -1]

        assert (
            swap_target_device_slots.numel() == swap_source_host_slots.numel()
        ), "Swap target device slots and source host slots must have the same number of elements"

        # Load cache from host to device
        if swap_target_device_slots.numel() > 0:
            self.host_mem_pool.load_to_device_per_layer(
                self.host_mem_pool.device_pool,
                swap_source_host_slots.flatten(),
                swap_target_device_slots.flatten(),
                layer_id,
                "kernel",
            )

        return self.req_states.curr_device_indices[
            :batch_size, : self.req_states.topk_tokens_cnt // page_size
        ]

    def offload_decode_token_kvcache(
        self, req_pool_indices, device_cache_locs, seq_lens
    ):
        """
        Offload newly generated decode token KV cache from device to host.

        Returns:
            Offload operation ID for tracking completion
        """
        self.next_offload_id += 1
        offload_id = self.next_offload_id
        host_indices = self.cache_controller.write(
            device_indices=device_cache_locs.long(),
            node_id=offload_id,
            sparse_ack_type="decode",
        )
        assert host_indices is not None, "Host out of memory"
        self.pending_sparse_decode_offloads[offload_id] = (
            host_indices,
            req_pool_indices,
            seq_lens,
        )
        return offload_id

    def poll_decode_offload_completion(self):
        """
        Poll and finalize completed decode token KV cache offload operations.

        Checks if pending decode offload operations have completed and updates
        the host indices mapping.
        """
        if len(self.pending_sparse_decode_offloads) == 0:
            return

        controller = self.cache_controller
        queue_sizes = torch.tensor(
            [len(controller.ack_sparse_decode_write_queue)],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_sizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        completed_count = queue_sizes.tolist()[0]
        assert completed_count == 1, f"Expected 1 completion, got {completed_count}"

        _, finish_event, offload_ids = controller.ack_sparse_decode_write_queue.pop(0)
        finish_event.synchronize()

        # Update host indices mapping
        host_indices, req_pool_indices, seq_lens = (
            self.pending_sparse_decode_offloads.pop(offload_ids[0])
        )
        self.req_states.req_to_tokens_host[req_pool_indices, seq_lens] = (
            host_indices.to(self.req_states.device)
        )

    def offload_prompt_kvcache(self, req):
        """
        Offload full prompt KV cache from device to host after prefill.

        Returns:
            Offload operation ID for tracking completion
        """
        prompt_len = len(req.origin_input_ids)
        token_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prompt_len
        ].long()

        self.next_offload_id += 1
        offload_id = self.next_offload_id
        host_indices = self.cache_controller.write(
            device_indices=token_indices, node_id=offload_id, sparse_ack_type="prompt"
        )
        assert host_indices is not None, "Host out of memory"
        self.pending_sparse_prompt_offloads[offload_id] = (host_indices, req)
        return offload_id

    def poll_prompt_offload_completion(self):
        """
        Poll and finalize completed prompt KV cache offload operations.

        Checks if pending prompt offload operations have completed and updates
        the host indices mapping for each completed request.

        Returns:
            List of requests whose prompt KV cache offload has completed
        """
        completed_reqs = []
        if len(self.pending_sparse_prompt_offloads) == 0:
            return completed_reqs

        completed_count = 0
        for _, finish_event, _ in self.cache_controller.ack_sparse_prompt_write_queue:
            if not finish_event.query():
                break
            completed_count += 1

        # Sync completion count across TP ranks
        if self.tp_world_size > 1:
            queue_size = torch.tensor(completed_count, dtype=torch.int, device="cpu")
            torch.distributed.all_reduce(
                queue_size, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
            completed_count = int(queue_size.item())

        # Process all completed offload operations
        while completed_count > 0:
            _, finish_event, offload_ids = (
                self.cache_controller.ack_sparse_prompt_write_queue.pop(0)
            )
            finish_event.synchronize()

            host_indices, req = self.pending_sparse_prompt_offloads.pop(offload_ids[0])
            self.req_states.req_to_tokens_host[req.req_pool_idx][
                : len(host_indices)
            ] = host_indices.to(self.req_states.device)
            completed_reqs.append(req)
            completed_count -= 1

        return completed_reqs
