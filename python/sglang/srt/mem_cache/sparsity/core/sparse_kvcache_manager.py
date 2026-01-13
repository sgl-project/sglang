from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheAck
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
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

device_module = get_device_module()
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
        self.mem_pool_device = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.mem_pool_device, MHATokenToKVPool):
            self.mem_pool_host = MHATokenToKVPoolHost(
                self.mem_pool_device,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
        elif isinstance(self.mem_pool_device, MLATokenToKVPool):
            self.mem_pool_host = MLATokenToKVPoolHost(
                self.mem_pool_device,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for sparse attention offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # Track pending offload operations
        self.pending_sparse_decode_offloads = {}
        self.pending_sparse_prompt_offloads = {}

        # Separate ack queues for prompt and decode offloads
        self.ack_sparse_prompt_write_queue = []
        self.ack_sparse_decode_write_queue = []

        self.write_queue = []
        self.write_stream = device_module.Stream()
        self.device = self.mem_pool_device.device
        self.io_backend = "kernel" 

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
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
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
        host_indices = self._write_to_host(
            device_indices=device_cache_locs.long(),
            node_id=offload_id,
            sparse_ack_type="decode_offload",
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

        queue_sizes = torch.tensor(
            [len(self.ack_sparse_decode_write_queue)],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_sizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        completed_count = queue_sizes.tolist()[0]
        assert completed_count == 1, f"Expected 1 completion, got {completed_count}"

        _, finish_event, offload_ids = self.ack_sparse_decode_write_queue.pop(0)
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
        host_indices = self._write_to_host(
            device_indices=token_indices, node_id=offload_id, sparse_ack_type="prompt_offload"
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
        for _, finish_event, _ in self.ack_sparse_prompt_write_queue:
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
                self.ack_sparse_prompt_write_queue.pop(0)
            )
            finish_event.synchronize()

            host_indices, req = self.pending_sparse_prompt_offloads.pop(offload_ids[0])
            self.req_states.req_to_tokens_host[req.req_pool_idx][
                : len(host_indices)
            ] = host_indices.to(self.req_states.device)
            completed_reqs.append(req)
            completed_count -= 1

        return completed_reqs

    def _write_to_host(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        sparse_ack_type = "prompt_offload"
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self._start_writing(sparse_ack_type=sparse_ack_type)
        return host_indices

    def _start_writing(self, sparse_ack_type: str) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        # Route ack to appropriate queue
        ack = HiCacheAck(start_event, finish_event, op.node_ids)

        if sparse_ack_type == "prompt_offload":
            self.ack_sparse_prompt_write_queue.append(ack)
        elif sparse_ack_type == "decode_offload":
            self.ack_sparse_decode_write_queue.append(ack)


    def move_indices(self, op: CacheOperation):
        """Move indices to device if needed."""
        host_indices = op.host_indices.to(self.device, non_blocking=True)
        device_indices = op.device_indices.to(self.device, non_blocking=True)
        return host_indices, device_indices