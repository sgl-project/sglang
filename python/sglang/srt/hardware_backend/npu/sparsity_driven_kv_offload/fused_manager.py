"""Sparsity-driven KV offload manager for the Ascend backend."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    unidex_copy_inplace,
)

from sglang.kernels.ops.attention.npu_copy_sfa import (
    fused_scatter_copy_sparse_flash_attention,
    load_copy_sfa_operator,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    prepare_prefill_gather_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    sparse_batch_capacity,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparse_kv_budget,
    get_sparse_kv_max_query_width,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    prepare_mtp_rows,
    sparse_query_width,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import get_eager_max_batch_size
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    import torch.npu

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class SparseKVCacheManager:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        sparse_context_len: int,
        copy_sfa_padded_query_heads: Optional[int] = None,
        graph_batch_capacity: int = 0,
        paged_kv_cache: Optional[MLATokenToKVPool] = None,
    ) -> None:
        enable_memory_saver = False
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        # Include the padding row because real request IDs can equal the
        # configured capacity when row 0 is reserved for graph padding.
        self.size = int(req_to_token_pool.req_to_token.shape[0])
        self.batch_capacity = sparse_batch_capacity(
            self.size,
            max(graph_batch_capacity, get_eager_max_batch_size(self.size - 1)),
        )
        self.last_decode_path = "unselected"
        self.logical_max_context_len = int(req_to_token_pool.max_context_len)
        self.max_query_width = get_sparse_kv_max_query_width()
        self.req_to_token_pool = req_to_token_pool
        self.max_context_len = (self.logical_max_context_len + 127) // 128 * 128
        self.sparse_context_len = int(sparse_context_len)
        if self.sparse_context_len <= 0:
            raise ValueError(
                "SparseKVCacheManager requires a positive sparse_context_len, "
                f"got {self.sparse_context_len}."
            )
        self.device = req_to_token_pool.device
        # EAGLE shares the target allocator's token IDs while owning a
        # separate draft KV pool. The allocator's pool is not the draft pool.
        if paged_kv_cache is None:
            paged_kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(paged_kv_cache, MLATokenToKVPool):
            raise TypeError(
                "SparseKVCacheManager requires an MLATokenToKVPool, "
                f"got {type(paged_kv_cache).__name__}"
            )
        self.paged_kv_cache = paged_kv_cache
        self.start_layer = paged_kv_cache.start_layer
        # MLA params
        self.head_num = 1
        self.kv_lora_rank = self.paged_kv_cache.kv_lora_rank
        self.qk_rope_head_dim = self.paged_kv_cache.qk_rope_head_dim
        # kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.head_dim = (
            self.paged_kv_cache.kv_lora_rank + self.paged_kv_cache.qk_rope_head_dim
        )
        self.store_dtype = self.paged_kv_cache.store_dtype
        self.layer_num = self.paged_kv_cache.layer_num
        self.host_token_capacity = (
            (int(paged_kv_cache.size) + int(paged_kv_cache.page_size) + 127) // 128
        ) * 128
        self.budget = get_sparse_kv_budget(
            request_rows=self.size,
            context_len=self.max_context_len,
            num_layers=self.layer_num,
            top_k=self.sparse_context_len,
            kv_bytes_per_token=self.head_dim * self.store_dtype.itemsize,
            copy_sfa_query_heads=copy_sfa_padded_query_heads or 128,
            graph_batch_capacity=self.batch_capacity,
        )
        self.device_cache_size = self.budget.cache_tokens
        self.budget = replace(
            self.budget,
            host_kv_bytes=(
                self.host_token_capacity
                * self.layer_num
                * self.head_num
                * self.head_dim
                * self.store_dtype.itemsize
            ),
        )
        self._slot_map_width = (self.max_context_len + 127) // 128 * 128
        logger.info(
            "Sparse KV per rank: K=%d C=%d, managed HBM=%.2f MiB, host KV=%.2f MiB, host_layout=paged",
            self.sparse_context_len,
            self.device_cache_size,
            self.budget.managed_hbm_bytes / 1024**2,
            self.budget.host_kv_bytes / 1024**2,
        )
        self.cache_capacity = self.device_cache_size
        if (
            not 2048 <= self.cache_capacity <= 32640
            or self.cache_capacity % 128
            or copy_sfa_padded_query_heads is None
        ):
            raise ValueError(
                "Copy-SFA requires C in [2048,32640], aligned to 128, and a padded query head count."
            )
        if (self.kv_lora_rank, self.qk_rope_head_dim) != (512, 64):
            raise ValueError(
                "Copy-SFA supports only MLA KV dimensions CKV=512 and KPE=64; "
                f"got CKV={self.kv_lora_rank}, KPE={self.qk_rope_head_dim}."
            )
        if self.store_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "Copy-SFA supports only BF16 or FP16 KV storage; "
                f"got {self.store_dtype}."
            )
        self._copy_sfa_operator = load_copy_sfa_operator()

        # device KV buffer
        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                # [bs, ctx_len, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.device_kv_buffer: list[torch.Tensor] = [
                    torch.empty(
                        (
                            self.size,
                            self.device_cache_size,
                            self.head_num,
                            self.head_dim,
                        ),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
        except Exception as e:
            self._raise_buffer_allocation_error("device_kv_buffer", e)
        self.resident_begin = torch.full(
            (self.layer_num, self.size), -1, dtype=torch.int32, device=self.device
        )
        self.resident_end = torch.full_like(self.resident_begin, -1)
        self.next_extents = tuple(
            torch.empty(self.size, dtype=torch.int32, device=self.device)
            for _ in range(3)
        )
        history_count = self.batch_capacity * self.device_cache_size
        hs = torch.empty(history_count, dtype=torch.int64, device=self.device)
        self.history_workspace = (
            hs,
            torch.empty_like(hs),
            torch.empty(history_count, dtype=torch.bool, device=self.device),
        )
        token_capacity = self.batch_capacity * self.max_query_width
        self.current_workspace = (
            torch.empty(token_capacity, dtype=torch.int64, device=self.device),
            torch.empty(token_capacity, dtype=torch.bool, device=self.device),
        )
        self.device_slot_map = [
            torch.full(
                (self.size + 1, self._slot_map_width),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.planner_actual_seq_lengths_kv = torch.empty(
            self.batch_capacity, dtype=torch.int32, device=self.device
        )
        self.planner_num_cache_tokens = torch.empty_like(
            self.planner_actual_seq_lengths_kv
        )
        self.planner_hbm_block_table = torch.empty(
            (self.batch_capacity, self.device_cache_size // 128),
            dtype=torch.int32,
            device=self.device,
        )
        self.planner_dram_block_table = torch.empty(
            (self.batch_capacity, self.max_context_len // 128),
            dtype=torch.int32,
            device=self.device,
        )
        self.fused_attn_out = torch.empty(
            (token_capacity, copy_sfa_padded_query_heads, self.kv_lora_rank),
            dtype=self.store_dtype,
            device=self.device,
        )

        # Host KV buffer
        # [bs, ctx_len, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.host_kv_buffer: list[torch.Tensor] = []
        self.host_ptr_list: list[int] = []
        self.dev_ptr_list: list[int] = []

        # Radix prefixes and accepted-token relocation are expressed in native
        # physical token IDs. Keep host payloads in that same address space;
        # request slots own only the bounded HBM residency maps.
        host_kv_shape = (
            1,
            self.host_token_capacity,
            self.head_num,
            self.head_dim,
        )
        logger.info("Sparse KV host buffer shape: %s", host_kv_shape)
        device_id = torch.npu.current_device()

        try:
            for layer_idx in range(self.layer_num):
                shm_cpu_tensor, host_ptr, dev_ptr = create_shm_tensor(
                    shape=host_kv_shape,
                    dtype=self.store_dtype,
                    device_id=device_id,
                    name=f"host_kv_pool_{id(paged_kv_cache)}_layer_{layer_idx}_rank_{device_id}",
                )
                self.host_kv_buffer.append(shm_cpu_tensor)
                self.host_ptr_list.append(host_ptr)
                self.dev_ptr_list.append(dev_ptr)
        except Exception as e:
            self._raise_buffer_allocation_error("host_kv_buffer", e)
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
            LIMDecodeAdapter,
        )

        self.lim_adapter = LIMDecodeAdapter(self)

        self._install_req_alloc_hook(req_to_token_pool)

    def _raise_buffer_allocation_error(
        self,
        buffer_name: str,
        exc: Exception,
    ) -> None:
        raise RuntimeError(
            "Failed to allocate sparse KV buffer "
            f"{buffer_name}: req_capacity={self.size}, "
            f"max_context_len={self.max_context_len}, "
            f"sparse_context_len={self.sparse_context_len}. "
            "The sparse KV request capacity may be too large; set a smaller "
            "--max-running-requests for sparse KV offload."
        ) from exc

    def host_source_indices(self, request_ids, positions):
        """Resolve logical positions through the allocator's shared token map."""
        # Invalid/padded entries are masked by the caller. Clamp only their
        # speculative address calculation, never an active token's mapping.
        rows = request_ids.long().clamp(0, self.size - 1)
        columns = positions.long().clamp(0, self.logical_max_context_len - 1)
        return self.req_to_token_pool.req_to_token[rows, columns].long()

    def move_host_kv(self, target, source):
        """Mirror accepted-token compaction before native physical IDs recycle."""
        if source.numel() == 0:
            return
        count = source.numel()
        scratch = torch.empty(
            (count, self.head_num, self.head_dim),
            dtype=self.store_dtype,
            device=self.device,
        )
        rows = torch.arange(count, device=self.device, dtype=torch.long)
        valid = torch.ones(count, device=self.device, dtype=torch.bool)
        # Stage the complete source set so overlapping/permuted moves have the
        # same snapshot semantics as native advanced-index assignment.
        for layer, pointer in zip(self.host_kv_buffer, self.dev_ptr_list):
            unidex_copy_inplace(
                layer,
                scratch,
                source,
                rows,
                valid,
                2,
                1,
                block_dim=48,
                src_ptr=pointer,
            )
            unidex_copy_inplace(
                scratch,
                layer,
                rows,
                target,
                valid,
                1,
                2,
                block_dim=48,
                dst_ptr=pointer,
            )

    def reset_requests(self, req_ids: List[int]) -> None:
        if not req_ids:
            return

        req_ids_tensor = torch.tensor(
            req_ids, dtype=torch.long, device=self.device
        ).contiguous()
        for layer_idx in range(self.layer_num):
            self.resident_begin[layer_idx].index_fill_(0, req_ids_tensor, -1)
            self.resident_end[layer_idx].index_fill_(0, req_ids_tensor, -1)
            self.device_slot_map[layer_idx].index_fill_(0, req_ids_tensor, -1)
        self.lim_adapter.reset_requests(req_ids_tensor)

    def copy_sfa_attention(
        self, layer, query_nope, query_rope, query_heads, scale_value, plan
    ):
        batch = plan.request_slots.numel()
        rows = batch * plan.query_width
        ends = self.lim_adapter.copy_sfa_metadata(self, plan)
        padded_heads = self.fused_attn_out.shape[1]
        if query_nope.shape[1] < padded_heads:
            padding = padded_heads - query_nope.shape[1]
            query_nope = torch.nn.functional.pad(query_nope, (0, 0, 0, padding))
            query_rope = torch.nn.functional.pad(query_rope, (0, 0, 0, padding))
        output = self.fused_attn_out[:rows]
        layer_idx = layer.layer_id - self.start_layer
        cache = self.device_kv_buffer[layer_idx].view(
            -1, 128, self.head_num, self.head_dim
        )
        fused_scatter_copy_sparse_flash_attention(
            query_nope.contiguous(),
            query_rope.contiguous(),
            cache,
            ends,
            self.planner_actual_seq_lengths_kv[:batch],
            self.planner_num_cache_tokens[:batch],
            plan.source_ids.unsqueeze(1),
            plan.device_slots.unsqueeze(1),
            plan.topk_misses,
            plan.miss_sources,
            plan.miss_slots,
            plan.miss_counts,
            self.planner_hbm_block_table[:batch],
            self.planner_dram_block_table[:batch],
            self.dev_ptr_list[layer_idx],
            output,
            scale_value,
            operator=self._copy_sfa_operator,
            full_visible=True,
            resident_refresh_tail=plan.query_width > 1,
        )
        valid = (
            plan.valid_rows[:, None].expand(batch, plan.query_width).reshape(rows, 1)
        )
        return output[:, :query_heads].reshape(rows, -1).masked_fill(~valid, 0)

    def _install_req_alloc_hook(self, req_to_token_pool: ReqToTokenPool) -> None:
        # Preserve the target manager's hook when adding the draft manager.
        original_alloc = req_to_token_pool.alloc

        def alloc_with_sparse_reset(reqs: list[Req]) -> Optional[List[int]]:
            newly_allocated = [not req.kv.holds_kv for req in reqs]
            req_pool_indices = original_alloc(reqs)
            if req_pool_indices is not None:
                self.reset_requests(
                    [
                        req_pool_indices[i]
                        for i, is_new in enumerate(newly_allocated)
                        if is_new
                    ]
                )
            return req_pool_indices

        setattr(req_to_token_pool, "alloc", alloc_with_sparse_reset)

    def offload_v2(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
        plan=None,
    ):
        """Publish new KV to paged host storage and maintain dense HBM rows."""
        layer_idx = layer.layer_id - self.start_layer
        device = k.device

        # k:         [num_new_tokens, nhead, kv_lora_rank]
        # k_rope:    [num_new_tokens, nhead, qk_rope_head_dim]
        # kv_device: [num_new_tokens, nhead, kv_lora_rank + qk_rope_head_dim]
        src_tensor = torch.cat([k, k_rope], dim=-1).contiguous()
        dst_tensor = self.host_kv_buffer[layer_idx]
        src_index = torch.arange(src_tensor.shape[0], device=device, dtype=torch.long)
        num_src_rows = int(src_tensor.shape[0])

        if not forward_batch.forward_mode.is_extend_without_speculative():
            req_ids, token_pos, valid_mask = prepare_mtp_rows(
                forward_batch, num_src_rows, self.size, self.logical_max_context_len
            )
            token_requests = req_ids.reshape(-1).contiguous()
            token_positions = token_pos.reshape(-1).contiguous()
            valid_mask = valid_mask.reshape(-1).contiguous()
            # TP token padding is a suffix, independent of the request window.
            padding = num_src_rows - token_requests.numel()
            token_requests = torch.nn.functional.pad(
                token_requests, (0, padding), value=0
            )
            token_positions = torch.nn.functional.pad(
                token_positions, (0, padding), value=-1
            )
            valid_mask = torch.nn.functional.pad(valid_mask, (0, padding), value=False)
        else:
            if (
                forward_batch.extend_seq_lens is None
                or forward_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "Sparse v2 prefill offload requires extend_seq_lens and "
                    "extend_prefix_lens in ForwardBatch."
                )

            req_ids = forward_batch.req_pool_indices.to(torch.long)
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            extend_prefix_lens = forward_batch.extend_prefix_lens.to(torch.long)

            batch_size = int(req_ids.shape[0])
            if batch_size <= 0:
                return

            if int(extend_seq_lens.shape[0]) != batch_size:
                raise RuntimeError(
                    "Sparse v2 prefill offload expects extend_seq_lens to be "
                    f"padded to batch size {batch_size}, got "
                    f"{int(extend_seq_lens.shape[0])}."
                )

            prefix_len_size = int(extend_prefix_lens.shape[0])
            if prefix_len_size < batch_size:
                extend_prefix_lens = torch.cat(
                    [
                        extend_prefix_lens,
                        torch.zeros(
                            batch_size - prefix_len_size,
                            device=device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
            elif prefix_len_size > batch_size:
                raise RuntimeError(
                    "Sparse v2 prefill offload expects extend_prefix_lens length "
                    f"<= batch size {batch_size}, got {prefix_len_size}."
                )

            if forward_batch.extend_seq_lens_cpu is not None:
                extend_seq_lens_sum = int(
                    sum(forward_batch.extend_seq_lens_cpu[:batch_size])
                )
            else:
                extend_seq_lens_sum = int(extend_seq_lens.sum().item())
            if extend_seq_lens_sum == num_src_rows:
                # Chunk prefill emits compact rows as [req0 tokens][req1 tokens]...
                # instead of graph-captured padded [B, tokens_per_req] rows.
                seq_starts = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens
                flat_req_ids = torch.repeat_interleave(
                    req_ids, extend_seq_lens, output_size=num_src_rows
                )
                flat_seq_starts = torch.repeat_interleave(
                    seq_starts, extend_seq_lens, output_size=num_src_rows
                )
                flat_prefix_lens = torch.repeat_interleave(
                    extend_prefix_lens, extend_seq_lens, output_size=num_src_rows
                )
                token_pos = (
                    flat_prefix_lens
                    + torch.arange(num_src_rows, device=device, dtype=torch.long)
                    - flat_seq_starts
                )
                token_requests, token_positions = flat_req_ids, token_pos
                valid_mask = (
                    (flat_req_ids > 0)
                    & (token_pos >= 0)
                    & (token_pos < self.max_context_len)
                )
            else:
                if num_src_rows % batch_size != 0:
                    raise RuntimeError(
                        "Sparse v2 prefill offload expects either compact ragged "
                        "layout with rows=sum(extend_seq_lens) or graph-style "
                        f"row-major layout [B, tokens_per_req], got rows={num_src_rows}, "
                        f"batch={batch_size}, extend_seq_lens_sum={extend_seq_lens_sum}."
                    )

                # Graph-friendly static layout:
                # compact rows are interpreted as [batch_size, tokens_per_req].
                # Invalid padded columns are masked by local_offsets < extend_seq_lens.
                tokens_per_req = num_src_rows // batch_size

                local_offsets = (
                    torch.arange(tokens_per_req, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, tokens_per_req)
                )
                req_ids_2d = req_ids.unsqueeze(1).expand(batch_size, tokens_per_req)
                token_pos_2d = extend_prefix_lens.unsqueeze(1) + local_offsets

                token_requests = req_ids_2d.reshape(-1).contiguous()
                token_positions = token_pos_2d.reshape(-1).contiguous()
                valid_mask = (
                    (local_offsets < extend_seq_lens.unsqueeze(1))
                    & (req_ids_2d > 0)
                    & (token_pos_2d >= 0)
                    & (token_pos_2d < self.max_context_len)
                ).reshape(-1)

            if (
                forward_batch.out_cache_loc is not None
                and int(forward_batch.out_cache_loc.numel()) == num_src_rows
            ):
                valid_mask = valid_mask & (
                    forward_batch.out_cache_loc.to(torch.long) >= 0
                )
            valid_mask = valid_mask.contiguous()

        dst_index = forward_batch.out_cache_loc.to(torch.long).reshape(-1)
        valid_mask = (
            valid_mask & (dst_index >= 0) & (dst_index < self.host_token_capacity)
        )
        assert src_tensor.shape[1:] == dst_tensor.shape[2:]
        assert src_index.shape == dst_index.shape == valid_mask.shape

        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                src_tensor,
                dst_tensor,
                src_index,
                dst_index,
                valid_mask,
                1,  # src_tensor: [num_new_tokens, nhead, head_dim]
                2,  # dst_tensor: [num_req, max_context_len, nhead, head_dim]
                block_dim=48,
                dst_ptr=self.dev_ptr_list[layer_idx],
            )

            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_kv_residency import (
                stage_resident_kv,
            )

            width = (
                sparse_query_width(forward_batch)
                if not forward_batch.forward_mode.is_extend_without_speculative()
                else 1
            )
            if plan is not None:
                lengths = plan.seq_lens
                first = forward_batch.positions[
                    : forward_batch.batch_size * width
                ].view(forward_batch.batch_size, width)[:, 0]
            elif not forward_batch.forward_mode.is_extend_without_speculative():
                logical = forward_batch.batch_size * width
                positions = forward_batch.positions[:logical].view(
                    forward_batch.batch_size, width
                )
                lengths = positions[:, -1] + 1
                first = positions[:, 0]
            else:
                lengths = forward_batch.seq_lens
                first = extend_prefix_lens
            current_workspace = self.current_workspace
            if num_src_rows > current_workspace[0].numel():
                # Chunked prefill can exceed decode scratch; these temporary
                # tensors belong to that forward/captured graph's memory pool.
                current_workspace = (
                    torch.empty(num_src_rows, dtype=torch.int64, device=device),
                    torch.empty(num_src_rows, dtype=torch.bool, device=device),
                )
            stage_resident_kv(
                current=src_tensor,
                source_indices=src_index,
                host=dst_tensor,
                host_address=self.dev_ptr_list[layer_idx],
                cache=self.device_kv_buffer[layer_idx],
                req_to_token=self.req_to_token_pool.req_to_token,
                requests=forward_batch.req_pool_indices,
                lengths=lengths,
                first_positions=first,
                token_requests=token_requests.contiguous(),
                token_positions=token_positions.contiguous(),
                token_valid=valid_mask,
                old_begin=self.resident_begin[layer_idx],
                old_end=self.resident_end[layer_idx],
                next_extents=self.next_extents,
                history_workspace=self.history_workspace,
                current_workspace=current_workspace,
                capacity=self.cache_capacity,
                context_len=self.logical_max_context_len,
            )

    def get_forward_kv(
        self,
        layer: Union[RadixAttention, int],
        forward_batch: ForwardBatch,
        stream: Optional[torch.npu.Stream] = None,
    ):
        """Gather full request KV from sparse storage as compact TND tensors.

        This helper is intended for the prefill/extend sparse path. It returns
        KV in the same request order as forward_batch.req_pool_indices:
        [req0 tokens][req1 tokens]... . The SFA caller should pair this with
        actual_seq_lengths_kv = cumsum(forward_batch.seq_lens).

        TODO: Add a prefill-resident device KV cache shaped like
        [max_prefill_parallel_reqs, max_prefill_len, nhead, head_dim]. Reuse a
        slot while the same req_id continues chunked prefill, and fully
        overwrite it when a new req_id takes that slot. This mirrors the decode
        device cache idea and avoids repeatedly copying the full prefix KV from
        host for every chunk.
        """
        layer_id = layer.layer_id if hasattr(layer, "layer_id") else int(layer)
        layer_idx = layer_id - self.start_layer
        if layer_idx < 0 or layer_idx >= self.layer_num:
            raise RuntimeError(
                f"Invalid sparse KV layer id {layer_id}; start_layer="
                f"{self.start_layer}, layer_num={self.layer_num}."
            )

        if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
            raise RuntimeError(
                "get_forward_kv requires req_pool_indices and seq_lens in ForwardBatch."
            )

        device = (
            forward_batch.req_pool_indices.device
            if forward_batch.req_pool_indices.device.type == "npu"
            else self.device
        )
        req_ids = forward_batch.req_pool_indices.to(device=device, dtype=torch.long)
        seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)

        if int(req_ids.numel()) != int(seq_lens.numel()):
            raise RuntimeError(
                "get_forward_kv expects req_pool_indices and seq_lens to have the "
                f"same length, got {int(req_ids.numel())} and "
                f"{int(seq_lens.numel())}."
            )

        keep_cpu, total_tokens = prepare_prefill_gather_rows(
            forward_batch.req_pool_indices_cpu,
            forward_batch.seq_lens_cpu,
            self.size,
            self.max_context_len,
        )
        if forward_batch.req_pool_indices_cpu.numel() != req_ids.numel():
            raise RuntimeError(
                "Prefill CPU and device metadata must have matching sizes"
            )
        if total_tokens == 0:
            empty_nope = torch.empty(
                (0, self.head_num, self.kv_lora_rank),
                dtype=self.store_dtype,
                device=device,
            )
            empty_pe = torch.empty(
                (0, self.head_num, self.qk_rope_head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            return empty_nope, empty_pe

        if keep_cpu.numel() != req_ids.numel():
            keep = keep_cpu.to(device=device, dtype=torch.long, non_blocking=True)
            req_ids = req_ids.index_select(0, keep)
            seq_lens = seq_lens.index_select(0, keep)

        kv_cat = torch.empty(
            (total_tokens, self.head_num, self.head_dim),
            dtype=self.store_dtype,
            device=device,
        )

        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        src_req_ids = torch.repeat_interleave(
            req_ids, seq_lens, output_size=total_tokens
        )
        src_seq_starts = torch.repeat_interleave(
            seq_starts, seq_lens, output_size=total_tokens
        )
        token_pos = (
            torch.arange(total_tokens, device=device, dtype=torch.long) - src_seq_starts
        )
        src_index = self.host_source_indices(src_req_ids, token_pos).contiguous()
        dst_index = torch.arange(total_tokens, device=device, dtype=torch.long)
        valid_mask = torch.ones(total_tokens, device=device, dtype=torch.bool)

        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                kv_cat,
                src_index,
                dst_index,
                valid_mask,
                2,  # host_kv_buffer: [num_req, max_context_len, nhead, head_dim]
                1,  # kv_cat: [total_tokens, nhead, head_dim]
                block_dim=48,
                src_ptr=self.dev_ptr_list[layer_idx],
            )

        k_nope, k_pe = kv_cat.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return k_nope.contiguous(), k_pe.contiguous()
