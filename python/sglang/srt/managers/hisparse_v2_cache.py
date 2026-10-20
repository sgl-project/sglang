"""DSA indexes addressed by Device/Host KV pages and the HiCache swap-in plane."""

from __future__ import annotations

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    HiSparseSpecState,
    load_cache_to_device_buffer_mla,
    load_cache_to_device_buffer_spec_mla,
)


class HiSparseV2Cache:
    @torch.inference_mode(False)
    def __init__(
        self,
        *,
        req_to_token_pool,
        token_to_kv_pool,
        top_k: int,
        buffer_tokens: int,
        block_size: int,
        device: str,
        num_query_steps: int = 1,
    ):
        self.pool = token_to_kv_pool
        self.req_to_token = req_to_token_pool.req_to_token
        self.page_size = self.pool.page_size
        self.top_k = top_k
        self.buffer_tokens = buffer_tokens
        self.block_size = block_size
        self.device = device
        if not 1 <= num_query_steps <= 4:
            raise ValueError("HiSparse V2 requires 1-4 query steps")
        self.num_query_steps = num_query_steps
        self.spec_enabled = num_query_steps > 1
        workspace = num_query_steps * top_k
        self.scratch_tokens = (
            min(workspace, max(1024, workspace - buffer_tokens))
            if self.spec_enabled
            else 0
        )
        self.reserved_buffer_tokens = buffer_tokens + self.scratch_tokens
        buffer_columns = buffer_tokens + 1
        rows, context = self.req_to_token.shape
        layers = self.pool.layer_num
        if self.spec_enabled and (rows > buffer_tokens or workspace > 8192):
            raise ValueError(
                "HiSparse V2 MTP requires request capacity <= device_buffer_size "
                "and at most 8192 top-k occurrences"
            )
        if buffer_tokens < top_k or buffer_tokens % self.page_size:
            raise ValueError("HiSparse V2 buffer must be page-aligned and cover top_k")
        if buffer_tokens > 32767:
            raise ValueError("HiSparse V2 LRU slots must fit int16")
        self.index_buffers = [
            b for b in self.pool.index_k_with_scale_buffer if b.shape[0] > 0
        ]
        self.base_pages = self.pool.size // self.page_size + 1
        self.index_pages = self.index_buffers[0].shape[0]
        if self.index_pages <= self.base_pages:
            raise ValueError("HiSparse V2 requires expanded indexer storage")
        self.host_locs = torch.full(
            (rows, context), -1, dtype=torch.int64, device=device
        )
        self.protected_lens = torch.zeros(rows, dtype=torch.int64, device=device)
        self.temp_locs = torch.full(
            (layers, rows, buffer_columns), -1, dtype=torch.int32, device=device
        )
        self.residency = torch.full(
            (layers, rows, buffer_columns),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = self.lru = None
        if not self.spec_enabled:
            self._lru_init = torch.arange(
                buffer_tokens, dtype=torch.int16, device=device
            )
            self.lru = self._lru_init.expand(layers, rows, -1).clone()
        self.result = torch.full(
            (rows * num_query_steps, top_k), -1, dtype=torch.int32, device=device
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        # HiCache attaches after decode graphs are captured.
        self.host_bindings = torch.zeros((layers, 2), dtype=torch.int64, device=device)
        self.host_pool = None
        self.spec_states: tuple[HiSparseSpecState, ...] = ()
        self.scratch_locs = torch.empty(
            (layers, rows, 0), dtype=torch.int32, device=device
        )
        if self.spec_enabled:
            hash_size = 1 << max(1, (2 * buffer_tokens - 1).bit_length())
            self._spec_index = torch.full(
                (layers, rows, 2, hash_size), -1, dtype=torch.int64, device=device
            )
            self._spec_policy = torch.zeros(
                (layers, rows + 1, buffer_tokens), dtype=torch.int32, device=device
            )
            self._spec_scratch = torch.full(
                (rows + 1, max(4 * rows, 5 * workspace)),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._spec_scratch[0].zero_()
            self.scratch_locs = torch.full(
                (layers, rows, self.scratch_tokens),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.spec_states = tuple(
                HiSparseSpecState(
                    self._spec_index[layer],
                    self._spec_policy[layer],
                    self.scratch_locs[layer],
                    self._spec_scratch,
                )
                for layer in range(layers)
            )

    def attach_host_pool(self, host_pool, *, layer_offset: int = 0) -> None:
        if self.base_pages + host_pool.size // self.page_size > self.index_pages:
            raise ValueError(
                "HiSparse V2 index expansion must cover the HiCache host pool; "
                "increase expansion_ratio or reduce the host pool size"
            )
        refs = host_pool.data_refs[layer_offset : layer_offset + self.pool.layer_num]
        if len(refs) != self.pool.layer_num:
            raise ValueError("HiSparse V2 host/device layer counts differ")
        bindings = []
        for host, device in zip(refs, self.pool.kv_buffer):
            host_u8, device_u8 = host.view(torch.uint8), device.view(torch.uint8)
            if not host.is_pinned() or host_u8.stride(0) < device_u8.stride(0):
                raise ValueError(
                    "HiSparse V2 needs pinned host rows covering device KV"
                )
            bindings.append((host_u8.data_ptr(), host_u8.stride(0)))
        self.host_bindings.copy_(
            torch.tensor(bindings, dtype=torch.int64, device=self.device)
        )
        self.host_pool = host_pool

    def prepare_forward(self, forward_batch) -> None:
        if self.num_real_reqs.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        self.num_real_reqs.fill_(
            0 if forward_batch.forward_mode.is_idle() else forward_batch.batch_size
        )

    def assign_temp_locations(self, req_id: int, temp: torch.Tensor) -> None:
        self.temp_locs[:, req_id, : self.buffer_tokens] = temp[: self.buffer_tokens].to(
            torch.int32
        )
        if self.spec_enabled:
            self.scratch_locs[:, req_id] = temp[self.buffer_tokens :].to(torch.int32)

    def preserve_index(self, device_tokens: torch.Tensor, host_tokens: torch.Tensor):
        source = device_tokens[:: self.page_size] // self.page_size
        destination = self.base_pages + host_tokens[:: self.page_size] // self.page_size
        # Host locks keep these rows alive for every request sharing the prefix.
        for buffer in self.index_buffers:
            buffer[destination] = buffer[source]

    def release(self, req_id: int) -> None:
        self.protected_lens[req_id] = 0
        self.host_locs[req_id].fill_(-1)
        self.temp_locs[:, req_id].fill_(-1)
        self.residency[:, req_id].fill_(-1)
        if self.spec_enabled:
            rows = self.req_to_token.shape[0]
            self._spec_index[:, req_id].fill_(-1)
            self._spec_policy[:, 0, req_id] = 0
            self._spec_policy[:, req_id + 1].zero_()
            self._spec_scratch[0, req_id : 4 * rows : rows] = 0
            self._spec_scratch[req_id + 1].fill_(-1)
            self.scratch_locs[:, req_id].fill_(-1)
        else:
            self.lru[:, req_id].copy_(self._lru_init)

    def indexer_page_table(
        self, *, req_pool_indices: torch.Tensor, num_pages: int
    ) -> torch.Tensor:
        req_ids = req_pool_indices.long()
        tokens = self.req_to_token[
            req_ids, : num_pages * self.page_size : self.page_size
        ]
        hosts = self.host_locs[req_ids, : num_pages * self.page_size : self.page_size]
        host_pages = torch.where(
            hosts >= 0, self.base_pages + hosts // self.page_size, -1
        )
        pages = torch.where(tokens < 0, host_pages, tokens // self.page_size)
        starts = torch.arange(num_pages, device=self.device) * self.page_size
        protected = starts[None, :] < self.protected_lens[req_ids, None]
        valid = (pages >= 0) & (pages < self.index_pages)
        torch._assert_async(
            (~protected | valid).all(), "HiSparse V2 lost an active index page"
        )
        return torch.where(valid, pages, 0).to(torch.int32)

    def swap_in_selected_pages(
        self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id
    ):
        local_layer = layer_id - self.pool.start_layer
        num_reqs = req_pool_indices.numel()
        num_queries = top_k_result.shape[0]
        num_query_steps = num_queries // num_reqs if num_reqs else 1
        if num_query_steps > self.num_query_steps:
            raise ValueError("HiSparse V2 query count exceeds its reserved workspace")
        result = self.result[:num_queries]
        device_buffer = self.pool.kv_buffer[local_layer]
        if self.spec_enabled:
            load_cache_to_device_buffer_spec_mla(
                top_k_tokens=top_k_result.reshape(
                    num_reqs, num_query_steps, self.top_k
                ),
                device_buffer_tokens=self.residency[local_layer],
                host_cache_locs=self.host_locs,
                device_buffer_locs=self.temp_locs[local_layer],
                host_cache=None,
                device_buffer=device_buffer,
                top_k_device_locs=result.view(num_reqs, num_query_steps, self.top_k),
                req_pool_indices=req_pool_indices.to(torch.int64),
                seq_lens=compressed_seq_lens.to(torch.int32),
                state=self.spec_states[local_layer],
                num_real_reqs=self.num_real_reqs,
                device_locs=self.req_to_token,
                host_binding=self.host_bindings[local_layer],
            )
            return result
        load_cache_to_device_buffer_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.residency[local_layer],
            host_cache_locs=self.host_locs,
            device_buffer_locs=self.temp_locs[local_layer],
            host_cache=None,
            device_buffer=device_buffer,
            top_k_device_locs=result,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru[local_layer],
            item_size_bytes=device_buffer.stride(0) * device_buffer.element_size(),
            num_top_k=self.top_k,
            hot_buffer_size=self.buffer_tokens,
            page_size=1,
            block_size=self.block_size,
            num_real_reqs=self.num_real_reqs,
            device_locs=self.req_to_token,
            host_binding=self.host_bindings[local_layer],
        )
        return result
