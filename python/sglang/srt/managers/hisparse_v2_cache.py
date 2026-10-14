"""DSA index copies and the shared HiCache swap-in data plane."""

from __future__ import annotations

import torch

from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla


class IndexPageAllocator:
    def __init__(self, num_pages: int, device: str):
        self._free = list(range(num_pages))
        self._allocated: set[int] = set()
        self.device = device

    def available(self) -> int:
        return len(self._free)

    def alloc(self, count: int) -> torch.Tensor | None:
        if count <= 0:
            raise ValueError("Index page allocation must be positive")
        if count > len(self._free):
            return None
        pages = self._free[-count:]
        del self._free[-count:]
        self._allocated.update(pages)
        return torch.tensor(pages, dtype=torch.int64, device=self.device)

    def free(self, pages: torch.Tensor) -> None:
        ids = pages.tolist()
        if len(set(ids)) != len(ids) or not set(ids) <= self._allocated:
            raise RuntimeError(
                "HiSparse V2 index pages freed twice or by another owner"
            )
        self._allocated.difference_update(ids)
        self._free.extend(ids)


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
    ):
        self.pool = token_to_kv_pool
        self.req_to_token = req_to_token_pool.req_to_token
        self.page_size = self.pool.page_size
        self.top_k = top_k
        self.buffer_tokens = buffer_tokens
        self.block_size = block_size
        self.device = device
        rows, context = self.req_to_token.shape
        layers = self.pool.layer_num
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
        self.index_allocator = IndexPageAllocator(
            self.index_pages - self.base_pages, device
        )
        self.index_table = torch.full(
            (rows, (context + self.page_size - 1) // self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.host_locs = torch.full(
            (rows, context), -1, dtype=torch.int64, device=device
        )
        self.protected_lens = torch.zeros(rows, dtype=torch.int64, device=device)
        self.temp_locs = torch.full(
            (rows, buffer_tokens + 1), -1, dtype=torch.int32, device=device
        )
        self.residency = torch.full(
            (layers, rows, buffer_tokens + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(buffer_tokens, dtype=torch.int16, device=device)
        self.lru = self._lru_init.expand(layers, rows, -1).clone()
        self.result = torch.full((rows, top_k), -1, dtype=torch.int32, device=device)
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        # HiCache attaches after decode graphs are captured.
        self.host_bindings = torch.zeros((layers, 2), dtype=torch.int64, device=device)
        self.host_pool = None

    def attach_host_pool(self, host_pool) -> None:
        refs = host_pool.data_refs
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

    def copy_index(self, req_id: int, tokens: torch.Tensor) -> bool:
        count = tokens.numel() // self.page_size
        pages = self.index_allocator.alloc(count)
        if pages is None:
            return False
        destination = pages + self.base_pages
        source = (tokens[:: self.page_size] // self.page_size).long()
        try:
            # A prefix loaded from HiCache may still be arriving on its IO stream.
            for layer_id in range(
                self.pool.start_layer, self.pool.start_layer + self.pool.layer_num
            ):
                buffer = self.pool.get_index_k_with_scale_buffer(layer_id)
                if buffer.shape[0]:
                    buffer[destination] = buffer[source]
            self.index_table[req_id, :count] = destination
        except Exception:
            self.index_allocator.free(pages)
            raise
        return True

    def release(self, req_id: int) -> None:
        row = self.index_table[req_id]
        pages = row[row >= 0]
        if pages.numel():
            self.index_allocator.free(pages - self.base_pages)
        row.fill_(-1)
        self.protected_lens[req_id] = 0
        self.host_locs[req_id].fill_(-1)
        self.temp_locs[req_id].fill_(-1)
        self.residency[:, req_id].fill_(-1)
        self.lru[:, req_id].copy_(self._lru_init)

    def indexer_page_table(
        self, *, req_pool_indices: torch.Tensor, num_pages: int
    ) -> torch.Tensor:
        req_ids = req_pool_indices.long()
        tokens = self.req_to_token[
            req_ids, : num_pages * self.page_size : self.page_size
        ]
        copied = self.index_table[req_ids, :num_pages]
        pages = torch.where(tokens < 0, copied, tokens // self.page_size)
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
        result = self.result[: req_pool_indices.numel()]
        device_buffer = self.pool.kv_buffer[local_layer]
        load_cache_to_device_buffer_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.residency[local_layer],
            host_cache_locs=self.host_locs,
            device_buffer_locs=self.temp_locs,
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
