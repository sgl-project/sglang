"""MXFP4 DSA index keys (--enable-dsa-fp4-indexer, ROCm gfx950)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


class Fp4IndexKeyCache(IndexKeyCache):
    """``buffer`` holds the e2m1 pages and ``scale_buffer`` their ue8m0 exponents,
    each dense as AITER's FP4 logits address them, on the same page ids."""

    def __init__(self, pool: DSATokenToKVPool, index_buf_size: int):
        super().__init__(pool, index_buf_size)
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        scale_bytes = pool.page_size // pool.index_kpool * pool.index_head_dim // 32
        with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(pool.custom_mem_pool)
                if pool.custom_mem_pool
                else nullcontext()
            ):
                self.scale_buffer = [
                    torch.zeros(
                        (self._layer_num_pages(i, num_pages), scale_bytes),
                        dtype=pool.index_k_with_scale_buffer_dtype,
                        device=pool.device,
                    )
                    for i in range(pool.layer_num)
                ]

    def _buffer_shape(self, num_pages: int) -> tuple[int, int]:
        # 64 e2m1 bytes a 128-dim key; the exponents live in scale_buffer
        pool = self.pool
        return (
            num_pages,
            pool.page_size // pool.index_kpool * (pool.index_head_dim // 2),
        )

    def clear(self) -> None:
        super().clear()
        del self.scale_buffer

    def move(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        assert tgt_loc.numel() == 0, "Moving FP4 index keys is not supported"

    def get_fp4_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """[pages, 4096] e2m1 bytes and [pages, 256] exponents of the layer's FP4 keys."""
        payload = self.get_local_buffer(layer_id)
        scale = self.scale_buffer[layer_id - self.pool.start_layer]
        return payload, scale.view(
            -1, self.pool.index_page_size * self.pool.index_head_dim // 32
        )

    def _page_buffers(self) -> list[torch.Tensor]:
        # every layer's key pages, then every layer's exponent pages
        return [*self.buffer, *self.scale_buffer]

    def cpu_copy(self, indices):
        # Retracted pages may be reused before resume, so offload index-K with KV.
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for buffer in self._page_buffers():
            index_k_cpu.append([])
            if buffer.shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = buffer[chunk_page_indices].to("cpu", non_blocking=True)
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()
        return index_k_cpu

    def load_cpu_copy(self, index_k_cpu, indices) -> None:
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for buffer_id, buffer in enumerate(self._page_buffers()):
            if buffer.shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[buffer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                buffer[chunk_page_indices] = idx_cpu.to(
                    buffer.device, non_blocking=True
                )
        torch.cuda.synchronize()

    def state_buf_infos(self):
        # Both PD peers register the same order, so pages move with the generic
        # page path; a 0-row layer (skip-topk) has no item.
        buffers = self._page_buffers()
        data_ptrs = [buf.data_ptr() for buf in buffers]
        data_lens = [buf.nbytes for buf in buffers]
        item_lens = [0 if buf.shape[0] == 0 else buf[0].nbytes for buf in buffers]
        return data_ptrs, data_lens, item_lens
