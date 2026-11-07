from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.kvcache import dsa_index_k_cache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.dsa.utils import (
    INDEXER_K_CACHE_PRESHUFFLE_TILE,
    aiter_can_use_preshuffle_paged_mqa,
)

_use_aiter_preshuffle = aiter_can_use_preshuffle_paged_mqa()

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


class IndexKeyCache:
    def __init__(self, pool: DSATokenToKVPool, index_buf_size: int):
        self.pool = pool
        # FP4 keys: ``buffer`` holds the e2m1 pages and ``scale_buffer`` their exponents,
        # each dense as AITER's FP4 logits address them
        self.fp4 = pool.index_k_fp4
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(pool.custom_mem_pool)
                if pool.custom_mem_pool
                else nullcontext()
            ):
                self.buffer = [
                    torch.zeros(
                        self._buffer_shape(self._layer_num_pages(i, num_pages)),
                        dtype=pool.index_k_with_scale_buffer_dtype,
                        device=pool.device,
                    )
                    for i in range(pool.layer_num)
                ]
                self.scale_buffer = [
                    torch.zeros(
                        (
                            self._layer_num_pages(i, num_pages),
                            pool.page_size
                            // pool.index_kpool
                            * pool.index_head_dim
                            // 32,
                        ),
                        dtype=pool.index_k_with_scale_buffer_dtype,
                        device=pool.device,
                    )
                    for i in range(pool.layer_num if self.fp4 else 0)
                ]

    def _key_bytes_per_token(self) -> int:
        # AMD FP4 keeps the exponents in scale_buffer
        if self.fp4:
            return self.pool.index_head_dim // 2
        return self.pool.index_bytes_per_token

    def _buffer_shape(self, num_pages: int) -> tuple[int, int]:
        pool = self.pool
        return (
            num_pages,
            pool.page_size // pool.index_kpool * self._key_bytes_per_token(),
        )

    def _kernel_view(self, buffer: torch.Tensor) -> torch.Tensor:
        pool = self.pool
        kernel_pages_per_row = (
            pool.page_size // pool.index_kpool // pool.index_page_size
        )
        return buffer.view(-1, buffer.shape[1] // kernel_pages_per_row)

    def _layer_num_pages(self, layer_idx: int, num_pages: int) -> int:
        # Layers that reuse the previous layer's top-k never write index-K, so
        # they get a 0-row placeholder that keeps ``buffer`` layer-aligned.
        return 0 if self.pool.skip_topk_layers[layer_idx] else num_pages

    def clear(self) -> None:
        del self.buffer
        del self.scale_buffer

    def move(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        assert not self.fp4, "Moving FP4 index keys is not supported"
        assert self.pool.index_kpool == 1, (
            "Moving pooled index keys requires regrouping tokens"
        )
        page_size = self.pool.index_page_size
        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        tgt_page, tgt_offset = tgt_loc_flat // page_size, tgt_loc_flat % page_size
        src_page, src_offset = src_loc_flat // page_size, src_loc_flat % page_size
        key_bytes = page_size * self.pool.index_key_bytes_per_token
        for index_k in self.buffer:
            if index_k.shape[0] == 0:
                continue
            index_k = self._kernel_view(index_k)
            # Each physical page stores all key bytes followed by all scale bytes.
            # Locations identify tokens; copying whole pages overwrites neighbors.
            keys = index_k[:, :key_bytes]
            if _use_aiter_preshuffle:
                tile = INDEXER_K_CACHE_PRESHUFFLE_TILE
                keys = keys.view(
                    index_k.shape[0], page_size // tile, -1, tile, tile
                ).permute(0, 1, 3, 2, 4)
                keys[tgt_page, tgt_offset // tile, tgt_offset % tile] = keys[
                    src_page, src_offset // tile, src_offset % tile
                ]
            else:
                keys = keys.view(index_k.shape[0], page_size, -1)
                keys[tgt_page, tgt_offset] = keys[src_page, src_offset]
            scales = index_k[:, key_bytes:].view(index_k.shape[0], page_size, -1)
            scales[tgt_page, tgt_offset] = scales[src_page, src_offset]

    def get_local_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pool.layer_transfer_counter is not None:
            self.pool.layer_transfer_counter.wait_until(
                layer_id - self.pool.start_layer
            )
        return self._kernel_view(self.buffer[layer_id - self.pool.start_layer])

    def get_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_local_buffer(layer_id)

    def get_fp4_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """[pages, 4096] e2m1 bytes and [pages, 256] exponents of the layer's FP4 keys."""
        assert self.fp4
        payload = self.get_local_buffer(layer_id)
        scale = self.scale_buffer[layer_id - self.pool.start_layer]
        return payload, scale.view(
            -1, self.pool.index_page_size * self.pool.index_head_dim // 32
        )

    def get_k_and_scale(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        buf = self.get_buffer(layer_id)
        return dsa_index_k_cache.GetKAndS.execute(
            self.pool,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def store_quantized(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self._kernel_view(self.buffer[layer_id - self.pool.start_layer])
        dsa_index_k_cache.SetKAndS.execute(
            pool=self.pool,
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )

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

    def _page_buffers(self) -> list[torch.Tensor]:
        return [*self.buffer, *self.scale_buffer]

    def _item_len(self, layer_idx: int) -> int:
        # 0-row layers (skip-topk, or non-owned under CP layer split) have no item.
        buf = self.buffer[layer_idx]
        return 0 if buf.shape[0] == 0 else buf[0].nbytes

    def state_buf_infos(self):
        assert not self.fp4, "PD transfer of FP4 index keys is not supported"
        layer_num = self.pool.layer_num
        data_ptrs = [self.buffer[i].data_ptr() for i in range(layer_num)]
        data_lens = [self.buffer[i].nbytes for i in range(layer_num)]
        item_lens = [self._item_len(i) for i in range(layer_num)]
        return data_ptrs, data_lens, item_lens
