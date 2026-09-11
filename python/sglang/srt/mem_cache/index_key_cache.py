from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


class IndexKeyCache:
    def __init__(self, pool: DSATokenToKVPool, index_buf_size: int):
        self.pool = pool
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
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

    def _buffer_shape(self, num_pages: int) -> tuple[int, int]:
        pool = self.pool
        return (
            num_pages,
            pool.page_size
            * (pool.index_head_dim + pool.index_head_dim // pool.quant_block_size * 4),
        )

    def _layer_num_pages(self, layer_idx: int, num_pages: int) -> int:
        # Layers that reuse the previous layer's top-k never write index-K, so
        # they get a 0-row placeholder that keeps ``buffer`` layer-aligned.
        return 0 if self.pool.skip_topk_layers[layer_idx] else num_pages

    def clear(self) -> None:
        del self.buffer

    def move(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return

        # The buffer is PAGE-indexed (dim-0 is the page; a row is a block of fp8 keys
        # then a block of fp32 scales -- see _buffer_shape), but tgt/src are per-TOKEN
        # locations. Indexing dim-0 with them is correct only for page_size == 1, so map
        # each token to its (page, offset) and move both sub-slices.
        pool = self.pool
        ps = pool.page_size
        hd = pool.index_head_dim
        sc = hd // pool.quant_block_size * 4  # scale bytes per token, as _buffer_shape
        tgt = tgt_loc.reshape(-1).long()
        src = src_loc.reshape(-1).long()
        tgt_page, tgt_off = tgt // ps, tgt % ps
        src_page, src_off = src // ps, src % ps
        for index_k in self.buffer:
            num_pages = index_k.shape[0]
            if num_pages == 0:
                continue
            # Page-dim OOB probe, mirroring the token-dim check in
            # MLATokenToKVPool.move_kv_cache: this buffer is page-indexed, so bound
            # the derived page ids by num_pages.
            maybe_detect_oob(
                tgt_page, 0, num_pages, "move_kv_cache tgt_page (DSA index)"
            )
            maybe_detect_oob(
                src_page, 0, num_pages, "move_kv_cache src_page (DSA index)"
            )
            row_stride = index_k.stride(0)
            base = index_k.storage_offset()
            fp8 = index_k.as_strided((num_pages, ps, hd), (row_stride, hd, 1), base)
            scale = index_k.as_strided(
                (num_pages, ps, sc), (row_stride, sc, 1), base + ps * hd
            )
            # RHS advanced-indexing gathers into a fresh tensor before the LHS write,
            # so overlapping src/tgt token locations are safe.
            fp8[tgt_page, tgt_off] = fp8[src_page, src_off]
            scale[tgt_page, tgt_off] = scale[src_page, src_off]

    def get_local_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pool.layer_transfer_counter is not None:
            self.pool.layer_transfer_counter.wait_until(
                layer_id - self.pool.start_layer
            )
        return self.buffer[layer_id - self.pool.start_layer]

    def get_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_local_buffer(layer_id)

    def get_k_continuous(self, layer_id: int, seq_len: int, page_indices: torch.Tensor):
        buf = self.get_buffer(layer_id)
        return index_buf_accessor.GetK.execute(
            self.pool, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_k_scale_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        buf = self.get_buffer(layer_id)
        return index_buf_accessor.GetS.execute(
            self.pool, buf, seq_len=seq_len, page_indices=page_indices
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
        return index_buf_accessor.GetKAndS.execute(
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
        buf = self.buffer[layer_id - self.pool.start_layer]
        index_buf_accessor.SetKAndS.execute(
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
        for layer_id in range(self.pool.layer_num):
            index_k_cpu.append([])
            if self.buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.buffer[layer_id][chunk_page_indices].to(
                    "cpu", non_blocking=True
                )
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()
        return index_k_cpu

    def load_cpu_copy(self, index_k_cpu, indices) -> None:
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for layer_id in range(self.pool.layer_num):
            if self.buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(self.buffer[layer_id].device, non_blocking=True)
                self.buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()

    def _item_len(self, layer_idx: int) -> int:
        # 0-row layers (skip-topk, or non-owned under CP layer split) have no item.
        buf = self.buffer[layer_idx]
        return 0 if buf.shape[0] == 0 else buf[0].nbytes

    def state_buf_infos(self):
        layer_num = self.pool.layer_num
        data_ptrs = [self.buffer[i].data_ptr() for i in range(layer_num)]
        data_lens = [self.buffer[i].nbytes for i in range(layer_num)]
        item_lens = [self._item_len(i) for i in range(layer_num)]
        return data_ptrs, data_lens, item_lens
