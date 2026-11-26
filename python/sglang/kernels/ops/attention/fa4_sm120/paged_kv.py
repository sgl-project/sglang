import math
from dataclasses import dataclass
from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from quack.cute_dsl_utils import ParamsBase

from sglang.kernels.ops.attention.flash_attn.cute import utils


@dataclass
class Sm120PagedKVManager(ParamsBase):
    """SM120 paged-KV loader for the stage-sliced FA4 pipeline."""

    mPageTable: cute.Tensor
    mK_paged: cute.Tensor
    mV_paged: cute.Tensor
    thread_idx: Int32

    page_size_divmod: FastDivmodDivisor
    seqlen_k: Int32
    leftpad_k: Int32
    n_block_size: cutlass.Constexpr[Int32]
    num_threads: cutlass.Constexpr[Int32]
    head_dim_padded: cutlass.Constexpr[Int32]
    head_dim_v_padded: cutlass.Constexpr[Int32]

    gmem_threads_per_row: cutlass.Constexpr[Int32]
    page_entry_per_thread: cutlass.Constexpr[Int32]
    async_copy_elems: cutlass.Constexpr[Int32]

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy
    tPrPage: cute.Tensor
    tPrPageOffset: cute.Tensor

    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        mK_paged: cute.Tensor,
        mV_paged: cute.Tensor,
        page_size_divmod: FastDivmodDivisor,
        bidb: Int32,
        bidh: Int32,
        thread_idx: Int32,
        seqlen_k: Int32,
        leftpad_k: Int32,
        n_block_size: cutlass.Constexpr[Int32],
        head_dim_padded: cutlass.Constexpr[Int32],
        head_dim_v_padded: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: Type[cutlass.Numeric],
    ):
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        dtype_bytes = dtype.width // 8
        gmem_k_block_size = math.gcd(
            head_dim_padded,
            head_dim_v_padded,
            128 // dtype_bytes,
        )
        assert gmem_k_block_size % async_copy_elems == 0
        gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        assert cute.arch.WARP_SIZE % gmem_threads_per_row == 0

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(
            atom_async_copy, thr_layout, val_layout
        )
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)

        # SM120 decode tiles can have fewer rows than DMA threads. Keep one
        # register entry per thread so those shapes do not create zero-sized
        # register tensors.
        page_entry_per_thread = max(1, (n_block_size + num_threads - 1) // num_threads)
        tPrPage = cute.make_rmem_tensor((page_entry_per_thread,), Int32)
        tPrPageOffset = cute.make_rmem_tensor((page_entry_per_thread,), Int32)

        return Sm120PagedKVManager(
            mPageTable[bidb, None],
            mK_paged[None, None, bidh, None],
            mV_paged[None, None, bidh, None],
            thread_idx,
            page_size_divmod,
            seqlen_k,
            leftpad_k,
            n_block_size,
            num_threads,
            head_dim_padded,
            head_dim_v_padded,
            gmem_threads_per_row,
            page_entry_per_thread,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            tPrPage,
            tPrPageOffset,
        )

    @cute.jit
    def _load_page_table_entry(self, i: Int32, n_block: Int32):
        row = (
            i * self.num_threads
            + (self.thread_idx % self.gmem_threads_per_row)
            * (self.num_threads // self.gmem_threads_per_row)
            + (self.thread_idx // self.gmem_threads_per_row)
        )
        row_idx = n_block * self.n_block_size + row
        page_idx, page_offset = divmod(row_idx + self.leftpad_k, self.page_size_divmod)
        is_valid = (
            (i + 1) * self.num_threads <= self.n_block_size or row < self.n_block_size
        ) and row_idx < self.seqlen_k
        page = self.mPageTable[page_idx] if is_valid else 0
        self.tPrPage[i] = page
        self.tPrPageOffset[i] = page_offset

    @cute.jit
    def load_page_table(self, n_block: Int32):
        # The entry count is a specialization constant for SM120. Expanding
        # this small loop removes a measurable dynamic-loop cost in decode.
        for i in cutlass.range_constexpr(self.page_entry_per_thread):
            self._load_page_table_entry(i, n_block)

    @cute.jit
    def compute_X_ptr(self, K_or_V: str):
        tPrXPtr = cute.make_rmem_tensor((self.page_entry_per_thread,), cutlass.Int64)
        mX = self.mK_paged if const_expr(K_or_V == "K") else self.mV_paged
        for i in cutlass.range_constexpr(self.page_entry_per_thread):
            page = self.tPrPage[i]
            page_offset = self.tPrPageOffset[i]
            # SGLang stores both paged K and paged V as
            # (page_size, head_dim, num_pages).
            tPrXPtr[i] = utils.elem_pointer(mX, (page_offset, 0, page)).toint()
        return tPrXPtr

    @cute.jit
    def _copy_row_async(
        self,
        tXsX: cute.Tensor,
        tXcX: cute.Tensor,
        mX_paged_cur_copy: cute.Tensor,
        m: Int32,
        should_load: cute.Tensor,
    ):
        for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
            ki = tXcX[0, 0, k][1] // self.async_copy_elems
            mX_paged_cur_copy_ki = mX_paged_cur_copy[None, ki]
            tXsX_k = tXsX[None, m, k]
            mX_paged_cur_copy_ki = cute.make_tensor(
                mX_paged_cur_copy_ki.iterator, tXsX_k.layout
            )
            cute.copy(
                self.gmem_tiled_copy_KV,
                mX_paged_cur_copy_ki,
                tXsX_k,
                pred=should_load,
            )

    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, K_or_V: str):
        assert K_or_V in ("K", "V")

        tPrXPtr = self.compute_X_ptr(K_or_V)

        # The SM120 pipeline passes one stage at a time. V has already been
        # transposed by the caller's shared-memory view.
        sX_pi = cute.group_modes(sX, 0, 1)
        head_dim = (
            self.head_dim_v_padded
            if const_expr(K_or_V == "V")
            else self.head_dim_padded
        )
        cX = cute.make_identity_tensor((self.n_block_size, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_pi)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)
        tXc0X = self.gmem_thr_copy_KV.get_slice(0).partition_S(cX)

        seqlenk_row_limit = (
            self.seqlen_k - n_block * self.n_block_size - tXcX[0][0]
            if n_block >= 0
            else 0
        )
        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_valid = tXc0X[0, m, 0][0] < seqlenk_row_limit
            should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)

            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // self.gmem_threads_per_row],
                m % self.gmem_threads_per_row,
                width=self.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                self.mK_paged.element_type,
                x_ptr_i64,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            mX_paged_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_paged_cur_copy = cute.tiled_divide(
                mX_paged_cur, (self.async_copy_elems,)
            )
            self._copy_row_async(tXsX, tXcX, mX_paged_cur_copy, m, should_load)
