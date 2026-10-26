"""RMSNorm over the whole hidden size followed by interleaved RoPE, in one pass.

Wan's self-attention normalizes q and k across all heads (``rms_norm_across_heads``)
and then rotates (even, odd) pairs of every head. The norm pass below is FlashInfer's
CuTe-DSL ``RMSNormKernel`` (single-CTA rows), with the same tiling and reduction, so
its bf16 output matches ``flashinfer.norm.rmsnorm``; the store then applies the
interleaved rotation of FlashInfer's ``BatchQKApplyRotaryPosIdsCosSinCache`` kernel
(``cos_sin_cache`` row = position, cos then sin) to that bf16 output in fp32.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.utils import (
    COPY_BITS,
    predicate_k,
    row_reduce_sum_multirow,
)

_COMPILE_CACHE = {}
_TORCH_TO_CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


def _threads_per_row(H: int) -> int:
    # FlashInfer RMSNormKernel._compute_threads_per_row
    if H <= 64:
        return 8
    if H <= 128:
        return 16
    if H <= 3072:
        return 32
    if H <= 6144:
        return 64
    if H <= 16384:
        return 128
    return 256


class RMSNormInterleavedRope:
    def __init__(self, dtype: cutlass.Numeric, H: int, head_dim: int):
        self.dtype = dtype
        self.H = H
        self.head_dim = head_dim
        elem_bytes = dtype.width // 8
        self.vec_size = min(H & (-H), COPY_BITS // 8 // elem_bytes)
        self.copy_bits = self.vec_size * dtype.width
        self.threads_per_row = _threads_per_row(H)
        self.num_threads = 128 if H <= 16384 else 256
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)
        self.num_vec_blocks = max(
            1,
            (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row
        tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.use_async_copy = tile_bytes <= props.shared_memory_per_block_optin // 2

    def _smem_size_in_bytes(self) -> int:
        tile_bytes = (
            self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            if self.use_async_copy
            else 0
        )
        return tile_bytes + self.rows_per_block * self.warps_per_row * 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mCS: cute.Tensor,
        M: Int64,
        seq_len: Int64,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        tv_shape = (
            (self.threads_per_row, self.rows_per_block),
            (self.vec_size, self.num_vec_blocks),
        )
        tv_stride = (
            (self.vec_size * self.rows_per_block, 1),
            (
                self.rows_per_block,
                self.rows_per_block * self.vec_size * self.threads_per_row,
            ),
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(mX, mW, mY, mCS, M, seq_len, eps, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mCS: cute.Tensor,
        M: Int64,
        seq_len: Int64,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H = self.H
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        half = self.head_dim // 2
        pairs = self.vec_size // 2

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))
        mW_2d = cute.make_tensor(
            mW.iterator,
            cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
        )
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=self.copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=self.copy_bits,
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=self.copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)
        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # 1. load the row (and the weight)
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        # this thread's 8-column runs sit at the same offset of every head (the runs are
        # one head-multiple apart), so one 4-pair cos / sin slice serves all of them
        pair0 = (tXcX[(0, 0), 0, 0][1] % self.head_dim) // 2
        tCS = cute.make_rmem_tensor(cute.make_layout((self.vec_size,)), Float32)
        tCS.fill(0.0)
        if row_in_bounds:
            pos = Int64(row_coord[0]) % seq_len
            gC = cute.local_tile(mCS, (1, pairs), (pos, pair0 // pairs))
            gS = cute.local_tile(mCS, (1, pairs), (pos, (half + pair0) // pairs))
            tC = cute.make_rmem_tensor(cute.make_layout((1, pairs)), Float32)
            tS = cute.make_rmem_tensor(cute.make_layout((1, pairs)), Float32)
            cute.autovec_copy(gC, tC)
            cute.autovec_copy(gS, tS)
            for i in cutlass.range_constexpr(pairs):
                tCS[i] = tC[0, i]
                tCS[pairs + i] = tS[0, i]

        # 2. RMSNorm, as FlashInfer's RMSNormKernel
        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, None, 1
        )
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
        cute.arch.barrier()
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)
        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(0.0))
        tXrO.store(y.to(mY.element_type))

        # 3. rotate each (even, odd) pair of the rounded output in fp32
        for blk in cutlass.range_constexpr(self.num_vec_blocks):
            for v in cutlass.range_constexpr(0, self.vec_size, 2):
                a = tXrO[(v, blk), 0, 0].to(Float32)
                b = tXrO[(v + 1, blk), 0, 0].to(Float32)
                c = tCS[v // 2]
                s = tCS[pairs + v // 2]
                tXrO[(v, blk), 0, 0] = (a * c + (-b) * s).to(mY.element_type)
                tXrO[(v + 1, blk), 0, 0] = (b * c + a * s).to(mY.element_type)

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


def can_use_fused_rmsnorm_interleaved_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
) -> bool:
    """Whether ``fused_rmsnorm_interleaved_rope_`` takes ``x`` [M, H]."""
    H = x.shape[-1]
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype in _TORCH_TO_CUTE
        and weight.dtype == x.dtype
        and weight.shape == (H,)
        and weight.is_contiguous()
        and x.stride(-1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and H % 8 == 0
        and H <= 6144
        and (_threads_per_row(H) * 8) % head_dim == 0
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[-1] == head_dim
        and cos_sin_cache.is_contiguous()
        and x.shape[0] % cos_sin_cache.shape[0] == 0
    )


def fused_rmsnorm_interleaved_rope_(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    head_dim: int,
) -> torch.Tensor:
    """In place: x [M, H] <- RoPE(RMSNorm(x) * weight), row m at position m % len(cos_sin_cache).

    ``cos_sin_cache`` is [seq_len, head_dim] fp32 (cos | sin of each pair); bit-identical
    to ``flashinfer.norm.rmsnorm`` then FlashInfer's interleaved cos/sin-cache RoPE.
    """
    M, H = x.shape
    key = (x.dtype, H, head_dim)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        dtype = _TORCH_TO_CUTE[x.dtype]
        sym_m = cute.sym_int(64)
        x_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, H), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
        cs_fake = cute.runtime.make_fake_compact_tensor(
            Float32, (cute.sym_int(64), head_dim), stride_order=(1, 0), assumed_align=16
        )
        compiled = _COMPILE_CACHE[key] = cute.compile(
            RMSNormInterleavedRope(dtype, H, head_dim),
            x_fake,
            w_fake,
            x_fake,
            cs_fake,
            Int64(1),
            Int64(1),
            Float32(1e-6),
            cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
            options="--enable-tvm-ffi",
        )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(x, weight, x, cos_sin_cache, M, cos_sin_cache.shape[0], eps, stream)
    return x
