# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from quack.cute_dsl_utils import ParamsBase, torch2cute_dtype_map
from quack.tile_scheduler import RasterOrderOption, TileSchedulerArguments

from sglang.srt.layers.moe.sonic_moe.utils import get_next_power_of_2, get_powers_of_2

from .tile_scheduler import SonicMoETileScheduler


def last_even(a: int):
    return a if a % 2 == 0 else a - 1


class Token_gather_and_sum_over_k:
    def __init__(
        self,
        x_dtype: cutlass.Numeric | torch.dtype,
        w_dtype: cutlass.Numeric | torch.dtype | None,
        N: int,
        MAX_K: int,
        K_STAGED: int,
        is_persistent=False,
        is_w_gathered=False,
    ):  # we can set it as `E`, or a runtime number
        if type(x_dtype) == torch.dtype:
            x_dtype = torch2cute_dtype_map[x_dtype]
        if w_dtype is not None and type(w_dtype) == torch.dtype:
            w_dtype = torch2cute_dtype_map[w_dtype]
        self.x_dtype = x_dtype
        self.w_dtype = w_dtype

        self.is_persistent = is_persistent
        # if w is gathered, w will follow the same fetching order as x (which is gathered).
        #   otherwise, we will do sequential scan over w
        self.is_w_gathered = is_w_gathered

        self.REG_LIMIT = 255
        self.SMEM_LIMIT = cutlass.utils.get_smem_capacity_in_bytes("sm_90")

        # register pressure
        assert N % 8 == 0
        if N % 256 == 0:
            self.no_padding_on_row_size = True
        else:
            self.no_padding_on_row_size = False

        # only optimize for K <= 128.
        #   K=256 will have register pressure and require another impl.
        self.universal_copy_bits = 128
        self.universal_copy_bytes = self.universal_copy_bits // 8

        MAX_K = int(math.ceil(MAX_K / K_STAGED)) * K_STAGED
        for t in range(1024, 0, -2 * cute.arch.WARP_SIZE):
            if t * K_STAGED * self.universal_copy_bytes <= self.SMEM_LIMIT - MAX_K * 6:
                self.num_threads = t
                break
        else:
            raise RuntimeError(f"K {MAX_K} is too large")

        self.copy_elems = self.universal_copy_bits // x_dtype.width

        self.N = N
        self.MAX_K = MAX_K
        self.K_STAGED = K_STAGED

        handled_rows_per_block, thread_per_row, handled_rows_size = (
            self._calculate_launch_settings()
        )

        self.smem_size = (
            handled_rows_per_block * handled_rows_size * K_STAGED
        ) * x_dtype.width // 8 + (handled_rows_per_block * MAX_K) * (
            (w_dtype.width if w_dtype is not None else 0) // 8 + 4
        )

        assert self.smem_size <= self.SMEM_LIMIT

        self.handled_rows_per_block = handled_rows_per_block
        self.thread_per_row = thread_per_row
        self.handled_rows_size = handled_rows_size

    def _make_tiled_copy_2D(
        self,
        copy_atom: cute.CopyAtom,
    ) -> cute.TiledCopy:
        thread_layout = cute.make_layout(
            (self.num_threads // self.thread_per_row, self.thread_per_row),
            stride=(self.thread_per_row, 1),
        )

        # Assume all K are loaded now
        value_layout = cute.make_layout((1, self.copy_elems))

        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    def _make_tiled_copy_3D(
        self,
        copy_atom: cute.CopyAtom,
    ) -> cute.TiledCopy:
        # thread layout for copy
        thread_layout = cute.make_layout(
            (self.num_threads // self.thread_per_row, self.thread_per_row, 1),
            stride=(self.thread_per_row, 1, 0),
        )

        # Value layout for copy
        value_layout = cute.make_layout(
            (1, self.copy_elems, self.K_STAGED), stride=(0, 1, self.copy_elems)
        )

        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    def _calculate_launch_settings(self):
        N, K_STAGED = self.N, self.K_STAGED

        thread_per_row = min(self.num_threads, int(math.ceil(N / self.copy_elems)))

        for t in range(get_next_power_of_2(thread_per_row), 0, -8):
            if (
                2 * self.copy_elems * t + 2 * t * K_STAGED
                <= min(t * self.REG_LIMIT, 65536)
                and self.num_threads % t == 0
                and t <= self.num_threads
            ):
                if self.no_padding_on_row_size:
                    handled_row_size = (
                        (
                            self.SMEM_LIMIT
                            // (self.num_threads * K_STAGED * self.universal_copy_bytes)
                        )
                        * t
                        * self.copy_elems
                    )
                    if self.N % handled_row_size == 0:
                        thread_per_row = t
                        break
                else:
                    thread_per_row = t
                    break
        else:
            raise RuntimeError(f"cannot find appropriate thread_per_row")

        handled_rows_per_block = self.num_threads // thread_per_row

        # smem caches gX and gW. Assumed gW is either FP32 or BF16
        handled_row_size = (
            (
                self.SMEM_LIMIT
                // (self.num_threads * K_STAGED * self.universal_copy_bytes)
            )
            * thread_per_row
            * self.copy_elems
        )
        handled_row_size = int(min(handled_row_size, N))

        assert handled_row_size > 0

        return handled_rows_per_block, thread_per_row, handled_row_size

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mO: cute.Tensor,
        mMoffset: cute.Tensor,
        mXIdx: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.x_dtype
        assert mO.element_type == self.x_dtype

        X_g2s_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.NONE
            ),
            mX.element_type,
            num_bits_per_copy=self.universal_copy_bits,
        )

        O_r2g_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=self.universal_copy_bits,
        )

        X_tiled_copy = self._make_tiled_copy_2D(X_g2s_copy_atom)

        O_tiled_copy = self._make_tiled_copy_2D(
            O_r2g_copy_atom,
        )

        problem_shape_ntile_mnl = (
            cute.ceil_div(mO.shape[0], self.handled_rows_per_block),
            1,
            1,
        )
        TileScheduler = SonicMoETileScheduler
        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=problem_shape_ntile_mnl,
            raster_order=RasterOrderOption.Heuristic,
            group_size=8,
            cluster_shape_mnk=(1, 1, 1),
            is_persistent=self.is_persistent,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid = TileScheduler.get_grid_shape(tile_sched_params, 1)

        self.kernel(
            mX,
            mW,
            mO,
            mMoffset,
            mXIdx,
            X_tiled_copy,
            O_tiled_copy,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            cluster=None,
            smem=self.smem_size,
            stream=stream,
        )

    @dsl_user_op
    def fma(
        self,
        a: float | Float32,
        b: float | Float32,
        c: float | Float32,
        *,
        loc=None,
        ip=None,
    ) -> Float32:
        return Float32(
            llvm.inline_asm(
                cutlass.cutlass_dsl.T.f32(),
                [
                    Float32(a).ir_value(loc=loc, ip=ip),
                    Float32(b).ir_value(loc=loc, ip=ip),
                    Float32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def elem_pointer(
        self, x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
    ) -> cute.Pointer:
        return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mO: cute.Tensor,
        mMoffset: cute.Tensor,
        mXIdx: cute.Tensor,
        X_tiled_copy: cute.core.TiledCopy,
        O_tiled_copy: cute.core.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        MAX_M = mO.shape[0]
        N, MAX_K, K_STAGED = self.N, self.MAX_K, self.K_STAGED

        handled_rows_per_block, thread_per_row, handled_rows_size = (
            const_expr(self.handled_rows_per_block),
            const_expr(self.thread_per_row),
            const_expr(self.handled_rows_size),
        )

        row_id_within_block = tidx // thread_per_row
        col_id_within_row = tidx % thread_per_row * self.copy_elems

        idO = cute.make_identity_tensor(mO.shape)

        smem = cutlass.utils.SmemAllocator()
        sX_slice = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(
                (handled_rows_per_block, handled_rows_size, K_STAGED), order=(1, 0, 2)
            ),
            byte_alignment=16,
        )
        sW = None
        if const_expr(mW is not None):
            sW = smem.allocate_tensor(
                mW.element_type,
                cute.make_ordered_layout((handled_rows_per_block, MAX_K), order=(1, 0)),
            )
        sXIdx = smem.allocate_tensor(
            mXIdx.element_type,
            cute.make_ordered_layout((handled_rows_per_block, MAX_K), order=(1, 0)),
        )

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        # !!! This impl assumes 1 thread will cover all K for 1 slice of row.
        while work_tile.is_valid_tile:
            tile_coord_mnkl = work_tile.tile_idx
            M_idx, _, _, _ = tile_coord_mnkl

            row_offset = handled_rows_per_block * M_idx

            cO = cute.local_tile(
                idO, (handled_rows_per_block, handled_rows_size), (M_idx, 0)
            )

            X_thr_copy = X_tiled_copy.get_slice(tidx)
            O_thr_copy = O_tiled_copy.get_slice(tidx)

            tOcO = O_thr_copy.partition_S(cO)[(None, None), 0, 0]

            if tOcO[0, 0][0] < MAX_M:
                tMIdx_s, tMIdx_e = (
                    mMoffset[row_offset + row_id_within_block],
                    mMoffset[row_offset + row_id_within_block + 1],
                )
                K = tMIdx_e - tMIdx_s

                for i in cutlass.range(
                    cute.ceil_div(K, thread_per_row), unroll_full=True
                ):
                    j = tidx % thread_per_row
                    fetched_idx = j + i * thread_per_row

                    if fetched_idx < K:
                        k_idx = mXIdx[tMIdx_s + fetched_idx]
                        sXIdx[row_id_within_block, fetched_idx] = k_idx
                        if const_expr(self.is_w_gathered and mW is not None):
                            sW[row_id_within_block, fetched_idx] = mW[k_idx]

                        if const_expr(
                            mW is not None
                        ):  # otherwise, we do sequential scan
                            sW[row_id_within_block, fetched_idx] = mW[
                                tMIdx_s + fetched_idx
                            ]

                # cannot be removed for correctness!
                cute.arch.sync_threads()

                if const_expr(mW is not None):
                    rW = cute.make_fragment((self.K_STAGED,), dtype=mW.element_type)
                rXIdx = cute.make_fragment((self.K_STAGED,), dtype=mXIdx.element_type)
                rX = cute.make_fragment(self.copy_elems, dtype=mX.element_type)

                for N_idx in cutlass.range_constexpr(
                    cute.ceil_div(self.N, handled_rows_size)
                ):
                    gO = cute.local_tile(
                        mO, (handled_rows_per_block, handled_rows_size), (M_idx, N_idx)
                    )

                    tOgO = O_thr_copy.partition_D(gO)
                    tOrO = cute.make_fragment_like(tOgO)

                    y = cute.make_fragment_like(tOrO, dtype=cutlass.Float32)
                    y.fill(0.0)

                    tNIdx = N_idx * handled_rows_size + col_id_within_row

                    if const_expr(self.no_padding_on_row_size) or tNIdx < N:
                        for k in cutlass.range(cute.ceil_div(K, K_STAGED), unroll=1):
                            if const_expr(mW is not None):
                                sW_slice = cute.make_tensor(
                                    self.elem_pointer(
                                        sW, (row_id_within_block, k * K_STAGED)
                                    ).align(K_STAGED),
                                    (K_STAGED,),
                                )
                            sXIdx_slice = cute.make_tensor(
                                self.elem_pointer(
                                    sXIdx, (row_id_within_block, k * K_STAGED)
                                ).align(K_STAGED),
                                (K_STAGED,),
                            )

                            if const_expr(mW is not None):
                                cute.autovec_copy(sW_slice, rW)
                            cute.autovec_copy(sXIdx_slice, rXIdx)

                            for i in cutlass.range_constexpr(K_STAGED):
                                k_idx = k * K_STAGED + i
                                if k_idx < K:
                                    tMIdx = rXIdx[i]

                                    XgPtr = self.elem_pointer(mX, (tMIdx, tNIdx)).align(
                                        self.universal_copy_bits // self.copy_elems
                                    )
                                    XsPtr = self.elem_pointer(
                                        sX_slice,
                                        (row_id_within_block, col_id_within_row, i),
                                    ).align(self.universal_copy_bits // self.copy_elems)

                                    Xg_thr_slice = cute.make_tensor(
                                        XgPtr, (self.copy_elems,)
                                    )
                                    Xs_thr_slice = cute.make_tensor(
                                        XsPtr, (self.copy_elems,)
                                    )

                                    cute.copy(X_thr_copy, Xg_thr_slice, Xs_thr_slice)

                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)

                            for i in cutlass.range_constexpr(K_STAGED):
                                k_idx = k * K_STAGED + i
                                if k_idx < K:
                                    XsPtr = self.elem_pointer(
                                        sX_slice,
                                        (row_id_within_block, col_id_within_row, i),
                                    ).align(self.universal_copy_bits // self.copy_elems)
                                    Xs_thr_slice = cute.make_tensor(
                                        XsPtr, (self.copy_elems,)
                                    )

                                    cute.autovec_copy(Xs_thr_slice, rX)

                                    for j in cutlass.range_constexpr(self.copy_elems):
                                        if const_expr(mW is not None):
                                            y[(j, 0), 0, 0] = self.fma(
                                                rW[i].to(cutlass.Float32),
                                                rX[j].to(cutlass.Float32),
                                                y[(j, 0), 0, 0],
                                            )
                                        else:
                                            y[(j, 0), 0, 0] = (
                                                rX[j].to(cutlass.Float32)
                                                + y[(j, 0), 0, 0]
                                            )

                        tOrO.store(y.load().to(tOrO.element_type))
                        cute.copy(O_thr_copy, tOrO, tOgO)

            # cannot be removed for correctness!
            if const_expr(self.is_persistent):
                cute.arch.sync_threads()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


### This triton impl is equivalent as the cute-dsl impl above,
# and also achieves similar memory bandwidth on H100 for large K and H.
# However, for small K and H, this impl is better by autotuning so we use it as default.
def _get_triton_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_H in get_powers_of_2(256, 4096):
        for BLOCK_K in get_powers_of_2(1, 128):
            for num_warps in [4, 8]:
                if BLOCK_K * BLOCK_H <= 32768:
                    configs.append(
                        triton.Config(
                            {"BLOCK_H": BLOCK_H, "BLOCK_K": BLOCK_K},
                            num_warps=num_warps,
                            num_stages=4,
                        )
                    )
    return configs


def _prune_triton_autotune_config(configs, nargs, **kw):
    pruned_configs = []
    for c in configs:
        BLOCK_H = c.kwargs["BLOCK_H"]
        BLOCK_K = c.kwargs["BLOCK_K"]
        H = kw["H"]
        MAX_K = kw["MAX_K"]
        if (
            BLOCK_H <= triton.next_power_of_2(H)
            and BLOCK_K <= triton.next_power_of_2(MAX_K)
            and min(H * MAX_K, 1024) <= (BLOCK_H * BLOCK_K)
        ):
            pruned_configs.append(c)

    if len(pruned_configs) == 0:
        return configs
    else:
        return pruned_configs


@triton.autotune(
    configs=_get_triton_autotune_configs(),
    key=["H", "MAX_K", "w_is_None", "is_varlen_K"],
    prune_configs_by={"early_config_prune": _prune_triton_autotune_config},
)
@triton.jit
def token_gather_sum_kernel(
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,)
    M_perm_ptr,  # (Mtotal,) int32
    M_offset_ptr,  # (T+1,)   int32
    out_ptr,  # (T, H)
    T,
    H: tl.constexpr,
    MAX_K: tl.constexpr,
    # strides
    stride_xM: tl.constexpr,
    stride_xH: tl.constexpr,
    stride_outT: tl.constexpr,
    stride_outH: tl.constexpr,
    # tile sizes
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,
    is_varlen_K: tl.constexpr,
):
    # 1D tiling over T only
    pid_t = tl.program_id(axis=0)
    t_idx = pid_t.to(tl.uint32)

    # Load segment starts and ends for this token
    if is_varlen_K:
        Ms = tl.load(M_offset_ptr + t_idx).to(tl.uint32)
        Me = tl.load(M_offset_ptr + t_idx + 1).to(tl.uint32)
        K_this_token = Me - Ms  # actual K for this token
    else:
        Ms = MAX_K * t_idx
        K_this_token: tl.constexpr = MAX_K

    # Outer loop over H tiles
    for h_tile in tl.static_range(triton.cdiv(H, BLOCK_H)):
        h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.uint32)  # [BLOCK_H]
        m_h = h_idx < H

        # Initialize accumulator for this H tile
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)  # [BLOCK_H]

        # Inner loop over K tiles
        for k_tile in tl.range(tl.cdiv(K_this_token, BLOCK_K)):
            k_offset = k_tile * BLOCK_K

            k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.uint32)  # [BLOCK_K]

            # Mask for valid K indices
            m_k = k_idx < K_this_token  # [BLOCK_K]

            # Absolute positions into M_perm and w
            m_abs = Ms + k_idx  # [BLOCK_K]

            # Gather permuted indices
            perm_idx = tl.load(M_perm_ptr + m_abs, mask=m_k, other=0).to(
                tl.uint32
            )  # [BLOCK_K]

            # Load x values: [BLOCK_K, BLOCK_H]
            x_ptrs = x_ptr + perm_idx[:, None] * stride_xM + h_idx[None, :] * stride_xH
            x_mask = m_k[:, None] & m_h[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            # Reduce along K dimension and add to accumulator
            if w_is_None:
                acc += tl.sum(x_vals, axis=0)  # [BLOCK_H]
            else:
                w_vals = tl.load(w_ptr + m_abs, mask=m_k, other=0.0).to(
                    tl.float32
                )  # [BLOCK_K]
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)  # [BLOCK_H]

        # Store final result for this H tile (only once!)
        out_ptrs = out_ptr + t_idx * stride_outT + h_idx * stride_outH
        tl.store(out_ptrs, acc, mask=m_h)


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,  # (Mtotal, H)
    w: Optional[torch.Tensor],  # (Mtotal,)
    out: torch.Tensor,  # (T, H)
    M_perm: torch.Tensor,  # (Mtotal,) int32
    M_offset: torch.Tensor,  # (T+1,)   int32, variable K per token
    T: int,
    MAX_K: int,  # maximum K across all tokens
    H: int,
    is_varlen_K: bool,
):
    """
    1D parallelization over T, with iterative accumulation over K tiles and H tiles.
    Supports variable K per token.

    out[i, :] = sum_{j=0..K[i]-1}  x[M_perm[M_offset[i] + j], :] * w[M_offset[i] + j]

    where K[i] = M_offset[i+1] - M_offset[i] can vary per token.
    """
    # 1D grid over T only
    token_gather_sum_kernel[(T,)](
        x,
        w,
        M_perm,
        M_offset,
        out,
        T=T,
        H=H,
        MAX_K=MAX_K,
        stride_xM=x.stride(0),
        stride_xH=x.stride(1),
        stride_outT=out.stride(0),
        stride_outH=out.stride(1),
        w_is_None=(w is None),
        is_varlen_K=is_varlen_K,
    )
