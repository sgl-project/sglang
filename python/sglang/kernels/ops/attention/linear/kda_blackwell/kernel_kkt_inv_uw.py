# SPDX-License-Identifier: Apache-2.0
# KDA (Kimi Delta Attention) SM100 KKT-inverse + U/W kernel.
#
# Adapted from gdn_blackwell/kernel_kkt_inv_uw.py. KDA's decay is PER-CHANNEL, so
# (as with kernel_h/o) the gate is folded OUTSIDE this kernel into pre-scaled keys:
#
#   kL [c,d] = k[c,d] * exp(g_cu[c,d] - g_cu_last[d])   (KKT left operand)
#   kR [j,d] = k[j,d] * exp(g_cu_last[d] - g_cu[j,d])   (KKT right operand, bounded)
#   kg [j,d] = k[j,d] * exp(g_cu[j,d])                  (W operand, bounded)
#
# Then  KKT[c,j] = sum_d kL[c,d]*kR[j,d] = sum_d k[c,d]*k[j,d]*exp(g_cu[c,d]-g_cu[j,d])
# carries the per-channel decay, so:
#   A  = strictLower(beta * KKT)         (NO post-MMA Gamma; decay already inside)
#   Ai = inverse(I + A)                  (Newton-Schulz, gate-independent -> verbatim)
#   U  = (Ai * beta) @ V
#   W  = (Ai * beta) @ kg                (NO Abg; the exp(g_cu) lives in kg)
#
# Net: this kernel has NO cumsum and NO g_cu — only beta survives, exactly like GDN.
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass.cute.nvgpu import cpasync, warp
from quack.compile_utils import make_fake_tensor

from sglang.kernels.ops.attention.cute_utils import (
    EVICT_FIRST,
    _tcgen05,
    cvt,
    fence_before_tma_store,
    mma_bf16,
    simple_tma_copy,
)


class Sm100KdaChunkUWKernel:
    """KDA per-chunk KKT-inverse + U/W (see module docstring)."""

    def __init__(
        self,
        H: int,
        Hv: int,
        K_dim: int,
        V_dim: int,
        num_stages: int = 2,
    ) -> None:
        assert Hv % H == 0
        assert K_dim == V_dim == 128
        self.H = H
        self.Hv = Hv
        self.K_dim = K_dim
        self.V_dim = V_dim
        self.num_stages = num_stages

        self.BT = 64
        self.num_warps = 2 + 4 + 4

    @cute.jit
    def _make_tma_args(
        self,
        tensor: cute.Tensor,
        dim: cutlass.Constexpr[int],
        num_stages: int,
        op: cpasync.TmaCopyOp,
    ):
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (self.BT, 1, (64, dim // 64), num_stages),
            stride=(64, 0, (1, self.BT * 64), self.BT * dim),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, 64)),
            slayout,
            cta_tiler=(self.BT, 1, dim),
        )
        return atom, tma_tensor, slayout

    @cute.jit
    def __call__(
        self,
        KL: cute.Tensor,  # k*exp(g_cu - g_cu_last)  [T, Hv, K]
        KR: cute.Tensor,  # k*exp(g_cu_last - g_cu)  [T, Hv, K]
        KG: cute.Tensor,  # k*exp(g_cu)              [T, Hv, K]
        V: cute.Tensor,
        U: cute.Tensor,
        W: cute.Tensor,
        beta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
        num_sms: Int32,
        stream: CUstream,
    ):
        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        tma_s2g = cpasync.CopyBulkTensorTileS2GOp()

        KL_args = self._make_tma_args(KL, self.K_dim, self.num_stages, tma_g2s)
        KR_args = self._make_tma_args(KR, self.K_dim, self.num_stages, tma_g2s)
        KG_args = self._make_tma_args(KG, self.K_dim, self.num_stages, tma_g2s)
        V_args = self._make_tma_args(V, self.V_dim, self.num_stages, tma_g2s)
        U_args = self._make_tma_args(U, self.V_dim, 1, tma_s2g)
        W_args = self._make_tma_args(W, self.K_dim, 1, tma_s2g)

        grid = (num_sms // self.Hv, self.Hv, 1)
        block = (self.num_warps * 32, 1, 1)
        self.kernel(
            KL_args,
            KR_args,
            KG_args,
            V_args,
            U_args,
            W_args,
            beta,
            cu_seqlens,
            chunk_indices,
            total_chunks,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        KL_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        KR_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        KG_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        V_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        U_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        W_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        beta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, head_id, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()

        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BT = self.BT
        K_dim = self.K_dim
        V_dim = self.V_dim
        num_stages = self.num_stages

        KL_tma_atom, tmaKL, sKL_layout = KL_args
        KR_tma_atom, tmaKR, sKR_layout = KR_args
        KG_tma_atom, tmaKG, sKG_layout = KG_args
        V_tma_atom, tmaV, sV_layout = V_args
        U_tma_atom, tmaU, sU_layout = U_args
        W_tma_atom, tmaW, sW_layout = W_args

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        sKL = allocate_tensor(smem, BFloat16, sKL_layout)[None, 0, None, None]
        sKR = allocate_tensor(smem, BFloat16, sKR_layout)[None, 0, None, None]
        sKG = allocate_tensor(smem, BFloat16, sKG_layout)[None, 0, None, None]
        sV = allocate_tensor(smem, BFloat16, sV_layout)[None, 0, None, None]
        sU = allocate_tensor(smem, BFloat16, sU_layout)[None, 0, None, 0]
        sW = allocate_tensor(smem, BFloat16, sW_layout)[None, 0, None, 0]

        swizzle_128B = cute.make_swizzle(3, 4, 3)
        sA_layout = cute.make_layout((BT, (64, 1)), stride=(64, (1, BT * 64)))
        sA_layout = cute.make_composed_layout(swizzle_128B, 0, sA_layout)
        sA = allocate_tensor(smem, BFloat16, sA_layout)
        sAi = allocate_tensor(smem, BFloat16, sA_layout)

        s_beta = smem.allocate_array(Float32, BT)

        tma_mbar = smem.allocate_array(Int64, num_stages)
        mma_kkt_mbar = smem.allocate_array(Int64, num_stages)
        inv_mbar = smem.allocate_array(Int64, num_stages)
        mma_u_mbar = smem.allocate_array(Int64, num_stages)
        mma_w_mbar = smem.allocate_array(Int64, num_stages)
        epi_mbar = smem.allocate_array(Int64, num_stages)
        taddr = smem.allocate(Int32, 4)

        kkt_tmem = 0
        U_tmem_base = kkt_tmem + BT
        Ab_tmem_base = U_tmem_base + V_dim * num_stages
        assert Ab_tmem_base + (BT // 2) * num_stages <= 512

        ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
        stsm_op = warp.StMatrix8x8x16bOp(num_matrices=4)
        ldsm_trans_op = warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True)
        ldsm_atom = cute.make_copy_atom(ldsm_op, BFloat16)
        stsm_atom = cute.make_copy_atom(stsm_op, BFloat16)
        ldsm_trans_atom = cute.make_copy_atom(ldsm_trans_op, BFloat16)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_mbar + i, 1)
                    cute.arch.mbarrier_init(mma_kkt_mbar + i, 1)
                    cute.arch.mbarrier_init(inv_mbar + i, 128)
                    cute.arch.mbarrier_init(mma_u_mbar + i, 1)
                    cute.arch.mbarrier_init(mma_w_mbar + i, 1)
                    cute.arch.mbarrier_init(epi_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(KL_tma_atom)
            cpasync.prefetch_descriptor(KR_tma_atom)
            cpasync.prefetch_descriptor(KG_tma_atom)
            cpasync.prefetch_descriptor(V_tma_atom)
            cpasync.prefetch_descriptor(U_tma_atom)
            cpasync.prefetch_descriptor(W_tma_atom)
        cute.arch.sync_threads()

        num_global_chunks = total_chunks[0]
        if warp_id == 9:
            # TMA warp
            stage_id = 0
            parity = 1

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]

                mbar = tma_mbar + stage_id
                # KDA: all keys are per v-head [T, Hv, K], index by head_id.
                gKL = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaKL[None, head_id, None]),
                    tiler=(BT, K_dim),
                    coord=(chunk_id, 0),
                )
                gKR = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaKR[None, head_id, None]),
                    tiler=(BT, K_dim),
                    coord=(chunk_id, 0),
                )
                gKG = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaKG[None, head_id, None]),
                    tiler=(BT, K_dim),
                    coord=(chunk_id, 0),
                )
                gV = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaV[None, head_id, None]),
                    tiler=(BT, V_dim),
                    coord=(chunk_id, 0),
                )

                cute.arch.mbarrier_wait(mma_u_mbar + stage_id, parity)

                with cute.arch.elect_one():
                    STAGE_SIZE = BT * (K_dim + K_dim + K_dim + V_dim) * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_copy(KL_tma_atom, gKL, sKL[None, None, stage_id], mbar)
                simple_tma_copy(KR_tma_atom, gKR, sKR[None, None, stage_id], mbar)
                simple_tma_copy(
                    KG_tma_atom, gKG, sKG[None, None, stage_id], mbar, EVICT_FIRST
                )
                simple_tma_copy(
                    V_tma_atom, gV, sV[None, None, stage_id], mbar, EVICT_FIRST
                )

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id == 8:
            # MMA warp
            _tcgen05.alloc(taddr)

            stage_id = 0
            parity = 0

            kkt_idesc = _tcgen05.make_bf16_idesc(BT, BT)
            u_idesc = _tcgen05.make_bf16_idesc(BT, V_dim, transpose_B=True)
            w_idesc = _tcgen05.make_bf16_idesc(BT, K_dim, transpose_B=True)

            sdesc_template = _tcgen05.make_sdesc_128B_swizzle(BT * 128)

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                U_tmem = U_tmem_base + V_dim * stage_id
                W_tmem = U_tmem | (16 << 16)
                Ab_tmem = Ab_tmem_base + (BT // 2) * stage_id
                Abg_tmem = Ab_tmem | (16 << 16)

                ##### KKT MMA: KKT = kL @ kR.T #####
                klraddr = sKL[None, None, stage_id].iterator.toint()
                krraddr = sKR[None, None, stage_id].iterator.toint()
                kldesc_base = sdesc_template | (klraddr >> 4)
                krdesc_base = sdesc_template | (krraddr >> 4)

                cute.arch.mbarrier_wait(tma_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(K_dim // 64):
                        for j in cutlass.range_constexpr(64 // 16):
                            off = (i * BT * 128 + j * 32) >> 4
                            _tcgen05.mma_f16(
                                kkt_tmem,
                                kldesc_base | off,
                                krdesc_base | off,
                                kkt_idesc,
                                (i > 0) or (j > 0),
                            )
                    _tcgen05.commit(mma_kkt_mbar + stage_id)

                ##### U/W MMA: U = Ab @ V, W = Ab @ kg #####
                vaddr = sV[None, None, stage_id].iterator.toint()
                kgaddr = sKG[None, None, stage_id].iterator.toint()
                vdesc = sdesc_template | (vaddr >> 4)
                kgdesc = sdesc_template | (kgaddr >> 4)

                cute.arch.mbarrier_wait(epi_mbar + stage_id, parity ^ 1)
                cute.arch.mbarrier_wait(inv_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(BT // 16):
                        _tcgen05.mma_ts_f16(
                            W_tmem, Abg_tmem + i * 8, kgdesc, w_idesc, i > 0
                        )
                        kgdesc += (16 * 128) >> 4
                    _tcgen05.commit(mma_w_mbar + stage_id)

                    for i in cutlass.range_constexpr(BT // 16):
                        _tcgen05.mma_ts_f16(
                            U_tmem, Ab_tmem + i * 8, vdesc, u_idesc, i > 0
                        )
                        vdesc += (16 * 128) >> 4
                    _tcgen05.commit(mma_u_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

            cute.arch.mbarrier_wait(epi_mbar + stage_id, parity ^ 1)
            _tcgen05.dealloc()

        elif warp_id >= 4:
            # inv warps
            tid_ = tid % 128
            warp_id_ = warp_id % 4

            stage_id = 0
            parity = 0

            sA_ldsm = cute.logical_divide(sA, (16, cute.make_layout((8, 2))))
            sAi_ldsm = cute.logical_divide(sAi, (16, cute.make_layout((8, 2))))
            sA_ldsm = sA_ldsm[(lane_id % 16, None), ((None, lane_id // 16), None)]
            sAi_ldsm = sAi_ldsm[(lane_id % 16, None), ((None, lane_id // 16), None)]

            for i in cutlass.range_constexpr((BT // 4 * 3) * BT // 128):
                idx = i * 128 + tid_
                sAi[idx // BT, idx % BT] = BFloat16(0.0)

            row_indices = cute.make_rmem_tensor((1, 2, 1), Int32)
            row_indices[0, 0, 0] = warp_id_ * 16 + (lane_id // 4)
            row_indices[0, 1, 0] = warp_id_ * 16 + (lane_id // 4) + 8
            row_indices = row_indices.load()

            col_indices = cute.make_rmem_tensor((2, 1, 2), Int32)
            col_indices[0, 0, 0] = (lane_id % 4) * 2 + 0
            col_indices[1, 0, 0] = (lane_id % 4) * 2 + 1
            col_indices[0, 0, 1] = (lane_id % 4) * 2 + 8
            col_indices[1, 0, 1] = (lane_id % 4) * 2 + 9
            col_indices = col_indices.load()

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]
                eos = cu_seqlens[seq_id + 1]
                off_t = bos + chunk_id * BT

                t = off_t + tid_

                ##### Phase 1: load beta (KDA: no cumsum) #####
                if tid_ < BT:
                    in_bounds = t < eos
                    beta_val = beta[t, head_id] if in_bounds else Float32(0.0)
                    s_beta[tid_] = beta_val

                ##### Phase 2: A = strictLower(beta * kkt) #####
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(mma_kkt_mbar + stage_id, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                row_coord = (lane_id // 4, None, warp_id_)
                s_beta_view = cute.make_tensor(s_beta, (8, 2, 4))
                beta_row = s_beta_view[row_coord].load().reshape((1, 2, 1))

                kkt = _tcgen05.ld(kkt_tmem, 0, "16x256b", BT // 8)
                kkt = kkt.reshape((2, 2, 2, BT // 16))

                for i in cutlass.range_constexpr(BT // 16):
                    # KDA: decay is already inside KKT; only beta + mask here.
                    A = kkt[None, None, None, i] * beta_row

                    A_masked = cute.where(row_indices > col_indices + i * 16, A, 0.0)

                    packed = cute.make_rmem_tensor(4, Uint32)
                    packed[0] = cvt.fp32x2_to_bf16x2(
                        A_masked[0, 0, 0], A_masked[1, 0, 0]
                    )
                    packed[1] = cvt.fp32x2_to_bf16x2(
                        A_masked[0, 1, 0], A_masked[1, 1, 0]
                    )
                    packed[2] = cvt.fp32x2_to_bf16x2(
                        A_masked[0, 0, 1], A_masked[1, 0, 1]
                    )
                    packed[3] = cvt.fp32x2_to_bf16x2(
                        A_masked[0, 1, 1], A_masked[1, 1, 1]
                    )

                    cute.copy(
                        stsm_atom,
                        cute.recast_tensor(packed, BFloat16),
                        sA_ldsm[warp_id_, None, i],
                    )

                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                ##### Phase 3: matrix inverse (VERBATIM from GDN) #####
                zeros_f32 = cute.make_rmem_tensor(4, Float32)
                zeros_f32.fill(0.0)

                def set_diagonal(A: cute.Tensor, lane_id: Int32):
                    "Set the diagonal to 1s"
                    if lane_id % 9 == 0:
                        A[0] = (A[0] & Uint32(0xFFFF0000)) | Uint32(0x00003F80)
                        A[3] = (A[3] & Uint32(0xFFFF0000)) | Uint32(0x00003F80)
                    elif lane_id % 9 == 4:
                        A[0] = (A[0] & Uint32(0x0000FFFF)) | Uint32(0x3F800000)
                        A[3] = (A[3] & Uint32(0x0000FFFF)) | Uint32(0x3F800000)

                Ai_bf16 = cute.make_rmem_tensor(8, BFloat16)
                mma_B_bf16 = cute.make_rmem_tensor(8, BFloat16)
                M_bf16 = cute.make_rmem_tensor(8, BFloat16)
                acc = cute.make_rmem_tensor((4, 2), Float32)

                Ai = cute.recast_tensor(Ai_bf16, Uint32)
                mma_B = cute.logical_divide(cute.recast_tensor(mma_B_bf16, Uint32), 2)
                M = cute.logical_divide(cute.recast_tensor(M_bf16, Uint32), 2)

                cute.copy(ldsm_atom, sA_ldsm[warp_id_, None, warp_id_], Ai_bf16)
                for i in cutlass.range_constexpr(4):
                    Ai[i] ^= Uint32(0x80008000)
                set_diagonal(Ai, lane_id)

                Ai_f32 = cute.logical_divide(cvt.bf16x2_to_fp32x2(Ai), 4)

                cute.copy(ldsm_trans_atom, sA_ldsm[warp_id_, None, warp_id_], M_bf16)
                set_diagonal(M, lane_id)
                for i in cutlass.range_constexpr(4):
                    M[i] ^= Uint32(0x80008000)

                for _ in cutlass.range_constexpr(3):
                    cute.copy(stsm_atom, Ai_bf16, sA_ldsm[warp_id_, None, warp_id_])
                    cute.arch.sync_warp()
                    acc[None, 0] = mma_bf16(Ai, M[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, M[None, 1], zeros_f32)
                    Ai_bf16.store(acc.load().to(BFloat16))

                    for j in cutlass.range_constexpr(8):
                        Ai_f32[j] *= 2.0
                    cute.copy(
                        ldsm_trans_atom,
                        sA_ldsm[warp_id_, None, warp_id_],
                        mma_B_bf16,
                    )
                    Ai_f32[None, 0] = mma_bf16(Ai, mma_B[None, 0], Ai_f32[None, 0])
                    Ai_f32[None, 1] = mma_bf16(Ai, mma_B[None, 1], Ai_f32[None, 1])
                    Ai_bf16.store(Ai_f32.load().to(BFloat16))

                cute.copy(stsm_atom, Ai_bf16, sAi_ldsm[warp_id_, None, warp_id_])
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                if warp_id_ > 0:
                    neg_Ai = cute.make_rmem_tensor(4, Uint32)
                    for i in cutlass.range_constexpr(4):
                        neg_Ai[i] = Ai[i] ^ Uint32(0x80008000)

                    cute.copy(
                        ldsm_trans_atom,
                        sA_ldsm[warp_id_, None, warp_id_ - 1],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(neg_Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(neg_Ai, mma_B[None, 1], zeros_f32)
                    Ai_bf16.store(acc.load().to(BFloat16))

                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[warp_id_ - 1, None, warp_id_ - 1],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)
                    Ai_bf16.store(acc.load().to(BFloat16))
                    cute.copy(
                        stsm_atom,
                        Ai_bf16,
                        sAi_ldsm[warp_id_, None, warp_id_ - 1],
                    )
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                if warp_id_ < 2:
                    cute.copy(
                        ldsm_atom,
                        sA_ldsm[warp_id_ + 2, None, warp_id_],
                        Ai_bf16,
                    )
                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[warp_id_, None, warp_id_],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)

                    cute.copy(
                        ldsm_atom,
                        sA_ldsm[warp_id_ + 2, None, warp_id_ + 1],
                        Ai_bf16,
                    )
                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[warp_id_ + 1, None, warp_id_],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], acc[None, 0])
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], acc[None, 1])

                    tmp = cute.make_rmem_tensor(8, BFloat16)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[warp_id_ + 2, None, warp_id_])
                    cute.arch.sync_warp()

                    cute.copy(
                        ldsm_atom, sAi_ldsm[warp_id_ + 2, None, warp_id_ + 2], Ai_bf16
                    )
                    for i in cutlass.range_constexpr(4):
                        Ai[i] ^= Uint32(0x80008000)
                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[warp_id_ + 2, None, warp_id_],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[warp_id_ + 2, None, warp_id_])
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                if warp_id_ == 0:
                    cute.copy(ldsm_atom, sA_ldsm[3, None, 0], Ai_bf16)
                    cute.copy(ldsm_trans_atom, sAi_ldsm[0, None, 0], mma_B_bf16)
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)

                    for i in cutlass.range_constexpr(1, 3):
                        cute.copy(ldsm_atom, sA_ldsm[3, None, i], Ai_bf16)
                        cute.copy(ldsm_trans_atom, sAi_ldsm[i, None, 0], mma_B_bf16)
                        acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], acc[None, 0])
                        acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], acc[None, 1])

                    tmp = cute.make_rmem_tensor(8, BFloat16)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[3, None, 0])
                    cute.arch.sync_warp()

                    cute.copy(ldsm_atom, sAi_ldsm[3, None, 3], Ai_bf16)
                    for i in cutlass.range_constexpr(4):
                        Ai[i] ^= Uint32(0x80008000)
                    cute.copy(ldsm_trans_atom, sAi_ldsm[3, None, 0], mma_B_bf16)
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[3, None, 0])

                ##### Phase 4: Ab = Ai * beta (KDA: no Abg) #####
                if warp_id_ == 3:
                    cute.arch.mbarrier_wait(mma_u_mbar + stage_id, parity ^ 1)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                for i in cutlass.range_constexpr(BT // 16):
                    cute.copy(ldsm_atom, sAi_ldsm[warp_id_, None, i], Ai_bf16)

                    col_coord = (None, lane_id % 4, None, i)
                    s_beta_view = cute.make_tensor(s_beta, (2, 4, 2, BT // 16))
                    beta_col = s_beta_view[col_coord].load().reshape((2, 1, 2))

                    Ai_f32 = cvt.bf16x2_to_fp32x2(Ai).load().reshape((2, 2, 2))

                    Ab_f32 = Ai_f32 * beta_col
                    Ab = Ab_f32.to(BFloat16)
                    Ab_tmem = Ab_tmem_base + (BT // 2) * stage_id + i * 8
                    _tcgen05.st(warp_id_ * 32, Ab_tmem, "16x128b", 2, Ab)
                    # KDA: Abg == Ab (no per-chunk g on the matrix). Duplicate into the
                    # +16 lane region so the W MMA (reads Abg_tmem) sees valid data,
                    # matching GDN's tmem layout exactly.
                    _tcgen05.st(warp_id_ * 32 + 16, Ab_tmem, "16x128b", 2, Ab)

                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(inv_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id < 4:
            # epi warps (store U, W) -- VERBATIM from GDN
            stage_id = 0
            parity = 0

            gU_tiles = cute.logical_divide(tmaU[None, head_id, None], (BT, None))
            gW_tiles = cute.logical_divide(tmaW[None, head_id, None], (BT, None))

            s_row = warp_id * 16 + lane_id % 16
            sW_view = cute.zipped_divide(
                sW[s_row, None],
                tiler=cute.make_layout((8, 2)),
            )
            sU_view = cute.zipped_divide(
                sU[s_row, None],
                tiler=cute.make_layout((8, 2)),
            )

            sW_view = sW_view[(None, lane_id // 16), None]
            sU_view = sU_view[(None, lane_id // 16), None]

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                U_tmem = U_tmem_base + V_dim * stage_id
                if warp_id == 0:
                    cute.arch.mbarrier_wait(mma_w_mbar + stage_id, parity)
                elif warp_id == 1:
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                w_f32 = _tcgen05.ld(warp_id * 32 + 16, U_tmem, "16x256b", K_dim // 8)
                _tcgen05.wait_ld()
                w_bf16 = cute.make_rmem_tensor((8, K_dim // 16), BFloat16)
                w_bf16.store(w_f32.to(BFloat16))
                cute.copy(stsm_atom, w_bf16, sW_view)

                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                fence_before_tma_store()
                if warp_id == 0:
                    cute.arch.mbarrier_wait(mma_u_mbar + stage_id, parity)
                elif warp_id == 1:
                    simple_tma_copy(
                        W_tma_atom, sW, gW_tiles[(None, global_chunk_id), None]
                    )
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                u_f32 = _tcgen05.ld(warp_id * 32, U_tmem, "16x256b", V_dim // 8)
                _tcgen05.wait_ld()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(epi_mbar + stage_id)
                u_bf16 = cute.make_rmem_tensor((8, V_dim // 16), BFloat16)
                u_bf16.store(u_f32.to(BFloat16))
                cute.copy(stsm_atom, u_bf16, sU_view)

                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                fence_before_tma_store()
                if warp_id == 1:
                    simple_tma_copy(
                        U_tma_atom, sU, gU_tiles[(None, global_chunk_id), None]
                    )
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

    @cache
    @staticmethod
    def compile(H: int, Hv: int, K_dim: int, V_dim: int, num_stages: int = 2):
        total_t = cute.sym_int()
        pad_t = cute.sym_int()
        total_chunks_n = cute.sym_int()
        num_sequences = cute.sym_int()

        KL = make_fake_tensor(BFloat16, (total_t, Hv, K_dim), divisibility=16)
        KR = make_fake_tensor(BFloat16, (total_t, Hv, K_dim), divisibility=16)
        KG = make_fake_tensor(BFloat16, (total_t, Hv, K_dim), divisibility=16)
        V = make_fake_tensor(BFloat16, (total_t, Hv, V_dim), divisibility=16)
        U = make_fake_tensor(BFloat16, (pad_t, Hv, V_dim), divisibility=16)
        W = make_fake_tensor(BFloat16, (pad_t, Hv, K_dim), divisibility=16)
        beta = make_fake_tensor(Float32, (total_t, Hv), divisibility=4)
        cu_seqlens = make_fake_tensor(Int32, (num_sequences,), divisibility=1)
        chunk_indices = make_fake_tensor(Int32, (total_chunks_n, 2), divisibility=2)
        total_chunks = make_fake_tensor(Int32, (1,), divisibility=1)

        kernel = Sm100KdaChunkUWKernel(H, Hv, K_dim, V_dim, num_stages)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            KL,
            KR,
            KG,
            V,
            U,
            W,
            beta,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            Int32(148),
            stream,
            options="--enable-tvm-ffi",
        )


def kkt_inv_uw_cutedsl(
    KL: torch.Tensor,
    KR: torch.Tensor,
    KG: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    total_chunks: torch.Tensor,
    num_sms: int = 148,
) -> None:
    """KDA KKT-inverse + U/W. KL/KR/KG are the pre-scaled keys (see module doc)."""
    _, Hv, K_dim = KL.shape
    _, _, V_dim = V.shape
    Sm100KdaChunkUWKernel.compile(Hv, Hv, K_dim, V_dim)(
        KL, KR, KG, V, U, W, beta, cu_seqlens, chunk_indices, total_chunks, num_sms
    )
