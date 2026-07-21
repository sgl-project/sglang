# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined. INTERNAL COLLABORATION ONLY.
# ruff: noqa  -- vendored kernel library, minimal local deltas

"""
Optimized Fused Gate Activation + Cumsum + Scaling Kernel for KDA (V2 Vec) - 4-Stage Pipeline.

4-warp parallel design with vec2 vectorized access + 4-stage K,Q pipelining:

          cols 0-63      cols 64-127
rows 0-31    warp 0        warp 2
rows 32-63   warp 1        warp 3

Each warp: 32 threads × 32 rows × 1 vec2 col/thread = 32 rows × 32 vec2 cols (64 scalar)

K,Q Pipeline Design:
- 4 stages, each 8 rows (32 rows / 4 = 8 rows per stage)
- Triple buffer (3 SMEM buffers)
- Prefetch 2 stages ahead

Pipeline Flow:
  Time 0: Prefetch stage 0 → buf 0, Prefetch stage 1 → buf 1
  Time 1: Compute stage 0 (buf 0), Prefetch stage 2 → buf 2
  Time 2: Compute stage 1 (buf 1), Prefetch stage 3 → buf 0 (wrap)
  Time 3: Compute stage 2 (buf 2)
  Time 4: Compute stage 3 (buf 0)
"""

import glob
import os
import subprocess

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute import KeepCUBIN, KeepPTX
from cutlass.cute.runtime import from_dlpack

# Configuration
BT = 64  # Chunk size (rows)
ROWS_PER_WARP = 32  # Each warp handles 32 rows
BS = 128  # Feature block size per CTA (128 scalar cols)
VEC_SIZE = 2  # Vec2 for loads/stores
BS_VEC = BS // VEC_SIZE  # 64 vec2 columns per row
COLS_VEC_PER_WARP = BS_VEC // 2  # 32 vec2 columns per warp
NUM_WARPS = 4  # 4 warps = 128 threads
WARP_SIZE = 32
THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE  # 128 threads

# Pipeline configuration
NUM_STAGES = 4  # 4 stages for K,Q
NUM_BUFFERS = 3  # Triple buffering
PREFETCH_STAGES = 2  # Prefetch 2 stages ahead
ROWS_PER_STAGE = ROWS_PER_WARP // NUM_STAGES  # 8 rows per stage

# cp.async configuration (128-bit = 8 fp16)
CPASYNC_SIZE = 8  # 8 fp16 per 128-bit cp.async
COLS_PER_WARP = 64  # scalar columns per warp (32 vec2)
LOAD_CHUNKS_PER_WARP = COLS_PER_WARP // CPASYNC_SIZE  # 8 chunks per row per warp
# For 8 rows: 8 rows × 8 chunks = 64 loads, 32 threads → 2 loads per thread
LOAD_ROWS_PER_ITER = WARP_SIZE // LOAD_CHUNKS_PER_WARP  # 4 rows per iteration
LOAD_ITERS_PER_STAGE = ROWS_PER_STAGE // LOAD_ROWS_PER_ITER  # 2 iterations for 8 rows


class ActCumsumScaleFusedV2VecPipe4:
    """Optimized fused kernel with 4-stage K,Q pipeline and triple buffering.

    Supports two activation modes:
    - Standard mode (USE_LOWER_BOUND=False): g = -exp(A_log) * softplus(g)
    - Safe gate mode (USE_LOWER_BOUND=True): g = lower_bound * sigmoid(exp(A_log) * g)
    """

    def __init__(self):
        self.num_threads = THREADS_PER_BLOCK

    @cute.jit
    def __call__(
        self,
        mG: cute.Tensor,  # [B, T, H, S//2, 2] raw gate (vec2 layout)
        mK: cute.Tensor,  # [B, T, H, S//2, 2] keys
        mQ: cute.Tensor,  # [B, T, H, S//2, 2] queries
        mK_128: cute.Tensor,  # [B, T, H, S//8, 8] keys (128-bit view for cp.async)
        mQ_128: cute.Tensor,  # [B, T, H, S//8, 8] queries (128-bit view for cp.async)
        mA_log: cute.Tensor,  # [H] log of decay rate
        mDt_bias: cute.Tensor,  # [H, S//2, 2] dt_bias (vec2 layout), or dummy if HAS_BIAS=False
        cumsum_scale: cutlass.Float32,
        attn_scale: cutlass.Float32,
        lower_bound: cutlass.Float32,  # lower bound for safe_gate mode (e.g., -5.0)
        mG_cumsum: cute.Tensor,  # [B, T, H, S//2, 2] output (fp32)
        mK_scaled: cute.Tensor,  # [B, T, H, S//2, 2] output
        mKg: cute.Tensor,  # [B, T, H, S//2, 2] output
        mQ_scaled: cute.Tensor,  # [B, T, H, S//2, 2] output
        mGk_last_exp: cute.Tensor,  # [B, NT, H, S//2, 2] output (fp32)
        mCuSeqlens: cute.Tensor,  # [N+1] int32 (varlen) or dummy
        mChunkIndices: cute.Tensor,  # [NT_total, 2] int32 (varlen) or dummy
        NT_total: cutlass.Int32,  # Runtime: total chunks (varlen) or NT (equal-len)
        B: cutlass.Constexpr,
        T: cutlass.Constexpr,
        H: cutlass.Constexpr,
        S: cutlass.Constexpr,
        USE_LOWER_BOUND: cutlass.Constexpr,  # Compile-time flag for activation mode
        HAS_BIAS: cutlass.Constexpr,  # Compile-time flag for dt_bias
        IS_VARLEN: cutlass.Constexpr,  # Compile-time flag for varlen mode
    ):
        """Launch the optimized kernel with 4-stage pipeline."""
        # Create 128-bit copy atom for cp.async
        copy_atom_128 = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        # Grid: (cdiv(S, BS), NT, B * H) for equal-len, (cdiv(S, BS), NT_total, H) for varlen
        grid_x = (S + BS - 1) // BS
        grid_y = NT_total  # Runtime value: NT for equal-len, NT_total for varlen
        grid_z = B * H
        if IS_VARLEN:
            grid_z = H  # B=1 for varlen

        self.kernel(
            mG,
            mK,
            mQ,
            mK_128,
            mQ_128,
            mA_log,
            mDt_bias,
            cumsum_scale,
            attn_scale,
            lower_bound,
            mG_cumsum,
            mK_scaled,
            mKg,
            mQ_scaled,
            mGk_last_exp,
            mCuSeqlens,
            mChunkIndices,
            copy_atom_128,
            B,
            T,
            H,
            S,
            USE_LOWER_BOUND,
            HAS_BIAS,
            IS_VARLEN,
        ).launch(
            grid=(grid_x, grid_y, grid_z),
            block=(THREADS_PER_BLOCK, 1, 1),
            smem=self._get_smem_size(),
        )

    def _get_smem_size(self):
        """Calculate shared memory size needed."""
        # K buffer: 3 buffers × 2 col_groups × 2 row_groups × 8 rows × 64 cols × 2 bytes = 12KB
        # Q buffer: same = 12KB
        # Partial last: 2 × 2 × 32 × 2 × 4 bytes = 1KB
        # Total ~25KB + alignment
        return 32 * 1024

    @cute.kernel
    def kernel(
        self,
        mG: cute.Tensor,
        mK: cute.Tensor,
        mQ: cute.Tensor,
        mK_128: cute.Tensor,
        mQ_128: cute.Tensor,
        mA_log: cute.Tensor,
        mDt_bias: cute.Tensor,  # [H, S//2, 2] dt_bias (vec2 layout)
        cumsum_scale: cutlass.Float32,
        attn_scale: cutlass.Float32,
        lower_bound: cutlass.Float32,
        mG_cumsum: cute.Tensor,
        mK_scaled: cute.Tensor,
        mKg: cute.Tensor,
        mQ_scaled: cute.Tensor,
        mGk_last_exp: cute.Tensor,
        mCuSeqlens: cute.Tensor,  # [N+1] int32 (varlen) or dummy
        mChunkIndices: cute.Tensor,  # [NT_total, 2] int32 (varlen) or dummy
        copy_atom_128: cute.CopyAtom,
        B: cutlass.Constexpr,
        T: cutlass.Constexpr,
        H: cutlass.Constexpr,
        S: cutlass.Constexpr,
        USE_LOWER_BOUND: cutlass.Constexpr,
        HAS_BIAS: cutlass.Constexpr,
        IS_VARLEN: cutlass.Constexpr,
    ):
        """Main kernel with 4-stage K,Q pipeline.

        Activation modes:
        - USE_LOWER_BOUND=False: g = -exp(A_log) * softplus(g)
        - USE_LOWER_BOUND=True:  g = lower_bound * sigmoid(exp(A_log) * g)

        Optional dt_bias:
        - HAS_BIAS=True: g = g + dt_bias before activation

        Varlen mode (IS_VARLEN=True):
        - B=1, sequences concatenated along T
        - cu_seqlens marks boundaries, chunk_indices maps chunks to sequences
        - Boundary checks on load/store/gate prevent OOB access
        """
        # Get block/thread indices
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # Decode block indices
        i_s = bidx
        i_t = bidy  # chunk index (global for varlen, per-batch for equal-len)

        # Initialize all variables before control flow (required by CuTe DSL)
        i_bh = bidz
        i_b = i_bh // H
        i_h = i_bh % H
        chunk_start = i_t * BT
        eos = i_t  # dummy init, will be overwritten

        if IS_VARLEN:
            # Varlen: grid_z = H, B=1
            i_b = 0
            i_h = bidz
            # Read chunk→sequence mapping.
            # Tensor may be int64 (FLA convention) — cast to Int32 in-register
            # to keep type consistency. This avoids .to(int32) GPU cast kernels.
            seq_id = cutlass.Int32(mChunkIndices[i_t, 0])
            local_chunk_id = cutlass.Int32(mChunkIndices[i_t, 1])
            bos = cutlass.Int32(mCuSeqlens[seq_id])
            eos = cutlass.Int32(mCuSeqlens[seq_id + 1])
            chunk_start = bos + local_chunk_id * BT

        # Warp and lane indices
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = tidx % WARP_SIZE

        # Warp layout: 2×2 grid
        warp_row_group = warp_idx % 2  # 0 or 1
        warp_col_group = warp_idx // 2  # 0 or 1

        # Global offsets
        warp_row_offset = warp_row_group * ROWS_PER_WARP
        col_base = i_s * BS_VEC + warp_col_group * COLS_VEC_PER_WARP
        col_vec = col_base + lane_idx

        # cp.async thread mapping
        load_row_offset = lane_idx // LOAD_CHUNKS_PER_WARP  # 0-3
        load_chunk_idx = lane_idx % LOAD_CHUNKS_PER_WARP  # 0-7
        read_chunk = lane_idx // (CPASYNC_SIZE // VEC_SIZE)
        read_offset = (lane_idx % (CPASYNC_SIZE // VEC_SIZE)) * VEC_SIZE

        # =====================================================================
        # SMEM Allocation - Triple Buffer for K,Q
        # =====================================================================
        smem = cutlass.utils.SmemAllocator()

        # K,Q triple buffer: (NUM_BUFFERS, 2, 2, ROWS_PER_STAGE, chunks, CPASYNC_SIZE)
        # Layout: (buffer, col_group, row_group, row, chunk, element)
        sK_layout = cute.make_layout(
            (NUM_BUFFERS, 2, 2, ROWS_PER_STAGE, LOAD_CHUNKS_PER_WARP, CPASYNC_SIZE),
            stride=(2048, 1024, 512, 64, CPASYNC_SIZE, 1),
        )
        sQ_layout = cute.make_layout(
            (NUM_BUFFERS, 2, 2, ROWS_PER_STAGE, LOAD_CHUNKS_PER_WARP, CPASYNC_SIZE),
            stride=(2048, 1024, 512, 64, CPASYNC_SIZE, 1),
        )
        sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
        sQ = smem.allocate_tensor(cutlass.BFloat16, sQ_layout, 16)

        # Partial last for prefix exchange
        sPartialLast = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((2, 2, WARP_SIZE, VEC_SIZE)), 16
        )

        # =====================================================================
        # Register arrays
        # =====================================================================
        rG_warp = cute.make_rmem_tensor(
            cute.make_layout((ROWS_PER_WARP, VEC_SIZE)), cutlass.BFloat16
        )
        rG = cute.make_rmem_tensor(cute.make_layout((VEC_SIZE,)), cutlass.BFloat16)
        rAcc = cute.make_rmem_tensor(cute.make_layout((VEC_SIZE,)), cutlass.Float32)
        rAcc[0] = cutlass.Float32(0.0)
        rAcc[1] = cutlass.Float32(0.0)

        rCsOut = cute.make_rmem_tensor(cute.make_layout((VEC_SIZE,)), cutlass.Float32)
        rKsOut = cute.make_rmem_tensor(cute.make_layout((VEC_SIZE,)), cutlass.BFloat16)
        rKgOut = cute.make_rmem_tensor(cute.make_layout((VEC_SIZE,)), cutlass.BFloat16)
        rQsOut = cute.make_rmem_tensor(cute.make_layout((VEC_SIZE,)), cutlass.BFloat16)

        # Precompute activation constants based on mode
        exp_A = cute.exp(mA_log[i_h])
        chunk_base = warp_col_group * LOAD_CHUNKS_PER_WARP

        # Load dt_bias into register (once per thread)
        # dt_bias is [H, S//2, 2] (vec2 layout), access via [i_h, col_vec, :]
        rBias = cute.make_rmem_tensor(cute.make_layout((VEC_SIZE,)), cutlass.Float32)
        rBias[0] = cutlass.Float32(0.0)
        rBias[1] = cutlass.Float32(0.0)
        if HAS_BIAS:
            cute.autovec_copy(mDt_bias[i_h, col_vec, None], rBias)

        # =====================================================================
        # Phase 0: Load all G into registers
        # =====================================================================
        for row in cutlass.range_constexpr(ROWS_PER_WARP):
            global_row = chunk_start + warp_row_offset + row
            cute.autovec_copy(mG[i_b, global_row, i_h, col_vec, None], rG)
            rG_warp[row, 0] = rG[0]
            rG_warp[row, 1] = rG[1]

        # =====================================================================
        # Phase 1: Prefetch first 2 stages of K,Q (stages 0 and 1)
        # =====================================================================
        # Prefetch stage 0 → buffer 0
        for load_iter in cutlass.range_constexpr(LOAD_ITERS_PER_STAGE):
            local_row = load_iter * LOAD_ROWS_PER_ITER + load_row_offset
            stage_row = 0 * ROWS_PER_STAGE + local_row
            global_row = chunk_start + warp_row_offset + stage_row
            global_chunk = chunk_base + load_chunk_idx
            cute.copy(
                copy_atom_128,
                mK_128[i_b, global_row, i_h, global_chunk, None],
                sK[0, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
            cute.copy(
                copy_atom_128,
                mQ_128[i_b, global_row, i_h, global_chunk, None],
                sQ[0, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
        cute.arch.cp_async_commit_group()

        # Prefetch stage 1 → buffer 1
        for load_iter in cutlass.range_constexpr(LOAD_ITERS_PER_STAGE):
            local_row = load_iter * LOAD_ROWS_PER_ITER + load_row_offset
            stage_row = 1 * ROWS_PER_STAGE + local_row
            global_row = chunk_start + warp_row_offset + stage_row
            global_chunk = chunk_base + load_chunk_idx
            cute.copy(
                copy_atom_128,
                mK_128[i_b, global_row, i_h, global_chunk, None],
                sK[1, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
            cute.copy(
                copy_atom_128,
                mQ_128[i_b, global_row, i_h, global_chunk, None],
                sQ[1, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
        cute.arch.cp_async_commit_group()

        # =====================================================================
        # Phase 2: Compute partial cumsum for all 32 rows
        # =====================================================================
        for row in cutlass.range_constexpr(ROWS_PER_WARP):
            g0 = rG_warp[row, 0].to(cutlass.Float32)
            g1 = rG_warp[row, 1].to(cutlass.Float32)

            # Add dt_bias if present (same bias for all rows in this column)
            if HAS_BIAS:
                g0 = g0 + rBias[0]
                g1 = g1 + rBias[1]

            # Initialize before control flow (required by CuTe DSL)
            g0_activated = cutlass.Float32(0.0)
            g1_activated = cutlass.Float32(0.0)

            if USE_LOWER_BOUND:
                # Safe gate mode: g = lower_bound * sigmoid(exp(A_log) * (g + bias))
                sigmoid0 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g0)
                )
                sigmoid1 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g1)
                )
                g0_activated = lower_bound * sigmoid0
                g1_activated = lower_bound * sigmoid1
            else:
                # Standard mode: g = -exp(A_log) * softplus(g + bias)
                softplus0 = cute.log(cutlass.Float32(1.0) + cute.exp(g0))
                softplus1 = cute.log(cutlass.Float32(1.0) + cute.exp(g1))
                g0_activated = -exp_A * softplus0
                g1_activated = -exp_A * softplus1

            # Zero out gate for invalid rows (varlen boundary)
            if IS_VARLEN:
                global_row_p2 = chunk_start + warp_row_offset + row
                if global_row_p2 >= eos:
                    g0_activated = cutlass.Float32(0.0)
                    g1_activated = cutlass.Float32(0.0)

            rAcc[0] = rAcc[0] + g0_activated
            rAcc[1] = rAcc[1] + g1_activated

        partial_last_0 = rAcc[0]
        partial_last_1 = rAcc[1]

        # =====================================================================
        # Phase 3: Exchange partial_last via SMEM
        # =====================================================================
        sPartialLast[warp_col_group, warp_row_group, lane_idx, 0] = partial_last_0
        sPartialLast[warp_col_group, warp_row_group, lane_idx, 1] = partial_last_1

        cute.arch.sync_threads()

        partner_row_group = warp_row_group ^ 1
        partner_last_0 = sPartialLast[warp_col_group, partner_row_group, lane_idx, 0]
        partner_last_1 = sPartialLast[warp_col_group, partner_row_group, lane_idx, 1]

        row_group_f32 = cutlass.Float32(warp_row_group)
        prefix_0 = partner_last_0 * row_group_f32
        prefix_1 = partner_last_1 * row_group_f32

        gk_last_0 = (partial_last_0 + partner_last_0) * cumsum_scale
        gk_last_1 = (partial_last_1 + partner_last_1) * cumsum_scale

        # =====================================================================
        # Phase 4: Process all 4 stages with pipelined K,Q loads
        # =====================================================================
        rAcc[0] = prefix_0
        rAcc[1] = prefix_1

        # Stage 0: wait for buf 0, prefetch stage 2 → buf 2
        cute.arch.cp_async_wait_group(1)  # Wait for stage 0 (group 0)
        cute.arch.sync_threads()

        # Prefetch stage 2 → buffer 2
        for load_iter in cutlass.range_constexpr(LOAD_ITERS_PER_STAGE):
            local_row = load_iter * LOAD_ROWS_PER_ITER + load_row_offset
            stage_row = 2 * ROWS_PER_STAGE + local_row
            global_row = chunk_start + warp_row_offset + stage_row
            global_chunk = chunk_base + load_chunk_idx
            cute.copy(
                copy_atom_128,
                mK_128[i_b, global_row, i_h, global_chunk, None],
                sK[2, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
            cute.copy(
                copy_atom_128,
                mQ_128[i_b, global_row, i_h, global_chunk, None],
                sQ[2, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
        cute.arch.cp_async_commit_group()

        # Process stage 0 (rows 0-7) from buffer 0
        for row in cutlass.range_constexpr(ROWS_PER_STAGE):
            global_row = chunk_start + warp_row_offset + row

            g0 = rG_warp[row, 0].to(cutlass.Float32)
            g1 = rG_warp[row, 1].to(cutlass.Float32)

            # Add dt_bias if present
            if HAS_BIAS:
                g0 = g0 + rBias[0]
                g1 = g1 + rBias[1]

            g0_activated = cutlass.Float32(0.0)
            g1_activated = cutlass.Float32(0.0)

            if USE_LOWER_BOUND:
                sigmoid0 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g0)
                )
                sigmoid1 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g1)
                )
                g0_activated = lower_bound * sigmoid0
                g1_activated = lower_bound * sigmoid1
            else:
                softplus0 = cute.log(cutlass.Float32(1.0) + cute.exp(g0))
                softplus1 = cute.log(cutlass.Float32(1.0) + cute.exp(g1))
                g0_activated = -exp_A * softplus0
                g1_activated = -exp_A * softplus1

            if IS_VARLEN:
                if global_row >= eos:
                    g0_activated = cutlass.Float32(0.0)
                    g1_activated = cutlass.Float32(0.0)

            rAcc[0] = rAcc[0] + g0_activated
            rAcc[1] = rAcc[1] + g1_activated

            cs0 = rAcc[0] * cumsum_scale
            cs1 = rAcc[1] * cumsum_scale

            k0 = sK[0, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            k1 = sK[
                0, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)
            q0 = sQ[0, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            q1 = sQ[
                0, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)

            exp2_cs0 = cute.exp2(cs0)
            exp2_cs1 = cute.exp2(cs1)
            exp2_kg0 = cute.exp2(gk_last_0 - cs0)
            exp2_kg1 = cute.exp2(gk_last_1 - cs1)

            rCsOut[0] = cs0
            rCsOut[1] = cs1
            rKsOut[0] = (k0 * exp2_cs0).to(cutlass.BFloat16)
            rKsOut[1] = (k1 * exp2_cs1).to(cutlass.BFloat16)
            rQsOut[0] = (q0 * exp2_cs0 * attn_scale).to(cutlass.BFloat16)
            rQsOut[1] = (q1 * exp2_cs1 * attn_scale).to(cutlass.BFloat16)
            rKgOut[0] = (k0 * exp2_kg0).to(cutlass.BFloat16)
            rKgOut[1] = (k1 * exp2_kg1).to(cutlass.BFloat16)

            if IS_VARLEN:
                if global_row < eos:
                    cute.autovec_copy(
                        rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])
            else:
                cute.autovec_copy(
                    rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])

        # Stage 1: wait for buf 1, prefetch stage 3 → buf 0 (wrap)
        cute.arch.cp_async_wait_group(1)  # Wait for stage 1
        cute.arch.sync_threads()

        # Prefetch stage 3 → buffer 0 (reuse)
        for load_iter in cutlass.range_constexpr(LOAD_ITERS_PER_STAGE):
            local_row = load_iter * LOAD_ROWS_PER_ITER + load_row_offset
            stage_row = 3 * ROWS_PER_STAGE + local_row
            global_row = chunk_start + warp_row_offset + stage_row
            global_chunk = chunk_base + load_chunk_idx
            cute.copy(
                copy_atom_128,
                mK_128[i_b, global_row, i_h, global_chunk, None],
                sK[0, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
            cute.copy(
                copy_atom_128,
                mQ_128[i_b, global_row, i_h, global_chunk, None],
                sQ[0, warp_col_group, warp_row_group, local_row, load_chunk_idx, None],
            )
        cute.arch.cp_async_commit_group()

        # Process stage 1 (rows 8-15) from buffer 1
        for row in cutlass.range_constexpr(ROWS_PER_STAGE):
            stage_row = ROWS_PER_STAGE + row
            global_row = chunk_start + warp_row_offset + stage_row

            g0 = rG_warp[stage_row, 0].to(cutlass.Float32)
            g1 = rG_warp[stage_row, 1].to(cutlass.Float32)

            if HAS_BIAS:
                g0 = g0 + rBias[0]
                g1 = g1 + rBias[1]

            g0_activated = cutlass.Float32(0.0)
            g1_activated = cutlass.Float32(0.0)

            if USE_LOWER_BOUND:
                sigmoid0 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g0)
                )
                sigmoid1 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g1)
                )
                g0_activated = lower_bound * sigmoid0
                g1_activated = lower_bound * sigmoid1
            else:
                softplus0 = cute.log(cutlass.Float32(1.0) + cute.exp(g0))
                softplus1 = cute.log(cutlass.Float32(1.0) + cute.exp(g1))
                g0_activated = -exp_A * softplus0
                g1_activated = -exp_A * softplus1

            if IS_VARLEN:
                if global_row >= eos:
                    g0_activated = cutlass.Float32(0.0)
                    g1_activated = cutlass.Float32(0.0)

            rAcc[0] = rAcc[0] + g0_activated
            rAcc[1] = rAcc[1] + g1_activated

            cs0 = rAcc[0] * cumsum_scale
            cs1 = rAcc[1] * cumsum_scale

            k0 = sK[1, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            k1 = sK[
                1, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)
            q0 = sQ[1, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            q1 = sQ[
                1, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)

            exp2_cs0 = cute.exp2(cs0)
            exp2_cs1 = cute.exp2(cs1)
            exp2_kg0 = cute.exp2(gk_last_0 - cs0)
            exp2_kg1 = cute.exp2(gk_last_1 - cs1)

            rCsOut[0] = cs0
            rCsOut[1] = cs1
            rKsOut[0] = (k0 * exp2_cs0).to(cutlass.BFloat16)
            rKsOut[1] = (k1 * exp2_cs1).to(cutlass.BFloat16)
            rQsOut[0] = (q0 * exp2_cs0 * attn_scale).to(cutlass.BFloat16)
            rQsOut[1] = (q1 * exp2_cs1 * attn_scale).to(cutlass.BFloat16)
            rKgOut[0] = (k0 * exp2_kg0).to(cutlass.BFloat16)
            rKgOut[1] = (k1 * exp2_kg1).to(cutlass.BFloat16)

            if IS_VARLEN:
                if global_row < eos:
                    cute.autovec_copy(
                        rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])
            else:
                cute.autovec_copy(
                    rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])

        # Stage 2: wait for buf 2, no more prefetch
        cute.arch.cp_async_wait_group(1)  # Wait for stage 2
        cute.arch.sync_threads()

        # Process stage 2 (rows 16-23) from buffer 2
        for row in cutlass.range_constexpr(ROWS_PER_STAGE):
            stage_row = 2 * ROWS_PER_STAGE + row
            global_row = chunk_start + warp_row_offset + stage_row

            g0 = rG_warp[stage_row, 0].to(cutlass.Float32)
            g1 = rG_warp[stage_row, 1].to(cutlass.Float32)

            if HAS_BIAS:
                g0 = g0 + rBias[0]
                g1 = g1 + rBias[1]

            g0_activated = cutlass.Float32(0.0)
            g1_activated = cutlass.Float32(0.0)

            if USE_LOWER_BOUND:
                sigmoid0 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g0)
                )
                sigmoid1 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g1)
                )
                g0_activated = lower_bound * sigmoid0
                g1_activated = lower_bound * sigmoid1
            else:
                softplus0 = cute.log(cutlass.Float32(1.0) + cute.exp(g0))
                softplus1 = cute.log(cutlass.Float32(1.0) + cute.exp(g1))
                g0_activated = -exp_A * softplus0
                g1_activated = -exp_A * softplus1

            if IS_VARLEN:
                if global_row >= eos:
                    g0_activated = cutlass.Float32(0.0)
                    g1_activated = cutlass.Float32(0.0)

            rAcc[0] = rAcc[0] + g0_activated
            rAcc[1] = rAcc[1] + g1_activated

            cs0 = rAcc[0] * cumsum_scale
            cs1 = rAcc[1] * cumsum_scale

            k0 = sK[2, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            k1 = sK[
                2, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)
            q0 = sQ[2, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            q1 = sQ[
                2, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)

            exp2_cs0 = cute.exp2(cs0)
            exp2_cs1 = cute.exp2(cs1)
            exp2_kg0 = cute.exp2(gk_last_0 - cs0)
            exp2_kg1 = cute.exp2(gk_last_1 - cs1)

            rCsOut[0] = cs0
            rCsOut[1] = cs1
            rKsOut[0] = (k0 * exp2_cs0).to(cutlass.BFloat16)
            rKsOut[1] = (k1 * exp2_cs1).to(cutlass.BFloat16)
            rQsOut[0] = (q0 * exp2_cs0 * attn_scale).to(cutlass.BFloat16)
            rQsOut[1] = (q1 * exp2_cs1 * attn_scale).to(cutlass.BFloat16)
            rKgOut[0] = (k0 * exp2_kg0).to(cutlass.BFloat16)
            rKgOut[1] = (k1 * exp2_kg1).to(cutlass.BFloat16)

            if IS_VARLEN:
                if global_row < eos:
                    cute.autovec_copy(
                        rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])
            else:
                cute.autovec_copy(
                    rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])

        # Stage 3: wait for buf 0 (last group)
        cute.arch.cp_async_wait_group(0)  # Wait for all
        cute.arch.sync_threads()

        # Process stage 3 (rows 24-31) from buffer 0
        for row in cutlass.range_constexpr(ROWS_PER_STAGE):
            stage_row = 3 * ROWS_PER_STAGE + row
            global_row = chunk_start + warp_row_offset + stage_row

            g0 = rG_warp[stage_row, 0].to(cutlass.Float32)
            g1 = rG_warp[stage_row, 1].to(cutlass.Float32)

            if HAS_BIAS:
                g0 = g0 + rBias[0]
                g1 = g1 + rBias[1]

            g0_activated = cutlass.Float32(0.0)
            g1_activated = cutlass.Float32(0.0)

            if USE_LOWER_BOUND:
                sigmoid0 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g0)
                )
                sigmoid1 = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-exp_A * g1)
                )
                g0_activated = lower_bound * sigmoid0
                g1_activated = lower_bound * sigmoid1
            else:
                softplus0 = cute.log(cutlass.Float32(1.0) + cute.exp(g0))
                softplus1 = cute.log(cutlass.Float32(1.0) + cute.exp(g1))
                g0_activated = -exp_A * softplus0
                g1_activated = -exp_A * softplus1

            if IS_VARLEN:
                if global_row >= eos:
                    g0_activated = cutlass.Float32(0.0)
                    g1_activated = cutlass.Float32(0.0)

            rAcc[0] = rAcc[0] + g0_activated
            rAcc[1] = rAcc[1] + g1_activated

            cs0 = rAcc[0] * cumsum_scale
            cs1 = rAcc[1] * cumsum_scale

            k0 = sK[0, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            k1 = sK[
                0, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)
            q0 = sQ[0, warp_col_group, warp_row_group, row, read_chunk, read_offset].to(
                cutlass.Float32
            )
            q1 = sQ[
                0, warp_col_group, warp_row_group, row, read_chunk, read_offset + 1
            ].to(cutlass.Float32)

            exp2_cs0 = cute.exp2(cs0)
            exp2_cs1 = cute.exp2(cs1)
            exp2_kg0 = cute.exp2(gk_last_0 - cs0)
            exp2_kg1 = cute.exp2(gk_last_1 - cs1)

            rCsOut[0] = cs0
            rCsOut[1] = cs1
            rKsOut[0] = (k0 * exp2_cs0).to(cutlass.BFloat16)
            rKsOut[1] = (k1 * exp2_cs1).to(cutlass.BFloat16)
            rQsOut[0] = (q0 * exp2_cs0 * attn_scale).to(cutlass.BFloat16)
            rQsOut[1] = (q1 * exp2_cs1 * attn_scale).to(cutlass.BFloat16)
            rKgOut[0] = (k0 * exp2_kg0).to(cutlass.BFloat16)
            rKgOut[1] = (k1 * exp2_kg1).to(cutlass.BFloat16)

            if IS_VARLEN:
                if global_row < eos:
                    cute.autovec_copy(
                        rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(
                        rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                    )
                    cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])
            else:
                cute.autovec_copy(
                    rCsOut, mG_cumsum[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rKsOut, mK_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(
                    rQsOut, mQ_scaled[i_b, global_row, i_h, col_vec, None]
                )
                cute.autovec_copy(rKgOut, mKg[i_b, global_row, i_h, col_vec, None])

        # Store gk_last_exp
        rCsOut[0] = cute.exp2(gk_last_0)
        rCsOut[1] = cute.exp2(gk_last_1)
        cute.autovec_copy(rCsOut, mGk_last_exp[i_b, i_t, i_h, col_vec, None])


# Global cache for compiled kernels
_compiled_kernels = {}


def act_cumsum_scale_fused_v2_vec(
    g: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    A_log: torch.Tensor,
    cumsum_scale: float = 1.0,
    attn_scale: float = 1.0,
    lower_bound: float | None = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple:
    """Fused gate activation + cumsum + scaling kernel with 4-stage pipeline.

    Args:
        g: Raw gate tensor [B, T, H, S] (B=1 for varlen)
        k: Key tensor [B, T, H, S]
        q: Query tensor [B, T, H, S]
        A_log: Log of decay rate [H]
        cumsum_scale: Scale for cumsum (typically RCP_LN2)
        attn_scale: Attention scale (typically 1/sqrt(d))
        lower_bound: If provided, uses safe_gate mode
        dt_bias: Optional bias tensor [H * S]
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen, or None for equal-len
        chunk_indices: Chunk-to-sequence mapping [NT_total, 2] for varlen, or None

    Returns:
        g_cumsum, k_scaled, kg, q_scaled, gk_last_exp
    """
    B, T, H, S = g.shape

    # Determine varlen mode (matching FLA convention)
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1, "Varlen requires B=1"
        if chunk_indices is None:
            from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices

            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
    else:
        assert T % BT == 0, f"T ({T}) must be divisible by {BT}"
        NT = T // BT

    assert S % BS == 0 or S == BS, f"S ({S}) must be divisible by {BS}"
    assert S % VEC_SIZE == 0, f"S ({S}) must be divisible by VEC_SIZE ({VEC_SIZE})"
    assert (
        S % CPASYNC_SIZE == 0
    ), f"S ({S}) must be divisible by CPASYNC_SIZE ({CPASYNC_SIZE})"

    # Determine activation mode and bias mode
    use_lower_bound = lower_bound is not None
    lower_bound_val = lower_bound if use_lower_bound else 0.0
    has_bias = dt_bias is not None

    S_VEC = S // VEC_SIZE
    S_128 = S // CPASYNC_SIZE

    g = g.contiguous().view(B, T, H, S_VEC, VEC_SIZE)
    k_orig = k.contiguous()
    q_orig = q.contiguous()
    k = k_orig.view(B, T, H, S_VEC, VEC_SIZE)
    q = q_orig.view(B, T, H, S_VEC, VEC_SIZE)

    k_128 = k_orig.view(B, T, H, S_128, CPASYNC_SIZE)
    q_128 = q_orig.view(B, T, H, S_128, CPASYNC_SIZE)

    A_log = A_log.contiguous()

    # Prepare dt_bias tensor (or dummy)
    if has_bias:
        assert dt_bias.shape == (
            H * S,
        ), f"dt_bias shape must be [H * S], got {dt_bias.shape}"
        dt_bias_vec = dt_bias.float().contiguous().view(H, S_VEC, VEC_SIZE)
    else:
        # Use empty (no GPU kernel) — dummy is never read when HAS_BIAS=False
        dt_bias_vec = torch.empty(
            H, S_VEC, VEC_SIZE, dtype=torch.float32, device=g.device
        )

    # Prepare varlen tensors (or dummy for equal-len)
    # NO .to(int32) — pass native dtype directly. CuTe DSL handles int64.
    # Dummies use torch.empty (no GPU fill kernel) to avoid pipeline bubbles.
    if is_varlen:
        idx_dtype = cu_seqlens.dtype  # native dtype (typically int64 from FLA)
        cu_seqlens_native = cu_seqlens
        chunk_indices_native = chunk_indices
    else:
        idx_dtype = torch.int64  # match varlen compilation
        cu_seqlens_native = torch.empty(2, dtype=idx_dtype, device=g.device)
        chunk_indices_native = torch.empty(1, 2, dtype=idx_dtype, device=g.device)

    g_cumsum = torch.empty(
        B, T, H, S_VEC, VEC_SIZE, dtype=torch.float32, device=g.device
    )
    k_scaled = torch.empty(
        B, T, H, S_VEC, VEC_SIZE, dtype=torch.bfloat16, device=k.device
    )
    kg = torch.empty(B, T, H, S_VEC, VEC_SIZE, dtype=torch.bfloat16, device=k.device)
    q_scaled = torch.empty(
        B, T, H, S_VEC, VEC_SIZE, dtype=torch.bfloat16, device=q.device
    )
    gk_last_exp = torch.empty(
        B, NT, H, S_VEC, VEC_SIZE, dtype=torch.float32, device=g.device
    )

    # Cache key includes all compile-time flags
    cache_key = (B, T, H, S, "pipe4", use_lower_bound, has_bias, is_varlen, idx_dtype)

    if cache_key not in _compiled_kernels:
        kernel_op = ActCumsumScaleFusedV2VecPipe4()

        os.environ["CUTLASS_NVCC_ARCHS"] = "90a"
        os.environ["NVCC_FLAGS"] = "--ptxas-options=-v,-O3"

        mode_str = "safe_gate (sigmoid)" if use_lower_bound else "standard (softplus)"
        bias_str = " + dt_bias" if has_bias else ""
        varlen_str = " + varlen" if is_varlen else ""
        print(
            f"Compiling kernel with 4-stage pipeline ({mode_str}{bias_str}{varlen_str})..."
        )
        compiled = cute.compile[KeepPTX, KeepCUBIN](
            kernel_op,
            from_dlpack(g, assumed_align=4),
            from_dlpack(k, assumed_align=4),
            from_dlpack(q, assumed_align=4),
            from_dlpack(k_128, assumed_align=16),
            from_dlpack(q_128, assumed_align=16),
            from_dlpack(A_log),
            from_dlpack(dt_bias_vec, assumed_align=4),
            cutlass.Float32(cumsum_scale),
            cutlass.Float32(attn_scale),
            cutlass.Float32(lower_bound_val),
            from_dlpack(g_cumsum, assumed_align=8),
            from_dlpack(k_scaled, assumed_align=4),
            from_dlpack(kg, assumed_align=4),
            from_dlpack(q_scaled, assumed_align=4),
            from_dlpack(gk_last_exp, assumed_align=8),
            from_dlpack(cu_seqlens_native),
            from_dlpack(chunk_indices_native),
            cutlass.Int32(NT),  # Runtime: NT_total
            cutlass.Int32(B),
            cutlass.Int32(T),
            cutlass.Int32(H),
            cutlass.Int32(S),
            use_lower_bound,
            has_bias,
            is_varlen,  # Constexpr
        )

        _compiled_kernels[cache_key] = compiled
        print("Compilation complete!")

        # Generate SASS
        try:
            cubin_files = glob.glob("*.cubin") + glob.glob("*.sm_90*.cubin")
            if cubin_files:
                cubin_file = cubin_files[0]
                result = subprocess.run(
                    ["cuobjdump", "-sass", cubin_file], capture_output=True, text=True
                )
                if result.returncode == 0:
                    sass_file = os.path.join(
                        os.path.dirname(__file__),
                        "act_cumsum_scale_fused_v2_vec_pipe4.sass",
                    )
                    with open(sass_file, "w") as f:
                        f.write(result.stdout)
                    print(f"SASS saved to: {sass_file}")

                    ldg_count = result.stdout.count("LDG.")
                    stg_count = result.stdout.count("STG.")
                    ldgsts_count = result.stdout.count("LDGSTS")
                    print(
                        f"SASS analysis: LDG={ldg_count}, STG={stg_count}, LDGSTS={ldgsts_count}"
                    )

                result = subprocess.run(
                    ["cuobjdump", "-res-usage", cubin_file],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print("\nResource usage:")
                    for line in result.stdout.split("\n"):
                        if (
                            "REG" in line
                            or "SMEM" in line
                            or "reg" in line
                            or "smem" in line
                        ):
                            print(f"  {line}")
        except Exception as e:
            print(f"Could not generate SASS: {e}")

    compiled = _compiled_kernels[cache_key]

    compiled(
        from_dlpack(g, assumed_align=4),
        from_dlpack(k, assumed_align=4),
        from_dlpack(q, assumed_align=4),
        from_dlpack(k_128, assumed_align=16),
        from_dlpack(q_128, assumed_align=16),
        from_dlpack(A_log),
        from_dlpack(dt_bias_vec, assumed_align=4),
        cutlass.Float32(cumsum_scale),
        cutlass.Float32(attn_scale),
        cutlass.Float32(lower_bound_val),
        from_dlpack(g_cumsum, assumed_align=8),
        from_dlpack(k_scaled, assumed_align=4),
        from_dlpack(kg, assumed_align=4),
        from_dlpack(q_scaled, assumed_align=4),
        from_dlpack(gk_last_exp, assumed_align=8),
        from_dlpack(cu_seqlens_native),
        from_dlpack(chunk_indices_native),
        cutlass.Int32(NT),
    )

    g_cumsum = g_cumsum.view(B, T, H, S)
    k_scaled = k_scaled.view(B, T, H, S)
    kg = kg.view(B, T, H, S)
    q_scaled = q_scaled.view(B, T, H, S)
    gk_last_exp = gk_last_exp.view(B, NT, H, S)

    return g_cumsum, k_scaled, kg, q_scaled, gk_last_exp


def test():
    """Test the 4-stage pipeline version."""
    import time

    B, T, H, S = 1, 8192, 96, 128

    print(f"Testing 4-Stage Pipeline Version: B={B}, T={T}, H={H}, S={S}")
    print(
        f"Pipeline config: {NUM_STAGES} stages, {ROWS_PER_STAGE} rows/stage, {NUM_BUFFERS} buffers"
    )

    device = torch.device("cuda")

    torch.manual_seed(42)
    g = torch.randn(B, T, H, S, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, T, H, S, dtype=torch.bfloat16, device=device)
    q = torch.randn(B, T, H, S, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    cumsum_scale = 0.5
    attn_scale = 1.0 / (S**0.5)

    print("\nRunning kernel...")

    # Warmup
    g_cumsum, k_scaled, kg, q_scaled, gk_last_exp = act_cumsum_scale_fused_v2_vec(
        g, k, q, A_log, cumsum_scale, attn_scale
    )
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        g_cumsum, k_scaled, kg, q_scaled, gk_last_exp = act_cumsum_scale_fused_v2_vec(
            g, k, q, A_log, cumsum_scale, attn_scale
        )
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iterations * 1000
    print(f"\nPerformance: {avg_time:.2f} ms per iteration")

    # Verify
    assert not torch.isnan(g_cumsum).any(), "g_cumsum has NaN!"
    assert not torch.isnan(k_scaled).any(), "k_scaled has NaN!"
    assert not torch.isnan(kg).any(), "kg has NaN!"
    assert not torch.isnan(q_scaled).any(), "q_scaled has NaN!"
    assert not torch.isnan(gk_last_exp).any(), "gk_last_exp has NaN!"

    print("Test passed!")


def reference_impl(g, k, q, A_log, cumsum_scale, attn_scale, lower_bound=None):
    """
    Reference PyTorch implementation for precision verification.

    Computes per chunk (BT rows):
    1. g_activated = activation(g, A_log)
       - Standard mode (lower_bound=None): g = -exp(A_log) * softplus(g)
       - Safe gate mode (lower_bound set): g = lower_bound * sigmoid(exp(A_log) * g)
    2. g_cumsum = cumsum(g_activated, dim=1) * cumsum_scale
    3. k_scaled = k * exp2(g_cumsum)
    4. q_scaled = q * exp2(g_cumsum) * attn_scale
    5. kg = k * exp2(gk_last - g_cumsum)  where gk_last = last cumsum per chunk
    6. gk_last_exp = exp2(gk_last)
    """
    B, T, H, S = g.shape
    NT = T // BT

    # Compute in float32 for precision
    g_f32 = g.float()
    k_f32 = k.float()
    q_f32 = q.float()

    exp_A = torch.exp(A_log).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, H, 1]

    g_cumsum_out = torch.zeros(B, T, H, S, dtype=torch.float32, device=g.device)
    k_scaled_out = torch.zeros(B, T, H, S, dtype=torch.bfloat16, device=g.device)
    kg_out = torch.zeros(B, T, H, S, dtype=torch.bfloat16, device=g.device)
    q_scaled_out = torch.zeros(B, T, H, S, dtype=torch.bfloat16, device=g.device)
    gk_last_exp_out = torch.zeros(B, NT, H, S, dtype=torch.float32, device=g.device)

    for chunk_idx in range(NT):
        start = chunk_idx * BT
        end = start + BT

        # Get chunk
        g_chunk = g_f32[:, start:end, :, :]  # [B, BT, H, S]
        k_chunk = k_f32[:, start:end, :, :]
        q_chunk = q_f32[:, start:end, :, :]

        # Activation
        if lower_bound is not None:
            # Safe gate mode: g = lower_bound * sigmoid(exp(A_log) * g)
            sigmoid_g = torch.sigmoid(exp_A * g_chunk)
            g_activated = lower_bound * sigmoid_g
        else:
            # Standard mode: g = -exp(A_log) * softplus(g)
            softplus_g = torch.log(1.0 + torch.exp(g_chunk))
            g_activated = -exp_A * softplus_g

        # Cumsum along time dimension within chunk
        g_cumsum = torch.cumsum(g_activated, dim=1) * cumsum_scale

        # gk_last is the last cumsum value per chunk
        gk_last = g_cumsum[:, -1:, :, :]  # [B, 1, H, S]

        # Compute outputs
        exp2_cs = torch.pow(2.0, g_cumsum)
        exp2_kg = torch.pow(2.0, gk_last - g_cumsum)

        k_scaled = k_chunk * exp2_cs
        q_scaled = q_chunk * exp2_cs * attn_scale
        kg = k_chunk * exp2_kg

        # Store
        g_cumsum_out[:, start:end, :, :] = g_cumsum
        k_scaled_out[:, start:end, :, :] = k_scaled.bfloat16()
        q_scaled_out[:, start:end, :, :] = q_scaled.bfloat16()
        kg_out[:, start:end, :, :] = kg.bfloat16()
        gk_last_exp_out[:, chunk_idx, :, :] = torch.pow(2.0, gk_last.squeeze(1))

    return g_cumsum_out, k_scaled_out, kg_out, q_scaled_out, gk_last_exp_out


def verify_precision():
    """Verify precision against reference implementation with large data."""
    print("=" * 60)
    print("Precision Verification")
    print("=" * 60)

    # Large data size
    B, T, H, S = 1, 8192, 96, 128

    print(f"Testing with: B={B}, T={T}, H={H}, S={S}")
    print(f"Total elements: {B * T * H * S:,}")

    device = torch.device("cuda")

    # Use fixed seed for reproducibility
    torch.manual_seed(12345)

    # Create test inputs (use smaller values to avoid overflow)
    g = torch.randn(B, T, H, S, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, T, H, S, dtype=torch.bfloat16, device=device)
    q = torch.randn(B, T, H, S, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    cumsum_scale = 1.4426950408889634  # RCP_LN2
    attn_scale = 0.125

    print("\nRunning reference implementation...")
    ref_g_cumsum, ref_k_scaled, ref_kg, ref_q_scaled, ref_gk_last_exp = reference_impl(
        g, k, q, A_log, cumsum_scale, attn_scale
    )
    torch.cuda.synchronize()

    print("Running CuTe implementation...")
    cute_g_cumsum, cute_k_scaled, cute_kg, cute_q_scaled, cute_gk_last_exp = (
        act_cumsum_scale_fused_v2_vec(g, k, q, A_log, cumsum_scale, attn_scale)
    )
    torch.cuda.synchronize()

    # Compare results
    def compare_tensors(name, ref, cute, rtol=1e-2, atol=1e-3):
        ref = ref.float()
        cute = cute.float()

        abs_diff = torch.abs(ref - cute)
        rel_diff = abs_diff / (torch.abs(ref) + 1e-8)

        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        mean_rel_diff = rel_diff.mean().item()

        # Count errors
        close = torch.isclose(ref, cute, rtol=rtol, atol=atol)
        num_errors = (~close).sum().item()
        total = ref.numel()
        error_rate = num_errors / total * 100

        passed = error_rate < 1.0  # Allow up to 1% errors
        status = "PASS" if passed else "FAIL"

        print(f"\n{name}:")
        print(f"  Max abs diff: {max_abs_diff:.6e}")
        print(f"  Max rel diff: {max_rel_diff:.6e}")
        print(f"  Mean abs diff: {mean_abs_diff:.6e}")
        print(f"  Mean rel diff: {mean_rel_diff:.6e}")
        print(f"  Errors: {num_errors}/{total} ({error_rate:.4f}%) [{status}]")

        return passed

    print("\n" + "-" * 40)
    print("Comparison Results (rtol=1e-2, atol=1e-3):")
    print("-" * 40)

    all_passed = True
    all_passed &= compare_tensors("g_cumsum (fp32)", ref_g_cumsum, cute_g_cumsum)
    all_passed &= compare_tensors("k_scaled (bf16)", ref_k_scaled, cute_k_scaled)
    all_passed &= compare_tensors("kg (bf16)", ref_kg, cute_kg)
    all_passed &= compare_tensors("q_scaled (bf16)", ref_q_scaled, cute_q_scaled)
    all_passed &= compare_tensors(
        "gk_last_exp (fp32)", ref_gk_last_exp, cute_gk_last_exp
    )

    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: ALL TESTS PASSED!")
    else:
        print("OVERALL: SOME TESTS FAILED!")
    print("=" * 60)

    return all_passed


def benchmark():
    """Benchmark CuTe implementation with 4-stage pipeline."""
    import time

    torch.manual_seed(42)

    # Real workload size
    B, T, H, S = 1, 8192, 96, BS  # S=128 (BS)
    chunk_size = BT
    NT = T // chunk_size

    g = torch.randn(B, T, H, S, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, S, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, S, dtype=torch.bfloat16, device="cuda")
    A_log = torch.randn(H, dtype=torch.float32, device="cuda") * 0.1

    cumsum_scale = 1.4426950408889634  # RCP_LN2
    attn_scale = 0.125

    print(
        f"Benchmark 4-Stage Pipeline: B={B}, T={T}, H={H}, S={S}, chunk_size={chunk_size}"
    )
    print(
        f"Pipeline config: {NUM_STAGES} stages, {ROWS_PER_STAGE} rows/stage, {NUM_BUFFERS} buffers"
    )
    print(f"Total blocks: {(S // BS) * NT * (B * H)}")
    print(f"Threads per block: {THREADS_PER_BLOCK}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = act_cumsum_scale_fused_v2_vec(g, k, q, A_log, cumsum_scale, attn_scale)
    torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking...")
    iterations = 20
    start = time.time()
    for _ in range(iterations):
        _ = act_cumsum_scale_fused_v2_vec(g, k, q, A_log, cumsum_scale, attn_scale)
    torch.cuda.synchronize()
    cute_time = (time.time() - start) / iterations * 1000
    print(f"CuTe v2 Vec2 4-Stage Pipeline:  {cute_time:.4f} ms per iteration")

    # Compute throughput
    total_bytes = g.numel() * 2 * 8  # inputs + outputs roughly
    throughput = total_bytes / (cute_time / 1000) / 1e9
    print(f"Throughput: ~{throughput:.1f} GB/s")

    return cute_time


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            benchmark()
        elif sys.argv[1] == "verify":
            verify_precision()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python script.py [test|benchmark|verify]")
    else:
        test()
