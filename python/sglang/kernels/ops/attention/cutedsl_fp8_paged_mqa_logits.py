# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CuTe DSL FP8 paged MQA logits kernel for Blackwell (SM100).

Architecture:
  - 384 threads: 256 math (2 WGs) + 128 specialized (2 TMA + 2 UMMA)
  - 1 TMA per KV block [128, 128], UMMA iterates 4x K=32
  - 2 warp groups process 2 KV blocks per iteration (kNumMathWarpGroups=2)
  - Q reloaded via TMA pipeline when q_idx (batch) changes
  - Persistent kernel: CTAs iterate through assigned (q_idx, kv_idx) pairs
  - Weights cached in registers: preloaded once per q_idx change (not per KV block)
  - KV Scales loaded via TMA to SMEM (separate pipeline per group, Math consumes)

Merged KV+Scale pipeline:
  - KV data and scales share a single TMA barrier per group
  - TMA loads both KV and Scale under one barrier (combined tx_count)
  - UMMA waits on merged barrier (for KV GEMM), does NOT release
  - Math waits on merged barrier (for scale read), Math releases

Fused KV layout:
  - KV data and scales stored contiguously per physical block:
    [num_phys_blocks, block_kv * (head_dim + 4)] bytes
  - Per block: [KV_all_tokens (block_kv * head_dim bytes)] [Scales (block_kv * 4 bytes)]
  - KV and Scale views are derived inside __call__ using CuTE pointer arithmetic

Scheduler:
  - schedule_meta[sm_idx] = (start_q_idx, start_kv_idx / kNumMathWarpGroups)
  - schedule_meta[sm_idx+1] = end boundary for this CTA
  - fetch_next_task pattern: each warp role independently advances (q_idx, kv_idx)
  - kv_idx in units of KV blocks, advances by kNumMathWarpGroups=2 per step

Dynamic shape support:
  - Model-constant dims (block_kv, head_dim, N, per_token) remain static for codegen
  - Runtime-varying dims (batch_size, num_phys_blocks, max_ctx, max_blocks_per_seq,
    num_ctas) are marked dynamic via mark_compact_shape_dynamic
  - Allows JIT cache reuse across different batch sizes / sequence lengths

Epilogue dtype flows (--acc_dtype / --epi_dtype):

  Flow 1: --acc_dtype fp32 --epi_dtype fp16
    Q(FP8) x K(FP8) -> MMA acc(FP32) -> TMEM(FP32)
      -> LDTM -> Reg(FP32) -> cvt FP16 -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  Flow 2: --acc_dtype fp16 --epi_dtype fp16
    Q(FP8) x K(FP8) -> MMA acc(FP16) -> TMEM(FP16, pack_16b)
      -> LDTM -> Reg(FP16) -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  --output_dtype: fp32 (default), fp16, bf16. Controls logits tensor dtype and final store conversion.
  Default: --acc_dtype fp32 --epi_dtype fp32 --output_dtype fp32
"""

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float16, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


@dsl_user_op
def pack_f16x2(
    a: Float16,
    b: Float16,
    *,
    loc=None,
    ip=None,
) -> Int32:
    f16_ty = Float16.mlir_type
    i32_ty = Int32.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = vector.from_elements(
        vec2_f16,
        (Float16(a).ir_value(loc=loc, ip=ip), Float16(b).ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.bitcast(i32_ty, v, loc=loc, ip=ip))


@dsl_user_op
def unpack_f16x2(
    packed: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float16, Float16]:
    f16_ty = Float16.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = llvm.bitcast(vec2_f16, Int32(packed).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    r0 = Float16(
        vector.extract(v, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    r1 = Float16(
        vector.extract(v, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return r0, r1


@dsl_user_op
def fma_f16x2(
    a: Int32,
    b: Int32,
    c: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


def relu2_fma_f32x2(v0, v1, v2, v3, w0, w1, w2, w3, s0x, s0y, s1x, s1y):
    # ReLU+weighted-accumulate of 4 heads via packed f32x2 (DeepGEMM SM100 pattern).
    # relu(x) = (x + |x|) * 0.5; here we accumulate 2*relu (the 0.5 is folded into
    # the output scale by the caller). abs folds into the add_packed_f32x2 source
    # modifier, so the ReLU stays on the FMA pipe instead of emitting scalar FMNMX
    # on the ALU pipe (the SM100 bottleneck for this kernel).
    r01 = cute.arch.add_packed_f32x2((v0, v1), (cute.math.absf(v0), cute.math.absf(v1)))
    r23 = cute.arch.add_packed_f32x2((v2, v3), (cute.math.absf(v2), cute.math.absf(v3)))
    s0x, s0y = cute.arch.fma_packed_f32x2(r01, (w0, w1), (s0x, s0y), rnd="rn")
    s1x, s1y = cute.arch.fma_packed_f32x2(r23, (w2, w3), (s1x, s1y), rnd="rn")
    return s0x, s0y, s1x, s1y


class FP8MQALogitsKernel:
    """FP8 paged MQA logits kernel for Blackwell (SM100).

    Each CTA processes a range of (q_idx, kv_split) pairs.
    A split = 2 consecutive KV blocks within a sequence (one per warp group).
    Q is shared between warp groups and reloaded when q_idx changes.
    """

    def __init__(
        self,
        block_kv: int = 128,
        phys_block_kv: int = 128,
        num_heads: int = 64,
        head_dim: int = 128,
        next_n: int = 1,
        num_sms: int = 148,
        remove_kv_wait_in_epilogue: bool = False,
        early_tmem_copy: bool = False,
        smem_subpartition_opt: bool = False,
        max_kv_pipeline: bool = False,
        max_umma_pipeline: bool = False,
        max_w_in_reg: int = 0,
        num_epi_subtiles: int = 1,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        enable_pdl: bool = True,
    ):
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.num_blocks_per_mma = block_kv // phys_block_kv
        assert (
            block_kv % phys_block_kv == 0
        ), f"block_kv={block_kv} must be divisible by phys_block_kv={phys_block_kv}"
        assert (
            self.num_blocks_per_mma <= 4
        ), f"num_blocks_per_mma={self.num_blocks_per_mma} exceeds max 4"
        self.remove_kv_wait_in_epilogue = remove_kv_wait_in_epilogue
        self.early_tmem_copy = early_tmem_copy
        self.smem_subpartition_opt = smem_subpartition_opt
        self.max_w_in_reg = max_w_in_reg
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n = next_n
        self.N = next_n * num_heads
        self.num_sms = num_sms
        self.enable_pdl = enable_pdl
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        self.epi_bytes = 2 if epi_dtype == cutlass.Float16 else 4
        # sW stage stride padded to 128-byte SMEM alignment for TMA bulk copy.
        # Without padding, e.g. fp16 + N=32 gives 64B per stage, so stage 1
        # at +64 would be misaligned (TMA requires 128-byte aligned SMEM dest).
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.epi_bytes
        self.output_dtype = output_dtype
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "num_heads // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 2

        self.num_math_threads = 256
        self.num_specialized_threads = 128
        self.threads_per_cta = 384
        self.num_math_warps = 8
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        self.num_q_stages = 3  # 3 stages for Q pipelining across batch sequences

        # TMEM: 512 columns total, each group needs N columns per UMMA stage
        # max_umma_stages = 512 // (2 * N)
        TMEM_COLS = 512
        if max_umma_pipeline:
            self.num_umma_stages = min(2, TMEM_COLS // (2 * self.N))
        else:
            self.num_umma_stages = 1

        if max_kv_pipeline:
            smem_capacity = utils.get_smem_capacity_in_bytes()
            # Reserve ~1 KB for barriers and misc
            SMEM_BUDGET = smem_capacity - 1024
            # KV+Scale per stage (×2 groups):
            #   2 * (block_kv * head_dim * 1B + block_kv * 4B)
            kv_scale_per_stage = 2 * (block_kv * head_dim + block_kv * 4)
            # Q+W per stage: Q is N * head_dim * 1B, W uses padded stride
            qw_per_stage = self.N * head_dim + self.w_stage_stride * self.epi_bytes
            qw_total = qw_per_stage * self.num_q_stages
            self.num_kv_stages = (SMEM_BUDGET - qw_total) // kv_scale_per_stage
        else:
            self.num_kv_stages = 3

        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB),
        # avoiding sub-bank conflicts with UMMA reading sKV.
        # Layout: barriers(~256B) | sKV_0 | sKV_1 | sQ | [pad] | sW | sScales
        if self.smem_subpartition_opt:
            BOUNDARY = 128 * 1024
            used = (
                256
                + 2 * (block_kv * head_dim * self.num_kv_stages)
                + self.N * head_dim * self.num_q_stages
            )
            used = ((used + 127) // 128) * 128
            if used < BOUNDARY:
                self.smem_pad_bytes = ((BOUNDARY - used + 1023) // 1024) * 1024
            else:
                self.smem_pad_bytes = 0
        else:
            self.smem_pad_bytes = 0

        self.acc_dtype = acc_dtype
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            a_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 32

        # Full-K: tile K = head_dim (128), 1 TMA per block
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k  # 4
        full_k = mma_inst_shape_k * mma_inst_tile_k  # 128
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            full_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]

        # KV SMEM: 3 stages per group, each stage holds full [128, 128]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            a_dtype,
            self.num_kv_stages,
        )
        # Q SMEM: 1 stage, holds full [N, 128]
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            b_dtype,
            self.num_q_stages,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
        )

        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,  # Fused KV: [num_phys_blocks, block_bytes] FP8
        b: cute.Tensor,  # Q: [N, head_dim, batch_size]
        weights: cute.Tensor,  # [N, batch_size] (transposed for TMA)
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        context_lens: cute.Tensor,  # [batch_size]
        schedule_meta: cute.Tensor,  # [num_sms+1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # Derive KV and Scale views from fused buffer using CuTE ops.
        # Fused layout per physical block: [KV data (phys_block_kv*head_dim)] [Scales (phys_block_kv*4)]
        phys_block_kv = self.phys_block_kv
        phys_block_bytes = phys_block_kv * (self.head_dim + 4)
        scale_offset_elems = phys_block_kv * self.head_dim  # in FP8 elements

        # Recast fused buffer to FP8 (same 1-byte elements, needed for MMA type inference)
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)

        # Q (b) was passed as uint8 to work around DLPack's lack of float8 support;
        # recast back to FP8 so MMA type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)

        # KV view: [phys_block_kv, head_dim, num_phys_blocks] FP8
        # Each TMA loads one physical block; multiple TMAs fill a compute tile.
        kv_layout = cute.make_layout(
            (phys_block_kv, self.head_dim, num_phys_blocks),
            stride=(self.head_dim, 1, phys_block_bytes),
        )
        a = cute.make_tensor(kv_fp8.iterator, kv_layout)

        # Scale view: offset pointer to scale region, recast FP8 → Float32
        # [phys_block_kv, num_phys_blocks] float32 (after recast)
        scale_fp8_layout = cute.make_layout(
            (phys_block_kv * 4, num_phys_blocks),
            stride=(1, phys_block_bytes),
        )
        scale_fp8 = cute.make_tensor(
            kv_fp8.iterator + scale_offset_elems, scale_fp8_layout
        )
        scales = cute.recast_tensor(scale_fp8, cutlass.Float32)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA for KV (A) — fmha_decode_paged pattern.
        # Build a TMA SMEM layout via tiled_divide on the full compute-tile
        # layout, then select to drop trivial K dim. Atom uses mode [0] as
        # single-tile SMEM layout and (phys, head) as cta_tiler.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_view_layout = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        self.a_tma_view_layout = cute.tiled_divide(
            self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
        )
        # ((tile_M, tile_K), rest_M, rest_K, stages) → drop trivial rest_K
        self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
        # ((tile_M, tile_K), rest_M=num_sub_blocks, stages)
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            a,
            self.a_tma_view_layout[0],  # atom SMEM = single-tile (mode 0)
            (self.phys_block_kv, self.head_dim),
        )

        # TMA for Q (B) — full K=128, L dim = batch_size (unchanged)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA for Weights — [N, batch_size], tile [N], L=batch_size
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.w_stage_stride),
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            self.w_smem_layout_staged.shape[:1],
        )

        # TMA for Scales — [phys_block_kv, num_phys_blocks], tile [phys_block_kv]
        # SMEM holds compute_block_kv scales per stage; filled by
        # num_blocks_per_mma sub-block TMAs at consecutive offsets.
        self.s_smem_layout_staged = cute.make_layout(
            (self.block_kv, self.num_kv_stages)
        )
        s_smem_per_subblock = cute.make_layout((phys_block_kv,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op,
            scales,
            s_smem_per_subblock,
            (phys_block_kv,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        # Per sub-block: phys_block_kv * head_dim (KV) + phys_block_kv * 4 (scales)
        kv_tma_bytes_per_subblock = phys_block_kv * self.head_dim
        scale_tma_bytes_per_subblock = phys_block_kv * 4
        # Total per compute tile = num_blocks_per_mma sub-blocks
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + scale_tma_bytes_per_subblock
        )
        # Q + Weights share barrier (like DeepGEMM)
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        num_ctas = self.num_sms

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            batch_size,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV pool
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = batch_size)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, batch_size]
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,  # Scales TMA coord tensor [block_kv, num_phys_blocks]
        mLogits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        mBlockTable: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        mContextLens: cute.Tensor,  # [batch_size]
        mScheduleMeta: cute.Tensor,  # [num_sms+1, 2] int32
        batch_size: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Warp roles (matches DeepGEMM SM100)
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        is_umma_warp_1 = warp_idx == 11

        # Early schedule metadata load: issue global loads ASAP so their
        # ~200-cycle L2 latency overlaps with subsequent prologue setup
        # (SMEM alloc, TMA partition, MMA fragment creation, etc.)
        NUM_MATH_WG = 2  # kNumMathWarpGroups
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        sm_idx = bidz
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_half = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_half = mScheduleMeta[(sm_idx + 1, 1)]
        # Early mContextLens load: overlap ~200-cycle L2 latency with the
        # entire prologue setup (pipelines, SMEM alloc, TMA partition, etc.)
        # Clamp to avoid OOB when start_q == batch_size (zero-work CTA sentinel).
        # Note: zero-work CTAs get a stale current_num_kv (from the last batch
        # element), but it is never used because has_work will be False.
        start_q_clamped = min(start_q, batch_size - 1)
        current_num_kv = (
            mContextLens[start_q_clamped] + self.block_kv - 1
        ) // self.block_kv

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        block_kv_val = self.block_kv
        num_heads = self.num_heads
        next_n = self.next_n
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer → Math consumer (8 math warps)
        # PipelineTmaAsync: consumer_release uses is_signalling_thread
        # (lane 0 per warp). 8 math warps × 1 lane-0 = 8 arrives.
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        # Both Math WGs share the same pipeline state (advance in lockstep)
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        # UMMA warps observe Q pipeline (wait only, no release)
        # to ensure Q is in SMEM before GEMM. Critical for UMMA warp 1
        # since TMA warp 1 only loads KV1 (not Q).
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        # Merged KV+Scale pipelines (per-group, 3 stages each)
        # Like DeepGEMM: KV data and scales share one barrier.
        # TMA loads both under one barrier. Math is consumer (releases).
        # UMMA also waits on this barrier (for KV GEMM) but does NOT release.
        math_warps_per_group = self.num_math_warps // 2  # 4 warps
        kv_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, math_warps_per_group
        )
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )

        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        # UMMA consumer states (wait only, no release)
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        # Math consumer states (wait + release)
        kv_cons_state_math_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )

        # TMEM — only Math warps (8×32=256) + UMMA warps (2×32=64) = 320 threads
        # TMA warps do NOT participate, so they can start TMA loads earlier.
        # Math warp 0 is the allocator (like fp16_gemm_3's epilogue warp 0),
        # because math warps are the last TMEM consumers (epilogue reads).
        tmem_alloc_num_threads = 320  # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,  # math warp 0 does alloc+free (last TMEM consumer)
            is_two_cta=False,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB)
        # to avoid sub-bank conflicts with UMMA reading sKV from
        # sub-partition 0.
        if cutlass.const_expr(self.smem_pad_bytes > 0):
            _ = smem.allocate(self.smem_pad_bytes)
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        # Scales SMEM: [block_kv, num_kv_stages] float32, per group
        sScales_0 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )
        sScales_1 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): fmha_decode_paged pattern.
        # SMEM view is ((tile), num_sub_blocks, stages) — built in __call__.
        # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
        # GMEM: local_tile by (phys, head), then group first 2 modes into tile.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
        sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(
            mA_mkl,
            (self.phys_block_kv, self.head_dim),
            coord=(None, None, None),
        )
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_0_for_tma,
            cute.group_modes(gA, 0, 2),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_1_for_tma,
            cute.group_modes(gA, 0, 2),
        )

        # Partition Q (B): shared SMEM, L dim = batch_size
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, None)]  # [tma, K, L]

        # Partition Weights: standalone TMA, [N, batch_size] → [N] per stage
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )

        # Partition Scales: explicit sub_blocks + stages dims.
        # Layout (phys_block_kv, num_sub, stages) K-major with custom strides.
        s_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
        sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
        # GMEM: local_tile by phys to match atom's tile size
        gS = cute.local_tile(mS_tma, (self.phys_block_kv,), coord=(None, None))
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_0_for_tma,
            gS,
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_1_for_tma,
            gS,
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)  # shared
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # Staged acc (fp16_gemm_3 pattern): append UMMA stage dim
        # shape: (*acc_shape, STAGE) — dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        # Epilogue setup
        c_layout = utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], num_heads // num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # ===== SCHEDULER: derive values from early-loaded schedule metadata =====
        end_kv_idx = end_kv_half * NUM_MATH_WG

        # Convert start to KV block units (matching DeepGEMM)
        current_q_idx = start_q
        current_kv_idx = start_kv_half * NUM_MATH_WG

        # ===== COMMON SCHEDULER STATE (before warp branches) =====
        # Each warp role independently maintains its own copy of these
        # variables (like DeepGEMM where each role creates its own scheduler).
        # Pre-fetch first task (current_num_kv loaded early above for latency hiding)
        next_q_idx = current_q_idx
        next_kv_idx = current_kv_idx
        next_num_kv = current_num_kv
        # Sentinel: no previous batch (matches DeepGEMM's q_idx = batch_size)
        q_idx = batch_size
        # While-loop termination flag (matches DeepGEMM's fetch_next_task pattern).
        # True if this CTA has work assigned (start != end in schedule_meta).
        has_work = (current_q_idx != end_q_idx) | (current_kv_idx != end_kv_idx)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            # TMA warp 0: loads Q (prefetch) + KV for group 0
            # Matches DeepGEMM's TMA warp with kv_group_idx == 0
            cute.arch.setmaxregister_decrease(24)
            lane_idx = tidx % 32

            # Block table prefetch: 32 lanes cache block indices,
            # distributed via shuffle. Each lane holds num_blocks_per_mma
            # physical block indices per compute tile. (DeepGEMM L233-244)
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use

            # Prefetch first Q before loop (like DeepGEMM line 203-204)
            q_pipeline.producer_acquire(q_prod_state)
            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0, next_q_idx)],
                tBsB[(None, q_prod_state.index)],
                tma_bar_ptr=q_bar,
                mcast_mask=b_mcast_mask,
            )
            cute.copy(
                tma_atom_w,
                tWgW[(None, next_q_idx)],
                tWsW[(None, q_prod_state.index)],
                tma_bar_ptr=q_bar,
            )
            q_prod_state.advance()

            while has_work:
                # fetch_next_task: commit next → current
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Q prefetch: when batch changes, load Q for NEXT batch
                if q_idx != q_idx_old:
                    kv_blk_ptr = cutlass.Int32(32)  # force re-prefetch
                    prefetch_next = q_idx + 1
                    if prefetch_next < end_q_idx:
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, prefetch_next)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[(None, prefetch_next)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()
                    elif prefetch_next == end_q_idx:
                        if end_kv_idx > 0:
                            q_pipeline.producer_acquire(q_prod_state)
                            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, 0, prefetch_next)],
                                tBsB[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                                mcast_mask=b_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_w,
                                tWgW[(None, prefetch_next)],
                                tWsW[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                            )
                            q_prod_state.advance()

                # Block table prefetch for group 0 (like DeepGEMM L233-241).
                # Each lane loads num_blocks_per_mma physical block indices
                # for one compute tile (kv_idx counts compute tiles).
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + lane_idx * NUM_MATH_WG
                    if prefetch_kv < num_kv:
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[(q_idx, base_phys + i)]
                    else:
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = cutlass.Int32(0)

                # Get block indices via shuffle before barrier (like DeepGEMM L244)
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1

                # Load KV + Scale for group 0: num_blocks_per_mma TMAs per tile.
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                stage = kv_prod_state_0.index
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cute.copy(
                        tma_atom_a,
                        tAgA_0[(None, 0, 0, phys_blks[i])],
                        tAsA_0[(None, i, stage)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_s,
                        tSgS_0[(None, 0, phys_blks[i])],
                        tSsS_0[(None, i, stage)],
                        tma_bar_ptr=bar,
                    )
                kv_prod_state_0.advance()

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_tma_warp_1:
            # TMA warp 1: loads KV + Scale for group 1 only
            # Matches DeepGEMM's TMA warp with kv_group_idx == 1
            cute.arch.setmaxregister_decrease(24)
            lane_idx = tidx % 32

            # Block table prefetch for group 1
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use

            while has_work:
                # fetch_next_task: commit next → current
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # New q_idx → force block table re-prefetch
                if q_idx != q_idx_old:
                    kv_blk_ptr = cutlass.Int32(32)

                # Block table prefetch for group 1 (like DeepGEMM L233-241).
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + 1 + lane_idx * NUM_MATH_WG
                    if prefetch_kv < num_kv:
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[(q_idx, base_phys + i)]
                    else:
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = cutlass.Int32(0)

                # Get block indices via shuffle before barrier (like DeepGEMM L244)
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1

                # Load KV + Scale for group 1: num_blocks_per_mma TMAs per tile.
                kv_pipeline_1.producer_acquire(kv_prod_state_1)
                bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                stage = kv_prod_state_1.index
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cute.copy(
                        tma_atom_a,
                        tAgA_1[(None, 0, 0, phys_blks[i])],
                        tAsA_1[(None, i, stage)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_s,
                        tSgS_1[(None, 0, phys_blks[i])],
                        tSsS_1[(None, i, stage)],
                        tma_bar_ptr=bar,
                    )
                kv_prod_state_1.advance()

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_umma_warp_0:
            # UMMA warp for group 0
            # Must wait on Q pipeline: TMA operations with different
            # barriers are NOT visibility-ordered even within the same
            # warp. KV0 barrier arriving does not guarantee Q SMEM
            # writes are visible.
            cute.arch.setmaxregister_decrease(24)

            # TMEM: wait for math warp 0's allocation, retrieve pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            if is_leader_cta:
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_stage_0 = cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Wait for Q pipeline when batch changes
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index

                    # Process KV block for group 0 (kv_idx + 0)
                    # Unconditional UMMA (like DeepGEMM): OOB iterations
                    # compute on garbage data; results written to aligned
                    # padding region in logits buffer.
                    # Wait KV first, then TMEM empty (like DeepGEMM)
                    kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                    umma_pipeline_0.producer_acquire(umma_prod_state_0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage = kv_cons_state_umma_0.index
                    tCtAcc_0 = tCtAcc_base_0[
                        (None, None, None, umma_prod_state_0.index)
                    ]
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_0,
                            tCrA_0[None, None, k_block, kv_stage],
                            tCrB[None, None, k_block, q_stage_0],
                            tCtAcc_0,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # No consumer_release here — Math WG 0 releases
                    kv_cons_state_umma_0.advance()

                    umma_pipeline_0.producer_commit(umma_prod_state_0)
                    umma_prod_state_0.advance()

                    # Advance: inline fetch_next_task
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_umma_warp_1:
            # UMMA warp for group 1
            # Explicitly waits on Q pipeline — critical because TMA warp 1
            # only loads KV1, not Q. Without this wait, UMMA warp 1 can
            # start GEMM before TMA warp 0 finishes loading Q into SMEM.
            cute.arch.setmaxregister_decrease(24)

            # TMEM: wait for umma_warp_0's allocation, retrieve pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            if is_leader_cta:
                num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                q_stage_1 = cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Wait for Q pipeline when batch changes
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional UMMA (like DeepGEMM)
                    # Wait KV first, then TMEM empty (like DeepGEMM)
                    kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                    umma_pipeline_1.producer_acquire(umma_prod_state_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage_1 = kv_cons_state_umma_1.index
                    tCtAcc_1 = tCtAcc_base_1[
                        (None, None, None, umma_prod_state_1.index)
                    ]
                    for k_block in cutlass.range_constexpr(num_k_blocks_1):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_1,
                            tCrA_1[None, None, k_block, kv_stage_1],
                            tCrB[None, None, k_block, q_stage_1],
                            tCtAcc_1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # No consumer_release here — Math WG 1 releases
                    kv_cons_state_umma_1.advance()

                    umma_pipeline_1.producer_commit(umma_prod_state_1)
                    umma_prod_state_1.advance()

                    # Advance: inline fetch_next_task
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_math_warp:
            cute.arch.setmaxregister_increase(240)

            # TMEM: math warp 0 is the allocator; all math warps wait + retrieve
            tmem.allocate(num_tmem_alloc_cols_total)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            local_tidx = tidx % 128
            cC = cute.make_identity_tensor(epi_sub_mn)

            if warpgroup_idx == 0:
                # Math WG 0: process group 0
                # Reference setup (stage 0) for m_coord
                # flat_divide by sub-tile to get sub-tile partitions
                tAcc_0_ref = tCtAcc_base_0[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_0_ref_epi = cute.flat_divide(tAcc_0_ref, epi_sub_mn)
                tiled_copy_ref_0 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_0_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_0 = tiled_copy_ref_0.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_0.partition_D(cC)
                m_coord = tTR_cC[0][0]

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                # Weight register cache: only first NUM_W_IN_REG
                # per next_n slot (like DeepGEMM min(52, kNumHeads)).
                # Remaining weights read from SMEM in epilogue.
                # FP16 weights use half the regs, so we can fit
                # all heads for next_n <= 3.
                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    MAX_NUM_W_IN_REG = 64 if next_n <= 3 else 48
                else:
                    MAX_NUM_W_IN_REG = 64 if next_n == 1 else 40 if next_n >= 4 else 52
                if cutlass.const_expr(self.max_w_in_reg > 0):
                    MAX_NUM_W_IN_REG = self.max_w_in_reg
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_rmem_tensor(NUM_W_IN_REG * next_n, self.epi_dtype)
                q_stage_local = cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Q pipeline consumer: wait for Q+Weights SMEM
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        # Preload first NUM_W_IN_REG weights per slot
                        for t_i in cutlass.range_constexpr(next_n):
                            for w_j in cutlass.range(NUM_W_IN_REG, unroll_full=True):
                                w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                    (t_i * num_heads + w_j, q_stage_local)
                                ]

                    # Process KV block for group 0 (kv_idx + 0)
                    # Unconditional Math (like DeepGEMM): OOB results
                    # written to aligned padding region in logits buffer.
                    kv_pos = kv_idx * block_kv_val + m_coord

                    if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                        # Skip KV wait, rely on UMMA barrier's
                        # transitive visibility.
                        umma_pipeline_0.consumer_wait(umma_cons_state_0)
                    else:
                        # Default: wait KV first to overlap lds with
                        # UMMA computation.
                        kv_pipeline_0.consumer_wait(kv_cons_state_math_0)
                        sc_stage_0 = kv_cons_state_math_0.index
                        scale_val = sScales_0[(m_coord, sc_stage_0)]
                        umma_pipeline_0.consumer_wait(umma_cons_state_0)

                    # --- TMEM sub-tile setup ---
                    # flat_divide accumulator by sub-tile shape;
                    # partition once, then loop over sub-tiles.
                    tCtAcc_c0 = tCtAcc_base_0[
                        (None, None, None, umma_cons_state_0.index)
                    ]
                    tAcc_c0 = tCtAcc_c0[((None, None), 0, 0)]
                    tAcc_c0_epi = cute.flat_divide(tAcc_c0, epi_sub_mn)
                    tc_0 = tcgen05.make_tmem_copy(
                        copy_atom_t2r, tAcc_c0_epi[(None, None, 0, 0)]
                    )
                    tr_0 = tc_0.get_slice(local_tidx)
                    tTR_0 = tr_0.partition_S(tAcc_c0_epi)

                    # --- First sub-tile LDTM + KV release ---
                    if cutlass.const_expr(self.early_tmem_copy):
                        # Issue first sub-tile LDTM early
                        cute.copy(tc_0, tTR_0[(None, None, None, 0, 0)], tTR_rAcc)
                        # Scale LDS + KV release fill latency
                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            sc_stage_0 = kv_cons_state_math_0.index
                            scale_val = sScales_0[(m_coord, sc_stage_0)]
                        kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                        kv_cons_state_math_0.advance()
                        cute.arch.fence_view_async_tmem_load()
                    else:
                        # Default: scale LDS + KV release first
                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            sc_stage_0 = kv_cons_state_math_0.index
                            scale_val = sScales_0[(m_coord, sc_stage_0)]
                        kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                        kv_cons_state_math_0.advance()
                        cute.copy(tc_0, tTR_0[(None, None, None, 0, 0)], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()

                    # --- Sub-tile compute loop ---
                    # Each sub-tile: LDTM.xN → fence → load →
                    # ReLU+FMA. Breaks FMA chain (16→4 per chunk)
                    # and interleaves LDTM with FP32 compute to
                    # reduce ShadowPipeThrottle.
                    # Sub-tiles are within each t-slot (num_heads
                    # // num_epi_subtiles wide). flat_divide yields
                    # next_n * num_epi_subtiles sub-tiles total;
                    # global index = t * num_epi_subtiles + i.
                    subtile_n = num_heads // num_epi_subtiles
                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                        packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
                    for t in cutlass.range_constexpr(next_n):
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps0 = packed_zero
                            ps1 = packed_zero
                        else:
                            s0x = cutlass.Float32(0.0)
                            s0y = cutlass.Float32(0.0)
                            s1x = cutlass.Float32(0.0)
                            s1y = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            # LDTM for sub-tiles 1..N-1
                            # (sub-tile 0 handled above)
                            if t > 0 or i > 0:
                                cute.copy(
                                    tc_0,
                                    tTR_0[
                                        (None, None, None, 0, t * num_epi_subtiles + i)
                                    ],
                                    tTR_rAcc,
                                )
                                cute.arch.fence_view_async_tmem_load()
                            # Release UMMA after last LDTM+fence
                            if t == next_n - 1 and i == num_epi_subtiles - 1:
                                umma_pipeline_0.consumer_release(umma_cons_state_0)
                                umma_cons_state_0.advance()
                            acc_vec = tTR_rAcc.load()
                            # Reg-path: weights from registers
                            reg_h_end = min(
                                subtile_n, max(0, NUM_W_IN_REG - i * subtile_n)
                            )
                            for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                n0 = h
                                h_g = i * subtile_n + h
                                if cutlass.const_expr(
                                    self.epi_dtype == cutlass.Float16
                                ):
                                    pa01 = pack_f16x2(
                                        Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                    )
                                    pa23 = pack_f16x2(
                                        Float16(acc_vec[n0 + 2]),
                                        Float16(acc_vec[n0 + 3]),
                                    )
                                    pa01 = max_f16x2(pa01, packed_zero)
                                    pa23 = max_f16x2(pa23, packed_zero)
                                    r0 = t * NUM_W_IN_REG + h_g
                                    pw01 = pack_f16x2(w_cache[r0], w_cache[r0 + 1])
                                    pw23 = pack_f16x2(w_cache[r0 + 2], w_cache[r0 + 3])
                                    ps0 = fma_f16x2(pa01, pw01, ps0)
                                    ps1 = fma_f16x2(pa23, pw23, ps1)
                                else:
                                    r0 = t * NUM_W_IN_REG + h_g
                                    s0x, s0y, s1x, s1y = relu2_fma_f32x2(
                                        acc_vec[n0],
                                        acc_vec[n0 + 1],
                                        acc_vec[n0 + 2],
                                        acc_vec[n0 + 3],
                                        w_cache[r0],
                                        w_cache[r0 + 1],
                                        w_cache[r0 + 2],
                                        w_cache[r0 + 3],
                                        s0x,
                                        s0y,
                                        s1x,
                                        s1y,
                                    )
                            # SMEM-path: weights from shared mem
                            smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
                            for h in cutlass.range_constexpr(
                                smem_h_start, subtile_n, 4
                            ):
                                n0 = h
                                h_g = i * subtile_n + h
                                if cutlass.const_expr(
                                    self.epi_dtype == cutlass.Float16
                                ):
                                    pa01 = pack_f16x2(
                                        Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                    )
                                    pa23 = pack_f16x2(
                                        Float16(acc_vec[n0 + 2]),
                                        Float16(acc_vec[n0 + 3]),
                                    )
                                    pa01 = max_f16x2(pa01, packed_zero)
                                    pa23 = max_f16x2(pa23, packed_zero)
                                    pw01 = pack_f16x2(
                                        sW[(t * num_heads + h_g, q_stage_local)],
                                        sW[(t * num_heads + h_g + 1, q_stage_local)],
                                    )
                                    pw23 = pack_f16x2(
                                        sW[(t * num_heads + h_g + 2, q_stage_local)],
                                        sW[(t * num_heads + h_g + 3, q_stage_local)],
                                    )
                                    ps0 = fma_f16x2(pa01, pw01, ps0)
                                    ps1 = fma_f16x2(pa23, pw23, ps1)
                                else:
                                    s0x, s0y, s1x, s1y = relu2_fma_f32x2(
                                        acc_vec[n0],
                                        acc_vec[n0 + 1],
                                        acc_vec[n0 + 2],
                                        acc_vec[n0 + 3],
                                        sW[(t * num_heads + h_g, q_stage_local)],
                                        sW[(t * num_heads + h_g + 1, q_stage_local)],
                                        sW[(t * num_heads + h_g + 2, q_stage_local)],
                                        sW[(t * num_heads + h_g + 3, q_stage_local)],
                                        s0x,
                                        s0y,
                                        s1x,
                                        s1y,
                                    )
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps_sum = add_f16x2(ps0, ps1)
                            sum_lo, sum_hi = unpack_f16x2(ps_sum)
                            result_t = sum_lo + sum_hi
                        else:
                            # 0.5 here completes relu(x)=(x+|x|)*0.5 folded across
                            # the packed-f32x2 ReLU accumulation in relu2_fma_f32x2.
                            result_t = (s0x + s0y + s1x + s1y) * cutlass.Float32(0.5)
                        out_row = q_idx * next_n + t
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            mLogits[(out_row, kv_pos)] = self.output_dtype(
                                result_t * Float16(scale_val)
                            )
                        else:
                            mLogits[(out_row, kv_pos)] = self.output_dtype(
                                result_t * scale_val
                            )

                    # Advance: inline fetch_next_task
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

                # Release last Q stage (WG 0)
                if q_idx < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            else:
                # Math WG 1: process group 1
                tAcc_1_ref = tCtAcc_base_1[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_1_ref_epi = cute.flat_divide(tAcc_1_ref, epi_sub_mn)
                tiled_copy_ref_1 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_1_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_1 = tiled_copy_ref_1.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_1.partition_D(cC)
                m_coord = tTR_cC[0][0]

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                # Weight register cache: only first NUM_W_IN_REG
                # per next_n slot (like DeepGEMM min(52, kNumHeads)).
                # FP16 weights use half the regs, so we can fit
                # all heads for next_n <= 3.
                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    MAX_NUM_W_IN_REG = 64 if next_n <= 3 else 48
                else:
                    MAX_NUM_W_IN_REG = 64 if next_n == 1 else 40 if next_n >= 4 else 52
                if cutlass.const_expr(self.max_w_in_reg > 0):
                    MAX_NUM_W_IN_REG = self.max_w_in_reg
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_rmem_tensor(NUM_W_IN_REG * next_n, self.epi_dtype)
                q_stage_local = cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Q pipeline consumer: wait for Q+Weights SMEM
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        # Preload first NUM_W_IN_REG weights per slot
                        for t_i in cutlass.range_constexpr(next_n):
                            for w_j in cutlass.range(NUM_W_IN_REG, unroll_full=True):
                                w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                    (t_i * num_heads + w_j, q_stage_local)
                                ]

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional Math (like DeepGEMM)
                    kv_idx_1 = kv_idx + 1

                    kv_pos = kv_idx_1 * block_kv_val + m_coord

                    if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                        umma_pipeline_1.consumer_wait(umma_cons_state_1)
                    else:
                        kv_pipeline_1.consumer_wait(kv_cons_state_math_1)
                        sc_stage_1 = kv_cons_state_math_1.index
                        scale_val = sScales_1[(m_coord, sc_stage_1)]
                        umma_pipeline_1.consumer_wait(umma_cons_state_1)

                    # --- TMEM sub-tile setup (WG1) ---
                    tCtAcc_c1 = tCtAcc_base_1[
                        (None, None, None, umma_cons_state_1.index)
                    ]
                    tAcc_c1 = tCtAcc_c1[((None, None), 0, 0)]
                    tAcc_c1_epi = cute.flat_divide(tAcc_c1, epi_sub_mn)
                    tc_1 = tcgen05.make_tmem_copy(
                        copy_atom_t2r, tAcc_c1_epi[(None, None, 0, 0)]
                    )
                    tr_1 = tc_1.get_slice(local_tidx)
                    tTR_1 = tr_1.partition_S(tAcc_c1_epi)

                    # --- First sub-tile LDTM + KV release (WG1) ---
                    if cutlass.const_expr(self.early_tmem_copy):
                        cute.copy(tc_1, tTR_1[(None, None, None, 0, 0)], tTR_rAcc)
                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            sc_stage_1 = kv_cons_state_math_1.index
                            scale_val = sScales_1[(m_coord, sc_stage_1)]
                        kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                        kv_cons_state_math_1.advance()
                        cute.arch.fence_view_async_tmem_load()
                    else:
                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            sc_stage_1 = kv_cons_state_math_1.index
                            scale_val = sScales_1[(m_coord, sc_stage_1)]
                        kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                        kv_cons_state_math_1.advance()
                        cute.copy(tc_1, tTR_1[(None, None, None, 0, 0)], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()

                    # --- Sub-tile compute loop (WG1) ---
                    subtile_n = num_heads // num_epi_subtiles
                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                        packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
                    for t in cutlass.range_constexpr(next_n):
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps0 = packed_zero
                            ps1 = packed_zero
                        else:
                            s0x = cutlass.Float32(0.0)
                            s0y = cutlass.Float32(0.0)
                            s1x = cutlass.Float32(0.0)
                            s1y = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            if t > 0 or i > 0:
                                cute.copy(
                                    tc_1,
                                    tTR_1[
                                        (None, None, None, 0, t * num_epi_subtiles + i)
                                    ],
                                    tTR_rAcc,
                                )
                                cute.arch.fence_view_async_tmem_load()
                            if t == next_n - 1 and i == num_epi_subtiles - 1:
                                umma_pipeline_1.consumer_release(umma_cons_state_1)
                                umma_cons_state_1.advance()
                            acc_vec = tTR_rAcc.load()
                            # Reg-path
                            reg_h_end = min(
                                subtile_n, max(0, NUM_W_IN_REG - i * subtile_n)
                            )
                            for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                n0 = h
                                h_g = i * subtile_n + h
                                if cutlass.const_expr(
                                    self.epi_dtype == cutlass.Float16
                                ):
                                    pa01 = pack_f16x2(
                                        Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                    )
                                    pa23 = pack_f16x2(
                                        Float16(acc_vec[n0 + 2]),
                                        Float16(acc_vec[n0 + 3]),
                                    )
                                    pa01 = max_f16x2(pa01, packed_zero)
                                    pa23 = max_f16x2(pa23, packed_zero)
                                    r0 = t * NUM_W_IN_REG + h_g
                                    pw01 = pack_f16x2(w_cache[r0], w_cache[r0 + 1])
                                    pw23 = pack_f16x2(w_cache[r0 + 2], w_cache[r0 + 3])
                                    ps0 = fma_f16x2(pa01, pw01, ps0)
                                    ps1 = fma_f16x2(pa23, pw23, ps1)
                                else:
                                    r0 = t * NUM_W_IN_REG + h_g
                                    s0x, s0y, s1x, s1y = relu2_fma_f32x2(
                                        acc_vec[n0],
                                        acc_vec[n0 + 1],
                                        acc_vec[n0 + 2],
                                        acc_vec[n0 + 3],
                                        w_cache[r0],
                                        w_cache[r0 + 1],
                                        w_cache[r0 + 2],
                                        w_cache[r0 + 3],
                                        s0x,
                                        s0y,
                                        s1x,
                                        s1y,
                                    )
                            # SMEM-path
                            smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
                            for h in cutlass.range_constexpr(
                                smem_h_start, subtile_n, 4
                            ):
                                n0 = h
                                h_g = i * subtile_n + h
                                if cutlass.const_expr(
                                    self.epi_dtype == cutlass.Float16
                                ):
                                    pa01 = pack_f16x2(
                                        Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                    )
                                    pa23 = pack_f16x2(
                                        Float16(acc_vec[n0 + 2]),
                                        Float16(acc_vec[n0 + 3]),
                                    )
                                    pa01 = max_f16x2(pa01, packed_zero)
                                    pa23 = max_f16x2(pa23, packed_zero)
                                    pw01 = pack_f16x2(
                                        sW[(t * num_heads + h_g, q_stage_local)],
                                        sW[(t * num_heads + h_g + 1, q_stage_local)],
                                    )
                                    pw23 = pack_f16x2(
                                        sW[(t * num_heads + h_g + 2, q_stage_local)],
                                        sW[(t * num_heads + h_g + 3, q_stage_local)],
                                    )
                                    ps0 = fma_f16x2(pa01, pw01, ps0)
                                    ps1 = fma_f16x2(pa23, pw23, ps1)
                                else:
                                    s0x, s0y, s1x, s1y = relu2_fma_f32x2(
                                        acc_vec[n0],
                                        acc_vec[n0 + 1],
                                        acc_vec[n0 + 2],
                                        acc_vec[n0 + 3],
                                        sW[(t * num_heads + h_g, q_stage_local)],
                                        sW[(t * num_heads + h_g + 1, q_stage_local)],
                                        sW[(t * num_heads + h_g + 2, q_stage_local)],
                                        sW[(t * num_heads + h_g + 3, q_stage_local)],
                                        s0x,
                                        s0y,
                                        s1x,
                                        s1y,
                                    )
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps_sum = add_f16x2(ps0, ps1)
                            sum_lo, sum_hi = unpack_f16x2(ps_sum)
                            result_t = sum_lo + sum_hi
                        else:
                            # 0.5 here completes relu(x)=(x+|x|)*0.5 folded across
                            # the packed-f32x2 ReLU accumulation in relu2_fma_f32x2.
                            result_t = (s0x + s0y + s1x + s1y) * cutlass.Float32(0.5)
                        out_row = q_idx * next_n + t
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            mLogits[(out_row, kv_pos)] = self.output_dtype(
                                result_t * Float16(scale_val)
                            )
                        else:
                            mLogits[(out_row, kv_pos)] = self.output_dtype(
                                result_t * scale_val
                            )

                    # Advance: inline fetch_next_task
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

                # Release last Q stage (WG 1)
                if q_idx < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            # TMEM dealloc: math warps are allocator + last consumer
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        else:
            cute.arch.setmaxregister_decrease(24)
