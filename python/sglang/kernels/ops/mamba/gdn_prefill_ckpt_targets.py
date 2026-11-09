"""FlashInfer's non-CP Blackwell GDN prefill kernel, storing one state checkpoint per sequence at a given token.

SGLANG_ENABLE_GDN_PREFILL_SPARSE_CHECKPOINTS: prefix-cache tracking reads one checkpoint row per tracked sequence,
while the stock kernel stores one every N tokens for every sequence of the call. This subclass of FlashInfer v0.7.2's
``GatedDeltaNetChunkedKernel`` (``flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py``, sha256 f5db6379519b82b6) overrides the two methods that store checkpoints,
copied from the v0.7.2 tag (d84335d8c5) with three edits each: a sequence's ``cu_checkpoints`` entry is a code,
row * 65536 + chunks (the checkpoint after ``chunks`` * 64 tokens, 0 for none), and the store fires when the chunk
boundary equals that token instead of at every multiple of ``checkpoint_every_n_tokens``; the row is never advanced.
Every other instruction, and so every output, final state and stored row, is the stock kernel's. The base class is
``gdn_prefill_tmem_dealloc.GatedDeltaNetChunkedKernelTmemDealloc`` (sha256 118a7eb45ae602f7): v0.7.2's kernel with
FlashInfer #8854's TMEM dealloc ordering, which overrides ``__init__`` and ``kernel`` only. The launcher is
``flashinfer/gdn_kernels/blackwell/gdn_prefill.py``'s ``chunk_gated_delta_rule_sm100`` (sha256 02da221978b5a2e1) with this class and its own cache name.
Generated from the v0.7.2 tag; regenerate rather than edit.
"""

import functools
import hashlib
import math

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flashinfer.cute_dsl.utils import get_num_sm
from flashinfer.gdn_kernels import cute_dsl_cache_naming as _naming
from flashinfer.gdn_kernels import device_target as _target
from flashinfer.gdn_kernels.blackwell import gated_delta_net_chunked as _fi_chunked
from flashinfer.gdn_kernels.blackwell import gated_delta_net_tile_scheduler as _fi_sched
from flashinfer.gdn_kernels.blackwell import gdn_prefill as _fi_prefill
from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

# Raises ImportError unless FlashInfer's GDN prefill files are v0.7.2's.
from sglang.kernels.ops.mamba import gdn_prefill_tmem_dealloc as _fixed

FI_CHUNKED_SHA256 = "f5db6379519b82b6e293d1f95560fa75af3a19a59c5aeb3aa00cc798df3957bc"
FI_PREFILL_SHA256 = "02da221978b5a2e1435b7a9a5edbb96ee7abcdcf80bc77f403ae9fa9d09fa3ec"
FIXED_SHA256 = "118a7eb45ae602f790342d5f1f657d1a787a2d1645b533094ca959e5a555143d"
CHECKPOINT_ROW_STRIDE = 65536


def _file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# The copied methods must run against the exact class they were copied from.
if _file_sha256(_fi_chunked.__file__) != FI_CHUNKED_SHA256:
    raise ImportError(
        f"{_fi_chunked.__file__} is not FlashInfer v0.7.2's (sha256 {_file_sha256(_fi_chunked.__file__)[:16]})"
    )
if _file_sha256(_fi_prefill.__file__) != FI_PREFILL_SHA256:
    raise ImportError(f"{_fi_prefill.__file__} is not FlashInfer v0.7.2's")
# The base class: v0.7.2's kernel with FlashInfer #8854 (its __init__ and kernel call the methods copied below).
if _file_sha256(_fixed.__file__) != FIXED_SHA256:
    raise ImportError(f"{_fixed.__file__} is not the #8854 module this subclass was generated against")

# The copied methods resolve the source module's globals (cute, cutlass, pipeline, tcgen05, utils, ...).
globals().update(
    {k: v for k, v in vars(_fi_chunked).items() if not (k.startswith("__") and k.endswith("__"))}
)


class GatedDeltaNetChunkedCheckpointTargetsKernel(_fixed.GatedDeltaNetChunkedKernelTmemDealloc):
    """``GatedDeltaNetChunkedKernel`` storing at most one checkpoint per sequence, at its coded token."""

    @cute.jit
    def _store_final_state(
        self,
        tidx,
        # full output-state GMEM tensor (DK, DV, (h_r, h_qv), B) fp32
        mS_out,
        mS_indices,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        # MMA -> CG1 consumer; waited+released inside this method
        kv_acc_consumer,
        seqlen_b,
        mS_checkpoints,
        checkpoint_offset,
        checkpoint_every_n_tokens,
    ):
        """Store final recurrent state from TMEM (fp32) to GMEM mS_out.

        Waits for the last GEMM-7 (kv_acc) to complete, reads state TMEM -> registers,
        writes registers -> GMEM fp32, then releases the consumer handle.
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM layout (mirrors compute_group_1 setup)
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState
        )
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )

        # TMEM -> registers  (Ld32x32b)
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.acc_dtype)
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        # Wait for last GEMM-7 to finish.
        ck_row = checkpoint_offset // CHECKPOINT_ROW_STRIDE
        ck_target = cutlass.Int32(checkpoint_offset - ck_row * CHECKPOINT_ROW_STRIDE) * self.b_t
        kv_acc_handle = kv_acc_consumer.wait_and_advance()

        for sub in cutlass.range(tTR_rState.shape[2]):
            cute.copy(
                tiled_state_t2r,
                tTR_tCtState[None, 0, sub, kv_acc_handle.index],
                tTR_rState[None, 0, sub],
            )
            if cutlass.const_expr(self.acc_dtype != self.state_dtype):
                tRG_rState[None, 0, sub].store(
                    tTR_rState[None, 0, sub].load().to(self.state_dtype)
                )
            else:
                tRG_rState = tTR_rState
            if cutlass.const_expr(self.enable_checkpoints):
                if seqlen_b == ck_target:
                    num_valid_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                    if num_valid_chunks_b % 2 == 0:
                        gS_checkpoints = cute.flat_divide(
                            mS_checkpoints[None, None, head_idx, ck_row],
                            (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                        )[None, None, 0, 0]
                        tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                        cute.autovec_copy(
                            tRG_rState[None, 0, sub],
                            tSgCheckpoints[None, 0, sub],
                            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                        )
            if cutlass.const_expr(self.store_final_state):
                if cutlass.const_expr(mS_indices is not None):
                    state_row = cutlass.Int32(mS_indices[batch_idx])
                else:
                    state_row = batch_idx
                gS_out = cute.flat_divide(
                    mS_out[None, None, head_idx, state_row],
                    (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                )[None, None, 0, 0]
                tRG_tCgState = thr_state_t2r.partition_D(gS_out)
                cute.autovec_copy(
                    tRG_rState[None, 0, sub],
                    tRG_tCgState[None, 0, sub],
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )
        kv_acc_handle.release()
        return kv_acc_consumer

    @cute.jit
    def compute_group_1_chunk(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        checkpoint_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        cutlass.Int32,
    ]:
        """Warps 4-7: process one chunk from the caller-owned chunk stream."""
        sV, sCumsumlog, sCumprod, sBeta, sO = smem_args
        mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens = checkpoint_args
        ck_row = checkpoint_offset // CHECKPOINT_ROW_STRIDE
        ck_target = cutlass.Int32(checkpoint_offset - ck_row * CHECKPOINT_ROW_STRIDE) * self.b_t
        (
            load_v_consumer,
            load_gate_consumer,
            cg1_shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            vks_ready_producer,
            nv_ready_producer,
            decay_v_ready_producer,
            o_store_producer,
        ) = pipeline_args
        tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv = mma_args
        chunk_iter, num_pairs_b, head_idx, seqlen_b, is_first_chunk = work_args

        # ------------------------------------------------------------------
        # Preamble (identical to compute_group_1)
        # ------------------------------------------------------------------
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState
        )
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        atom_state_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        tiled_state_r2t = tcgen05.make_tmem_copy(atom_state_r2t, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.acc_dtype)
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )
        tCtState_inp_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState_inp
        )
        tCcState_inp = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        atom_state_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.io_dtype
        )
        tCtState_inp_for_r2t = tCtState_inp_mn_view[None, None, 0]
        tiled_state_inp_r2t = tcgen05.make_tmem_copy(
            atom_state_inp_r2t, tCtState_inp_for_r2t
        )
        thr_state_inp_r2t = tiled_state_inp_r2t.get_slice(cg1_tidx)
        tRT_tCcState_inp = thr_state_inp_r2t.partition_S(tCcState_inp)
        tRT_tCtState_inp = thr_state_inp_r2t.partition_D(tCtState_inp_mn_view)
        tRT_rState_inp = cute.make_rmem_tensor_like(tRT_tCcState_inp, self.io_dtype)

        qkv_acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(qkv_acc_shape)
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_cg1_shared_acc_offset,
            cute.flat_product(
                tCtShared_fake.layout,
                cute.make_layout((self.tmem_cg1_shared_acc_stages,)),
            ),
        )
        tCtShared_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtShared
        )
        tCcShared = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtShared_for_t2r = tCtShared[(None, None), 0, 0, 0]
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tCtShared_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg1_tidx)
        tTR_tCtShared = thr_shared_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcShared = thr_shared_t2r.partition_D(tCcShared)

        # Dedicated full-tile KS t2r (separate atom so it can be tuned
        # independently of atom_shared_t2r used by NV/decay_v reads).
        atom_ks_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_ks_t2r = tcgen05.make_tmem_copy(atom_ks_t2r, tCtShared_for_t2r)
        thr_ks_t2r = tiled_ks_t2r.get_slice(cg1_tidx)
        tTR_tCtKS = thr_ks_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcKS = thr_ks_t2r.partition_D(tCcShared)

        qkv_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(qkv_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )
        tCtShared_inp_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtShared_inp
        )
        tCcShared_inp = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        atom_shared_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tCtShared_inp_for_r2t = tCtShared_inp_mn_view[None, None, 0]
        tiled_shared_inp_r2t = tcgen05.make_tmem_copy(
            atom_shared_inp_r2t, tCtShared_inp_for_r2t
        )
        thr_shared_inp_r2t = tiled_shared_inp_r2t.get_slice(cg1_tidx)
        tRT_tCtShared_inp = thr_shared_inp_r2t.partition_D(tCtShared_inp_mn_view)

        # Dedicated full-tile VKS r2t (separate atom so it can be tuned
        # independently of atom_shared_inp_r2t used by NV/decay_v writes).
        atom_vks_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tiled_vks_r2t = tcgen05.make_tmem_copy(atom_vks_r2t, tCtShared_inp_for_r2t)
        thr_vks_r2t = tiled_vks_r2t.get_slice(cg1_tidx)
        tRT_tCtVKS_inp = thr_vks_r2t.partition_D(tCtShared_inp_mn_view)

        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )
        tCtQState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtQState
        )
        tCcQState = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        atom_qs_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        atom_qs_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtQState_for_t2r = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_t2r = tcgen05.make_tmem_copy(atom_qs_t2r, tCtQState_for_t2r)
        tCtQState_for_r2t = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_r2t = tcgen05.make_tmem_copy(atom_qs_r2t, tCtQState_for_r2t)
        thr_qs_t2r = tiled_qs_t2r.get_slice(cg1_tidx)
        thr_qs_r2t = tiled_qs_r2t.get_slice(cg1_tidx)
        tTR_tCtQS = thr_qs_t2r.partition_S(tCtQState_mn_view)
        tTR_tCcQS = thr_qs_t2r.partition_D(tCcQState)
        tRT_tCtQS = thr_qs_r2t.partition_D(tCtQState_mn_view)
        tTR_rQS = cute.make_rmem_tensor_like(tTR_tCcQS, self.acc_dtype)

        tRT_tCcV = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        atom_v_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_v_s2r = cute.make_tiled_copy_S(
            atom_v_s2r,
            tiled_shared_inp_r2t,
        )
        thr_v_s2r = tiled_v_s2r.get_slice(cg1_tidx)

        atom_o_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_o_t2r = tcgen05.make_tmem_copy(atom_o_t2r, tCtQState_for_t2r)
        thr_o_t2r = tiled_o_t2r.get_slice(cg1_tidx)
        tTR_tOtO = thr_o_t2r.partition_S(tCtQState_mn_view)
        tTR_tOcO = thr_o_t2r.partition_D(tCcQState)
        atom_o_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_o_r2s = cute.make_tiled_copy_D(atom_o_r2s, tiled_o_t2r)
        thr_o_r2s = tiled_o_r2s.get_slice(cg1_tidx)
        tCsO = thr_o_r2s.partition_D(sO)

        sub_tile_size = 32
        sV_vt_view = utils.gemm.sm100.transform_partitioned_tensor_layout(sV)
        tCsV = thr_v_s2r.partition_S(sV_vt_view)

        rCumprod = cute.make_rmem_tensor((1, cute.size(tTR_tCcShared)), self.acc_dtype)
        tGrCumprod = thr_shared_t2r.partition_D(rCumprod)
        tGrDecayScale = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
        # Pair membership is derived from the caller-provided chunk index.
        is_pair_first = (chunk_iter & 1) == 0

        valid_state = is_first_chunk == False  # noqa: E712
        if cutlass.const_expr(self.use_initial_state):
            valid_state = True
            if is_pair_first:
                kv_acc_producer.advance()
                kv_acc_producer.advance()

        # Only the total decay is needed to finish the state update.  Defer
        # the per-row gate work until the previous state has been published
        # and decayed, keeping its register fragment in one contiguous region.
        gate_handle = load_gate_consumer.wait_and_advance()
        # valid_len = max(0, min(self.b_t, seqlen_b - chunk_iter * self.b_t))
        # gamma_end = 1.0 if valid_len == 0 else sCumprod[valid_len - 1, 0, gate_handle.index]
        # OOB alpha is padded with 1 before the prefix scan, so the last physical slot equals gamma_end.
        cumprod_total = sCumprod[self.b_t - 1, 0, gate_handle.index]

        kv_prev_handle = kv_acc_consumer.current_handle()
        if valid_state:
            kv_prev_handle = kv_acc_consumer.wait_and_advance()
            # No empty-stage wait is needed before reusing state_inp.  For the
            # first publication the stage is initialized empty.  Thereafter,
            # CG1 reaches this point only after waiting for the preceding NewV;
            # the MMA warp releases state_inp before publishing that NewV.
            state_inp_handle = state_inp_ready_producer.current_handle()
            state_inp_ready_producer.advance()
            cute.copy(
                tiled_state_t2r,
                tTR_tCtState[None, 0, None, kv_prev_handle.index],
                tTR_rState[None, 0, None],
            )
            tRT_rState_inp[None, 0, None].store(
                tTR_rState[None, 0, None].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_state_inp_r2t,
                tRT_rState_inp[None, 0, None],
                tRT_tCtState_inp[None, 0, None, state_inp_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
            state_inp_handle.commit()
            checkpoint_token = self.b_t * chunk_iter
            if cutlass.const_expr(self.enable_checkpoints):
                if checkpoint_token > 0:
                    if checkpoint_token <= seqlen_b:
                        if checkpoint_token == ck_target:
                            gS_checkpoints = cute.flat_divide(
                                mS_checkpoints[
                                    None,
                                    None,
                                    head_idx,
                                    ck_row,
                                ],
                                (
                                    self.mma_tiler_kv[0],
                                    self.mma_tiler_kv[1],
                                ),
                            )[None, None, 0, 0]
                            tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                            if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                                tRG_rState[None, 0, None].store(
                                    tTR_rState[None, 0, None]
                                    .load()
                                    .to(self.state_dtype)
                                )
                            else:
                                tRG_rState = tTR_rState
                            cute.autovec_copy(
                                tRG_rState[None, 0, None],
                                tSgCheckpoints[None, 0, None],
                                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                            )
            for k in cutlass.range(cute.size(tTR_rState), vectorize=True):
                tTR_rState[k] = tTR_rState[k] * cumprod_total
            cute.copy(
                tiled_state_r2t,
                tTR_rState[None, 0, None],
                tRT_tCtState[None, 0, None, kv_prev_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
            kv_prev_handle.release()
        for k in cutlass.range_constexpr(cute.size(tTR_tCcShared)):
            coord = tTR_tCcShared[k]
            tGrCumprod[k] = sCumprod[coord[1], 0, gate_handle.index]
        last_cumsumlog = sCumsumlog[self.b_t - 1, 0, gate_handle.index]
        for k in cutlass.range_constexpr(0, cute.size(tTR_tCcShared), 2):
            coord0 = tTR_tCcShared[k]
            coord1 = tTR_tCcShared[k + 1]
            decay_diff = cute.arch.add_packed_f32x2(
                (last_cumsumlog, last_cumsumlog),
                (
                    -sCumsumlog[coord0[1], 0, gate_handle.index],
                    -sCumsumlog[coord1[1], 0, gate_handle.index],
                ),
                ftz=False,
                rnd="rn",
            )
            tGrDecayScale[k] = cute.math.exp2(decay_diff[0], fastmath=True)
            tGrDecayScale[k + 1] = cute.math.exp2(decay_diff[1], fastmath=True)
        gate_handle.release()

        vks_handle = vks_ready_producer.current_handle()
        vks_ready_producer.advance()
        v_handle = load_v_consumer.wait_and_advance()
        tRT_rV = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)
        tCrV = tiled_v_s2r.retile(tRT_rV)
        # Always publish V through fixed TMEM slot 0. The SMEM V ring cursor
        # survives persistent work boundaries, so a new work item cannot
        # assume that its first V tile resides in SMEM stage 0.
        cute.copy(
            tiled_v_s2r,
            tCsV[None, None, None, v_handle.index],
            tCrV,
        )

        if valid_state:
            ks_handle = cg1_shared_acc_consumer.wait_and_advance()
            tTR_rKS = cute.make_rmem_tensor_like(tTR_tCcKS, self.acc_dtype)
            cute.copy(
                tiled_ks_t2r,
                tTR_tCtKS[None, None, None, ks_handle.index],
                tTR_rKS,
            )
            for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
                tTR_rKS[k] = tTR_rKS[k] * tGrCumprod[k]
            ks_handle.release()
            for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
                tRT_rV[k] = tRT_rV[k] - tTR_rKS[k].to(self.io_dtype)
        cute.copy(
            tiled_vks_r2t,
            tRT_rV,
            tRT_tCtVKS_inp[None, None, None, 0],
        )
        cute.arch.fence_view_async_tmem_store()
        vks_handle.commit()

        if valid_state:
            qs_handle = q_state_acc_consumer.wait_and_advance()
            cute.copy(
                tiled_qs_t2r,
                tTR_tCtQS[None, None, 0, qs_handle.index],
                tTR_rQS[None, None, 0],
            )
            for k in cutlass.range(cute.size(tTR_rQS), vectorize=True):
                tTR_rQS[k] = tTR_rQS[k] * tGrCumprod[k] * scale
            cute.copy(
                tiled_qs_r2t,
                tTR_rQS[None, None, 0],
                tRT_tCtQS[None, None, 0, qs_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
            qs_handle.release()

        nv_handle = cg1_shared_acc_consumer.wait_and_advance()
        v_handle.release()
        tTR_rNv = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
        tTR_rNv_inp = cute.make_rmem_tensor_like(tTR_rNv, self.io_dtype)
        for sub in cutlass.range(tTR_rNv.shape[1]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, sub, 0, nv_handle.index],
                tTR_rNv[None, sub, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rNv[None, sub, 0].load().to(self.io_dtype)
            )
        nv_handle.release()

        tTR_rDv = tTR_rNv
        for sub in cutlass.range(tTR_rDv.shape[1]):
            for k in cutlass.range(sub_tile_size, vectorize=True):
                tTR_rDv[k, sub, 0] = tTR_rDv[k, sub, 0] * tGrDecayScale[k, sub, 0]

        nv_ready_handle = nv_ready_producer.current_handle()
        nv_ready_producer.advance()
        decay_v_ready_handle = decay_v_ready_producer.current_handle()
        decay_v_ready_producer.advance()
        # Both chunks publish NV for GEMM6/QKV before decayV for GEMM7/KV.
        for sub in cutlass.range(tTR_rDv.shape[1]):
            cute.copy(
                tiled_shared_inp_r2t,
                tTR_rNv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rDv[None, sub, 0].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_shared_inp_r2t,
                tTR_rNv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, 1],
            )
        cute.arch.fence_view_async_tmem_store()
        nv_ready_handle.commit()
        decay_v_ready_handle.commit()

        # Drain this chunk's output at the end of the same chunk.  This keeps
        # O ownership uniform and removes the first/last pending-output cases.
        o_handle = o_store_producer.acquire_and_advance()
        o_qs_handle = q_state_acc_consumer.wait_and_advance()
        tTR_tOrO = cute.make_rmem_tensor_like(tTR_tOcO, self.acc_dtype)
        tTR_rO_out = cute.make_rmem_tensor_like(tTR_tOrO, self.io_dtype)
        tRS_tOrO = tiled_o_r2s.retile(tTR_rO_out)
        cute.copy(
            tiled_o_t2r,
            tTR_tOtO[None, None, None, o_qs_handle.index],
            tTR_tOrO,
        )
        tTR_rO_out.store(tTR_tOrO.load().to(self.io_dtype))
        cute.copy(
            tiled_o_r2s,
            tRS_tOrO,
            tCsO[None, None, None, o_handle.index],
        )
        cute.arch.fence_view_async_shared()
        o_qs_handle.release()
        o_handle.commit()

        return (
            load_v_consumer,
            load_gate_consumer,
            cg1_shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            vks_ready_producer,
            nv_ready_producer,
            decay_v_ready_producer,
            o_store_producer,
            checkpoint_offset,
        )


_CUTE_DSL_MODULE = "sglang_gdn_prefill_ckpt_targets"


def _kernel_source_files() -> tuple:
    return (__file__, _fi_chunked.__file__, _fi_sched.__file__, _fi_prefill.__file__, _fixed.__file__)


@functools.cache
def _get_compiled_cache(*key):
    return {}


def chunk_gated_delta_rule_sm100_checkpoint_targets(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    output_state: torch.Tensor,
    scale: float,
    checkpoint_codes: torch.Tensor,
    output_checkpoints: torch.Tensor,
) -> None:
    """``chunk_gated_delta_rule_sm100`` with packed initial and final states and coded checkpoints:
    ``checkpoint_codes`` [num_seqs] int64, row * 65536 + chunks; ``output_checkpoints`` [rows, HO, DK, DK]."""
    HQ = q.size(1)
    HV = v.size(1)
    DK = q.size(2)
    is_GQA = HQ >= HV
    io_dtype = _fi_prefill._cutlass_io_dtype(q.dtype)
    state_torch_dtype = initial_state.dtype
    state_dtype = _fi_prefill._cutlass_state_dtype(state_torch_dtype)
    B = cu_seqlens.size(0) - 1
    target = _target.gdn_device_target(q.device)
    num_sm = get_num_sm(q.device)
    cache_key = (
        target.compile_key,
        num_sm,
        str(q.dtype),
        str(state_torch_dtype),
        HQ,
        HV,
        is_GQA,
        True,
        True,
        True,
        False,
        str(cu_seqlens.dtype),
        "none",
        str(checkpoint_codes.dtype),
        None,
        None,
    )
    cache = _get_compiled_cache(*cache_key)
    if "compiled" not in cache:
        gdn = GatedDeltaNetChunkedCheckpointTargetsKernel(
            io_dtype=io_dtype,
            inverse_dtype=io_dtype,
            acc_dtype=cutlass.Float32,
            state_dtype=state_dtype,
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(128, 64, 128),
            mma_tiler_qkv=(128, 64, 64),
            mma_tiler_kv=(128, 128, 64),
            max_active_clusters=num_sm,
            num_sm=num_sm,
            is_GQA=is_GQA,
            use_initial_state=True,
            store_final_state=True,
            enable_checkpoints=True,
            is_persistent=True,
        )
        tensors = []
        for t in (q, k, v, output):
            c = from_dlpack(t, assumed_align=16)
            c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
            tensors.append(c)
        q_cute, k_cute, v_cute, o_cute = tensors
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        s_in_cute = from_dlpack(initial_state, assumed_align=16)
        _fi_prefill._mark_state_layout(s_in_cute, False, DK)
        s_out_cute = from_dlpack(output_state, assumed_align=16)
        _fi_prefill._mark_state_layout(s_out_cute, False, DK)
        s_ck_cute = from_dlpack(output_checkpoints, assumed_align=16)
        s_ck_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
            mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
        )
        codes_cute = from_dlpack(checkpoint_codes, assumed_align=4).mark_layout_dynamic()
        workspace = torch.empty(
            _fi_chunked.GatedDeltaNetChunkedKernel.get_workspace_size(num_sm, B, HQ, HV, True),
            dtype=torch.int8,
            device=q.device,
        )
        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
        compiled = build_and_load_cute_dsl_kernel(
            _CUTE_DSL_MODULE,
            _naming.make_kernel_name(_target.target_arch(target.compile_key), *cache_key[1:]),
            lambda: cute.compile[
                _target.gdn_compile_options(q.device, cute.EnableTVMFFI(True), cute.OptLevel(3))
            ](
                gdn,
                q_cute,
                k_cute,
                v_cute,
                gate_cute,
                beta_cute,
                o_cute,
                cu_seqlens_cute,
                s_in_cute,
                s_out_cute,
                None,
                s_ck_cute,
                codes_cute,
                64,
                scale,
                from_dlpack(workspace, assumed_align=16),
                stream,
            ),
            extra_key_files=_kernel_source_files(),
        )
        cache["compiled"] = compiled
        cache["num_sm"] = num_sm
    compiled = cache["compiled"]
    # Fresh per call, as FlashInfer: the kernel writes TMA descriptors into it on every launch.
    workspace = torch.empty(
        _fi_chunked.GatedDeltaNetChunkedKernel.get_workspace_size(cache["num_sm"], B, HQ, HV, True),
        dtype=torch.int8,
        device=q.device,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        None,
        output_checkpoints,
        checkpoint_codes,
        64,
        scale,
        workspace,
        stream,
    )
