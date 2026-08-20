# Vendored from flashinfer 0.6.18.dev20260807 (SM100 GDN CP prefill closure);
# pending a FlashInfer release that ships it.
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Context-parallel Gated Delta Net prefill kernel for Blackwell SM100.

Each CTA processes one CP chunk as a recurrence of 64-token blocks. It follows
the optimized non-CP SM100 state pipeline, but replaces the KK and hierarchical
inverse path with a TMA load of the signed, beta-folded inverse produced by CP
preprocessing. CG0 applies the gate sandwich and publishes the same A-inverse
operand contract consumed by the common recurrence.

Warp assignments (12 warps):
  warps 0-3 : transform T and materialize gamma-scaled QK
  warps 4-7 : recurrent state and output epilogues
  warp 8    : issue paired QK UTCMMA
  warp 9    : TMA load Q, K, V, and T
  warp 10   : issue state/output UTCMMA
  warp 11   : preprocess gates and store O
"""

import math
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import TensorMapManager, TensorMapUpdateMode

# ---------------------------------------------------------------------------
# cutlass-dsl 4.4.2 compatibility: TmaInfo was removed; make_tiled_tma_atom_*
# now returns a plain (CopyAtom, Tensor) tuple instead of TmaInfo.
# ---------------------------------------------------------------------------
try:
    from cutlass.cute.nvgpu.cpasync import TmaInfo
except ImportError:
    from cutlass.base_dsl import extract_mlir_attributes as _ema
    from cutlass.base_dsl import extract_mlir_values as _emv
    from cutlass.base_dsl import get_mlir_types as _gmt
    from cutlass.base_dsl import new_from_mlir_values as _nfmv

    class TmaInfo:  # type: ignore[no-redef]
        """Compatibility shim replacing cpasync.TmaInfo for cutlass-dsl >= 4.4.2."""

        def __init__(self, atom, tma_tensor, smem_layout=None):
            self._atom = atom
            self._tma_tensor = tma_tensor

        @property
        def atom(self):
            return self._atom

        @property
        def tma_tensor(self):
            return self._tma_tensor

        def __extract_mlir_values__(self):
            return _emv(self._atom) + _emv(self._tma_tensor)

        def __extract_mlir_attributes__(self):
            return _ema(self._atom) + _ema(self._tma_tensor)

        def __new_from_mlir_values__(self, values):
            n = len(_gmt(self._atom))
            return TmaInfo(
                _nfmv(self._atom, values[:n]),
                _nfmv(self._tma_tensor, values[n:]),
            )

        def __iter__(self):
            yield self._atom
            yield self._tma_tensor

        def __getitem__(self, i):
            return (self._atom, self._tma_tensor)[i]

        def __len__(self):
            return 2


def _wrap_tma(ret):
    """Wrap make_tiled_tma_atom_* return value in TmaInfo if not already."""
    if isinstance(ret, TmaInfo):
        return ret
    # 4.4.2: returns (CopyAtom, Tensor) tuple
    return TmaInfo(ret[0], ret[1])


from sglang.kernels.ops.attention.gdn_cp_prefill.custom_compile_cache import (
    KeyedCompileMixin,
)
from sglang.kernels.ops.attention.gdn_cp_prefill.varlen_helper import (
    chunks_for_len,
    varlen_chunk_idx,
    varlen_chunk_valid_len,
)

# ---------------------------------------------------------------------------
# Combined configuration + execution class
# ---------------------------------------------------------------------------


class CPDeltaRulePrefillTcgen05Sm100(KeyedCompileMixin):
    """
    Configuration and execution class for the Chunked GDN kernel.

    Main responsibilities:
      - __init__    : warp IDs, barriers, tile shapes, SMEM/TMEM sizes
      - __call__    : @cute.jit host entry point (TMA setup, kernel launch)
      - kernel      : @cute.kernel device entry point (warp dispatch)
      - per-warp methods called from kernel's chunk loop

    Args:
        io_dtype   : input/output dtype (Float16)
        acc_dtype  : accumulator dtype  (Float32)
        b_t        : fixed chunk size / block tile (64)
        DK         : key/query hidden dim     (128)
        DV         : value hidden dim         (128)
    """

    # TMA descriptor size in bytes
    arch = "sm_100"
    bytes_per_tensormap = 128
    num_tensormaps = 5

    def __init__(
        self,
        io_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        state_dtype: Type[cutlass.Numeric],
        mma_tiler_qk: Tuple[int, int, int],
        mma_tiler_qs: Tuple[int, int, int],
        mma_tiler_qkv: Tuple[int, int, int],
        mma_tiler_kv: Tuple[int, int, int],
        max_active_clusters: int,
        num_sm: int,
        is_GQA: bool,
        head_ratio: int,
        use_initial_state: bool,
        store_final_state: bool = True,
        enable_checkpoints: bool = False,
        is_persistent: bool = True,
        cu_seqlens_dtype: Type[cutlass.Numeric] = cutlass.Int32,
    ):
        self.io_dtype = io_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.acc_dtype = acc_dtype
        self.state_dtype = state_dtype
        self.mma_tiler_qk = mma_tiler_qk
        self.mma_tiler_qs = mma_tiler_qs
        self.mma_tiler_qkv = mma_tiler_qkv
        self.mma_tiler_kv = mma_tiler_kv
        self.max_active_clusters = max_active_clusters
        self.num_sm = num_sm
        self.is_GQA = is_GQA
        self.head_ratio = head_ratio
        self.needs_initial_state = use_initial_state
        # Every CP CTA receives an explicit state, either the user-provided
        # initial state, zero, or the fixed-up state from the previous CP chunk.
        self.use_initial_state = True
        self.store_final_state = store_final_state
        self.enable_checkpoints = False
        self.is_persistent = False

        # ------------------------------------------------------------------
        # Warp assignments  (12 warps total)
        # ------------------------------------------------------------------
        # Precomputed-T transform and QK epilogue.
        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        # Recurrent state and output epilogues.
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.mma_warp_id = 8
        self.tma_qkv_warp_id = 9
        # The second issuer owns the five state/output GEMMs.
        self.mma_cg1_warp_id = 10
        # store O
        self.epilogue_warp_id = 11
        # The lightly loaded O epilogue warp also preprocesses gate.
        self.gate_warp_id = self.epilogue_warp_id

        # Give the MMA/TMA/gate/epilogue warps enough registers to keep their
        # pipeline handles resident. Fund the increase from CG1 first while
        # preserving the existing 64,512-register CTA budget.
        self.num_regs_compute_group_0 = 224
        self.num_regs_compute_group_1 = 256
        self.num_regs_other = 24
        if not self.use_initial_state:
            # The peeled zero-state MMA carries more pipeline cursors.  Transfer
            # twenty-four registers from each CG1 warp to each lightweight warp while
            # retaining the same 64,512-register CTA allocation.
            self.num_regs_compute_group_1 = 232
            self.num_regs_other = 48

        self.threads_per_cta = 32 * (
            len(
                (
                    self.mma_warp_id,
                    self.tma_qkv_warp_id,
                    self.mma_cg1_warp_id,
                    self.epilogue_warp_id,
                )
            )
            + len(self.compute_group_0_warp_ids)
            + len(self.compute_group_1_warp_ids)
        )

        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        self.threads_per_warp = 32
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        # ------------------------------------------------------------------
        # Named barriers - only TmemAllocator requires a NamedBarrier;
        # all other inter-warp synchronization uses mbarrier-based pipelines
        # created inside kernel().
        # ------------------------------------------------------------------
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp
            * len(
                (
                    self.mma_warp_id,
                    self.mma_cg1_warp_id,
                    *self.compute_group_0_warp_ids,
                    *self.compute_group_1_warp_ids,
                )
            ),
        )
        self.t_store_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.compute_group_0_warp_ids),
        )
        self.init_state_store_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )
        self.manual_cache_key(
            "io_dtype",
            "cu_seqlens_dtype",
            "acc_dtype",
            "state_dtype",
            "mma_tiler_qk",
            "mma_tiler_qs",
            "mma_tiler_qkv",
            "mma_tiler_kv",
            "is_GQA",
            "head_ratio",
            "needs_initial_state",
            "store_final_state",
        )

    def _setup_attributes(self):
        # ------------------------------------------------------------------
        # SMEM sizes (bytes per stage) and stage counts
        # ------------------------------------------------------------------
        self.smem_q_stages = 2
        # CP does not issue KK, so three K stages cover the QK/CG1 consumers.
        self.smem_k_stages = 3
        # Three V stages let TMA stay ahead while CG1 consumes the current pair.
        self.smem_v_stages = 3
        self.smem_t_stages = 2
        # Current Ainv1 and the next pair's two transformed T tiles can coexist.
        self.smem_ainv_stages = 3
        self.smem_qk_stages = 2
        # Gate work shares the epilogue warp, so O uses two stages to
        # avoid back-pressuring CG1 while the warp publishes the next gate.
        self.smem_o_stages = 2
        # Five resident stages preserve four chunks of gate lookahead.
        # Cumulative gate buffers are placed last in SMEM.
        self.smem_gate_stages = 5

        # ------------------------------------------------------------------
        # TMEM column offsets and buffer sizes (fp32, 32B per column)
        # ------------------------------------------------------------------
        self.tmem_kv_acc_stages = 1
        self.tmem_q_state_acc_stages = 1
        self.tmem_state_inp_stages = 1
        self.tmem_shared_inp_stages = 2
        # CG0 owns QK and CG1 owns KS/NV. Separate rings remove the
        # cross-group ownership handoff from the shared-acc critical path.
        self.tmem_cg0_shared_acc_stages = 2
        self.tmem_cg1_shared_acc_stages = 1

        self.tmem_state_offset = 0
        self.tmem_q_state_offset = (
            self.tmem_state_offset + self.tmem_kv_acc_stages * 128
        )
        self.tmem_state_inp_offset = (
            self.tmem_q_state_offset + self.tmem_q_state_acc_stages * 64
        )
        self.tmem_cg0_shared_acc_offset = (
            self.tmem_state_inp_offset + self.tmem_state_inp_stages * 64
        )
        self.tmem_cg1_shared_acc_offset = (
            self.tmem_cg0_shared_acc_offset + self.tmem_cg0_shared_acc_stages * 64
        )
        self.tmem_shared_inp_offset = (
            self.tmem_cg1_shared_acc_offset + self.tmem_cg1_shared_acc_stages * 64
        )
        self.buffer_align_bytes = 1024

    # -----------------------------------------------------------------------
    # Capability check
    # -----------------------------------------------------------------------

    @staticmethod
    def can_implement(
        io_dtype,
        acc_dtype,
        mma_tiler_qk,
        mma_tiler_qs,
        mma_tiler_qkv,
        mma_tiler_kv,
    ):
        """Raise CantImplementError if this configuration is not supported."""
        if io_dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise testing.CantImplementError(
                f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported"
            )
        if acc_dtype != cutlass.Float32:
            raise testing.CantImplementError(
                f"acc_dtype={acc_dtype} not supported; only Float32 is supported"
            )
        if mma_tiler_qk != (64, 64, 128):
            raise testing.CantImplementError(
                f"mma_tiler_qk={mma_tiler_qk} not supported; only (64, 64, 128) is supported"
            )
        if mma_tiler_qs != (128, 64, 128):
            raise testing.CantImplementError(
                f"mma_tiler_qs={mma_tiler_qs} not supported; only (128, 64, 128) is supported"
            )
        if mma_tiler_qkv != (128, 64, 64):
            raise testing.CantImplementError(
                f"mma_tiler_qkv={mma_tiler_qkv} not supported; only (128, 64, 64) is supported"
            )
        if mma_tiler_kv != (128, 128, 64):
            raise testing.CantImplementError(
                f"mma_tiler_kv={mma_tiler_kv} not supported; only (128, 128, 64) is supported"
            )

    # -----------------------------------------------------------------------
    # Host entry point
    # -----------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        t: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        fixed_state: cute.Tensor,
        initial_state: Optional[cute.Tensor],
        state_out: cute.Tensor,
        cp_chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        scale: cutlass.Float32,
        tensormap_workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # chunk size
        self.b_t = 64
        h_q = q.shape[1]
        h_v = v.shape[1]
        self._setup_attributes()

        if cutlass.const_expr(self.is_GQA):
            h_r = h_q // h_v
            h_qv = h_v
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_v)),
                    stride=(q.stride[0], q.stride[2], (q.stride[1], h_r * q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_v)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_v)),
                    stride=(v.stride[2], v.stride[0], (0, v.stride[1])),
                ),
            )
        else:
            h_r = h_v // h_q
            h_qv = h_q
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_q)),
                    stride=(q.stride[0], q.stride[2], (0, q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_q)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_q)),
                    stride=(v.stride[2], v.stride[0], (v.stride[1], h_r * v.stride[1])),
                ),
            )

        gate = cute.make_tensor(
            gate.iterator,
            cute.make_layout(
                (gate.shape[0], (h_r, h_qv)),
                stride=(gate.stride[0], (gate.stride[1], h_r * gate.stride[1])),
            ),
        )
        t = cute.make_tensor(
            t.iterator,
            cute.make_layout(
                (t.shape[0], t.shape[1], (h_r, h_qv), t.shape[3]),
                stride=(
                    t.stride[0],
                    t.stride[1],
                    (t.stride[2], h_r * t.stride[2]),
                    t.stride[3],
                ),
            ),
        )
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[0], (h_r, h_qv)),
                stride=(o.stride[2], o.stride[0], (o.stride[1], h_r * o.stride[1])),
            ),
        )
        fixed_state = cute.make_tensor(
            fixed_state.iterator,
            cute.make_layout(
                (
                    fixed_state.shape[2],
                    fixed_state.shape[3],
                    (h_r, h_qv),
                    fixed_state.shape[0],
                ),
                stride=(
                    fixed_state.stride[2],
                    fixed_state.stride[3],
                    (fixed_state.stride[1], h_r * fixed_state.stride[1]),
                    fixed_state.stride[0],
                ),
            ),
        )
        if cutlass.const_expr(initial_state is not None):
            initial_state = cute.make_tensor(
                initial_state.iterator,
                cute.make_layout(
                    (
                        initial_state.shape[2],
                        initial_state.shape[3],
                        (h_r, h_qv),
                        initial_state.shape[0],
                    ),
                    stride=(
                        initial_state.stride[2],
                        initial_state.stride[3],
                        (initial_state.stride[1], h_r * initial_state.stride[1]),
                        initial_state.stride[0],
                    ),
                ),
            )
        if cutlass.const_expr(self.store_final_state):
            state_out = cute.make_tensor(
                state_out.iterator,
                cute.make_layout(
                    (
                        state_out.shape[2],
                        state_out.shape[3],
                        (h_r, h_qv),
                        state_out.shape[0],
                    ),
                    stride=(
                        state_out.stride[2],
                        state_out.stride[3],
                        (state_out.stride[1], h_r * state_out.stride[1]),
                        state_out.stride[0],
                    ),
                ),
            )
        else:
            state_out = fixed_state

        # ------------------------------------------------------------------
        # Build tiled MMAs  (one per logical GEMM group, differing in operand major modes)
        # ------------------------------------------------------------------
        def _mma_op(mma_tiler, a_major, b_major, OperandSourceA):
            # Derive MMA atom (M, N) from the first two dims of the tile shape;
            # K=16 is the hardware fp16 atom depth (fixed for SM100 tcgen05).
            return tcgen05.MmaF16BF16Op(
                self.io_dtype,
                self.acc_dtype,
                (mma_tiler[0], mma_tiler[1], 16),
                self.cta_group,
                OperandSourceA,
                a_major,
                b_major,
            )

        # QK = Q @ K.T.
        tiled_mma_qk = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qk,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # K/Q applied to the recurrent state.
        tiled_mma_qs = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qs,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # New-V correction and intra-block output.
        tiled_mma_qkv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # for v_smem_layout_staged
        tiled_mma_qkv_ss = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.MN,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # Recurrent state update.
        tiled_mma_kv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_kv,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                tcgen05.OperandSource.TMEM,
            )
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # ------------------------------------------------------------------
        # SMEM layouts - computed before SharedStorage so cosize() is available
        # ------------------------------------------------------------------
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Q and K are the A and B operands of QK.
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_q_stages
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_k_stages
        )
        k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_kv, self.mma_tiler_kv, self.io_dtype, self.smem_k_stages
        )
        # V is the A operand of the new-V correction.
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qkv_ss, self.mma_tiler_qkv, self.io_dtype, self.smem_v_stages
        )
        # A_inv is the B operand of the new-V correction.
        ainv_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv,
            self.mma_tiler_qkv,
            self.io_dtype,
            self.smem_ainv_stages,
        )
        t_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv,
            self.mma_tiler_qkv,
            self.io_dtype,
            self.smem_t_stages,
        )
        # QK is the B operand of the intra-block output update.
        qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_qk_stages
        )

        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            cutlass.utils.LayoutEnum.from_tensor(o),
            self.mma_tiler_qkv[:2],
            self.smem_o_stages,
        )
        # Gate scalar arrays (1D Float32, flat layout - no swizzle needed)
        cumsumlog_smem_layout_staged = cute.make_layout(
            (self.b_t, 1, self.smem_gate_stages)
        )

        # ------------------------------------------------------------------
        # Shared memory struct  (defined here to capture layout cosizes)
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            # Pipeline mbarriers - one entry per stage, 2 Int64 words per barrier
            # TMA load warp -> MMA warp (K is staged for next-chunk prefetch)
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_k_stages * 2]
            # TMA load warp -> MMA warp
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_q_stages * 2]
            # TMA load warp -> CG1 V-load signal
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_v_stages * 2]
            # Gate load warp -> CG0/CG1.
            load_gate_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_gate_stages * 2
            ]
            # TMA T load -> CG0
            load_t_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_t_stages * 2]
            # MMA warp -> CG1 (Q*state acc ready in TMEM)
            q_state_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_q_state_acc_stages * 2
            ]
            # MMA warp -> CG1 (state update done).
            kv_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_kv_acc_stages * 2
            ]
            # MMA warp -> CG0 (QK ready).
            cg0_shared_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_cg0_shared_acc_stages * 2
            ]
            # MMA warp -> CG1 (KS/NV ready)
            cg1_shared_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_cg1_shared_acc_stages * 2
            ]
            # CG0 -> MMA warp (A_inv ready in SMEM)
            ainv_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_ainv_stages * 2
            ]
            # CG0 -> MMA warp (QK ready in SMEM)
            qk_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_qk_stages * 2
            ]
            state_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_state_inp_stages * 2
            ]
            # CG1 -> MMA warp: fixed-slot TMEM inputs.  The empty halves are
            # unused because downstream accumulator-full signals prove reuse.
            vks_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            nv_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            decay_v_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            # CG1 -> epilogue warp
            o_store_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_o_stages * 2
            ]
            # TMEM allocation token
            tmem_holding_buf: cutlass.Int32
            # SMEM tensor buffers (aligned, in SMEM layout order)
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sT: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(t_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # A_inv result consumed by the next MMA; keep this in io_dtype.
            sAinv: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype, cute.cosize(ainv_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # W_qk scores
            sQk: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(qk_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Cumulative gate scalars - placed last in SMEM
            cumsumlog: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]
            cumprod: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]

        self.shared_storage = SharedStorage

        # ------------------------------------------------------------------
        # Build TMA atoms
        # ------------------------------------------------------------------
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        t_smem_layout = cute.select(t_smem_layout_staged, mode=[0, 1, 2])

        tma_q = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                q,
                q_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_k = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                k,
                k_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_v = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                v,
                v_smem_layout,
                self.mma_tiler_qkv,
                tiled_mma_qkv_ss,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_t = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                t,
                t_smem_layout,
                self.mma_tiler_qkv,
                tiled_mma_qkv,
                self.cluster_layout_vmnk.shape,
            )
        )

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_o = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout,
                self.mma_tiler_qkv[:2],
            )
        )

        self.tma_q_bytes = cute.size_in_bytes(self.io_dtype, q_smem_layout)
        self.tma_k_bytes = cute.size_in_bytes(self.io_dtype, k_smem_layout)
        self.tma_v_bytes = cute.size_in_bytes(self.io_dtype, v_smem_layout)
        self.tma_t_bytes = cute.size_in_bytes(self.io_dtype, t_smem_layout)
        self.tma_o_bytes = cute.size_in_bytes(self.io_dtype, o_smem_layout)

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        grid_shape = (h_r * h_qv * max_cp_chunks_per_seq, num_seqs, 1)

        self.kernel(
            tiled_mma_qk,
            tiled_mma_qs,
            tiled_mma_qkv,
            tiled_mma_qkv_ss,
            tiled_mma_kv,
            tma_q,
            tma_k,
            tma_v,
            tma_t,
            gate,
            tma_o,
            cu_seqlens,
            fixed_state,
            initial_state,
            state_out,
            cp_chunk_len,
            h_r * h_qv,
            scale,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_trans_smem_layout_staged,
            v_smem_layout_staged,
            cumsumlog_smem_layout_staged,
            t_smem_layout_staged,
            ainv_smem_layout_staged,
            qk_smem_layout_staged,
            o_smem_layout_staged,
            q,
            k,
            v,
            t,
            o,
            tensormap_workspace,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def get_cp_work(
        self,
        cu_seqlens: cute.Tensor,
        seq_idx: cutlass.Int32,
        flat_work_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        cp_chunk_len: cutlass.Int32,
    ):
        head_idx = flat_work_idx % num_sab_heads
        chunk_idx = flat_work_idx // num_sab_heads
        seq_start = cutlass.Int32(cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, cp_chunk_len)
        valid_chunk_len = cutlass.Int32(0)
        if chunk_idx < num_cp_chunks:
            valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx, cp_chunk_len)
        tok_offset = seq_start + chunk_idx * cp_chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx, cp_chunk_len)
        t_blocks_per_cp_chunk = cute.ceil_div(cp_chunk_len, self.b_t)
        t_block_start = varlen_chunk_idx(
            seq_idx, seq_start, chunk_idx * t_blocks_per_cp_chunk, self.b_t
        )
        return (
            head_idx,
            chunk_idx,
            tok_offset,
            valid_chunk_len,
            num_cp_chunks,
            cp_chunk_idx,
            t_block_start,
        )

    # -----------------------------------------------------------------------
    # Device kernel
    # -----------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        # Tiled MMAs (one per logical GEMM group)
        # QK.
        tiled_mma_qk: cute.TiledMma,
        # K/Q applied to state.
        tiled_mma_qs: cute.TiledMma,
        # New-V correction and intra-block output.
        tiled_mma_qkv: cute.TiledMma,
        # New-V correction with an SMEM V operand.
        tiled_mma_qkv_ss: cute.TiledMma,
        # State update.
        tiled_mma_kv: cute.TiledMma,
        # TMA descriptors and cute tensors
        tma_q: TmaInfo,
        tma_k: TmaInfo,
        tma_v: TmaInfo,
        tma_t: TmaInfo,
        mGate: cute.Tensor,
        tma_o: TmaInfo,
        cu_seqlens: cute.Tensor,
        mFixedState: cute.Tensor,
        mInitialState: Optional[cute.Tensor],
        mStateOut: cute.Tensor,
        cp_chunk_len: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        scale: cutlass.Float32,
        # SMEM staged layouts (needed to view shared_storage tensor buffers)
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        cumsumlog_smem_layout_staged: cute.Layout,
        t_smem_layout_staged: cute.ComposedLayout,
        ainv_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        # TMA descriptor workspace in GMEM (one q/k/v/t/o descriptor set per CTA).
        mQ,
        mK,
        mV,
        mT,
        # used for TMA descriptor update
        mO,
        # (num_ctas, num_tensormaps, bytes_per_tensormap) Int8 workspace
        tensormap_workspace: cute.Tensor,
    ):
        """
        Process one explicitly mapped CP chunk for one sequence and SAB head.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()

        if cutlass.const_expr(self.needs_initial_state):
            assert (
                mInitialState is not None
            ), "mInitialState must be provided if needs_initial_state is True"
        else:
            assert (
                mInitialState is None
            ), "mInitialState must be None if needs_initial_state is False"

        (
            head_idx,
            cp_chunk_idx_in_seq,
            chunk_start,
            chunk_len,
            num_cp_chunks,
            cp_chunk_idx,
            t_block_start,
        ) = self.get_cp_work(cu_seqlens, bidy, bidx, num_sab_heads, cp_chunk_len)
        # ------------------------------------------------------------------
        # TMA descriptor GMEM workspace - one q/k/v/o descriptor set per CTA.
        # Slots: Q=0, K=1, V=2, T=3, O=4.
        # ------------------------------------------------------------------
        cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx

        tensormap_manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, self.bytes_per_tensormap
        )

        tensormap_workspace = self.initialize_workspace(tensormap_workspace, grid_dim)
        tensormap_q_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 0, None)].iterator
        )
        tensormap_k_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 1, None)].iterator
        )
        tensormap_v_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 2, None)].iterator
        )
        tensormap_o_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 4, None)].iterator
        )

        # ------------------------------------------------------------------
        # 1. Allocate SMEM / TMEM, prefetch TMA descriptors
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sK_trans = storage.sK.get_tensor(
            k_trans_smem_layout_staged.outer, swizzle=k_trans_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        sT = storage.sT.get_tensor(
            t_smem_layout_staged.outer, swizzle=t_smem_layout_staged.inner
        )
        # A_inverse MMA input.
        sAinv = storage.sAinv.get_tensor(
            ainv_smem_layout_staged.outer, swizzle=ainv_smem_layout_staged.inner
        )
        # QK output / O store  (W_qk first, then O epilogue staging)
        sQk = storage.sQk.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        # Gate scalar arrays (1D Float32, flat - no swizzle)
        sCumsumlog = storage.cumsumlog.get_tensor(cumsumlog_smem_layout_staged)
        sCumprod = storage.cumprod.get_tensor(cumsumlog_smem_layout_staged)

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_q.atom)
            cpasync.prefetch_descriptor(tma_k.atom)
            cpasync.prefetch_descriptor(tma_v.atom)
            cpasync.prefetch_descriptor(tma_t.atom)
            cpasync.prefetch_descriptor(tma_o.atom)

        # TMEM allocator object - CG1 will issue the actual allocation
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # CG1 owns allocation and is the last group to release TMEM state.
            allocator_warp_id=self.compute_group_1_warp_ids[0],
        )

        # ------------------------------------------------------------------
        # mbarrier-based pipelines
        # Each pipeline is created by all threads; barrier_storage points into SMEM.
        # defer_sync=True means pipeline_init_arrive() flushes all at once below.
        # ------------------------------------------------------------------
        def _cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        # 1 thread (TMA issuer)
        cg_tma = _cg(len([self.tma_qkv_warp_id]))
        # 1 warp (gate scalar load warp).
        cg_gate = _cg(self.threads_per_warp * len([self.gate_warp_id]))
        # One producer per result pipeline; K/Q are consumed by both issuers.
        cg_mma = _cg(len([self.mma_warp_id]))
        cg_mma_both = _cg(len([self.mma_warp_id, self.mma_cg1_warp_id]))
        # 128 threads (CG0)
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        # One elected thread per CG0 warp releases each TMA T stage.
        cg_cg0_t = _cg(len(self.compute_group_0_warp_ids))
        # 128 threads (CG1)
        cg_cg1 = _cg(self.threads_per_warp * len(self.compute_group_1_warp_ids))
        # 4 threads (one per CG1 warp, used for V load signaling)
        cg_cg1_v = _cg(len(self.compute_group_1_warp_ids))
        # 256 threads (CG0 + CG1)
        cg_both = _cg(
            self.threads_per_warp * len(self.compute_group_0_warp_ids)
            + self.threads_per_warp * len(self.compute_group_1_warp_ids)
        )
        # 32 threads (epilogue warp)
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))

        # TMA load pipelines: K/Q feed MMA; V is signaled to CG1 for ALU work.
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_q_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_v_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # Gate producer -> CG0 / CG1 (software-signaled).
        # Scalar-load paths do not use TMA barriers; producer calls commit() after writes.
        load_gate_producer, load_gate_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_gate_stages,
            producer_group=cg_gate,
            consumer_group=cg_both,
            barrier_storage=storage.load_gate_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_t_producer, load_t_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_t_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg0_t,
            tx_count=self.tma_t_bytes,
            barrier_storage=storage.load_t_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  kv_acc
        kv_acc_producer, kv_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_kv_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  q_state_acc
        q_state_acc_producer, q_state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_q_state_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # QK issuer -> CG0 accumulator ring.
        cg0_shared_acc_producer, cg0_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg0_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg0,
                barrier_storage=storage.cg0_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # MMA warp -> CG1: KS/NV accumulator ring.
        cg1_shared_acc_producer, cg1_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg1_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg1,
                barrier_storage=storage.cg1_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG0 -> MMA warp:  a_inv_done
        a_inv_ready_producer, a_inv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  qk_done
        qk_ready_producer, qk_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_qk_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG1 -> MMA warp: state input.
        state_inp_ready_producer, state_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_state_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> MMA warp: fixed-slot, ready-only input notifications.
        vks_ready_producer, vks_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.vks_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        nv_ready_producer, nv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.nv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        decay_v_ready_producer, decay_v_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.decay_v_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> epilogue warp:  output_ready
        o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_o_stages,
            producer_group=cg_cg1,
            consumer_group=cg_epi,
            barrier_storage=storage.o_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline_init_arrive(is_relaxed=True)

        pipeline_init_wait()

        num_pairs = 2
        num_pairs_b = cute.ceil_div(chunk_len, self.b_t * num_pairs)
        num_chunks_b = num_pairs_b * num_pairs
        num_valid_chunks_b = cute.ceil_div(chunk_len, self.b_t)
        chunk_end = chunk_start + chunk_len

        # COMPUTE WARP GROUP 0 (warps 0-3): T transform and QK epilogue.
        if (
            warp_idx >= self.compute_group_0_warp_ids[0]
            and warp_idx <= self.compute_group_0_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            sQk_pisl = self._transform_to_position_independent_layout(
                sQk, qk_smem_layout_staged.inner
            )
            sAinv_pisl = self._transform_to_position_independent_layout(
                sAinv, ainv_smem_layout_staged.inner
            )
            for pair_idx in cutlass.range(num_pairs_b):
                (
                    load_gate_consumer,
                    load_t_consumer,
                    cg0_shared_acc_consumer,
                    a_inv_ready_producer,
                    qk_ready_producer,
                ) = self.compute_group_0_pair_cp(
                    tidx,
                    tmem_ptr,
                    scale,
                    (tiled_mma_qk,),
                    (sCumsumlog, sT, sAinv_pisl, sQk_pisl),
                    (
                        load_gate_consumer,
                        load_t_consumer,
                        cg0_shared_acc_consumer,
                        a_inv_ready_producer,
                        qk_ready_producer,
                    ),
                    (pair_idx, num_pairs_b, num_valid_chunks_b, chunk_len),
                )
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()

        # COMPUTE WARP GROUP 1 (warps 4-7): recurrent state and output.
        if (
            warp_idx >= self.compute_group_1_warp_ids[0]
            and warp_idx <= self.compute_group_1_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            if chunk_len > 0:
                kv_acc_producer = self._load_cp_initial_state(
                    tidx,
                    mFixedState,
                    mInitialState,
                    head_idx,
                    bidy,
                    cp_chunk_idx_in_seq,
                    cp_chunk_idx,
                    tmem_ptr,
                    tiled_mma_kv,
                    kv_acc_producer,
                )
                sV_pisl = self._transform_to_position_independent_layout(
                    sV, v_smem_layout_staged.inner
                )
                sO_pisl = self._transform_to_position_independent_layout(
                    sO, o_smem_layout_staged.inner
                )
                checkpoint_offset = 0
                is_first_chunk = True
                for chunk_idx in cutlass.range(num_chunks_b):
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
                        checkpoint_offset,
                    ) = self.compute_group_1_chunk(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (sV_pisl, sCumsumlog, sCumprod, sCumprod, sO_pisl),
                        (mStateOut, checkpoint_offset, cutlass.Int32(0)),
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
                        ),
                        (chunk_idx, num_pairs_b, head_idx, chunk_len, is_first_chunk),
                    )
                    is_first_chunk = False

                if cp_chunk_idx_in_seq == num_cp_chunks - 1:
                    kv_acc_consumer = self._store_final_state(
                        tidx,
                        mStateOut,
                        None,
                        head_idx,
                        bidy,
                        tmem_ptr,
                        tiled_mma_kv,
                        kv_acc_consumer,
                        chunk_len,
                        mStateOut,
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    )
                else:
                    kv_acc_handle = kv_acc_consumer.wait_and_advance()
                    kv_acc_handle.release()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            o_store_producer.tail()
            state_inp_ready_producer.tail()

        # CG0 MMA ISSUER (warp 8): paired QK.
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            for _pair_idx in cutlass.range(num_pairs_b):
                (
                    cg0_shared_acc_producer,
                    load_k_consumer,
                    load_q_consumer,
                ) = self.mma_cg0_pair_cp(
                    tmem_ptr,
                    (tiled_mma_qk, tiled_mma_qkv),
                    (sQ, sK),
                    (
                        cg0_shared_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                    ),
                )
            cg0_shared_acc_producer.tail()

        # CG1 MMA ISSUER (warp 10): KS/QS/NV/QKV/KV.
        elif warp_idx == self.mma_cg1_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            for chunk_idx in cutlass.range(num_chunks_b):
                (
                    cg1_shared_acc_producer,
                    q_state_acc_producer,
                    kv_acc_producer,
                    load_k_consumer,
                    load_q_consumer,
                    a_inv_ready_consumer,
                    qk_ready_consumer,
                    state_inp_ready_consumer,
                    vks_ready_consumer,
                    nv_ready_consumer,
                    decay_v_ready_consumer,
                ) = self.mma_cg1_chunk(
                    tmem_ptr,
                    (
                        tiled_mma_qs,
                        tiled_mma_qkv,
                        tiled_mma_qkv_ss,
                        tiled_mma_kv,
                    ),
                    (sQ, sK, sK_trans, sV, sAinv, sQk),
                    (
                        cg1_shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        vks_ready_consumer,
                        nv_ready_consumer,
                        decay_v_ready_consumer,
                    ),
                    chunk_idx == 0,
                )
            cg1_shared_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()

        # TMA LOAD WARP (warp 9): Q/K/V and precomputed T.
        elif warp_idx == self.tma_qkv_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tensormap_manager.init_tensormap_from_atom(
                tma_q.atom, tensormap_q_ptr, self.tma_qkv_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_k.atom, tensormap_k_ptr, self.tma_qkv_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_v.atom, tensormap_v_ptr, self.tma_qkv_warp_id
            )
            tensormap_manager.fence_tensormap_initialization()
            bounded_q = cute.make_tensor(
                mQ.iterator,
                cute.make_layout(
                    (chunk_end, mQ.shape[1], mQ.shape[2]),
                    stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                ),
            )
            bounded_k = cute.make_tensor(
                mK.iterator,
                cute.make_layout(
                    (chunk_end, mK.shape[1], mK.shape[2]),
                    stride=(mK.stride[0], mK.stride[1], mK.stride[2]),
                ),
            )
            bounded_v = cute.make_tensor(
                mV.iterator,
                cute.make_layout(
                    (mV.shape[0], chunk_end, mV.shape[2]),
                    stride=(mV.stride[0], mV.stride[1], mV.stride[2]),
                ),
            )
            tensormap_manager.update_tensormap(
                (bounded_q, bounded_k, bounded_v),
                (tma_q.atom, tma_k.atom, tma_v.atom),
                (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr),
                self.tma_qkv_warp_id,
                (None, None, None),
            )
            for chunk_idx in cutlass.range(num_chunks_b):
                chunk_offset = chunk_start + chunk_idx * self.b_t
                load_q_producer, load_k_producer, load_v_producer = self.tma_qkv_warp(
                    (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                    (tma_q, tma_k, tma_v),
                    (sQ, sK, sV),
                    (load_q_producer, load_k_producer, load_v_producer),
                    (chunk_offset, chunk_idx, bidy, head_idx),
                    (
                        tensormap_manager,
                        tensormap_q_ptr,
                        tensormap_k_ptr,
                        tensormap_v_ptr,
                    ),
                )
                t_chunk_idx = chunk_idx
                if chunk_idx >= num_valid_chunks_b:
                    t_chunk_idx = num_valid_chunks_b - 1
                load_t_producer = self.tma_t_warp(
                    tiled_mma_qkv,
                    tma_t,
                    sT,
                    load_t_producer,
                    t_block_start + t_chunk_idx,
                    head_idx,
                )
            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()
            load_t_producer.tail()

        # EPILOGUE WARP (warp 11): gate preprocessing and O store.
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tensormap_manager.init_tensormap_from_atom(
                tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id
            )
            tensormap_manager.fence_tensormap_initialization()
            bounded_o = cute.make_tensor(
                mO.iterator,
                cute.make_layout(
                    (mO.shape[0], chunk_end, mO.shape[2]),
                    stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                ),
            )
            tensormap_manager.update_tensormap(
                (bounded_o,),
                (tma_o.atom,),
                (tensormap_o_ptr,),
                self.epilogue_warp_id,
                (None,),
            )
            tensormap_manager.fence_tensormap_update(tensormap_o_ptr)

            if chunk_len > 0:
                num_gate_prefetch = 4
                for prefetch_idx in range(2):
                    load_gate_producer = self.load_gate_warp(
                        tidx,
                        mGate,
                        (sCumsumlog, sCumprod),
                        load_gate_producer,
                        (
                            chunk_start + prefetch_idx * self.b_t,
                            head_idx,
                            prefetch_idx >= num_valid_chunks_b - 1,
                            chunk_end,
                        ),
                    )
                if num_chunks_b > 2:
                    for prefetch_idx in range(2, num_gate_prefetch):
                        load_gate_producer = self.load_gate_warp(
                            tidx,
                            mGate,
                            (sCumsumlog, sCumprod),
                            load_gate_producer,
                            (
                                chunk_start + prefetch_idx * self.b_t,
                                head_idx,
                                prefetch_idx >= num_valid_chunks_b - 1,
                                chunk_end,
                            ),
                        )
                for chunk_idx in cutlass.range(num_chunks_b):
                    prefetch_idx = chunk_idx + num_gate_prefetch
                    if prefetch_idx < num_chunks_b:
                        load_gate_producer = self.load_gate_warp(
                            tidx,
                            mGate,
                            (sCumsumlog, sCumprod),
                            load_gate_producer,
                            (
                                chunk_start + prefetch_idx * self.b_t,
                                head_idx,
                                prefetch_idx >= num_valid_chunks_b - 1,
                                chunk_end,
                            ),
                        )
                    o_store_consumer = self.epilogue_warp(
                        (sO,),
                        (tma_o,),
                        (o_store_consumer,),
                        (head_idx, chunk_start + chunk_idx * self.b_t),
                        (tensormap_manager, tensormap_o_ptr),
                    )
            load_gate_producer.tail()

    # ------------------------------------------------------------------
    # Per-warp methods  (called from kernel's chunk loop)
    @cute.jit
    def tma_t_warp(
        self,
        tiled_mma_qkv: cute.TiledMma,
        tma_t: TmaInfo,
        sT: cute.Tensor,
        load_t_producer: pipeline.PipelineProducer,
        t_block_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
    ) -> pipeline.PipelineProducer:
        """Load one signed, beta-folded T tile into its MMA-B SMEM layout."""
        t_handle = load_t_producer.acquire_and_advance()
        mT = tma_t.tma_tensor[None, None, head_idx, t_block_idx]
        gT = cute.flat_divide(
            mT,
            (self.b_t, self.b_t),
        )
        tCgT = tiled_mma_qkv.get_slice(0).partition_B(gT)
        tTsT, tTgT = cpasync.tma_partition(
            tma_t.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sT, 0, 3),
            cute.group_modes(tCgT, 0, 3),
        )
        cute.copy(
            tma_t.atom,
            tTgT[(None, 0, 0)],
            tTsT[(None, t_handle.index)],
            tma_bar_ptr=t_handle.barrier,
        )
        return load_t_producer

    @cute.jit
    def load_gate_warp(
        self,
        tidx: cutlass.Int32,
        gate: cute.Tensor,
        smem_args: tuple,
        load_gate_producer: pipeline.PipelineProducer,
        work_args: tuple,
    ) -> pipeline.PipelineProducer:
        """Warp 10: load and prefix-process gate[BT] for the current chunk.

        Gate is loaded via ldg (sync G->R), then preprocessed into cumsum-log
        and cumprod channels in SMEM.

        The last tile uses predicated copies: elements with linear index >= valid_tokens
        are out-of-bounds and receive the neutral gate value one.

        Thread tidx (lane 0..31) owns positions tidx, tidx+32, tidx+64, tidx+96.
        """
        sCumsumlog, sCumprod = smem_args
        chunk_offset, head_idx, is_last_tile, batch_end = work_args

        # lane index
        lidx = tidx % self.threads_per_warp

        gGate = cute.domain_offset((chunk_offset,), gate[None, head_idx])
        cGate = cute.domain_offset(
            (chunk_offset,), cute.make_identity_tensor(gate[None, head_idx].shape)
        )
        gGate = cute.flat_divide(gGate, (self.b_t,))[None, 0]
        cGate = cute.flat_divide(cGate, (self.b_t,))[None, 0]

        # Tiled copy: 1D thread/value layouts; partition_S/D handle element mapping.
        # thread_layout (32,): each of the 32 lanes maps to one row of the b_t tile.
        # value_layout  (4,) : each lane owns 4 elements strided by threads_per_warp.
        thread_layout = cute.make_layout((self.threads_per_warp,), stride=(1,))
        value_layout = cute.make_layout((1,), stride=(1,))

        # Gate: sync G->R (ldg), apply ln + prefix sum, then R->S (sts)
        atom_gate_g2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
        )
        tiled_copy_gate_g2r = cute.make_tiled_copy_tv(
            atom_gate_g2r, thread_layout, value_layout
        )

        # Per-thread partitions (1D tensors; no manual 2D reshaping needed)
        thr_copy_gate_g2r = tiled_copy_gate_g2r.get_slice(lidx)
        tGgGate = thr_copy_gate_g2r.partition_S(gGate)
        tGsCumsumlog = thr_copy_gate_g2r.partition_D(sCumsumlog)
        tGsCumprod = thr_copy_gate_g2r.partition_D(sCumprod)

        gate_handle = load_gate_producer.acquire_and_advance()

        rGate = cute.make_rmem_tensor_like(tGgGate, self.acc_dtype)
        tGrGate = tiled_copy_gate_g2r.retile(rGate)
        rCumprod = cute.make_rmem_tensor_like(tGgGate, self.acc_dtype)
        tGrCumprod = tiled_copy_gate_g2r.retile(rCumprod)

        # Compute the tail predicate once for the gate tile.
        tGcGate = thr_copy_gate_g2r.partition_S(cGate)
        tGpGate = cute.make_rmem_tensor(
            ((tGcGate.shape[0][1],), tGcGate.shape[1]), cutlass.Boolean
        )
        if is_last_tile:
            valid_tokens = batch_end  # noqa: F841
            for i in range(cute.size(tGpGate)):
                tGpGate[i] = cute.elem_less(tGcGate[i][0], batch_end)

        # --- Gate load ---
        if is_last_tile:
            # OOB neutral: 1.0 -> log2 ~= 0.0 (no decay contribution)
            tGrGate.fill(1.0)
            cute.copy(tiled_copy_gate_g2r, tGgGate, tGrGate, pred=tGpGate)
        else:
            cute.copy(tiled_copy_gate_g2r, tGgGate, tGrGate)

        # --- log2 + warp-wide inclusive prefix sum + SMEM store (always) ---
        for i in range(cute.size(tGrGate)):
            tGrGate[i] = cute.math.log2(tGrGate[i] + 1e-10, fastmath=True)
        for offset in [1, 2, 4, 8, 16]:
            for col in range(cute.size(tGrGate)):
                n = cute.arch.shuffle_sync_up(
                    tGrGate[col], offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lidx >= offset:
                    tGrGate[col] = tGrGate[col] + n
        sum_v = 0.0  # noqa: F841
        for col in range(1, cute.size(tGrGate)):
            last_v = cute.arch.shuffle_sync(
                tGrGate[col - 1],
                self.threads_per_warp - 1,
                mask=0xFFFFFFFF,
                mask_and_clamp=self.threads_per_warp - 1,
            )
            tGrGate[col] += last_v
        for col in range(cute.size(tGrGate)):
            tGrCumprod[col] = cute.math.exp2(tGrGate[col], fastmath=True)
        cute.copy(
            tiled_copy_gate_g2r, tGrGate, tGsCumsumlog[None, None, 0, gate_handle.index]
        )
        cute.copy(
            tiled_copy_gate_g2r,
            tGrCumprod,
            tGsCumprod[None, None, 0, gate_handle.index],
        )
        gate_handle.commit()

        return load_gate_producer

    @cute.jit
    def mma_cg0_warp(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Warp 8: issue Q @ K.T for the CP score path."""
        tiled_mma_qk, tiled_mma_qkv = mma_args
        sQ, sK = smem_args
        shared_acc_producer, load_k_consumer, load_q_consumer = pipeline_args

        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtAcc_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_cg0_shared_acc_stages)
        )
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_cg0_shared_acc_offset, tCtAcc_fake.layout
        )

        tCrK_B = tiled_mma_qk.make_fragment_B(sK)
        tCrQ_A = tiled_mma_qk.make_fragment_A(sQ)

        k_handle = load_k_consumer.wait_and_advance()
        q_handle = load_q_consumer.wait_and_advance()
        qk_handle = shared_acc_producer.acquire_and_advance()
        num_kphases = cute.size(tCrQ_A, mode=[2])
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtShared[None, None, None, qk_handle.index],
                tCrQ_A[None, None, kphase_idx, q_handle.index],
                tCrK_B[None, None, kphase_idx, k_handle.index],
                tCtShared[None, None, None, qk_handle.index],
            )
        qk_handle.commit()
        q_handle.release()
        k_handle.release()

        return shared_acc_producer, load_k_consumer, load_q_consumer

    @cute.jit
    def compute_group_0_cp(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ):
        """Materialize gamma-scaled QK and signed-T operands for tcgen05."""
        (tiled_mma_qk,) = mma_args
        sCumsumlog, sT, sAinv, sQk = smem_args
        (
            load_gate_consumer,
            load_t_consumer,
            aux_acc_consumer,
            t_ready_producer,
            qk_ready_producer,
        ) = pipeline_args
        is_final_block, valid_tokens = work_args

        num_threads_cg0 = self.threads_per_warp * len(self.compute_group_0_warp_ids)
        cg0_tidx = tidx % num_threads_cg0
        tAcc_shape = tiled_mma_qk.partition_shape_C(
            (self.mma_tiler_qk[0], self.mma_tiler_qk[1])
        )
        tAcc = tiled_mma_qk.make_fragment_C(tAcc_shape)
        tAcc = cute.make_tensor(
            tAcc.iterator,
            cute.flat_product(
                tAcc.layout,
                cute.make_layout((self.tmem_cg0_shared_acc_stages,), stride=(1,)),
            ),
        )
        tStS = cute.make_tensor(tmem_ptr + self.tmem_cg0_shared_acc_offset, tAcc.layout)
        tStS_mn = self.transform_partitioned_tensor_layout(tStS)
        cS = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tStS[(None, None), 0, 0, 0])
        thr_t2r = tiled_t2r.get_slice(cg0_tidx)
        tTR_tStS = thr_t2r.partition_S(tStS_mn)
        tTR_cS = thr_t2r.partition_D(cS)

        sT_mn = self.transform_partitioned_tensor_layout(sT)
        sAinv_mn = self.transform_partitioned_tensor_layout(sAinv)

        gate_handle = load_gate_consumer.wait_and_advance()
        t_handle = load_t_consumer.wait_and_advance()
        t_ready_handle = t_ready_producer.acquire_and_advance()
        copy_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(self.io_dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(self.b_t, self.b_t, 16),
        )
        atom_t_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_t_s2r = cute.make_tiled_copy_C(atom_t_s2r, copy_mma)
        thr_t_s2r = tiled_t_s2r.get_slice(cg0_tidx)
        tTsT = thr_t_s2r.partition_S(sT_mn)
        rT = cute.make_rmem_tensor(
            copy_mma.partition_shape_C((self.b_t, self.b_t)),
            self.io_dtype,
        )
        tTrT = thr_t_s2r.retile(rT)
        cute.copy(
            tiled_t_s2r,
            tTsT[None, None, None, t_handle.index],
            tTrT,
        )

        cT = cute.make_identity_tensor((self.b_t, self.b_t))
        tTcT = thr_t_s2r.partition_D(cT)
        tTcT = thr_t_s2r.retile(tTcT)
        for i in cutlass.range_constexpr(cute.size(tTrT)):
            t, s = tTcT[i]
            pred = s >= t
            if is_final_block:
                pred = pred and s < valid_tokens and t < valid_tokens
            gamma = cutlass.Float32(0.0)
            if pred:
                gamma = cute.math.exp2(
                    sCumsumlog[s, 0, gate_handle.index]
                    - sCumsumlog[t, 0, gate_handle.index],
                    fastmath=True,
                )
            tTrT[i] = self.io_dtype(-gamma * cutlass.Float32(tTrT[i]))

        sAinv_t = cute.make_tensor(
            sAinv_mn.iterator,
            cute.select(sAinv_mn.layout, mode=[1, 0, 2]),
        )
        atom_t_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_t_r2s = cute.make_tiled_copy_C(atom_t_r2s, copy_mma)
        thr_t_r2s = tiled_t_r2s.get_slice(cg0_tidx)
        tTsAinv = thr_t_r2s.partition_D(sAinv_t)
        tTsAinv = cute.make_tensor(tTsAinv.iterator.align(16), tTsAinv.layout)
        tTrT_r2s = thr_t_r2s.retile(rT)
        cute.copy(
            tiled_t_r2s,
            tTrT_r2s,
            tTsAinv[None, None, None, t_ready_handle.index],
        )
        cute.arch.fence_view_async_shared()
        self.t_store_barrier.arrive_and_wait()
        t_handle.release()
        t_ready_handle.commit()

        qk_ready_handle = qk_ready_producer.acquire_and_advance()
        qk_handle = aux_acc_consumer.wait_and_advance()
        tQKrQK = cute.make_rmem_tensor_like(tTR_cS, self.acc_dtype)
        tQKrQK_out = cute.make_rmem_tensor_like(tQKrQK, self.io_dtype)
        atom_qk_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_qk_r2s = cute.make_tiled_copy_D(atom_qk_r2s, tiled_t2r)
        tQsQK = tiled_qk_r2s.get_slice(cg0_tidx).partition_D(
            self.transform_partitioned_tensor_layout(sQk)
        )
        tQrQK = tiled_qk_r2s.retile(tQKrQK_out)
        for sub in cutlass.range_constexpr(tQKrQK.shape[2]):
            cute.copy(
                tiled_t2r,
                tTR_tStS[None, 0, sub, qk_handle.index],
                tQKrQK[None, 0, sub],
            )
            for i in cutlass.range(32):
                s, t = tTR_cS[i, 0, sub]
                pred = s >= t
                if is_final_block:
                    pred = pred and s < valid_tokens and t < valid_tokens
                gamma = cutlass.Float32(0.0)
                if pred:
                    gamma = cute.math.exp2(
                        sCumsumlog[s, 0, gate_handle.index]
                        - sCumsumlog[t, 0, gate_handle.index],
                        fastmath=True,
                    )
                tQKrQK_out[i, 0, sub] = self.io_dtype(tQKrQK[i, 0, sub] * gamma * scale)
            cute.copy(
                tiled_qk_r2s,
                tQrQK[None, 0, sub],
                tQsQK[None, 0, sub, qk_ready_handle.index],
            )
        cute.arch.fence_view_async_shared()
        qk_handle.release()
        qk_ready_handle.commit()
        gate_handle.release()
        return (
            load_gate_consumer,
            load_t_consumer,
            aux_acc_consumer,
            t_ready_producer,
            qk_ready_producer,
        )

    def transform_partitioned_tensor_layout(self, tensor: cute.Tensor) -> cute.Tensor:
        """
        Transform MMA layout from ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
        to ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest).

        This groups MMA_ATOM_M with MMA_M and MMA_ATOM_N with MMA_N.

        :param tensor: Input tensor with layout ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
        :type tensor: cute.Tensor
        :return: Transformed tensor with layout ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest)
        :rtype: cute.Tensor
        """
        layout = tensor.layout
        # Save original layout in case it is a composed layout
        stored_layout = layout

        if isinstance(stored_layout, cute.ComposedLayout):
            # For composed layouts, we only modify the outer layout
            layout = layout.outer

        shape = layout.shape
        stride = layout.stride

        # Build new shape: ((shape[0][0], shape[1]), (shape[0][1], shape[2]), ...rest)
        new_shape = ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:])

        # Build new stride: ((stride[0][0], stride[1]), (stride[0][1], stride[2]), ...rest)
        new_stride = ((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:])

        new_layout = cute.make_layout(shape=new_shape, stride=new_stride)

        if isinstance(stored_layout, cute.ComposedLayout):
            # Recreate the composed layout
            new_layout = cute.make_composed_layout(
                stored_layout.inner, stored_layout.offset, new_layout
            )

        return cute.make_tensor(tensor.iterator, new_layout)

    # -----------------------------------------------------------------------
    @cute.jit
    def mma_cg0_pair_cp(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
    ):
        for _ in range(2):
            pipeline_args = self.mma_cg0_warp(
                tmem_ptr,
                mma_args,
                smem_args,
                pipeline_args,
                (),
            )
        return pipeline_args

    @cute.jit
    def compute_group_0_pair_cp(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ):
        pair_idx, _, num_valid_chunks, chunk_len = work_args
        for chunk_in_pair in range(2):
            chunk_idx = pair_idx * 2 + chunk_in_pair
            valid_tokens = chunk_len - chunk_idx * self.b_t
            pipeline_args = self.compute_group_0_cp(
                tidx,
                tmem_ptr,
                scale,
                mma_args,
                smem_args,
                pipeline_args,
                (chunk_idx >= num_valid_chunks - 1, valid_tokens),
            )
        return pipeline_args

    @cute.jit
    def tma_qkv_warp(
        self,
        mma_args: tuple,
        tma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        tensormap_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer, pipeline.PipelineProducer, pipeline.PipelineProducer
    ]:
        """Warp 9: load Q, K (double-buffered), V for the current chunk.

        TMA load pattern:
          1. domain_offset the TMA tensor to (chunk_offset, head_idx, 0) so that
             the logical tile (0, ...) maps to the current chunk.
          2. flat_divide to obtain the tiled global view.
          3. thr_mma.partition_{A,B} to get the TMA-compatible per-thread view.
          4. cpasync.tma_partition -> (tXsX, tXgX) SMEM/global pairs.
          5. acquire pipeline stage, issue cute.copy, signal mbarrier.

        Note on head coordinate: head_idx is the flat KV-head index in [0, h_qv).
        For the hierarchical head layout (h_r, h_qv) with h_r having stride 0
        (broadcast), the flat index maps correctly as long as head_idx < h_qv.
        """
        tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv = mma_args
        tma_q, tma_k, tma_v = tma_args
        sQ, sK, sV = smem_args
        load_q_producer, load_k_producer, load_v_producer = pipeline_args
        chunk_offset, chunk_idx, batch_idx, head_idx = work_args
        tensormap_manager, tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr = (
            tensormap_args
        )

        # Single-CTA mode: no multicast, cta_v = 0.
        cta_layout = cute.make_layout(1)

        # Per-thread MMA slices (cta_v=0 for ONE-CTA mode).
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_qkv_ss = tiled_mma_qkv_ss.get_slice(0)

        # Tile shapes from the MMA tiler (128, 128, 128):
        #   mode[0,2] = (BT, DK) - M,K tile for A (Q) and for B (K) of tiled_mma_qk
        #   mode[1,2] = (BT, DV) - tile shape for loading V (B operand in GEMMs 5/6)
        # (BT, DK)
        qk_tile = cute.select(self.mma_tiler_qk, mode=[0, 2])
        # (DV, BT)
        v_tile = cute.select(self.mma_tiler_qkv, mode=[0, 2])

        # ------------------------------------------------------------------
        # K  (B operand of GEMM-kk / GEMM-qk, double-buffered)
        # Tensor shape: (total_tokens, H_hier, DK)
        # TMA tile:     (BT, DK)
        # ------------------------------------------------------------------
        mK = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_k.tma_tensor[None, None, head_idx]
        )
        # (..., num_k_tiles, ...)
        gK = cute.flat_divide(mK, qk_tile)
        tCgK = thr_mma_qk.partition_B(gK)
        tKsK, tKgK = cpasync.tma_partition(
            tma_k.atom,
            0,
            cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )

        # Load K for the current chunk into the next available pipeline stage.
        k_handle = load_k_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_k_ptr)

        cute.copy(
            tma_k.atom,
            tKgK[(None, 0, 0)],
            tKsK[(None, k_handle.index)],
            tma_bar_ptr=k_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_k_ptr, cute.AddressSpace.generic
            ),
        )

        # ------------------------------------------------------------------
        # Q  (A operand of GEMM-qk, single-buffered)
        # ------------------------------------------------------------------
        mQ = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_q.tma_tensor[None, None, head_idx]
        )
        gQ = cute.flat_divide(mQ, qk_tile)
        tCgQ = thr_mma_qk.partition_A(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q.atom,
            0,
            cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )

        q_handle = load_q_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_q_ptr)
        cute.copy(
            tma_q.atom,
            tQgQ[(None, 0, 0)],
            tQsQ[(None, q_handle.index)],
            tma_bar_ptr=q_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_q_ptr, cute.AddressSpace.generic
            ),
        )

        # ------------------------------------------------------------------
        # V  (B operand of GEMM-new_v / GEMM-qkv, single-buffered)
        # ------------------------------------------------------------------
        mV = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset), tma_v.tma_tensor[None, None, head_idx]
        )
        gV = cute.flat_divide(mV, v_tile)
        tCgV = thr_mma_qkv_ss.partition_A(gV)
        tVsV, tVgV = cpasync.tma_partition(
            tma_v.atom,
            0,
            cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tCgV, 0, 3),
        )

        v_handle = load_v_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_v_ptr)
        cute.copy(
            tma_v.atom,
            tVgV[(None, 0, 0)],
            tVsV[(None, v_handle.index)],
            tma_bar_ptr=v_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_v_ptr, cute.AddressSpace.generic
            ),
        )

        return load_q_producer, load_k_producer, load_v_producer

    @cute.jit
    def _load_cp_initial_state(
        self,
        tidx,
        mFixedState,
        mInitialState,
        head_idx,
        seq_idx,
        cp_chunk_idx_in_seq,
        cp_chunk_idx,
        tmem_ptr,
        tiled_mma_kv,
        kv_acc_producer,
    ):
        if cutlass.const_expr(self.needs_initial_state):
            if cp_chunk_idx_in_seq > 0:
                kv_acc_producer = self._load_initial_state(
                    tidx,
                    mFixedState,
                    None,
                    head_idx,
                    cp_chunk_idx - 1,
                    tmem_ptr,
                    tiled_mma_kv,
                    kv_acc_producer,
                )
            else:
                kv_acc_producer = self._load_initial_state(
                    tidx,
                    mInitialState,
                    None,
                    head_idx,
                    seq_idx,
                    tmem_ptr,
                    tiled_mma_kv,
                    kv_acc_producer,
                )
        else:
            if cp_chunk_idx_in_seq > 0:
                kv_acc_producer = self._load_initial_state(
                    tidx,
                    mFixedState,
                    None,
                    head_idx,
                    cp_chunk_idx - 1,
                    tmem_ptr,
                    tiled_mma_kv,
                    kv_acc_producer,
                )
            else:
                kv_acc_producer = self._zero_initial_state(
                    tidx, tmem_ptr, tiled_mma_kv, kv_acc_producer
                )
        return kv_acc_producer

    @cute.jit
    def _zero_initial_state(self, tidx, tmem_ptr, tiled_mma_kv, kv_acc_producer):
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
        state_r2t_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        cState = cute.make_identity_tensor((self.mma_tiler_kv[0], self.mma_tiler_kv[1]))
        tiled_state_r2t = tcgen05.make_tmem_copy(
            state_r2t_atom, tCtState[(None, None), 0, 0, 0]
        )
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tRT_tCcState = thr_state_r2t.partition_S(cState)
        rState = cute.make_rmem_tensor_like(tRT_tCcState, self.acc_dtype)
        rState.fill(0.0)

        kv_acc_handle = kv_acc_producer.acquire_and_advance()
        for sub in cutlass.range(rState.shape[2]):
            cute.copy(
                tiled_state_r2t,
                rState[None, 0, sub],
                tRT_tCtState[None, 0, sub, kv_acc_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()
        self.init_state_store_barrier.arrive_and_wait()
        if cg1_tidx == 0:
            cute.arch.mbarrier_arrive(kv_acc_handle.barrier)
        return kv_acc_producer

    @cute.jit
    def _load_initial_state(
        self,
        tidx,
        mS_init,
        mS_indices,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        kv_acc_producer,
    ) -> pipeline.PipelineProducer:
        """Load S_init from GMEM into state TMEM (fp32).

        Two steps:
          1. GMEM fp32 -> registers
          2. registers -> state TMEM (fp32), then signal the state-update issuer
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM store copy (registers -> state TMEM)
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
        state_r2t_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        cState = cute.make_identity_tensor((self.mma_tiler_kv[0], self.mma_tiler_kv[1]))
        tCtState_for_r2t = tCtState[(None, None), 0, 0, 0]
        tiled_state_r2t = tcgen05.make_tmem_copy(state_r2t_atom, tCtState_for_r2t)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tRT_tCcState = thr_state_r2t.partition_S(cState)
        tRT_tCrState = cute.make_rmem_tensor_like(tRT_tCcState, self.acc_dtype)
        tGR_tCrState = cute.make_rmem_tensor_like(tRT_tCcState, mS_init.element_type)

        if cutlass.const_expr(mS_indices is not None):
            state_idx = mS_indices[batch_idx]
        else:
            state_idx = batch_idx
        gS_init = cute.flat_divide(
            mS_init[None, None, head_idx, state_idx],
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
        )[None, None, 0, 0]
        tGR_tCgState = thr_state_r2t.partition_S(gS_init)
        kv_acc_handle = kv_acc_producer.acquire_and_advance()
        for sub in cutlass.range(tGR_tCrState.shape[2]):
            # 1. Load S_init state_dtype GMEM -> state_dtype registers
            cute.autovec_copy(
                tGR_tCgState[None, 0, sub],
                tGR_tCrState[None, 0, sub],
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            if cutlass.const_expr(self.acc_dtype != mS_init.element_type):
                tRT_tCrState[None, 0, sub].store(
                    tGR_tCrState[None, 0, sub].load().to(self.acc_dtype)
                )
            else:
                tRT_tCrState = tGR_tCrState

            # 2. fp32 registers -> state TMEM; the state update accumulates on top.
            cute.copy(
                tiled_state_r2t,
                tRT_tCrState[None, 0, sub],
                tRT_tCtState[None, 0, sub, kv_acc_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()

        # Manually sync before committing - CG1 is not the MMA warp so uses mbarrier_arrive.
        self.init_state_store_barrier.arrive_and_wait()
        if cg1_tidx == 0:
            cute.arch.mbarrier_arrive(kv_acc_handle.barrier)

        return kv_acc_producer

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
                if seqlen_b % checkpoint_every_n_tokens == 0:
                    num_valid_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                    if num_valid_chunks_b % 2 == 0:
                        gS_checkpoints = cute.flat_divide(
                            mS_checkpoints[None, None, head_idx, checkpoint_offset],
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
                    state_idx = mS_indices[batch_idx]
                else:
                    state_idx = batch_idx
                gS_out = cute.flat_divide(
                    mS_out[None, None, head_idx, state_idx],
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
    def mma_cg1_chunk(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        is_first_chunk: cutlass.Boolean,
    ) -> tuple:
        """Issue KS/QS/NV/QKV/KV for one chunk."""
        tiled_mma_qs, tiled_mma_qkv, _, tiled_mma_kv = mma_args
        sQ, sK, sK_trans, sV, sAinv, sQk = smem_args
        (
            cg1_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            vks_ready_consumer,
            nv_ready_consumer,
            decay_v_ready_consumer,
        ) = pipeline_args

        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_cg1_shared_acc_stages)
        )
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_cg1_shared_acc_offset, tCtShared_fake.layout
        )

        shared_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtSharedInp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(shared_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtSharedInp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtSharedInp_fake.layout,
        )

        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )

        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtStateInp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtStateInp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtStateInp_fake.layout,
        )

        tCrK_B_qs = tiled_mma_qs.make_fragment_B(sK)
        tCrQ_B_qs = tiled_mma_qs.make_fragment_B(sQ)
        tCrAinv_B = tiled_mma_qkv.make_fragment_B(sAinv)
        tCrNv_B = tiled_mma_qkv.make_fragment_B(sQk)
        tCrKt_B = tiled_mma_kv.make_fragment_B(sK_trans)
        num_kphases_qs = cute.size(tCtStateInp, mode=[2])
        num_kphases_qkv = cute.size(tCrAinv_B, mode=[2])
        num_kphases_kv = cute.size(tCrKt_B, mode=[2])

        k_handle = load_k_consumer.wait_and_advance()
        q_handle = load_q_consumer.wait_and_advance()
        valid_state = is_first_chunk == False  # noqa: E712
        if cutlass.const_expr(self.use_initial_state):
            valid_state = True

        if valid_state:
            ks_handle = cg1_acc_producer.acquire_and_advance()
            state_handle = state_inp_ready_consumer.wait_and_advance()
            for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qs,
                    tCtShared[None, None, None, ks_handle.index],
                    tCtStateInp[None, None, kphase_idx, state_handle.index],
                    tCrK_B_qs[None, None, kphase_idx, k_handle.index],
                    tCtShared[None, None, None, ks_handle.index],
                )
            ks_handle.commit()

            qs_handle = q_state_acc_producer.acquire_and_advance()
            for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qs,
                    tCtQState[None, None, None, qs_handle.index],
                    tCtStateInp[None, None, kphase_idx, state_handle.index],
                    tCrQ_B_qs[None, None, kphase_idx, q_handle.index],
                    tCtQState[None, None, None, qs_handle.index],
                )
            qs_handle.commit()
            state_handle.release()

        q_handle.release()

        nv_handle = cg1_acc_producer.acquire_and_advance()
        vks_ready_consumer.wait_and_advance()
        ainv_handle = a_inv_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qkv,
                tCtShared[None, None, None, nv_handle.index],
                tCtSharedInp[None, None, kphase_idx, 0],
                tCrAinv_B[None, None, kphase_idx, ainv_handle.index],
                tCtShared[None, None, None, nv_handle.index],
            )
        nv_handle.commit()
        ainv_handle.release()

        q_state_handle = q_state_acc_producer.acquire_and_advance()
        qk_handle = qk_ready_consumer.wait_and_advance()
        nv_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            tiled_mma_qkv.set(
                tcgen05.Field.ACCUMULATE, valid_state or (kphase_idx != 0)
            )
            cute.gemm(
                tiled_mma_qkv,
                tCtQState[None, None, None, q_state_handle.index],
                tCtSharedInp[None, None, kphase_idx, 0],
                tCrNv_B[None, None, kphase_idx, qk_handle.index],
                tCtQState[None, None, None, q_state_handle.index],
            )
        qk_handle.release()
        q_state_handle.commit()

        if cutlass.const_expr(self.use_initial_state):
            if is_first_chunk:
                kv_acc_producer.advance()
        kv_handle = kv_acc_producer.acquire_and_advance()
        decay_v_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_kv, unroll_full=True):
            tiled_mma_kv.set(tcgen05.Field.ACCUMULATE, valid_state or (kphase_idx != 0))
            cute.gemm(
                tiled_mma_kv,
                tCtState[None, None, None, kv_handle.index],
                tCtSharedInp[None, None, kphase_idx, 1],
                tCrKt_B[None, None, kphase_idx, k_handle.index],
                tCtState[None, None, None, kv_handle.index],
            )
        kv_handle.commit()
        k_handle.release()

        return (
            cg1_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            vks_ready_consumer,
            nv_ready_consumer,
            decay_v_ready_consumer,
        )

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
                        if checkpoint_token % checkpoint_every_n_tokens == 0:
                            gS_checkpoints = cute.flat_divide(
                                mS_checkpoints[
                                    None,
                                    None,
                                    head_idx,
                                    checkpoint_offset,
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
            if cutlass.const_expr(self.enable_checkpoints):
                if checkpoint_token > 0:
                    if checkpoint_token <= seqlen_b:
                        if checkpoint_token % checkpoint_every_n_tokens == 0:
                            checkpoint_offset += 1
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

    @cute.jit
    def epilogue_warp(
        self,
        smem_args,
        tma_args,
        pipeline_args,
        work_args,
        tensormap_args,
    ) -> pipeline.PipelineConsumer:
        """Warp 11: TMA bulk-store O from SMEM staging buffer to global memory.

        Steps:
          1. Wait for CG1 to signal O is ready in sO (via o_store_consumer).
          2. Domain-offset the TMA tensor to (chunk_offset, head_idx), flat-divide
             into tiles, tma_partition -> (tOsO, tOgO).
          3. Issue TMA S2G bulk copy using the per-work-tile updated descriptor.
          4. Commit the async group and wait for the store to land in GMEM.
          5. Release the pipeline slot back to CG1.
        """
        (sO,) = smem_args
        (tma_o,) = tma_args
        (o_store_consumer,) = pipeline_args
        head_idx, chunk_offset = work_args
        tensormap_manager, tensormap_o_ptr = tensormap_args

        o_handle = o_store_consumer.wait_and_advance()

        cta_layout = cute.make_layout(1)
        # (BT, DV)
        o_tile = cute.select(self.mma_tiler_qkv, mode=[0, 1])

        # Position global O tile at current chunk / head
        mO = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset),
            tma_o.tma_tensor[None, None, head_idx],
        )
        # (BT, DV, num_o_tiles, ...)
        gO = cute.flat_divide(mO, o_tile)

        # TMA partition: tOsO = SMEM source, tOgO = GMEM destination
        tOsO, tOgO = cpasync.tma_partition(
            tma_o.atom,
            0,
            cta_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )

        # TMA bulk store SMEM -> GMEM using the descriptor updated per work tile
        cute.copy(
            tma_o.atom,
            tOsO[(None, o_handle.index)],
            tOgO[(None, 0, 0)],
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_o_ptr, cute.AddressSpace.generic
            ),
        )

        # Wait for the store to complete before releasing the SMEM slot
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)

        o_handle.release()

        return o_store_consumer

    @cute.jit
    def _transform_to_position_independent_layout(
        self, tensor: cute.Tensor, swizzle_inner: cute.Swizzle
    ) -> cute.Tensor:
        wo_swizzle_iter = cute.recast_ptr(tensor.iterator, swizzle_=None)
        pisl_swizzle_base = int(math.log2(self.io_dtype.width)) - 1
        pisl_swizzle = cute.make_swizzle(
            swizzle_inner.num_bits, pisl_swizzle_base, swizzle_inner.num_shift
        )
        tensor_pisl = cute.make_composed_layout(pisl_swizzle, 0, tensor.layout)
        return cute.make_tensor(wo_swizzle_iter, tensor_pisl)

    @staticmethod
    def get_workspace_size(num_sm: int, B: int, HQ: int, HV: int, is_persistent: bool):
        # q, k, v, o
        if is_persistent:
            return (
                CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap
                * CPDeltaRulePrefillTcgen05Sm100.num_tensormaps
                * num_sm
            )
        HO = HQ if HQ >= HV else HV
        return (
            CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap
            * CPDeltaRulePrefillTcgen05Sm100.num_tensormaps
            * (B * HO)
        )

    @cute.jit
    def initialize_workspace(
        self, workspace: cute.Tensor, grid_dim: Tuple[int, int, int]
    ):
        workspace = cute.make_tensor(
            workspace.iterator,
            cute.make_layout(
                (
                    grid_dim[0] * grid_dim[1] * grid_dim[2],
                    CPDeltaRulePrefillTcgen05Sm100.num_tensormaps,
                    CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap,
                ),
                stride=(
                    CPDeltaRulePrefillTcgen05Sm100.num_tensormaps
                    * CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap,
                    CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap,
                    1,
                ),
            ),
        )
        return workspace


# ---------------------------------------------------------------------------
# Test / validation entry point
# ---------------------------------------------------------------------------
