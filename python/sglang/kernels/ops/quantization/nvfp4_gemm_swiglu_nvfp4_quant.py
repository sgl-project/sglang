# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from flashinfer.fused_moe.cute_dsl.blackwell.utils import fmin, silu_f32

"""
This example provides an experimental implementation of the SM100 batched dense blockscaled
GEMM kernel, please note that the APIs and implementation details related to this kernel
may change in future releases.

A high-performance persistent batched dense blockscaled GEMM example for the NVIDIA Blackwell
SM100 architecture using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
  for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
  for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk,
  which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk,
  which has N×ceil_div(K, sf_vec_size)×L elements respectively

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global
      memory (GMEM) with TMA operations, or directly store C matrix from registers (RMEM)
      to global memory (GMEM) without TMA operations.
    - Optionally accept an elementwise lambda function epilogue_op to apply to the output tensor:
      e.g., relu can set epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is shown below:

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py            \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py        \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1                                                  \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 128 or 256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""


class Sm100BlockScaledPersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
        # Note: We don't have SFD generation support in this example for now,
        # so Float4E2M1FN output is only for internal testing and will not be released.
        - Float4E2M1FN

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        # TODO: Add 64 and 192 support
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        prefetch_dist: int = 3,
        vectorized_f32: bool = True,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel with SwiGLU fusion.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  SwiGLU Fusion:
            - The kernel computes C = up * silu(gate) where up and gate come from
              interleaved weight matrix B (granularity=64)
            - Output N dimension is N/2 due to SwiGLU fusion

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_prefetch: Enable prefetch operations (default: False).
        :type use_prefetch: bool
        :param prefetch_dist: Prefetch distance for TMA operations (default: 3).
        :type prefetch_dist: int
        :param vectorized_f32: Enable vectorized f32x2 operations for better performance (default: True).
        :type vectorized_f32: bool
        """

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.use_prefetch = use_prefetch
        self.prefetch_dist = prefetch_dist
        self.vectorized_f32 = vectorized_f32

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        # TODO: round up to 128, it is prepared for supporting N=64 or 192.
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Output tile shape for C (N dimension is halved due to SwiGLU fusion)
        self.mma_tiler_c = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1] // 2,
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        # self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
        #     self.cta_tile_shape_mnk,
        #     self.use_2cta_instrs,
        #     self.c_layout,
        #     self.c_dtype,
        # )
        self.epi_tile = (128, 64)
        # Compute epilogue tile count for SFC quantization (use output tile shape)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile[1],
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        mma_inst_tile_k = 4
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk[1] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        self.epi_tile_n_required = 2 * cute.size(self.epi_tile[1])
        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n_required
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        alpha_scale: cute.Tensor,
        c_tensor: cute.Tensor,
        sfc_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        use_pdl: cutlass.Constexpr = False,
    ):
        """Execute the GEMM operation with SwiGLU fusion and optional quantization.

        This method performs FC1 layer computation:
        1. GEMM: acc = alpha * (SFA * A) * (SFB * B)
        2. SwiGLU: out = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
        3. Optional Quant: When sfc_tensor is provided, generates SFC and quantizes output

        Steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B (weights are interleaved: [up_0:64, gate_64:128, ...])
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param alpha_scale: Pre-SwiGLU alpha scaling tensor for each L batch.
        :type alpha_scale: cute.Tensor
        :param c_tensor: Output tensor C (N dimension is N/2 due to SwiGLU)
        :type c_tensor: cute.Tensor
        :param sfc_tensor: Scale factor tensor C for quantized output (None if not quantizing)
        :type sfc_tensor: Optional[cute.Tensor]
        :param norm_const_tensor: Normalization constant for scale factor generation (None if not quantizing)
        :type norm_const_tensor: Optional[cute.Tensor]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :param use_pdl: Enable Programmatic Dependent Launch.
        :type use_pdl: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        # Determine if we need to generate scale factor C for quantization
        self.generate_sfc = sfc_tensor is not None and norm_const_tensor is not None
        if cutlass.const_expr(self.generate_sfc):
            sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(
                c_tensor.shape, self.sf_vec_size
            )
            sfc_tensor = cute.make_tensor(sfc_tensor.iterator, sfc_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size (use output tile shape for C due to SwiGLU fusion)
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Extract M dimension for epilogue boundary checks.
        m_total = cute.size(a_tensor.shape, mode=[0])

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            sfc_tensor,
            norm_const_tensor,
            alpha_scale,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            m_total,
            use_pdl,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFC_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        alpha_scale: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        m_total: cutlass.Int32,
        use_pdl: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL) - use mma_tiler_c for output due to SwiGLU fusion
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                prefetch_dist = self.prefetch_dist
                # Sending a batch of inflight Prefetches before starting TMALDG loop
                # Prefetch logic: use_prefetch for both A&B, or explicit A-only/B-only
                if self.use_prefetch:
                    # Prefetch both A and B (default behavior)
                    for k_tile in cutlass.range(
                        0, min(prefetch_dist, k_tile_cnt), unroll=1
                    ):
                        # Prefetch both A and B (default behavior)
                        cute.prefetch(
                            tma_atom_a,
                            tAgA_slice[(None, k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_b,
                            tBgB_slice[(None, k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, k_tile)],
                        )

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Prefetch logic in the loop: use_prefetch for both A&B, or explicit A-only/B-only
                    if k_tile < k_tile_cnt - prefetch_dist:
                        if self.use_prefetch:
                            # Prefetch both A and B (default behavior)
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[
                                    (None, ab_producer_state.count + prefetch_dist)
                                ],
                            )
                            cute.prefetch(
                                tma_atom_b,
                                tBgB_slice[
                                    (None, ab_producer_state.count + prefetch_dist)
                                ],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[
                                    (None, ab_producer_state.count + prefetch_dist)
                                ],
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[
                                    (None, ab_producer_state.count + prefetch_dist)
                                ],
                            )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
                tTR_gC,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )

            # Setup SFC tensor partition for quantization (if needed)
            if cutlass.const_expr(self.generate_sfc):
                norm_const = norm_const_tensor[0]
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gSFC_mnl = cute.local_tile(mSFC_mnl, epi_tile, (None, None, None))
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
                tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrSFC = cute.make_rmem_tensor(
                    tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype
                )
                tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            current_alpha_scale = alpha_scale[work_tile.tile_idx[2]]

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_index == 0
                        else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                if cutlass.const_expr(self.generate_sfc):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgSFC_mn = tCgSFC_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                batch_idx = cur_tile_coord[2]

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Process accumulator subtiles with SwiGLU fusion and store to global memory
                # Each iteration processes a pair of subtiles (up, gate) and computes
                # up * silu(gate)
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                    real_subtile_idx = subtile_idx // 2
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n_required
                                - 1
                                - subtile_idx // 2
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    # Load both up and gate subtiles
                    #
                    tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
                    tTR_tAcc_mn_gate = tTR_tAcc[
                        (None, None, None, real_subtile_idx * 2 + 1)
                    ]

                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx // 2 == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    acc_vec_up = tTR_rAcc_up.load()
                    acc_vec_gate = tTR_rAcc_gate.load()

                    #
                    # SwiGLU activation: output = up * silu(gate)
                    # where silu(x) = x * sigmoid(x)
                    # up and gate are extracted from interleaved accumulator subtiles
                    #
                    tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                    if cutlass.const_expr(self.vectorized_f32):
                        # SwiGLU Packed Version: uses f32x2 packed operations for better performance
                        # Computes: output = (alpha * up) * silu(alpha * gate)
                        # where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                        LOG2_E = cutlass.Float32(1.4426950408889634)
                        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_up), 2):
                            acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                (acc_vec_up[i], acc_vec_up[i + 1]),
                                (
                                    cutlass.Float32(current_alpha_scale),
                                    cutlass.Float32(current_alpha_scale),
                                ),
                            )
                            acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                (
                                    cutlass.Float32(current_alpha_scale),
                                    cutlass.Float32(current_alpha_scale),
                                ),
                            )
                            tCompute_log2e = cute.arch.mul_packed_f32x2(
                                (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                (-LOG2_E, -LOG2_E),
                            )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.add_packed_f32x2(
                                (
                                    cute.math.exp2(tCompute_log2e[0], fastmath=True),
                                    cute.math.exp2(tCompute_log2e[1], fastmath=True),
                                ),
                                (1.0, 1.0),
                            )
                            tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                            tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (tCompute[i], tCompute[i + 1]),
                                (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                            )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (tCompute[i], tCompute[i + 1]),
                                (acc_vec_up_alpha[0], acc_vec_up_alpha[1]),
                            )
                    else:
                        # SwiGLU Unpacked Version: scalar operations
                        # Computes: output = (alpha * up) * silu(alpha * gate)
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                            acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(
                                current_alpha_scale
                            )
                            acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(
                                current_alpha_scale
                            )
                            tCompute[i] = acc_vec_up_alpha * silu_f32(
                                acc_vec_gate_alpha, fastmath=True
                            )

                    if cutlass.const_expr(self.generate_sfc):
                        #
                        # Quantization path for Float4E2M1FN output:
                        # 1. Compute per-vector absolute max from SwiGLU result
                        # 2. Generate scale factor C (SFC) based on max values
                        # 3. Store SFC to global memory
                        # 4. Quantize output by scaling with reciprocal of SFC
                        #
                        # Guard: skip SFC store for CTA tiles beyond the M boundary.
                        # In 2CTA mode, CTA 1 may have cur_tile_coord[0] beyond
                        # the actual M tile count; direct memory SFC store would
                        # write OOB and corrupt adjacent GPU memory.
                        m_tile_in_bounds = (
                            cur_tile_coord[0] * self.epi_tile[0] < m_total
                        )
                        # Assume subtile partitioned always happens on n dimension
                        sfc_subtile_idx_mn = (
                            cur_tile_coord[0] * self.epi_tile_cnt[0],
                            cur_tile_coord[1] * self.epi_tile_cnt[1] + real_subtile_idx,
                        )
                        tCgSFC = tCgSFC_mn[
                            (
                                None,
                                None,
                                None,
                                *sfc_subtile_idx_mn,
                            )
                        ]

                        #
                        # Get absolute max across a vector and Compute SFC
                        #
                        tTR_rAcc_frg = cute.logical_divide(
                            tCompute, cute.make_layout(self.sf_vec_size)
                        )
                        acc_frg = tTR_rAcc_frg.load()
                        acc_frg = epilogue_op(acc_frg)

                        # Apply element-wise absolute value using math.absf (supports vectors)
                        abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                        abs_acc_frg = type(acc_frg)(
                            abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype
                        )

                        if cutlass.const_expr(self.vectorized_f32):
                            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                    cute.ReductionOp.MAX,
                                    cutlass.Float32(0.0),
                                    0,  # Use 0.0 as init for abs values
                                )
                            for vi in cutlass.range_constexpr(
                                0, abs_acc_frg.shape[1], 2
                            ):
                                tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                        (
                                            self.get_dtype_rcp_limits(self.c_dtype),
                                            self.get_dtype_rcp_limits(self.c_dtype),
                                        ),
                                    )
                                )
                                tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                        (norm_const, norm_const),
                                    )
                                )
                        else:
                            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                tCrSFC_pvscale[vi] = (
                                    abs_acc_frg[None, vi].reduce(
                                        cute.ReductionOp.MAX,
                                        cutlass.Float32(0.0),
                                        0,  # Use 0.0 as init for abs values
                                    )
                                    * self.get_dtype_rcp_limits(self.c_dtype)
                                    * norm_const
                                )

                        # Store SFC to register
                        tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))

                        #
                        # Store SFC to global memory (guarded for M boundary)
                        #
                        if m_tile_in_bounds:
                            cute.autovec_copy(tCrSFC, tCgSFC)

                        #
                        # Compute quantized output values and convert to C type
                        #
                        tCrSFC_qpvscale_up = tCrSFC.load().to(cutlass.Float32)
                        fp32_max = cutlass.Float32(3.40282346638528859812e38)
                        if cutlass.const_expr(self.vectorized_f32):
                            for vi in cutlass.range_constexpr(0, cute.size(tCrSFC), 2):
                                acc_scale = cute.arch.mul_packed_f32x2(
                                    (
                                        cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi]),
                                        cute.arch.rcp_approx(
                                            tCrSFC_qpvscale_up[vi + 1]
                                        ),
                                    ),
                                    (norm_const, norm_const),
                                )
                                acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
                                acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)

                                vec0 = tTR_rAcc_frg[None, vi]
                                vec1 = tTR_rAcc_frg[None, vi + 1]
                                for ei in cutlass.range_constexpr(self.sf_vec_size):
                                    vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                        (vec0[ei], vec1[ei]),
                                        (acc_scale_min0, acc_scale_min1),
                                    )
                        else:
                            for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                                acc_scale = norm_const * cute.arch.rcp_approx(
                                    tCrSFC_qpvscale_up[vi]
                                )
                                acc_scale = fmin(acc_scale, fp32_max, nan=True)

                                vec = tTR_rAcc_frg[None, vi]
                                for ei in cutlass.range_constexpr(self.sf_vec_size):
                                    vec[ei] = vec[ei] * acc_scale

                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                    else:
                        #
                        # Convert to C type (non-quantization path)
                        #
                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tRS_rC.store(acc_vec)

                    #
                    # Store C to shared memory
                    #
                    num_prev_subtiles = num_prev_subtiles + 1
                    c_buffer = num_prev_subtiles % self.num_c_stage

                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_view_async_shared()
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()

                work_tile = tile_sched.get_current_work()

                # Update dense alpha when the L batch changes.
                if work_tile.is_valid_tile and batch_idx != work_tile.tile_idx[2]:
                    current_alpha_scale = alpha_scale[work_tile.tile_idx[2]]

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()
            if cutlass.const_expr(use_pdl):
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.griddepcontrol_launch_dependents()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it
        to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination). Returns separate accumulator tensors
        for up and gate values used in SwiGLU fusion, plus tTR_gC.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate, tTR_gC) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_up: The accumulated tensor in register for up values
            - tTR_rAcc_gate: The accumulated tensor in register for gate values
            - tTR_gC: The partitioned output tensor for coordinate mapping
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N) - for up values
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N) - for gate values
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate, tTR_gC

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition
        register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Get the reciprocal of the maximum representable value for quantization.

        This is used to compute the scale factor for block-scaled quantization.
        The scale factor is computed as: sf = max_abs_value * rcp_limit * norm_const

        :param dtype: The target data type for quantization
        :type dtype: Type[cutlass.Numeric]
        :return: The reciprocal of the maximum representable value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1.0 / 6.0  # 6.0 is max value for FP4 E2M1
        elif dtype == cutlass.Float8E4M3FN:
            return 1.0 / 448.0  # Max value for FP8 E4M3
        elif dtype == cutlass.Float8E5M2:
            return 1.0 / 57344.0  # Max value for FP8 E5M2
        else:
            return 1.0

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        # TODO: Currently we don't support m major output for Float4E2M1FN
        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        # TODO: Add tile_n=64 and tile_n=192 support
        if mma_tiler_mn[1] not in [64, 128, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False

        # Skip invalid cluster shape
        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False

        # Dense N must be divisible by the CTA N tile size.
        # CTA N tile = mma_tiler_mn[1] (e.g. 128 for (128,128), 256 for (256,256))
        cta_tile_shape_n = mma_tiler_mn[1]
        if n % cta_tile_shape_n != 0:
            can_implement = False

        # cluster_m > 1 requires 2CTA mode (mma_m=256)
        if cluster_shape_mn[0] > 1 and mma_tiler_mn[0] != 256:
            can_implement = False

        return can_implement

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: Optional[cute.Pointer],
        alpha_ptr: cute.Pointer,
        norm_const_ptr: Optional[cute.Pointer],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        use_pdl: cutlass.Constexpr = False,
    ):
        """Wrapper function to create cute tensors from raw pointers and call the kernel.

        This wrapper is designed for integration with TensorRT-LLM custom ops.
        It creates the appropriate cute tensor layouts from raw pointers and dimensions.

        :param a_ptr: Pointer to input activation tensor A (M, K, L)
        :type a_ptr: cute.Pointer
        :param b_ptr: Pointer to weight tensor B (N, K, L)
        :type b_ptr: cute.Pointer
        :param a_sf_ptr: Pointer to scale factor tensor for A
        :type a_sf_ptr: cute.Pointer
        :param b_sf_ptr: Pointer to scale factor tensor for B
        :type b_sf_ptr: cute.Pointer
        :param c_ptr: Pointer to output tensor C (M, N//2, L) - N//2 due to SwiGLU
        :type c_ptr: cute.Pointer
        :param c_sf_ptr: Pointer to scale factor tensor for C (can be null)
        :type c_sf_ptr: Optional[cute.Pointer]
        :param alpha_ptr: Pointer to dense alpha scale tensor (L)
        :type alpha_ptr: cute.Pointer
        :param norm_const_ptr: Pointer to normalization constant for SFC generation (can be null)
        :type norm_const_ptr: Optional[cute.Pointer]
        :param m: M dimension (number of tokens/rows)
        :type m: cutlass.Int64
        :param n: N dimension (full weight width, before SwiGLU)
        :type n: cutlass.Int64
        :param k: K dimension (hidden size)
        :type k: cutlass.Int64
        :param l: L dimension, typically 1 for dense
        :type l: cutlass.Int64
        :param scaling_vector_size: Vector size for block scaling (typically 16)
        :type scaling_vector_size: cutlass.Constexpr
        :param max_active_clusters: Maximum number of active clusters for persistent scheduling
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for kernel execution
        :type stream: cuda.CUstream
        :param use_pdl: Enable Programmatic Dependent Launch.
        :type use_pdl: cutlass.Constexpr
        """
        # Compute derived dimensions
        scale_k = k // scaling_vector_size
        n_out = n // 2  # Output N dimension after SwiGLU fusion
        scale_n_out = n_out // scaling_vector_size

        # Create A tensor: (M, K, L) row-major K
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2))
        )

        # Create B tensor: (N, K, L) row-major K
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )

        # Create C tensor: (M, N//2, L) row-major N
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, n_out, l), order=(1, 0, 2))
        )

        # Create scale factor tensors with blockscaled MMA layout
        # Layout: (32, 4, ceil_div(dim, 128), 4, scale_k // 4, L) with order (2, 1, 4, 0, 3, 5)
        # Use ceil_div for M dimension to avoid zero-volume tensors when M < 128
        m_blocks = (m + 127) // 128
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m_blocks, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )

        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )

        # Create C scale factor tensor (optional, for FP4 output quantization)
        # Use m_blocks (ceil_div) to avoid zero-volume tensors when M < 128
        c_sf = None
        if cutlass.const_expr(c_sf_ptr is not None):
            c_sf = cute.make_tensor(
                c_sf_ptr,
                layout=cute.make_ordered_layout(
                    (32, 4, m_blocks, 4, scale_n_out // 4, l), order=(2, 1, 4, 0, 3, 5)
                ),
            )

        # Create dense alpha scale tensor: (L)
        alpha = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_layout((l,)),
        )

        # Create norm_const tensor (optional, for FP4 output quantization)
        norm_const = None
        if cutlass.const_expr(norm_const_ptr is not None):
            norm_const = cute.make_tensor(norm_const_ptr, layout=cute.make_layout((1,)))

        return self(
            a,
            b,
            a_sf,
            b_sf,
            alpha,
            c,
            c_sf,
            norm_const,
            max_active_clusters=max_active_clusters,
            stream=stream,
            use_pdl=use_pdl,
        )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.jit
def cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
    sf_mma_tensor: cute.Tensor,
    sf_ref_tensor: cute.Tensor,
):
    """Convert scale factor tensor from mma specification M(32x4xrest_m)xK(4xrest_k)xL layout to MKL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_ref_tensor[mkl_coord] = sf_mma_tensor[mkl_coord]


# ---------------------------------------------------------------------------
# SGLang-side wrapper, helpers, and per-shape compile cache.
# ---------------------------------------------------------------------------

import torch  # noqa: E402
from flashinfer.cute_dsl.utils import (  # noqa: E402
    get_cutlass_dtype,
    get_max_active_clusters,
    make_ptr,
)
from flashinfer.utils import get_compute_capability  # noqa: E402


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def interleave_linear_and_gate(
    tensor: torch.Tensor,
    group_size: int = 64,
    dim: int = 0,
) -> torch.Tensor:
    """Rewrite ``[linear all][gate all]`` along ``dim`` as
    ``[linear chunk][gate chunk]…`` with ``group_size`` rows per chunk.

    Matches the FC1 GEMM+SwiGLU layout the fused-gemm kernel expects.
    """
    if tensor.ndim == 0:
        raise ValueError("expected a tensor with at least one dimension")
    dim = dim % tensor.ndim
    sizes = tensor.size()
    dim_size = sizes[dim]
    if dim_size % (group_size * 2) != 0:
        raise ValueError(
            f"dimension {dim} size {dim_size} must be divisible by "
            f"2 * group_size={2 * group_size}"
        )
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    return (
        tensor.reshape(
            *prev_sizes,
            2,
            dim_size // (group_size * 2),
            group_size,
            *post_sizes,
        )
        .transpose(dim, dim + 1)
        .reshape(*sizes)
        .contiguous()
    )


def swizzle_blockscale_2d(scales: torch.Tensor) -> torch.Tensor:
    """Standard CUTLASS block-scale 2D swizzle: pad to (128, 4) tiles then
    permute into the layout the FP4 GEMM kernel reads."""
    M, K = scales.shape
    M_padded = _round_up(M, 128)
    K_padded = _round_up(K, 4)
    padded = torch.zeros((M_padded, K_padded), dtype=scales.dtype, device=scales.device)
    padded[:M, :K] = scales
    rows, cols = padded.shape
    padded = padded.reshape(rows // 128, 4, 32, cols // 4, 4)
    padded = padded.permute((0, 3, 2, 1, 4))
    return padded.contiguous().reshape(M_padded, K_padded)


_compiled_kernel_cache: dict[tuple, object] = {}


def _get_compiled(
    *,
    a_ptr,
    b_ptr,
    a_sf_ptr,
    b_sf_ptr,
    c_ptr,
    c_sf_ptr,
    alpha_ptr,
    norm_const_ptr,
    m: int,
    n: int,
    k: int,
    l: int,
    max_active_clusters: int,
    stream,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_prefetch: bool,
    prefetch_dist: int,
    vectorized_f32: bool,
    enable_pdl: bool,
):
    cache_key = (
        n,
        k,
        ab_dtype,
        sf_dtype,
        c_dtype,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        use_prefetch,
        prefetch_dist,
        vectorized_f32,
        enable_pdl,
    )
    cached = _compiled_kernel_cache.get(cache_key)
    if cached is not None:
        return cached
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_prefetch=use_prefetch,
        prefetch_dist=prefetch_dist,
        vectorized_f32=vectorized_f32,
    )
    compiled = cute.compile(
        gemm.wrapper,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        c_sf_ptr,
        alpha_ptr,
        norm_const_ptr,
        m,
        n,
        k,
        l,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
        use_pdl=enable_pdl,
    )
    _compiled_kernel_cache[cache_key] = compiled
    return compiled


def nvfp4_gemm_swiglu_nvfp4_quant(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_global_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    ab_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    c_dtype: str = "float4_e2m1fn",
    sf_vec_size: int = 16,
    use_prefetch: bool = False,
    prefetch_dist: int = 3,
    vectorized_f32: bool = True,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 GEMM fused with SwiGLU and NVFP4 output quantization.

    Args:
        a: FP4-packed input activation, shape ``[M, K / 2]``.
        a_scale: Swizzled NVFP4 input scales,
            shape ``[round_up(M,128), round_up(K/16,4)]``.
        b: FP4-packed interleaved FC1 weight, shape ``[2 * I, K / 2]``.
        b_scale: Swizzled interleaved FC1 weight scales.
        alpha: GEMM global dequant scale, scalar or ``[1, 1]``.
        output_global_scale: Output quantization scale-up factor (= 1 /
            down_proj.input_scale_inv).
        enable_pdl: Enable Programmatic Dependent Launch for the fused kernel.

    Returns:
        ``(out_fp4, out_scale)`` directly consumable by the NVFP4 ``down_proj``.
    """
    if ab_dtype != "float4_e2m1fn" or c_dtype != "float4_e2m1fn":
        raise ValueError(
            "nvfp4_gemm_swiglu_nvfp4_quant currently supports NVFP4 input "
            "and output only"
        )
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("nvfp4_gemm_swiglu_nvfp4_quant requires CUDA tensors")

    major, minor = get_compute_capability(a.device)
    if major != 10:
        raise ValueError(
            f"nvfp4_gemm_swiglu_nvfp4_quant requires SM100, got SM{major}{minor}"
        )

    m = a.shape[0]
    k = a.shape[1] * 2
    n = b.shape[0]
    if b.shape[1] * 2 != k:
        raise ValueError(f"Shape mismatch: A K={k}, B K={b.shape[1] * 2}")
    if n % 2 != 0:
        raise ValueError(f"Interleaved FC1 N must be even, got {n}")

    l = 1
    n_out = n // 2
    if n_out % sf_vec_size != 0:
        raise ValueError(
            f"Output N={n_out} must be divisible by sf_vec_size={sf_vec_size}"
        )
    scale_n_out = n_out // sf_vec_size
    padded_m = _round_up(m, 128)
    padded_scale_n = _round_up(scale_n_out, 4)

    ab_dtype_cutlass = get_cutlass_dtype(ab_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    c_dtype_cutlass = get_cutlass_dtype(c_dtype)

    if m <= 128:
        mma_tiler_mn, cluster_shape_mn = (128, 128), (1, 2)
    else:
        mma_tiler_mn, cluster_shape_mn = (256, 128), (2, 1)

    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        c_dtype_cutlass,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major="k",
        b_major="k",
        c_major="n",
    ):
        raise ValueError(
            "Unsupported nvfp4_gemm_swiglu_nvfp4_quant configuration: "
            f"shape=(M={m}, N={n}, K={k}), mma_tiler_mn={mma_tiler_mn}, "
            f"cluster_shape_mn={cluster_shape_mn}"
        )

    if out is None:
        out = torch.empty((m, n_out // 2), dtype=torch.uint8, device=a.device)
    if out_scale is None:
        out_scale = torch.empty(
            (padded_m, padded_scale_n),
            dtype=torch.float8_e4m3fn,
            device=a.device,
        )

    if alpha.dim() == 0 or alpha.dim() == 1:
        alpha = alpha.view(1, 1)
    if output_global_scale.dim() == 0:
        output_global_scale = output_global_scale.view(1)

    a_ptr = make_ptr(
        ab_dtype_cutlass, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    b_ptr = make_ptr(
        ab_dtype_cutlass, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    a_sf_ptr = make_ptr(
        sf_dtype_cutlass, a_scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_sf_ptr = make_ptr(
        sf_dtype_cutlass, b_scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype_cutlass, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    c_sf_ptr = make_ptr(
        sf_dtype_cutlass, out_scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    norm_const_ptr = make_ptr(
        cutlass.Float32,
        output_global_scale.data_ptr(),
        cute.AddressSpace.gmem,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    compiled_gemm = _get_compiled(
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        a_sf_ptr=a_sf_ptr,
        b_sf_ptr=b_sf_ptr,
        c_ptr=c_ptr,
        c_sf_ptr=c_sf_ptr,
        alpha_ptr=alpha_ptr,
        norm_const_ptr=norm_const_ptr,
        m=m,
        n=n,
        k=k,
        l=l,
        max_active_clusters=max_active_clusters,
        stream=stream,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_prefetch=use_prefetch,
        prefetch_dist=prefetch_dist,
        vectorized_f32=vectorized_f32,
        enable_pdl=bool(enable_pdl),
    )

    compiled_gemm(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        c_sf_ptr,
        alpha_ptr,
        norm_const_ptr,
        m,
        n,
        k,
        l,
        stream=stream,
    )
    return out, out_scale


__all__ = [
    "interleave_linear_and_gate",
    "nvfp4_gemm_swiglu_nvfp4_quant",
    "swizzle_blockscale_2d",
]
