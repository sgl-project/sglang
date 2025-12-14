import nvshmem.core
import torch.distributed._symmetric_memory as symm_mem

import os
import argparse
from typing import Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed as dist
import cuda.bindings.driver as cuda
from cuda.core.experimental import Device
from cuda.pathfinder import load_nvidia_dynamic_lib


import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float16, BFloat16, Float32, Float8E5M2, Float8E4M3FN
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

try:
    import nvshmem.core
except ImportError as exc:
    raise ImportError(
        "nvshmem4py is required but not installed. Please install it using:\n"
        "  For CUDA 12: pip install nvshmem4py-cu12\n"
        "  For CUDA 13: pip install nvshmem4py-cu13\n"
        "Note: nvshmem4py version >= 0.1.3 is recommended."
    ) from None

try:
    load_nvidia_dynamic_lib("nvshmem_host")
except RuntimeError as exc:
    raise ImportError(
        "nvshmem lib is required but not installed. Please install it using:\n"
        "  For CUDA 12: pip install nvidia-nvshmem-cu12\n"
        "  For CUDA 13: pip install nvidia-nvshmem-cu13\n"
    ) from None

def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    use_tma_store: bool,
    c_smem_layout: Union[cute.Layout, None],
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
    :param c_dtype: Data type of operand C (output).
    :type c_dtype: type[cutlass.Numeric]
    :param smem_capacity: Total available shared memory capacity in bytes.
    :type smem_capacity: int
    :param occupancy: Target number of CTAs per SM (occupancy).
    :type occupancy: int
    :param use_tma_store: Whether TMA store is enabled.
    :type use_tma_store: bool
    :param c_smem_layout: Layout of C operand in shared memory, or None if not using TMA store.
    :type c_smem_layout: Union[cute.Layout, None]

    :return: A tuple containing the computed number of stages for:
             (ACC stages, A/B operand stages, C stages)
    :rtype: tuple[int, int, int]
    """
    # Default ACC stages
    num_acc_stage = 2

    # Default C stages
    num_c_stage = 2 if use_tma_store else 0

    # Calculate smem layout and size for one stage of A, B, and C with 1-stage
    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a_dtype, 1
    )
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b_dtype, 1
    )

    ab_bytes_per_stage = cute.size_in_bytes(
        a_dtype, a_smem_layout_stage_one
    ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    # Calculate A/B stages:
    # Start with total smem per CTA (capacity / occupancy)
    # Subtract reserved bytes and initial C stages bytes
    # Divide remaining by bytes needed per A/B stage
    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    # Refine epilogue stages:
    # Calculate remaining smem after allocating for A/B stages and reserved bytes
    # Add remaining unused smem to epilogue
    if use_tma_store:
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
    return num_acc_stage, num_ab_stage, num_c_stage


class PersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
    :type use_tma_store: bool
    :param all_reduce: All-reduce mode, can be "none", "LDMCxSTMC"
    :type all_reduce: str

    :note: only support two_shot all-reduce mode in a single node right now

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    Example:
        >>> gemm = PersistentDenseGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        rank_id: int,
        world_size: int,
        all_reduce="two_shot",
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3. Output C tensor store mode:
            - use_tma_store: Boolean indicating whether to use Tensor Memory Access (TMA) for storing results.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Use Tensor Memory Access (TMA) or normal store for output C tensor.
        :type use_tma_store: bool
        :param all_reduce: All-reduce mode, can be "none", "LDMCxSTMC"
        :type all_reduce: str
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.all_reduce_warp_id = (6, 7, 8, 9)
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilogue_warp_id,
                *self.all_reduce_warp_id,
            )
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.epilogue_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.all_reduce_sync_bar_id = 4
        self.all_reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.all_reduce_sync_bar_id,
            num_threads=32 * len(self.all_reduce_warp_id),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.all_reduce = all_reduce
        self.rank_id = rank_id
        self.num_ranks = world_size

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        c_smem_layout = None
        if cutlass.const_expr(self.use_tma_store):
            c_smem_layout = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, 1
            )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,
        )

        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        self.c_smem_layout_staged = None
        if self.use_tma_store:
            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        c_mc: cute.Tensor,
        barrier_flag: cute.Tensor,
        barrier_flag_mc: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param c_mc: Output symmetric tensor C_mc, any write or read to a multicast tensor will be broadcasted to all GPUs
        :type c_mc: cute.Tensor
        :param barrier_flag: flag tensor acting as a barrier for GPUs to synchronize
        :type barrier_flag: cute.Tensor
        :param barrier_flag_mc: the multicast tensor of barrier_flag
        :type barrier_flag_mc: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # Setup TMA load for B
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
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
            )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            c_mc,
            barrier_flag,
            barrier_flag_mc,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        c_mc: cute.Tensor,
        barrier_flag: cute.Tensor,
        barrier_flag_mc: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
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
            if cutlass.const_expr(self.use_tma_store):
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
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
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
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
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
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
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
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

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
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

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

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

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

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

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
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblk_crd],
                                tCrB[kblk_crd],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

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

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            if cutlass.const_expr(self.use_tma_store):

                assert tma_atom_c is not None and sC is not None
                self.epilogue_tma_store_release_flag(
                    tidx,
                    warp_idx,
                    acc_pipeline,
                    tiled_mma,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    epi_tile,
                    tile_sched,
                    epilogue_op,
                    flag_base=barrier_flag_mc,
                    flag_mem_scope="gpu",
                )
            else:
                self.epilogue_release_flag(
                    tidx,
                    acc_pipeline,
                    tiled_mma,
                    tCtAcc_base,
                    tCgC,
                    epi_tile,
                    tile_sched,
                    epilogue_op,
                    tmem_dealloc_barrier,
                    flag_base=barrier_flag_mc,
                    flag_mem_scope="gpu",
                )

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Allreduce warps
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.all_reduce == "LDMCxSTMC"):
            if warp_idx >= self.all_reduce_warp_id[0]:
                # ///////////////////////////////////////////////////////////////////////////////
                # Add persistent tile loop
                # ///////////////////////////////////////////////////////////////////////////////

                rank_id = self.rank_id
                num_ranks = Int32(self.num_ranks)
                lane_id = cute.arch.lane_idx()

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                # we want 128bit ld/st for better performance
                atom_val = 128 // c_mc.element_type.width
                atom_thr_n = self.mma_tiler[1] // atom_val
                atom_thr_m = len(self.all_reduce_warp_id) * cute.arch.WARP_SIZE // atom_thr_n
                thr_layout = cute.make_layout(
                    (atom_thr_m, atom_thr_n), stride=(atom_thr_n, 1)
                )
                val_layout = cute.make_layout((1, atom_val), stride=(atom_val, 1))

                copy_atom_load = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), c_mc.element_type
                )
                tiled_copy_fake = cute.make_tiled_copy_tv(
                    copy_atom_load, thr_layout, val_layout
                )
                thr_copy_fake = tiled_copy_fake.get_slice(
                    tidx - self.all_reduce_warp_id[0] * 32
                )
                # predicate tensor
                idC = cute.make_identity_tensor(c_mc.shape)

                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    tile_id = Int32(
                        tile_sched._current_work_linear_idx
                        * cute.size(self.cluster_shape_mn)
                        + cute.arch.block_idx_in_cluster()
                    )
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )

                    # System barrier to make sure that data from each GPU is in memory before allreduce
                    if warp_idx == self.all_reduce_warp_id[0]:
                        with cute.arch.elect_one():
                            flag = barrier_flag.iterator + tile_id
                            utils.distributed.spin_lock_atom_cas_relaxed_wait(
                                lock_ptr=flag,
                                expected_val=num_ranks,
                                reset_val=0,
                                scope="gpu",
                            )

                    self.all_reduce_sync_barrier.arrive_and_wait()
                    # partition and slice at tile level
                    gC_mc = cute.local_tile(
                        c_mc,
                        cute.slice_(self.mma_tiler, (None, None, 0)),
                        (None, None, None),
                    )
                    cC = cute.local_tile(
                        idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
                    )
                    
                    tCgC_mc = thr_mma.partition_C(gC_mc)
                    tCpC = thr_mma.partition_C(cC)
                    tCgC_mc_slice = tCgC_mc[((None, None), 0, 0, *mma_tile_coord_mnl)]
                    tCpC_slice = tCpC[((None, None), 0, 0, *mma_tile_coord_mnl)]

                    # partition based on the number of GPUs
                    cta_mma_tile_m = self.mma_tiler[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    m_local_rank = int(cta_mma_tile_m / self.num_ranks)
                    tCgC_mc_slice_partitioned = cute.zipped_divide(
                        tCgC_mc_slice, (m_local_rank, self.mma_tiler[1])
                    )
                    tCpC_slice_partitioned = cute.zipped_divide(tCpC_slice, (m_local_rank, self.mma_tiler[1]))
                    tCgC_mc_local_rank = cute.slice_(
                        tCgC_mc_slice_partitioned, ((None, None), (rank_id, 0))
                    )
                    tCpC_local_rank = cute.slice_(tCpC_slice_partitioned, ((None, None), (rank_id, 0)))

                    # partition at thread level
                    frgC_mc = thr_copy_fake.partition_S(tCgC_mc_local_rank)
                    frpC = thr_copy_fake.partition_S(tCpC_local_rank)
                    atom, loop_m, loop_n = frgC_mc.shape
                    for i in cutlass.range_constexpr(loop_m):
                        for j in cutlass.range_constexpr(loop_n):
                            if cute.elem_less(frpC[0, i, j], c_mc.shape):
                                mc_ptr = frgC_mc[None, i, j].iterator
                                x, y, z, w = 0, 0, 0, 0
                                if cutlass.const_expr(self.c_dtype == Float16):
                                    x, y, z, w = utils.distributed.multimem_ld_reduce_8xf16(
                                        mc_ptr
                                    )
                                elif cutlass.const_expr(self.c_dtype == Float32):
                                    x, y, z, w = utils.distributed.multimem_ld_reduce_4xf32(
                                        mc_ptr
                                    )
                                elif cutlass.const_expr(self.c_dtype == BFloat16):
                                    x, y, z, w = (
                                        utils.distributed.multimem_ld_reduce_8xbf16(mc_ptr)
                                    )
                                elif cutlass.const_expr(self.c_dtype == Float8E4M3FN):
                                    x, y, z, w = (
                                        utils.distributed.multimem_ld_reduce_16xe4m3(mc_ptr)
                                    )
                                elif cutlass.const_expr(self.c_dtype == Float8E5M2):
                                    x, y, z, w = (
                                        utils.distributed.multimem_ld_reduce_16xe5m2(mc_ptr)
                                    )
                                utils.distributed.multimem_st_4xb32(mc_ptr, x, y, z, w)
                    # Advance to next tile
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                self.all_reduce_sync_barrier.arrive_and_wait()

                #
                # Set Per SM Flag with Release
                #
                # This ensure
                # 1. no rank early exit while other ranks are still issuing multimem.ld_reduce
                # 2. each rank's prior multimem.st have become visiable to all other ranks in the system (w/ .SYS scope)
                if warp_idx == self.all_reduce_warp_id[0]:
                    with cute.arch.elect_one():
                        # Offset to last tile flag idx
                        last_tile_id_linear = cute.size(
                            tile_sched.params.problem_layout_ncluster_mnl
                        ) * cute.size(self.cluster_shape_mn)
                        # Linear id of current SM.
                        sm_id_linear = (
                            cute.arch.block_idx()[0]
                            + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
                            + cute.arch.block_idx()[2]
                            * cute.arch.grid_dim()[0]
                            * cute.arch.grid_dim()[1]
                        )
                        # Release flag with sys scope
                        utils.distributed.multimem_red_add1(
                            lock_ptr=barrier_flag_mc.iterator
                            + last_tile_id_linear
                            + sm_id_linear,
                            scope="sys",
                            order="release",
                        )
                        # Relaxed spin-lock wait flag with sys scope
                        utils.distributed.spin_lock_atom_cas_relaxed_wait(
                            lock_ptr=barrier_flag.iterator
                            + last_tile_id_linear
                            + sm_id_linear,
                            expected_val=num_ranks,
                            reset_val=0,
                            scope="sys",
                        )

    @cute.jit
    def epilogue_tma_store_release_flag(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        acc_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tma_atom_c: cute.CopyAtom,
        # Input of epilogue
        tCtAcc_base: cute.Tensor,
        # Staging of epilogue
        sC: cute.Tensor,
        # Output of epilogue
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        tile_sched: utils.StaticPersistentTileScheduler,
        epilogue_op: cutlass.Constexpr,
        flag_base: cute.Tensor,
        flag_mem_scope: str,
    ) -> None:
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilogue_tmem_copy_and_partition(
            epi_tidx, tCtAcc_base, tCgC, epi_tile, self.use_2cta_instrs
        )

        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = self.epilogue_smem_copy_and_partition(
            tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        # Threads/warps participating in tma store pipeline
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * len(self.epilogue_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage, producer_group=c_producer_group
        )

        epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilogue_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        work_tile = tile_sched.initial_work_tile_info()
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
            bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

            # Set tensor memory buffer for current tile
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
            tTR_tAcc = tTR_tAcc_base[
                (None, None, None, None, None, acc_consumer_state.index)
            ]

            #
            # Wait for accumulator buffer full
            #
            acc_pipeline.consumer_wait(acc_consumer_state)

            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

            #
            # Store accumulator to global memory in subtiles
            #
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
            for subtile_idx in cutlass.range(subtile_cnt):
                #
                # Load accumulator from tensor memory buffer to register
                #
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                #
                # Convert to C type
                #
                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                tRS_rC.store(acc_vec)

                #
                # Store C to shared memory
                #
                c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                # Fence and barrier to make sure shared memory store is visible to TMA store
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                epilogue_sync_barrier.arrive_and_wait()

                #
                # TMA store C to global memory
                #
                if warp_idx == self.epilogue_warp_id[0]:
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, c_buffer)],
                        bSG_gC[(None, subtile_idx)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()
                epilogue_sync_barrier.arrive_and_wait()

            epilogue_sync_barrier.arrive_and_wait()

            #
            # Async arrive accumulator buffer empty
            #
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            #
            # Set Per Output Tile Flag with Release
            #
            # 1D linear index of current output tile
            tile_id_linear = Int32(
                tile_sched._current_work_linear_idx * cute.size(self.cluster_shape_mn)
                + cute.arch.block_idx_in_cluster()
            )
            # Wait for C store complete
            # Unlike regular epilogue where we only wait C store complete once at end of each kernel.
            # Here we need to wait for C store complete for each output tile before we set the release flag.
            c_pipeline.producer_tail()
            # Update flag with release semantic with GPU scope
            if warp_idx == self.epilogue_warp_id[0]:
                with cute.arch.elect_one():
                    flag_curr_tile = flag_base.iterator + tile_id_linear
                    utils.distributed.multimem_red_add1(
                        lock_ptr=flag_curr_tile,
                        scope=flag_mem_scope,
                        order="release",
                    )

            #
            # Advance to next tile
            #
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    @cute.jit
    def epilogue_release_flag(
        self,
        epi_tidx: cutlass.Int32,
        acc_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tCtAcc_base: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        tile_sched: utils.StaticPersistentTileScheduler,
        epilogue_op: cutlass.Constexpr,
        tmem_dealloc_barrier: pipeline.NamedBarrier,
        flag_base: cute.Tensor,
        flag_mem_scope: str,
    ) -> None:
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilogue_tmem_copy_and_partition(
            epi_tidx, tCtAcc_base, tCgC, epi_tile, self.use_2cta_instrs
        )

        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].shape, self.c_dtype
        )
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        work_tile = tile_sched.initial_work_tile_info()
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
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_gC = tTR_gC_partitioned[
                (None, None, None, None, None, *mma_tile_coord_mnl)
            ]

            # Set tensor memory buffer for current tile
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_tAcc = tTR_tAcc_base[
                (None, None, None, None, None, acc_consumer_state.index)
            ]

            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

            #
            # Wait for accumulator buffer full
            #
            acc_pipeline.consumer_wait(acc_consumer_state)

            #
            # Store accumulator to global memory in subtiles
            #
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                #
                # Load accumulator from tensor memory buffer to register
                #
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                #
                # Convert to C type
                #
                acc_vec = tTR_rAcc.load()
                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                tTR_rC.store(acc_vec)

                #
                # Store C to global memory
                #
                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

            #
            # Async arrive accumulator buffer empty
            #
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            #
            # Set Per Output Tile Flag with Release
            #
            # 1D linear index of current output tile
            tile_id_linear = Int32(
                tile_sched._current_work_linear_idx * cute.size(self.cluster_shape_mn)
                + cute.arch.block_idx_in_cluster()
            )
            # Wait for C store complete
            # Unlike regular epilogue where we only wait C store complete once at end of each kernel.
            # Here we need to wait for C store complete for each output tile before we set the release flag.
            c_pipeline.producer_tail()
            # Update flag with release semantic with GPU scope
            if warp_idx == self.epilogue_warp_id[0]:
                with cute.arch.elect_one():
                    flag_curr_tile = flag_base.iterator + tile_id_linear
                    utils.distributed.multimem_red_add1(
                        lock_ptr=flag_curr_tile,
                        scope=flag_mem_scope,
                        order="release",
                    )

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Synchronize before TMEM dealloc (done by the caller)
        tmem_dealloc_barrier.arrive_and_wait()

    def epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

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

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
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
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
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
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilogue_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

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
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        return num_tmem_alloc_cols

    def is_valid_dtypes(
        self, ab_dtype: Type[cutlass.Numeric], c_dtype: Type[cutlass.Numeric]
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Uint8,
            cutlass.Int8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }
        if ab_dtype not in valid_ab_dtypes:
            return False

        if self.acc_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.Int32}:
            return False

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.TFloat32,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },  # Float32 accumulator supports floating point AB types only
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and AB type
        if ab_dtype not in acc_ab_compatibility[self.acc_dtype]:
            return False

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
            cutlass.Float16: {
                cutlass.BFloat16,
                cutlass.Float16,
            },
            cutlass.Int32: {
                cutlass.BFloat16,
                cutlass.Float16,
                cutlass.Float32,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[self.acc_dtype]:
            return False

        # check if c_dtype is supported by multimem all-reduce
        if cutlass.const_expr(
            c_dtype not in {cutlass.Float16, cutlass.Float32, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}
        ):
            return False

        return True

    def is_valid_mma_tiler_and_cluster_shape(self) -> bool:
        """Check if the mma tiler and cluster shape are valid.

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if not (
            (not self.use_2cta_instrs and self.mma_tiler_mn[0] in [64, 128])
            or (self.use_2cta_instrs and self.mma_tiler_mn[0] in [128, 256])
        ):
            is_valid = False
        if self.mma_tiler_mn[1] not in range(32, 257, 32):
            is_valid = False
        # Skip illegal cluster shape
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16
            or self.cluster_shape_mn[0] <= 0
            or self.cluster_shape_mn[1] <= 0
            or not is_power_of_2(self.cluster_shape_mn[0])
            or not is_power_of_2(self.cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    def is_valid_tensor_alignment(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
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

    def is_valid_epilogue_store_option(self, m: int, n: int) -> bool:
        """
        Check if the epilogue store option is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int

        :return: True if the epilogue store option is valid, False otherwise
        :rtype: bool
        """

        is_valid = True
        # None TMA store version does not have predication, can not support OOB tiles
        cta_tile_shape_mn = (
            self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1),
            self.mma_tiler_mn[1],
        )
        if not self.use_tma_store:
            if not (m % cta_tile_shape_mn[0] == 0 and n % cta_tile_shape_mn[1] == 0):
                is_valid = False
        return is_valid

    def can_implement(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> bool:
        """Check if the given tensors can be implemented by this kernel.

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor

        :return: True if the gemm supports the given config, False otherwise
        :rtype: bool
        """
        m, n, k, l = a.shape[0], b.shape[0], a.shape[1], a.shape[2]

        # infer a_major, b_major, c_major
        is_m_major_a = utils.LayoutEnum.from_tensor(a).is_m_major_a()
        is_n_major_b = utils.LayoutEnum.from_tensor(b).is_n_major_b()
        is_m_major_c = utils.LayoutEnum.from_tensor(c).is_m_major_c()
        a_major = "m" if is_m_major_a else "k"
        b_major = "n" if is_n_major_b else "k"
        c_major = "m" if is_m_major_c else "n"

        can_implement = True
        # Skip unsupported types
        if not self.is_valid_dtypes(a.element_type, c.element_type):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not self.is_valid_mma_tiler_and_cluster_shape():
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not self.is_valid_tensor_alignment(
            m, n, k, l, a.element_type, c.element_type, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid epilogue store option
        if not self.is_valid_epilogue_store_option(m, n):
            can_implement = False
        # currently only support world_size = 2, 4, 8
        if dist.get_world_size() not in [2, 4, 8]:
            can_implement = False
        return can_implement


class GemmARLayer:
    """A high-performance GEMM+AllReduce layer based on CUTLASS PersistentDenseGemmKernel.
    
    This layer combines matrix multiplication with all-reduce communication in a single
    fused operation, optimized for distributed training on Blackwell GPUs.
    """
    
    def __init__(
        self,
        tp_group,
        max_M: int,
        N: int, 
        K: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        local_world_size: int,
        c_tensor_mc: cute.Tensor,
        torch_symm_tensor: torch.Tensor,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Tuple[int, int] = (2, 1),
        use_2cta_instrs: bool = True,
        use_tma_store: bool = True,
        all_reduce: str = "none"
    ):
        self.tp_group = tp_group
        self.max_M = max_M
        self.N = N
        self.K = K
        
        self.input_torch_dtype = input_dtype
        self.output_torch_dtype = output_dtype
        self.input_dtype = self._torch_dtype_to_cutlass_dtype(input_dtype)
        self.output_dtype = self._torch_dtype_to_cutlass_dtype(output_dtype)
        # Set accumulator dtype - usually Float32 for better precision
        self.acc_dtype = cutlass.Float32
        self.c_tensor_mc = c_tensor_mc
        self.torch_symm_tensor = torch_symm_tensor
        self.local_world_size = local_world_size

        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.use_2cta_instrs = use_2cta_instrs
        self.use_tma_store = use_tma_store
        self.all_reduce = all_reduce

        if all_reduce != "none" and local_world_size not in [2, 4, 8]:
            raise ValueError(f"AllReduce only supports world_size of 2, 4, 8, got {local_world_size}")
        
        self._init_gemm_allreduce()
    
    def _torch_dtype_to_cutlass_dtype(self, torch_dtype):
        """Convert PyTorch dtype to CUTLASS dtype"""
        dtype_map = {
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
            torch.float32: cutlass.Float32,
            torch.int8: cutlass.Int8,
            torch.uint8: cutlass.Uint8,
        }
        if hasattr(torch, 'float8_e4m3fn'):
            dtype_map[torch.float8_e4m3fn] = cutlass.Float8E4M3FN
        if hasattr(torch, 'float8_e5m2'):
            dtype_map[torch.float8_e5m2] = cutlass.Float8E5M2
        
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported PyTorch dtype: {torch_dtype}")
        return dtype_map[torch_dtype]
    
    def _init_gemm_allreduce(self):
        """Initialize the CUTLASS GEMM+AllReduce kernel."""

        self.gemm_kernel = PersistentDenseGemmKernel(
            acc_dtype=self.acc_dtype,
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            use_tma_store=self.use_tma_store,
            rank_id=dist.get_rank(self.tp_group),
            world_size=self.local_world_size,            
            all_reduce=self.all_reduce,
        )
        
        # Get hardware info for max active clusters
        self.max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )
        
        # Pre-compile kernel with template tensors
        self._precompile_kernel()
     
    def _create_template_tensors(self):
        """Create template tensors for kernel compilation following distributed version exactly."""
        l = 1  # Batch dimension
        
        a_major = "k"
        b_major = "k"
        c_major = "n"
        a_torch_cpu = cutlass_torch.matrix(l, self.max_M, self.K, a_major == "m", self.input_dtype)
        b_torch_cpu = cutlass_torch.matrix(l, self.N, self.K, b_major == "n", self.input_dtype)
        c_torch_cpu = cutlass_torch.matrix(l, self.max_M, self.N, c_major == "m", self.output_dtype)
        # print(f"a_torch_cpu: {a_torch_cpu.shape, a_torch_cpu.stride(), a_torch_cpu.dtype}, ")
        # print(f"b_torch_cpu: {b_torch_cpu.shape, b_torch_cpu.stride(), b_torch_cpu.dtype}, ")
        # print(f"c_torch_cpu: {c_torch_cpu.shape, c_torch_cpu.stride(), c_torch_cpu.dtype}, ")

        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, self.input_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, self.input_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
            c_torch_cpu, self.output_dtype, is_dynamic_layout=True, assumed_align=16
        )
        # print(f"a_tensor: {a_tensor.shape, a_tensor._dtype}")
        # print(f"b_tensor: {b_tensor.shape, b_tensor._dtype}")
        # print(f"c_tensor: {c_tensor.shape, c_tensor._dtype}")
        if self.all_reduce != "none":
            # c_tensor, c_tensor_mc, c_torch_gpu = create_mc_tensor(
            #     c_torch_cpu, self.output_dtype, (1 if c_major == "n" else 0), is_dynamic_layout=True
            # )
            torch_tensor_gpu = self.torch_symm_tensor
            cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
            cute_tensor.element_type = self.output_dtype
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1) # if is_dynamic_layout 
            cute_tensor = cutlass_torch.convert_cute_tensor(
                torch_tensor_gpu,
                cute_tensor,
                self.output_dtype,
                is_dynamic_layout=True,
            )

            c_tensor = cute_tensor
            c_torch_gpu = torch_tensor_gpu
        return a_tensor, b_tensor, c_tensor, c_torch_gpu
    
    def _precompile_kernel(self):
        """Pre-compile the GEMM kernel with template tensors."""
        torch_stream = torch.cuda.current_stream()
        current_stream = cuda.CUstream(torch_stream.cuda_stream)
        
        a_tensor, b_tensor, c_tensor, c_torch_gpu = self._create_template_tensors()
        # c_torch_gpu is used in compare of verity
        # self.c_tensor = c_tensor
        self.c_torch_gpu = c_torch_gpu

        if not self.gemm_kernel.can_implement(a_tensor, b_tensor, c_tensor):
            raise RuntimeError(f"GEMM kernel cannot implement configuration")
        
        
        if self.all_reduce == "none":
            # print("[GEMM_ALLREDUCE] Starting compilation (all_reduce=none)...")
            # compile_start_time = time.time()
            self.compiled_gemm = cute.compile(
                self.gemm_kernel, a_tensor, b_tensor, c_tensor, 
                self.max_active_clusters, current_stream
            )
            # compile_end_time = time.time()
            # compile_duration = compile_end_time - compile_start_time
            # print(f"[GEMM_ALLREDUCE] Compilation completed in {compile_duration:.4f} seconds")
        else:
            self.barrier_flag_memref, self.barrier_flag_mc_memref = self._create_barrier_flags(self.max_M)
            # print(f"[GEMM_ALLREDUCE] Starting compilation (all_reduce={self.all_reduce})...")
            # compile_start_time = time.time()
            self.compiled_gemm = cute.compile(
                self.gemm_kernel,
                a_tensor, b_tensor, c_tensor,
                self.c_tensor_mc,
                self.barrier_flag_memref,
                self.barrier_flag_mc_memref,
                self.max_active_clusters, current_stream,
            )
            # compile_end_time = time.time()
            # compile_duration = compile_end_time - compile_start_time
            # print(f"[GEMM_ALLREDUCE] Compilation completed in {compile_duration:.4f} seconds")
    
    def _create_barrier_flags(self, m: int):
        """Create barrier flags for synchronization in all-reduce."""
        cta_tile_shape_mn = (
            self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1),
            self.mma_tiler_mn[1],
        )
        problem_shape_ntile_mn = (m // cta_tile_shape_mn[0], self.N // cta_tile_shape_mn[1])
        num_tiles = problem_shape_ntile_mn[0] * problem_shape_ntile_mn[1]
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        # Try NVSHMEM first, fallback to torch symm_mem  
        from cuda.core.experimental import Device, system
        import nvshmem.core
        from sglang.srt.distributed.device_communicators.nvshmem_communicator import get_nvshmem_comm
        nvshmem_comm = get_nvshmem_comm()
        
        if nvshmem_comm:
            # Use NVSHMEM_SYMM_MEM
            barrier_flag = nvshmem_comm.create_symmetric_tensor((num_tiles + num_sms,), dtype=torch.int32)
            barrier_flag_mc = nvshmem_comm.create_multicast_pointer(barrier_flag)
        else:
            # Use TORCH_SYMM_MEM
            barrier_flag = symm_mem.empty((num_tiles + num_sms,), device="cuda", dtype=torch.int32)
            barrier_flag.fill_(0)
            symm = symm_mem.rendezvous(barrier_flag, group=dist.group.WORLD.group_name)
            barrier_flag_mc = symm.multicast_ptr

        barrier_flag_memref = from_dlpack(barrier_flag)
        barrier_flag_memref = barrier_flag_memref.mark_layout_dynamic()
        barrier_flag_mc_memref = from_dlpack(
            cutlass_torch.as_tensor(
                barrier_flag_mc, barrier_flag.shape, barrier_flag.dtype
            ),
        )
        barrier_flag_mc_memref = barrier_flag_mc_memref.mark_layout_dynamic()
        
        return barrier_flag_memref, barrier_flag_mc_memref
    
    def forward(self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_tensor: Input tensor of shape [M, K] 
            weight: Weight tensor of shape [N, K]
        Returns:
            Output tensor of shape [M, N] with all-reduce applied
        """
        assert input_tensor.shape[1] == self.K, f"Input K dim mismatch: {input_tensor.shape[1]} vs {self.K}"
        assert weight.shape[0] == self.N, f"Weight N dim mismatch: {weight.shape[0]} vs {self.N}"
        assert weight.shape[1] == self.K, f"Weight K dim mismatch: {weight.shape[1]} vs {self.K}"
        assert bias is None, f"Bias is not supported yet"
        
        torch_stream = torch.cuda.current_stream()
        current_stream = cuda.CUstream(torch_stream.cuda_stream)

        actual_m = input_tensor.shape[0]

        input_tensor_3d = input_tensor.unsqueeze(-1)
        weight_3d = weight.unsqueeze(-1)

        # a_tensor, _ = cutlass_torch.cute_tensor_like(
        #     input_tensor_3d, self.input_dtype, is_dynamic_layout=True, assumed_align=16
        # )
        # b_tensor, _ = cutlass_torch.cute_tensor_like(
        #     weight_3d, self.input_dtype, is_dynamic_layout=True, assumed_align=16
        # )
        a_tensor = from_dlpack(input_tensor_3d, assumed_align=16)
        a_tensor.element_type = self.input_dtype
        a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1)
        b_tensor = from_dlpack(weight_3d, assumed_align=16)
        b_tensor.element_type = self.input_dtype
        b_tensor = b_tensor.mark_layout_dynamic(leading_dim=1)            

        c_tensor = from_dlpack(self.c_torch_gpu[:actual_m, :], assumed_align=16)
        c_tensor.element_type = self.output_dtype
        c_tensor = c_tensor.mark_layout_dynamic(leading_dim=1)
        
        # print(f"a_tensor: {a_tensor.shape, a_tensor._dtype}")
        # print(f"b_tensor: {b_tensor.shape, b_tensor._dtype}")
        # print(f"c_tensor: {c_tensor.shape, c_tensor._dtype}, actual_m: {actual_m}")
        # dist.barrier(group=self.tp_group)
        if self.all_reduce == "none":
            self.compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
        else:
            self.compiled_gemm(
                a_tensor, b_tensor, c_tensor,
                self.c_tensor_mc,
                self.barrier_flag_memref,
                self.barrier_flag_mc_memref,
                current_stream,
            )
        
        result = self.c_torch_gpu.squeeze(-1)
        # print(f"result: {result.shape}")
        return result[:actual_m, :]
    