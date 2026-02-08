/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Stream-K Gemm kernel compatible with fused epilogues
    that broadcast a bias vector over the MMA output.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"
#include "cutlass/semaphore.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool IsSingleSource = Epilogue_::kIsSingleSource
>
struct GemmStreamkWithFusedEpilogue;

// GemmStreamkWithFusedEpilogue with two sources
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmStreamkWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, false> {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;

  /// The per-thread tile of raw accumulators
  using AccumulatorTile = typename Mma::FragmentC;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Workspace bytes per thread block
  static size_t const kWorkspaceBytesPerBlock =
    __NV_STD_MAX(
      kThreadCount * sizeof(AccumulatorTile),
      Epilogue::kWorkspaceBytesPerBlock);

  /// Block-striped reduction utility
  using BlockStripedReduceT = BlockStripedReduce<kThreadCount, AccumulatorTile>;



  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode{GemmUniversalMode::kGemm};
    GemmCoord problem_size{};
    int batch_count{1};        // Either (mode == GemmUniversalMode::kBatched) the batch count, or (mode == GemmUniversalMode::kGemm) the tile-splitting factor

    typename EpilogueOutputOp::Params epilogue{};

    void const * ptr_A{nullptr};
    void const * ptr_B{nullptr};
    void const * ptr_C1{nullptr};
    void const * ptr_C2{nullptr};
    void * ptr_D{nullptr};

    void * ptr_Vector;
    void * ptr_Tensor;

    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};
    int64_t batch_stride_C1{0};
    int64_t batch_stride_C2{0};
    int64_t batch_stride_D{0};
    int64_t batch_stride_Vector{0};
    int64_t batch_stride_Tensor{0};

    typename LayoutA::Stride::Index lda{};
    typename LayoutB::Stride::Index ldb{};
    typename LayoutC::Stride::Index ldc1{};
    typename LayoutC::Stride::Index ldc2{};
    typename LayoutC::Stride::Index ldd{};
    typename LayoutC::Stride::Index ldr{};
    typename LayoutC::Stride::Index ldt{};

    int avail_sms{-1};          /// The number of SMs that StreamK dispatch heuristics will attempt to load-balance across (-1 defaults to device width, 1 implies classic data-parallel scheduling)


    //
    // Methods
    //

    /// Default Constructor
    Arguments() = default;

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_split,                              /// Either (mode == GemmUniversalMode::kBatched) the batch count, or (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1 defaults to StreamK, >1 emulates Split-K)
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C1,
      void const * ptr_C2,
      void * ptr_D,
      void * ptr_Vector,
      void * ptr_Tensor,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C1,
      int64_t batch_stride_C2,
      int64_t batch_stride_D,
      int64_t batch_stride_Vector,
      int64_t batch_stride_Tensor,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldc1,
      typename LayoutC::Stride::Index ldc2,
      typename LayoutC::Stride::Index ldd,
      typename LayoutC::Stride::Index ldr,
      typename LayoutC::Stride::Index ldt,
      int avail_sms = -1)                           /// The number of SMs that StreamK dispatch heuristics will attempt to load-balance across (-1 defaults to device width, 1 implies classic data-parallel scheduling)
    :
      mode(mode),
      problem_size(problem_size),
      batch_count(batch_split),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C1(ptr_C1), ptr_C2(ptr_C2), ptr_D(ptr_D),
      ptr_Vector(ptr_Vector),
      ptr_Tensor(ptr_Tensor),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C1(batch_stride_C1),
      batch_stride_C2(batch_stride_C2),
      batch_stride_Vector(batch_stride_Vector),
      batch_stride_Tensor(batch_stride_Tensor),
      lda(lda), ldb(ldb), ldc1(ldc1), ldc2(ldc2), ldd(ldd), ldr(ldr), ldt(ldt), avail_sms(avail_sms)
    {
      CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::Arguments::Arguments() - problem_size: " << problem_size);
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << this->ldt);
      CUTLASS_TRACE_HOST("  avail_sms: " << this->avail_sms);
    }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.batch_stride_A, args.batch_stride_B);

      return args;
    }
  };


  /// Parameters structure
  struct Params
  {
  public:

    //
    // Data members
    //

    void * ptr_A{nullptr};
    void * ptr_B{nullptr};

    typename Mma::IteratorA::Params params_A{};
    typename Mma::IteratorB::Params params_B{};

    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};

    GemmUniversalMode mode{GemmUniversalMode::kGemm};

    ThreadblockSwizzle block_mapping{};

    void *barrier_workspace{nullptr};
    void *partials_workspace{nullptr};

    typename EpilogueOutputOp::Params output_op{};

    void * ptr_C1{nullptr};
    void * ptr_C2{nullptr};
    void * ptr_D{nullptr};
    void * ptr_Tensor{nullptr};
    void * ptr_Vector{nullptr};

    typename Epilogue::OutputTileIterator::Params params_C1{};
    typename Epilogue::OutputTileIterator::Params params_C2{};
    typename Epilogue::OutputTileIterator::Params params_D{};
    typename Epilogue::TensorTileIterator::Params params_Tensor{};

    int64_t batch_stride_C1{0};
    int64_t batch_stride_C2{0};
    int64_t batch_stride_D{0};
    int64_t batch_stride_Vector{0};
    int64_t batch_stride_Tensor{0};

    typename LayoutC::Stride::Index ldr{};

  protected:

    //
    // Host-only dispatch-utilities
    //

    /// Pad the given allocation size up to the nearest cache line
    static size_t cacheline_align_up(size_t size)
    {
      static const int CACHELINE_SIZE = 128;
      return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
    }

    /// Get the workspace size needed for barrier
    size_t get_barrier_workspace_size() const
    {
      // For atomic reduction, each SK-block needs a synchronization flag.  For parallel reduction,
      // each reduction block needs its own synchronization flag.
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      int num_flags = fast_max(sk_blocks, block_mapping.reduction_blocks);

      return cacheline_align_up(sizeof(typename Barrier::T) * num_flags);
    }

    /// Get the workspace size needed for intermediate partial sums
    size_t get_partials_workspace_size() const
    {
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      return cacheline_align_up(kWorkspaceBytesPerBlock * sk_blocks);
    }


  public:

    //
    // Host dispatch API
    //

    /// Default constructor
    Params() = default;

    /// Constructor
    Params(
      Arguments const &args,  /// GEMM application arguments
      int device_sms,         /// Number of SMs on the device
      int sm_occupancy)       /// Kernel SM occupancy (in thread blocks)
    :
      params_A(args.lda),
      params_B(args.ldb),
      params_C1(args.ldc1),
      params_C2(args.ldc2),
      params_D(args.ldd),
      params_Tensor(args.ldt),
      output_op(args.epilogue),
      mode(args.mode),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C1(const_cast<void *>(args.ptr_C1)),
      ptr_C2(const_cast<void *>(args.ptr_C2)),
      ptr_D(args.ptr_D),
      ptr_Vector(args.ptr_Vector),
      ldr(args.ldr),
      ptr_Tensor(args.ptr_Tensor),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C1(args.batch_stride_C1),
      batch_stride_C2(args.batch_stride_C2),
      batch_stride_D(args.batch_stride_D),
      batch_stride_Vector(args.batch_stride_Vector),
      batch_stride_Tensor(args.batch_stride_Tensor),
      barrier_workspace(nullptr),
      partials_workspace(nullptr)
    {
      CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::Params::Params()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << args.ldt);

      // Number of SMs to make available for StreamK decomposition
      int avail_sms = (args.avail_sms == -1) ?
                        device_sms :
                        fast_min(args.avail_sms, device_sms);
      CUTLASS_TRACE_HOST("  avail_sms: " << avail_sms);

      // Initialize the block mapping structure
      block_mapping = ThreadblockSwizzle(
        args.mode,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.batch_count,
        sm_occupancy,
        device_sms,
        avail_sms,
        sizeof(ElementA),
        sizeof(ElementB),
        sizeof(ElementC),
        Epilogue::kAccumulatorFragments);
    }

    /// Returns the workspace size (in bytes) needed for these parameters
    size_t get_workspace_size() const
    {
      return
        get_barrier_workspace_size() +
        get_partials_workspace_size();
    }

    /// Assign and initialize the specified workspace buffer.  Assumes
    /// the memory allocated to workspace is at least as large as get_workspace_size().
    Status init_workspace(
      void *workspace,
      cudaStream_t stream = nullptr)
    {
      uint8_t *ptr = static_cast<uint8_t*>(workspace);

      // Establish partials workspace
      partials_workspace = nullptr;
      size_t partials_workspace_bytes = get_partials_workspace_size();
      if (partials_workspace_bytes > 0)
      {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        partials_workspace = ptr;
        ptr += partials_workspace_bytes;
      }

      // Establish barrier workspace
      barrier_workspace = nullptr;
      size_t barrier_workspace_bytes = get_barrier_workspace_size();
      if (barrier_workspace_bytes > 0)
      {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        barrier_workspace = ptr;
        ptr += barrier_workspace_bytes;
      }

      // Zero-initialize barrier workspace
      if (barrier_workspace)
      {
        size_t barrier_workspace_bytes = get_barrier_workspace_size();

        CUTLASS_TRACE_HOST("  Initialize " << barrier_workspace_bytes << " barrier bytes");

        cudaError_t result = cudaMemsetAsync(
          barrier_workspace,
          0,
          barrier_workspace_bytes,
          stream);

        if (result != cudaSuccess) {
          CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }

      return Status::kSuccess;
    }


    /// Returns the GEMM volume in thread block tiles
    cutlass::gemm::GemmCoord get_tiled_shape() const
    {
      return block_mapping.tiled_shape();
    }

    /// Returns the total number of thread blocks to launch
    int get_grid_blocks() const
    {
      dim3 grid_dims = get_grid_dims();
      return grid_dims.x * grid_dims.y * grid_dims.z;
    }

    /// Returns the grid extents in thread blocks to launch
    dim3 get_grid_dims() const
    {
      return block_mapping.get_grid_dims();
    }

    /// Lightweight update given a subset of arguments.  Problem geometry is assumed
    /// to remain the same.
    CUTLASS_HOST_DEVICE
    void update(Arguments const &args)
    {
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C1 = const_cast<void *>(args.ptr_C1);
      ptr_C2 = const_cast<void *>(args.ptr_C2);
      ptr_D = args.ptr_D;

      ptr_Vector = args.ptr_Vector;
      ldr = args.ldr;
      ptr_Tensor = args.ptr_Tensor;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C1 = args.batch_stride_C1;
      batch_stride_C2 = args.batch_stride_C2;
      batch_stride_D = args.batch_stride_D;
      batch_stride_Vector = args.batch_stride_Vector;
      batch_stride_Tensor = args.batch_stride_Tensor;

      output_op = args.epilogue;

      CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::Params::update()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
    }
  };

  /// Tile work descriptor
  struct TileWorkDesc
  {
    /// The linear tile index
    int tile_idx;

    /// The location of this tile (in threadblock-tile coordinates) in the output matrix
    cutlass::gemm::GemmCoord tiled_coord;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    int iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_begin;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_end;

    /// The number of remaining MAC-iterations this threadblock will perform for this tile
    int k_iters_remaining;

    // Whether this block will perform the first iteration of this tile
    CUTLASS_DEVICE
    bool tile_started()
    {
      return (k_begin == 0);
    }

    // Whether this block will perform the last iteration of this tile
    CUTLASS_DEVICE
    bool tile_finished(Params const &params)
    {
      return (k_end == params.block_mapping.problem_size.k());
    }
  };


  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };


protected:

  //
  // Data members
  //

  /// GEMM problem parameters
  Params const &params;

  /// Shared storage reference
  SharedStorage &shared_storage;

  /// ID within the threadblock
  int thread_idx;

  /// ID of warp
  int warp_idx;

  /// ID of each thread within a warp
  int lane_idx;

  /// Threadblock scoped epilogue
  Epilogue epilogue;


public:

  //
  // Host dispatch API
  //

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value
            || platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

protected:

  //
  // Device-only utility methods
  //

  /// Iterator for fetching tile fragments from A
  CUTLASS_DEVICE
  typename Mma::IteratorA init_iterator_A(
    TileWorkDesc &tile_work,
    GemmUniversalMode mode)
  {
    // The input A matrix
    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);

    // Update input pointers based on batched/array mode
    if (mode == GemmUniversalMode::kBatched) {
      ptr_A += tile_work.tiled_coord.k() * params.batch_stride_A;
    }
    if (mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[tile_work.tiled_coord.k()];
    }

    int m_begin = tile_work.tiled_coord.m() * Mma::Shape::kM;
    int m_end = params.block_mapping.problem_size.m();
    return Mma::IteratorA(
        params.params_A,
        ptr_A,
        { m_end, tile_work.k_end },
        threadIdx.x,
        { m_begin, tile_work.k_begin });

  }


  /// Iterator for fetching tile fragments from B
  CUTLASS_DEVICE
  typename Mma::IteratorB init_iterator_B(
    TileWorkDesc &tile_work,
    GemmUniversalMode mode)
  {
    // The input B matrix
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    // Update input pointers based on batched/array mode
    if (mode == GemmUniversalMode::kBatched) {
      ptr_B += tile_work.tiled_coord.k() * params.batch_stride_B;
    }
    if (mode == GemmUniversalMode::kArray) {
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[tile_work.tiled_coord.k()];
    }

    int n_begin = tile_work.tiled_coord.n() * Mma::Shape::kN;
    int n_end = params.block_mapping.problem_size.n();
    return Mma::IteratorB(
        params.params_B,
        ptr_B,
        { tile_work.k_end, n_end },
        threadIdx.x,
        { tile_work.k_begin, n_begin });
  }


  CUTLASS_DEVICE
  void init_dp_tile_work(
      TileWorkDesc &tile_work,
      int tile_idx)
  {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    tile_work.iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = params.block_mapping.iters_per_tile();

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_begin = 0;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_end = params.block_mapping.problem_size.k();

    // The location of this tile (in threadblock-tile coordinates) in the output matrix
    tile_work.tiled_coord = params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }


  CUTLASS_DEVICE
  void init_sk_tile_work(
      TileWorkDesc &tile_work,
      int tile_idx,
      int block_iter_begin,
      int block_iter_end)
  {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration for this tile
    int tile_iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    tile_work.iter_begin = max(block_iter_begin, tile_iter_begin);

    // The first tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_begin = tile_work.iter_begin - tile_iter_begin;

    // The last (one past) tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_end = block_iter_end - tile_iter_begin;

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = k_iter_end - k_iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_begin = k_iter_begin * Mma::Shape::kK;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_end = min(
        params.block_mapping.problem_size.k(),            // extent of k domain
        (k_iter_end * Mma::Shape::kK));                   // extent of the threadblock's global iteration assignment

    // The location of this tile (in threadblock-tile coordinates) in the output matrix
    tile_work.tiled_coord = params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }


  /// Share accumulators with peers
  CUTLASS_DEVICE
  void share_accumulators(
    AccumulatorTile const &accumulator_tile,
    int block_idx,
    int first_block_idx)
  {
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    int accum_tile_offset = first_block_idx * kThreadCount;

    if (block_idx == first_block_idx)
    {
      // First peer initializes the workspace partials
      BlockStripedReduceT::store(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }
    else
    {
      // Subsequent peers atomically accumulate into the workspace partials
      if (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic)
      {
        // Non-deterministic reduction order: wait for the first peer to have initialized the partials before we add to them
        Barrier::wait_lt(params.barrier_workspace, thread_idx, first_block_idx, 1);
      }
      else
      {
        // Turnstile reduction order: wait until the previous peer has written
        int wait_count = block_idx - first_block_idx;
        Barrier::wait_eq(params.barrier_workspace, thread_idx, first_block_idx, wait_count);
      }

      // Perform reduction in workspace
      BlockStripedReduceT::reduce(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }

    // Signal our arrival
    Barrier::arrive_inc(params.barrier_workspace, thread_idx, first_block_idx);
  }


  /// Acquire accumulators from peers
  CUTLASS_DEVICE
  void acquire_accumulators(
    AccumulatorTile &accumulator_tile,
    int block_idx,
    int first_block_idx)
  {
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    // Wait for arrival
    int num_carry_in = block_idx - first_block_idx;
    Barrier::wait_eq_reset(params.barrier_workspace, thread_idx, first_block_idx, num_carry_in);

    // Load and add peer-partials accumulator tile to local accumulator tile
    int accum_tile_offset = first_block_idx * kThreadCount;
    BlockStripedReduceT::load_add(accumulator_tile, accum_tile_workspace + accum_tile_offset, thread_idx);
  }


  /// Perform epilogue computations and output
  CUTLASS_DEVICE
  void do_epilogue(
    TileWorkDesc &tile_work,
    AccumulatorTile &accumulator_tile)
  {
    ElementC *ptr_C1 = static_cast<ElementC *>(params.ptr_C1);
    ElementC *ptr_C2 = static_cast<ElementC *>(params.ptr_C2);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);
    typename Epilogue::ElementTensor *ptr_Tensor = static_cast<typename Epilogue::ElementTensor *>(params.ptr_Tensor);

    // Define the reduction output pointer and move to the appropriate place
    typename Epilogue::ElementVector *ptr_Vector =
      static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

    // Update pointers for batched/array mode(s)
    if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C1 += tile_work.tiled_coord.k() * params.batch_stride_C1;
      if (ptr_C2) {
        ptr_C2 += tile_work.tiled_coord.k() * params.batch_stride_C2;
      }
      ptr_D += tile_work.tiled_coord.k() * params.batch_stride_D;
      if (ptr_Tensor) {
        ptr_Tensor = ReferenceFactory<typename Epilogue::ElementTensor>::add_pointer_offset(
          ptr_Tensor,
          tile_work.tiled_coord.k() * params.batch_stride_Tensor);
      }
      if (ptr_Vector) {
        ptr_Vector += tile_work.tiled_coord.k() * params.batch_stride_Vector;
      }
    }
    if (params.mode == GemmUniversalMode::kArray) {
      ptr_C1 = static_cast<ElementC * const *>(params.ptr_C1)[tile_work.tiled_coord.k()];
      if (ptr_C2) {
        ptr_C2 = static_cast<ElementC * const *>(params.ptr_C2)[tile_work.tiled_coord.k()];
      }
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[tile_work.tiled_coord.k()];
      if (ptr_Tensor) {
        ptr_Tensor = static_cast<typename Epilogue::ElementTensor * const *>(params.ptr_Tensor)[tile_work.tiled_coord.k()];
      }
      if (ptr_Vector) {
        ptr_Vector = static_cast<typename Epilogue::ElementVector * const *>(params.ptr_Vector)[tile_work.tiled_coord.k()];
      }
    }

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
      tile_work.tiled_coord.m() * Mma::Shape::kM,
      tile_work.tiled_coord.n() * Mma::Shape::kN
    );

    // Tile iterator loading from residual1.
    typename Epilogue::OutputTileIterator iterator_C1(
        params.params_C1,
        ptr_C1,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator loading from residual2.
    typename Epilogue::OutputTileIterator iterator_C2(
        params.params_C2,
        ptr_C2,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Additional tensor to load from
    typename Epilogue::TensorTileIterator tensor_iterator(
        params.params_Tensor,
        ptr_Tensor,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_item_begin.column() + tile_work.tiled_coord.m() * params.ldr;
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(
        EpilogueOutputOp(params.output_op),
        ptr_Vector,
        iterator_D,
        accumulator_tile,
        iterator_C1,
        iterator_C2,
        tensor_iterator,
        params.block_mapping.problem_size.mn(),
        threadblock_item_begin);
  }


  CUTLASS_DEVICE
  void separate_reduction(int reduce_idx)
  {
    int peer_idx_begin, peer_idx_last, reduce_tile_idx, reduce_fragment_idx;

    // Reduce by sk-tile (every tile contributed to by one or more blocks)
    reduce_tile_idx = reduce_idx / Epilogue::kAccumulatorFragments;
    reduce_fragment_idx = reduce_idx % Epilogue::kAccumulatorFragments;

    int iter_tile_first = reduce_tile_idx * params.block_mapping.iters_per_tile();
    int iter_tile_last = iter_tile_first + params.block_mapping.iters_per_tile() - 1;

    peer_idx_begin = params.block_mapping.get_sk_block_idx(iter_tile_first);
    peer_idx_last = params.block_mapping.get_sk_block_idx(iter_tile_last);

    // Wait for peers to complete
    int peer_idx_end = peer_idx_last + 1;
    int num_peers = peer_idx_end - peer_idx_begin;
    Barrier::wait_eq_reset(
        params.barrier_workspace,
        thread_idx,
        (reduce_tile_idx * Epilogue::kAccumulatorFragments) + reduce_fragment_idx,
        num_peers);

    /// The location of this tile (in threadblock-tile coordinates) in the output matrix
    GemmCoord tiled_coord = params.block_mapping.get_tile_offset(reduce_tile_idx);

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
      tiled_coord.m() * Mma::Shape::kM,
      tiled_coord.n() * Mma::Shape::kN
    );

    ElementC *ptr_C1 = static_cast<ElementC *>(params.ptr_C1);
    ElementC *ptr_C2 = static_cast<ElementC *>(params.ptr_C2);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);
    typename Epilogue::ElementTensor *ptr_Tensor = static_cast<typename Epilogue::ElementTensor *>(params.ptr_Tensor);

    // Define the reduction output pointer and move to the appropriate place
    typename Epilogue::ElementVector *ptr_Vector =
      static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

    // Tile iterator loading from residual1.
    typename Epilogue::OutputTileIterator iterator_C1(
        params.params_C1,
        ptr_C1,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator loading from residual2.
    typename Epilogue::OutputTileIterator iterator_C2(
        params.params_C2,
        ptr_C2,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Additional tensor to load from
    typename Epilogue::TensorTileIterator tensor_iterator(
        params.params_Tensor,
        ptr_Tensor,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_item_begin.column() + tiled_coord.m() * params.ldr;
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue.reduce(
        peer_idx_begin,
        peer_idx_end,
        reduce_fragment_idx,
        params.partials_workspace,
        EpilogueOutputOp(params.output_op),
        ptr_Vector,
        iterator_D,
        iterator_C1,
        iterator_C2,
        tensor_iterator,
        params.block_mapping.problem_size.mn(),
        threadblock_item_begin);
  }


  CUTLASS_DEVICE
  void process_tile(
    TileWorkDesc tile_work,
    int block_idx,
    int dp_start_block_idx,
    int block_iter_begin)
  {
    // Initialize input iterators
    typename Mma::IteratorA iterator_A = init_iterator_A(tile_work, params.mode);
    typename Mma::IteratorB iterator_B = init_iterator_B(tile_work, params.mode);

    // Initialize accumulators
    AccumulatorTile accumulator_tile;
    accumulator_tile.clear();

    // Initialize MMA abstraction
    Mma mma(
      shared_storage.main_loop,
      thread_idx,
      warp_idx,
      lane_idx);

    // Perform this tile's range of multiply-accumulate (MAC) iterations
    mma(tile_work.k_iters_remaining, accumulator_tile, iterator_A, iterator_B, accumulator_tile);

    if ((ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic) ||
        (params.block_mapping.reduction_blocks == 0) ||
        (block_idx >= dp_start_block_idx))
    {
      //
      // Cooperative SK peer reduction or DP block
      //

      int first_block_idx = params.block_mapping.get_first_block_idx(tile_work.tile_idx, block_idx);

      if (!tile_work.tile_finished(params)) {
        // Non "finishing" SK blocks must share their partial accumulator sums through global scratch workspace
        share_accumulators(accumulator_tile, block_idx, first_block_idx);
      }
      else
      {
        // DP blocks and "finishing" SK blocks must perform epilogue operations and write the output tile
        if (!tile_work.tile_started())
        {
          // A "finishing" SK block must first aggregate its accumulator partial sums with those shared by peer threadblocks
          acquire_accumulators(accumulator_tile, block_idx, first_block_idx);
        }

        do_epilogue(tile_work, accumulator_tile);
      }
    }
    else
    {
      //
      // Separate peer reduction
      //

      // Share accumulator partial sums with peer threadblock(s) through scratch workspace
      epilogue.share(block_idx, params.partials_workspace, accumulator_tile, tile_work.tile_started());

      // Signal arrival
      Barrier::arrive_range_inc(
        params.barrier_workspace,
        thread_idx,
        tile_work.tile_idx * Epilogue::kAccumulatorFragments,
        Epilogue::kAccumulatorFragments);
    }
  }


  /// Executes one GEMM
  CUTLASS_DEVICE
  void gemm()
  {
    // Initialize block's iteration range
    int tile_idx = 0;
    int block_iter_begin = 0;
    int block_iters_remaining = 0;

    int block_idx = params.block_mapping.get_block_idx();

    int sk_padding_start_block_idx =  params.block_mapping.sk_regions() * params.block_mapping.sk_blocks_per_region();
    int dp_start_block_idx = params.block_mapping.sk_waves * params.block_mapping.avail_sms;
    int reduce_start_block_idx = dp_start_block_idx + params.block_mapping.dp_blocks;
    int grid_padding_start_block_idx = reduce_start_block_idx + params.block_mapping.reduction_blocks;

    // Initialize tile work descriptor
    TileWorkDesc tile_work;

    bool dp_block = (block_idx >= dp_start_block_idx) && (block_idx < reduce_start_block_idx);
    bool sk_block = (block_idx < sk_padding_start_block_idx);
    bool reduce_block = (block_idx >= reduce_start_block_idx) &&
            (block_idx < grid_padding_start_block_idx) &&
            (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kMixed);

    if (dp_block)
    {
      // This is a DP block
      int dp_block_idx = block_idx - dp_start_block_idx;
      int first_dp_tile = (params.block_mapping.cohort_raster) ? 0 : params.block_mapping.sk_tiles;

      // Blocks in first DP wave get configured number of tiles
      tile_idx = first_dp_tile + dp_block_idx;
      int tile_allottment = params.block_mapping.dp_first_wave_tiles;

      // Blocks in subsequent DP waves get 1 tile
      if (dp_block_idx >= params.block_mapping.avail_sms) {
          tile_allottment = 1;
          tile_idx += (params.block_mapping.dp_first_wave_tiles - 1) * params.block_mapping.avail_sms;
      }

      block_iters_remaining = params.block_mapping.iters_per_tile() * tile_allottment;

      init_dp_tile_work(tile_work, tile_idx);

      // DP blocks exit if out of bounds or overlap an SK tile (only possible during cohort rasterization, where dp_first_wave_tiles must be 1)
      if ((tile_idx < params.block_mapping.sk_tiles) ||
          (tile_work.tiled_coord.m() >= params.block_mapping.tiled_shape().m()) ||
          (tile_work.tiled_coord.n() >= params.block_mapping.tiled_shape().n()))
      {
        return;
      }
    }
    else if (sk_block)
    {
      // This is a SK block
      int block_iter_end;
      params.block_mapping.get_iter_extents(block_idx, block_iter_begin, block_iter_end);
      block_iters_remaining = block_iter_end - block_iter_begin;

      tile_idx = params.block_mapping.get_sk_tile_idx(block_iter_end - 1);
      init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
    }
    else
    {
      if (reduce_block)
      {
        // This is a reduction threadblock
        int reduce_block_idx = block_idx - reduce_start_block_idx;
        separate_reduction(reduce_block_idx);
      }

      return;
    }

    // Iteration-processing loop body
    CUTLASS_PRAGMA_NO_UNROLL
    while (true)
    {
      // Perform this block's share of work for this tile
      process_tile(
        tile_work,
        block_idx,
        dp_start_block_idx,
        block_iter_begin);

      block_iters_remaining -= tile_work.k_iters_remaining;

      if (block_iters_remaining == 0)
      {
        break;
      }

      // Continue to next tile
      __syncthreads();

      if (block_idx >= dp_start_block_idx)
      {
        // DP block consume their tiles at stride
        tile_idx += params.block_mapping.avail_sms;
        init_dp_tile_work(tile_work, tile_idx);
      }
      else
      {
        // SK blocks consume their tiles in backwards order
        tile_idx--;
        init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
      }
    }

  }


public:

  //
  // Device-only API
  //

  // Factory invocation
  CUTLASS_DEVICE
  static void invoke(
    Params const &params,
    SharedStorage &shared_storage)
  {
    GemmStreamkWithFusedEpilogue op(params, shared_storage);
    op();
  }


  // Constructor
  CUTLASS_DEVICE
  GemmStreamkWithFusedEpilogue(
      Params const &params,
      SharedStorage &shared_storage)
    :
      params(params),
      shared_storage(shared_storage),
      thread_idx(threadIdx.x),
      warp_idx(__shfl_sync(0xffffffff, threadIdx.x / 32, 0)),   // broadcast the warp_id computed by lane 0 to ensure dependent code
      lane_idx(threadIdx.x % 32),
      epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx)
  {}

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()() {
    // Generic SK code path
    gemm();

  }
};


// GemmStreamkWithFusedEpilogue with one source
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmStreamkWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, true> {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;

  /// The per-thread tile of raw accumulators
  using AccumulatorTile = typename Mma::FragmentC;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Workspace bytes per thread block
  static size_t const kWorkspaceBytesPerBlock =
    __NV_STD_MAX(
      kThreadCount * sizeof(AccumulatorTile),
      Epilogue::kWorkspaceBytesPerBlock);

  /// Block-striped reduction utility
  using BlockStripedReduceT = BlockStripedReduce<kThreadCount, AccumulatorTile>;



  //
  // Structures
  //

  /// Argument structure
  struct Arguments
  {

    //
    // Data members
    //

    GemmUniversalMode mode{GemmUniversalMode::kGemm};
    GemmCoord problem_size{};
    int batch_count{1};        // Either (mode == GemmUniversalMode::kBatched) the batch count, or (mode == GemmUniversalMode::kGemm) the tile-splitting factor

    typename EpilogueOutputOp::Params epilogue{};

    void const * ptr_A{nullptr};
    void const * ptr_B{nullptr};
    void const * ptr_C{nullptr};
    void * ptr_D{nullptr};

    void * ptr_Vector{nullptr};
    void * ptr_Tensor{nullptr};

    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};
    int64_t batch_stride_C{0};
    int64_t batch_stride_D{0};
    int64_t batch_stride_Vector{0};
    int64_t batch_stride_Tensor{0};

    typename LayoutA::Stride::Index lda{};
    typename LayoutB::Stride::Index ldb{};
    typename LayoutC::Stride::Index ldc{};
    typename LayoutC::Stride::Index ldd{};
    typename LayoutC::Stride::Index ldr{};
    typename LayoutC::Stride::Index ldt{};

    int avail_sms{-1};          /// The number of SMs that StreamK dispatch heuristics will attempt to load-balance across (-1 defaults to device width, 1 implies classic data-parallel scheduling)


    //
    // Methods
    //

    /// Default Constructor
    Arguments() = default;

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_split,                              /// Either (mode == GemmUniversalMode::kBatched) the batch count, or (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1 defaults to StreamK, >1 emulates Split-K)
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      void * ptr_Vector,
      void * ptr_Tensor,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      int64_t batch_stride_Vector,
      int64_t batch_stride_Tensor,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldc,
      typename LayoutC::Stride::Index ldd,
      typename LayoutC::Stride::Index ldr,
      typename LayoutC::Stride::Index ldt,
      int avail_sms = -1)                           /// The number of SMs that StreamK dispatch heuristics will attempt to load-balance across (-1 defaults to device width, 1 implies classic data-parallel scheduling)
    :
      mode(mode),
      problem_size(problem_size),
      batch_count(batch_split),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      ptr_Vector(ptr_Vector),
      ptr_Tensor(ptr_Tensor),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_Vector(batch_stride_Vector),
      batch_stride_Tensor(batch_stride_Tensor),
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd), ldr(ldr), ldt(ldt), avail_sms(avail_sms)
    {
      CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::Arguments::Arguments() - problem_size: " << problem_size);
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << this->ldt);
      CUTLASS_TRACE_HOST("  avail_sms: " << this->avail_sms);
    }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.batch_stride_A, args.batch_stride_B);

      return args;
    }
  };


  /// Parameters structure
  struct Params
  {

  public:

    //
    // Data members
    //

    void * ptr_A{nullptr};
    void * ptr_B{nullptr};

    typename Mma::IteratorA::Params params_A{};
    typename Mma::IteratorB::Params params_B{};

    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};

    GemmUniversalMode mode{GemmUniversalMode::kGemm};

    ThreadblockSwizzle block_mapping{};

    void *barrier_workspace{nullptr};
    void *partials_workspace{nullptr};

    typename EpilogueOutputOp::Params output_op{};

    void * ptr_C{nullptr};
    void * ptr_D{nullptr};
    void * ptr_Tensor{nullptr};
    void * ptr_Vector{nullptr};

    typename Epilogue::OutputTileIterator::Params params_C{};
    typename Epilogue::OutputTileIterator::Params params_D{};
    typename Epilogue::TensorTileIterator::Params params_Tensor{};

    int64_t batch_stride_C{0};
    int64_t batch_stride_D{0};
    int64_t batch_stride_Vector{0};
    int64_t batch_stride_Tensor{0};

    typename LayoutC::Stride::Index ldr{};

  protected:

    //
    // Host-only dispatch-utilities
    //

    /// Pad the given allocation size up to the nearest cache line
    static size_t cacheline_align_up(size_t size)
    {
      static const int CACHELINE_SIZE = 128;
      return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
    }

    /// Get the workspace size needed for barrier
    size_t get_barrier_workspace_size() const
    {
      // For atomic reduction, each SK-block needs a synchronization flag.  For parallel reduction,
      // each reduction block needs its own synchronization flag.
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      int num_flags = fast_max(sk_blocks, block_mapping.reduction_blocks);

      return cacheline_align_up(sizeof(typename Barrier::T) * num_flags);
    }

    /// Get the workspace size needed for intermediate partial sums
    size_t get_partials_workspace_size() const
    {
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      return cacheline_align_up(kWorkspaceBytesPerBlock * sk_blocks);
    }


  public:
    //
    // Host dispatch API
    //

    /// Default constructor
    Params() = default;

    /// Constructor
    Params(
      Arguments const &args,  /// GEMM application arguments
      int device_sms,         /// Number of SMs on the device
      int sm_occupancy)       /// Kernel SM occupancy (in thread blocks)
    :
      params_A(args.lda),
      params_B(args.ldb),
      params_C(args.ldc),
      params_D(args.ldd),
      params_Tensor(args.ldt),
      output_op(args.epilogue),
      mode(args.mode),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      ptr_Vector(args.ptr_Vector),
      ldr(args.ldr),
      ptr_Tensor(args.ptr_Tensor),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      batch_stride_D(args.batch_stride_D),
      batch_stride_Vector(args.batch_stride_Vector),
      batch_stride_Tensor(args.batch_stride_Tensor),
      barrier_workspace(nullptr),
      partials_workspace(nullptr)
    {
      CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::Params::Params()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << args.ldt);

      // Number of SMs to make available for StreamK decomposition
      int avail_sms = (args.avail_sms == -1) ?
                        device_sms :
                        fast_min(args.avail_sms, device_sms);
      CUTLASS_TRACE_HOST("  avail_sms: " << avail_sms);

      // Initialize the block mapping structure
      block_mapping = ThreadblockSwizzle(
        args.mode,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.batch_count,
        sm_occupancy,
        device_sms,
        avail_sms,
        sizeof(ElementA),
        sizeof(ElementB),
        sizeof(ElementC),
        Epilogue::kAccumulatorFragments);
    }

    /// Returns the workspace size (in bytes) needed for these parameters
    size_t get_workspace_size() const
    {
      return
        get_barrier_workspace_size() +
        get_partials_workspace_size();
    }


    /// Assign and initialize the specified workspace buffer.  Assumes
    /// the memory allocated to workspace is at least as large as get_workspace_size().
    Status init_workspace(
      void *workspace,
      cudaStream_t stream = nullptr)
    {
      uint8_t *ptr = static_cast<uint8_t*>(workspace);

      // Establish partials workspace
      partials_workspace = nullptr;
      size_t partials_workspace_bytes = get_partials_workspace_size();
      if (partials_workspace_bytes > 0)
      {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        partials_workspace = ptr;
        ptr += partials_workspace_bytes;
      }

      // Establish barrier workspace
      barrier_workspace = nullptr;
      size_t barrier_workspace_bytes = get_barrier_workspace_size();
      if (barrier_workspace_bytes > 0)
      {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        barrier_workspace = ptr;
        ptr += barrier_workspace_bytes;
      }

      // Zero-initialize barrier workspace
      if (barrier_workspace)
      {
        size_t barrier_workspace_bytes = get_barrier_workspace_size();

        CUTLASS_TRACE_HOST("  Initialize " << barrier_workspace_bytes << " barrier bytes");

        cudaError_t result = cudaMemsetAsync(
          barrier_workspace,
          0,
          barrier_workspace_bytes,
          stream);

        if (result != cudaSuccess) {
          CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }

      return Status::kSuccess;
    }


    /// Returns the GEMM volume in thread block tiles
    cutlass::gemm::GemmCoord get_tiled_shape() const
    {
      return block_mapping.tiled_shape();
    }


    /// Returns the total number of thread blocks to launch
    int get_grid_blocks() const
    {
      dim3 grid_dims = get_grid_dims();
      return grid_dims.x * grid_dims.y * grid_dims.z;
    }


    /// Returns the grid extents in thread blocks to launch
    dim3 get_grid_dims() const
    {
      return block_mapping.get_grid_dims();
    }

    /// Lightweight update given a subset of arguments.  Problem geometry is assumed
    /// to remain the same.
    CUTLASS_HOST_DEVICE
    void update(Arguments const &args)
    {
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;

      ptr_Vector = args.ptr_Vector;
      ldr = args.ldr;
      ptr_Tensor = args.ptr_Tensor;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_D = args.batch_stride_D;
      batch_stride_Vector = args.batch_stride_Vector;
      batch_stride_Tensor = args.batch_stride_Tensor;

      output_op = args.epilogue;

      CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::Params::update()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
    }
  };

  /// Tile work descriptor
  struct TileWorkDesc
  {
    /// The linear tile index
    int tile_idx;

    /// The location of this tile (in threadblock-tile coordinates) in the output matrix
    cutlass::gemm::GemmCoord tiled_coord;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    int iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_begin;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_end;

    /// The number of remaining MAC-iterations this threadblock will perform for this tile
    int k_iters_remaining;

    // Whether this block will perform the first iteration of this tile
    CUTLASS_DEVICE
    bool tile_started()
    {
      return (k_begin == 0);
    }

    // Whether this block will perform the last iteration of this tile
    CUTLASS_DEVICE
    bool tile_finished(Params const &params)
    {
      return (k_end == params.block_mapping.problem_size.k());
    }
  };


  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };


protected:

  //
  // Data members
  //

  /// GEMM problem parameters
  Params const &params;

  /// Shared storage reference
  SharedStorage &shared_storage;

  /// ID within the threadblock
  int thread_idx;

  /// ID of warp
  int warp_idx;

  /// ID of each thread within a warp
  int lane_idx;

  /// Threadblock scoped epilogue
  Epilogue epilogue;


public:

  //
  // Host dispatch API
  //

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmStreamkWithFusedEpilogue::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value
            || platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

protected:

  //
  // Device-only utility methods
  //

  /// Iterator for fetching tile fragments from A
  CUTLASS_DEVICE
  typename Mma::IteratorA init_iterator_A(
    TileWorkDesc &tile_work,
    GemmUniversalMode mode)
  {
    // The input A matrix
    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);

    // Update input pointers based on batched/array mode
    if (mode == GemmUniversalMode::kBatched) {
      ptr_A += tile_work.tiled_coord.k() * params.batch_stride_A;
    }
    if (mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[tile_work.tiled_coord.k()];
    }

    int m_begin = tile_work.tiled_coord.m() * Mma::Shape::kM;
    int m_end = params.block_mapping.problem_size.m();
    return Mma::IteratorA(
        params.params_A,
        ptr_A,
        { m_end, tile_work.k_end },
        threadIdx.x,
        { m_begin, tile_work.k_begin });

  }


  /// Iterator for fetching tile fragments from B
  CUTLASS_DEVICE
  typename Mma::IteratorB init_iterator_B(
    TileWorkDesc &tile_work,
    GemmUniversalMode mode)
  {
    // The input B matrix
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    // Update input pointers based on batched/array mode
    if (mode == GemmUniversalMode::kBatched) {
      ptr_B += tile_work.tiled_coord.k() * params.batch_stride_B;
    }
    if (mode == GemmUniversalMode::kArray) {
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[tile_work.tiled_coord.k()];
    }

    int n_begin = tile_work.tiled_coord.n() * Mma::Shape::kN;
    int n_end = params.block_mapping.problem_size.n();
    return Mma::IteratorB(
        params.params_B,
        ptr_B,
        { tile_work.k_end, n_end },
        threadIdx.x,
        { tile_work.k_begin, n_begin });
  }


  CUTLASS_DEVICE
  void init_dp_tile_work(
      TileWorkDesc &tile_work,
      int tile_idx)
  {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    tile_work.iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = params.block_mapping.iters_per_tile();

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_begin = 0;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_end = params.block_mapping.problem_size.k();

    // The location of this tile (in threadblock-tile coordinates) in the output matrix
    tile_work.tiled_coord = params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }


  CUTLASS_DEVICE
  void init_sk_tile_work(
      TileWorkDesc &tile_work,
      int tile_idx,
      int block_iter_begin,
      int block_iter_end)
  {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration for this tile
    int tile_iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    tile_work.iter_begin = max(block_iter_begin, tile_iter_begin);

    // The first tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_begin = tile_work.iter_begin - tile_iter_begin;

    // The last (one past) tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_end = block_iter_end - tile_iter_begin;

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = k_iter_end - k_iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_begin = k_iter_begin * Mma::Shape::kK;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_end = min(
        params.block_mapping.problem_size.k(),            // extent of k domain
        (k_iter_end * Mma::Shape::kK));                   // extent of the threadblock's global iteration assignment

    // The location of this tile (in threadblock-tile coordinates) in the output matrix
    tile_work.tiled_coord = params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }


  /// Share accumulators with peers
  CUTLASS_DEVICE
  void share_accumulators(
    AccumulatorTile const &accumulator_tile,
    int block_idx,
    int first_block_idx)
  {
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    int accum_tile_offset = first_block_idx * kThreadCount;

    if (block_idx == first_block_idx)
    {
      // First peer initializes the workspace partials
      BlockStripedReduceT::store(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }
    else
    {
      // Subsequent peers atomically accumulate into the workspace partials
      if (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic)
      {
        // Non-deterministic reduction order: wait for the first peer to have initialized the partials before we add to them
        Barrier::wait_lt(params.barrier_workspace, thread_idx, first_block_idx, 1);
      }
      else
      {
        // Turnstile reduction order: wait until the previous peer has written
        int wait_count = block_idx - first_block_idx;
        Barrier::wait_eq(params.barrier_workspace, thread_idx, first_block_idx, wait_count);
      }

      // Perform reduction in workspace
      BlockStripedReduceT::reduce(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }

    // Signal our arrival
    Barrier::arrive_inc(params.barrier_workspace, thread_idx, first_block_idx);
  }


  /// Acquire accumulators from peers
  CUTLASS_DEVICE
  void acquire_accumulators(
    AccumulatorTile &accumulator_tile,
    int block_idx,
    int first_block_idx)
  {
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    // Wait for arrival
    int num_carry_in = block_idx - first_block_idx;
    Barrier::wait_eq_reset(params.barrier_workspace, thread_idx, first_block_idx, num_carry_in);

    // Load and add peer-partials accumulator tile to local accumulator tile
    int accum_tile_offset = first_block_idx * kThreadCount;
    BlockStripedReduceT::load_add(accumulator_tile, accum_tile_workspace + accum_tile_offset, thread_idx);
  }


  /// Perform epilogue computations and output
  CUTLASS_DEVICE
  void do_epilogue(
    TileWorkDesc &tile_work,
    AccumulatorTile &accumulator_tile)
  {
    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);
    typename Epilogue::ElementTensor *ptr_Tensor = static_cast<typename Epilogue::ElementTensor *>(params.ptr_Tensor);

    // Define the reduction output pointer and move to the appropriate place
    typename Epilogue::ElementVector *ptr_Vector =
      static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

    // Update pointers for batched/array mode(s)
    if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C += tile_work.tiled_coord.k() * params.batch_stride_C;
      ptr_D += tile_work.tiled_coord.k() * params.batch_stride_D;
      if (ptr_Tensor) {
        ptr_Tensor = ReferenceFactory<typename Epilogue::ElementTensor>::add_pointer_offset(
          ptr_Tensor,
          tile_work.tiled_coord.k() * params.batch_stride_Tensor);
      }
      if (ptr_Vector) {
        ptr_Vector += tile_work.tiled_coord.k() * params.batch_stride_Vector;
      }
    }
    if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[tile_work.tiled_coord.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[tile_work.tiled_coord.k()];
      if (ptr_Tensor) {
        ptr_Tensor = static_cast<typename Epilogue::ElementTensor * const *>(params.ptr_Tensor)[tile_work.tiled_coord.k()];
      }
      if (ptr_Vector) {
        ptr_Vector = static_cast<typename Epilogue::ElementVector * const *>(params.ptr_Vector)[tile_work.tiled_coord.k()];
      }
    }

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
      tile_work.tiled_coord.m() * Mma::Shape::kM,
      tile_work.tiled_coord.n() * Mma::Shape::kN
    );

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Additional tensor to load from
    typename Epilogue::TensorTileIterator tensor_iterator(
        params.params_Tensor,
        ptr_Tensor,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_item_begin.column() + tile_work.tiled_coord.m() * params.ldr;
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(
        EpilogueOutputOp(params.output_op),
        ptr_Vector,
        iterator_D,
        accumulator_tile,
        iterator_C,
        tensor_iterator,
        params.block_mapping.problem_size.mn(),
        threadblock_item_begin);
  }


  CUTLASS_DEVICE
  void separate_reduction(int reduce_idx)
  {
    int peer_idx_begin, peer_idx_last, reduce_tile_idx, reduce_fragment_idx;

    // Reduce by sk-tile (every tile contributed to by one or more blocks)
    reduce_tile_idx = reduce_idx / Epilogue::kAccumulatorFragments;
    reduce_fragment_idx = reduce_idx % Epilogue::kAccumulatorFragments;

    int iter_tile_first = reduce_tile_idx * params.block_mapping.iters_per_tile();
    int iter_tile_last = iter_tile_first + params.block_mapping.iters_per_tile() - 1;

    peer_idx_begin = params.block_mapping.get_sk_block_idx(iter_tile_first);
    peer_idx_last = params.block_mapping.get_sk_block_idx(iter_tile_last);

    // Wait for peers to complete
    int peer_idx_end = peer_idx_last + 1;
    int num_peers = peer_idx_end - peer_idx_begin;
    Barrier::wait_eq_reset(
        params.barrier_workspace,
        thread_idx,
        (reduce_tile_idx * Epilogue::kAccumulatorFragments) + reduce_fragment_idx,
        num_peers);

    /// The location of this tile (in threadblock-tile coordinates) in the output matrix
    GemmCoord tiled_coord = params.block_mapping.get_tile_offset(reduce_tile_idx);

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
      tiled_coord.m() * Mma::Shape::kM,
      tiled_coord.n() * Mma::Shape::kN
    );

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);
    typename Epilogue::ElementTensor *ptr_Tensor = static_cast<typename Epilogue::ElementTensor *>(params.ptr_Tensor);

    // Define the reduction output pointer and move to the appropriate place
    typename Epilogue::ElementVector *ptr_Vector =
      static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Additional tensor to load from
    typename Epilogue::TensorTileIterator tensor_iterator(
        params.params_Tensor,
        ptr_Tensor,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_item_begin.column() + tiled_coord.m() * params.ldr;
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue.reduce(
        peer_idx_begin,
        peer_idx_end,
        reduce_fragment_idx,
        params.partials_workspace,
        EpilogueOutputOp(params.output_op),
        ptr_Vector,
        iterator_D,
        iterator_C,
        tensor_iterator,
        params.block_mapping.problem_size.mn(),
        threadblock_item_begin);
  }


  CUTLASS_DEVICE
  void process_tile(
    TileWorkDesc tile_work,
    int block_idx,
    int dp_start_block_idx,
    int block_iter_begin)
  {
    // Initialize input iterators
    typename Mma::IteratorA iterator_A = init_iterator_A(tile_work, params.mode);
    typename Mma::IteratorB iterator_B = init_iterator_B(tile_work, params.mode);

    // Initialize accumulators
    AccumulatorTile accumulator_tile;
    accumulator_tile.clear();

    // Initialize MMA abstraction
    Mma mma(
      shared_storage.main_loop,
      thread_idx,
      warp_idx,
      lane_idx);

    // Perform this tile's range of multiply-accumulate (MAC) iterations
    mma(tile_work.k_iters_remaining, accumulator_tile, iterator_A, iterator_B, accumulator_tile);

    if ((ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic) ||
        (params.block_mapping.reduction_blocks == 0) ||
        (block_idx >= dp_start_block_idx))
    {
      //
      // Cooperative SK peer reduction or DP block
      //

      int first_block_idx = params.block_mapping.get_first_block_idx(tile_work.tile_idx, block_idx);

      if (!tile_work.tile_finished(params)) {
        // Non "finishing" SK blocks must share their partial accumulator sums through global scratch workspace
        share_accumulators(accumulator_tile, block_idx, first_block_idx);
      }
      else
      {
        // DP blocks and "finishing" SK blocks must perform epilogue operations and write the output tile
        if (!tile_work.tile_started())
        {
          // A "finishing" SK block must first aggregate its accumulator partial sums with those shared by peer threadblocks
          acquire_accumulators(accumulator_tile, block_idx, first_block_idx);
        }

        do_epilogue(tile_work, accumulator_tile);
      }
    }
    else
    {
      //
      // Separate peer reduction
      //

      // Share accumulator partial sums with peer threadblock(s) through scratch workspace
      epilogue.share(block_idx, params.partials_workspace, accumulator_tile, tile_work.tile_started());

      // Signal arrival
      Barrier::arrive_range_inc(
        params.barrier_workspace,
        thread_idx,
        tile_work.tile_idx * Epilogue::kAccumulatorFragments,
        Epilogue::kAccumulatorFragments);
    }
  }


  /// Executes one GEMM
  CUTLASS_DEVICE
  void gemm()
  {
    // Initialize block's iteration range
    int tile_idx = 0;
    int block_iter_begin = 0;
    int block_iters_remaining = 0;

    int block_idx = params.block_mapping.get_block_idx();

    int sk_padding_start_block_idx =  params.block_mapping.sk_regions() * params.block_mapping.sk_blocks_per_region();
    int dp_start_block_idx = params.block_mapping.sk_waves * params.block_mapping.avail_sms;
    int reduce_start_block_idx = dp_start_block_idx + params.block_mapping.dp_blocks;
    int grid_padding_start_block_idx = reduce_start_block_idx + params.block_mapping.reduction_blocks;

    // Initialize tile work descriptor
    TileWorkDesc tile_work;

    bool dp_block = (block_idx >= dp_start_block_idx) && (block_idx < reduce_start_block_idx);
    bool sk_block = (block_idx < sk_padding_start_block_idx);
    bool reduce_block = (block_idx >= reduce_start_block_idx) &&
            (block_idx < grid_padding_start_block_idx) &&
            (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kMixed);

    if (dp_block)
    {
      // This is a DP block
      int dp_block_idx = block_idx - dp_start_block_idx;
      int first_dp_tile = (params.block_mapping.cohort_raster) ? 0 : params.block_mapping.sk_tiles;

      // Blocks in first DP wave get configured number of tiles
      tile_idx = first_dp_tile + dp_block_idx;
      int tile_allottment = params.block_mapping.dp_first_wave_tiles;

      // Blocks in subsequent DP waves get 1 tile
      if (dp_block_idx >= params.block_mapping.avail_sms) {
          tile_allottment = 1;
          tile_idx += (params.block_mapping.dp_first_wave_tiles - 1) * params.block_mapping.avail_sms;
      }

      block_iters_remaining = params.block_mapping.iters_per_tile() * tile_allottment;

      init_dp_tile_work(tile_work, tile_idx);

      // DP blocks exit if out of bounds or overlap an SK tile (only possible during cohort rasterization, where dp_first_wave_tiles must be 1)
      if ((tile_idx < params.block_mapping.sk_tiles) ||
          (tile_work.tiled_coord.m() >= params.block_mapping.tiled_shape().m()) ||
          (tile_work.tiled_coord.n() >= params.block_mapping.tiled_shape().n()))
      {
        return;
      }
    }
    else if (sk_block)
    {
      // This is a SK block
      int block_iter_end;
      params.block_mapping.get_iter_extents(block_idx, block_iter_begin, block_iter_end);
      block_iters_remaining = block_iter_end - block_iter_begin;

      tile_idx = params.block_mapping.get_sk_tile_idx(block_iter_end - 1);
      init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
    }
    else
    {
      if (reduce_block)
      {
        // This is a reduction threadblock
        int reduce_block_idx = block_idx - reduce_start_block_idx;
        separate_reduction(reduce_block_idx);
      }

      return;
    }

    // Iteration-processing loop body
    CUTLASS_PRAGMA_NO_UNROLL
    while (true)
    {
      // Perform this block's share of work for this tile
      process_tile(
        tile_work,
        block_idx,
        dp_start_block_idx,
        block_iter_begin);

      block_iters_remaining -= tile_work.k_iters_remaining;

      if (block_iters_remaining == 0)
      {
        break;
      }

      // Continue to next tile
      __syncthreads();

      if (block_idx >= dp_start_block_idx)
      {
        // DP block consume their tiles at stride
        tile_idx += params.block_mapping.avail_sms;
        init_dp_tile_work(tile_work, tile_idx);
      }
      else
      {
        // SK blocks consume their tiles in backwards order
        tile_idx--;
        init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
      }
    }

  }


public:

  //
  // Device-only API
  //

  // Factory invocation
  CUTLASS_DEVICE
  static void invoke(
    Params const &params,
    SharedStorage &shared_storage)
  {
    GemmStreamkWithFusedEpilogue op(params, shared_storage);
    op();
  }


  // Constructor
  CUTLASS_DEVICE
  GemmStreamkWithFusedEpilogue(
      Params const &params,
      SharedStorage &shared_storage)
    :
      params(params),
      shared_storage(shared_storage),
      thread_idx(threadIdx.x),
      warp_idx(__shfl_sync(0xffffffff, threadIdx.x / 32, 0)),   // broadcast the warp_id computed by lane 0 to ensure dependent code
      lane_idx(threadIdx.x % 32),
      epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx)
  {}

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()() {
    // Generic SK code path
    gemm();

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
