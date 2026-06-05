/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/be1788106245496872d18e702978e59b6bfd50e0/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h

#pragma once

#include <cutlass/complex.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/trace.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename Mma_,                ///! Threadblock-scoped matrix multiply-accumulate
    typename Epilogue_,           ///! Epilogue
    typename ThreadblockSwizzle_  ///! Threadblock swizzling function
    >
struct GemmWithEpilogueVisitor {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using TensorRefB = TensorRef<ElementB, LayoutB>;

  using ElementCompute = typename EpilogueVisitor::ElementCompute;
  using LayoutAlphaCol = cutlass::layout::RowMajor;
  using LayoutAlphaRow = cutlass::layout::ColumnMajor;
  using TensorRefAlphaCol = TensorRef<ElementCompute, LayoutAlphaCol>;
  using TensorRefAlphaRow = TensorRef<ElementCompute, LayoutAlphaRow>;

  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename Epilogue::Layout;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;
  using EpilogueOutputOp =
      typename Epilogue::Visitor::ElementwiseFunctor;  // Define type so GemmUniversalBase doesn't complain

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    GemmUniversalMode mode;
    GemmCoord problem_size;
    int batch_count;

    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefAlphaCol ref_alpha_col;
    TensorRefAlphaRow ref_alpha_row;
    TensorRefC ref_C;
    TensorRefC ref_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_D;

    typename EpilogueVisitor::Arguments epilogue_visitor;

    //
    // Methods
    //

    Arguments() : mode(GemmUniversalMode::kGemm), batch_count(1) {}

    /// constructs an arguments structure
    Arguments(
        GemmCoord problem_size_,
        TensorRefA ref_A_,
        TensorRefB ref_B_,
        TensorRefAlphaCol ref_alpha_col_,
        TensorRefAlphaRow ref_alpha_row_,
        TensorRefC ref_C_,
        TensorRefC ref_D_,
        typename EpilogueVisitor::Arguments epilogue_visitor_)
        : mode(GemmUniversalMode::kGemm),
          problem_size(problem_size_),
          batch_count(1),
          ref_A(ref_A_),
          ref_B(ref_B_),
          ref_alpha_col(ref_alpha_col_),
          ref_alpha_row(ref_alpha_row_),
          ref_C(ref_C_),
          ref_D(ref_D_),
          batch_stride_A(0),
          batch_stride_B(0),
          batch_stride_D(0),
          epilogue_visitor(epilogue_visitor_) {}
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename EpilogueVisitor::ScaleTileIterator::Params params_alpha_col;
    typename EpilogueVisitor::ScaleTileIterator::Params params_alpha_row;
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;

    GemmUniversalMode mode;
    int batch_count;
    int gemm_k_size;

    void* ptr_A;
    void* ptr_B;
    typename EpilogueVisitor::ScaleTileIterator::Element* ptr_alpha_col;
    typename EpilogueVisitor::ScaleTileIterator::Element* ptr_alpha_row;
    ElementC* ptr_C;
    ElementC* ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;

    typename EpilogueVisitor::Params epilogue_visitor;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : swizzle_log_tile(0),
          params_A(0),
          params_B(0),
          params_alpha_col(0),
          params_C(0),
          params_D(0),
          batch_count(0),
          gemm_k_size(0),
          mode(cutlass::gemm::GemmUniversalMode::kGemm),
          ptr_A(nullptr),
          ptr_B(nullptr),
          ptr_alpha_col(nullptr),
          ptr_alpha_row(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          batch_stride_A(0),
          batch_stride_B(0) {}

    Params(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape_, int gemm_k_size_, int* workspace_)
        : problem_size(args.problem_size),
          swizzle_log_tile(0),
          params_A(args.ref_A.layout()),
          params_B(args.ref_B.layout()),
          params_alpha_col(args.ref_alpha_col.layout()),
          params_alpha_row(args.ref_alpha_col.layout()),
          params_C(args.ref_C.layout()),
          params_D(args.ref_D.layout()),
          mode(args.mode),
          batch_count(args.batch_count),
          gemm_k_size(args.problem_size.k()),
          ptr_A(args.ref_A.data()),
          ptr_B(args.ref_B.data()),
          ptr_alpha_col(args.ref_alpha_col.data()),
          ptr_alpha_row(args.ref_alpha_row.data()),
          ptr_C(args.ref_C.data()),
          ptr_D(args.ref_D.data()),
          batch_stride_A(args.batch_stride_A),
          batch_stride_B(args.batch_stride_B),
          epilogue_visitor(args.epilogue_visitor) {
      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
          args.problem_size, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, args.batch_count);

      if (args.mode == GemmUniversalMode::kGemm || args.mode == GemmUniversalMode::kGemmSplitKParallel) {
        int const kAlignK =
            const_max(const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value), 1);

        gemm_k_size = round_up(ceil_div(args.problem_size.k(), args.batch_count), kAlignK);

        if (gemm_k_size) {
          grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
        }
      }

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;

    struct {
      typename Epilogue::SharedStorage epilogue;
      typename EpilogueVisitor::SharedStorage visitor;
    } epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmWithEpilogueVisitor() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
    CUTLASS_TRACE_HOST("GemmWithEpilogueVisitor::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = EpilogueVisitor::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (
        platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value ||
        platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (
        platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value ||
        platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (
        platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value ||
        platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
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

  static Status can_implement(Arguments const& args) {
    return can_implement(args.problem_size);
  }

  static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape) {
    return 0;
  }

#define SPLIT_K_ENABLED 1

  /// Executes one GEMM
  CUTLASS_DEVICE
  void run_kernel_(Params const& params, SharedStorage& shared_storage) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA* ptr_A = static_cast<ElementA*>(params.ptr_A);
    ElementB* ptr_B = static_cast<ElementB*>(params.ptr_B);

#if SPLIT_K_ENABLED
    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm || params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {
        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    } else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
    } else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA* const*>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB* const*>(params.ptr_B)[threadblock_tile_offset.k()];
    }
#endif

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{offset_k, threadblock_tile_offset.n() * Mma::Shape::kN};

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
        params.params_A, ptr_A, {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_A);

    typename Mma::IteratorB iterator_B(
        params.params_B, ptr_B, {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // assume identity swizzle
    MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM, threadblock_tile_offset.n() * Mma::Shape::kN);

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    //
    // Construct the epilogue visitor
    //

    bool with_bias = true;
    if (params.ptr_C == nullptr) {
      with_bias = false;
    }

    EpilogueVisitor epilogue_visitor(
        params.epilogue_visitor,
        shared_storage.epilogue.visitor,
        params.problem_size.mn(),
        thread_idx,
        warp_idx,
        lane_idx,
        params.params_alpha_col,
        params.params_C,
        params.params_D,
        with_bias,
        true,
        true,
        params.ptr_alpha_row,
        params.ptr_alpha_col,
        params.ptr_C,
        params.ptr_D,
        threadblock_offset,
        blockIdx.y * params.problem_size.m());

    if (params.mode == GemmUniversalMode::kGemm) {
      // Indicate which position in a serial reduction the output operator is currently updating
      epilogue_visitor.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    } else if (params.mode == GemmUniversalMode::kBatched || params.mode == GemmUniversalMode::kArray) {
      epilogue_visitor.set_batch_index(threadblock_tile_offset.k());
    }

    // Construct the epilogue
    Epilogue epilogue(shared_storage.epilogue.epilogue, thread_idx, warp_idx, lane_idx);

    // Execute the epilogue operator to update the destination tensor.
    epilogue(epilogue_visitor, accumulators);
  }

  template <typename CompilationArch>
  CUTLASS_DEVICE void run_kernel(Params const& params, SharedStorage& shared_storage) {
    if constexpr (platform::is_same<ArchTag, CompilationArch>::value) {
      run_kernel_(params, shared_storage);
    } else {
      CUTLASS_NOT_IMPLEMENTED();
    }
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    run_kernel<ArchTag>(params, shared_storage);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
