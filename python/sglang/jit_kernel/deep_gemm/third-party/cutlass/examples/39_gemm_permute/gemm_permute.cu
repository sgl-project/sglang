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
    \brief GEMM Permute Example.

    This example computes batched GEMM operations with output results permuted as reshaped tensors.

    We provide layout plugin as a flexible tool for users to add any customized input/output tensor permute operation, 
    or any other generalized global memory writeout address computation. To add a customized layout, add new class
    in include/cutlass/layout/permute.h

    In this example we use several permute operations (permute([0, 2, 1, 3]))

    In this example, we used Tensor4DPermuteBMM0213 layout to perform Batched GEMM with permute([0, 2, 1, 3]) on BMM
    whole output tensor, and used Tensor5DPermute20314 layout to perform Normal GEMM with permute([2, 0, 3, 1, 4]) on
    output matrix. The address computations are performed in compute(col_init, row_init, stride_init, 
    BMM_batch_idx) with {col_permute, row_permute and stride_permute} as new addresses after permute op.
    (check include/cutlass/layout/permute.h)

    Tips:
    
      1) Make sure to set batch_stride to zero for BMM permute; also the BMM GEMM should be in mode
      cutlass::gemm::GemmUniversalMode::kBatched instead of kArray.

      2) When the contiguous dimension is touched in permute op (for example [0, 2, 3, 1] for row-major matrix 
      or [1, 0, 2, 3] for column-major), Alignment should be set to 1 for the corresponding matrix. 
      If the last dimension is untouched,  one can set Alignment to be larger like 8 in our example.
      As a result, permute op without touching the unit stride dimension is recommended to obtain the best performance.

    Examples:

      # Runs a batched GEMM with 96 batches
      $ ./examples/39_gemm_permute/39_gemm_permute --problem-count=96

      # Runs a batched GEMM with 96 batches (with GEMM-K dimension equal to 1024)
      $ ./examples/39_gemm_permute/39_gemm_permute --problem-count=96 --k=1024 --verbose=true

      # Execute batched GEMM and profile with NSight
      $ nv-nsight-cu-cli ./examples/39_gemm_permute/39_gemm_permute --m=256 --n=192 --k=256 --verbose=true --iterations=1 --reference-check=false

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/layout/permute.h"

#include "layouts.h"
#include "permute_info.h"

/// Tensor4DPermuteBMM0213 --->
/// Permute layout function for 4-D permuted tensors for BMM with BMM tensor (dimension as [B, M, N]) reshaped
/// as [B/D1, D1, M, N]. Then perform permute([0, 2, 1, 3]) on the corresponding whole BMM tensor.
int constexpr D1 = 12;

/// Tensor5DPermute20314 --->
/// Permute layout function for 5-D permuted tensors with matrix (dimension as [M, N]) reshaped
/// as [M/T1, T1, T2, T3, N/T2/T3]. Then perform permute([2, 0, 3, 1, 4]) on the corresponding tensor.
int constexpr T1 = 16; 
int constexpr T2 = 3;
int constexpr T3 = 8;

/// Tensor4DPermute0213 --->
/// Permute layout function for 4-D permuted tensors with matrix (dimension as [M, N]) reshaped
/// as [M/S1, S1, S2, N/S2]. Then perform permute([0, 2, 1, 3]) on the corresponding tensor.
int constexpr S1 = 8; 
int constexpr S2 = 4;

// // // Alignments
int constexpr AlignmentA = 8;
int constexpr AlignmentB = 8;
int constexpr AlignmentC = 8;

/// GEMM element types
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Useful macros

#define CHECK_CUDA_CALL(call, handler) \
do { \
  cudaError_t __err = (call); \
  if (__err != cudaSuccess) { \
    std::cerr << #call " failed: " << cudaGetErrorString(__err) << std::endl; \
    handler; \
  } \
} while(0)

#define CHECK_CUTLASS_CALL(call, handler) \
do { \
  cutlass::Status __status = (call); \
  if (__status != cutlass::Status::kSuccess) { \
    std::cerr << #call " failed: " << cutlass::cutlassGetStatusString(__status) << std::endl; \
    handler; \
  } \
} while(0)

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;

  cutlass::gemm::GemmCoord problem_each;

  int batch_count;
  int iterations;
  int cuda_streams;
  bool verbose;
  float alpha;
  float beta;

  //
  // Methods
  // 

  Options():
    help(false),
    error(false),
    reference_check(true),
    batch_count(-1),
    iterations(20),
    cuda_streams(0),
    verbose(false),
    alpha(1),
    beta()
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);    
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("verbose", verbose, false);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);

    int m, n, k;

    cmd.get_cmd_line_argument("m", m, 384);
    cmd.get_cmd_line_argument("n", n, 192);
    cmd.get_cmd_line_argument("k", k, 384);
    cmd.get_cmd_line_argument("batch-count", batch_count, 96);

    problem_each = cutlass::gemm::GemmCoord(m, n, k);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << 
      "39_gemm_permute\n"
      "\n"
      " This example tests and profiles the performance of normal GEMM and batched GEMM with different"
      " combinations of fused permutations of input and output tensors."
      "\n"
      " Permutations considered in this example:\n"
      "\n"
      " Normal GEMM:\n"
      " 1) Tensor4DPermute0213: matrix of shape [X, Y] is reshaped as [X/S1, S1, S2, Y/S2] and has its dimensions"
      " permuted as [0, 2, 1, 3], resulting in shape [X/S1, S2, S1, Y/S2] viewed as matrix of shape [X*S2/S1, Y*S1/S2].\n"
      " 2) Tensor5DPermute20314: matrix of shape [X, Y] is reshaped as [X/T1, T1, T2, T3, Y/T2/T3] and has its dimensions"
      " permuted as [2, 0, 3, 1, 4], resulting in shape [T2, X/T1, T3, T1, Y/T2/T3] viewed as matrix of shape [X*T2/T1, Y*T1/T2].\n"
       "\n"
      " Batched GEMM:\n"
      " 3) Tensor4DPermuteBMM0213: batched tensor of 3D shape [B, X, Y] is reshaped as 4D shape [B/D1, D1, X, Y]"
      " and has its dimensions permuted as [0, 2, 1, 3], resulting in shape [B/D1, X, D1, Y] viewed as"
      " a matrix of shape [B/D1, X, Y*D1] for batched GEMM purposes.\n"
      "\n"
      " Note: S1, S2, D1, D2, T1, T2, T3 are compile-time constants defined in gemm_permute.cu."
      " Runtime specification of these values is not supported."
      " These values along with alignment requirements place constraints on supported matrix sizes.\n"
      "\n"
      " Note: X, Y above may refer to M, N or K dimensions of GEMM problem, depending on the tensor considered (A, B or D)."
      " For the output tensor D the values correspond directly to dimensions of D, whereas for A and B the original dimensions"
      " X', Y' are inferred from the ones supplied to the GEMM, taking into account the permute operation.\n"
      "\n"
      "Options:\n"
      "\n"
      "  --help                      If specified, displays this usage statement.\n\n"
      "  --batch-count=<int>         Sets the number of batches in batched GEMM (batch number for BMM). (default: --batch-count=768)\n"
      "  --m=<int>                   Sets the M dimension for both batched GEMM and normal GEMM problems. (default: --m=128)\n"
      "  --n=<int>                   Sets the N dimension for both batched GEMM and normal GEMM problems. (default: --n=192)\n"
      "  --k=<int>                   Sets the K dimension for both batched GEMM and normal GEMM problems. (default: --k=384)\n"
      "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
      "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
      "  --iterations=<int>          Number of profiling iterations to perform.\n"
      "  --reference-check=<bool>    If true, performs reference check.\n"
      "  --verbose=<bool>            If true, prints problem sizes and batching structure.\n"
      "\n"
      "Examples:\n"
      "\n"
      "# Runs a batched GEMM with 96 batches\n"
      "$ ./examples/39_gemm_permute/39_gemm_permute --batch-count=96\n"
      "\n"
      "# Runs a batched GEMM with 96 batches (with GEMM-K dimension equal to 1024)\n"
      "$ ./examples/39_gemm_permute/39_gemm_permute --batch-count=96 --k=1024 --verbose=true\n"
      "\n"
      "# Execute batched GEMM and profile with NSight\n"
      "$ nv-nsight-cu-cli ./examples/39_gemm_permute/39_gemm_permute --m=256 --n=192 --k=256 --verbose=true --iterations=1 --reference-check=false\n"
      "\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, bool batched) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = int64_t();

    fmas += problem_each.product() * (batched ? batch_count : 1);
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace { // (anonymous)

/// Dimension-generic permutation loop
template<int I, typename Element, typename Layout, typename PermuteOp, typename Coord>
void permute_host_impl(
    cutlass::TensorView<Element const, Layout> const & input,
    cutlass::TensorView<Element, Layout> const & output,
    PermuteOp && permute,
    Coord & coord
) {
  static_assert(Layout::kRank == Coord::kRank, "Incompatible Layout and Coord types");
  if constexpr (I == Coord::kRank) {
    output.at(permute(coord)) = input.at(coord);
  }
  else {
    for (coord[I] = 0; coord[I] < input.extent(I); ++coord[I]) {
      permute_host_impl<I+1>(input, output, std::forward<PermuteOp>(permute), coord);
    }
  }
}

} // namespace (anonymous)

/// Perform a reference (host-based) permutation of an input tensor
template<typename PermuteLayout, typename Element, typename Layout>
void permute_host(
    cutlass::TensorView<Element const, Layout> const &input,
    cutlass::TensorView<Element, Layout> const &output,
    int batch_count) {
  Layout layout = input.layout();
  cutlass::MatrixCoord extent = input.extent();

  std::size_t num_elems = layout.capacity(extent) * batch_count;
  std::vector<Element> h_input(num_elems);
  cutlass::device_memory::copy_to_host(h_input.data(), input.data(), num_elems);

  std::vector<Element> h_output(num_elems);

  using Info = PermuteInfo<PermuteLayout>;
  using TensorLayout = typename Info::Layout;

  auto shape_orig = Info::original_shape(extent, batch_count);
  auto shape_perm = Info::permute(shape_orig);

  cutlass::TensorView<Element const, TensorLayout> view_input(h_input.data(), TensorLayout::packed(shape_orig), shape_orig); 
  cutlass::TensorView<Element, TensorLayout> view_output(h_output.data(), TensorLayout::packed(shape_perm), shape_perm);

  decltype(shape_orig) coord;
  permute_host_impl<0>(view_input, view_output, Info::permute, coord);

  cutlass::device_memory::copy_to_device(output.data(), h_output.data(), num_elems);
}

template<typename Layout>
struct LayoutInfo;

template<>
struct LayoutInfo<cutlass::layout::RowMajor> {
  static std::string name() { return "RowMajor"; }
};

template<>
struct LayoutInfo<cutlass::layout::ColumnMajor> {
  static std::string name() { return "ColumnMajor"; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementA, typename ElementB, typename ElementC>
class Testbed {
private:

  //
  // Data members
  //

  Options & options;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint32_t seed;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

public:

  //
  // Methods
  //

  Testbed(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3090
  ):
    options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

private:

  /// Print permutation info for one tensor
  template<typename PermuteLayout>
  void print_tensor_info(
      std::ostream & os,
      std::string const &tensor_name,
      int row_dim,
      int col_dim) {

    cutlass::MatrixCoord extent(options.problem_each.at(row_dim), options.problem_each.at(col_dim));
    using Info = PermuteInfo<PermuteLayout>;

    os << "tensor " << tensor_name << ": " << Info::desc() << "\n";
    os << "    extent: [" << extent.row() << ", " << extent.column() << "]";
    if (Info::kBatched) {
      os << ", batch count: " << options.batch_count;
    }
    os << "\n";
    if (!cutlass::layout::is_trivial_permute<PermuteLayout>) {
      auto shape_orig = Info::original_shape(extent, options.batch_count);
      auto shape_perm = Info::permute(shape_orig);
      os << "  original: [" << shape_orig << "]\n";
      os << "  permuted: [" << shape_perm << "]\n";
    }
  }

  /// Check shape compatibility for one tensor
  template<typename Layout, typename PermuteLayout, int Alignment>
  bool check_tensor_shape(
      std::string const &tensor_name,
      int row_dim,
      int col_dim) {

    cutlass::MatrixCoord extent(options.problem_each.at(row_dim), options.problem_each.at(col_dim));

    using Info = PermuteInfo<PermuteLayout>;

    auto rowAlign = cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value ? Alignment : 1;
    auto colAlign = cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value ? Alignment : 1;

    auto rowFactor = Info::kRowFactor * rowAlign;
    auto colFactor = Info::kColumnFactor * colAlign;

    // Assumes row-major layout
    bool const valid_row = extent.row() % rowFactor == 0;
    if (!valid_row) {
      std::cerr << "Invalid tensor " << tensor_name << " row size = " << extent.row() << ", "
                   "must be divisible by " << rowFactor << ", "
                   "required by " << Info::name() << 
                   (rowAlign > 1 ? (" and alignment of " + std::to_string(rowAlign)) : "") << std::endl;
    }

    bool const valid_col = extent.column() % colFactor == 0;
    if (!valid_col) {
      std::cerr << "Invalid tensor " << tensor_name << " column size = " << extent.column() << ", "
                   "must be divisible by " << colFactor << ", "
                   "required by " << Info::name() << 
                   (colAlign > 1 ? (" and alignment of " + std::to_string(colAlign)) : "") << std::endl;
    }

    bool const valid_bsz = options.batch_count % Info::kBatchFactor == 0;
    if (!valid_bsz) {
      std::cerr << "Invalid batch count = " << options.batch_count << ", "
                   "must be divisible by " << Info::kBatchFactor << ", "
                   "required by " << Info::name() << std::endl;
    }

    return valid_row && valid_col && valid_bsz;
  }

  /// Helper to initialize a tensor view
  template <typename Element>
  void initialize_tensor_(
      Element *ptr,
      size_t capacity, 
      cutlass::Distribution::Kind dist_kind,
      uint32_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      Element scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope_max = 5;
          scope_min = -5;
        }
        else {
          scope_max = 8;
          scope_min = -8;
        }
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::device::BlockFillRandomUniform(
        ptr, capacity, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::device::BlockFillRandomGaussian(
        ptr, capacity, seed, Element(), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      // Fill with increasing elements
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(1), Element());
    } 
    else {

      // Fill with all 1s
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(), Element(1));
    }
  }

  /// Initializes data structures
  void initialize(int batch_count) {

    srand(seed);

    int64_t total_elements_A = options.problem_each.m() * options.problem_each.k() * batch_count;
    int64_t total_elements_B = options.problem_each.n() * options.problem_each.k() * batch_count;
    int64_t total_elements_C = options.problem_each.m() * options.problem_each.n() * batch_count;
    int64_t total_elements_D = options.problem_each.m() * options.problem_each.n() * batch_count;

    // Allocate space
    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);

    // Initialize input tensors
    initialize_tensor_(block_A.get(), total_elements_A, init_A, seed * 2021);
    initialize_tensor_(block_B.get(), total_elements_B, init_B, seed * 2022);
    initialize_tensor_(block_C.get(), total_elements_C, init_C, seed * 2023);

    cutlass::reference::device::BlockFillSequential(
      block_D.get(), total_elements_D, ElementC(), ElementC());
  }


  /// Check device GEMM results against a reference implementation with separate host-based permutation
  template<typename Gemm>
  bool validate(Gemm const &gemm) {

    bool constexpr kBatched = PermuteInfo<typename Gemm::PermuteALayout>::kBatched 
                           || PermuteInfo<typename Gemm::PermuteBLayout>::kBatched 
                           || PermuteInfo<typename Gemm::PermuteDLayout>::kBatched;
                      
    int const batch_count = kBatched ? options.batch_count : 1;

    cutlass::gemm::GemmCoord problem = options.problem_each;

    cutlass::MatrixCoord extent_A{problem.m(), problem.k()};
    cutlass::MatrixCoord extent_B{problem.k(), problem.n()};
    cutlass::MatrixCoord extent_C{problem.m(), problem.n()};

    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    LayoutA layout_A(LayoutA::packed(extent_A));
    LayoutB layout_B(LayoutB::packed(extent_B));
    LayoutC layout_C(LayoutC::packed(extent_C));

    auto size_A = layout_A.capacity(extent_A) * batch_count;
    auto size_B = layout_B.capacity(extent_B) * batch_count;
    auto size_C = layout_C.capacity(extent_C) * batch_count;
    
    cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get(), layout_A, extent_A);
    cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get(), layout_B, extent_B);
    cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get(), layout_C, extent_C);
    cutlass::TensorView<ElementC, LayoutC> view_D(block_D.get(), layout_C, extent_C);

    cutlass::DeviceAllocation<ElementA> block_A_perm(size_A);
    cutlass::DeviceAllocation<ElementA> block_B_perm(size_B);

    cutlass::TensorView<ElementA, LayoutA> view_A_perm(block_A_perm.get(), layout_A, extent_A);
    cutlass::TensorView<ElementB, LayoutB> view_B_perm(block_B_perm.get(), layout_B, extent_B);

    permute_host<typename Gemm::PermuteALayout>(view_A.const_view(), view_A_perm, batch_count);
    permute_host<typename Gemm::PermuteBLayout>(view_B.const_view(), view_B_perm, batch_count);

    cutlass::DeviceAllocation<ElementC>    block_D_ref(size_C);
    cutlass::TensorView<ElementC, LayoutC> view_D_ref(block_D_ref.get(), layout_C, extent_C);

    using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;

    // Reference GEMM
    cutlass::reference::device::GemmComplex<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC, 
        typename EpilogueOutputOp::ElementCompute,
        typename Gemm::ElementAccumulator
    >(
      problem,
      options.alpha, 
      view_A_perm,
      Gemm::kTransformA,
      view_B_perm,
      Gemm::kTransformB,
      options.beta, 
      view_C, 
      view_D_ref, 
      ElementAccumulator(0),
      batch_count,
      options.problem_each.m() * options.problem_each.k(),
      options.problem_each.n() * options.problem_each.k(),
      options.problem_each.m() * options.problem_each.n(),
      options.problem_each.m() * options.problem_each.n()
    );

    cutlass::DeviceAllocation<ElementC>    block_D_perm(size_C);
    cutlass::TensorView<ElementC, LayoutC> view_D_perm(block_D_perm.get(), layout_C, extent_C);
    permute_host<typename Gemm::PermuteDLayout>(view_D_ref.const_view(), view_D_perm, batch_count);

    // Reference check
    return cutlass::reference::device::BlockCompareEqual(view_D_perm.data(), view_D.data(), size_C);
}

public:

  template<typename Gemm>
  bool profile_GEMM_permute() {

    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    using PermuteALayout = typename Gemm::PermuteALayout;
    using PermuteBLayout = typename Gemm::PermuteBLayout;
    using PermuteDLayout = typename Gemm::PermuteDLayout;

    bool constexpr kBatched = PermuteInfo<PermuteALayout>::kBatched 
                           || PermuteInfo<PermuteBLayout>::kBatched 
                           || PermuteInfo<PermuteDLayout>::kBatched;

    std::cout << "\n"
                 "====================================================\n"
                 << (kBatched ? "Batched" : "Normal") << " GEMM:"
                 << "\n  A=" << LayoutInfo<LayoutA>::name() << "," << PermuteInfo<PermuteALayout>::name()
                 << "\n  B=" << LayoutInfo<LayoutB>::name() << "," << PermuteInfo<PermuteBLayout>::name()
                 << "\n  D=" << LayoutInfo<LayoutC>::name() << "," << PermuteInfo<PermuteDLayout>::name()
                 << "\n"
                 "====================================================\n";

    if (options.verbose) {
      print_tensor_info<PermuteALayout>(std::cout, "A", 0, 2);
      print_tensor_info<PermuteBLayout>(std::cout, "B", 2, 1);
      print_tensor_info<PermuteDLayout>(std::cout, "D", 0, 1);
    }
    std::cout << std::endl;

    bool valid = true;
    valid &= check_tensor_shape<LayoutA, PermuteALayout, Gemm::kAlignmentA>("A", 0, 2);
    valid &= check_tensor_shape<LayoutB, PermuteBLayout, Gemm::kAlignmentB>("B", 2, 1);
    valid &= check_tensor_shape<LayoutC, PermuteDLayout, Gemm::kAlignmentC>("D", 0, 1);
    if (!valid)
    {
      std::cout << "Skipped test" << std::endl;
      return true;
    }

    int const batch_count = kBatched ? options.batch_count : 1;

    // Initialize the problem
    initialize(batch_count);

    // Configure the GEMM arguments
    using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Please make sure all problem_sizes are the same for kBatched mode
    auto problem = options.problem_each;

    cutlass::MatrixCoord extent_A{problem.m(), problem.k()};
    cutlass::MatrixCoord extent_B{problem.k(), problem.n()};
    cutlass::MatrixCoord extent_C{problem.m(), problem.n()};

    LayoutA layout_A(LayoutA::packed(extent_A));
    LayoutB layout_B(LayoutB::packed(extent_B));
    LayoutC layout_C(LayoutC::packed(extent_C));

    // Configure GEMM arguments
    typename Gemm::Arguments arguments{
      kBatched ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      batch_count,
      epilogue_op,
      (void*)block_A.get(),
      (void*)block_B.get(),
      (void*)block_C.get(),
      (void*)block_D.get(),
      // For any non-trivial permute the batch stride must be set to 0
      cutlass::layout::is_trivial_permute<PermuteALayout> ? layout_A.capacity(extent_A) : 0,
      cutlass::layout::is_trivial_permute<PermuteBLayout> ? layout_B.capacity(extent_B) : 0,
      layout_C.capacity(extent_C),
      cutlass::layout::is_trivial_permute<PermuteDLayout> ? layout_C.capacity(extent_C) : 0,
      layout_A.stride(0),
      layout_B.stride(0),
      layout_C.stride(0),
      layout_C.stride(0),
    };

    // Initialize the GEMM object
    Gemm gemm_normal;

    CHECK_CUTLASS_CALL(gemm_normal.initialize(arguments, nullptr), return false);

    // Run the normal GEMM object
    CHECK_CUTLASS_CALL(gemm_normal.run(), return false);

    // Wait for completion
    CHECK_CUDA_CALL(cudaDeviceSynchronize(), return false);

    //
    // Verify correctness
    //
    if (options.reference_check) {
      if (validate(gemm_normal)) {
        std::cout << "\nPassed verification\n" << std::endl;
      }
      else {
        std::cerr << "\n*** Error - problem failed the QA check ***\n" << std::endl;
        return false;
      }
    }

    // Warm-up run of the normal GEMM object
    CHECK_CUTLASS_CALL(gemm_normal.run(), return false);

    // Construct events
    cudaEvent_t events[2];
    for (auto & event : events) {
      CHECK_CUDA_CALL(cudaEventCreate(&event), return false);
    }

    // Record an event at the start of a series of GEMM operations
    CHECK_CUDA_CALL(cudaEventRecord(events[0]), return false);

    // Run profiling loop
    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_normal();
    }

    // Record an event when the GEMM operations have been launched.
    CHECK_CUDA_CALL(cudaEventRecord(events[1]), return false);

    // Wait for work on the device to complete.
    CHECK_CUDA_CALL(cudaEventSynchronize(events[1]), return false);

    // Measure elapsed runtime
    float runtime_total_ms = 0;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&runtime_total_ms, events[0], events[1]), return false);

    // Compute average runtime and GFLOPs.
    double runtime_avg_ms = double(runtime_total_ms) / double(options.iterations);
    double gflops = options.gflops(runtime_avg_ms / 1000.0, kBatched);

    // Cleanup
    for (auto event : events) {
      CHECK_CUDA_CALL(cudaEventDestroy(event), return false);
    }

    std::cout << "    Runtime: " << runtime_avg_ms << " ms\n"
                 "     GFLOPs: " << gflops << std::endl;

    return true;
  }
};

/// Shorthand alist for GEMM instantiations
template<typename LayoutA, typename PermuteALayout,
         typename LayoutB, typename PermuteBLayout,
         typename LayoutC, typename PermuteDLayout>
using GemmPermute = cutlass::gemm::device::GemmUniversal<
  ElementInput, LayoutA,
  ElementInput, LayoutB,
  ElementOutput, LayoutC,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 
    AlignmentC, //128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, 
    ElementAccumulator
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  4,              /*kStages*/
  AlignmentA,     /*AlignmentA*/
  AlignmentB,     /*AlignmentB*/
  cutlass::arch::OpMultiplyAdd,
  cutlass::ComplexTransform::kNone,
  cutlass::ComplexTransform::kNone,
  false,  /*GatherA*/
  false,  /*GatherB*/
  false,  /*ScatterD*/
  PermuteDLayout,  /*PermuteDLayout*/
  typename cutlass::layout::InversePermute<PermuteALayout>::type,  /*PermuteALayout*/
  typename cutlass::layout::InversePermute<PermuteBLayout>::type   /*PermuteBLayout*/
>;

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  //
  // This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
  //

  cudaDeviceProp props;

  CHECK_CUDA_CALL(cudaGetDeviceProperties(&props, 0), return EXIT_FAILURE);

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
  
    //
    // This example requires an NVIDIA Ampere-architecture GPU.
    //

    std::cout << "CUTLASS's GEMM+Permute example requires a GPU of NVIDIA's Ampere Architecture "
                 "or later (compute capability 80 or greater).\n";

    return EXIT_SUCCESS;
  }

  //
  // Parse options
  //

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return EXIT_SUCCESS;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Define GEMM types to test
  //

  //
  // TTT (Row-major) GEMMs
  //

  using TTTGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteA = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using TTTGemmNormalPermuteB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using TTTGemmNormalPermuteD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using TTTGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  //
  // NNN (Col-major) GEMMs
  //

  using NNNGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteA = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using NNNGemmNormalPermuteB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using NNNGemmNormalPermuteD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using NNNGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  //
  // NNT (Col-major inputs, row-major output) GEMMs
  //

  using NNTGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteA = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using NNTGemmNormalPermuteB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using NNTGemmNormalPermuteD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  using NNTGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute
  >;

  using NNTGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<T1, T2, T3>
  >;

  //
  // TTN (Row-major inputs, col-major output) GEMMs
  //

  using TTNGemmNormalPermuteNone = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteA = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteAD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using TTNGemmNormalPermuteB = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteBD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using TTNGemmNormalPermuteD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  using TTNGemmNormalPermuteAB = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using TTNGemmNormalPermuteABD = GemmPermute<
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<S1, S2>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<T1, T2, T3>
  >;

  //
  // TTT (Row-major) BMMs
  //

  using TTTGemmBatchedPermuteA = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmBatchedPermuteAD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute
  >;

  using TTTGemmBatchedPermuteBD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteAB = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::NoPermute,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  using TTTGemmBatchedPermuteABD = GemmPermute<
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>,
    cutlass::layout::RowMajor, cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>
  >;

  //
  // NNN (Col-major) BMMs
  //

  using NNNGemmBatchedPermuteA = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmBatchedPermuteAD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  using NNNGemmBatchedPermuteB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmBatchedPermuteBD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  using NNNGemmBatchedPermuteD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  using NNNGemmBatchedPermuteAB = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::NoPermute
  >;

  using NNNGemmBatchedPermuteABD = GemmPermute<
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>,
    cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>
  >;

  //
  // Profile it
  //

  Testbed<ElementInput, ElementInput, ElementOutput> testbed(options);

  bool result = true;

  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<TTTGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<NNNGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<NNTGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteNone>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteA>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteAD>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteB>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteBD>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteD>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteAB>();
  result &= testbed.profile_GEMM_permute<TTNGemmNormalPermuteABD>();

  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteA>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteAD>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteB>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteBD>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteD>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteAB>();
  result &= testbed.profile_GEMM_permute<TTTGemmBatchedPermuteABD>();

  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteA>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteAD>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteB>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteBD>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteD>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteAB>();
  result &= testbed.profile_GEMM_permute<NNNGemmBatchedPermuteABD>();

  std::cout << "\n"
               "====================================================\n"
               "Finished (" << (result ? "PASS" : "FAIL") << ")\n"
               "====================================================" << std::endl;

  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
