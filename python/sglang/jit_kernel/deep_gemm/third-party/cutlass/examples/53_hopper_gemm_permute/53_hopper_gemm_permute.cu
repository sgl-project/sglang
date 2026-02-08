/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Hopper GEMM+permute example.

    This example demonstrates the fusion of tensor permutation operations with a Hopper GEMM kernel.
    It is similar in spirit to example 39_gemm_permute, but uses CUTLASS 3 CollectiveBuilder API to
    construct kernels that make use of Hopper architecture features: Tensor Memory Accelerator (TMA)
    units and warpgroup-level MMA instructions.

    Background
    ----------

    While a GEMM kernel computes a product of two matrices (rank-2 tensors), the source data may
    come from higher-rank tensors by combining some if its modes (dimensions) into the row and column
    modes of the matrix. These tensors are often outputs from previous layers of a network, and the 
    data may sometimes need to be reordered in memory before a GEMM is computed. Similarly, the output
    of a GEMM may need to be reordered before a subsequent operation can be executed.
    
    Consider this sample PyTorch code:

    # Forward pass
    D = torch.mm(A, B).view(M/D1, D1, D2, N/D2).permute(0, 2, 1, 3)

    # Backward pass
    grad_A = torch.mm(grad_D.permute(0, 2, 1, 3).view(M, N), B)

    Executing the reordering as a separate operation requires committing intermediate tensor to memory 
    and increases the latency and memory footprint of the model. By fusing the permutation with either 
    reading of A/B matrices or writing of D matrix, we can avoid the unnecessary global memory traffic
    and kernel launch overhead.

    Implementation
    --------------

    The approach relies on two things:
    - The ability of CUTLASS 3 to naturally perform general tensor contractions (GETT) owing to the
    flexibility of CuTe's hierarchical layouts (see example 51_hopper_gett for more details).
    - The hardware capabilities of Hopper TMA units that allow for loading multidimensional tensors with
    (almost) arbitrary strides, which can be used to represent a permuted view of the data.

    In this example we reuse the permutation classes of examples 39_gemm_permute as operation tags.
    For each tag, a specialization of struct PermuteTraits<> provides the necessary information about
    the target tensor shape and ordering of modes. The main class, ExampleRunner, then figures out the
    overall (hierarchical) shape of the GEMM operation and computes the shape and strides for each
    tensor taking into account the permutation applied. We highlight the importance of specifying
    consistent multidimensional shapes for all tensors (even those that are not permuted), as well as
    choosing hierarchical GEMM tile sizes that best fit those shapes (in cases where some tensor
    dimensions are known at compile time).

    In addition, this example implements a standalone permutation kernel that is used to both verify
    correctness of the fused kernel and benchmark the fused kernel against an unfused version that
    writes intermediate tensor to memory.
*/

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "helper.h"
#include "permute_kernel.cuh"
#include "permute_traits.hpp"

namespace example
{

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct Options {

  bool help;

  cutlass::gemm::BatchedGemmCoord problem_size;

  float alpha;
  float beta;

  bool reference_check;
  int iterations;

  bool verbose;

  Options():
    help(false),
    problem_size({2048, 2048, 2048, 8}),
    alpha(1.0),
    beta(1.0),
    reference_check(true),
    iterations(20),
    verbose(false) { }

  bool valid() const {
    return problem_size.m() > 0
        && problem_size.n() > 0
        && problem_size.k() > 0
        && problem_size.batch() > 0
        && iterations > 0;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("batch_size", problem_size.batch());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("check", reference_check, true);
    cmd.get_cmd_line_argument("iterations", iterations);

    cmd.get_cmd_line_argument("verbose", verbose, false);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out <<
      "53_hopper_gemm_permute example\n"
      "\n"
      "  This example uses the CUTLASS Library to fuse permute() on input/output tensors with GEMM\n"
      "\n"
      "Options:\n"
      "  --help                      If specified, displays this usage statement.\n"
      "  --m=<int>                   GEMM M dimension\n"
      "  --n=<int>                   GEMM N dimension\n"
      "  --k=<int>                   GEMM K dimension\n"
      "  --alpha=<float>             GEMM alpha parameter\n"
      "  --beta=<float>              GEMM beta parameter\n"
      "  --iterations=<int>          Number of profiling iterations to perform.\n"
      "  --check=<bool>              Validate results against a reference (unfused) imlementation"
      "  --verbose=<bool>            Enable verbose output"
      "\n"
      "Examples:\n"
      "\n"
      "$ ./examples/53_hopper_gemm_permute/53_hopper_gemm_permute --m=4096 --n=2048 --k=3072 --batch_size=8\n";

    return out;
  }
};

using namespace cute;

// Check the shapes assigned to the same mode of different tensors,
// ensure all permuted shapes are the same and return that shape.
template<class ... Shapes>
auto
select_mode_shape(Shapes const & ... shapes) {
  auto permuted_shapes = filter_tuple(cute::make_tuple(shapes...), [](auto shape) {
    if constexpr (cute::rank(shape) > 1) {
      return cute::make_tuple(shape);
    }
    else {
      return cute::make_tuple();
    }
  });
  if constexpr (cute::rank(permuted_shapes) == 0) {
    return get<0>(cute::make_tuple(shapes...));
  }
  else {
    auto ref_shape = get<0>(permuted_shapes);
    for_each(permuted_shapes, [&](auto shape) {
      // This static assert fails to compile on GCC 7.5
      // static_assert(is_same<decltype(shape), decltype(ref_shape)>::value, "Inconsistent shapes for the same mode");
      // This runtime check can be skipped if all permutations are required to be static.
      if (shape != ref_shape)
      {
        print("Inconsistent shapes for the same mode: ");
        print(ref_shape); print(" and "); print(shape); print("\n");
        exit(EXIT_FAILURE);
      }
    });
    return ref_shape;
  }
}

template<class Shape, class StrideOrig>
auto
compute_default_stride(Shape const & shape, StrideOrig const & stride_orig) {
  // Only supports column-major and row-major, batch stride always comes last
  if constexpr (is_constant<1, decltype(get<0>(stride_orig))>::value) {
    return compact_col_major(shape);
  }
  else
  {
    return compact_order(shape, Step<_1,_0,_2>{});
  }
}

// Divide a static scalar TileSize into static modes of Shape until either:
// - a dynamic mode is encountered
// - we run out of size to divide
// - no longer divisible by next shape
// Examples:
//   select_tile_shape(_128, (_8,_16)) -> (_8,_16)
//   select_tile_shape(_128, (_8,_32)) -> (_8,_16)
//   select_tile_shape(_128, (_8, _4)) -> (_8,_4,_4)
//   select_tile_shape(_128, (_8,  4)) -> (_8,_16)
template<class TileSize, class Shape>
auto
select_tile_shape(TileSize size, Shape const& shape)
{
  static_assert(is_static<TileSize>::value, "Tile size must be static");
  if constexpr (cute::rank(Shape{}) == 0) {
    return cute::make_tuple(size);
  }
  else {
    if constexpr (is_static<tuple_element_t<0, Shape>>::value) {
      auto div = front(shape);
      if constexpr (size > div and size % div == 0) {
        return prepend(select_tile_shape(size / div, take<1,tuple_size_v<Shape>>(shape)), div);
      }
      else {
        return cute::make_tuple(size);
      }
    }
    else {
      return cute::make_tuple(size);
    }
  }
}

template<class ElementA, class LayoutA, class PermuteA,
         class ElementB, class LayoutB, class PermuteB,
         class ElementC, class LayoutC, class PermuteC,
         class ElementD, class LayoutD, class PermuteD,
         class ElementAccumulator, class ElementEpilogue,
         class TileShape, class ClusterShape>
class ExampleRunner
{
private:

  // Define shapes for each operand and original GEMM problem as a whole.

  using MatrixShape = Shape<int,int,int>;      // [M,N,L]/[M,K,L]/[N,K,L]
  using ProblemShape = Shape<int,int,int,int>; // [M,N,K,L]

  // Determine the CuTe stride for each of the four operands.

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
  using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

  // Flags to check which operands will be permuted.

  static constexpr bool DoPermuteA = not cutlass::layout::is_trivial_permute<PermuteA>;
  static constexpr bool DoPermuteB = not cutlass::layout::is_trivial_permute<PermuteB>;
  static constexpr bool DoPermuteC = not cutlass::layout::is_trivial_permute<PermuteC>;
  static constexpr bool DoPermuteD = not cutlass::layout::is_trivial_permute<PermuteD>;

  // For input operands, we must use inverse of the permutation operation 
  // to read data that is stored in original (un-permuted) order.

  using PermuteAReal = typename cutlass::layout::InversePermute<PermuteA>::type;
  using PermuteBReal = typename cutlass::layout::InversePermute<PermuteB>::type;
  using PermuteCReal = typename cutlass::layout::InversePermute<PermuteC>::type;
  using PermuteDReal = PermuteD;

  // Get permutation layout for each operand.
  // A permutation layout is a rank-3 layout in the usual CuTe mode ordering,
  // but each mode may have a nested shape corresponding to the reshaping of
  // the matrix into a multidimensional tensor, and the strides are computed
  // taking the desired permutation into account.

  template<class Permute, class Stride, bool Transpose>
  using LayoutPermute = remove_cvref_t<decltype(make_permute_layout<Permute, Transpose>(make_layout(MatrixShape{}, Stride{})))>;

  using LayoutAP = LayoutPermute<PermuteAReal, StrideA, false>;
  using LayoutBP = LayoutPermute<PermuteBReal, StrideB, true >;
  using LayoutCP = LayoutPermute<PermuteCReal, StrideC, false>;
  using LayoutDP = LayoutPermute<PermuteDReal, StrideD, false>;

  // Now we want to build the unified problem shape for permute-GEMM.
  // To do this, we check the corresponding mode in each tensor that has it.
  // If at least one tensor has a mode that has been reshaped (i.e. rank > 1),
  // its shape will be used as the reference shape for that mode in all tensors.
  // If multiple tensors have reshaped mode, we additionally check that their 
  // shapes for that mode match. Otherwise, we can't define a consistent GEMM shape.

  using ShapeM = decltype(select_mode_shape(shape<0>(LayoutAP{}), shape<0>(LayoutCP{}), shape<0>(LayoutDP{})));
  using ShapeN = decltype(select_mode_shape(shape<0>(LayoutBP{}), shape<1>(LayoutCP{}), shape<1>(LayoutDP{})));
  using ShapeK = decltype(select_mode_shape(shape<1>(LayoutAP{}), shape<1>(LayoutBP{})));
  using ShapeL = decltype(select_mode_shape(shape<2>(LayoutAP{}), shape<2>(LayoutBP{}), shape<2>(LayoutCP{}), shape<2>(LayoutDP{})));

  using ProblemShapePermute = Shape<ShapeM, ShapeN, ShapeK, ShapeL>;

  using ShapeAPermute = Shape<ShapeM, ShapeK, ShapeL>;
  using ShapeBPermute = Shape<ShapeN, ShapeK, ShapeL>;
  using ShapeCPermute = Shape<ShapeM, ShapeN, ShapeL>;
  using ShapeDPermute = Shape<ShapeM, ShapeN, ShapeL>;

  // Next, we must define the strides for each tensor.
  // If the tensor is permuted, we take the strides produced by the permutation function.
  // Otherwise, we compute default strides induced by the new (multidimensional) shape of the tensor.
  //
  // This won't always work in general if multiple tensors are permuted: e.g. if PermuteA affects
  // modes M and K, and PermuteB affects modes N and L, the single stride for mode L of tensor A
  // computed by PermuteA will be non-congruent with it's shape that is changed by PermuteB. 
  // To handle this correctly, a more complicated logic is needed to reconstruct multi-mode strides.
  // This is not addressed here, as it's not a common requirement to permute multiple tensors in one GEMM.

  using StrideAPermute = conditional_t<DoPermuteA, remove_cvref_t<decltype(stride(LayoutAP{}))>, decltype(compute_default_stride(ShapeAPermute{}, StrideA{}))>;
  using StrideBPermute = conditional_t<DoPermuteB, remove_cvref_t<decltype(stride(LayoutBP{}))>, decltype(compute_default_stride(ShapeBPermute{}, StrideB{}))>;
  using StrideCPermute = conditional_t<DoPermuteC, remove_cvref_t<decltype(stride(LayoutCP{}))>, decltype(compute_default_stride(ShapeCPermute{}, StrideC{}))>;
  using StrideDPermute = conditional_t<DoPermuteD, remove_cvref_t<decltype(stride(LayoutDP{}))>, decltype(compute_default_stride(ShapeDPermute{}, StrideD{}))>;

  // We need to select optimal tile shape based on the tile size specified by the user. 
  // This is done by dividing the tile size in each mode by the mode shape as much
  // as possible (i.e. until we run out of tile size or encounter a dynamic sub-shape).

  using TileMPermute = decltype(select_tile_shape(get<0>(TileShape{}), ShapeM{}));
  using TileNPermute = decltype(select_tile_shape(get<1>(TileShape{}), ShapeN{}));
  using TileKPermute = decltype(select_tile_shape(get<2>(TileShape{}), ShapeK{}));

  using TileShapePermute = Shape<TileMPermute, TileNPermute, TileKPermute>;

  // Now we are ready to define the GEMM kernel types for both fused permute and reference paths.

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementEpilogue,
    ElementC, StrideC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementD, StrideD, 128 / cutlass::sizeof_bits<ElementD>::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveEpiloguePermute = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShapePermute, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementEpilogue,
    ElementC, StrideCPermute, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementD, StrideDPermute, 128 / cutlass::sizeof_bits<ElementD>::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, StrideA, 128 / cutlass::sizeof_bits<ElementA>::value,
    ElementB, StrideB, 128 / cutlass::sizeof_bits<ElementB>::value,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using CollectiveMainloopPermute = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, StrideAPermute, 128 / cutlass::sizeof_bits<ElementA>::value,
    ElementB, StrideBPermute, 128 / cutlass::sizeof_bits<ElementB>::value,
    ElementAccumulator,
    TileShapePermute, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpiloguePermute::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  using GemmKernelPermute = cutlass::gemm::kernel::GemmUniversal<
    ProblemShapePermute,
    CollectiveMainloopPermute,
    CollectiveEpiloguePermute
  >;

  using GemmReference = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using GemmPermute = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelPermute>;

  // Data members

  cutlass::gemm::BatchedGemmCoord problem_size;
  ProblemShape problem_shape;
  cutlass::KernelHardwareInfo hw_info;

  ElementEpilogue alpha;
  ElementEpilogue beta;

  MatrixShape shape_A;
  MatrixShape shape_B;
  MatrixShape shape_C;
  MatrixShape shape_D;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  LayoutAP layout_AP;
  LayoutBP layout_BP;
  LayoutCP layout_CP;
  LayoutDP layout_DP;

  ShapeM shape_M;
  ShapeN shape_N;
  ShapeK shape_K;
  ShapeL shape_L;

  ProblemShapePermute problem_shape_permute;

  ShapeAPermute shape_A_permute;
  ShapeBPermute shape_B_permute;
  ShapeCPermute shape_C_permute;
  ShapeDPermute shape_D_permute;

  StrideAPermute stride_A_permute;
  StrideBPermute stride_B_permute;
  StrideCPermute stride_C_permute;
  StrideDPermute stride_D_permute;

  cutlass::device_memory::allocation<ElementA> tensor_a;
  cutlass::device_memory::allocation<ElementB> tensor_b;
  cutlass::device_memory::allocation<ElementC> tensor_c;
  cutlass::device_memory::allocation<ElementD> tensor_d;

  cutlass::device_memory::allocation<ElementA> tensor_a_permuted;
  cutlass::device_memory::allocation<ElementB> tensor_b_permuted;
  cutlass::device_memory::allocation<ElementC> tensor_c_permuted;
  cutlass::device_memory::allocation<ElementD> tensor_d_unpermuted;
  cutlass::device_memory::allocation<ElementD> tensor_d_reference;

  cutlass::gemm::GemmUniversalMode gemm_mode;

  GemmPermute gemm_permute;
  typename GemmPermute::Arguments arguments_permute;
  cutlass::device_memory::allocation<uint8_t> workspace_permute;

  GemmReference gemm_reference;
  typename GemmReference::Arguments arguments_reference;
  cutlass::device_memory::allocation<uint8_t> workspace_reference;

  public:

  ExampleRunner(Options const & options, cutlass::KernelHardwareInfo const & hw_info)
  : problem_size(options.problem_size),
    problem_shape(problem_size.m(), problem_size.n(), problem_size.k(), problem_size.batch()),
    hw_info(hw_info),
    alpha(options.alpha),
    beta(options.beta),
    shape_A(make_shape(problem_size.m(), problem_size.k(), problem_size.batch())),
    shape_B(make_shape(problem_size.n(), problem_size.k(), problem_size.batch())),
    shape_C(make_shape(problem_size.m(), problem_size.n(), problem_size.batch())),
    shape_D(make_shape(problem_size.m(), problem_size.n(), problem_size.batch())),
    stride_A(cutlass::make_cute_packed_stride(StrideA{}, shape_A)),
    stride_B(cutlass::make_cute_packed_stride(StrideB{}, shape_B)),
    stride_C(cutlass::make_cute_packed_stride(StrideC{}, shape_C)),
    stride_D(cutlass::make_cute_packed_stride(StrideD{}, shape_D)),
    layout_AP(make_permute_layout<PermuteAReal, false>(make_layout(shape_A, stride_A))),
    layout_BP(make_permute_layout<PermuteBReal, true >(make_layout(shape_B, stride_B))),
    layout_CP(make_permute_layout<PermuteCReal, false>(make_layout(shape_C, stride_C))),
    layout_DP(make_permute_layout<PermuteDReal, false>(make_layout(shape_D, stride_D))),
    shape_M(select_mode_shape(shape<0>(layout_AP), shape<0>(layout_CP), shape<0>(layout_DP))),
    shape_N(select_mode_shape(shape<0>(layout_BP), shape<1>(layout_CP), shape<1>(layout_DP))),
    shape_K(select_mode_shape(shape<1>(layout_AP), shape<1>(layout_BP))),
    shape_L(select_mode_shape(shape<2>(layout_AP), shape<2>(layout_BP), shape<2>(layout_CP), shape<2>(layout_DP))),
    problem_shape_permute(shape_M, shape_N, shape_K, shape_L),
    shape_A_permute(make_shape(shape_M, shape_K, shape_L)),
    shape_B_permute(make_shape(shape_N, shape_K, shape_L)),
    shape_C_permute(make_shape(shape_M, shape_N, shape_L)),
    shape_D_permute(make_shape(shape_M, shape_N, shape_L)),
    stride_A_permute(conditional_return<DoPermuteA>(layout_AP.stride(), compute_default_stride(shape_A_permute, stride_A))),
    stride_B_permute(conditional_return<DoPermuteB>(layout_BP.stride(), compute_default_stride(shape_B_permute, stride_B))),
    stride_C_permute(conditional_return<DoPermuteC>(layout_CP.stride(), compute_default_stride(shape_C_permute, stride_C))),
    stride_D_permute(conditional_return<DoPermuteD>(layout_DP.stride(), compute_default_stride(shape_D_permute, stride_D))),
    tensor_a(problem_size.m() * problem_size.k() * problem_size.batch()),
    tensor_b(problem_size.k() * problem_size.n() * problem_size.batch()),
    tensor_c(problem_size.m() * problem_size.n() * problem_size.batch()),
    tensor_d(problem_size.m() * problem_size.n() * problem_size.batch()),
    tensor_a_permuted(problem_size.m() * problem_size.k() * problem_size.batch()),
    tensor_b_permuted(problem_size.k() * problem_size.n() * problem_size.batch()),
    tensor_c_permuted(problem_size.m() * problem_size.n() * problem_size.batch()),
    tensor_d_unpermuted(problem_size.m() * problem_size.n() * problem_size.batch()),
    tensor_d_reference(problem_size.m() * problem_size.n() * problem_size.batch()),
    gemm_mode(problem_size.batch() > 1 ?  cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm),
    arguments_permute{
      gemm_mode,
      problem_shape_permute,
      {
        tensor_a.get(), stride_A_permute,
        tensor_b.get(), stride_B_permute,
      },
      { 
        { alpha, beta },
        tensor_c.get(), stride_C_permute,
        tensor_d.get(), stride_D_permute
      },
      hw_info
    },
    workspace_permute(GemmPermute::get_workspace_size(arguments_permute)),
    arguments_reference{
      gemm_mode,
      problem_shape,
      {
        DoPermuteA ? tensor_a_permuted.get() : tensor_a.get(), stride_A,
        DoPermuteB ? tensor_b_permuted.get() : tensor_b.get(), stride_B
      },
      { 
        { alpha, beta },
        DoPermuteC ? tensor_c_permuted.get()   : tensor_c.get(),           stride_C, 
        DoPermuteD ? tensor_d_unpermuted.get() : tensor_d_reference.get(), stride_D
      },
      hw_info
    },
    workspace_reference(GemmReference::get_workspace_size(arguments_reference))
  {
    if (options.verbose) {
      print("Original GEMM problem:\n");
      print("  Problem shape: "); print(problem_shape); print("\n");
      print("  Layout A: "); print(make_layout(shape_A, stride_A)); print("\n");
      print("  Layout B: "); print(make_layout(shape_B, stride_B)); print("\n");
      print("  Layout C: "); print(make_layout(shape_C, stride_C)); print("\n");
      print("  Layout D: "); print(make_layout(shape_D, stride_D)); print("\n");
      print("  Tile shape: "); print(TileShape{}); print("\n");
      print("With fused permutations:\n");
      print("  Problem shape: "); print(problem_shape_permute); print("\n");
      print("  Layout A: "); print(make_layout(shape_A_permute, stride_A_permute)); print("\n");
      print("  Layout B: "); print(make_layout(shape_B_permute, stride_B_permute)); print("\n");
      print("  Layout C: "); print(make_layout(shape_C_permute, stride_C_permute)); print("\n");
      print("  Layout D: "); print(make_layout(shape_D_permute, stride_D_permute)); print("\n");
      print("  Tile shape: "); print(TileShapePermute{}); print("\n");
    }

    cutlass::reference::device::BlockFillRandomUniform(tensor_a.get(), tensor_a.size(), 1, ElementA(7), ElementA(-8), 0);
    cutlass::reference::device::BlockFillRandomUniform(tensor_b.get(), tensor_b.size(), 2, ElementB(7), ElementB(-8), 0);
    cutlass::reference::device::BlockFillRandomUniform(tensor_c.get(), tensor_c.size(), 3, ElementC(7), ElementC(-8), 0);
    cutlass::reference::device::BlockFillSequential(tensor_d.get(), tensor_d.size(), ElementD(0), ElementD(0));

    auto const gemm_init = [](auto & gemm, auto const & arguments, auto & workspace) {
      cutlass::Status status = gemm.can_implement(arguments);
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Requested GEMM kernel cannot be used for this problem.\n" 
                  << "Check problem sizes and alignment requirements." << std::endl;
        exit(EXIT_FAILURE);
      }     
      status = gemm.initialize(arguments, workspace.get());
      CUTLASS_CHECK(status);
    };

    gemm_init(gemm_permute,   arguments_permute,   workspace_permute  );
    gemm_init(gemm_reference, arguments_reference, workspace_reference);
  }

  void debug_output(std::ostream & os)
  {
    auto print_tensor = [](std::ostream &os, char const * name, auto const & data, auto shape, auto stride)
    {
      std::vector<remove_cvref_t<decltype(*data.get())>> h_data(data.size());
      data.copy_to_host(h_data.data());
      Tensor t = make_tensor(h_data.data(), shape, stride);
      os << "\n" << name << ": " << std::setw(4) << t << std::endl;
    };
    auto [M,N,K,L] = problem_shape;
    print_tensor(os, "A", tensor_a, make_shape(M,K,L), stride_A);
    print_tensor(os, "B", tensor_b, make_shape(N,K,L), stride_B);
    print_tensor(os, "C", tensor_c, make_shape(M,N,L), stride_C);
    print_tensor(os, "D", tensor_d, make_shape(M,N,L), stride_D);
    print_tensor(os, "D reference", tensor_d_reference, make_shape(M,N,L), stride_D);
  }

  template<bool DoTime, class Gemm>
  static float 
  run_gemm(Gemm &gemm)
  {
    GpuTimer timer;
    if constexpr (DoTime) timer.start();
    cutlass::Status status = gemm.run();
    CUTLASS_CHECK(status);
    if constexpr (DoTime) timer.stop();
    if constexpr (DoTime) return timer.elapsed_millis();
    else return 0;
  }

  template<bool DoTime, class Permute, class Element, class Shape, class Stride>
  static float
  run_permute(cutlass::device_memory::allocation<Element> const & input, 
              cutlass::device_memory::allocation<Element> & output,
              Layout<Shape, Stride> const& layout,
              cutlass::KernelHardwareInfo const & hw_info)
  {
    auto idx = find_if(layout.stride(), [](auto x){ return not is_constant<1, decltype(x)>{}; });
    auto stride = get<decltype(idx)::value>(layout.stride());

    GpuTimer timer;
    if constexpr (DoTime) timer.start();
    permute<PermuteTraits<Permute>::kBatched, Permute>(input.get(), 
                                                       output.get(),
                                                       size(take<0,2>(layout)),
                                                       static_cast<int>(stride),
                                                       shape<2>(layout),
                                                       hw_info);
    if constexpr (DoTime) timer.stop();
    if constexpr (DoTime) return timer.elapsed_millis();
    else return 0;
  };

  template<bool DoTime, class Gemm2>
  auto run_reference(Gemm2 &gemm)
  {
    float permute_time = 0.f;
    if constexpr (DoPermuteA) {
      auto orig_layout = make_original_layout<PermuteAReal, false>(make_layout(shape_A, stride_A));
      permute_time += run_permute<DoTime, PermuteA>(tensor_a, tensor_a_permuted, orig_layout, hw_info);
    }
    if constexpr (DoPermuteB) {
      auto orig_layout = make_original_layout<PermuteBReal, true>(make_layout(shape_B, stride_B));
      permute_time += run_permute<DoTime, PermuteB>(tensor_b, tensor_b_permuted, select<1,0,2>(orig_layout), hw_info);
    }
    if constexpr (DoPermuteC) {
      auto orig_layout = make_original_layout<PermuteCReal, false>(make_layout(shape_C, stride_C));
      permute_time += run_permute<DoTime, PermuteC>(tensor_c, tensor_c_permuted, orig_layout, hw_info);
    }

    float gemm_time = run_gemm<DoTime>(gemm);

    if constexpr (DoPermuteD) {
      auto orig_layout = make_layout(shape_D, stride_D);
      permute_time += run_permute<DoTime, PermuteD>(tensor_d_unpermuted, tensor_d_reference, orig_layout, hw_info);
    }

    return cute::make_tuple(gemm_time, permute_time);
  }

  bool verify()
  {
    run_gemm<false>(gemm_permute);
    run_reference<false>(gemm_reference);
    return cutlass::reference::device::BlockCompareEqual(tensor_d.get(), tensor_d_reference.get(), tensor_d.size());
  }

  bool run(Options const &options)
  {
    if (options.reference_check) {
      if (!verify()) {
        std::cout << "Failed validation" << std::endl;
#if 1
        debug_output(std::cout);
#endif
        return false;
      } 
      else {
        std::cout << "Passed validation" << std::endl;
      }
    }

    //
    // Run profiling loop
    //

    auto const benchmark = [&](auto name, auto func)
    {
      GpuTimer timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        func();
      }
      timer.stop();

      double runtime = timer.elapsed_millis() / double(options.iterations);
      double gflops = 2 * double(problem_size.product()) / 1e6 / runtime; // Two flops per multiply-add

      std::cout << name << ":\n";
      std::cout << "  Runtime: " << runtime << " ms\n";
      std::cout << "   GFLOPs: " << gflops  << "\n";
    };

    benchmark("Fused GEMM+permute", [&](){ run_gemm<false>(gemm_permute); });
    benchmark("Unfused GEMM+permute", [&](){ run_reference<false>(gemm_reference); });
    benchmark("Standalone GEMM only", [&](){ run_gemm<false>(gemm_reference); });
    std::cout << "\n";

    return true;
  }
};
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

} // namespace example


int main(int argc, char const **argv)
{
  bool notSupported = false;

  // CUDA 12 minimum required
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA Toolkit version 12 or later.\n";
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (props.major < 9) {
    std::cerr << "This example requires a device with compute capability 90 or higher.\n";
    notSupported = true;
  }
  
  else if (props.major != 9 || props.minor != 0) {
    std::cerr << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
    notSupported = true;
  }
  

  if (notSupported) {
    return EXIT_SUCCESS; // Do not fail CI checks on unsupported systems
  }
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  example::Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return EXIT_SUCCESS;
  }

  if (!options.valid()) {
    std::cerr << "Invalid arguments." << "\n";
    return EXIT_FAILURE;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using namespace cute;

  // Define the data types
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementEpilogue    = float;

  // M=64 for TMA epilogue
  using TileShape = Shape<_128,_128,_64>;

  // Cluster launch with TMA multicast for better perf
  using ClusterShape = Shape<_2,_2,_1>;

  bool result = true;

#define COMPILE_ALL_EXAMPLES 0

  // REGULAR GEMMS

  {
    print("===================================================\n");
    print("Tensor A: RowMajor, Tensor4DPermute0213<8,16>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<8, 16>,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#if COMPILE_ALL_EXAMPLES
  {
    print("===================================================\n");
    print("Tensor A: ColumnMajor, Tensor4DPermute0213<8,16>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<8, 16>,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
  {
    print("===================================================\n");
    print("Tensor B: RowMajor, Tensor4DPermute0213<8,16>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::ColumnMajor, cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<8, 16>,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#endif
  {
    print("===================================================\n");
    print("Tensor B: ColumnMajor, Tensor4DPermute0213<8,16>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<8, 16>,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
  {
    print("===================================================\n");
    print("Tensor D: RowMajor, Tensor4DPermute0213<8,16>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermute0213RowMajor<8, 16>,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#if COMPILE_ALL_EXAMPLES
  {
    print("===================================================\n");
    print("Tensor D: ColumnMajor, Tensor4DPermute0213<8,16>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermute0213ColumnMajor<8, 16>,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#endif
  {
    print("===================================================\n");
    print("Tensor A: RowMajor, Tensor5DPermute20314<16,8,4>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<16,8,4>,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#if COMPILE_ALL_EXAMPLES
  {
    print("===================================================\n");
    print("Tensor A: ColumnMajor, Tensor5DPermute02413<16,8,4>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<16,8,4>,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#endif
  {
    print("===================================================\n");
    print("Tensor D: RowMajor, Tensor5DPermute20314<16,8,4>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::Tensor5DPermute20314RowMajor<16,8,4>,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#if COMPILE_ALL_EXAMPLES
  {
    print("===================================================\n");
    print("Tensor D: ColumnMajor, Tensor5DPermute02413<16,8,4>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::ColumnMajor, cutlass::layout::Tensor5DPermute02413ColumnMajor<16,8,4>,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#endif

  // BATCHED GEMMS

  {
    print("===================================================\n");
    print("Tensor A: RowMajor, Tensor4DPermuteBMM0213<8>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermuteBMM0213RowMajor<8>,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
  {
    print("===================================================\n");
    print("Tensor D: RowMajor, Tensor4DPermuteBMM0213<8>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::Tensor4DPermuteBMM0213RowMajor<8>,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#if COMPILE_ALL_EXAMPLES
  {
    print("===================================================\n");
    print("Tensor A: ColumnMajor, Tensor4DPermuteBMM0321<8>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<8>,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
  {
    print("===================================================\n");
    print("Tensor D: RowMajor, Tensor4DPermuteBMM0321<8>\n");
    using Runner = example::ExampleRunner<ElementA, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementB, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementC, cutlass::layout::RowMajor,    cutlass::layout::NoPermute,
                                          ElementD, cutlass::layout::ColumnMajor, cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<8>,
                                          ElementAccumulator, ElementEpilogue,
                                          TileShape, ClusterShape>;
    Runner runner(options, hw_info);
    result &= runner.run(options);
  }
#endif
  return result ? EXIT_SUCCESS : EXIT_FAILURE;
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
}
