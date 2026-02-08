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
    \brief Tests for device-wide GEMM interface
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <numeric> // std::lcm

#include "../../common/cutlass_unit_test.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/complex.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/detail/collective.hpp"

#include "testbed_utils.h"

#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/gemm.h"

#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class ScalarLoc {
  ON_HOST = 0,
  ON_DEVICE = 1
};

enum class VectorScale {
  DISABLED = 0,
  ENABLED = 1
};

enum class CheckEquality {
  EXACT = 0,
  RELATIVE = 1
};

namespace detail {

inline constexpr auto decomp_mode_to_string =
  [] (cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode mode) -> std::string {
    using Mode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    if (mode == Mode::Heuristic) {
      return "Heuristic";
    }
    else if (mode == Mode::DataParallel) {
      return "DataParallel";
    }
    else if (mode == Mode::SplitK) {
      return "SplitK";
    }
    else if (mode == Mode::StreamK) {
      return "StreamK";
    }
    else {
      return "Unknown";
    }
  };

inline constexpr auto raster_order_to_string =
  [] (cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions mode) -> std::string {
    using Mode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
    if (mode == Mode::Heuristic) {
      return "Heuristic";
    }
    else if (mode == Mode::AlongM) {
      return "AlongM";
    }
    else if (mode == Mode::AlongN) {
      return "AlongN";
    }
    else {
      return "Unknown";
    }
  };

// Helper classes that take default data type when
// the Gemm::EpilogueOutputOp does not have ElementCompute
// and ElementScalar.
// (e.g. when Sm90TreeVisitor is used as FusionCallbacks)
template <typename Gemm, typename Default, typename = void>
struct ElementComputeType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementComputeType<Gemm, Default, std::enable_if_t<not std::is_void_v<typename Gemm::EpilogueOutputOp::ElementCompute>>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementCompute;
};

template <typename Gemm, typename Default, typename = void>
struct ElementScalarType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementScalarType<Gemm, Default, std::enable_if_t<not std::is_void_v<typename Gemm::EpilogueOutputOp::ElementScalar>>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementScalar;
};


template <typename Gemm, typename = void>
struct IsF8F6F4Kernel {
  static constexpr bool value = false;
};

template <typename Gemm>
struct IsF8F6F4Kernel<Gemm, std::void_t<decltype(Gemm::GemmKernel::CollectiveMainloop::IsF8F6F4)>> {
  static constexpr bool value = true;
};


template<class CollectiveEpilogue, class = void>
struct IsSfdEpi : cute::false_type {};

template<class CollectiveEpilogue>
struct IsSfdEpi<CollectiveEpilogue, cute::void_t<typename CollectiveEpilogue::FusionCallbacks::Operation::GmemLayoutTagScalefactor>> : cute::true_type {};

// The maximum swizzle size to use
//
// This class, like Splits above makes it harder to confuse
// the order of arguments of the various run(...) functions in this file.
class MaxSwizzleSize {
public:
  MaxSwizzleSize() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !cute::is_same_v<IntegralNotBool, bool>)) >
  explicit MaxSwizzleSize(IntegralNotBool max_swizzle_size) : max_swizzle_size_(max_swizzle_size) {}
  explicit operator int() const { return max_swizzle_size_; }
private:
  int max_swizzle_size_ = 1;
};

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

template<class T>
struct IsDefaultEpilogue {
  static constexpr bool value = false;
};

template<class ...args>
struct IsDefaultEpilogue<cutlass::epilogue::collective::DefaultEpilogue<args...>> {
  static constexpr bool value = true;
};

template<class ...args>
struct IsDefaultEpilogue<cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<args...>> {
  static constexpr bool value = true;
};

template <typename Epilogue, typename = void>
struct IsLegacyEpiloguePolicy {
  static constexpr bool value = false;
};

template <typename Epilogue>
struct IsLegacyEpiloguePolicy<Epilogue, cute::void_t<decltype(Epilogue::DispatchPolicy::FragmentSize)>> {
  using EpiloguePolicy = typename Epilogue::DispatchPolicy;
  static constexpr bool value = cute::is_same_v<
                                      EpiloguePolicy,
                                      cutlass::epilogue::Sm90TmaWarpSpecializedBiasElementwise<
                                        EpiloguePolicy::StagesC, EpiloguePolicy::StagesD, EpiloguePolicy::FragmentSize>>;
};

// The number of splits to test.
//
// This class makes it harder to confuse the order of arguments
// of the various run(...) functions in this file.  The constructor
// is explicit, so one can't just type 42 (or false, which the
// compiler unhelpfully turns into 0); one has to type Splits(42).
// Splits() picks the default number of splits, 1.
//
// The conversion-to-int operator (operator int()) MUST be explicit!
// Conversion to int MUST require static_cast<int>.
// Otherwise, that defeats a key purpose of this class,
// which is to catch common errors of confusing the order
// of function arguments.
class Splits {
public:
  Splits() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !cute::is_same_v<IntegralNotBool, bool>)) >
  explicit Splits(IntegralNotBool splits) : splits_(splits) {}
  explicit operator int() const { return splits_; }
private:
  int splits_ = 1;
};

// The number of iterations to test.
//
// This class, like Splits above makes it harder to confuse
// the order of arguments of the various run(...) functions in this file.
// Iterations() picks the default number of iterations, 20.
class Iterations {
public:
  Iterations() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !cute::is_same_v<IntegralNotBool, bool>)) >
  explicit Iterations(IntegralNotBool iterations) : iterations_(iterations) {}
  explicit operator int() const { return iterations_; }
private:
  int iterations_ = 20;
};

template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {
    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    }

    else if (bits_input <= 6) {
      scope_max = 2;
      scope_min = -2;
    }

    else if (bits_input <= 8) {

      if constexpr (
                    cute::is_same_v<Element, cutlass::float_ue8m0_t>){
        scope_max = 4;
        scope_min = 1;
      }
      else {

        scope_max = 1;
        scope_min = -1;

      }

    }
    else{
      scope_max = 4;
      scope_min = -4;
    }
    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }

  else if (dist_kind == cutlass::Distribution::Identity) {
    cutlass::reference::host::TensorFillIdentity(view);
  }

  else if (dist_kind == cutlass::Distribution::Gaussian) {
    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }

  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(
      view.data(), view.capacity());
  }

  else if (dist_kind == cutlass::Distribution::AllOnes) {
    cutlass::reference::host::TensorFill(view, Element(1));
  }

  else {
    EXPECT_TRUE(false) << "Not implemented";
    return false;
  }

  return true;
}

// Looks at Cute Stride to check Row / Column Major
template<typename Stride>
static constexpr bool is_row_or_col_major(){
  int stride_0 = int(cute::size<0>(Stride{}));
  int stride_1 = int(cute::size<1>(Stride{}));
  int depth = cute::depth(Stride{});
  return ((stride_0 == 1) || (stride_1 == 1)) && (depth == 1);
}


//
// Default MMA input Operands : A , B
//
template<
  class ScheduleType_,
  class Gemm,
  class ElementA_ = typename Gemm::GemmKernel::ElementA,
  class ElementB_ = typename Gemm::GemmKernel::ElementB,
  class Enable = void>
struct HostCollectiveMainloop {
  // Kernel data types
  using ElementA = ElementA_;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using ElementB = ElementB_;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using LayoutTagA = cutlass::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;

  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;

  using Arguments = typename Gemm::GemmKernel::MainloopArguments;

  cutlass::ComplexTransform TransformA = Gemm::kTransformA;
  cutlass::ComplexTransform TransformB = Gemm::kTransformB;

  StrideA stride_a;
  StrideB stride_b;

  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;

  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");

  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed,
    typename LayoutTagA::Stride stride_factor_A_ = typename LayoutTagA::Stride(),
    typename LayoutTagB::Stride stride_factor_B_ = typename LayoutTagB::Stride()
  ):
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    init_A(init_A_), init_B(init_B_), seed(seed_),
    check_relative_equality(check_relative_equality_) { }

  template<class ProblemShapeType>
  bool initialize(ProblemShapeType problem_size) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("HostCollectiveMainloop (generic)::initialize(problem_shape)");
#endif
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto a_coord = cutlass::make_Coord(M * L, K);
    // Cutlass has Row/Col major refers to MxK times KxN matrix product,
    // so the HostTensorB should be treated as KxN in "coord"'s view
    auto b_coord = cutlass::make_Coord(K, N * L);

    try {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor_A.resize");
#endif
      tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor_B.resize");
#endif
      tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor A or B resize threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor A or B resize threw an unknown exception");
      throw;
    }

    try {
      EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
      EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));
    }
    catch (cutlass::cuda_exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: checked initialize_tensor threw cutlass::cuda_exception: " << e);
      throw;
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: checked initialize_tensor threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: checked_initialize_tensor threw an unknown exception");
      throw;
    }

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: Check last error before sync_device()");
      cudaError_t error = cudaGetLastError();
      const auto error_str = cudaGetErrorString(error);
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: cudaGetLastError() is " << error_str);
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor_A.host_data()=" << tensor_A.host_data() << ", tensor_A.device_data()=" << tensor_A.device_data());
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor_B.host_data()=" << tensor_B.host_data() << ", tensor_B.device_data()=" << tensor_B.device_data());
    }
#endif
    try {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor_A.sync_device");
#endif
      tensor_A.sync_device();
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: tensor_B.sync_device");
#endif
      tensor_B.sync_device();
    }
    catch (cutlass::cuda_exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: sync_device() threw cutlass::cuda_exception: " << e);
      throw;
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: sync_device() threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: sync_device() threw an unknown exception");
      throw;
    }

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("HostCollectiveMainloop::initialize: Reached end");
#endif
    return true;
  }

  Arguments to_args() {


    // Runtime datatype selection
    if constexpr (not cute::is_same_v<ElementA, typename Gemm::GemmKernel::ElementA>) {
      using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
      using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
      return {
        reinterpret_cast<ArrayElementA *>(tensor_A.device_data()), stride_a,
        reinterpret_cast<ArrayElementB *>(tensor_B.device_data()), stride_b
      };
    }
    else {

    Arguments arguments =
    {
      tensor_A.device_data(), stride_a, tensor_B.device_data(), stride_b
    };
    return arguments;
    }
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);
    auto A = make_tensor(make_iterator(tensor_A.host_data()),
          make_layout(make_shape(M, K, L), stride_a));
    auto B = make_tensor(make_iterator(tensor_B.host_data()),
        make_layout(make_shape(N, K, L), stride_b));


    auto dummy_SFA = cute::make_tensor(static_cast<ElementA*>(nullptr),
        cute::make_layout(cute::make_shape(M, K, L), stride_a));
    auto dummy_SFB = cute::make_tensor(static_cast<ElementB*>(nullptr),
        cute::make_layout(cute::make_shape(N, K, L), stride_b));

    cutlass::reference::host::GettMainloopParams<ElementAccumulator,
                                                 decltype(A),
                                                 decltype(B)

                                                 , decltype(dummy_SFA),
                                                 decltype(dummy_SFB)

                                                 > mainloop_params{};

    mainloop_params.A = A;
    mainloop_params.B = B;
    mainloop_params.transform_A = TransformA;
    mainloop_params.transform_B = TransformB;

    return mainloop_params;
  }

  void print_tensors(std::ofstream& file) {
    file << "A =\n" << tensor_A.host_view()
         << "\nB =\n" << tensor_B.host_view();
  }

  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {

    // Factors used for calculating relative equality. CUTLASS's relative-equality
    // checks in include/cutlass/relatively_equal.h  are inspired by
    // https://floating-point-gui.de/errors/comparison/. This reference suggests using
    // the minimum normal value of a given type as the nonzero_floor.
    Element epsilon(static_cast<Element>(0.1f));
    Element nonzero_floor(std::numeric_limits<Element>::min());

    if constexpr (!cutlass::is_complex<Element>::value) {
      if (check_relative_equality == CheckEquality::RELATIVE) {
        return cutlass::reference::host::TensorRelativelyEquals(
          lhs, rhs, epsilon, nonzero_floor);
      }
      else {
        return cutlass::reference::host::TensorEquals(lhs, rhs);
      }
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL) {
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);

    bool passed = true;
    return passed;
  }
};

//
// Sparse MMA host implementation
//
template<
  class Gemm,
  class ElementA_,
  class ElementB_>
struct HostCollectiveMainloopSparse
{

  // Kernel data types
  using ElementA = ElementA_;
  // CuTe layout A for the kernel's sparse tensorA.
  using LayoutA  = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
  using ElementB = ElementB_;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;

  using ElementE = typename Gemm::GemmKernel::CollectiveMainloop::ElementE;
  // CuTe layout E for the kernel's metadata tensor.
  using LayoutE  = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
  using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;

  // The following typenames are for the reference host tensors. They are non-sparse tensors.
  using LayoutTagA = decltype(SparseConfig::deduce_layoutA_tag(LayoutA{}));
  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  // We don't care about the actual strideE for the host tensor, but just need one to allocate memory.
  using StrideE = StrideA;

  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagE = cutlass::detail::StrideToLayoutTagA_t<StrideE>;

  using ArchTag = typename Gemm::ArchTag;

  using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                              cute::Shape<int, int, int, int>,
                              ElementA,
                              LayoutTagA,
                              SparseConfig>;

  using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
                              cute::Shape<int, int, int, int>,
                              ElementA,
                              LayoutTagA,
                              SparseConfig,
                              ArchTag>;

  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  using Arguments = typename Gemm::GemmKernel::MainloopArguments;
  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");

  StrideA stride_a;
  StrideA stride_a_compressed;
  StrideB stride_b;
  StrideE stride_e;

  LayoutA layout_a;
  LayoutE layout_e;

  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;
  typename LayoutTagE::Stride stride_factor_E;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A_Comp;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  cutlass::HostTensor<ElementE, LayoutTagE> tensor_E;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;
  static constexpr int MaxSmCount = 16;

  HostCollectiveMainloopSparse(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed,
    typename LayoutTagA::Stride stride_factor_A_ = typename LayoutTagA::Stride(),
    typename LayoutTagB::Stride stride_factor_B_ = typename LayoutTagB::Stride(),
    typename LayoutTagE::Stride stride_factor_E_ = typename LayoutTagE::Stride()
  ):
    check_relative_equality(check_relative_equality_),
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    stride_factor_E(stride_factor_E_),
    init_A(init_A_), init_B(init_B_), seed(seed_) { }

  template<class ProblemShapeType>
  bool initialize(ProblemShapeType problem_size) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("HostCollectiveMainloopSparse::initialize");
#endif
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));

    CompressorUtility compressor_utility(problem_shape_MNKL, stride_a);

    // TensorE
    // In unit of ElementE (uint8_t), after alignment requirement
    // M-dim: TensorEAtom_M alignment
    // K-dim: TensorEAtom_K alignment
    int KAlignedE = compressor_utility.get_metadata_k_physical();
    int MAlignedE = compressor_utility.get_metadata_m_physical();

    // TensorA Compressed
    // In unit of ElementARaw, after alignment requirement
    // M-dim: TMA alignment
    // K-dim: TMA alignment
    int KAlignedAC = compressor_utility.get_tensorA_k_physical();
    int MAlignedAC = compressor_utility.get_tensorA_m_physical();

    stride_a_compressed = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, KAlignedAC, L));
    stride_e = cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(MAlignedE, KAlignedE, L));

    auto a_coord = cutlass::make_Coord(M * L, K);
    auto b_coord = cutlass::make_Coord(K, N * L);
    auto e_coord = cutlass::make_Coord(MAlignedE * L, KAlignedE);
    auto a_comp_coord = cutlass::make_Coord(MAlignedAC * L, KAlignedAC);

    tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_A_Comp.resize(a_comp_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_comp_coord, stride_factor_A));
    tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
    tensor_E.resize(e_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagE>::layout_factory(e_coord, stride_factor_E));

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);

    compressor_utility.structure_sparse_zero_mask_fill(tensor_A.host_data(), static_cast<int>(seed + 2023));

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_E.sync_device();
    tensor_A_Comp.sync_device();

    cutlass::Status status {cutlass::Status::kSuccess };

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    typename Compressor::Arguments arguments{
      {M, N, K, L},
      {tensor_A.device_data(),
       stride_a,
       tensor_A_Comp.device_data(),
       tensor_E.device_data()},
      {hw_info}
    };

    Compressor compressor_op;
    size_t workspace_size = Compressor::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = compressor_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = compressor_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = compressor_op.run();

    auto result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
      return false;
    }

    layout_a = SparseConfig::fill_layoutA(problem_shape_MNKL);
    layout_e = SparseConfig::fill_layoutE(problem_shape_MNKL);

    tensor_E.sync_host();
    tensor_A_Comp.sync_host();

    return true;
  }

  Arguments to_args() {
    using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
    return {
      reinterpret_cast<ArrayElementA *>(tensor_A_Comp.device_data()), layout_a,
      reinterpret_cast<ArrayElementB *>(tensor_B.device_data()), stride_b,
      tensor_E.device_data(), layout_e
    };
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);
    auto A = make_tensor(make_iterator(tensor_A.host_data()),
          make_layout(make_shape(M, K, L), stride_a));
    auto B = make_tensor(make_iterator(tensor_B.host_data()),
        make_layout(make_shape(N, K, L), stride_b));

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};
    return mainloop_params;
  }

  void print_tensors(std::ofstream& file) {
    file << "A =\n" << tensor_A.host_view()
         << "\nB =\n" << tensor_B.host_view();
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL) {
    auto [M, N, K, L] = problem_shape_MNKL;

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    return true;
  }
};

template<
  class ScheduleType_,
  class Gemm,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<ScheduleType_, Gemm, ElementA_, ElementB_,
    cute::enable_if_t<
      cute::is_base_of_v<
        cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedSparse<Gemm::CollectiveMainloop::DispatchPolicy::Stages,
                                                                typename Gemm::CollectiveMainloop::DispatchPolicy::ClusterShape,
                                                                ScheduleType_>,
        typename Gemm::CollectiveMainloop::DispatchPolicy>>>
  : HostCollectiveMainloopSparse<Gemm, ElementA_, ElementB_>
{
  using HostCollectiveMainloopSparse<Gemm, ElementA_, ElementB_>::HostCollectiveMainloopSparse;
};

//
// Sparse MMA input Operands : A_compressed, B, metadata
//
// Structured Sparse Gemm Input Operands

template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  typename ElementA_,
  typename ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedSm100<SchedulerPipelineStageCount_,
                                                                                 AccumulatorPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_>
  : HostCollectiveMainloopSparse<Gemm, ElementA_, ElementB_>
{
  using HostCollectiveMainloopSparse<Gemm, ElementA_, ElementB_>::HostCollectiveMainloopSparse;
};

//
// Sparse Gemm Input Operands : A , B, E
//
template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseSm120<SchedulerPipelineStageCount_, false /*isAsymmetric*/>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedSm100<0/*SchedulerPipelineStageCount_*/,
                                                                                 0/*AccumulatorPipelineStageCount_*/>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedSm100<0,0>,
                                      Gemm, ElementA_, ElementB_ >;
  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = Base::kDefaultSeed,
    typename Base::LayoutTagA::Stride stride_factor_A_ = typename Base::LayoutTagA::Stride(),
    typename Base::LayoutTagB::Stride stride_factor_B_ = typename Base::LayoutTagB::Stride(),
    typename Base::LayoutTagE::Stride stride_factor_E_ = typename Base::LayoutTagE::Stride()
  ) : Base::HostCollectiveMainloop(check_relative_equality_, init_A_, init_B_, seed_, stride_factor_A_,
                                                                                      stride_factor_B_,
                                                                                      stride_factor_E_) {}
};

//
// Sparse Gemm Input Operands : A , B, E
//
template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseSm120<SchedulerPipelineStageCount_, true /*isAsymmetric*/>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedSm100<0/*SchedulerPipelineStageCount_*/,
                                                                                 0/*AccumulatorPipelineStageCount_*/>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedSm100<0,0>,
                                      Gemm, ElementA_, ElementB_ >;
  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = Base::kDefaultSeed,
    typename Base::LayoutTagA::Stride stride_factor_A_ = typename Base::LayoutTagA::Stride(),
    typename Base::LayoutTagB::Stride stride_factor_B_ = typename Base::LayoutTagB::Stride(),
    typename Base::LayoutTagE::Stride stride_factor_E_ = typename Base::LayoutTagE::Stride()
  ) : Base::HostCollectiveMainloop(check_relative_equality_, init_A_, init_B_, seed_, stride_factor_A_,
                                                                                      stride_factor_B_,
                                                                                      stride_factor_E_) {}
};

//
// Block Scaled Gemm Input Operands : A , B, scalefactorA, scalefactorB
//
template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_,
                                                                                      AccumulatorPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> {
  // Kernel data types
  using ElementA = ElementA_;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using ElementB = ElementB_;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using LayoutTagA = cutlass::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;

  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;

  static constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;

  using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using Blk_MN   = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF   = typename Sm1xxBlkScaledConfig::Blk_SF;
  using SfAtom   = typename Sm1xxBlkScaledConfig::SfAtom;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

  using Arguments = typename Gemm::GemmKernel::MainloopArguments;

  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;

  StrideA stride_a;
  StrideB stride_b;

  LayoutSFA layout_sfa;
  LayoutSFB layout_sfb;

  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  cutlass::HostTensor<ElementSF, LayoutTagA> tensor_SFA;
  cutlass::HostTensor<ElementSF, LayoutTagB> tensor_SFB;

  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");

  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed,
    typename LayoutTagA::Stride stride_factor_A_ = typename LayoutTagA::Stride(),
    typename LayoutTagB::Stride stride_factor_B_ = typename LayoutTagB::Stride()
  ):
    check_relative_equality(check_relative_equality_),
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    init_A(init_A_), init_B(init_B_), seed(seed_) { }

  template<class ProblemShapeType>
  bool initialize(ProblemShapeType problem_size) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("HostCollectiveMainloop (KernelTmaWarpSpecializedBlockScaledSm100)::initialize");
#endif
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto a_coord = cutlass::make_Coord(M * L, K);
    // Cutlass has Row/Col major refers to MxK times KxN matrix product,
    // so the HostTensorB should be treated as KxN in "coord"'s view
    auto b_coord = cutlass::make_Coord(K, N * L);

    tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);

    tensor_A.sync_device();
    tensor_B.sync_device();

    using namespace cute;
    auto k_blks = cutlass::ceil_div(K, size<1>(shape(SfAtom{})));
    auto m_blks = cutlass::ceil_div(M, Blk_MN{});
    auto n_blks = cutlass::ceil_div(N, Blk_MN{});
    layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
    layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto sfa_coord   = cutlass::make_Coord(m_blks * Blk_MN{} * L, k_blks * Blk_SF{});
    auto sfb_coord   = cutlass::make_Coord(n_blks * Blk_MN{} * L, k_blks * Blk_SF{});

    tensor_SFA.resize(sfa_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(sfa_coord, stride_factor_A));
    tensor_SFB.resize(sfb_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(sfb_coord, stride_factor_B));

    EXPECT_TRUE(initialize_tensor(tensor_SFA.host_view(), init_A, seed + 2024));
    EXPECT_TRUE(initialize_tensor(tensor_SFB.host_view(), init_B, seed + 2025));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_SFA.host_view().at({0, 0}) = ElementSF(1);
    tensor_SFB.host_view().at({0, 0}) = ElementSF(1);

    tensor_SFA.sync_device();
    tensor_SFB.sync_device();

    return true;
  }

  Arguments to_args() {
    using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
    return {
      reinterpret_cast<ArrayElementA *>(tensor_A.device_data()), stride_a,
      reinterpret_cast<ArrayElementB *>(tensor_B.device_data()), stride_b,
      tensor_SFA.device_data(), layout_sfa,
      tensor_SFB.device_data(), layout_sfb
    };
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);
    auto A = make_tensor(make_iterator(tensor_A.host_data()),
          make_layout(make_shape(M, K, L), stride_a));
    auto SfA = make_tensor(tensor_SFA.host_data(), layout_sfa);

    auto B = make_tensor(make_iterator(tensor_B.host_data()),
        make_layout(make_shape(N, K, L), stride_b));
    auto SfB = make_tensor(tensor_SFB.host_data(), layout_sfb);

    cutlass::reference::host::GettMainloopParams<ElementAccumulator,
        decltype(A),
        decltype(B),
        decltype(SfA),
        decltype(SfB)
      >
      mainloop_params{A, SfA, B, SfB};
    return mainloop_params;
  }

  void print_tensors(std::ofstream& file) {
    file << "A =\n" << tensor_A.host_view()
         << "\nB =\n" << tensor_B.host_view()
         << "\nSFA =\n" << tensor_SFA.host_view()
         << "\nSFB =\n" << tensor_SFB.host_view();
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL) {
    auto [M, N, K, L] = problem_shape_MNKL;

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_SFA.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_SFB.host_view()), 0);
    return true;
  }
};


//
// Block Scaled Gemm Input Operands : A , B, scalefactorA, scalefactorB
//
template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedPingpongBlockScaledSm120<SchedulerPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<0,0>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<0,0>,
                                      Gemm, ElementA_, ElementB_>;
  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = Base::kDefaultSeed,
    typename Base::LayoutTagA::Stride stride_factor_A_ = typename Base::LayoutTagA::Stride(),
    typename Base::LayoutTagB::Stride stride_factor_B_ = typename Base::LayoutTagB::Stride()
  ) : Base::HostCollectiveMainloop(check_relative_equality_, init_A_, init_B_, seed_, stride_factor_A_, stride_factor_B_) {}
};

//
// Block Scaled Gemm Input Operands : A , B, scalefactorA, scalefactorB
//
template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedCooperativeBlockScaledSm120<SchedulerPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<0,0>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<0,0>,
                                      Gemm, ElementA_, ElementB_>;
  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = Base::kDefaultSeed,
    typename Base::LayoutTagA::Stride stride_factor_A_ = typename Base::LayoutTagA::Stride(),
    typename Base::LayoutTagB::Stride stride_factor_B_ = typename Base::LayoutTagB::Stride()
  ) : Base::HostCollectiveMainloop(check_relative_equality_, init_A_, init_B_, seed_, stride_factor_A_, stride_factor_B_) {}
};

//
// Block Scaled Gemm Input Operands : A , B, scalefactorA, scalefactorB
//
template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm103<SchedulerPipelineStageCount_,
                                                                                      AccumulatorPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_,AccumulatorPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_,AccumulatorPipelineStageCount_>,
                                      Gemm, ElementA_, ElementB_>;
  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = Base::kDefaultSeed,
    typename Base::LayoutTagA::Stride stride_factor_A_ = typename Base::LayoutTagA::Stride(),
    typename Base::LayoutTagB::Stride stride_factor_B_ = typename Base::LayoutTagB::Stride()
  ) : Base::HostCollectiveMainloop(check_relative_equality_, init_A_, init_B_, seed_, stride_factor_A_, stride_factor_B_) {}
};

//
// Block Scaled Structured Sparse Gemm Input Operands : A_compressed, B, metadata, scalefactorA, scalefactorB
//
template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  typename ElementA_,
  typename ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_,
                                                                                            AccumulatorPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> {
  // Kernel data types
  using ElementA = ElementA_;
  // CuTe layout A for the kernel's sparse tensorA.
  using LayoutA  = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
  using ElementB = ElementB_;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;

  using ElementE = typename Gemm::GemmKernel::CollectiveMainloop::ElementE;
  // CuTe layout E for the kernel's metadata tensor.
  using LayoutE  = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
  using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;

  // The following typenames are for the reference host tensors. They are non-sparse tensors.
  using LayoutTagA = decltype(SparseConfig::deduce_layoutA_tag(LayoutA{}));
  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  // We don't care about the actual strideE for the host tensor, but just need one to allocate memory.
  using StrideE = StrideA;

  static constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;
  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;

  using LayoutTagE = cutlass::detail::StrideToLayoutTagA_t<StrideE>;

  using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using Blk_MN   = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF   = typename Sm1xxBlkScaledConfig::Blk_SF;
  using SfAtom   = typename Sm1xxBlkScaledConfig::SfAtom;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

  using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                              cute::Shape<int, int, int, int>,
                              ElementA,
                              LayoutTagA,
                              SparseConfig>;
  using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
                        cute::Shape<int, int, int, int>,
                        ElementA,
                        LayoutTagA,
                        SparseConfig,
                        cutlass::arch::Sm100>;

  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  using Arguments = typename Gemm::GemmKernel::MainloopArguments;
  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;

  StrideA stride_a;
  StrideA stride_a_compressed;
  StrideB stride_b;
  StrideE stride_e;

  LayoutA layout_a;
  LayoutE layout_e;
  LayoutSFA layout_sfa;
  LayoutSFB layout_sfb;

  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;
  typename LayoutTagE::Stride stride_factor_E;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A_Comp;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  cutlass::HostTensor<ElementE, LayoutTagE> tensor_E;
  cutlass::HostTensor<ElementSF, LayoutTagA> tensor_SFA;
  cutlass::HostTensor<ElementSF, LayoutTagB> tensor_SFB;

  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");

  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed,
    typename LayoutTagA::Stride stride_factor_A_ = typename LayoutTagA::Stride(),
    typename LayoutTagB::Stride stride_factor_B_ = typename LayoutTagB::Stride(),
    typename LayoutTagE::Stride stride_factor_E_ = typename LayoutTagE::Stride()
  ):
    check_relative_equality(check_relative_equality_),
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    stride_factor_E(stride_factor_E_),
    init_A(init_A_), init_B(init_B_), seed(seed_) { }

  template<class ProblemShapeType>
  bool initialize(ProblemShapeType problem_size) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("HostCollectiveMainloop (KernelSparseTmaWarpSpecializedBlockScaledSm100)::initialize");
#endif
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));

    CompressorUtility compressor_utility(problem_shape_MNKL, stride_a);

    // TensorE
    // In unit of ElementE (uint8_t), after alignment requirement
    // M-dim: TensorEAtom_M alignment
    // K-dim: TensorEAtom_K alignment
    int KAlignedE = compressor_utility.get_metadata_k_physical();
    int MAlignedE = compressor_utility.get_metadata_m_physical();

    // TensorA Compressed
    // In unit of ElementARaw, after alignment requirement
    // M-dim: TMA alignment
    // K-dim: TMA alignment
    int KAlignedAC = compressor_utility.get_tensorA_k_physical();
    int MAlignedAC = compressor_utility.get_tensorA_m_physical();

    stride_a_compressed = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, KAlignedAC, L));
    stride_e = cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(MAlignedE, KAlignedE, L));

    auto a_coord = cutlass::make_Coord(M * L, K);
    auto b_coord = cutlass::make_Coord(K, N * L);
    auto e_coord = cutlass::make_Coord(MAlignedE * L, KAlignedE);
    auto a_comp_coord = cutlass::make_Coord(MAlignedAC * L, KAlignedAC);

    tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_A_Comp.resize(a_comp_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_comp_coord, stride_factor_A));
    tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
    tensor_E.resize(e_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagE>::layout_factory(e_coord, stride_factor_E));

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);

    compressor_utility.structure_sparse_zero_mask_fill(tensor_A.host_data(), static_cast<int>(seed + 2023));

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_E.sync_device();
    tensor_A_Comp.sync_device();

    cutlass::Status status {cutlass::Status::kSuccess };

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    typename Compressor::Arguments arguments{
      {M, N, K, L},
      {tensor_A.device_data(),
       stride_a,
       tensor_A_Comp.device_data(),
       tensor_E.device_data()},
      {hw_info}
    };

    Compressor compressor_op;
    size_t workspace_size = Compressor::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = compressor_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = compressor_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = compressor_op.run();

    auto result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
      return false;
    }

    layout_a = SparseConfig::fill_layoutA(problem_shape_MNKL);
    layout_e = SparseConfig::fill_layoutE(problem_shape_MNKL);

    tensor_E.sync_host();
    tensor_A_Comp.sync_host();

    using namespace cute;
    auto k_blks = cutlass::ceil_div(K, size<1>(shape(SfAtom{})));
    auto m_blks = cutlass::ceil_div(M, Blk_MN{});
    auto n_blks = cutlass::ceil_div(N, Blk_MN{});
    layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
    layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto sfa_coord   = cutlass::make_Coord(m_blks * Blk_MN{} * L, k_blks * Blk_SF{});
    auto sfb_coord   = cutlass::make_Coord(n_blks * Blk_MN{} * L, k_blks * Blk_SF{});

    tensor_SFA.resize(sfa_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(sfa_coord, stride_factor_A));
    tensor_SFB.resize(sfb_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(sfb_coord, stride_factor_B));

    EXPECT_TRUE(initialize_tensor(tensor_SFA.host_view(), init_A, seed + 2024));
    EXPECT_TRUE(initialize_tensor(tensor_SFB.host_view(), init_B, seed + 2025));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_SFA.host_view().at({0, 0}) = ElementSF(1);
    tensor_SFB.host_view().at({0, 0}) = ElementSF(1);

    tensor_SFA.sync_device();
    tensor_SFB.sync_device();

    return true;
  }

  Arguments to_args() {
    using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
    return {
      reinterpret_cast<ArrayElementA *>(tensor_A_Comp.device_data()), layout_a,
      reinterpret_cast<ArrayElementB *>(tensor_B.device_data()), stride_b,
      tensor_E.device_data(), layout_e,
      tensor_SFA.device_data(), layout_sfa,
      tensor_SFB.device_data(), layout_sfb
    };
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);
    auto A = make_tensor(make_iterator(tensor_A.host_data()),
          make_layout(make_shape(M, K, L), stride_a));
    auto SfA = make_tensor(tensor_SFA.host_data(), layout_sfa);

    auto B = make_tensor(make_iterator(tensor_B.host_data()),
        make_layout(make_shape(N, K, L), stride_b));
    auto SfB = make_tensor(tensor_SFB.host_data(), layout_sfb);

    // return {A, SfA, B, SfB};
    cutlass::reference::host::GettMainloopParams<ElementAccumulator,
        decltype(A),
        decltype(B),
        decltype(SfA),
        decltype(SfB)
      >
          mainloop_params{A, SfA, B, SfB};
    return mainloop_params;
  }

  void print_tensors(std::ofstream& file) {
    file << "A =\n" << tensor_A.host_view()
         << "\nB =\n" << tensor_B.host_view()
         << "\nSFA =\n" << tensor_SFA.host_view()
         << "\nSFB =\n" << tensor_SFB.host_view();
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL) {
    auto [M, N, K, L] = problem_shape_MNKL;

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_SFA.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_SFB.host_view()), 0);
    return true;
  }
};

template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120<SchedulerPipelineStageCount_, true>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedBlockScaledSm100<0,0>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedBlockScaledSm100<0,0>,
                                      Gemm, ElementA_, ElementB_>;
  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = Base::kDefaultSeed,
    typename Base::LayoutTagA::Stride stride_factor_A_ = typename Base::LayoutTagA::Stride(),
    typename Base::LayoutTagB::Stride stride_factor_B_ = typename Base::LayoutTagB::Stride(),
    typename Base::LayoutTagE::Stride stride_factor_E_ = typename Base::LayoutTagE::Stride()
  ) : Base::HostCollectiveMainloop(check_relative_equality_, init_A_, init_B_, seed_, stride_factor_A_,
                                                                                      stride_factor_B_,
                                                                                      stride_factor_E_) {}
};

template<
  class Gemm,
  int SchedulerPipelineStageCount_,
  class ElementA_,
  class ElementB_
>
struct HostCollectiveMainloop<cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120<SchedulerPipelineStageCount_, false>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedBlockScaledSm100<0,0>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelSparseTmaWarpSpecializedBlockScaledSm100<0,0>,
                                      Gemm, ElementA_, ElementB_>;
  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = Base::kDefaultSeed,
    typename Base::LayoutTagA::Stride stride_factor_A_ = typename Base::LayoutTagA::Stride(),
    typename Base::LayoutTagB::Stride stride_factor_B_ = typename Base::LayoutTagB::Stride(),
    typename Base::LayoutTagE::Stride stride_factor_E_ = typename Base::LayoutTagE::Stride()
  ) : Base::HostCollectiveMainloop(check_relative_equality_, init_A_, init_B_, seed_, stride_factor_A_,
                                                                                      stride_factor_B_,
                                                                                      stride_factor_E_) {}
};

template<class Gemm>
struct HostCollectiveDefaultEpilogue {
  // fusion types are potentially void if the fusion is not supported
  // helper so we don't try to construct HostTensor with void type
  template <typename T, typename U = uint8_t>
  using non_void_t = cute::conditional_t<cute::is_void_v<T>, U, T>;

  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using kernel   = typename Gemm::GemmKernel;
  using Epilogue = typename kernel::CollectiveEpilogue;

  using ElementD = typename kernel::ElementD;
  using StrideD  = typename kernel::StrideD;
  using ElementC = non_void_t<typename kernel::ElementC, ElementD>;
  using StrideC  = typename kernel::StrideC;

  using FusionOp = typename Gemm::EpilogueOutputOp;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static_assert(is_row_or_col_major<StrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideD>(),
    "ERROR : D Layout is neither Row / Column Major)");

  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagC = cutlass::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::detail::StrideToLayoutTagC_t<StrideD>;
  using LayoutTagScalar = cutlass::layout::PackedVectorLayout; // scalars are size-1 vectors
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  using ElementAccumulator = typename kernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename kernel::ProblemShape;
  using ElementCompute = typename ElementComputeType<Gemm, ElementAccumulator>::Type;
  using ElementScalar = typename ElementScalarType<Gemm, ElementCompute>::Type;

  using Arguments = typename Gemm::GemmKernel::EpilogueArguments;

  /// Initialization
  StrideC stride_c;
  StrideD stride_d;

  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;

  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  // Inputs
  ElementScalar alpha;
  ElementScalar beta;

  cutlass::HostTensor<ElementD, LayoutTagD> tensor_D;
  cutlass::HostTensor<ElementD, LayoutTagD> reference_D;

  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;
  // Are scalars copied to device memory before kernel launch
  ScalarLoc use_device_scalars = ScalarLoc::ON_HOST;
  // If per-row scale is enabled and this is disabled, alpha/beta are passed as a host or device scalar instead of device vector
  VectorScale vector_scale_mode = VectorScale::DISABLED;

  cutlass::Distribution::Kind init_C;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  HostCollectiveDefaultEpilogue(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorScale vector_scale_mode_ = VectorScale::DISABLED,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): init_C(init_C_), seed(seed_),
     stride_factor_C(typename LayoutTagC::Stride()),
     stride_factor_D(typename LayoutTagD::Stride()),
     check_relative_equality(check_relative_equality_),
     use_device_scalars(use_device_scalars_){ }

  bool initialize(ProblemShapeType problem_size, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("HostCollectiveDefaultEpilogue::initialize(problem_size, alpha, beta)");
#endif
    // Initialize Epilogue tensors
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto c_coord = cutlass::make_Coord(M * L, N);
    try {
      tensor_C.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
      tensor_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
      reference_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveDefaultEpilogue::initialize: resizing tensors threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveDefaultEpilogue::initialize: resizing tensors threw an unknown exception");
      throw;
    }
    {
      const bool init_succeeded = initialize_tensor(tensor_C.host_view(), init_C, seed + 2020);
      if (not init_succeeded) {
        CUTLASS_TRACE_HOST("HostCollectiveDefaultEpilogue::initialize: initialize_tensor returned false");
      }
      EXPECT_TRUE(init_succeeded);
    }
    tensor_C.host_view().at({0, 0}) = ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    try {
      tensor_C.sync_device();
      tensor_D.sync_device();
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveDefaultEpilogue::initialize: sync_device() threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveDefaultEpilogue::initialize: sync_device() threw an unknown exception");
      throw;
    }

    alpha = alpha_;
    beta = beta_;

    return true;
  }

  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {

    // Factors used for calculating relative equality. CUTLASS's relative-equality
    // checks in include/cutlass/relatively_equal.h  are inspired by
    // https://floating-point-gui.de/errors/comparison/. This reference suggests using
    // the minimum normal value of a given type as the nonzero_floor.
    Element epsilon(static_cast<Element>(0.1f));
    Element nonzero_floor(std::numeric_limits<Element>::min());

    if constexpr (!cutlass::is_complex<Element>::value) {
      if (check_relative_equality == CheckEquality::RELATIVE) {
        return cutlass::reference::host::TensorRelativelyEquals(
          lhs, rhs, epsilon, nonzero_floor);
      }
      else {
        return cutlass::reference::host::TensorEquals(lhs, rhs);
      }
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta) {
    auto [M, N, K, L] = problem_shape_MNKL;

    tensor_D.sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    }

    if (reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    }

    bool passed = equality_check(reference_D.host_view(), tensor_D.host_view());
    if(!passed) {
      std::cout<<"D is incorrect"<<std::endl;
    }
    return passed;
  }

  void print_tensors(std::ofstream& file) {
    file
    << "\nC =\n" << tensor_C.host_view()
    << "\n\nReference =\n" << reference_D.host_view()
    << "\n\nComputed =\n" << tensor_D.host_view();
  }

  Arguments to_args(ProblemShapeType problem_size) {
    Arguments arguments =
      {
        {alpha, beta},
        tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
      };

    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);
    auto coord_0 = cutlass::make_Coord(0);
    auto C = cute::make_tensor(detail::make_iterator(tensor_C.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_c));
    auto D = cute::make_tensor(detail::make_iterator(reference_D.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));

    cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D)>
        epilogue_params{};

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha;
    epilogue_params.beta = beta;

    return epilogue_params;
  }
};

template<class Gemm>
struct HostCollectiveEpilogue {
  // fusion types are potentially void if the fusion is not supported
  // helper so we don't try to construct HostTensor with void type
  template <typename T, typename U = uint8_t>
  using non_void_t = cute::conditional_t<cute::is_void_v<T>, U, T>;

  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using kernel   = typename Gemm::GemmKernel;
  using Epilogue = typename kernel::CollectiveEpilogue;
  static_assert(IsDefaultEpilogue<Epilogue>::value == false, "Default Epilogue is not supported");

  using ElementD = typename kernel::ElementD;
  using StrideD  = typename kernel::StrideD;
  using ElementC = non_void_t<typename kernel::ElementC, ElementD>;
  using StrideC  = typename kernel::StrideC;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static_assert(is_row_or_col_major<StrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideD>(),
    "ERROR : D Layout is neither Row / Column Major)");

  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagC = cutlass::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::detail::StrideToLayoutTagC_t<StrideD>;
  using LayoutTagScalar = cutlass::layout::PackedVectorLayout; // scalars are size-1 vectors
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  using ElementAccumulator = typename kernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename kernel::ProblemShape;

  //
  // FusionOperation derived types/queries
  //
  static constexpr bool IsLegacy = detail::IsLegacyEpiloguePolicy<Epilogue>::value;

  // FFMA2 SGEMM uses ThreadEpilogueOp for bias and relu support instead of FusionOp, so we compose LinCombPerRowBiasEltAct FusionOp by hand to test the functionality.
  static constexpr bool IsFfma2Kernel = cute::is_same_v<ScheduleType, cutlass::gemm::KernelMultistage>;
  using FusionOp = cute::conditional_t<IsFfma2Kernel,
                                       cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<cutlass::epilogue::thread::Clamp, float, float>,
                                       typename Gemm::EpilogueOutputOp>;
  static_assert(cute::is_base_of_v<cutlass::epilogue::fusion::FusionOperation, FusionOp>);


  // Scale factor Generation related
  using SfStrategy = cutlass::reference::host::SfStrategy;
  static constexpr bool IsBlockScaleSupported            = FusionOp::IsBlockScaleSupported;
  static constexpr SfStrategy SfGenStrategy              = (!IsBlockScaleSupported) ? SfStrategy::None : SfStrategy::SfDGen;
  static constexpr int32_t SFD_VectorSize = IsBlockScaleSupported ? FusionOp::SFVecSize : 1;
  static constexpr bool IsKMajorSFD = cute::is_same_v<typename FusionOp::GmemLayoutTagScalefactor, cutlass::layout::RowMajor>;
  using ElementSFD = non_void_t<typename FusionOp::ElementBlockScaleFactor, ElementD>;
  using Sm1xxBlockScaledOutputConfig= cutlass::detail::Sm1xxBlockScaledOutputConfig<SFD_VectorSize,
                                        IsKMajorSFD ? cute::UMMA::Major::K : cute::UMMA::Major::MN>;
  using Blk_MN = typename Sm1xxBlockScaledOutputConfig::Blk_MN;
  using Blk_SF = typename Sm1xxBlockScaledOutputConfig::Blk_SF;
  using OutputSFAtom = typename Sm1xxBlockScaledOutputConfig::SfAtom;
  cutlass::HostTensor<ElementSFD, LayoutTagD> tensor_SFD;
  cutlass::HostTensor<ElementSFD, LayoutTagD> reference_SFD;

  using ElementCompute    = typename FusionOp::ElementCompute;
  using ElementScalar     = typename FusionOp::ElementScalar;
  using ElementBias       = non_void_t<typename FusionOp::ElementBias>;
  using ElementAux        = non_void_t<typename FusionOp::ElementAux>;
  using ElementAmax       = non_void_t<typename FusionOp::ElementAmax>;
  using LayoutTagAux      = non_void_t<typename FusionOp::GmemLayoutTagAux, LayoutTagD>;
  using ActivationFunctor = non_void_t<typename FusionOp::ActivationFn,
                              cutlass::epilogue::thread::Identity<ElementCompute>>;

  static constexpr bool IsRowBiasEnabled        = FusionOp::IsPerRowBiasSupported;
  static constexpr bool IsColBiasEnabled        = FusionOp::IsPerColBiasSupported;
  static_assert(not (IsColBiasEnabled && IsRowBiasEnabled));

  static constexpr bool IsDeBiasEnabled      = FusionOp::IsDePerRowBiasSupported;
  static constexpr bool IsPerRowScaleEnabled = FusionOp::IsPerRowScaleSupported;
  static constexpr bool IsPerColScaleEnabled = FusionOp::IsPerColScaleSupported;
  static constexpr bool IsScaleFactorEnabled = FusionOp::IsScaleFactorSupported;
  static constexpr bool IsAuxInEnabled       = FusionOp::IsAuxInSupported;
  static constexpr bool IsAuxOutEnabled      = FusionOp::IsAuxOutSupported;
  static constexpr bool IsAbsMaxEnabledD     = FusionOp::IsAbsMaxSupported &&
                                                (cute::is_same_v<ElementD, cutlass::float_e4m3_t> ||
                                                 cute::is_same_v<ElementD, cutlass::float_e5m2_t>);
  static constexpr bool IsAbsMaxEnabledAux   = IsAuxOutEnabled && FusionOp::IsAbsMaxSupported &&
                                                (cute::is_same_v<ElementAux, cutlass::float_e4m3_t> ||
                                                 cute::is_same_v<ElementAux, cutlass::float_e5m2_t>);
  using Arguments = typename Gemm::GemmKernel::EpilogueArguments;

  /// Initialization
  StrideC stride_c;
  StrideD stride_d;

  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;

  // Inputs
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> alpha;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> beta;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_A;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_B;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_C;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_D;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_Aux;
  cutlass::HostTensor<ElementBias  , LayoutTagVector> bias;
  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  cutlass::HostTensor<ElementCompute, LayoutTagScalar> norm_constant;

  // Outputs
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_D;
  cutlass::HostTensor<ElementAux , LayoutTagAux   > tensor_Aux;
  cutlass::gemm::TagToStrideC_t<   LayoutTagAux   > stride_Aux;
  cutlass::HostTensor<ElementD, LayoutTagD> tensor_D;
  cutlass::HostTensor<ElementD, LayoutTagD> reference_D;

  // References
  cutlass::HostTensor<ElementBias, LayoutTagVector> reference_dbias;
  cutlass::HostTensor<ElementAux , LayoutTagAux   > reference_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_D;

  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;
  // Are scalars copied to device memory before kernel launch
  ScalarLoc use_device_scalars = ScalarLoc::ON_HOST;
  // If vector scale is supported and this is disabled, alpha/beta are passed as a host or device scalar instead of device vector
  VectorScale vector_scale_mode = VectorScale::DISABLED;

  // Random distribution with which to initialize the A/B/C/D/Aux scaling factors
  cutlass::Distribution::Kind init_scale = cutlass::Distribution::Uniform;
  // Random distribution with which to initialize the bias vector
  cutlass::Distribution::Kind init_bias = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  HostCollectiveEpilogue(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorScale vector_scale_mode_ = VectorScale::DISABLED,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): init_scale(init_scale_), init_bias(init_bias_),
     init_C(init_C_), seed(seed_),
     stride_factor_C(typename LayoutTagC::Stride()),
     stride_factor_D(typename LayoutTagD::Stride()),
     check_relative_equality(check_relative_equality_),
     use_device_scalars(use_device_scalars_){ }

  bool initialize(ProblemShapeType problem_size, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize(problem_size, alpha, beta)");
#endif
    // Initialize Epilogue tensors
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto c_coord = cutlass::make_Coord(M * L, N);
    try {
      tensor_C.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
      tensor_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
      reference_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize: resizing tensors threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize: resizing tensors threw an unknown exception");
      throw;
    }

    try {
      const bool initialize_tensor_C_succeeded =
        initialize_tensor(tensor_C.host_view(), init_C, seed + 2020);
      if (not initialize_tensor_C_succeeded) {
        CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize: initialize_tensor returned false");
      }
      EXPECT_TRUE(initialize_tensor_C_succeeded);
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize: initialize_tensor threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize: initialize_tensor threw an unknown exception");
      throw;
    }

    tensor_C.host_view().at({0, 0}) = ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());
    try {
      tensor_C.sync_device();
      tensor_D.sync_device();
    }
    catch (std::exception const& e) {
      CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize: sync_device() threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("HostCollectiveEpilogue::initialize: sync_device() threw an unknown exception");
      throw;
    }

    auto scalar_coord = cutlass::make_Coord(1);
    auto col_vector_coord = cutlass::make_Coord(M);
    auto row_vector_coord = cutlass::make_Coord(N);
    auto batch_vector_coord = cutlass::make_Coord(L);
    if constexpr (IsPerRowScaleEnabled or IsPerColScaleEnabled) {
      // scalars
      if (vector_scale_mode == VectorScale::DISABLED) {
        // batched scalars
        if (use_device_scalars == ScalarLoc::ON_DEVICE) {
          alpha.resize(batch_vector_coord, true);
          beta.resize(batch_vector_coord, true);
          EXPECT_TRUE(initialize_tensor(alpha.host_view(), init_scale, seed + 2023));
          if (beta_ != ElementScalar(0)) {
            EXPECT_TRUE(initialize_tensor(beta.host_view(), init_scale, seed + 2024));
          }
          else {
            cutlass::reference::host::TensorFill(beta.host_view(), beta_);
          }
        }
        // non-batched scalars
        else {
          alpha.resize(scalar_coord, false);
          beta.resize(scalar_coord, false);
          cutlass::reference::host::TensorFill(alpha.host_view(), alpha_);
          cutlass::reference::host::TensorFill(beta.host_view(), beta_);
        }
      }
      // batched vectors
      else {
        auto batched_vector_coord = cutlass::make_Coord((IsPerRowScaleEnabled ? M : N) * L);
        alpha.resize(batched_vector_coord, true);
        beta.resize(batched_vector_coord, true);
        EXPECT_TRUE(initialize_tensor(alpha.host_view(), init_scale, seed + 2023));
        if (beta_ != ElementScalar(0)) {
          EXPECT_TRUE(initialize_tensor(beta.host_view(), init_scale, seed + 2024));
        }
        else {
          cutlass::reference::host::TensorFill(beta.host_view(), beta_);
        }
      }
    }
    else {
      if (use_device_scalars == ScalarLoc::ON_DEVICE) {
        // Set alpha  beta for different batches.
        alpha.resize(batch_vector_coord, true);
        beta.resize(batch_vector_coord, true);
        cutlass::reference::host::TensorFill(alpha.host_view(), alpha_);
        for (int l = 0; l < L; ++l) {
          beta.host_view().at(cutlass::make_Coord(l)) = beta_ + ElementScalar(l);
        }
      }
      else {
        alpha.resize(scalar_coord, false);
        beta.resize(scalar_coord, false);
        cutlass::reference::host::TensorFill(alpha.host_view(), alpha_);
        cutlass::reference::host::TensorFill(beta.host_view(), beta_);
      }
    }
    alpha.sync_device();
    beta.sync_device();

    if constexpr (IsScaleFactorEnabled) {
      scale_A.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      scale_B.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      scale_C.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      scale_D.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      EXPECT_TRUE(initialize_tensor(scale_A.host_view(), init_scale, seed + 2023));
      EXPECT_TRUE(initialize_tensor(scale_B.host_view(), init_scale, seed + 2024));
      EXPECT_TRUE(initialize_tensor(scale_C.host_view(), init_scale, seed + 2025));
      EXPECT_TRUE(initialize_tensor(scale_D.host_view(), init_scale, seed + 2026));
      scale_A.sync_device();
      scale_B.sync_device();
      scale_C.sync_device();
      scale_D.sync_device();
    }

    if constexpr (IsRowBiasEnabled or IsColBiasEnabled) {
      bias.resize(IsRowBiasEnabled ? col_vector_coord : row_vector_coord);
      EXPECT_TRUE(initialize_tensor(bias.host_view(), init_bias, seed + 2023));
      bias.sync_device();
    }

    if constexpr (IsDeBiasEnabled) {
      bias.resize(col_vector_coord);
      reference_dbias.resize(col_vector_coord);
      cutlass::reference::host::TensorFill(bias.host_view(), ElementBias(0));
      cutlass::reference::host::TensorFill(reference_dbias.host_view(), ElementBias(0));
      bias.sync_device();
    }

    if constexpr (IsAbsMaxEnabledD) {
      abs_max_D.resize(scalar_coord);
      // ensure in-place device reductions perform their own initialization
      cutlass::reference::host::TensorFill(abs_max_D.host_view(),
                                           CUTLASS_STL_NAMESPACE::numeric_limits<ElementAmax>::max());
      abs_max_D.sync_device();
      reference_abs_max_D.resize(scalar_coord);
      cutlass::reference::host::TensorFill(reference_abs_max_D.host_view(), ElementAmax(0));
    }

    if constexpr (IsAuxInEnabled) {
      auto aux_coord = cutlass::make_Coord(M * L, N);
      auto aux_layout = cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(aux_coord, typename LayoutTagAux::Stride{});
      tensor_Aux.resize(aux_coord, aux_layout);
      EXPECT_TRUE(initialize_tensor(tensor_Aux.host_view(), init_C, seed + 2023));
      tensor_Aux.sync_device();
      stride_Aux = cutlass::make_cute_packed_stride(cutlass::gemm::TagToStrideC_t<LayoutTagAux>{}, cute::make_shape(M, N, L));
    }

    if constexpr (IsAuxOutEnabled) {
      auto aux_coord = cutlass::make_Coord(M * L, N);
      auto aux_layout = cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(aux_coord, typename LayoutTagAux::Stride{});
      tensor_Aux.resize(aux_coord, aux_layout);
      reference_Aux.resize(aux_coord, aux_layout, false);
      tensor_Aux.sync_device();
      stride_Aux = cutlass::make_cute_packed_stride(cutlass::gemm::TagToStrideC_t<LayoutTagAux>{}, cute::make_shape(M, N, L));

      if constexpr (IsScaleFactorEnabled) {
        scale_Aux.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
        EXPECT_TRUE(initialize_tensor(scale_Aux.host_view(), init_scale, seed + 2027));
        scale_Aux.sync_device();
      }

      if constexpr (IsAbsMaxEnabledAux) {
        abs_max_Aux.resize(scalar_coord);
        // ensure in-place device reductions perform their own initialization
        cutlass::reference::host::TensorFill(abs_max_Aux.host_view(),
                                             CUTLASS_STL_NAMESPACE::numeric_limits<ElementAmax>::max());
        abs_max_Aux.sync_device();
        reference_abs_max_Aux.resize(scalar_coord);
        cutlass::reference::host::TensorFill(reference_abs_max_Aux.host_view(), ElementAmax(0));
      }
    }


    if constexpr (IsBlockScaleSupported) {
      auto m_blks = cutlass::ceil_div(M, cute::size<0>(cute::shape(OutputSFAtom{})));
      auto n_blks = cutlass::ceil_div(N, cute::size<1>(cute::shape(OutputSFAtom{})));
      auto sfd_coord = [&] () {
        if constexpr (IsKMajorSFD) {
          return cutlass::make_Coord(m_blks * Blk_MN{} * L, n_blks * Blk_SF{});
        }
        else {
          return cutlass::make_Coord(m_blks * Blk_SF{} * L, n_blks * Blk_MN{});
        }
      }();
      tensor_SFD.resize(sfd_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(sfd_coord, stride_factor_D));
      reference_SFD.resize(sfd_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(sfd_coord, stride_factor_D), false);
      tensor_SFD.sync_device();
      norm_constant.resize(scalar_coord, true);
      EXPECT_TRUE(initialize_tensor(norm_constant.host_view(), init_scale, seed + 2023));
      norm_constant.sync_device();
    }


    return true;
  }

  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {

    // Factors used for calculating relative equality. CUTLASS's relative-equality
    // checks in include/cutlass/relatively_equal.h  are inspired by
    // https://floating-point-gui.de/errors/comparison/. This reference suggests using
    // the minimum normal value of a given type as the nonzero_floor.
    Element epsilon(static_cast<Element>(0.1f));
    Element nonzero_floor(std::numeric_limits<Element>::min());

    if constexpr (!cutlass::is_complex<Element>::value) {
      if (check_relative_equality == CheckEquality::RELATIVE) {
        return cutlass::reference::host::TensorRelativelyEquals(
          lhs, rhs, epsilon, nonzero_floor);
      }
      else {
        return cutlass::reference::host::TensorEquals(lhs, rhs);
      }
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta) {
    tensor_D.sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    }

    if (reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    }

    bool passed = equality_check(reference_D.host_view(), tensor_D.host_view());
    if(!passed) {
      #if 0
      auto [M, N, K, L] = problem_shape_MNKL;
      auto ref = cute::make_tensor(detail::make_iterator(reference_D.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));
      auto comp = cute::make_tensor(detail::make_iterator(tensor_D.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));
      for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
          for(int l=0; l<L; l++) {
            if(static_cast<float>(ElementD(ref(i, j, l))) != static_cast<float>((ElementD(comp(i, j, l))))) {
              printf("<m %d, n %d, l %d> ref: %f comp: %f\n", i, j, l, static_cast<float>(ElementD(ref(i, j, l))), static_cast<float>((ElementD(comp(i, j, l)))));
            }
          }
        }
      }
      #endif
      std::cout<<"D is incorrect"<<std::endl;
    }

    if constexpr (IsAbsMaxEnabledD) {
      abs_max_D.sync_host();
      passed &= equality_check(reference_abs_max_D.host_view(), abs_max_D.host_view());
    }

    if constexpr (IsDeBiasEnabled) {
      bias.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(bias.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_dbias.host_view()), 0);
      passed &= equality_check(reference_dbias.host_view(), bias.host_view());
    }

    if constexpr (IsAuxOutEnabled) {
      tensor_Aux.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Aux.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_Aux.host_view()), 0);
      passed &= equality_check(reference_Aux.host_view(), tensor_Aux.host_view());
      if(!passed) {
        std::cout<<"Aux is incorrect"<<std::endl;
      }
      if constexpr (IsAbsMaxEnabledAux) {
        abs_max_Aux.sync_host();
        bool tmp =  equality_check(reference_abs_max_Aux.host_view(), abs_max_Aux.host_view());
        if(!tmp) {
          std::cout<<"AbsMax of Aux is incorrect"<<std::endl;
        }
        passed &= tmp;
      }
    }


    if constexpr (IsBlockScaleSupported) {
      tensor_SFD.sync_host();
      bool passed_sf = equality_check(reference_SFD.host_view(), tensor_SFD.host_view());
      if(!passed_sf) {
        std::cout<<"SF is incorrect"<<std::endl;
      }
      passed &= passed_sf;
    }

    return passed;
  }

  void print_tensors(std::ofstream& file) {
    auto coord_0 = cutlass::make_Coord(0);
    if constexpr (IsScaleFactorEnabled) {
      file
        << ", scale_a: " << scale_A.at(coord_0)
        << ", scale_b: " << scale_B.at(coord_0)
        << ", scale_c: " << scale_C.at(coord_0);
    }
    if constexpr (IsPerRowScaleEnabled or IsPerColScaleEnabled) {
      file << "\n\nvalpha = \n" << alpha.host_view();
      file << "\n\nvbeta = \n" << beta.host_view();
    } else {
      file
        << "\n\nalpha= \n" << alpha.host_view()
        << "\n\nbeta= \n " << beta.host_view();
    }
    file << "\n\n";

    if constexpr (IsAbsMaxEnabledD) {
      file << "scale_d: " << float(scale_D.at(coord_0));
      file << "\nReference abs_max_D :";
      file << " " << float(reference_abs_max_D.at(coord_0));

      file << "\nComputed abs_max_D :";
      file << " " << float(abs_max_D.at(coord_0));
      file << "\n\n";
    }

    if constexpr (IsAbsMaxEnabledAux) {
      file << "scale_aux: " << float(scale_Aux.at(coord_0));
      file << "\nReference abs_max_Aux :";
      file << " " << float(reference_abs_max_Aux.at(coord_0));

      file << "\nComputed abs_max_Aux :";
      file << " " << float(abs_max_Aux.at(coord_0));
      file << "\n\n";
    }

    if constexpr (IsRowBiasEnabled or IsColBiasEnabled) {
      file << "\n\nBias = \n" << bias.host_view();
    }

    if constexpr (IsAuxInEnabled) {
      file << "\n\nAux Input = \n" << tensor_Aux.host_view();
    }

    if constexpr (IsDeBiasEnabled) {
      file << "\n\nReference dBias = \n" << reference_dbias.host_view();
      file << "\n\nComputed dBias = \n" << bias.host_view();
    }

    if constexpr (IsAuxOutEnabled) {
      file
        << "\n\nReference Aux =\n" << reference_Aux.host_view()
        << "\n\nComputed Aux =\n" << tensor_Aux.host_view();
    }

    if constexpr (IsBlockScaleSupported) {
      file
        << "\n\nSFD Reference =\n" << reference_SFD.host_view()
        << "\n\nSFD Computed =\n" << tensor_SFD.host_view();
    }

    file
    << "\nC =\n" << tensor_C.host_view()
    << "\n\nReference =\n" << reference_D.host_view()
    << "\n\nComputed =\n" << tensor_D.host_view();

  }

  Arguments to_args(ProblemShapeType problem_size) {
    auto coord_0 = cutlass::make_Coord(0);
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    Arguments arguments =
      {
        {},
        tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
      };

    auto &fusion_args = arguments.thread;
    if constexpr (IsLegacy) {
      arguments.thread = {
        alpha.at(coord_0),
        beta.at(coord_0),
        alpha.device_data(),
        beta.device_data()
      };
      arguments.ptr_Bias = bias.device_data();
      arguments.ptr_T = tensor_Aux.device_data();
    }
    else {
      fusion_args.alpha = alpha.at(coord_0);
      fusion_args.alpha_ptr = alpha.device_data();
      // Only initializing beta/beta_ptr for non-void source
      if constexpr (not cute::is_void_v<typename kernel::ElementC>) {
        fusion_args.beta = beta.at(coord_0);
        fusion_args.beta_ptr = beta.device_data(); // if vector_scale_mode is true this is nullptr
      }

      if constexpr (IsPerRowScaleEnabled) {
        int32_t m_stride = vector_scale_mode == VectorScale::ENABLED ? 1 : 0;
        int64_t l_stride = vector_scale_mode == VectorScale::ENABLED ? M : (use_device_scalars == ScalarLoc::ON_DEVICE ? 1 : 0);
        fusion_args.dAlpha = cute::make_stride(bool(m_stride),cute::_0{}, l_stride);
        fusion_args.dBeta = cute::make_stride(bool(m_stride),cute::_0{}, l_stride);
      }
      else if constexpr (IsPerColScaleEnabled) {
        int32_t n_stride = vector_scale_mode == VectorScale::ENABLED ? 1 : 0;
        int64_t l_stride = vector_scale_mode == VectorScale::ENABLED ? N : (use_device_scalars == ScalarLoc::ON_DEVICE ? 1 : 0);
        fusion_args.dAlpha = cute::make_stride(cute::_0{}, bool(n_stride), l_stride);
        fusion_args.dBeta = cute::make_stride(cute::_0{}, bool(n_stride), l_stride);
      }
      else {
        if constexpr (not IsFfma2Kernel) {
          if (use_device_scalars == ScalarLoc::ON_DEVICE) {
            if (L > 1) {
              fusion_args.dAlpha = cute::make_stride(cute::_0{},cute::_0{}, int64_t(1));
              fusion_args.dBeta  = cute::make_stride(cute::_0{},cute::_0{}, int64_t(1));
            }
          }
        }
      }

      if constexpr (IsScaleFactorEnabled) {
        fusion_args.scale_a = scale_A.at(coord_0);
        fusion_args.scale_b = scale_B.at(coord_0);
        fusion_args.scale_c = scale_C.at(coord_0);
        fusion_args.scale_d = scale_D.at(coord_0);
        fusion_args.scale_a_ptr = scale_A.device_data();
        fusion_args.scale_b_ptr = scale_B.device_data();
        fusion_args.scale_c_ptr = scale_C.device_data();
        fusion_args.scale_d_ptr = scale_D.device_data();
      }

      if constexpr (IsRowBiasEnabled or IsColBiasEnabled) {
        fusion_args.bias_ptr = bias.device_data();
      }

      if constexpr (IsDeBiasEnabled) {
        fusion_args.dbias_ptr = bias.device_data();
      }

      // example of how to set kernel activation arguments
      // see ActivationFunctor::Arguments in activation.h for definition
      // if Arguments doesn't exist then fusion_args.activation is empty
      auto init_activation_args = [] (auto activation, auto& args) {
        using Activation = cute::remove_cvref_t<decltype(activation)>;
        if constexpr (cute::is_same_v<Activation, cutlass::epilogue::thread::Clamp<ElementCompute>>) {
          args.lower_bound = 0; // Treat Clamp as ReLU
          args.upper_bound = cutlass::platform::identity_for_minimum<ElementCompute>();
        }
        if constexpr (cute::is_same_v<Activation, cutlass::epilogue::thread::ScaledGELU_taylor<ElementCompute>>) {
          args.scale = ElementCompute(1);
        }
      };

      if constexpr (not cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Identity<ElementCompute>>) {
        init_activation_args(ActivationFunctor{}, fusion_args.activation);
      }
      if constexpr (IsAbsMaxEnabledD) {
        fusion_args.amax_D_ptr = abs_max_D.device_data();
      }

      if constexpr (IsAuxInEnabled) {
        fusion_args.aux_ptr = tensor_Aux.device_data();
        fusion_args.dAux = stride_Aux;
      }

      if constexpr (IsAuxOutEnabled) {
        fusion_args.aux_ptr = tensor_Aux.device_data();
        fusion_args.dAux = stride_Aux;
        if constexpr (IsScaleFactorEnabled) {
          fusion_args.scale_aux = scale_Aux.at(coord_0);
          fusion_args.scale_aux_ptr = scale_Aux.device_data();
        }
        if constexpr (IsAbsMaxEnabledAux) {
          fusion_args.amax_aux_ptr = abs_max_Aux.device_data();
        }
      }


      if constexpr (IsBlockScaleSupported) {
        arguments.thread.block_scale_factor_ptr = tensor_SFD.device_data();
        arguments.thread.norm_constant_ptr = norm_constant.device_data();
      }
    }

    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);
    auto coord_0 = cutlass::make_Coord(0);
    auto C = cute::make_tensor(detail::make_iterator(tensor_C.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_c));
    auto D = cute::make_tensor(detail::make_iterator(reference_D.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));
    auto Bias = cute::make_tensor(detail::make_iterator(IsDeBiasEnabled ? reference_dbias.host_data() : bias.host_data()),
        cute::make_layout(cute::make_shape(IsRowBiasEnabled ? M : N)));
    auto Aux = cute::make_tensor(detail::make_iterator(IsAuxInEnabled ? tensor_Aux.host_data() : reference_Aux.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_Aux));
    auto Valpha = [&](){
      if constexpr (IsPerRowScaleEnabled) {
        int m_stride = vector_scale_mode == VectorScale::ENABLED ? 1 : 0;
        int l_stride = vector_scale_mode == VectorScale::ENABLED ? M : (use_device_scalars == ScalarLoc::ON_DEVICE ? 1 : 0);
        return cute::make_tensor(detail::make_iterator(alpha.host_data()),
            cute::make_layout(cute::make_shape(M, N, L), make_stride(m_stride, cute::_0{}, l_stride)));
      }
      else if constexpr (IsPerColScaleEnabled) {
        int n_stride = vector_scale_mode == VectorScale::ENABLED ? 1 : 0;
        int l_stride = vector_scale_mode == VectorScale::ENABLED ? N : (use_device_scalars == ScalarLoc::ON_DEVICE ? 1 : 0);
        return cute::make_tensor(detail::make_iterator(alpha.host_data()),
            cute::make_layout(cute::make_shape(M, N, L), make_stride(cute::_0{}, n_stride, l_stride)));
      }
      else {
        return cute::make_tensor(detail::make_iterator(alpha.host_data()),
            cute::make_layout(cute::make_shape(M, N, L), make_stride(cute::_0{}, cute::_0{}, cute::_1{})));
      }
    }();

    auto Vbeta = [&]() {
      if constexpr (IsPerRowScaleEnabled) {
        int m_stride = vector_scale_mode == VectorScale::ENABLED ? 1 : 0;
        int l_stride = vector_scale_mode == VectorScale::ENABLED ? M : (use_device_scalars == ScalarLoc::ON_DEVICE ? 1 : 0);
        return cute::make_tensor(detail::make_iterator(beta.host_data()),
            cute::make_layout(cute::make_shape(M, N, L), make_stride(m_stride, cute::_0{}, l_stride)));
      }
      else if constexpr (IsPerColScaleEnabled) {
        int n_stride = vector_scale_mode == VectorScale::ENABLED ? 1 : 0;
        int l_stride = vector_scale_mode == VectorScale::ENABLED ? N : (use_device_scalars == ScalarLoc::ON_DEVICE ? 1 : 0);
        return cute::make_tensor(detail::make_iterator(beta.host_data()),
            cute::make_layout(cute::make_shape(M, N, L), make_stride(cute::_0{}, n_stride, l_stride)));
      }
      else {
        return  cute::make_tensor(detail::make_iterator(beta.host_data()),
            cute::make_layout(cute::make_shape(M, N, L), make_stride(cute::_0{}, cute::_0{}, cute::_1{})));
      }
    }();

    auto SfD = [&](){
      if constexpr (IsBlockScaleSupported) {
        auto tensor = make_tensor(detail::make_iterator(reference_SFD.host_data()),
          Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape_MNKL));
        return tensor;
      }
      else {
        // Reference kernel has a logic to ignore scalefactor computation if we pass the tensor type same as output D tensor.
        return D;
      }
    }();
    cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      decltype(Bias),
      decltype(Aux),
      decltype(Valpha),
      decltype(Vbeta),
      ActivationFunctor,
      decltype(SfD),
      Int<SFD_VectorSize>,
      cutlass::plus<ElementCompute>,
      IsColBiasEnabled
      , SfGenStrategy
    > epilogue_params{};

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha.at(coord_0);
    epilogue_params.beta = beta.at(coord_0);

    if constexpr (IsScaleFactorEnabled) {
      epilogue_params.scale_a = scale_A.at(coord_0);
      epilogue_params.scale_b = scale_B.at(coord_0);
      epilogue_params.scale_c = scale_C.at(coord_0);
      epilogue_params.scale_d = scale_D.at(coord_0);
    }

    if constexpr (IsRowBiasEnabled or IsColBiasEnabled or IsDeBiasEnabled)
    {
      epilogue_params.Bias = Bias;
    }

    if constexpr (IsAbsMaxEnabledD) {
      epilogue_params.abs_max_D = reference_abs_max_D.host_data();
    }

    if constexpr (IsAuxInEnabled) {
      epilogue_params.Aux = Aux;
    }

    if constexpr (IsAuxOutEnabled) {
      epilogue_params.Aux = Aux;
      if constexpr (IsScaleFactorEnabled) {
        epilogue_params.scale_aux = scale_Aux.at(coord_0);
      }
      if constexpr (IsAbsMaxEnabledAux) {
        epilogue_params.abs_max_Aux = reference_abs_max_Aux.host_data();
      }
    }

    if constexpr (IsPerRowScaleEnabled or IsPerColScaleEnabled) {
      epilogue_params.Valpha = Valpha;
      if (vector_scale_mode == VectorScale::ENABLED) {
        epilogue_params.Vbeta = Vbeta;
      }
    }
    else {
      if (use_device_scalars == ScalarLoc::ON_DEVICE) {
        epilogue_params.Valpha = Valpha;
        epilogue_params.Vbeta = Vbeta;
      }
    }

    if constexpr (IsBlockScaleSupported) {
      epilogue_params.SfD = SfD;
      epilogue_params.st = norm_constant.at(coord_0);
    }
    return epilogue_params;
  }
};

template <
  typename Gemm,
  template <class T> class ActivationFunctor_ = cutlass::epilogue::thread::Identity,
  bool force_legacy_epilogue = false,
  typename ElementA = typename Gemm::GemmKernel::ElementA,
  typename ElementB = typename Gemm::GemmKernel::ElementB
  , typename RuntimeDatatypeA = void*
  , typename RuntimeDatatypeB = void*
>
struct TestbedImpl {
  // Kernel data types
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  // All Collective MMA operands are defined by HostCollectiveMainloopType based on the schedule type
  using HostCollectiveMainloopType = HostCollectiveMainloop<ScheduleType, Gemm, ElementA, ElementB>;

  using CollectiveEpilogue = cute::conditional_t<IsDefaultEpilogue<typename Gemm::GemmKernel::CollectiveEpilogue>::value || force_legacy_epilogue,
                                                HostCollectiveDefaultEpilogue<Gemm>,
                                                HostCollectiveEpilogue<Gemm>>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementCompute = typename ElementComputeType<Gemm, ElementAccumulator>::Type;
  using ElementScalar = typename ElementScalarType<Gemm, ElementCompute>::Type;

  using LayoutTagA = typename HostCollectiveMainloopType::LayoutTagA;
  using LayoutTagB = typename HostCollectiveMainloopType::LayoutTagB;
  using LayoutTagC = typename CollectiveEpilogue::LayoutTagC;
  using LayoutTagD = typename CollectiveEpilogue::LayoutTagD;


  using InternalElementA = typename Gemm::GemmKernel::ElementA;
  using InternalElementB = typename Gemm::GemmKernel::ElementB;
  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<InternalElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<InternalElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB in a GEMM kernel should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;


  uint32_t sm_count;
  // Used to force multi-wave tests for persistent kernel schedules
  constexpr static int MaxSmCount = 16;
  static constexpr uint64_t kDefaultSeed = 4096;
  static constexpr uint32_t mma_promotion_interval = 4;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

  HostCollectiveMainloopType collective_mma_inputs;
  CollectiveEpilogue collective_epilogue;

  //
  // Methods
  //

  TestbedImpl(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorScale vector_scale_mode_ = VectorScale::DISABLED,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): collective_mma_inputs(HostCollectiveMainloopType(check_relative_equality_, init_A_, init_B_, seed_)),
     collective_epilogue(CollectiveEpilogue(check_relative_equality_, use_device_scalars_, vector_scale_mode_, init_C_, init_scale_, init_bias_, seed_)) { }

  TestbedImpl(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorScale vector_scale_mode_ = VectorScale::DISABLED,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): collective_mma_inputs(HostCollectiveMainloopType(check_relative_equality_, stride_factor_A_, stride_factor_B_, init_A_, init_B_, seed_)),
     collective_epilogue(CollectiveEpilogue(check_relative_equality_, use_device_scalars_, vector_scale_mode_, init_C_, init_scale_, init_bias_, seed_)) { }

  /// Initializes data structures
  bool initialize(ProblemShapeType problem_size, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::initialize(problem_size, alpha, beta)");
#endif
    collective_mma_inputs.initialize(problem_size);
    collective_epilogue.initialize(problem_size, alpha_, beta_);

    return true;
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta)
  {
    auto [M, N, K, L] = problem_shape_MNKL;

    bool passed = collective_mma_inputs.compare_reference(problem_shape_MNKL);
    passed &= collective_epilogue.compare_reference(problem_shape_MNKL, alpha, beta);
    EXPECT_TRUE(passed);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << L
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      collective_mma_inputs.print_tensors(file);
      collective_epilogue.print_tensors(file);
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
      ProblemShapeType problem_size,
      ElementScalar alpha,
      ElementScalar beta)
  {
    using namespace cute;
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto mainloop_params = collective_mma_inputs.to_host_args(problem_size);
    auto epilogue_params = collective_epilogue.to_host_args(problem_size);

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    bool passed = compare_reference(problem_shape_MNKL, alpha, beta);
    return passed;
  }

	/// Determine if the CUDA device is sufficient to run the kernel
  bool sufficient() {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = static_cast<size_t>(Gemm::GemmKernel::SharedStorageSize);

    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    cudaDeviceProp properties;
    result = cudaGetDeviceProperties(&properties, device_idx);
    this->sm_count = properties.multiProcessorCount;

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      printf("failed due to smem_size\n");
      printf("hardware smem_size: %d, required smem_size: %d\n\n", int(properties.sharedMemPerBlockOptin), int(smem_size));
      return false;
    }

    return true;
  }

  bool profile(
    ProblemShapeType problem_size,
    int iterations,
    Gemm& gemm_op,
    typename Gemm::Arguments& arguments,
    cutlass::device_memory::allocation<uint8_t>& workspace) {
    int M = cute::size<0>(problem_size);
    int N = cute::size<1>(problem_size);
    int K = cute::size<2>(problem_size);
    int L = 1;
    if constexpr(cute::rank(ProblemShapeType{}) == 4) {
      L = cute::size<3>(problem_size);
    }


    cutlass::Status status;
    //
    // Run the GEMM
    //
    cudaError_t result;

    for (int iter = 0; iter < iterations; ++iter) {
      status = gemm_op(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        return false;
      }
    }

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    bool profiling = false,
    detail::Iterations iterations = detail::Iterations{},
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic,
    detail::MaxSwizzleSize max_swizzle = detail::MaxSwizzleSize{},
    detail::Splits splits = detail::Splits{},
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic
    , RuntimeDatatypeA runtime_input_datatype_a = {}
    , RuntimeDatatypeB runtime_input_datatype_b = {}
    )
  {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run");
#endif

    // Fail test if insufficient CUDA device
    if (!sufficient()) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: Test failed due to insufficient CUDA device");
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    else {
      CUTLASS_TRACE_HOST("TestbedImpl::run: sufficient() returned true");
    }
#endif

    try {
      const bool initialized = this->initialize(problem_size, alpha, beta);
      if (not initialized) {
        CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize returned false");
        std::cerr << "Initialization failed \n";
        return false;
      }
    }
    catch ([[maybe_unused]] std::exception const& e) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize threw an unknown exception");
      throw;
    }

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize() returned true");
#endif

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      this->sm_count = std::min(MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = this->sm_count;
    }
    else {
      this->sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = this->sm_count;
    }

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { static_cast<int>(splits), static_cast<int>(max_swizzle), raster_order, decomposition_mode };
    }
    else {
      scheduler_args = { static_cast<int>(max_swizzle), raster_order };
    }
    typename HostCollectiveMainloopType::Arguments mainloop_args;

    mainloop_args = collective_mma_inputs.to_args();


    if constexpr (IsRuntimeDataType) {
      mainloop_args.runtime_data_type_a = runtime_input_datatype_a;
      mainloop_args.runtime_data_type_b = runtime_input_datatype_b;
    }


    arguments =
    {
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      mainloop_args,
      collective_epilogue.to_args(problem_size),
      hw_info,
      scheduler_args
    };

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Creating gemm_op");
#endif
    Gemm gemm_op;

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling Gemm::get_workspace_size");
#endif
    size_t workspace_size = Gemm::get_workspace_size(arguments);
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Allocating workspace of size " << workspace_size);
#endif
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling gemm_op.can_implement");
#endif
    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      const auto error_str = cudaGetErrorString(error);
      CUTLASS_TRACE_HOST("TestbedImpl::run: cudaGetLastError() is " << error_str);
      std::cerr << "This test is not supported: " << error_str << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    if (profiling) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("TestbedImpl::run: Calling profile");
#endif
      return profile(problem_size, static_cast<int>(iterations), gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("TestbedImpl::run: Calling gemm_op.initialize");
#endif
      status = gemm_op.initialize(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        cudaError_t error = cudaGetLastError();
        const auto error_str = cudaGetErrorString(error);
        CUTLASS_TRACE_HOST("TestbedImpl::run: cudaGetLastError() is " << error_str);
      }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("TestbedImpl::run: Calling gemm_op.run");
#endif
      status = gemm_op.run();
      if (status != cutlass::Status::kSuccess) {
        cudaError_t error = cudaGetLastError();
        const auto error_str = cudaGetErrorString(error);
        CUTLASS_TRACE_HOST("TestbedImpl::run: cudaGetLastError() is " << error_str);
      }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("TestbedImpl::run: Calling cudaDeviceSynchronize");
#endif
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        CUTLASS_TRACE_HOST("TestbedImpl::run: cudaDeviceSynchronize reports non-success");
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }

      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

      //
      // Verify
      //
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("TestbedImpl::run: Calling this->verify");
#endif
      bool passed = this->verify(problem_size, alpha, beta);
      if (!passed) {
        CUTLASS_TRACE_HOST("TestbedImpl::run: this->verify FAILED");
        cudaError_t error = cudaGetLastError();
        const auto error_str = cudaGetErrorString(error);
        CUTLASS_TRACE_HOST("TestbedImpl::run: cudaGetLastError() is " << error_str);

        std::cout << "Error : Failed : with alpha: " << alpha << ", beta: " << beta
                  << "\n";
      }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      else {
        CUTLASS_TRACE_HOST("TestbedImpl::run: this->verify passed");
      }
#endif

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("TestbedImpl::run: Reached end");
#endif
      return passed;
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  template <class T> class ActivationFunctor = cutlass::epilogue::thread::Identity,
  bool force_legacy_epilogue = false,
  typename ElementA = typename Gemm::GemmKernel::ElementA,
  typename ElementB = typename Gemm::GemmKernel::ElementB
  , typename RuntimeDatatypeA = void*
  , typename RuntimeDatatypeB = void*
>
struct Testbed3x {

  using TestBedImpl = typename detail::TestbedImpl<
                        Gemm,
                        ActivationFunctor,
                        force_legacy_epilogue,
                        ElementA,
                        ElementB
                        , RuntimeDatatypeA
                        , RuntimeDatatypeB
                        >;
  using Kernel      = typename Gemm::GemmKernel;
  using Epilogue    = typename Gemm::GemmKernel::CollectiveEpilogue;

  using ElementAccumulator   = typename TestBedImpl::ElementAccumulator;
  using ElementCompute       = typename TestBedImpl::ElementCompute;
  using ElementScalar        = typename TestBedImpl::ElementScalar;

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

  // Detail Implementation
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3x(
      CheckEquality check_relative_equality_ = CheckEquality::EXACT,
      ScalarLoc use_device_scalars_ = ScalarLoc::ON_DEVICE,
      VectorScale vector_scale_mode_ = VectorScale::DISABLED,
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = TestBedImpl::kDefaultSeed)
      : impl_(check_relative_equality_, use_device_scalars_, vector_scale_mode_, init_A_, init_B_, init_C_, init_scale_, init_bias_, seed_) {}

  /// Executes one test
  bool run(
   typename TestBedImpl::ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic,
    detail::MaxSwizzleSize max_swizzle = detail::MaxSwizzleSize{},
    detail::Splits splits = detail::Splits{},
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic,
    bool profiling = false,
    detail::Iterations iterations = detail::Iterations{}
    , RuntimeDatatypeA runtime_input_datatype_a = {}
    , RuntimeDatatypeB runtime_input_datatype_b = {}
    )
  {
    return impl_.run(
        problem_size, alpha, beta, profiling, iterations, raster_order, max_swizzle, splits, decomposition_mode
        , runtime_input_datatype_a, runtime_input_datatype_b
        );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestGemmPerf3x(int iterations = 20) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalar = ElementAccumulator;
  bool passed = true;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;

  std::vector<int> problem_size_m = { 4608 };
  std::vector<int> problem_size_n = { 4608 };
  std::vector<int> problem_size_k = { 8192 };

  Testbed3x<Gemm> testbed;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        ProblemShapeType problem_size;
        if constexpr (cute::rank(ProblemShapeType{}) == 4) {
          problem_size = ProblemShapeType{m, n, k, /* l */ 1};
        }
        else {
          problem_size = ProblemShapeType{m, n, k};
        }

        passed = testbed.run(
          problem_size,
          cutlass::from_real<ElementScalar>(1),
          cutlass::from_real<ElementScalar>(0),
          RasterOrderOptions{}, detail::MaxSwizzleSize(1), detail::Splits{1}, DecompositionMode{},
          true, // profiling
          detail::Iterations{iterations});

        if (!passed) {
          return false;
        }
      }
    }
  }

  return true;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
//
template <
  typename Gemm,
  typename RuntimeDataTypeA,
  typename RuntimeDataTypeB,
  bool force_legacy_epilogue = false>
bool TestRuntimeDataTypeSmall(
  RuntimeDataTypeA runtime_input_datatype_a,
  RuntimeDataTypeB runtime_input_datatype_b,
  double alpha = 1.0, double beta = cute::is_same_v<typename Gemm::GemmKernel::ElementC, void> ? 0.0 : 1.0,
  CheckEquality check_relative_equality = CheckEquality::RELATIVE, ScalarLoc use_device_scalars = ScalarLoc::ON_DEVICE, VectorScale vector_scale_mode = VectorScale::ENABLED, std::vector<int> override_problem_size_k = {}) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using CtaShape_MNK = typename Gemm::GemmKernel::CollectiveMainloop::CtaShape_MNK;
  using DispatchPolicy = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy;

  using InternalElementA = typename Gemm::GemmKernel::ElementA;
  using InternalElementB = typename Gemm::GemmKernel::ElementB;

  CtaShape_MNK cta_shape;
  static constexpr int SmCount  = 16;
  static constexpr int MultiplierOffsetM = 1;
  static constexpr int MultiplierOffsetN = 2;
  static constexpr int MultiplierOffsetK = 3;
  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);

  float waves[] = {0.5, 1.25, 2.5};
  int cluster_m = 1;
  int cluster_n = 1;

  std::vector<int> problem_size_k;
  if (override_problem_size_k.empty()) {
    problem_size_k = {256 + max_alignment * MultiplierOffsetK, 512 + max_alignment * MultiplierOffsetK};
  }
  else {
    problem_size_k = override_problem_size_k;
  }

  if constexpr(DispatchPolicy::ArchTag::kMinComputeCapability >= 90) {
    typename DispatchPolicy::ClusterShape cluster_shape;
    cluster_m = cute::size<0>(cluster_shape);
    cluster_n = cute::size<1>(cluster_shape);
  }

  [[maybe_unused]] constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;

  std::vector<DecompositionMode> decomposition_modes = {DecompositionMode::Heuristic};
  static constexpr bool UsesStreamKScheduler = cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;
  if constexpr (UsesStreamKScheduler) {
    decomposition_modes.push_back(DecompositionMode::DataParallel);
    decomposition_modes.push_back(DecompositionMode::SplitK);
    decomposition_modes.push_back(DecompositionMode::StreamK);
  }
  bool passed = true;

  for (float wave : waves) {
    for (int k : problem_size_k) {
      int grid_m, grid_n = 0;
      int num_grid = int(wave * SmCount);

      if (cluster_m >= cluster_n) {
        grid_m = cluster_m;
        grid_n = num_grid / grid_m;
        // Align grid_n to cluster_n
        grid_n = std::max((grid_n + cluster_n - 1 ) / cluster_n * cluster_n, 1);
      }
      else {
        grid_n = cluster_n;
        grid_m = num_grid / grid_n;
        // Align grid_m to cluster_m
        grid_m = std::max((grid_m + cluster_m - 1 ) / cluster_m * cluster_m, 1);
      }

      int m = grid_m * cute::size<0>(cta_shape) + MultiplierOffsetM * max_alignment;
      int n = grid_n * cute::size<1>(cta_shape) + MultiplierOffsetN * max_alignment;

      ProblemShapeType problem_size;
      if constexpr (cute::rank(ProblemShapeType{}) == 4) {
        problem_size = ProblemShapeType{m, n, k, /* l */ 1};
      }
      else {
        problem_size = ProblemShapeType{m, n, k};
      }

      for (DecompositionMode decomp_mode : decomposition_modes) {
        std::vector problem_splits = {detail::Splits{1}};
        if (decomp_mode == DecompositionMode::Heuristic || decomp_mode == DecompositionMode::SplitK) {
          problem_splits.push_back(detail::Splits{2});
        }
        for (auto splits : problem_splits) {

          if constexpr (cute::is_same_v<RuntimeDataTypeA, cute::UMMA::MXF4Format> &&
                        cute::is_same_v<RuntimeDataTypeB, cute::UMMA::MXF4Format>) {
            // e2m1_e2m1
            if (runtime_input_datatype_a == cute::UMMA::MXF4Format::E2M1 &&
                runtime_input_datatype_b == cute::UMMA::MXF4Format::E2M1) {
              Testbed3x<Gemm,
                        cutlass::epilogue::thread::Identity,
                        force_legacy_epilogue,
                        cutlass::float_e2m1_t,
                        cutlass::float_e2m1_t,
                        cute::UMMA::MXF4Format,
                        cute::UMMA::MXF4Format> testbed(check_relative_equality,
                                                        use_device_scalars,
                                                        vector_scale_mode);
              passed = testbed.run(
                problem_size,
                cutlass::from_real<ElementScalar>(alpha),
                cutlass::from_real<ElementScalar>(beta),
                RasterOrderOptions::Heuristic, // raster_order
                detail::MaxSwizzleSize(1),
                splits,
                decomp_mode,
                false,
                detail::Iterations{},
                runtime_input_datatype_a,
                runtime_input_datatype_b
              );
            }
            else {
              std::cout << "Unsupported configuration for runtime datatype MXFP4." << std::endl;
              return false;
            }
          }

          else
          if constexpr (cute::is_same_v<RuntimeDataTypeA, cute::UMMA::MXF8F6F4Format> &&
                             cute::is_same_v<RuntimeDataTypeB, cute::UMMA::MXF8F6F4Format>) {
            static_assert((cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float8_t> ||
                           cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float6_t> ||
                           cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float4_t>) &&
                          (cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float8_t> ||
                           cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float6_t> ||
                           cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float4_t>),
                          "Runtime datatype must be selected with an appropriate static umbrella data type.");
            if constexpr (cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float8_t> &&
                          cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float4_t>) {
              // e4m3_e2m1
              if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E4M3 &&
                  runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E2M1) {
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e4m3_t,
                          cutlass::float_e2m1_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // Unsupport
              else {
                std::cout << "Unsupported configuration for runtime datatype Mxf8f6f4." << std::endl;
                return false;
              }
            }
            // f6xf4
            else if constexpr (cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float6_t> &&
                               cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float4_t>) {
              // e3m2_e2m1
              if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E3M2 &&
                  runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E2M1) {
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e3m2_t,
                          cutlass::float_e2m1_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);

                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // Unsupport
              else {
                std::cout << "Unsupported configuration for runtime datatype Mxf8f6f4." << std::endl;
                return false;
              }
            }
            else if constexpr (cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float4_t> &&
                               cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float4_t>) {
              // e2m1_e2m1
              if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E2M1 &&
                  runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E2M1) {
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e2m1_t,
                          cutlass::float_e2m1_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // Unsupport
              else {
                std::cout << "Unsupported configuration for runtime datatype Mxf8f6f4." << std::endl;
                return false;
              }
            }
            else if constexpr (cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float8_t> &&
                               cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float6_t>) {
              // e4m3_e3m2
              if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E4M3 &&
                  runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E3M2) {
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e4m3_t,
                          cutlass::float_e3m2_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // Unsupport
              else {
                std::cout << "Unsupported configuration for runtime datatype Mxf8f6f4." << std::endl;
                return false;
              }
            }
            else if constexpr (cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float6_t> &&
                               cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float6_t>) {
              // e3m2_e2m3
              if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E3M2 &&
                  runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E2M3) {
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e3m2_t,
                          cutlass::float_e2m3_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // Unsupported
              else {
                std::cout << "Unsupported configuration for runtime datatype Mxf8f6f4." << std::endl;
                return false;
              }
            }
            else
            if constexpr (cute::is_same_v<InternalElementA, cutlass::type_erased_dynamic_float8_t> &&
                               cute::is_same_v<InternalElementB, cutlass::type_erased_dynamic_float8_t>) {
              // e5m2_e5m2
              if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E5M2 &&
                  runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E5M2) {
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e5m2_t,
                          cutlass::float_e5m2_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // e4m3_e5m2
              else if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E4M3 &&
                       runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E5M2){
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e4m3_t,
                          cutlass::float_e5m2_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // e5m2_e4m3
              else if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E5M2 &&
                       runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E4M3){
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e5m2_t,
                          cutlass::float_e4m3_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // e4m3_e4m3
              else if (runtime_input_datatype_a == cute::UMMA::MXF8F6F4Format::E4M3 &&
                       runtime_input_datatype_b == cute::UMMA::MXF8F6F4Format::E4M3){
                Testbed3x<Gemm,
                          cutlass::epilogue::thread::Identity,
                          force_legacy_epilogue,
                          cutlass::float_e4m3_t,
                          cutlass::float_e4m3_t,
                          cute::UMMA::MXF8F6F4Format,
                          cute::UMMA::MXF8F6F4Format> testbed(check_relative_equality,
                                                          use_device_scalars,
                                                          vector_scale_mode);
                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  RasterOrderOptions::Heuristic, // raster_order
                  detail::MaxSwizzleSize(1),
                  splits,
                  decomp_mode,
                  false,
                  detail::Iterations{},
                  runtime_input_datatype_a,
                  runtime_input_datatype_b
                );
              }
              // Unsupported
              else {
                std::cout << "Unsupported configuration for runtime datatype Mxf8f6f4." << std::endl;
                return false;
              }
            }
            // Unsupported
            else {
              std::cout << "Unsupported configuration for runtime datatype Mxf8f6f4." << std::endl;
              return false;
            }
          }

          else {
            static_assert(cutlass::detail::dependent_false<RuntimeDataTypeA>,
                "Unsupported configuration for runtime datatype.");
          }

          if (!passed) {
            std::cout << __FILE__ << ':' << __LINE__ << " : GEMM MNK " << m << " " << n << " " << k << " FAILED.\n";
            return false;
          }
        } // splits
      } // decomposition_mode
    } // k
  } // waves

  return passed;
}

template <typename Gemm, bool force_legacy_epilogue = false, bool apply_alignment_offset = true, bool test_batched_alpha_beta = false>
bool TestSmall(double alpha = 1.0, double beta = cute::is_same_v<typename Gemm::GemmKernel::ElementC, void> ? 0.0 : 1.0,
  CheckEquality check_relative_equality = CheckEquality::RELATIVE,
  ScalarLoc use_device_scalars = ScalarLoc::ON_DEVICE,
  VectorScale vector_scale_mode = VectorScale::ENABLED,
  std::vector<int> override_problem_size_k = {}) {

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using CtaShape_MNK = typename Gemm::GemmKernel::CollectiveMainloop::CtaShape_MNK;
  using DispatchPolicy = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy;
  CtaShape_MNK cta_shape;
  Testbed3x<Gemm, cutlass::epilogue::thread::Identity, force_legacy_epilogue> testbed(check_relative_equality, use_device_scalars, vector_scale_mode);
  static constexpr int SmCount  = 16;
  static constexpr int MultiplierOffsetM = 1;
  static constexpr int MultiplierOffsetN = 2;
  static constexpr int MultiplierOffsetK = 3;
  int max_alignment_k = 0;
  int max_alignment_m = 0;
  int max_alignment_n = 0;

  if constexpr (apply_alignment_offset) {
    max_alignment_k = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
    max_alignment_n = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
    max_alignment_m = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  }
  // Alignment for SFD
  if constexpr (detail::IsSfdEpi<typename Gemm::GemmKernel::CollectiveEpilogue>::value) {
    using GmemLayoutTagScalefactor = typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Operation::GmemLayoutTagScalefactor;
    constexpr int SFDVecSize = Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Operation::SFVecSize;
    if constexpr (cute::is_same_v<GmemLayoutTagScalefactor, cutlass::layout::RowMajor>) {
      max_alignment_n = std::lcm(max_alignment_n, SFDVecSize);
    }
    else {
      max_alignment_m = std::lcm(max_alignment_m, SFDVecSize);
    }
  }

  float waves[] = {0.5, 1.25, 2.5};
  int cluster_m = 1;
  int cluster_n = 1;

  std::vector<int> problem_size_k;
  if (override_problem_size_k.empty()) {
    problem_size_k = {256 + max_alignment_k * MultiplierOffsetK, 512 + max_alignment_k * MultiplierOffsetK};
  }
  else {
    problem_size_k = override_problem_size_k;
  }

  if constexpr(DispatchPolicy::ArchTag::kMinComputeCapability >= 90) {
    typename DispatchPolicy::ClusterShape cluster_shape;
    cluster_m = cute::size<0>(cluster_shape);
    cluster_n = cute::size<1>(cluster_shape);
  }

  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;

  std::vector<DecompositionMode> decomposition_modes = {DecompositionMode::Heuristic};
  static constexpr bool UsesStreamKScheduler = cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;
  if constexpr (UsesStreamKScheduler) {
    decomposition_modes.push_back(DecompositionMode::DataParallel);
    decomposition_modes.push_back(DecompositionMode::SplitK);
    decomposition_modes.push_back(DecompositionMode::StreamK);
  }
  bool passed = true;

  std::vector<RasterOrderOptions> raster_order_options = {RasterOrderOptions::Heuristic};
  for (float wave : waves) {
    for (int k : problem_size_k) {
      int grid_m, grid_n = 0;
      int num_grid = int(wave * SmCount);

      if (cluster_m >= cluster_n) {
        grid_m = cluster_m;
        grid_n = num_grid / grid_m;
        // Align grid_n to cluster_n
        grid_n = std::max((grid_n + cluster_n - 1 ) / cluster_n * cluster_n, 1);
      }
      else {
        grid_n = cluster_n;
        grid_m = num_grid / grid_n;
        // Align grid_m to cluster_m
        grid_m = std::max((grid_m + cluster_m - 1 ) / cluster_m * cluster_m, 1);
      }

      int m = grid_m * cute::size<0>(cta_shape) + MultiplierOffsetM * max_alignment_m;
      int n = grid_n * cute::size<1>(cta_shape) + MultiplierOffsetN * max_alignment_n;
      int l = test_batched_alpha_beta && wave == waves[0] && k == problem_size_k[0] ? 2 : 1; // only test the smallest problem size
      ProblemShapeType problem_size;
      if constexpr (cute::rank(ProblemShapeType{}) == 4) {
        problem_size = ProblemShapeType{m, n, k, l};
      }
      else {
        problem_size = ProblemShapeType{m, n, k};
      }

      for (DecompositionMode decomp_mode : decomposition_modes) {
        for (RasterOrderOptions raster_order : raster_order_options) {
          std::vector problem_splits = {detail::Splits{1}};
          if constexpr (UsesStreamKScheduler) {
            if (decomp_mode == DecompositionMode::SplitK) {
              problem_splits.push_back(detail::Splits{2});
              problem_splits.push_back(detail::Splits{4});
            }
          }
          for (auto splits : problem_splits) {
            try {
              passed = testbed.run(
                problem_size,
                cutlass::from_real<ElementScalar>(alpha),
                cutlass::from_real<ElementScalar>(beta),
                raster_order, // raster_order
                detail::MaxSwizzleSize(0),
                splits,
                decomp_mode
              );
            }
            catch (std::exception const& e) {
              EXPECT_TRUE(false) << "TestSmall: testbed.run {"
                << "m: " << m << ", n: " << n << ", k: " << k << ", l: " << l
                << ", alpha: " << alpha << ", beta: " << beta
                << ", raster_order: " << detail::raster_order_to_string(raster_order)
                << ", max_swizzle_size: 1"
                << ", splits: " << static_cast<int>(splits)
                << ", decomp_mode: " << detail::decomp_mode_to_string(decomp_mode)
                << "} threw an exception: " << e.what();
              throw;
            }
            catch (...) {
              EXPECT_TRUE(false) << "TestSmall: testbed.run {"
                << "m: " << m << ", n: " << n << ", k: " << k << ", l: " << l
                << ", alpha: " << alpha << ", beta: " << beta
                << ", raster_order: " << detail::raster_order_to_string(raster_order)
                << ", max_swizzle_size: 1"
                << ", splits: " << static_cast<int>(splits)
                << ", decomp_mode: " << detail::decomp_mode_to_string(decomp_mode)
                << "} threw an exception (unknown)";
              throw;
            }
            EXPECT_TRUE(passed) << "TestSmall: testbed.run {"
              << "m: " << m << ", n: " << n << ", k: " << k << ", l: " << l
              << ", alpha: " << alpha << ", beta: " << beta
              << ", raster_order: " << detail::raster_order_to_string(raster_order)
              << ", max_swizzle_size: 1"
              << ", splits: " << static_cast<int>(splits)
              << ", decomp_mode: " << detail::decomp_mode_to_string(decomp_mode)
              << "} failed";

            if (!passed) {
              std::cout << __FILE__ << ':' << __LINE__ << " : GEMM MNKL " << m << " " << n << " " << k << " " << l << " FAILED.\n";
              return false;
            }
          } // splits
        } // raster_order
      } // decomposition_mode
    } // k
  } // waves

  return passed;
}

template <typename Gemm, bool force_legacy_epilogue = false, bool apply_alignment_offset = true, bool test_batched_alpha_beta = false>
bool TestSmallFusion(double alpha = 1.0, double beta = cute::is_same_v<typename Gemm::GemmKernel::ElementC, void> ? 0.0 : 1.0,
                     CheckEquality check_relative_equality = CheckEquality::RELATIVE,
                     ScalarLoc use_device_scalars = ScalarLoc::ON_DEVICE,
                     VectorScale vector_scale_mode = VectorScale::ENABLED,
                     std::vector<int> override_problem_size_k = {}) {
  return TestSmall<Gemm,
                   force_legacy_epilogue,
                   apply_alignment_offset,
                   test_batched_alpha_beta>(alpha,
                                            beta,
                                            check_relative_equality,
                                            use_device_scalars,
                                            vector_scale_mode,
                                            override_problem_size_k);
}



template <
  typename Gemm,
  template <class T> class ActivationFunctor = cutlass::epilogue::thread::Identity
>
bool TestAll(double alpha = 1.0, double beta = cute::is_same_v<typename Gemm::GemmKernel::ElementC, void> ? 0.0 : 1.0, CheckEquality check_relative_equality = CheckEquality::RELATIVE) {
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  Testbed3x<Gemm, ActivationFunctor> testbed(check_relative_equality, ScalarLoc::ON_HOST, VectorScale::DISABLED);

  int max_alignment_m = std::max({Gemm::kAlignmentA, Gemm::kAlignmentC, Gemm::kAlignmentD});
  int max_alignment_n = std::max({Gemm::kAlignmentB, Gemm::kAlignmentC, Gemm::kAlignmentD});
  if constexpr (std::is_base_of_v<cutlass::epilogue::fusion::FusionOperation, typename Gemm::EpilogueOutputOp>) {
    max_alignment_m = std::max(max_alignment_m, Gemm::EpilogueOutputOp::AlignmentAux);
    max_alignment_n = std::max(max_alignment_n, Gemm::EpilogueOutputOp::AlignmentAux);
  }
  std::vector<int> problem_size_m = {max_alignment_m, 512 - 3 * max_alignment_m};
  std::vector<int> problem_size_n = {max_alignment_n, 512 - 2 * max_alignment_n};

  if constexpr (cute::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong>) {
    problem_size_m.push_back(768);
    problem_size_n.push_back(768);
  }

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  int max_alignment_k = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_k = {max_alignment_k, TileShapeK * (Stages + 1) - max_alignment_k};

  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  std::vector<DecompositionMode> decomposition_modes = {DecompositionMode::Heuristic};
  std::vector problem_splits = {detail::Splits{1}};
  static constexpr bool UsesStreamKScheduler = cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;
  if constexpr (UsesStreamKScheduler) {
    problem_splits.push_back(detail::Splits{2});
    problem_splits.push_back(detail::Splits{3});

    decomposition_modes.push_back(DecompositionMode::DataParallel);
    decomposition_modes.push_back(DecompositionMode::SplitK);
    decomposition_modes.push_back(DecompositionMode::StreamK);

    // Use larger K sizes for stream-K tests
    static constexpr int min_tiles_per_sk_unit = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::min_iters_per_sk_unit_;
    problem_size_k = {TileShapeK * min_tiles_per_sk_unit, TileShapeK * 3 * min_tiles_per_sk_unit - max_alignment_k};
  }

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  std::vector<RasterOrderOptions> raster_orders = {RasterOrderOptions::AlongM, RasterOrderOptions::AlongN};
  std::vector max_swizzle_sizes{detail::MaxSwizzleSize{1}, detail::MaxSwizzleSize{4}};

  bool passed = true;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (auto raster_order : raster_orders) {
          for (auto max_swizzle_size : max_swizzle_sizes) {
            for (DecompositionMode decomp_mode : decomposition_modes) {

              std::vector problem_splits = {detail::Splits{1}};
              if (decomp_mode == DecompositionMode::Heuristic || decomp_mode == DecompositionMode::SplitK) {
                auto max_splits = (k + TileShapeK - 1) / TileShapeK;
                if (max_splits > 2) {
                  problem_splits.push_back(detail::Splits{2});
                }
                if (max_splits > 3) {
                  problem_splits.push_back(detail::Splits{3});
                }

                problem_splits.push_back(detail::Splits{max_splits});

                // Test the case in which we ask for more splits than there are K tiles in the GEMM. In this
                // case, split-K will fall back to a splitting factor of `max_splits`.
                problem_splits.push_back(detail::Splits{max_splits + 1});
              }
              for (auto splits : problem_splits) {
                ProblemShapeType problem_size;
                if constexpr (cute::rank(ProblemShapeType{}) == 4) {
                  problem_size = ProblemShapeType{m, n, k, /* l */ 1};
                }
                else {
                  problem_size = ProblemShapeType{m, n, k};
                }

                try {
                  passed = testbed.run(
                    problem_size,
                    cutlass::from_real<ElementScalar>(alpha),
                    cutlass::from_real<ElementScalar>(beta),
                    raster_order,
                    max_swizzle_size,
                    splits,
                    decomp_mode
                  );
                }
                catch (std::exception const& e) {
                  EXPECT_TRUE(false) << "TestAll: testbed.run {"
                    << "m: " << m << ", n: " << n << ", k: " << k
                    << ", alpha: " << alpha << ", beta: " << beta
                    << ", raster_order: ???"
                    << ", max_swizzle_size: " << static_cast<int>(max_swizzle_size)
                    << ", splits: " << static_cast<int>(splits)
                    << ", decomp_mode: " << detail::decomp_mode_to_string(decomp_mode)
                    << "} threw an exception: " << e.what();
                  throw;
                }
                catch (...) {
                  EXPECT_TRUE(false) << "TestAll: testbed.run {"
                    << "m: " << m << ", n: " << n << ", k: " << k
                    << ", alpha: " << alpha << ", beta: " << beta
                    << ", raster_order: ???"
                    << ", max_swizzle_size: " << static_cast<int>(max_swizzle_size)
                    << ", splits: " << static_cast<int>(splits)
                    << ", decomp_mode: " << detail::decomp_mode_to_string(decomp_mode)
                    << "} threw an exception (unknown)";
                  throw;
                }

                EXPECT_TRUE(passed) << "TestAll: testbed.run {"
                  << "m: " << m << ", n: " << n << ", k: " << k
                  << ", alpha: " << alpha << ", beta: " << beta
                  << ", raster_order: ???"
                  << ", max_swizzle_size: " << static_cast<int>(max_swizzle_size)
                  << ", splits: " << static_cast<int>(splits)
                  << ", decomp_mode: " << detail::decomp_mode_to_string(decomp_mode)
                  << "} failed";

                if (!passed) {
                  std::cout << __FILE__ << ':' << __LINE__ << " : GEMM MNK " << m << " " << n << " " << k << " FAILED.\n";
                  return false;
                }
              } // splits
            } // decomposition_mode
          } // max_swizzle_size
        } // raster_order
      } // k
    } // n
  } // m

  // if we do support batched GEMM, just run one test on it to save on test time
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
    auto problem_size = ProblemShapeType{256 + max_alignment_m, 256 + max_alignment_n, 160 + max_alignment_k, /* l */ 3};
    passed = testbed.run(
      problem_size,
      cutlass::from_real<ElementScalar>(alpha),
      cutlass::from_real<ElementScalar>(beta)
    );

    if (!passed) {
      return false;
    }
  }

  return passed;
}

template <typename Gemm>
bool TestAllBiasElementwise(double alpha = 1.0, double beta = cute::is_same_v<typename Gemm::GemmKernel::ElementC, void> ? 0.0 : 1.0, CheckEquality check_relative_equality = CheckEquality::EXACT) {
  return TestAll<Gemm>(alpha, beta, check_relative_equality);
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
