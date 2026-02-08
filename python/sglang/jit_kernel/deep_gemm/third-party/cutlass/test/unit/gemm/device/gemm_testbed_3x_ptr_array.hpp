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
    \brief Testbed for Ptr-Array and Grouped GEMM interface
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

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

namespace detail{

// Helper classes that take default data type when
// the Gemm::EpilogueOutputOp does not have ElementCompute
// and ElementScalar.
// (e.g. when Sm90TreeVisitor is used as FusionCallbacks)
template <typename Gemm, typename Default, typename = void>
struct ElementComputeType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementComputeType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementCompute>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementCompute;
};

template <typename Gemm, typename Default, typename = void>
struct ElementScalarType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementScalarType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementScalar>> {
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
  class ElementB_ = typename Gemm::GemmKernel::ElementB>
struct HostCollectiveMainloop {
  // Kernel data types
  using ElementA = ElementA_;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using InternalStrideA  = typename Gemm::GemmKernel::InternalStrideA;
  using ElementB = ElementB_;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using InternalStrideB  = typename Gemm::GemmKernel::InternalStrideB;
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using LayoutTagA = cutlass::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;

  static constexpr bool IsGroupGemm = !cute::is_same_v<StrideA, InternalStrideA>;

  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;

  using Arguments = typename Gemm::GemmKernel::MainloopArguments;

  cutlass::ComplexTransform TransformA = Gemm::kTransformA;
  cutlass::ComplexTransform TransformB = Gemm::kTransformB;

  std::vector<InternalStrideA> stride_a_host;
  std::vector<InternalStrideB> stride_b_host;

  cutlass::DeviceAllocation<InternalStrideA> stride_a_device;
  cutlass::DeviceAllocation<InternalStrideB> stride_b_device;

  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;

  std::vector<cutlass::HostTensor<ElementA, LayoutTagA>> tensors_A;
  std::vector<cutlass::HostTensor<ElementB, LayoutTagB>> tensors_B;
  cutlass::DeviceAllocation<const ElementA *> device_tensors_A;
  cutlass::DeviceAllocation<const ElementB *> device_tensors_B;
  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;

  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<InternalStrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<InternalStrideB>(),
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

  bool initialize(ProblemShapeType problem_shapes) {
    //
    // Allocate the GEMM workspace
    //
    // for pointer array problem_shapes.groups() is 1

    tensors_A.clear();
    tensors_B.clear();
    stride_a_host.clear();
    stride_b_host.clear();

    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = cutlass::platform::max(problem_shapes.groups(), L);

    for(int32_t i = 0; i < L; ++i) {
      auto [M, N, K, mock_L] = cute::append<4>(problem_shapes.get_host_problem_shape(i), 1);

      stride_a_host.push_back(cutlass::make_cute_packed_stride(InternalStrideA{}, {M, K, 1}));
      stride_b_host.push_back(cutlass::make_cute_packed_stride(InternalStrideB{}, {N, K, 1}));

      // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
      auto a_coord = cutlass::make_Coord(M, K);
      // Cutlass has Row/Col major refers to MxK times KxN matrix product,
      // so the HostTensorB should be treated as KxN in "coord"'s view
      auto b_coord = cutlass::make_Coord(K, N);

      tensors_A.push_back(cutlass::HostTensor<ElementA, LayoutTagA>(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A)));
      tensors_B.push_back(cutlass::HostTensor<ElementB, LayoutTagB>(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B)));

      EXPECT_TRUE(initialize_tensor(tensors_A[i].host_view(), init_A, seed + 2022 + i));
      EXPECT_TRUE(initialize_tensor(tensors_B[i].host_view(), init_B, seed + 2021 + i));

      // It is possible to randomly initialize to all zeros, so override this with non-zeros
      // in the upper left corner of each operand.
      tensors_A[i].host_view().at({0, 0}) = ElementA(1);
      tensors_B[i].host_view().at({0, 0}) = ElementB(1);

      tensors_A[i].sync_device();
      tensors_B[i].sync_device();
    }

    return true;
  }

  Arguments to_args(ProblemShapeType problem_shapes) {
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = cutlass::platform::max(problem_shapes.groups(), L);

    std::vector<ElementA *> ptr_A_host(L);
    std::vector<ElementB *> ptr_B_host(L);

    for (int32_t i = 0; i < L; ++i) {
      ptr_A_host.at(i) = tensors_A[i].device_data();
      ptr_B_host.at(i) = tensors_B[i].device_data();
    }

    device_tensors_A.reset(L);
    device_tensors_A.copy_from_host(ptr_A_host.data());

    device_tensors_B.reset(L);
    device_tensors_B.copy_from_host(ptr_B_host.data());

    stride_a_device.reset(problem_shapes.groups());
    stride_a_device.copy_from_host(stride_a_host.data());
    stride_b_device.reset(problem_shapes.groups());
    stride_b_device.copy_from_host(stride_b_host.data());

    Arguments arguments;

    if constexpr (IsGroupGemm) {
      arguments
      =
      {
        device_tensors_A.get(), stride_a_device.get(), device_tensors_B.get(), stride_b_device.get()
      };
    }
    else {
      arguments =
      {
        device_tensors_A.get(), stride_a_host[0], device_tensors_B.get(), stride_b_host[0]
      };
    }

    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_shapes, int batch) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(batch), 1);
    auto A = make_tensor(make_iterator(tensors_A[batch].host_data()),
          make_layout(make_shape(M, K, 1), stride_a_host[batch]));
    auto B = make_tensor(make_iterator(tensors_B[batch].host_data()),
        make_layout(make_shape(N, K, 1), stride_b_host[batch]));

    cutlass::reference::host::GettMainloopParams<ElementAccumulator,
                                                 decltype(A),
                                                 decltype(B)
                                                 > mainloop_params{};

    mainloop_params.A = A;
    mainloop_params.B = B;
    mainloop_params.transform_A = TransformA;
    mainloop_params.transform_B = TransformB;

    return mainloop_params;
  }

  void print_tensors(std::ofstream& file, int batch) {
    file << "A =\n" << tensors_A[batch].host_view()
         << "\nB =\n" << tensors_B[batch].host_view();
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
      ProblemShapeType problem_shapes, int batch) {
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_A[batch].host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_B[batch].host_view()), 0);

    bool passed = true;
    return passed;
  }
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
struct HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<
                                SchedulerPipelineStageCount_,
                                AccumulatorPipelineStageCount_>,
                                Gemm, ElementA_, ElementB_> {
  // Kernel data types
  using ElementA = ElementA_;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using InternalStrideA  = typename Gemm::GemmKernel::InternalStrideA;
  using ElementB = ElementB_;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using InternalStrideB  = typename Gemm::GemmKernel::InternalStrideB;
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using LayoutTagA = cutlass::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;

  static constexpr bool IsGroupGemm = !cute::is_same_v<StrideA, InternalStrideA>;

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
  using InternalLayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using InternalLayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;

  using Arguments = typename Gemm::GemmKernel::MainloopArguments;

  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;

  std::vector<InternalStrideA> stride_a_host;
  std::vector<InternalStrideB> stride_b_host;
  cutlass::DeviceAllocation<InternalStrideA> stride_a_device;
  cutlass::DeviceAllocation<InternalStrideB> stride_b_device;

  std::vector<InternalLayoutSFA> layout_sfa_host;
  std::vector<InternalLayoutSFB> layout_sfb_host;
  cutlass::DeviceAllocation<InternalLayoutSFA> layout_sfa_device;
  cutlass::DeviceAllocation<InternalLayoutSFB> layout_sfb_device;

  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;

  std::vector<cutlass::HostTensor<ElementA, LayoutTagA>> tensors_A;
  std::vector<cutlass::HostTensor<ElementB, LayoutTagB>> tensors_B;
  std::vector<cutlass::HostTensor<ElementSF, LayoutTagA>> tensors_SFA;
  std::vector<cutlass::HostTensor<ElementSF, LayoutTagB>> tensors_SFB;

  cutlass::DeviceAllocation<const ElementA *> device_tensors_A;
  cutlass::DeviceAllocation<const ElementB *> device_tensors_B;
  cutlass::DeviceAllocation<const ElementSF *> device_tensors_SFA;
  cutlass::DeviceAllocation<const ElementSF *> device_tensors_SFB;

  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<InternalStrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<InternalStrideB>(),
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
  bool initialize(ProblemShapeType problem_shapes) {
    //
    // Allocate the GEMM workspace
    //

    tensors_A.clear();
    tensors_B.clear();
    stride_a_host.clear();
    stride_b_host.clear();
    tensors_SFA.clear();
    tensors_SFB.clear();
    layout_sfa_host.clear();
    layout_sfb_host.clear();

    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = std::max(problem_shapes.groups(), L);

    for (int32_t i = 0; i < L; ++i) {
      auto [M, N, K, mock_L] = cute::append<4>(problem_shapes.get_host_problem_shape(i), 1);

      stride_a_host.push_back(cutlass::make_cute_packed_stride(InternalStrideA{}, {M, K, 1}));
      stride_b_host.push_back(cutlass::make_cute_packed_stride(InternalStrideB{}, {N, K, 1}));

      // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
      auto a_coord = cutlass::make_Coord(M, K);
      // Cutlass has Row/Col major refers to MxK times KxN matrix product,
      // so the HostTensorB should be treated as KxN in "coord"'s view
      auto b_coord = cutlass::make_Coord(K, N);

      tensors_A.push_back(cutlass::HostTensor<ElementA, LayoutTagA>(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A)));
      tensors_B.push_back(cutlass::HostTensor<ElementB, LayoutTagB>(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B)));

      EXPECT_TRUE(initialize_tensor(tensors_A[i].host_view(), init_A, seed + 2022 + i));
      EXPECT_TRUE(initialize_tensor(tensors_B[i].host_view(), init_B, seed + 2021 + i));

      // It is possible to randomly initialize to all zeros, so override this with non-zeros
      // in the upper left corner of each operand.
      tensors_A[i].host_view().at({0, 0}) = ElementA(1);
      tensors_B[i].host_view().at({0, 0}) = ElementB(1);

      tensors_A[i].sync_device();
      tensors_B[i].sync_device();

      using namespace cute;

      auto k_blks = cutlass::ceil_div(K, size<1>(shape(SfAtom{})));
      auto m_blks = cutlass::ceil_div(M, Blk_MN{});
      auto n_blks = cutlass::ceil_div(N, Blk_MN{});
      layout_sfa_host.push_back(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1)));
      layout_sfb_host.push_back(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1)));

      // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
      auto sfa_coord   = cutlass::make_Coord(m_blks * Blk_MN{}, k_blks * Blk_SF{});
      auto sfb_coord   = cutlass::make_Coord(n_blks * Blk_MN{}, k_blks * Blk_SF{});

      tensors_SFA.push_back(cutlass::HostTensor<ElementSF, LayoutTagA>(sfa_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(sfa_coord, stride_factor_A)));
      tensors_SFB.push_back(cutlass::HostTensor<ElementSF, LayoutTagB>(sfb_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(sfb_coord, stride_factor_B)));

      EXPECT_TRUE(initialize_tensor(tensors_SFA[i].host_view(), init_A, seed + 2024 + i));
      EXPECT_TRUE(initialize_tensor(tensors_SFB[i].host_view(), init_B, seed + 2025 + i));

      // It is possible to randomly initialize to all zeros, so override this with non-zeros
      // in the upper left corner of each operand.
      tensors_SFA[i].host_view().at({0, 0}) = ElementSF(1);
      tensors_SFB[i].host_view().at({0, 0}) = ElementSF(1);

      tensors_SFA[i].sync_device();
      tensors_SFB[i].sync_device();
    }

    return true;
  }

  Arguments to_args(ProblemShapeType problem_shapes) {
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = std::max(problem_shapes.groups(), L);

    std::vector<ElementA *> ptr_A_host(L);
    std::vector<ElementB *> ptr_B_host(L);
    std::vector<ElementSF *> ptr_SFA_host(L);
    std::vector<ElementSF *> ptr_SFB_host(L);

    for (int32_t i = 0; i < L; ++i) {
      ptr_A_host.at(i) = tensors_A[i].device_data();
      ptr_B_host.at(i) = tensors_B[i].device_data();
      ptr_SFA_host.at(i) = tensors_SFA[i].device_data();
      ptr_SFB_host.at(i) = tensors_SFB[i].device_data();
    }

    device_tensors_A.reset(L);
    device_tensors_A.copy_from_host(ptr_A_host.data());

    device_tensors_B.reset(L);
    device_tensors_B.copy_from_host(ptr_B_host.data());

    device_tensors_SFA.reset(L);
    device_tensors_SFA.copy_from_host(ptr_SFA_host.data());

    device_tensors_SFB.reset(L);
    device_tensors_SFB.copy_from_host(ptr_SFB_host.data());

    stride_a_device.reset(problem_shapes.groups());
    stride_a_device.copy_from_host(stride_a_host.data());

    stride_b_device.reset(problem_shapes.groups());
    stride_b_device.copy_from_host(stride_b_host.data());

    layout_sfa_device.reset(problem_shapes.groups());
    layout_sfa_device.copy_from_host(layout_sfa_host.data());

    layout_sfb_device.reset(problem_shapes.groups());
    layout_sfb_device.copy_from_host(layout_sfb_host.data());

    if constexpr (IsGroupGemm) {
      return Arguments{
        device_tensors_A.get(), stride_a_device.get(),
        device_tensors_B.get(), stride_b_device.get(),
        device_tensors_SFA.get(), layout_sfa_device.get(),
        device_tensors_SFB.get(), layout_sfb_device.get()
      };
    }
    else {
      return Arguments{
        device_tensors_A.get(), stride_a_host[0],
        device_tensors_B.get(), stride_b_host[0],
        device_tensors_SFA.get(), layout_sfa_host[0],
        device_tensors_SFB.get(), layout_sfb_host[0]
      };
    }
  }

  auto to_host_args(ProblemShapeType problem_shapes, int batch) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(batch), 1);
    auto A = make_tensor(make_iterator(tensors_A[batch].host_data()),
          make_layout(make_shape(M, K, 1), stride_a_host[batch]));
    auto SfA = make_tensor(tensors_SFA[batch].host_data(), layout_sfa_host[batch]);

    auto B = make_tensor(make_iterator(tensors_B[batch].host_data()),
        make_layout(make_shape(N, K, 1), stride_b_host[batch]));
    auto SfB = make_tensor(tensors_SFB[batch].host_data(), layout_sfb_host[batch]);

    return cutlass::reference::host::GettMainloopParams<ElementAccumulator,
        decltype(A),
        decltype(B),
        decltype(SfA),
        decltype(SfB)
      >
      {A, SfA, B, SfB};
  }

  void print_tensors(std::ofstream& file, int batch) {
    file << "A =\n" << tensors_A[batch].host_view()
         << "\nB =\n" << tensors_B[batch].host_view()
         << "\nSFA =\n" << tensors_SFA[batch].host_view()
         << "\nSFB =\n" << tensors_SFB[batch].host_view();
  }

  bool compare_reference(
      ProblemShapeType problem_shapes, int batch) {

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_A[batch].host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_B[batch].host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_SFA[batch].host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_SFB[batch].host_view()), 0);
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
struct HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120<SchedulerPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<0,0>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<0,0>,
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
struct HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<SchedulerPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<0,0>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<0,0>,
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
struct HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm103<SchedulerPipelineStageCount_,
                                                                                              AccumulatorPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> : public
       HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_,AccumulatorPipelineStageCount_>,
                              Gemm, ElementA_, ElementB_> {
  using Base = HostCollectiveMainloop<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_,AccumulatorPipelineStageCount_>,
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
  using InternalStrideD  = typename kernel::InternalStrideD;
  using ElementC = non_void_t<typename kernel::ElementC, ElementD>;
  using StrideC  = typename kernel::StrideC;
  using InternalStrideC  = typename kernel::InternalStrideC;

  static constexpr bool IsGroupGemm = !cute::is_same_v<StrideD, InternalStrideD>;

  using FusionOp = typename Gemm::EpilogueOutputOp;

  static_assert(rank(InternalStrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(InternalStrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static_assert(is_row_or_col_major<InternalStrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<InternalStrideD>(),
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
  cutlass::DeviceAllocation<InternalStrideC> stride_c_device;
  cutlass::DeviceAllocation<InternalStrideD> stride_d_device;

  std::vector<InternalStrideC> stride_c_host;
  std::vector<InternalStrideD> stride_d_host;

  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;

  // Inputs
  ElementScalar alpha;
  ElementScalar beta;

  std::vector<cutlass::HostTensor<ElementC, LayoutTagC>> tensors_C;
  std::vector<cutlass::HostTensor<ElementD, LayoutTagD>> tensors_D;
  std::vector<cutlass::HostTensor<ElementD, LayoutTagD>> references_D;
  cutlass::DeviceAllocation<const ElementC *> device_tensors_C;
  cutlass::DeviceAllocation<ElementD *> device_tensors_D;

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

  bool initialize(ProblemShapeType problem_shapes, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
    // Initialize Epilogue tensors

    tensors_C.clear();
    tensors_D.clear();
    references_D.clear();
    stride_c_host.clear();
    stride_d_host.clear();

    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = cutlass::platform::max(problem_shapes.groups(), L);

    for (int32_t i = 0; i < L; ++i) {
      auto [M, N, K, mock_L] = cute::append<4>(problem_shapes.get_host_problem_shape(i), 1);

      stride_c_host.push_back(cutlass::make_cute_packed_stride(InternalStrideC{}, {M, N, 1}));
      stride_d_host.push_back(cutlass::make_cute_packed_stride(InternalStrideD{}, {M, N, 1}));

      // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
      auto c_coord = cutlass::make_Coord(M, N);

      tensors_C.push_back(cutlass::HostTensor<ElementC, LayoutTagC>(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C)));
      tensors_D.push_back(cutlass::HostTensor<ElementD, LayoutTagD>(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D)));
      references_D.push_back(cutlass::HostTensor<ElementD, LayoutTagD>(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false));
      EXPECT_TRUE(initialize_tensor(tensors_C[i].host_view(), init_C, seed + 2020));
      tensors_C[i].host_view().at({0, 0}) = ElementC(1);

      cutlass::reference::host::TensorCopy(references_D[i].host_view(), tensors_C[i].host_view());
      tensors_C[i].sync_device();
      tensors_D[i].sync_device();
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
      ProblemShapeType problem_shapes,
      ElementScalar alpha,
      ElementScalar beta,
      int batch) {
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = cutlass::platform::max(problem_shapes.groups(), L);

    tensors_D[batch].sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_C[batch].host_view()), 0);

    if (tensors_D[batch].size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_D[batch].host_view()), 0);
    }

    if (references_D[batch].size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(references_D[batch].host_view()), 0);
    }

    bool passed = equality_check(references_D[batch].host_view(), tensors_D[batch].host_view());
    if(!passed) {
      std::cout<<"D is incorrect"<<std::endl;
    }
    return passed;
  }

  void print_tensors(std::ofstream& file, int batch) {
    file
    << "\nC =\n" << tensors_C[batch].host_view()
    << "\n\nReference =\n" << references_D[batch].host_view()
    << "\n\nComputed =\n" << tensors_D[batch].host_view();
  }

  Arguments to_args(ProblemShapeType problem_shapes) {
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = cutlass::platform::max(problem_shapes.groups(), L);

    std::vector<ElementC *> ptr_C_host(L);
    std::vector<ElementD *> ptr_D_host(L);

    for (int32_t i = 0; i < L; ++i) {
      ptr_C_host.at(i) = tensors_C[i].device_data();
      ptr_D_host.at(i) = tensors_D[i].device_data();
    }

    device_tensors_C.reset(L);
    device_tensors_C.copy_from_host(ptr_C_host.data());

    device_tensors_D.reset(L);
    device_tensors_D.copy_from_host(ptr_D_host.data());

    stride_c_device.reset(problem_shapes.groups());
    stride_c_device.copy_from_host(stride_c_host.data());

    stride_d_device.reset(problem_shapes.groups());
    stride_d_device.copy_from_host(stride_d_host.data());

    Arguments arguments;
    if constexpr (IsGroupGemm) {
      arguments =
      {
        {alpha, beta},
        device_tensors_C.get(), stride_c_device.get(), device_tensors_D.get(), stride_d_device.get()
      };
    }
    else {
      arguments =
      {
        {alpha, beta},
        device_tensors_C.get(), stride_c_host[0], device_tensors_D.get(), stride_d_host[0]
      };
    }

    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_shapes, int batch) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(batch), 1);
    L = std::max(problem_shapes.groups(), L);

    auto coord_0 = cutlass::make_Coord(0);
    auto C = cute::make_tensor(detail::make_iterator(tensors_C[batch].host_data()),
        cute::make_layout(cute::make_shape(M, N, 1), stride_c_host[batch]));
    auto D = cute::make_tensor(detail::make_iterator(references_D[batch].host_data()),
        cute::make_layout(cute::make_shape(M, N, 1), stride_d_host[batch]));

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
  using InternalStrideD  = typename kernel::InternalStrideD;
  using ElementC = non_void_t<typename kernel::ElementC, ElementD>;
  using StrideC  = typename kernel::StrideC;
  using InternalStrideC  = typename kernel::InternalStrideC;

  static constexpr bool IsGroupGemm = !cute::is_same_v<StrideD, InternalStrideD>;

  static_assert(rank(InternalStrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(InternalStrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static_assert(is_row_or_col_major<InternalStrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<InternalStrideD>(),
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
  using EpiloguePolicy = typename Epilogue::DispatchPolicy;
  static constexpr bool IsLegacy =
  cute::is_same_v<
    EpiloguePolicy,
    cutlass::epilogue::Sm90TmaWarpSpecializedBiasElementwise<
      EpiloguePolicy::StagesC, EpiloguePolicy::StagesD, EpiloguePolicy::FragmentSize>
  >;

  using FusionOp = typename Gemm::EpilogueOutputOp;
  static_assert(cute::is_base_of_v<cutlass::epilogue::fusion::FusionOperation, FusionOp>);


  // Scale factor Generation related
  using SfStrategy = cutlass::reference::host::SfStrategy;
  static constexpr bool IsBlockScaleSupported            = FusionOp::IsBlockScaleSupported;
  static constexpr SfStrategy SfGenStrategy              = (!IsBlockScaleSupported) ? SfStrategy::None : SfStrategy::SfDGen;
  static constexpr int32_t SFD_VectorSize = IsBlockScaleSupported ? FusionOp::SFVecSize : 1;
  using ElementSFD = non_void_t<cute::remove_pointer_t<typename FusionOp::ElementBlockScaleFactor>, ElementD>;
  using Sm1xxBlockScaledOutputConfig= cutlass::detail::Sm1xxBlockScaledOutputConfig<
                                          SFD_VectorSize
                                        >;
  using Blk_MN = typename Sm1xxBlockScaledOutputConfig::Blk_MN;
  using Blk_SF = typename Sm1xxBlockScaledOutputConfig::Blk_SF;
  using OutputSFAtom = typename Sm1xxBlockScaledOutputConfig::SfAtom;
  std::vector<cutlass::HostTensor<ElementSFD, LayoutTagD>> tensors_SFD;
  std::vector<cutlass::HostTensor<ElementSFD, LayoutTagD>> references_SFD;
  cutlass::DeviceAllocation<ElementSFD *> device_tensors_SFD;

  using ElementCompute    = typename FusionOp::ElementCompute;
  using ElementScalar     = typename FusionOp::ElementScalar;
  using ElementBias       = non_void_t<typename FusionOp::ElementBias>;
  using ElementAux        = non_void_t<typename FusionOp::ElementAux>;
  using ElementAmax       = non_void_t<typename FusionOp::ElementAmax>;
  using LayoutTagAux      = non_void_t<typename FusionOp::GmemLayoutTagAux, LayoutTagD>;
  using ActivationFunctor = non_void_t<typename FusionOp::ActivationFn,
                              cutlass::epilogue::thread::Identity<ElementCompute>>;

  static constexpr bool IsBiasEnabled        = FusionOp::IsPerRowBiasSupported;
  static constexpr bool IsDeBiasEnabled      = FusionOp::IsDePerRowBiasSupported;
  static constexpr bool IsPerRowScaleEnabled = FusionOp::IsPerRowScaleSupported;
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
  cutlass::DeviceAllocation<InternalStrideC> stride_c_device;
  cutlass::DeviceAllocation<InternalStrideD> stride_d_device;

  std::vector<InternalStrideC> stride_c_host;
  std::vector<InternalStrideD> stride_d_host;

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
  std::vector<cutlass::HostTensor<ElementC, LayoutTagC>> tensors_C;
  cutlass::DeviceAllocation<const ElementC *> device_tensors_C;
  cutlass::HostTensor<ElementCompute, LayoutTagScalar> norm_constant;

  // Outputs
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_D;
  std::vector<cutlass::HostTensor<ElementAux , LayoutTagAux>> tensors_Aux;
  cutlass::DeviceAllocation<ElementAux *> device_tensors_Aux;
  cutlass::gemm::TagToStrideC_t<   LayoutTagAux   > stride_Aux;
  std::vector<cutlass::HostTensor<ElementD, LayoutTagD>> tensors_D;
  std::vector<cutlass::HostTensor<ElementD, LayoutTagD>> references_D;
  cutlass::DeviceAllocation<ElementD *> device_tensors_D;

  // References
  cutlass::HostTensor<ElementBias, LayoutTagVector> reference_dbias;
  std::vector<cutlass::HostTensor<ElementAux , LayoutTagAux>> references_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_D;

  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;
  // Are scalars copied to device memory before kernel launch
  ScalarLoc use_device_scalars = ScalarLoc::ON_HOST;
  // If per-row scale is enabled and this is disabled, alpha/beta are passed as a host or device scalar instead of device vector
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

  bool initialize(ProblemShapeType problem_shapes, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
    // Initialize Epilogue tensors

    tensors_C.clear();
    tensors_D.clear();
    references_D.clear();
    stride_c_host.clear();
    stride_d_host.clear();

    tensors_SFD.clear();
    references_SFD.clear();


    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = std::max(problem_shapes.groups(), L);

    for (int32_t i = 0; i < L; ++i) {
      auto [M, N, K, mock_L] = cute::append<4>(problem_shapes.get_host_problem_shape(i), 1);

      stride_c_host.push_back(cutlass::make_cute_packed_stride(InternalStrideC{}, {M, N, 1}));
      stride_d_host.push_back(cutlass::make_cute_packed_stride(InternalStrideD{}, {M, N, 1}));

      auto c_coord = cutlass::make_Coord(M, N);
      tensors_C.push_back(cutlass::HostTensor<ElementC, LayoutTagC>(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C)));
      tensors_D.push_back(cutlass::HostTensor<ElementD, LayoutTagD>(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D)));
      references_D.push_back(cutlass::HostTensor<ElementD, LayoutTagD>(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false));
      EXPECT_TRUE(initialize_tensor(tensors_C[i].host_view(), init_C, seed + 2020));
      tensors_C[i].host_view().at({0, 0}) = ElementC(1);

      cutlass::reference::host::TensorCopy(references_D[i].host_view(), tensors_C[i].host_view());
      tensors_C[i].sync_device();
      tensors_D[i].sync_device();
    }

    auto scalar_coord = cutlass::make_Coord(1);
    auto col_vector_coord = cutlass::make_Coord(M);
    if constexpr (IsPerRowScaleEnabled) {
      alpha.resize(col_vector_coord);
      EXPECT_TRUE(initialize_tensor(alpha.host_view(), init_scale, seed + 2023));
      if (vector_scale_mode == VectorScale::DISABLED) {
        beta.resize(scalar_coord, false);
        cutlass::reference::host::TensorFill(beta.host_view(), beta_);
      }
      else {
        beta.resize(col_vector_coord);
        EXPECT_TRUE(initialize_tensor(beta.host_view(), init_scale, seed + 2024));
      }
    }
    else {
      alpha.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      beta.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      cutlass::reference::host::TensorFill(alpha.host_view(), alpha_);
      cutlass::reference::host::TensorFill(beta.host_view(), beta_);
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

    if constexpr (IsBiasEnabled) {
      bias.resize(col_vector_coord);
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

    tensors_Aux.clear();
    references_Aux.clear();

    static_assert(!IsGroupGemm or (IsGroupGemm and !IsAuxInEnabled));

    if constexpr (IsAuxInEnabled) {
      auto aux_coord = cutlass::make_Coord(M, N);
      auto aux_layout = cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(aux_coord, typename LayoutTagAux::Stride{});
      for (int32_t i = 0; i < L; ++i) {
        tensors_Aux.push_back(cutlass::HostTensor<ElementAux , LayoutTagAux>(aux_coord, aux_layout));
        EXPECT_TRUE(initialize_tensor(tensors_Aux[i].host_view(), init_C, seed + 2023));
        tensors_Aux[i].sync_device();
      }
      stride_Aux = cutlass::make_cute_packed_stride(cutlass::gemm::TagToStrideC_t<LayoutTagAux>{}, cute::make_shape(M, N, 1));
    }

    static_assert(!IsGroupGemm or (IsGroupGemm and !IsAuxOutEnabled));

    if constexpr (IsAuxOutEnabled) {
      for (int32_t i = 0; i < L; ++i) {
        auto [M, N, K, mock_L] = cute::append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto aux_coord = cutlass::make_Coord(M, N);
        auto aux_layout = cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(aux_coord, typename LayoutTagAux::Stride{});
        tensors_Aux.push_back(cutlass::HostTensor<ElementAux , LayoutTagAux>(aux_coord, aux_layout));
        references_Aux.push_back(cutlass::HostTensor<ElementAux , LayoutTagAux>(aux_coord, aux_layout, false));
        tensors_Aux[i].sync_device();
      }

      stride_Aux = cutlass::make_cute_packed_stride(cutlass::gemm::TagToStrideC_t<LayoutTagAux>{}, cute::make_shape(M, N, 1));

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
      for (int32_t i = 0; i < L; ++i) {
        auto [M, N, K, _] = cute::append<4>(problem_shapes.get_host_problem_shape(i), 1);
        // If block scaled output is supported we always have at least 1 SFD
        auto m_blks = cutlass::ceil_div(M, cute::size<0>(cute::shape(OutputSFAtom{})));
        auto n_blks = cutlass::ceil_div(N, cute::size<1>(cute::shape(OutputSFAtom{})));
        auto sfd_coord = [&] () {
            return cutlass::make_Coord(m_blks * Blk_MN{}, n_blks * Blk_SF{});
        }();
        tensors_SFD.push_back(cutlass::HostTensor<ElementSFD, LayoutTagD>(sfd_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(sfd_coord, stride_factor_D)));
        references_SFD.push_back(cutlass::HostTensor<ElementSFD, LayoutTagD>(sfd_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(sfd_coord, stride_factor_D), false));
        tensors_SFD[i].sync_device();
      }
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
      ProblemShapeType problem_shapes,
      ElementScalar alpha,
      ElementScalar beta,
      int batch) {
    tensors_D[batch].sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_C[batch].host_view()), 0);

    if (tensors_D[batch].size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_D[batch].host_view()), 0);
    }

    if (references_D[batch].size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(references_D[batch].host_view()), 0);
    }

    bool passed = equality_check(references_D[batch].host_view(), tensors_D[batch].host_view());
    if(!passed) {
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
      tensors_Aux[batch].sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensors_Aux[batch].host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(references_Aux[batch].host_view()), 0);
      passed &= equality_check(references_Aux[batch].host_view(), tensors_Aux[batch].host_view());
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
      tensors_SFD[batch].sync_host();
      bool passed_sf = equality_check(references_SFD[batch].host_view(), tensors_SFD[batch].host_view());
      if(!passed_sf) {
        std::cout<<"SF is incorrect"<<std::endl;
      }
      passed &= passed_sf;
    }


    return passed;
  }

  void print_tensors(std::ofstream& file, int batch) {
    auto coord_0 = cutlass::make_Coord(0);
    if constexpr (IsScaleFactorEnabled) {
      file
        << ", scale_a: " << scale_A.at(coord_0)
        << ", scale_b: " << scale_B.at(coord_0)
        << ", scale_c: " << scale_C.at(coord_0);
    }
    if constexpr (IsPerRowScaleEnabled) {
      file << "\n\nvalpha = \n" << alpha.host_view();
      file << "\n\nvbeta = \n" << beta.host_view();
    }
    else {
      file
        << ", alpha: " << alpha.at(coord_0) << ", beta: " << beta.at(coord_0);
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

    if constexpr (IsBiasEnabled) {
      file << "\n\nBias = \n" << bias.host_view();
    }

    if constexpr (IsAuxInEnabled) {
      file << "\n\nAux Input = \n" << tensors_Aux[batch].host_view();
    }

    if constexpr (IsDeBiasEnabled) {
      file << "\n\nReference dBias = \n" << reference_dbias.host_view();
      file << "\n\nComputed dBias = \n" << bias.host_view();
    }

    if constexpr (IsAuxOutEnabled) {
      file
        << "\n\nReference Aux =\n" << references_Aux[batch].host_view()
        << "\n\nComputed Aux =\n" << tensors_Aux[batch].host_view();
    }

    if constexpr (IsBlockScaleSupported) {
      file
        << "\n\nReference SFD =\n" << references_SFD[batch].host_view()
        << "\n\nComputed SFD =\n" << tensors_SFD[batch].host_view();
    }

    file
    << "\nC =\n" << tensors_C[batch].host_view()
    << "\n\nReference =\n" << references_D[batch].host_view()
    << "\n\nComputed =\n" << tensors_D[batch].host_view();

  }

  Arguments to_args(ProblemShapeType problem_shapes) {
    auto coord_0 = cutlass::make_Coord(0);
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = std::max(problem_shapes.groups(), L);

    std::vector<ElementC *> ptr_C_host(L);
    std::vector<ElementD *> ptr_D_host(L);

    for (int32_t i = 0; i < L; ++i) {
      ptr_C_host.at(i) = tensors_C[i].device_data();
      ptr_D_host.at(i) = tensors_D[i].device_data();
    }

    device_tensors_C.reset(L);
    device_tensors_C.copy_from_host(ptr_C_host.data());

    device_tensors_D.reset(L);
    device_tensors_D.copy_from_host(ptr_D_host.data());

    stride_c_device.reset(problem_shapes.groups());
    stride_c_device.copy_from_host(stride_c_host.data());

    stride_d_device.reset(problem_shapes.groups());
    stride_d_device.copy_from_host(stride_d_host.data());

    std::vector<ElementAux *> ptr_Aux_host(L);
    if constexpr (IsAuxInEnabled || IsAuxOutEnabled) {
      for (int32_t i = 0; i < L; ++i) {
        ptr_Aux_host.at(i) = tensors_Aux[i].device_data();
      }
      device_tensors_Aux.reset(L);
      device_tensors_Aux.copy_from_host(ptr_Aux_host.data());
    }

    auto device_tensors_C_ptr = cute::is_void_v<typename kernel::ElementC> ? nullptr :
                                  reinterpret_cast<typename kernel::ElementC const**>(device_tensors_C.get());

    Arguments arguments;
    if constexpr (IsGroupGemm) {
      arguments =
      {
        {},
        device_tensors_C_ptr, stride_c_device.get(), device_tensors_D.get(), stride_d_device.get()
      };
    }
    else {
      arguments =
      {
        {},
        device_tensors_C_ptr, stride_c_host[0], device_tensors_D.get(), stride_d_host[0]
      };
    }

    auto &fusion_args = arguments.thread;
    if constexpr (IsLegacy) {
      arguments.thread = {
        alpha.at(coord_0),
        beta.at(coord_0),
        alpha.device_data(),
        beta.device_data()
      };
      arguments.ptr_Bias = bias.device_data();
      arguments.ptr_T = device_tensors_Aux.get();
    }
    else {
      fusion_args.alpha = alpha.at(coord_0);
      fusion_args.beta = beta.at(coord_0);

      fusion_args.alpha_ptr = alpha.device_data();
      // can_implement requires beta_ptr to not be set if its voidC
      fusion_args.beta_ptr = cute::is_void_v<typename kernel::ElementC> ? nullptr :
                               beta.device_data();

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

      if constexpr (IsBiasEnabled) {
        fusion_args.bias_ptr = bias.device_data();
      }

      if constexpr (IsDeBiasEnabled) {
        fusion_args.dbias_ptr = bias.device_data();
      }

      // example of how to set kernel activation arguments
      // see ActivationFunctor::Arguments in activation.h for definition
      // if Arguments doesn't exist then fusion_args.activation is empty
      if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ScaledGELU_taylor<ElementCompute>>) {
        fusion_args.activation.scale = ElementCompute(1);
      }

      // Treat Clamp as ReLU
      if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Clamp<ElementCompute>>) {
        fusion_args.activation.lower_bound = 0;
        fusion_args.activation.upper_bound = std::numeric_limits<ElementCompute>::max();
      }

      if constexpr (IsAbsMaxEnabledD) {
        fusion_args.amax_D_ptr = abs_max_D.device_data();
      }

      if constexpr (IsAuxInEnabled) {
        fusion_args.aux_ptr = device_tensors_Aux.get();
        fusion_args.dAux = stride_Aux;
      }

      if constexpr (IsAuxOutEnabled) {
        fusion_args.aux_ptr = device_tensors_Aux.get();
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
        std::vector<ElementSFD *> ptr_SFD_host(L);
        for (int32_t i = 0; i < L; ++i) {
          ptr_SFD_host.at(i) = tensors_SFD[i].device_data();
        }
        device_tensors_SFD.reset(L);
        device_tensors_SFD.copy_from_host(ptr_SFD_host.data());

        arguments.thread.block_scale_factor_ptr = device_tensors_SFD.get();
        arguments.thread.norm_constant_ptr = norm_constant.device_data();
      }

    }

    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_shapes, int batch) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_shapes.get_host_problem_shape(batch), 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    auto coord_0 = cutlass::make_Coord(0);
    auto C = cute::make_tensor(detail::make_iterator(tensors_C[batch].host_data()),
        cute::make_layout(cute::make_shape(M, N, 1), stride_c_host[batch]));
    auto D = cute::make_tensor(detail::make_iterator(references_D[batch].host_data()),
        cute::make_layout(cute::make_shape(M, N, 1), stride_d_host[batch]));
    auto Bias = cute::make_tensor(detail::make_iterator(IsDeBiasEnabled ? reference_dbias.host_data() : bias.host_data()),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    auto Aux_layout = cute::make_layout(cute::make_shape(M, N, 1), stride_Aux);
    auto Aux = [&]() {
      auto ptr = recast_ptr<ElementAux>(nullptr);
      if (IsAuxInEnabled) {
        ptr = detail::make_iterator(tensors_Aux[batch].host_data());
      } else if (IsAuxOutEnabled) {
        ptr = detail::make_iterator(references_Aux[batch].host_data());
      }
      return cute::make_tensor(ptr, Aux_layout);
    }();
    auto Valpha = cute::make_tensor(detail::make_iterator(alpha.host_data()),
        cute::make_layout(cute::make_shape(M, N, cute::_1{}), cute::make_stride(cute::_1{}, cute::_0{}, M)));
    auto Vbeta = cute::make_tensor(detail::make_iterator(beta.host_data()),
        cute::make_layout(cute::make_shape(M, N, cute::_1{}), cute::make_stride(cute::_1{}, cute::_0{}, N)));

    auto SfD = [&](){
      if constexpr (IsBlockScaleSupported) {
        auto tensor = make_tensor(detail::make_iterator(references_SFD[batch].host_data()),
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
      ActivationFunctor
      , decltype(SfD)
      , Int<SFD_VectorSize>
      , cutlass::plus<ElementCompute>
      , false
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

    if constexpr (IsBiasEnabled or IsDeBiasEnabled) {
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

    if constexpr (IsPerRowScaleEnabled) {
      epilogue_params.Valpha = Valpha;
      if (vector_scale_mode == VectorScale::ENABLED) {
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

  uint32_t sm_count;
  // Used to force multi-wave tests for persistent kernel schedules
  constexpr static int MaxSmCount = 16;
  static constexpr uint64_t kDefaultSeed = 4096;
  static constexpr uint32_t mma_promotion_interval = 4;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

  HostCollectiveMainloopType collective_mma_inputs;
  CollectiveEpilogue collective_epilogue;

  static constexpr bool IsGroupGemm = CollectiveEpilogue::IsGroupGemm;

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
  bool initialize(ProblemShapeType problem_shapes, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
    collective_mma_inputs.initialize(problem_shapes);
    collective_epilogue.initialize(problem_shapes, alpha_, beta_);

    return true;
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      ProblemShapeType problem_shapes,
      ElementScalar alpha,
      ElementScalar beta,
      int batch)
  {
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(batch), 1);

    bool passed = collective_mma_inputs.compare_reference(problem_shapes, batch);
    passed &= collective_epilogue.compare_reference(problem_shapes, alpha, beta, batch);
    EXPECT_TRUE(passed);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << batch << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << batch
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      collective_mma_inputs.print_tensors(file, batch);
      collective_epilogue.print_tensors(file, batch);
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
      ProblemShapeType problem_shapes,
      ElementScalar alpha,
      ElementScalar beta)
  {
    using namespace cute;
    auto [M, N, K, L] = cute::append<4>(problem_shapes.get_host_problem_shape(0), 1);
    L = std::max(problem_shapes.groups(), L);

    bool passed = true;
    for (int32_t i = 0; i < L; ++i) {
      auto mainloop_params = collective_mma_inputs.to_host_args(problem_shapes, i);
      auto epilogue_params = collective_epilogue.to_host_args(problem_shapes, i);

      cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

      passed &= compare_reference(problem_shapes, alpha, beta, i);
    }
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

  /// Executes one test
  bool run(
    ProblemShapeType problem_shapes,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    detail::Iterations iterations = detail::Iterations{}
    )
  {

    // Fail test if insufficient CUDA device
    if (!sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }

    if (!this->initialize(problem_shapes, alpha, beta)) {
      std::cerr << "Initialization failed \n";
      return false;
    }

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    this->sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    hw_info.sm_count = this->sm_count;

    typename HostCollectiveMainloopType::Arguments mainloop_args;

    mainloop_args = collective_mma_inputs.to_args(problem_shapes);

    if constexpr (IsGroupGemm) {
      arguments =
      {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        problem_shapes,
        mainloop_args,
        collective_epilogue.to_args(problem_shapes),
        hw_info
      };
    }
    else {
      arguments =
      {
        cutlass::gemm::GemmUniversalMode::kArray,
        problem_shapes,
        mainloop_args,
        collective_epilogue.to_args(problem_shapes),
        hw_info
      };
    }


    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return false;
    }

    //
    // Run the GEMM
    //

    cudaError_t result;
    status = gemm_op.initialize(arguments, workspace.get());
    status = gemm_op.run();
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
      return false;
    }

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //
    bool passed = this->verify(problem_shapes, alpha, beta);
    if (!passed) {
      std::cout << "Error : Failed : with alpha: " << alpha << ", beta: " << beta
                << "\n";
    }

    return passed;
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
>
struct Testbed3x {

  using TestBedImpl = typename detail::TestbedImpl<
                        Gemm,
                        ActivationFunctor,
                        force_legacy_epilogue,
                        ElementA,
                        ElementB
                        >;
  using Kernel      = typename Gemm::GemmKernel;
  using Epilogue    = typename Gemm::GemmKernel::CollectiveEpilogue;

  using ElementAccumulator   = typename TestBedImpl::ElementAccumulator;
  using ElementCompute       = typename TestBedImpl::ElementCompute;
  using ElementScalar        = typename TestBedImpl::ElementScalar;

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

  static constexpr bool IsGroupGemm = TestBedImpl::IsGroupGemm;

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
   typename TestBedImpl::ProblemShapeType problem_shapes,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    detail::Iterations iterations = detail::Iterations{}
    )
  {
    return impl_.run(
        problem_shapes, alpha, beta, iterations);
  }
};

template <
  typename Gemm,
  template <class T> class ActivationFunctor = cutlass::epilogue::thread::Identity
>
bool TestAll(double alpha = 1.0, double beta = 0.0, CheckEquality check_relative_equality = CheckEquality::RELATIVE) {
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  Testbed3x<Gemm, ActivationFunctor> testbed(check_relative_equality, ScalarLoc::ON_DEVICE, VectorScale::DISABLED);

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {max_alignment, 512 - 3 * max_alignment};
  std::vector<int> problem_size_n = {max_alignment, 512 - 2 * max_alignment};

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {max_alignment, TileShapeK * (Stages + 1) - max_alignment};

  int batches[] = {5, 10};

  bool passed = true;

  for (int batch : batches) {
    for (int m : problem_size_m) {
      for (int n : problem_size_n) {
        for (int k : problem_size_k) {

          if constexpr (Testbed3x<Gemm, ActivationFunctor>::IsGroupGemm) {
            std::vector<typename ProblemShapeType::UnderlyingProblemShape> problem_sizes_host;
            cutlass::DeviceAllocation<typename ProblemShapeType::UnderlyingProblemShape> problem_sizes_device;

            for (int i = 0; i < batch; ++i) {
              problem_sizes_host.push_back({m * ((i % 3) + 1), n * ((i % 4) + 1), k * ((i % 5) + 1)});
            }

            problem_sizes_device.reset(problem_sizes_host.size());
            problem_sizes_device.copy_from_host(problem_sizes_host.data());

            passed = testbed.run(
              ProblemShapeType{static_cast<int>(problem_sizes_host.size()), problem_sizes_device.get(), problem_sizes_host.data()},
              cutlass::from_real<ElementScalar>(alpha),
              cutlass::from_real<ElementScalar>(beta)
            );
          }
          else {
            ProblemShapeType problem_size{{m, n, k, batch}};

            passed = testbed.run(
              problem_size,
              cutlass::from_real<ElementScalar>(alpha),
              cutlass::from_real<ElementScalar>(beta)
            );
          }

          if (!passed) {
            std::cout << __FILE__ << ':' << __LINE__ << " : GEMM MNKL " << m << " " << n << " " << k << " " << batch << " FAILED.\n";
            return false;
          }
        } // k
      } // n
    } // m
  } // batch

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool force_legacy_epilogue = false, bool apply_alignment_offset = false>
bool TestSmall(double alpha = 1.0, double beta = 1.0,
  CheckEquality check_relative_equality = CheckEquality::RELATIVE,
  ScalarLoc use_device_scalars = ScalarLoc::ON_DEVICE,
  VectorScale vector_scale_mode = VectorScale::ENABLED,
  std::vector<int> override_problem_size_k = {}) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using ElementA = typename Gemm::GemmKernel::ElementA;
  using ElementB = typename Gemm::GemmKernel::ElementB;
  using TiledMma = typename Gemm::GemmKernel::TiledMma;

  static constexpr bool IsF8F6F4 = cutlass::gemm::collective::detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();
  // For fp4 and fp6 kernels, the min alignment_input is 128 elements, so we don't need to add alignment_input in test problem sizes.  
  int alignment_bits_a = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
  int alignment_input_a = (alignment_bits_a / cute::sizeof_bits<ElementA>::value == 128) ? 0 : (alignment_bits_a / cute::sizeof_bits<ElementA>::value);
  
  int alignment_bits_b = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();
  int alignment_input_b = (alignment_bits_b / cute::sizeof_bits<ElementB>::value == 128) ? 0 : (alignment_bits_b / cute::sizeof_bits<ElementB>::value);
  
  int alignment_input = (alignment_input_a == 0 || alignment_input_b == 0) ? 0 : std::max(alignment_input_a, alignment_input_b);

  if constexpr (apply_alignment_offset) {
    // If BlockScaled, then min alignment is SFVecSize
    static constexpr bool IsBlockScaleSupported = Gemm::EpilogueOutputOp::IsBlockScaleSupported;
    static constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;
    if constexpr (IsBlockScaleSupported) {
      alignment_input = cutlass::round_up(alignment_input, SFVecSize);
    }
  }


  using CtaShape_MNK = typename Gemm::GemmKernel::CollectiveMainloop::CtaShape_MNK;
  using DispatchPolicy = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy;
  CtaShape_MNK cta_shape;
  Testbed3x<Gemm, cutlass::epilogue::thread::Identity, force_legacy_epilogue> testbed(check_relative_equality, use_device_scalars, vector_scale_mode);
  // For Ptr-Array and Grouped GEMM ideally we need to know SM count at runtime
  static constexpr int SmCount = 16;

  float waves[] = {0.5, 2.5};
  int batches[] = {3};
  int cluster_m = 1;
  int cluster_n = 1;

  std::vector<int> problem_size_k;
  if (override_problem_size_k.empty()) {
    // this is to test with min alignment
    problem_size_k = {256 - alignment_input, 512 + alignment_input};
  }
  else {
    problem_size_k = override_problem_size_k;
  }

  if constexpr(DispatchPolicy::ArchTag::kMinComputeCapability >= 90) {
    typename DispatchPolicy::ClusterShape cluster_shape;
    cluster_m = cute::size<0>(cluster_shape);
    cluster_n = cute::size<1>(cluster_shape);
  }

  bool passed = true;

  for (int batch : batches) {
    for (float wave : waves) {
      for (int k : problem_size_k) {
        int grid_m, grid_n = 0;
        float num_grid = wave * SmCount;

        if (cluster_m >= cluster_n) {
          grid_m = cluster_m;
          grid_n = static_cast<int>(num_grid) / grid_m;
          // Align grid_n to cluster_n
          grid_n = std::max((grid_n + cluster_n - 1 ) / cluster_n * cluster_n, 1);
        }
        else {
          grid_n = cluster_n;
          grid_m = static_cast<int>(num_grid) / grid_n;
          // Align grid_m to cluster_m
          grid_m = std::max((grid_m + cluster_m - 1 ) / cluster_m * cluster_m, 1);
        }

        int m = grid_m * cute::size<0>(cta_shape) - alignment_input; // this is just to test with unusual problem shapes
        int n = grid_n * cute::size<1>(cta_shape) + alignment_input;

        if constexpr (Testbed3x<Gemm, cutlass::epilogue::thread::Identity, force_legacy_epilogue>::IsGroupGemm) {
          std::vector<typename ProblemShapeType::UnderlyingProblemShape> problem_sizes_host;
          cutlass::DeviceAllocation<typename ProblemShapeType::UnderlyingProblemShape> problem_sizes_device;
          for (int i = 0; i < batch; ++i) {
            problem_sizes_host.push_back({m * ((i % 2) + 1), n * ((i % 3) + 1), k * ((i % 2) + 1)});
          }
          problem_sizes_device.reset(problem_sizes_host.size());
          problem_sizes_device.copy_from_host(problem_sizes_host.data());

          ProblemShapeType problem_shapes{batch, problem_sizes_device.get(), problem_sizes_host.data()};

          if (CUTLASS_DEBUG_TRACE_LEVEL > 0) {
            for (int i = 0; i < batch; ++i) {
              std::cout << "problem_shapes : "  << problem_shapes.get_host_problem_shape(i) << " \n";
            }
          }
          passed = testbed.run(
            problem_shapes,
            cutlass::from_real<ElementScalar>(alpha),
            cutlass::from_real<ElementScalar>(beta)
          );
        }
        else {
          ProblemShapeType problem_shapes{{m, n, k, batch}};
          if (CUTLASS_DEBUG_TRACE_LEVEL > 0) {
            std::cout << "problem_shapes : "  << problem_shapes.get_host_problem_shape() << " \n";
          }
          passed = testbed.run(
            problem_shapes,
            cutlass::from_real<ElementScalar>(alpha),
            cutlass::from_real<ElementScalar>(beta)
          );
        }

        if (!passed) {
          std::cout << __FILE__ << ':' << __LINE__ << " : GEMM MNK " << m << " " << n << " " << k << " FAILED.\n";
          return false;
        }
      } // k
    } // waves
  } // batches

  return passed;
}

template <typename Gemm, bool force_legacy_epilogue = false, bool apply_alignment_offset = true>
bool TestSmallFusion(double alpha = 1.0, double beta = 0.0,
    CheckEquality check_relative_equality = CheckEquality::RELATIVE,
    ScalarLoc use_device_scalars = ScalarLoc::ON_DEVICE,
    VectorScale vector_scale_mode = VectorScale::ENABLED) {
  return TestSmall<Gemm, force_legacy_epilogue, apply_alignment_offset>(
    alpha, beta, check_relative_equality, use_device_scalars, vector_scale_mode);
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
