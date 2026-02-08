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
    \brief Testbed and host reference for EVT unittest
*/


#pragma once
#include "gemm_testbed_3x.hpp" 

namespace test {
namespace gemm {
namespace device {

/// Host-side tapply, tapply in cute is HOST_DEVICE
template <class T, class F, class G, int... I>
constexpr auto
tapply(T&& t, F&& f, G&& g, cute::seq<I...>)
{
  return g(f(std::get<I>(static_cast<T&&>(t)))...);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT: Base class for EVT Node

template < class ElementCompute_ >
class HostEVTNodeBase {
public:
  using ElementCompute = ElementCompute_;

private:
  bool check_relative_equality_;
  // Factors used for calculating relative equality. These default
  // values are borrowed from those used by default in the CUTLASS
  // profiler for performing relative equality checks.
  float epsilon_ = 0.05f;
  float nonzero_floor_ = 1.0f / 256.0f;

public:
  HostEVTNodeBase(){}
  HostEVTNodeBase(bool check_relative_equality):
    check_relative_equality_(check_relative_equality) { }


  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {
    if (check_relative_equality_) {
      return cutlass::reference::host::TensorRelativelyEquals(
        lhs, rhs, Element(epsilon_), Element(nonzero_floor_)
      );
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  void* get_tensor_C_ptr() {
    return nullptr;
  }

  void* get_tensor_D_ptr() {
    return nullptr;
  }

  bool compare_reference(std::stringstream& error_ss) {
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Accumulator

template< class ElementCompute = float >
class HostAccumulator: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;

  struct Arguments { };
  
public:
  HostAccumulator(){}
  template<typename ProblemShapeType>
  HostAccumulator(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    :Base(check_relative_equality) {}

  template<typename ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> accumulator_converter;
    return accumulator_converter(acc);
  }

  Arguments get_arguments() {
    return Arguments{};
  }

  auto get_flatten_arguments() {
    return cute::make_tuple();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Scalar Broadcast

template <
  int Value,
  int BroadcastCount = 1,
  class StrideMNL = cute::Stride<cute::_0,cute::_0,cute::_0>,
  template <class> class ReductionFn = cutlass::multiplies,
  class ElementCompute = float
>
class HostScalarBroadcast : public HostEVTNodeBase<ElementCompute> {
public:

  using Base = HostEVTNodeBase<ElementCompute>;
  struct Arguments {
    ElementCompute scalar[BroadcastCount] = {0};
    ElementCompute const* scalar_ptrs[BroadcastCount] = { nullptr };
    StrideMNL dScalar[BroadcastCount] = {};
  };
private:
  ElementCompute scalar_{};
  StrideMNL dScalar{};
  ElementCompute scalar_reduced_{};
public:
  HostScalarBroadcast(){}

  template<typename ProblemShapeType>
  HostScalarBroadcast(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    : Base(check_relative_equality), scalar_(ElementCompute(Value)) {
    scalar_ = ElementCompute(Value);
    scalar_reduced_ = scalar_;
    for (int i = 1; i < BroadcastCount; ++i) {
      scalar_reduced_ = ReductionFn<ElementCompute>{}(scalar_reduced_, ElementCompute(Value));
    }
  }
  
  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    
    return scalar_reduced_;
  }

  bool compare_reference(std::stringstream& error_ss) {
    error_ss << "Scalar: " << float(scalar_) << "\n\n";
    return true;
  }

  Arguments get_arguments() {
    if constexpr (BroadcastCount == 1)
      return Arguments{{scalar_}, {nullptr}, {dScalar}};
    else if constexpr (BroadcastCount == 2)
      return Arguments{{scalar_, scalar_}, {nullptr, nullptr}, {dScalar,  dScalar}};
    else if constexpr (BroadcastCount == 3)
      return Arguments{{scalar_, scalar_, scalar_}, {nullptr, nullptr, nullptr}, {dScalar, dScalar, dScalar}};
    else
      return Arguments{{scalar_}, {nullptr}, {dScalar}};
  }

  auto get_flatten_arguments() {
    if constexpr (BroadcastCount == 1) {
      return cute::make_tuple(scalar_, nullptr);
    } 
    else if constexpr (BroadcastCount == 2) {
      return cute::make_tuple(scalar_, scalar_, nullptr, nullptr);
    } 
    else if constexpr (BroadcastCount == 3) {
      return cute::make_tuple(scalar_, scalar_, scalar_, nullptr, nullptr, nullptr);
    } 
    else {
      return cute::make_tuple(scalar_, nullptr);
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Row Broadcast
template <
  typename ElementBias_,
  typename StrideMNL = cute::Stride<cute::_0,cute::_1,cute::_0>,
  typename ElementCompute = float
>
class HostRowBroadcast: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;
  using ElementBias = ElementBias_;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;
  
  struct Arguments {
    ElementBias const* ptr_row = nullptr;
    ElementBias null_default = ElementBias(0);
    StrideMNL dRow = {};
  };
private:
  cutlass::NumericConverter<ElementCompute, ElementBias> bias_converter_;
  cutlass::HostTensor<ElementBias, LayoutTagVector> bias_;
  int N_;
public:
  HostRowBroadcast(){}
  template<typename ProblemShapeType>
  HostRowBroadcast(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    : Base(check_relative_equality) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    N_ = cute::get<1>(problem_shape_MNKL);
    bias_.resize(cutlass::Coord<1>(N_));
    
    EXPECT_TRUE(
      detail::initialize_tensor(
        bias_.host_view(), cutlass::Distribution::Uniform, 
        seed
      )
    );
    bias_.sync_device();
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    auto TensorBias = cute::make_tensor(bias_.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{}, N_)));
    
    return bias_converter_(TensorBias(1, n + n_b));
  }

  bool compare_reference(std::stringstream& error_ss) {
    error_ss
      << "PerColumnBias = \n" << bias_.host_view() << "\n\n";
    return true;
  }

  Arguments get_arguments() {
    return {bias_.device_data()};
  }

  auto get_flatten_arguments() {
    return cute::make_tuple(bias_.device_data(), ElementBias(0), StrideMNL{});
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Column Broadcast
template <
  typename ElementBias_,
  typename StrideMNL = cute::Stride<cute::_1,cute::_0,cute::_0>,
  typename ElementCompute = float
>
class HostColBroadcast: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;
  using ElementBias = ElementBias_;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;
  
  struct Arguments {
    ElementBias const* ptr_row = nullptr;
    ElementBias null_default = ElementBias(0);
    StrideMNL dRow = {};
  };
private:
  cutlass::NumericConverter<ElementCompute, ElementBias> bias_converter_;
  cutlass::HostTensor<ElementBias, LayoutTagVector> bias_;
  int M_;
public:
  HostColBroadcast(){}
  template<typename ProblemShapeType>
  HostColBroadcast(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    : Base(check_relative_equality) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    M_ = cute::get<0>(problem_shape_MNKL);
    bias_.resize(cutlass::Coord<1>(M_));
    
    EXPECT_TRUE(
      detail::initialize_tensor(
        bias_.host_view(), cutlass::Distribution::Uniform, 
        seed
      )
    );
    bias_.sync_device();
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    auto TensorBias = cute::make_tensor(bias_.host_data(),
      cute::make_layout(cute::make_shape(M_, cute::_1{})));
    
    return bias_converter_(TensorBias(m + m_b, 1));
  }

  bool compare_reference(std::stringstream& error_ss) {
    error_ss
      << "PerRowBias = \n" << bias_.host_view() << "\n\n";
    return true;
  }

  Arguments get_arguments() {
    return {bias_.device_data()};
  }

  auto get_flatten_arguments() {
    return cute::make_tuple(bias_.device_data(), ElementBias(0), StrideMNL{});
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Aux Load

template <
  typename ElementAuxLoad_,
  typename LayoutTagAux_,
  bool isC = false,
  typename ElementCompute = float
>
class HostAuxLoad: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;
  using ElementAuxLoad = ElementAuxLoad_;
  using LayoutTagAux = LayoutTagAux_;

  using StrideAux = cutlass::gemm::TagToStrideC_t<LayoutTagAux>;
  struct Arguments_Aux {
    ElementAuxLoad const *ptr_aux = nullptr;
    ElementAuxLoad null_default = ElementAuxLoad(0);
    StrideAux dAux = {};
  };

  struct Arguments_C {};

  using Arguments = cute::conditional_t<isC, Arguments_C, Arguments_Aux>;

private:
  cutlass::NumericConverter<ElementCompute, ElementAuxLoad> aux_load_converter_;
  cutlass::HostTensor<ElementAuxLoad, LayoutTagAux> tensor_aux_load_;

  int M_, N_, L_;

  StrideAux stride_aux_;
public:
  HostAuxLoad(){}
  template<typename ProblemShapeType>
  HostAuxLoad(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    : Base(check_relative_equality) {
    auto problem_shape_NMKL = cute::append<4>(problem_size, 1);
    auto [M_, N_, K, L_] = problem_shape_NMKL;
    auto aux_coord = cutlass::make_Coord(M_ * L_, N_);
    tensor_aux_load_.resize(
      aux_coord, 
      cutlass::layout::Affine2Layout_Factory<LayoutTagAux>::layout_factory(
        aux_coord, typename LayoutTagAux::Stride()
      )
    );
    EXPECT_TRUE(
      detail::initialize_tensor(
        tensor_aux_load_.host_view(), 
        cutlass::Distribution::Uniform, 
        seed
      )
    );
    tensor_aux_load_.sync_device();
    stride_aux_ = cutlass::make_cute_packed_stride(StrideAux{}, cute::make_shape(M_, N_, L_));
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {

    
    auto TensorAuxLoad = cute::make_tensor(tensor_aux_load_.host_data(),
      cute::make_layout(cute::make_shape(M_, N_, L_), stride_aux_));
    return aux_load_converter_(TensorAuxLoad(m + m_b, n + n_b, l));
  }

  bool compare_reference(std::stringstream& error_ss) {
    if constexpr (!isC) {
      error_ss
        << "AuxLoad = \n" << tensor_aux_load_.host_view()<< "\n\n";
    }
    return true;
  }

  void* get_tensor_C_ptr() {
    if constexpr (isC) {
      return static_cast<void*>(tensor_aux_load_.device_data());
    } 
    else {
      return nullptr;
    }
  }

  Arguments get_arguments() {
    if constexpr (isC)
      return {};
    else
      return {tensor_aux_load_.device_data(), ElementAuxLoad(0), stride_aux_};
  }

  auto get_flatten_arguments() {
    if constexpr (isC)
      return cute::make_tuple();
    else
      return cute::make_tuple(tensor_aux_load_.device_data(), ElementAuxLoad(0), stride_aux_);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Compute

template<typename T>
T* findNonNullPtr(T* first_ptr) {
  return first_ptr;
}

template <typename T, typename... Args>
T* findNonNullPtr(T* first_ptr, Args... args) {
  if (first_ptr) {
    return first_ptr;
  }
  return findNonNullPtr(args...);
}

template <
  template <class> class ComputeOp_,
  typename ElementCompute = float
>
class HostCompute: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;
  using ComputeOp = ComputeOp_<ElementCompute>;

  struct Arguments {
    struct OpArgs {} op;
  };
private:
  ComputeOp op_;
public:
  HostCompute(){}
  template <typename ProblemShapeType>
  HostCompute(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024):
    Base(check_relative_equality) { }

  template <class ElementAccumulator, typename... Args>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, Args... frg_inputs) {
    return op_(frg_inputs...);
  }

  Arguments get_arguments(){
    return {};
  }

  auto get_flatten_arguments() {
    return cute::make_tuple();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Aux Store

template <
  class ElementAuxStore_,
  typename LayoutTagAux_,
  bool isD = false,
  bool isRelu = false,
  typename ElementCompute = float
>
class HostAuxStore: public HostEVTNodeBase<ElementCompute> {
public:
  using ElementAuxStore = ElementAuxStore_;
  using LayoutTagAux = LayoutTagAux_;

  using Base = HostEVTNodeBase<ElementCompute>;

  using StrideAux = cutlass::gemm::TagToStrideC_t<LayoutTagAux>;
  struct Arguments_Aux {
    struct OpArgs {
      ElementAuxStore* ptr_aux = nullptr;
      StrideAux dAux = {};
    } op;
  };

  struct Arguments_D {};

  using Arguments = cute::conditional_t<isD, Arguments_D, Arguments_Aux>;


private:
  cutlass::NumericConverter<ElementAuxStore, ElementCompute> destination_converter_;
  cutlass::HostTensor<ElementAuxStore, LayoutTagAux> tensor_aux_store_;
  cutlass::HostTensor<ElementAuxStore, LayoutTagAux> reference_aux_store_;
  int M_, N_, L_;
  StrideAux stride_aux_;
public:
  HostAuxStore(){}
  template <typename ProblemShapeType>
  HostAuxStore(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024):
    Base(check_relative_equality) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M_, N_, K, L_] = problem_shape_MNKL;
    auto aux_coord = cutlass::make_Coord(M_ * L_, N_);
    tensor_aux_store_.resize(
      aux_coord, 
      cutlass::layout::Affine2Layout_Factory<LayoutTagAux>::layout_factory(
        aux_coord, typename LayoutTagAux::Stride()
      )
    );

    reference_aux_store_.resize(
      aux_coord,
      cutlass::layout::Affine2Layout_Factory<LayoutTagAux>::layout_factory(
        aux_coord, typename LayoutTagAux::Stride()
      )
    );
    tensor_aux_store_.sync_device();
    stride_aux_ = cutlass::make_cute_packed_stride(StrideAux{}, cute::make_shape(M_, N_, L_));
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {

    auto TensorAuxStore = cute::make_tensor(detail::make_iterator(static_cast<ElementAuxStore*>(reference_aux_store_.host_data())),
      cute::make_layout(cute::make_shape(M_, N_, L_), stride_aux_));
    if constexpr (isRelu)
      TensorAuxStore(m + m_b, n + n_b, l) = destination_converter_(child_0_result >= 0);
    else
      TensorAuxStore(m + m_b, n + n_b, l) = destination_converter_(child_0_result);
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    // Verify the store node
    tensor_aux_store_.sync_host();

    bool equal = this->equality_check(reference_aux_store_.host_view(), tensor_aux_store_.host_view());
    if (!equal) {
      error_ss 
        << "\n\nReference =\n" << reference_aux_store_.host_view()
        << "\n\nComputed =\n" << tensor_aux_store_.host_view() << "\n\n";
    }
    return equal;
  }

  void* get_tensor_D_ptr() {
    if constexpr (isD) 
      return static_cast<void*>(tensor_aux_store_.device_data());
    else
      return nullptr;
  }

  Arguments get_arguments() {
    if constexpr (isD) {
      return {};
    } 
    else {
      return {tensor_aux_store_.device_data(), stride_aux_};
    }
  }

  auto get_flatten_arguments() {
    if constexpr (isD) {
      return cute::make_tuple();
    } 
    else {
      return cute::make_tuple(tensor_aux_store_.device_data(), stride_aux_);
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Row Reduce

template <
  template <class> class ReduceFn,
  typename ElementReduce,
  bool FinalReduction = true, // Should match the FinalReduction in Device type
  typename CtaTileShapeMNK = cute::Shape<cute::_1,cute::_1,cute::_1>,
  typename ElementCompute = float
>
class HostRowReduce: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  using ElementDst = cute::conditional_t<FinalReduction, ElementReduce, ElementCompute>;

  static constexpr int TileM = cute::get<0>(CtaTileShapeMNK{});
  static constexpr int TileN = cute::get<1>(CtaTileShapeMNK{});

  struct Arguments {
    struct OpArgs {
      ElementReduce* ptr_row = nullptr;
      ElementCompute reduce_identity = 0;
      cute::Stride<cute::_0, cute::_1, cute::_0> dRow = {};
    } op;
  };

private:
  cutlass::NumericConverter<ElementReduce, ElementDst> destination_converter_;
  cutlass::HostTensor<ElementDst, LayoutTagVector> tensor_row_reduce_;
  cutlass::HostTensor<ElementCompute, LayoutTagVector> reduce_buffer_;
  cutlass::HostTensor<ElementDst, LayoutTagVector> reference_row_reduce_;
  int N_;
  ReduceFn<ElementCompute> reduce_fn_;

  int extent_m_;
  int extent_n_;
  int extent_l_;
public:
  HostRowReduce(){}
  template <typename ProblemShapeType>
  HostRowReduce(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024):
    Base(check_relative_equality) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    N_ = cute::get<1>(problem_shape_MNKL);
    if constexpr (FinalReduction) {
      tensor_row_reduce_.resize(cutlass::Coord<1>(N_));
      reference_row_reduce_.resize(cutlass::Coord<1>(N_));
      reduce_buffer_.resize(cutlass::Coord<1>(N_));
    } 
    else {
      auto NumTile = cute::ceil_div(cute::select<0,1,3>(problem_shape_MNKL), cute::take<0,2>(CtaTileShapeMNK{}));
      extent_m_ = cute::get<0>(NumTile);
      extent_n_ = cute::get<1>(NumTile) * TileN;
      extent_l_ = cute::get<2>(NumTile);
      auto shape = cutlass::make_Coord(extent_m_ * extent_n_ * extent_l_);
      tensor_row_reduce_.resize(shape);
      reference_row_reduce_.resize(shape);
      reduce_buffer_.resize(shape);
    }

    cutlass::reference::host::TensorFill(reduce_buffer_.host_view());
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {
    if constexpr (FinalReduction) {
      auto TensorRowReduce = cute::make_tensor(reduce_buffer_.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{}, N_)));
      TensorRowReduce(1, n + n_b) = reduce_fn_(TensorRowReduce(1, n + n_b), child_0_result);
    } 
    else {
      auto TensorRowReduce = cute::make_tensor(
        reduce_buffer_.host_data(),
        cute::make_layout(
          cute::make_shape(extent_m_, extent_n_, extent_l_),
          cute::make_stride(extent_n_, 1, extent_m_ * extent_l_)
        )
      );
      TensorRowReduce((m+m_b)/TileM, n+n_b, l) = reduce_fn_(TensorRowReduce((m+m_b)/TileM, n+n_b, l), child_0_result);
    }
    
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    // Verify the store node
    tensor_row_reduce_.sync_host();

    auto TensorRowReduce = cute::make_tensor(reference_row_reduce_.host_data(),
      cute::make_layout(cute::make_shape(reference_row_reduce_.size())));
    
    auto TensorReduceBuffer = cute::make_tensor(reduce_buffer_.host_data(),
      cute::make_layout(cute::make_shape(reduce_buffer_.size())));

    // Filling the reference tensor with the reduce buffer
    for (uint64_t n = 0; n < size(TensorRowReduce); n ++) {
      TensorRowReduce(n) = destination_converter_(TensorReduceBuffer(n));
    }

    bool equal = this->equality_check(reference_row_reduce_.host_view(), tensor_row_reduce_.host_view());
    if (!equal) {
      error_ss 
        << "\n\nRow Reduce Reference =\n" << reference_row_reduce_.host_view()
        << "\n\nRow Reduce Computed =\n" << tensor_row_reduce_.host_view() << "\n\n";
    }
    return equal;
  }

  Arguments get_arguments() {
    return {tensor_row_reduce_.device_data()};
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Column Reduce

template <
  template <class> class ReduceFn,
  typename ElementReduce,
  bool FinalReduction = true,  // Should match the FinalReduction in Device type
  typename CtaTileShapeMNK = cute::Shape<cute::_1,cute::_1,cute::_1>,
  typename ElementCompute = float
>
class HostColumnReduce: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  using ElementDst = cute::conditional_t<FinalReduction, ElementReduce, ElementCompute>;

  static constexpr int TileM = cute::get<0>(CtaTileShapeMNK{});
  static constexpr int TileN = cute::get<1>(CtaTileShapeMNK{});

  struct Arguments {
    struct OpArgs {
      ElementReduce* ptr_col = nullptr;
      ElementCompute reduce_identity = 0;
      cute::Stride<cute::_1, cute::_0, cute::_0> dRow = {};
    } op;
  };

private:
  cutlass::NumericConverter<ElementDst, ElementCompute> destination_converter_;
  cutlass::HostTensor<ElementDst, LayoutTagVector> tensor_column_reduce_;
  cutlass::HostTensor<ElementCompute, LayoutTagVector> reduce_buffer_;
  cutlass::HostTensor<ElementDst, LayoutTagVector> reference_column_reduce_;
  int M_;
  ReduceFn<ElementCompute> reduce_fn_;

  int extent_m_;
  int extent_n_;
  int extent_l_;
public:
  HostColumnReduce(){}
  template <typename ProblemShapeType>
  HostColumnReduce(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024):
    Base(check_relative_equality) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    M_ = cute::get<0>(problem_shape_MNKL);

    if constexpr (FinalReduction) {
      tensor_column_reduce_.resize(cutlass::Coord<1>(M_));
      reference_column_reduce_.resize(cutlass::Coord<1>(M_));
      reduce_buffer_.resize(cutlass::Coord<1>(M_));
    } 
    else {
      auto NumTile = cute::ceil_div(cute::select<0,1,3>(problem_shape_MNKL), cute::take<0,2>(CtaTileShapeMNK{}));
      extent_m_ = cute::get<0>(NumTile) * TileM;
      extent_n_ = cute::get<1>(NumTile);
      extent_l_ = cute::get<2>(NumTile);
      auto shape = cutlass::make_Coord(extent_m_ * extent_n_ * extent_l_);
      tensor_column_reduce_.resize(shape);
      reference_column_reduce_.resize(shape);
      reduce_buffer_.resize(shape);
    }

    cutlass::reference::host::TensorFill(reduce_buffer_.host_view());
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {
    auto TensorColReduce = cute::make_tensor(reduce_buffer_.host_data(),
      cute::make_layout(cute::make_shape(M_, cute::_1{})));
    if constexpr (FinalReduction) {
      TensorColReduce(m + m_b, 1) = reduce_fn_(TensorColReduce(m + m_b, 1), child_0_result);
    } 
    else {
      auto shape = reduce_buffer_.extent();
      auto TensorColReduce = cute::make_tensor(
        reduce_buffer_.host_data(),
        cute::make_layout(
          cute::make_shape(extent_m_, extent_n_, extent_l_),
          cute::make_stride(1, extent_m_, extent_m_ * extent_l_)
        )
      );
      TensorColReduce(m+m_b, (n+n_b)/TileN, l) = reduce_fn_(TensorColReduce(m+m_b, (n+n_b)/TileN, l), child_0_result);
    }
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    // Verify the store node
    tensor_column_reduce_.sync_host();

    auto TensorColReduce = cute::make_tensor(reference_column_reduce_.host_data(),
      cute::make_layout(cute::make_shape(reference_column_reduce_.size())));
    
    auto TensorReduceBuffer = cute::make_tensor(reduce_buffer_.host_data(),
    cute::make_layout(cute::make_shape(reduce_buffer_.size())));

    // Filling the reference tensor with the reduce buffer
    for (uint64_t m = 0; m < size(TensorColReduce); m ++) {
      TensorColReduce(m) = destination_converter_(TensorReduceBuffer(m));
    }

    bool equal = this->equality_check(reference_column_reduce_.host_view(), tensor_column_reduce_.host_view());
    if (!equal) {
      error_ss 
        << "\n\nColumn Reduce Reference =\n" << reference_column_reduce_.host_view()
        << "\n\nColumn Reduce Computed =\n" << tensor_column_reduce_.host_view() << "\n\n";
    }
    return equal;
  }

  Arguments get_arguments() {
    return {tensor_column_reduce_.device_data()};
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Scalar Reduce

template <
  template <class> class ReduceFn,
  typename ElementReduce,
  typename ElementCompute = float,
  bool enabled = true
>
class HostScalarReduce: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  struct Arguments {
    struct OpArgs {
      ElementReduce* ptr_scalar = nullptr;
      ElementCompute reduce_identity = 0;
      cute::Stride<cute::_0, cute::_0, cute::_0> dScalar = {};
    } op;
  };

private:
  cutlass::NumericConverter<ElementReduce, ElementCompute> destination_converter_;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> tensor_scalar_reduce_;
  cutlass::HostTensor<ElementCompute, LayoutTagVector> reduce_buffer_;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> reference_scalar_reduce_;
  ReduceFn<ElementCompute> reduce_fn_;
public:
  HostScalarReduce(){}
  template <typename ProblemShapeType>
  HostScalarReduce(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024):
    Base(check_relative_equality) {
    tensor_scalar_reduce_.resize(cutlass::Coord<1>(1));
    reference_scalar_reduce_.resize(cutlass::Coord<1>(1));
    reduce_buffer_.resize(cutlass::Coord<1>(1));

    tensor_scalar_reduce_.sync_device();
    cutlass::reference::host::TensorFill(reduce_buffer_.host_view());
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {
    auto TensorRowReduce = cute::make_tensor(reduce_buffer_.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{})));
    TensorRowReduce(0) = reduce_fn_(TensorRowReduce(0), child_0_result);
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    if constexpr (enabled) {
      // Verify the store node
      tensor_scalar_reduce_.sync_host();

      auto TensorRowReduce = cute::make_tensor(reference_scalar_reduce_.host_data(),
        cute::make_layout(cute::make_shape(cute::_1{})));
      
      auto TensorReduceBuffer = cute::make_tensor(reduce_buffer_.host_data(),
        cute::make_layout(cute::make_shape(cute::_1{})));

      // Filling the reference tensor with the reduce buffer
      TensorRowReduce(0) = destination_converter_(TensorReduceBuffer(0));

      bool equal = this->equality_check(reference_scalar_reduce_.host_view(), tensor_scalar_reduce_.host_view());
      if (!equal) {
        error_ss 
          << "\n\nScalar Reduce Reference =\n" << reference_scalar_reduce_.host_view()
          << "\n\nScalar Reduce Computed =\n" << tensor_scalar_reduce_.host_view() << "\n\n";
      }
      return equal;
    }
    else {
      return true;
    }
    
  }

  Arguments get_arguments() {
    return {tensor_scalar_reduce_.device_data()};
  }

  auto get_flatten_arguments() {
    return cute::make_tuple(tensor_scalar_reduce_.device_data());
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Host EVT wrapper

/// The ArgumentPack is used to model the alignment when num ops <= 4
template <typename... Ops>
struct ArgumentPack;

template <typename T>
struct ArgumentPack<T> {
  T arg;
  ArgumentPack(T first):
    arg(first) {}
};

template <typename First, typename... Rest>
struct ArgumentPack<First, Rest...> {
  First arg;
  ArgumentPack<Rest...> rest_args;

  ArgumentPack(First first, Rest... rest) :
    arg(first), rest_args(rest...) {}
};


/// Base class for Host Visitor
template <class ElementCompute, class... Ops>
struct HostVisitorBase: public HostEVTNodeBase<ElementCompute> {
public:
  using Base = HostEVTNodeBase<ElementCompute>;

  using Arguments_struct = ArgumentPack<typename Ops::Arguments...>;
  using Arguments_tuple = cute::tuple<typename Ops::Arguments...>;

  constexpr static int Rm1 = sizeof...(Ops);
  constexpr static bool cond = Rm1 > 4;
  using Arguments = cute::conditional_t<cond, Arguments_tuple, Arguments_struct>;

  std::tuple<Ops...> ops;

  HostVisitorBase(){}
  template<typename ProblemShapeType>
  HostVisitorBase(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    :Base(check_relative_equality),
    ops(test::gemm::device::tapply(std::tuple<Ops...>{}, 
      [&] (auto&& op) {
        using Op = cute::remove_cvref_t<decltype(op)>;
        return Op(problem_size, check_relative_equality, seed);
      },
      [] (auto&&... _ops) { 
        return std::make_tuple(_ops...); 
      },
      cute::make_seq<Rm1>{}
    )){ }

  bool compare_reference(std::stringstream& error_ss) {
    return cute::detail::tapply(ops,
      [&](auto& op) {
        return op.compare_reference(error_ss);
      },
      [&] (auto&&... inputs) {
        return arrayAnd(inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }

  void* get_tensor_C_ptr() {
    return cute::detail::tapply(ops,
      [&](auto& op) {
        return op.get_tensor_C_ptr();
      },
      [&] (auto&&... inputs) {
        return findNonNullPtr(inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }

  void* get_tensor_D_ptr() {
    return cute::detail::tapply(ops,
      [&](auto& op) {
        return op.get_tensor_D_ptr();
      },
      [&] (auto&&... inputs) {
        return findNonNullPtr(inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }

  Arguments get_arguments() {
    return test::gemm::device::tapply(ops,
      [&](auto& op) {
        return op.get_arguments();
      },
      [&] (auto&&... args) {
        if constexpr (Rm1 > 4) {
          return cute::make_tuple(args...);
        } 
        else {
          return Arguments(args...);
        }  
      },
      cute::make_seq<Rm1>{}
    );
  }

  auto get_flatten_arguments() {
    return test::gemm::device::tapply(ops,
      [&](auto& op) {
        return op.get_flatten_arguments();
      },
      [&] (auto&&... args) {
        return flatten(cute::make_tuple(args...));
      },
      cute::make_seq<Rm1>{}
    );
  }

  bool arrayAnd(bool passed) {
    return passed;
  }

  template <typename... Args>
  bool arrayAnd(bool first_passed, Args... passed) {
    if (first_passed) {
      return arrayAnd(passed...);
    }
    return first_passed;
  }

};


/// Tree-struct visitor
template <class NodeOp, class... ChildOps>
struct HostTreeVisitor: public HostVisitorBase<typename NodeOp::Base::ElementCompute, ChildOps..., NodeOp> {
public:
  using ElementCompute = typename NodeOp::Base::ElementCompute;
  using Base = HostVisitorBase<ElementCompute, ChildOps..., NodeOp>;
  using Arguments = typename Base::Arguments;
  
  constexpr static int Rm1 = sizeof...(ChildOps);

  HostTreeVisitor(){}
  template<typename ProblemShapeType>
  HostTreeVisitor(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    :Base(problem_size, check_relative_equality, seed){ }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    return cute::detail::tapply(this->ops,
      [&] (auto& op) {
        return op.visit(m, n, l, m_b, n_b, acc);
      },
      [&] (auto&&... frg_inputs) {
        return std::get<Rm1>(this->ops).visit(m, n, l, m_b, n_b, acc, frg_inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }
};


/// General Graph visitor
template <class ElementCompute, class EdgeTuple, class... Ops>
struct HostTopoVisitor: public HostVisitorBase<ElementCompute, Ops...> {
public:
  using Base = HostVisitorBase<ElementCompute, Ops...>;
  constexpr static int Rm1 = Base::Rm1;
  using Arguments = typename Base::Arguments;
  
private:
  ElementCompute frg_outputs_[Rm1];
public:
  HostTopoVisitor(){}
  template<typename ProblemShapeType>
  HostTopoVisitor(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    :Base(problem_size, check_relative_equality, seed) { }

  template<class ElementAccumulator, int I>
  ElementCompute visit_(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
      frg_outputs_[I] = cute::transform_apply(cute::get<I>(EdgeTuple{}),
        [&] (auto&& _E) {
          constexpr int e = cute::remove_cvref_t<decltype(_E)>::value;
          return frg_outputs_[e];
        },
        [&] (auto const&... frg_inputs) {
          ElementCompute res = std::get<I>(this->ops).visit(m, n, l, m_b, n_b, acc, frg_inputs...);
          return res;
        }
      );

      if constexpr (I < Rm1 - 1) {
        return visit_<ElementAccumulator, I+1>(m, n, l, m_b, n_b, acc);
      } 
      else {
        return frg_outputs_[I];
      }
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {

    return visit_<ElementAccumulator, 0>(m, n, l, m_b, n_b, acc);
  }

};


/// SplitTree visitor
template <class ElementCompute, class InputTree, class OutputTree, class... AuxOutTrees>
struct HostSplitTreeVisitor: public HostVisitorBase<ElementCompute, InputTree, AuxOutTrees..., OutputTree> {
public:
  using Base = HostVisitorBase<ElementCompute, InputTree, AuxOutTrees..., OutputTree>;
  using Arguments = typename Base::Arguments;

  constexpr static int Rm2 = sizeof...(AuxOutTrees);

private:
  ElementCompute frg_input_;
public:
  HostSplitTreeVisitor(){}
  template<typename ProblemShapeType>
  HostSplitTreeVisitor(ProblemShapeType problem_size, bool check_relative_equality = false, int64_t seed = 2024)
    :Base(problem_size, check_relative_equality, seed) { }

  template<class ElementAccumulator, int I>
  void visitAux(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator frag) {
    std::get<I+1>(this->ops).visit(m, n, l, m_b, n_b, frag);

    if constexpr (I < Rm2 - 1) {
      return visitAux<ElementAccumulator, I+1>(m, n, l, m_b, n_b, frag);
    } 
    else {
      return;
    }
  }

  template<class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    
    /// Compute the input tree
    frg_input_ = std::get<0>(this->ops).visit(m, n, l, m_b, n_b, acc);

    /// Compute the aux out tree
    visitAux<ElementAccumulator, 0>(m, n, l, m_b, n_b, frg_input_);
    /// Visit the output tree
    return std::get<Rm2+1>(this->ops).visit(m, n, l, m_b, n_b, frg_input_);
  }
};

/// Universal testbed for EVT w/o smem
template <class Gemm, typename EVT, bool FlatArgs = false>
class Testbed3xEVTnoSmem {
public:
  // The EVT Module to test
  using EVTModule = EVT; //typename EVT::EVTModule;

  using TestBedImpl = typename detail::TestbedImpl<Gemm, cutlass::epilogue::thread::Identity, true>;
  using Kernel = typename Gemm::GemmKernel;
  using Epilogue = typename Gemm::GemmKernel::CollectiveEpilogue;
  using ElementAccumulator = typename Kernel::ElementAccumulator;
  using ElementC = typename Kernel::ElementC;
  using ElementD = typename Kernel::ElementD;

  using ProblemShapeType = typename Kernel::ProblemShape;

  using LayoutTagA = typename TestBedImpl::LayoutTagA;
  using LayoutTagB = typename TestBedImpl::LayoutTagB;

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

  //
  // Methods
  //
  Testbed3xEVTnoSmem(
      bool check_relative_equality_,
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = TestBedImpl::kDefaultSeed ) :
    impl_((check_relative_equality_ ? CheckEquality::RELATIVE : CheckEquality::EXACT), ScalarLoc::ON_DEVICE, VectorScale::ENABLED,
          init_A_, init_B_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
          check_relative_equality(check_relative_equality_) { }

  Testbed3xEVTnoSmem(
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = TestBedImpl::kDefaultSeed ) :
    impl_(CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED,
          init_A_, init_B_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
          check_relative_equality(false)  { }
  
  /// Initializes data structures
  void initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace for A/B tensor
    //
    impl_.initialize(problem_size);
  }
  // Detail Implementation
  TestBedImpl impl_;
  
  // Whether to use relative equality checks
  bool check_relative_equality;
  
  bool verify(ProblemShapeType problem_size, EVTModule& host_reference) {
    
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);

    auto A = cute::make_tensor(impl_.collective_mma_inputs.tensor_A.host_data(),
      cute::make_layout(cute::make_shape(M, K, L), impl_.collective_mma_inputs.stride_a));
    auto B = cute::make_tensor(impl_.collective_mma_inputs.tensor_B.host_data(),
      cute::make_layout(cute::make_shape(N, K, L), impl_.collective_mma_inputs.stride_b));
    auto LayoutD = cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_d);

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    /// Reference Kernel
    static int constexpr kBlockM = 64;
    static int constexpr kBlockN = 64;

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int64_t l = 0; l < cute::size<2>(mainloop_params.A.layout()); ++l) {
      for (int64_t m = 0; m < cute::size<0>(mainloop_params.A.layout()); m += kBlockM) {
        for (int64_t n = 0; n < cute::size<0>(mainloop_params.B.layout()); n += kBlockN) {
          ElementAccumulator acc[kBlockM][kBlockN];
          gett_mainloop(mainloop_params, m, n, l, acc);
          /// Epilogue EVT
          for (int n_b = 0; n_b < kBlockN; ++n_b) {
            for (int m_b = 0; m_b < kBlockM; ++m_b) {
              if (m + m_b < cute::size<0>(LayoutD) && n + n_b < cute::size<1>(LayoutD)) {
                host_reference.visit(m, n, l, m_b, n_b, acc[m_b][n_b]);
              }
            }
          }
        }
      }
    }

    std::stringstream error_ss;
    bool passed = host_reference.compare_reference(error_ss);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";
      
      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K
        << ", Batch count = " << L << "\n\n";
      
      file
        << "A =\n" << impl_.collective_mma_inputs.tensor_A.host_view()
        << "\nB =\n" << impl_.collective_mma_inputs.tensor_B.host_view();
      
      file << error_ss.str();
    }

    return passed;
  }

  bool run(
    ProblemShapeType problem_size,
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic,
    detail::MaxSwizzleSize max_swizzle = detail::MaxSwizzleSize{},
    detail::Splits splits = detail::Splits{},
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic,
    int iterations = 20,
    bool profiling = false) {   
    // Fail test if insufficient CUDA device
    if (!impl_.sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }
    //
    // Initialize the Gemm operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      impl_.sm_count = std::min(impl_.MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = impl_.sm_count;
    }
    else {
      impl_.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = impl_.sm_count;
    }

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { static_cast<int>(splits), static_cast<int>(max_swizzle), raster_order, decomposition_mode };
    }
    else {
      scheduler_args = { static_cast<int>(max_swizzle), raster_order };
    }

    /// Initializes data structures
    /// A/B/C/D Tensor
    initialize(problem_size);

    /// Initialize the epilogue arguments
    EVTModule host_reference(problem_size, check_relative_equality, 2024);

    arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        impl_.collective_mma_inputs.tensor_A.device_data(), impl_.collective_mma_inputs.stride_a,
        impl_.collective_mma_inputs.tensor_B.device_data(), impl_.collective_mma_inputs.stride_b
      },
      {},
      hw_info,
      scheduler_args
    };

    // Filling in the thread arguments
    if constexpr (FlatArgs) {
      auto epilogue_args = host_reference.get_flatten_arguments();
      std::memcpy(&arguments.epilogue.thread, &epilogue_args, sizeof(epilogue_args));

      arguments.epilogue.ptr_C = static_cast<ElementC*>(host_reference.get_tensor_C_ptr());
      arguments.epilogue.dC = impl_.collective_epilogue.stride_c;

      arguments.epilogue.ptr_D = static_cast<ElementD*>(host_reference.get_tensor_D_ptr());
      arguments.epilogue.dD = impl_.collective_epilogue.stride_d;
    } 
    else {
      auto epilogue_args = host_reference.get_arguments();
      std::memcpy(&arguments.epilogue, &epilogue_args, sizeof(epilogue_args));
    }

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }
    
    //
    // Run the GEMM
    //
    if (profiling) {
      return impl_.profile(problem_size, iterations, gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }
    }

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //
    bool passed = this->verify(problem_size, host_reference);
    if (!passed) {
      std::cout << "Error : Failed \n";
    }

    return passed;
  }
};

/// Universal testbed for EVT
template <class Gemm, typename EVT>
class Testbed3xEVT {
public:
  // The EVT Module to test
  using EVTModule = typename EVT::EVTModule;

  using TestBedImpl = typename detail::TestbedImpl<Gemm, cutlass::epilogue::thread::Identity, true>;
  using Kernel = typename Gemm::GemmKernel;
  using Epilogue = typename Gemm::GemmKernel::CollectiveEpilogue;
  using ElementAccumulator = typename Kernel::ElementAccumulator;
  using ElementC = typename Kernel::ElementC;
  using ElementD = typename Kernel::ElementD;

  using ProblemShapeType = typename Kernel::ProblemShape;

  using LayoutTagA = typename TestBedImpl::LayoutTagA;
  using LayoutTagB = typename TestBedImpl::LayoutTagB;
  using LayoutTagC = typename TestBedImpl::LayoutTagC;
  using LayoutTagD = typename TestBedImpl::LayoutTagD;

  //
  // Methods
  //
  Testbed3xEVT(
    bool check_relative_equality_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
     impl_((check_relative_equality_ ? CheckEquality::RELATIVE : CheckEquality::EXACT), ScalarLoc::ON_DEVICE, VectorScale::ENABLED,
           init_A_, init_B_, init_C_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
           check_relative_equality(check_relative_equality_) { }

  Testbed3xEVT(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
     impl_(CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED,
           init_A_, init_B_, init_C_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
           check_relative_equality(false)  { }

  Testbed3xEVT(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
    impl_(stride_factor_A_, stride_factor_B_, stride_factor_C_, stride_factor_D_,
          CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED,
          init_A_, init_B_, init_C_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
          check_relative_equality(false)  { }
  
  /// Initializes data structures
  void initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace for A/B tensor
    //
    impl_.initialize(problem_size);
  }
  // Detail Implementation
  TestBedImpl impl_;

  // Whether to use relative equality checks
  bool check_relative_equality;

  bool verify(ProblemShapeType problem_size, EVTModule& host_reference) {
    
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);

    auto A = cute::make_tensor(impl_.collective_mma_inputs.tensor_A.host_data(),
      cute::make_layout(cute::make_shape(M, K, L), impl_.collective_mma_inputs.stride_a));
    auto B = cute::make_tensor(impl_.collective_mma_inputs.tensor_B.host_data(),
      cute::make_layout(cute::make_shape(N, K, L), impl_.collective_mma_inputs.stride_b));
    auto LayoutD = cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_d);

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    /// Reference Kernel
    static int constexpr kBlockM = 64;
    static int constexpr kBlockN = 64;

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int64_t l = 0; l < cute::size<2>(mainloop_params.A.layout()); ++l) {
      for (int64_t m = 0; m < cute::size<0>(mainloop_params.A.layout()); m += kBlockM) {
        for (int64_t n = 0; n < cute::size<0>(mainloop_params.B.layout()); n += kBlockN) {
          ElementAccumulator acc[kBlockM][kBlockN];
          gett_mainloop(mainloop_params, m, n, l, acc);
          /// Epilogue EVT
          for (int n_b = 0; n_b < kBlockN; ++n_b) {
            for (int m_b = 0; m_b < kBlockM; ++m_b) {
              if (m + m_b < cute::size<0>(LayoutD) && n + n_b < cute::size<1>(LayoutD)) {
                host_reference.visit(m, n, l, m_b, n_b, acc[m_b][n_b]);
              }
            }
          }
        }
      }
    }

    std::stringstream error_ss;
    bool passed = host_reference.compare_reference(error_ss);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";
      
      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K
        << ", Batch count = " << L << "\n\n";
      
      file
        << "A =\n" << impl_.collective_mma_inputs.tensor_A.host_view()
        << "\nB =\n" << impl_.collective_mma_inputs.tensor_B.host_view()
        << "\nC =\n" << impl_.collective_epilogue.tensor_C.host_view() << "\n\n";
      
      file << error_ss.str();
    }

    return passed;
  }

  bool run(
    ProblemShapeType problem_size,
    bool profiling = false,
    int iterations = 20,
    int splits = 1) {   
    // Fail test if insufficient CUDA device
    if (!impl_.sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }
    //
    // Initialize the Gemm operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      impl_.sm_count = std::min(impl_.MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = impl_.sm_count;
    }
    else {
      impl_.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = impl_.sm_count;
    }

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { splits };
    }

    /// Initializes data structures
    /// A/B/C/D Tensor
    initialize(problem_size);

    /// Initialize the epilogue arguments
    EVTModule host_reference(problem_size, check_relative_equality, 2024);

    arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        impl_.collective_mma_inputs.tensor_A.device_data(), impl_.collective_mma_inputs.stride_a,
        impl_.collective_mma_inputs.tensor_B.device_data(), impl_.collective_mma_inputs.stride_b
      },
      {   // Epilogue arguments
        {}, // thread
        static_cast<ElementC*>(host_reference.get_tensor_C_ptr()),
        impl_.collective_epilogue.stride_c,
        static_cast<ElementD*>(host_reference.get_tensor_D_ptr()),
        impl_.collective_epilogue.stride_d
      },  // Epilogue arguments end
      hw_info,
      scheduler_args
    };

    // Filling in the thread arguments
    typename EVTModule::Arguments epilogue_args = host_reference.get_arguments();
    std::memcpy(&arguments.epilogue.thread, &epilogue_args.arg, sizeof(epilogue_args.arg));

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }
    
    //
    // Run the GEMM
    //
    if (profiling) {
      return impl_.profile(problem_size, iterations, gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }
    }

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //
    bool passed = this->verify(problem_size, host_reference);
    if (!passed) {
      std::cout << "Error : Failed \n";
    }

    return passed;
  }
};

template <typename Gemm, typename EVT>
bool TestAllEVT(bool check_relative_equality = false) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {max_alignment, 512 - 3 * max_alignment};
  std::vector<int> problem_size_n = {max_alignment, 512 - 2 * max_alignment};

  if constexpr (cute::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>) {
  problem_size_m.push_back(768);
  problem_size_n.push_back(768);
  }

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {max_alignment, TileShapeK * (Stages + 1) - max_alignment};

  Testbed3xEVT<Gemm, EVT> testbed(check_relative_equality);
  bool passed = true;

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

    passed = testbed.run(problem_size);

    if (!passed) {
      return false;
    }
    }
  }
  }

  // if we do support batched GEMM, just run one test on it to save on test time
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
  auto problem_size = ProblemShapeType{256 + max_alignment, 256 + max_alignment, 160 + max_alignment, /* l */ 3};
  passed = testbed.run(
    problem_size
  );

  if (!passed) {
    return false;
  }
  }

  return passed;
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
