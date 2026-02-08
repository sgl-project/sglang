/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * @brief Test for structured sparse gemm compressor device kernel
 */

#pragma once

#include <cuda_runtime_api.h>  // cudaGetLastError

#include <cstdint>             // uint64_t
#include <cstdio>              // printf
#include <cstdlib>             // malloc
#include <iostream>            // std::cout
#include <vector>
#include <array>

#include "cute/layout.hpp"                                    // cute::make_shape
#include "cute/util/type_traits.hpp"                          // cute::is_same_v
#include "cutlass/coord.h"                                    // cutlass::make_Coord
#include "cutlass/cutlass.h"                                  // cutlass::Status
#include "cutlass/kernel_hardware_info.hpp"                          // cutlass::KernelHardwareInfo
#include "cutlass/layout/matrix.h"                                   // cutlass::layout::Affine2Layout_Factory
#include "cutlass/numeric_types.h"                                   // cutlass::sizeof_bits, cutlass::float_
#include "cutlass/tensor_view.h"                                     // cutlass::TensorView
#include "cutlass/transform/device/transform_universal_adapter.hpp"  // cutlass::transform::device::TransformUniversalAdapter
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"       // cutlass::transform::kernel::StructuredSparseCompressorUtility
#include "cutlass/util/device_memory.h"                              // cutlass::device_memory::allocation
#include "cutlass/util/distribution.h"                               // cutlass::Distribution
#include "cutlass/util/host_tensor.h"                                // cutlass::HostTensor
#include "cutlass/util/packed_stride.hpp"                            // cutlass::make_cute_packed_stride
#include "cutlass/util/reference/host/tensor_compare.h"              // cutlass::reference::host::TensorEquals
#include "cutlass/util/reference/host/tensor_fill.h"  // cutlass::reference::host::TensorFillRandomUniform, TensorFillIdentity, TensorFillRandomGaussian, BlockFillSequential, TensorFill
#include "cutlass/detail/collective.hpp"

#include "sm90_sparse_gemm_compressor_legacy.hpp"     // Legacy host compressor
#include "../../common/cutlass_unit_test.h"           // CUTLASS UT, EXPECT_TRUE


#define CUDA_CHECK_FALSE(cuda_error)                                                           \
  {                                                                                            \
    if (cuda_error != cudaSuccess) {                                                           \
      printf("cudaError %s in %s:%d\n", cudaGetErrorString(cuda_error), __func__, __LINE__ );  \
      return false;                                                                            \
    }                                                                                          \
  }

#define CUDA_CHECK(cuda_error)                                                                 \
  {                                                                                            \
    if (cuda_error != cudaSuccess) {                                                           \
      printf("cudaError %s in %s:%d\n", cudaGetErrorString(cuda_error), __func__, __LINE__ );  \
      return;                                                                                  \
    }                                                                                          \
  }


///////////////////////////////////////////////////////////////////////////////////////////////////
// * Test Bed
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace test
{
namespace transform
{
namespace device
{

// Helper Functions
template <typename Element, typename Layout>
bool
initialize_tensor(cutlass::TensorView<Element, Layout> view, cutlass::Distribution::Kind dist_kind, uint64_t seed)
{
  if (dist_kind == cutlass::Distribution::Uniform) {
    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    }
    else if (bits_input <= 8) {
        scope_max = 1;
        scope_min = -1;
    } else {
      scope_max = 4;
      scope_min = -4;
    }
    cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
  }

  else if (dist_kind == cutlass::Distribution::Identity) {
    cutlass::reference::host::TensorFillIdentity(view);
  }

  else if (dist_kind == cutlass::Distribution::Gaussian) {
    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }

  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }

  else if (dist_kind == cutlass::Distribution::AllOnes) {
    cutlass::reference::host::TensorFill(view, Element(1));
  }

  else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view, Element(0));
  }

  else {
    EXPECT_TRUE(false) << "Not implemented";
    return false;
  }

  return true;
}

// Testbed
template <typename Compressor_>
struct TestbedSparseGemmCompressor {
public:
  using Compressor = Compressor_;
  using CompressorKernel = typename Compressor::TransformKernel;

  using ElementA = typename CompressorKernel::ElementA;
  using LayoutATag = typename CompressorKernel::LayoutATag;
  using StrideA = typename CompressorKernel::StrideA;
  using ArrayElementA = 
    ElementA
  ;

  using ElementE = typename CompressorKernel::ElementEMmaRaw;
  using LayoutETag = cutlass::layout::RowMajor;  // We don't care about the major here, just to allocate tensor

  using SparseConfig = typename CompressorKernel::SparseConfig;
  using ProblemShapeType = typename CompressorKernel::ProblemShape;

  using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                              ProblemShapeType,
                              ElementA,
                              LayoutATag,
                              SparseConfig>;

  using CompressorKernelHost = cutlass::transform::kernel::SM90StructuredSparseCompressorLegacy<
                                ProblemShapeType,
                                ElementA,
                                LayoutATag,
                                SparseConfig>;

  using CompressorHost = cutlass::transform::device::TransformUniversalAdapter<CompressorKernelHost>;

  static constexpr auto LogicalElemsAPerChunk = CompressorKernel::LogicalElemsAPerChunk;
  static constexpr auto PhysicalElemsAPerChunk = CompressorKernel::PhysicalElemsAPerChunk;

  struct Data {
    // Data Storage
    cutlass::HostTensor<ArrayElementA, LayoutATag> tensor_A;
    cutlass::HostTensor<ArrayElementA, LayoutATag> tensor_A_Comp;
    cutlass::HostTensor<ElementE, LayoutETag> tensor_E;
    cutlass::HostTensor<ArrayElementA, LayoutATag> tensor_A_Comp_ref;
    cutlass::HostTensor<ElementE, LayoutETag> tensor_E_ref;
  };

  struct CudaRAII {
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
  
    CudaRAII(){
      CUDA_CHECK(cudaStreamCreate( &stream ));
      CUDA_CHECK(cudaEventCreate( &start ));
      CUDA_CHECK(cudaEventCreate( &stop ));
    };

    CudaRAII(const CudaRAII&) = delete;  
    CudaRAII& operator=(const CudaRAII&) = delete;  
    CudaRAII(CudaRAII&&) = delete;  
    CudaRAII& operator=(CudaRAII&&) = delete;  

    ~CudaRAII(){
      CUDA_CHECK(cudaStreamDestroy( stream ));
      CUDA_CHECK(cudaEventDestroy( start ));
      CUDA_CHECK(cudaEventDestroy( stop ));
    }
  };

public:
  TestbedSparseGemmCompressor(
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_E_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_A_Comp_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = 7)
      : init_A(init_A_)
      , init_E(init_E_)
      , init_A_Comp(init_A_Comp_)
      , seed(seed_)
  {
  }

  bool valid_test(ProblemShapeType problem_shape_MNKL)
  {
    const int GemmK = cute::size<2>(problem_shape_MNKL);

    if ( GemmK % LogicalElemsAPerChunk != 0 ) {
      printf("GemmK needs to be multiplier of LogicalElemsAPerChunk\n");
      return false;
    }

    return true;
  }

  bool initialize(ProblemShapeType problem_shape_MNKL, Data& datas)
  {
    CUDA_CHECK_FALSE(cudaGetLastError());

    // In unit of ElementARaw
    const int GemmM = cute::size<0>(problem_shape_MNKL);
    const int GemmN = cute::size<1>(problem_shape_MNKL);
    const int GemmK = cute::size<2>(problem_shape_MNKL);
    const int GemmL = cute::size<3>(problem_shape_MNKL);

    // Compressor utility to get allocated data size
    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(GemmM, GemmK, GemmL));
    CompressorUtility compressor_utility(problem_shape_MNKL, stride_a);

    // TensorA
    // In unit of ElementARaw, after alignment requirement
    // M-dim: no alignment requirement
    // K-dim: multiplier of chunk size

    // TensorA Compressed
    // In unit of ElementARaw, after alignment requirement
    // M-dim: TMA alignment
    // K-dim: TMA alignment
    const int GemmMAlignedAC = compressor_utility.get_tensorA_m_physical();
    const int GemmKAlignedAC = compressor_utility.get_tensorA_k_physical();

    // TensorE
    // In unit of ElementE (uint8_t), after alignment requirement
    // M-dim: TensorEAtom_M alignment
    // K-dim: TensorEAtom_K alignment
    const int GemmMAlignedE = compressor_utility.get_metadata_m_physical();
    const int GemmKAlignedE = compressor_utility.get_metadata_k_physical();

    auto a_coord = cutlass::make_Coord(GemmM * GemmL, GemmK);
    auto e_coord = cutlass::make_Coord(GemmMAlignedE * GemmL, GemmKAlignedE);
    auto a_comp_coord = cutlass::make_Coord(GemmMAlignedAC * GemmL, GemmKAlignedAC);

    typename LayoutATag::Stride stride_factor_A;
    typename LayoutETag::Stride stride_factor_E;

    datas.tensor_A.resize(a_coord,
                          cutlass::layout::Affine2Layout_Factory<LayoutATag>::layout_factory(a_coord, stride_factor_A));
    datas.tensor_A_Comp.resize(a_comp_coord,
                               cutlass::layout::Affine2Layout_Factory<LayoutATag>::layout_factory(a_comp_coord, stride_factor_A));
    datas.tensor_A_Comp_ref.resize(a_comp_coord,
                                   cutlass::layout::Affine2Layout_Factory<LayoutATag>::layout_factory(a_comp_coord, stride_factor_A),
                                   false);
    datas.tensor_E.resize(e_coord,
                          cutlass::layout::Affine2Layout_Factory<LayoutETag>::layout_factory(e_coord, stride_factor_E));
    datas.tensor_E_ref.resize(e_coord,
                              cutlass::layout::Affine2Layout_Factory<LayoutETag>::layout_factory(e_coord, stride_factor_E),
                              false);

    EXPECT_TRUE(initialize_tensor(datas.tensor_A.host_view(), init_A, seed + 1));
    EXPECT_TRUE(initialize_tensor(datas.tensor_E.host_view(), init_E, seed + 2));
    EXPECT_TRUE(initialize_tensor(datas.tensor_E_ref.host_view(), init_E, seed + 3));
    EXPECT_TRUE(initialize_tensor(datas.tensor_A_Comp.host_view(), init_A_Comp, seed + 4));
    EXPECT_TRUE(initialize_tensor(datas.tensor_A_Comp_ref.host_view(), init_A_Comp, seed + 5));

    compressor_utility.structure_sparse_zero_mask_fill(datas.tensor_A.host_data(), seed + 6);

    // Check for failed devide
    CUDA_CHECK_FALSE(cudaGetLastError());

    datas.tensor_A.sync_device();
    datas.tensor_A_Comp.sync_device();
    datas.tensor_E.sync_device();

    // Check for failed devide
    CUDA_CHECK_FALSE(cudaGetLastError());

    return true;
  }

  bool run_device(ProblemShapeType problem_shape_MNKL, Data& datas, float* time = nullptr)
  {
    CudaRAII cuda_raii;

    const int GemmM = cute::size<0>(problem_shape_MNKL);
    const int GemmN = cute::size<1>(problem_shape_MNKL);
    const int GemmK = cute::size<2>(problem_shape_MNKL);
    const int GemmL = cute::size<3>(problem_shape_MNKL);

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(GemmM, GemmK, GemmL));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    typename Compressor::Arguments arguments{
        {GemmM, GemmN, GemmK, GemmL},
        {datas.tensor_A.device_data(),
         stride_a,
         datas.tensor_A_Comp.device_data(),
         datas.tensor_E.device_data()},
        {hw_info}
    };

    Compressor compressor_op;
    size_t workspace_size = Compressor::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status {cutlass::Status::kSuccess };

    status = compressor_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      CUDA_CHECK_FALSE(cudaGetLastError());
    }

    status = compressor_op.initialize(arguments, workspace.get(), cuda_raii.stream);
    if (status != cutlass::Status::kSuccess) {
      CUDA_CHECK_FALSE(cudaGetLastError());
    }

    CUDA_CHECK_FALSE(cudaStreamSynchronize(cuda_raii.stream));
    CUDA_CHECK_FALSE(cudaEventRecord(cuda_raii.start, cuda_raii.stream));

    status = compressor_op.run(cuda_raii.stream);
    if (status != cutlass::Status::kSuccess) {
      CUDA_CHECK_FALSE(cudaGetLastError());
    }

    CUDA_CHECK_FALSE(cudaEventRecord(cuda_raii.stop, cuda_raii.stream));
    CUDA_CHECK_FALSE(cudaEventSynchronize(cuda_raii.stop));
    CUDA_CHECK_FALSE(cudaStreamSynchronize(cuda_raii.stream));
    if ( time != nullptr ){
      CUDA_CHECK_FALSE(cudaEventElapsedTime(time, cuda_raii.start, cuda_raii.stop));
    }

    datas.tensor_A_Comp.sync_host();
    datas.tensor_E.sync_host();

    #if 0
    {
      printf("\n--> DEVICE OUTPUT\n");
      printf("datas.tensor_A\n");
      std::cout << datas.tensor_A.host_view() << std::endl << std::endl;
      printf("datas.tensor_A_Comp\n");
      std::cout << datas.tensor_A_Comp.host_view() << std::endl << std::endl;
      printf("datas.tensor_E\n");
      std::cout << datas.tensor_E.host_view() << std::endl << std::endl;
    }
    #endif

    return true;
  }

  bool run_host_ref(ProblemShapeType problem_shape_MNKL, Data& datas)
  {
    const int GemmM = cute::size<0>(problem_shape_MNKL);
    const int GemmN = cute::size<1>(problem_shape_MNKL);
    const int GemmK = cute::size<2>(problem_shape_MNKL);
    const int GemmL = cute::size<3>(problem_shape_MNKL);

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(GemmM, GemmK, GemmL));

    typename CompressorKernelHost::Arguments arguments{
        {GemmM, GemmN, GemmK, GemmL},
        {datas.tensor_A.host_data(),
         stride_a,
         datas.tensor_A_Comp_ref.host_data(),
         datas.tensor_E_ref.host_data()},
        {}};

    const auto can_imp = CompressorKernelHost::can_implement(arguments);
    if (can_imp != cutlass::Status::kSuccess) {
      printf("can_implement() check failed\n");
      return false;
    }

    // Relies on std::vector for RAII
    auto workspace_size =
        static_cast<std::vector<uint8_t>::size_type>(CompressorKernelHost::get_workspace_size(arguments));
    std::vector<uint8_t> workspace_vector(workspace_size);
    auto workspace = static_cast<void*>(workspace_vector.data());

    cutlass::Status status = CompressorKernelHost::initialize_workspace(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
      printf("initialize_workspace() failed\n");
      return false;
    }

    auto params = CompressorKernelHost::to_underlying_arguments(arguments, workspace);
    CompressorKernelHost::run(params);

    return true;
  }

  bool compare_reference(Data& datas)
  {
    bool check_tensor_a_compressed =
        cutlass::reference::host::TensorEquals(datas.tensor_A_Comp_ref.host_view(), datas.tensor_A_Comp.host_view());
    if (!check_tensor_a_compressed) {
      printf("A-Compressed Mismatch\n");
    }

    bool check_tensor_e = cutlass::reference::host::TensorEquals(datas.tensor_E_ref.host_view(), datas.tensor_E.host_view());
    if (!check_tensor_e) {
      printf("E Mismatch\n");
    }

    return check_tensor_a_compressed && check_tensor_e;
  }

  bool run_auto_small()
  {
    return run_auto(true);
  }

  bool run_auto(bool run_small = false)
  {
    constexpr auto TensorEAlignmentM = typename SparseConfig::TensorEAlignmentM{};
    constexpr auto TensorEAlignmentK = typename SparseConfig::TensorEAlignmentK{};
    constexpr int LogicalElemsAPerChunk = typename SparseConfig::LogicalElemsAPerChunk{};

    constexpr int GemmN = 1;

    using ProblemType = typename std::array<int, 4>;

    std::vector<ProblemType> problems;

    const std::vector<ProblemType> problems_multiplier_of_tensor_e_atom = {
      // * Regular Cases (multiplier of TensorEAlignment)
      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 2, 1},
      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 2, 1},
      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 3, 1},

      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 2, 1},
      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 2, 1},
      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 3, 1},

      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 2, 1},
      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 2, 1},
      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 3, 1},

      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 2, 2},
      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 2, 2},
      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 3, 2},

      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 2, 2},
      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 2, 2},
      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 3, 2},

      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 2, 2},
      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 2, 2},
      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 3, 2},

      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 2, 3},
      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 2, 3},
      {TensorEAlignmentM * 1, GemmN, TensorEAlignmentK * 3, 3},

      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 2, 3},
      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 2, 3},
      {TensorEAlignmentM * 2, GemmN, TensorEAlignmentK * 3, 3},

      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 2, 3},
      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 2, 3},
      {TensorEAlignmentM * 3, GemmN, TensorEAlignmentK * 3, 3},
    };

    const std::vector<ProblemType> problems_multiplier_of_tensor_e_atom_large = {
      // * Large Case (multiplier of TensorEAlignment)
      {TensorEAlignmentM * 10, GemmN, TensorEAlignmentK * 13, 1},
      // {TensorEAlignmentM * 11, GemmN, TensorEAlignmentK * 14, 2},
      // {TensorEAlignmentM * 12, GemmN, TensorEAlignmentK * 15, 3},
    };

    const std::vector<ProblemType> problems_multiplier_of_twochunk {
      // * Corner Cases
      {4, GemmN, LogicalElemsAPerChunk * 2, 1},
      {4, GemmN, LogicalElemsAPerChunk * 4, 1},
      {4, GemmN, LogicalElemsAPerChunk * 6, 1},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 1},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 1},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 1},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 1},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 1},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 1},

      {4, GemmN, LogicalElemsAPerChunk * 2, 2},
      {4, GemmN, LogicalElemsAPerChunk * 4, 2},
      {4, GemmN, LogicalElemsAPerChunk * 6, 2},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 2},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 2},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 2},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 2},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 2},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 2},

      {4, GemmN, LogicalElemsAPerChunk * 2, 3},
      {4, GemmN, LogicalElemsAPerChunk * 4, 3},
      {4, GemmN, LogicalElemsAPerChunk * 6, 3},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 3},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 3},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 3},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 3},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 3},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 3},

      {32 + 4, GemmN, LogicalElemsAPerChunk * 2, 1},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 4, 1},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 6, 1},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 1},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 1},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 1},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 1},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 1},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 1},

      {32 + 4, GemmN, LogicalElemsAPerChunk * 2, 2},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 4, 2},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 6, 2},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 2},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 2},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 2},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 2},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 2},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 2},

      {32 + 4, GemmN, LogicalElemsAPerChunk * 2, 3},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 4, 3},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 6, 3},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 3},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 3},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 3},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 3},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 3},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 3},

      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 2, 1},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 4, 1},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 6, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 1},

      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 2, 2},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 4, 2},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 6, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 2},

      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 2, 3},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 4, 3},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 6, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 3},

      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 2, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 4, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 6, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 1},

      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 2, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 4, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 6, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 2},

      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 2, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 4, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 6, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 2, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 4, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 6, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 2, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 4, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 6, 3},
    };

    const std::vector<ProblemType> problems_multiplier_of_onechunk {
      {4, GemmN, LogicalElemsAPerChunk * 1, 1},
      {4, GemmN, LogicalElemsAPerChunk * 3, 1},
      {4, GemmN, LogicalElemsAPerChunk * 5, 1},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 1},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 1},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 1},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 1},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 1},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 1},

      {4, GemmN, LogicalElemsAPerChunk * 1, 2},
      {4, GemmN, LogicalElemsAPerChunk * 3, 2},
      {4, GemmN, LogicalElemsAPerChunk * 5, 2},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 2},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 2},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 2},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 2},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 2},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 2},

      {4, GemmN, LogicalElemsAPerChunk * 1, 3},
      {4, GemmN, LogicalElemsAPerChunk * 3, 3},
      {4, GemmN, LogicalElemsAPerChunk * 5, 3},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 3},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 3},
      {4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 3},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 3},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 3},
      {4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 3},

      {32 + 4, GemmN, LogicalElemsAPerChunk * 1, 1},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 3, 1},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 5, 1},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 1},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 1},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 1},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 1},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 1},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 1},

      {32 + 4, GemmN, LogicalElemsAPerChunk * 1, 2},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 3, 2},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 5, 2},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 2},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 2},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 2},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 2},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 2},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 2},

      {32 + 4, GemmN, LogicalElemsAPerChunk * 1, 3},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 3, 3},
      {32 + 4, GemmN, LogicalElemsAPerChunk * 5, 3},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 3},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 3},
      {32 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 3},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 3},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 3},
      {32 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 3},

      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 1, 1},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 3, 1},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 5, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 1},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 1},

      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 1, 2},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 3, 2},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 5, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 2},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 2},

      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 1, 3},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 3, 3},
      {TensorEAlignmentM + 4, GemmN, LogicalElemsAPerChunk * 5, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 3},
      {TensorEAlignmentM + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 3},

      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 1, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 3, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 5, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 1},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 1},

      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 1, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 3, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 5, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 2},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 2},

      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 1, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 3, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, LogicalElemsAPerChunk * 5, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 1, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 3, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK + LogicalElemsAPerChunk * 5, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 1, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 3, 3},
      {TensorEAlignmentM * 2 + 4, GemmN, TensorEAlignmentK * 2 + LogicalElemsAPerChunk * 5, 3},
    };

    // Run small only run multiplier of chunk size cases
    if (run_small) {
      problems.insert(problems.end(), problems_multiplier_of_tensor_e_atom.begin(), problems_multiplier_of_tensor_e_atom.end());
    }
    // Run full run all corner cases
    else {
      problems.insert(problems.end(), problems_multiplier_of_tensor_e_atom_large.begin(), problems_multiplier_of_tensor_e_atom_large.end());
      problems.insert(problems.end(), problems_multiplier_of_tensor_e_atom.begin(), problems_multiplier_of_tensor_e_atom.end());
      problems.insert(problems.end(), problems_multiplier_of_twochunk.begin(), problems_multiplier_of_twochunk.end());
      problems.insert(problems.end(), problems_multiplier_of_onechunk.begin(), problems_multiplier_of_onechunk.end());
    }

    for (const auto& problem_shape_MNKL : problems) {
      const auto [GemmM, GemmN, GemmK, GemmL] = problem_shape_MNKL;
      bool passed = run({GemmM, GemmN, GemmK, GemmL});
      printf("run() (%.4d,%.4d,%.4d,%.4d) %s\n", GemmM, GemmN, GemmK, GemmL, passed ? "PASS" : "FAIL");
      CUTLASS_TRACE_HOST("run() " << GemmM << " " << GemmN << " " << GemmK << " " << GemmL << passed ? " PASS" : " FAIL");
      if (not passed) {
        return false;
      }
    }

    return true;
  }

  bool run(ProblemShapeType problem_shape_MNKL)
  {
    // Check if valid test
    if (not valid_test(problem_shape_MNKL)) {
      CUTLASS_TRACE_HOST("valid_test() fail\n");
      return false;
    }

    // Data Storage
    Data datas;

    // Initialize Data
    if (not initialize(problem_shape_MNKL, datas)) {
      CUTLASS_TRACE_HOST("initialize() fail\n");
      return false;
    }

    // Run Compressor (Host Ref)
    if (not run_host_ref(problem_shape_MNKL, datas)) {
      CUTLASS_TRACE_HOST("run_host() fail\n");
      return false;
    }

    // Run Compressor (Device)
    if (not run_device(problem_shape_MNKL, datas)) {
      CUTLASS_TRACE_HOST("run_device() fail\n");
      return false;
    }

    // Verify
    if (not compare_reference(datas)) {
      CUTLASS_TRACE_HOST("compare_reference() DEVICE <-> LEGACY HOST fail\n");
      printf("compare_reference() DEVICE <-> LEGACY HOST fail\n");
      return false;
    }
    // else {
    //   printf("DEVICE <-> HOST PASS\n");
    // }

    return true;
  }

  bool benchmark(ProblemShapeType problem_shape_MNKL) {
    const auto [GemmM, GemmN, GemmK, GemmL] = problem_shape_MNKL;
    printf("Benchmark() (%.4d,%.4d,%.4d,%.4d) START\n", GemmM, GemmN, GemmK, GemmL);

    // Check if valid test
    if (valid_test(problem_shape_MNKL) == false) {
      CUTLASS_TRACE_HOST("valid_test() fail\n");
      return false;
    }

    // 2 warm-up iterations and 10 timing iterations
    constexpr int num_warmup = 5;
    constexpr int num_iter = 10;

    // Duplicate data to mimic cold cache
    Data data[num_warmup + num_iter];
    double total_time_milliseconds{0.0};

    for (int i = 0; i < num_warmup + num_iter; ++i ) {
      printf("Benchmark() (%.4d,%.4d,%.4d,%.4d) ITER %d\n", GemmM, GemmN, GemmK, GemmL, i );

      auto& datum_i = data[i];

      // Initialize Data  
      if (initialize(problem_shape_MNKL, datum_i) == false) {
        CUTLASS_TRACE_HOST("initialize() fail\n");
        return false;
      }

      // Run Compressor (Device)
      double time_i_milliseconds{0.0f};
      if (not run_device(problem_shape_MNKL, datum_i, &time_i_milliseconds)) {
        CUTLASS_TRACE_HOST("run_device() fail\n");
        return false;
      }

      if ( i >= num_warmup ) {
        total_time_milliseconds += time_i_milliseconds;
      }
    }

    const double mean_time_milliseconds = total_time_milliseconds / num_iter;
    printf("Mean time (ms): %.5f\n", mean_time_milliseconds);

    return true;
  }

public:
  // Data Init Setting
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_A_Comp;
  cutlass::Distribution::Kind init_E;
  uint64_t seed;
};

}  // namespace device
}  // namespace transform
}  // namespace test
