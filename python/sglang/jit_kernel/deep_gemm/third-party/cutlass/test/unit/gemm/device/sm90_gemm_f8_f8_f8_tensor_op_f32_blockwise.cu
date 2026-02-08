/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>

#include <thrust/universal_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template<cute::GMMA::Major SFAMajor,
         cute::GMMA::Major SFBMajor,
         int ScaleGranularityM,
         int ScaleGranularityN,
         int ScaleGranularityK,
         class LayoutA,
         class LayoutB,
         class LayoutCD,
         class MmaTileShape,
         class ClusterShape>
bool groupwise_test(
    Int<ScaleGranularityM>, Int<ScaleGranularityN>, Int<ScaleGranularityK>,
    LayoutA, LayoutB, LayoutCD,
    MmaTileShape, ClusterShape) {

  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK, SFAMajor, SFBMajor>;
  using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
  using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::float_e4m3_t, LayoutCD, 16,
      cutlass::float_e4m3_t, LayoutCD, 16,
      cutlass::epilogue::TmaWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, cute::tuple<LayoutA, LayoutSFA>, 16,
      cutlass::float_e4m3_t, cute::tuple<LayoutB, LayoutSFB>, 16,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename CollectiveEpilogue::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  // Strides just iterate over scalars and have no zeros
  LayoutSFA layout_SFA;
  LayoutSFB layout_SFB;

  int alignment_M = max(max((is_same_v<LayoutA, cutlass::layout::ColumnMajor> ? 16 : 1) , 
                            (SFAMajor == cute::GMMA::Major::MN ? CollectiveMainloop::AlignmentSFA : 1)),
                        (is_same_v<LayoutCD, cutlass::layout::ColumnMajor> ? 16 : 1));

  int alignment_N = max(max((is_same_v<LayoutB, cutlass::layout::RowMajor> ? 16 : 1) , 
                            (SFBMajor == cute::GMMA::Major::MN ? CollectiveMainloop::AlignmentSFB : 1)),
                         (is_same_v<LayoutCD, cutlass::layout::RowMajor> ? 16 : 1));

  int alignment_K = max(max((is_same_v<LayoutA, cutlass::layout::RowMajor> ? 16 : 1) , 
                            (SFAMajor == cute::GMMA::Major::K ? CollectiveMainloop::AlignmentSFA : 1)),
                        max((is_same_v<LayoutB, cutlass::layout::ColumnMajor> ? 16 : 1) , 
                            (SFBMajor == cute::GMMA::Major::K ? CollectiveMainloop::AlignmentSFB : 1)));

  alignment_K = (alignment_K / size<2>(MmaTileShape{}) + 1) * size<2>(MmaTileShape{});

  int M = 1024 + alignment_M;
  int N = 1024 + alignment_N;
  int K = 512  + alignment_K;
  EXPECT_TRUE(M % alignment_M == 0);
  EXPECT_TRUE(N % alignment_N == 0);
  EXPECT_TRUE(K % alignment_K == 0);
  EXPECT_TRUE(K % size<2>(MmaTileShape{}) == 0);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  thrust::universal_vector<cutlass::float_e4m3_t> tensor_A(M * K);
  thrust::universal_vector<float> tensor_SFA(cute::size(cute::filter_zeros(layout_SFA)));
  thrust::universal_vector<cutlass::float_e4m3_t> tensor_B(N * K);
  thrust::universal_vector<float> tensor_SFB(cute::size(cute::filter_zeros(layout_SFB)));
  thrust::universal_vector<cutlass::float_e4m3_t> tensor_C(M * N);
  thrust::universal_vector<cutlass::float_e4m3_t> tensor_D(M * N);
  thrust::universal_vector<cutlass::float_e4m3_t> tensor_ref_D(M * N);

  thrust::random::default_random_engine engine(2025);
  thrust::random::uniform_int_distribution<int> dist(-2, 2);

  std::generate(tensor_A.begin(), tensor_A.end(), [&] () {
    return static_cast<cutlass::float_e4m3_t>(dist(engine));
  });
  std::generate(tensor_SFA.begin(), tensor_SFA.end(), [&] () {
    return static_cast<float>(dist(engine));
  });
  std::generate(tensor_B.begin(), tensor_B.end(), [&] () {
    return static_cast<cutlass::float_e4m3_t>(dist(engine));
  });
  std::generate(tensor_SFB.begin(), tensor_SFB.end(), [&] () {
    return static_cast<float>(dist(engine));
  });
  std::generate(tensor_C.begin(), tensor_C.end(), [&] () {
    return static_cast<cutlass::float_e4m3_t>(dist(engine));
  });

  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {thrust::raw_pointer_cast(tensor_A.data()), stride_A, 
     thrust::raw_pointer_cast(tensor_B.data()), stride_B,
     thrust::raw_pointer_cast(tensor_SFA.data()), layout_SFA,
     thrust::raw_pointer_cast(tensor_SFB.data()), layout_SFB},
    {
      {}, // epilogue.thread
      thrust::raw_pointer_cast(tensor_C.data()), stride_C,
      thrust::raw_pointer_cast(tensor_D.data()), stride_D
    }
  };

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 1.0f;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm;

  EXPECT_TRUE(gemm.can_implement(arguments) == cutlass::Status::kSuccess);
  EXPECT_TRUE(gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess);
  EXPECT_TRUE(gemm.run() == cutlass::Status::kSuccess);
  EXPECT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  auto A = cute::make_tensor(thrust::raw_pointer_cast(tensor_A.data()),
      cute::make_layout(cute::make_shape(M, K, 1), stride_A));
  auto B = cute::make_tensor(thrust::raw_pointer_cast(tensor_B.data()),
      cute::make_layout(cute::make_shape(N, K, 1), stride_B));
  auto C = cute::make_tensor(thrust::raw_pointer_cast(tensor_C.data()),
      cute::make_layout(cute::make_shape(M, N, 1), stride_C));
  auto D = cute::make_tensor(thrust::raw_pointer_cast(tensor_ref_D.data()),
      cute::make_layout(cute::make_shape(M, N, 1), stride_D));
  auto SFA = cute::make_tensor(thrust::raw_pointer_cast(tensor_SFA.data()), layout_SFA);
  auto SFB = cute::make_tensor(thrust::raw_pointer_cast(tensor_SFB.data()), layout_SFB);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      float,
      decltype(A), 
      decltype(SFA), 
      decltype(B),
      decltype(SFB)
    > mainloop_params{A, SFA, B, SFB};

  cutlass::reference::host::GettEpilogueParams<
      float,
      float,
      float,
      float,
      decltype(C),
      decltype(D)
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.alpha = 1.0f;
  epilogue_params.beta = 1.0f;

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  bool equal = true;
  for (size_t i = 0; i < tensor_ref_D.size(); ++i) {
    equal &= (tensor_ref_D[i] == tensor_D[i]);
  }
  return equal;
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_e4m3t_tensorop_f32_align16_blockwise, 128x128x128_1x1x1_1x128x128_scale) {

  bool passed = groupwise_test<cute::GMMA::Major::MN, cute::GMMA::Major::K>(
      Int<1>{}, Int<128>{}, Int<128>{},
      cutlass::layout::RowMajor{}, cutlass::layout::ColumnMajor{}, 
      cutlass::layout::RowMajor{}, 
      Shape<_128,_128,_128>{},
      Shape<_1,_1,_1>{});

  EXPECT_TRUE(passed);
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_e4m3t_tensorop_f32_align16_blockwise, 128x128x128_1x1x1_1x1x128_scale) {

  bool passed = groupwise_test<cute::GMMA::Major::MN, cute::GMMA::Major::MN>(
      Int<1>{}, Int<128>{}, Int<128>{},
      cutlass::layout::RowMajor{}, cutlass::layout::ColumnMajor{}, 
      cutlass::layout::RowMajor{}, 
      Shape<_256,_128,_128>{},
      Shape<_2,_1,_1>{});

  EXPECT_TRUE(passed);

}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_e4m3t_tensorop_f32_align16_blockwise, 128x128x128_1x1x1_1x128x128_k_maj_k_maj_scale) {

  bool passed = groupwise_test<cute::GMMA::Major::K, cute::GMMA::Major::K>(
      Int<1>{}, Int<128>{}, Int<128>{},
      cutlass::layout::RowMajor{}, cutlass::layout::ColumnMajor{}, 
      cutlass::layout::RowMajor{}, 
      Shape<_128,_128,_128>{},
      Shape<_1,_1,_1>{});

  EXPECT_TRUE(passed);

}

#endif // #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
