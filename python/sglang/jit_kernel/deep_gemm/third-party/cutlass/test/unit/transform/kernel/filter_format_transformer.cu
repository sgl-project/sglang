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
    \brief Tests cutlass::transform::kernel::ConvFilterFormatTransformer
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/cutlass.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/kernel/filter_format_transformer.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"

#include "thrust/universal_vector.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element, class Shape_S>
auto verify_ckrs_to_crsk(thrust::host_vector<Element> const &S, thrust::host_vector<Element> const &D, Shape_S shape_s) {
  using namespace cute;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  if (S.size() != D.size()) {
    return false;
  }

  auto shape_d = select<2, 0, 1, 3>(shape_s);

  for (int i = 0; i < (int)S.size(); ++i) {
    auto [s, r, k, c] = idx2crd(i, shape_s);
    auto d_idx = crd2idx(make_coord(k, s, r, c), shape_d);

    if (S[i] != D[d_idx]) {
      std::cerr << "Error. S[" << i << "]: " << S[i] << ",   D[" << d_idx << "]: " << D[d_idx] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return false;
      }
    }
  }

  return errors == 0;
}

template <class Element, class Shape_S>
auto verify_ckrs_to_krsc(thrust::host_vector<Element> const &S, thrust::host_vector<Element> const &D, Shape_S shape_s) {
  using namespace cute;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  if (S.size() != D.size()) {
    return false;
  }

  auto shape_d = select<3, 0, 1, 2>(shape_s);

  for (int i = 0; i < (int)S.size(); ++i) {
    auto [s, r, k, c] = idx2crd(i, shape_s);
    auto d_idx = crd2idx(make_coord(c, s, r, k), shape_d);

    if (S[i] != D[d_idx]) {
      std::cerr << "Error. S[" << i << "]: " << S[i] << ",   D[" << d_idx << "]: " << D[d_idx] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return false;
      }
    }
  }

  return errors == 0;
}

template <class Element,
          cutlass::transform::kernel::FilterFormat SrcFormat,
          cutlass::transform::kernel::FilterFormat DstFormat,
          int Alignment = 16>
bool transform_test() {
  using namespace cute;

  using TransformKernel = cutlass::transform::kernel::ConvFilterFormatTransformer<SrcFormat, DstFormat, 4, Element, Alignment>;
  using Transform = cutlass::transform::device::TransformUniversalAdapter<TransformKernel>;

  auto s = 3;
  auto r = 3;
  auto k = 64 + Alignment / (int)(sizeof(Element));
  auto c = 64 + Alignment / (int)(sizeof(Element));

  thrust::host_vector<Element> h_S(s * r * k * c);
  thrust::host_vector<Element> h_D(s * r * k * c);

  //
  // Initialize
  //

  for (int i = 0; i < (int)h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  Transform transform_op;

  const void* src_ptr = static_cast<const void *>(d_S.data().get());
  void* dst_ptr = static_cast<void *>(d_D.data().get());

  typename TransformKernel::FilterExtent filter_extent;
  filter_extent[0] = k;
  filter_extent[1] = r;
  filter_extent[2] = s;
  filter_extent[3] = c;

  auto args = typename Transform::Arguments {
    src_ptr,
    dst_ptr,
    filter_extent
  };

  cutlass::Status status = cutlass::Status::kInvalid;

  size_t workspace_size = Transform::get_workspace_size(args);
  thrust::universal_vector<uint8_t> workspace(workspace_size);

  status = transform_op.initialize(args, workspace.data().get());
  if (status != cutlass::Status::kSuccess) {
    cudaError_t error = cudaGetLastError();
    std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
    return false;
  }

  status = transform_op();

  EXPECT_TRUE(status == cutlass::Status::kSuccess);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  cudaError_t result = cudaDeviceSynchronize();
  EXPECT_EQ(result, cudaSuccess) << " Kernel execution error: "
                                 << cudaGetErrorString(result);

  // Verification
  h_D = d_D;
  auto tensor_shape_S = make_shape(s, r, k, c);

  bool passed = false;
  if constexpr(DstFormat == cutlass::transform::kernel::FilterFormat::KTRSC) {
    // KTRSC
    passed = verify_ckrs_to_krsc(h_S, h_D, tensor_shape_S);
  }
  else if constexpr(DstFormat == cutlass::transform::kernel::FilterFormat::CTRSK) {
    // CTRSK;
    passed = verify_ckrs_to_crsk(h_S, h_D, tensor_shape_S);
  }

  return passed;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

TEST(Transform_kernel_ConvFilterFormatTransformer, ckrs_to_crsk) {
  bool passed = true;

  // fp16 kernel with alignment bytes from 16 to 2.
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK>();
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 8>();
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 4>();
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 2>();

  // fp8 kernel with alignment bytes from 16 to 1.
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 8>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 4>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 2>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 1>();

  // int8 kernel with alignment bytes from 16 to 1.
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 8>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 4>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 2>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 1>();

  // fp32 kernel with alignment bytes from 16 to 4.
  passed &= transform_test<float, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK>();
  passed &= transform_test<float, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 8>();
  passed &= transform_test<float, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::CTRSK, 4>();

  EXPECT_TRUE(passed);
}

// CKRS -> KRSC
TEST(Transform_kernel_ConvFilterFormatTransformer, ckrs_to_krsc) {
  bool passed = true;

  // fp16 kernel with alignment bytes from 16 to 2.
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC>();
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 8>();
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 4>();
  passed &= transform_test<cutlass::half_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 2>();

  // fp8 kernel with alignment bytes from 16 to 1.
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 8>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 4>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 2>();
  passed &= transform_test<cutlass::float_e4m3_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 1>();

  // int8 kernel with alignment bytes from 16 to 1.
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 8>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 4>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 2>();
  passed &= transform_test<int8_t, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 1>();

  // fp32 kernel with alignment bytes from 16 to 4.
  passed &= transform_test<float, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC>();
  passed &= transform_test<float, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 8>();
  passed &= transform_test<float, cutlass::transform::kernel::FilterFormat::CKTRS, cutlass::transform::kernel::FilterFormat::KTRSC, 4>();

  EXPECT_TRUE(passed);
}

#endif
