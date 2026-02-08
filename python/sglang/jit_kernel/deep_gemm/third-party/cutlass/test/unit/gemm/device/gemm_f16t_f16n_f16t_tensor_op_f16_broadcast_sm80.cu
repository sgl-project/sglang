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
    \brief Tests for GEMM + broadcast interface
*/

#include <fstream>

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/kernel/default_gemm_with_broadcast.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_elementwise.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

template<typename GemmElement, typename LayoutA, typename LayoutB, typename LayoutC>
struct TestbedUtils {
  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<GemmElement, LayoutA> tensor_A;          // Input A
  cutlass::HostTensor<GemmElement, LayoutB> tensor_B;          // Input B
  cutlass::HostTensor<GemmElement, LayoutC> tensor_C;          // Input C
  cutlass::HostTensor<GemmElement, LayoutC> tensor_D1;         // Input D
  cutlass::HostTensor<GemmElement, LayoutC> tensor_D2;         // Input D
  cutlass::HostTensor<GemmElement, LayoutC> tensor_Y1;         // Input Y
  cutlass::HostTensor<GemmElement, LayoutC> tensor_Y2;         // Input Y
  cutlass::HostTensor<GemmElement, LayoutC> tensor_Y_ref;

  //
  // Methods
  //

  TestbedUtils(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
   init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<Element>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view);
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
    else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(cutlass::gemm::GemmCoord problem_size) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize({1, problem_size.n()});
    tensor_D1.resize(problem_size.mn());
    tensor_D2.resize(problem_size.mn());
    tensor_Y1.resize(problem_size.mn());
    tensor_Y2.resize(problem_size.mn());
    tensor_Y_ref.resize(problem_size.mn());

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    // Initialize D data to smaller data range. This helps avoid large roundoff errors.
    int d_scope_min = -2;
    int d_scope_max =  2;
    cutlass::reference::host::TensorFillRandomUniform(tensor_D1.host_view(), seed + 2016, d_scope_max, d_scope_min, 0);
    cutlass::reference::host::TensorFillRandomUniform(tensor_D2.host_view(), seed + 2015, d_scope_max, d_scope_min, 0);

    EXPECT_TRUE(initialize_tensor(tensor_Y1.host_view(), cutlass::Distribution::AllZeros, 0));
    EXPECT_TRUE(initialize_tensor(tensor_Y2.host_view(), cutlass::Distribution::AllZeros, 0));
    EXPECT_TRUE(initialize_tensor(tensor_Y_ref.host_view(), cutlass::Distribution::AllZeros, 0));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = GemmElement(1);
    tensor_B.host_view().at({0, 0}) = GemmElement(1);
    tensor_C.host_view().at({0, 0}) = GemmElement(1);
    tensor_D1.host_view().at({0, 0}) = GemmElement(1);
    tensor_D2.host_view().at({0, 0}) = GemmElement(1);

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D1.sync_device();
    tensor_D2.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::gemm::GemmCoord problem_size, cutlass::HostTensor<GemmElement, LayoutC>& tensor_Y_ref, cutlass::HostTensor<GemmElement, LayoutC>& tensor_Y) {

    tensor_Y_ref.sync_host();
    tensor_Y.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D1.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D2.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Y_ref.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Y.host_view()), 0);

    bool passed = true;
    float norm_diff = 0;

    norm_diff = cutlass::reference::host::TensorNormDiff(tensor_Y_ref.host_view(), tensor_Y.host_view(), float());
    passed = (norm_diff <= 0.1f);
    EXPECT_LT(norm_diff, 0.1f) << " tensor_Y is incorrect";


    if (!passed) {
      std::ofstream file("errors_testbed_gemm_broadcast_new.txt");


      file
        << "problem: " << problem_size << "\n\n";

      file
        << "capacity: \n"
        << "A: " << tensor_A.capacity()
        << "\nB: " << tensor_B.capacity()
        << "\nC: " << tensor_C.capacity()
        << "\nD1: " << tensor_D1.capacity()
        << "\nD2: " << tensor_D2.capacity()
        << "\nY: " << tensor_Y.capacity()
        << "\n\n"
        << "\nY_ref: " << tensor_Y_ref.capacity()
        << "\n\n";
      file
        << "A =\n" << tensor_A.host_view()
        << "\n\nB =\n" << tensor_B.host_view()
        << "\n\nC =\n" << tensor_C.host_view()
        << "\n\nD1 =\n" << tensor_D1.host_view()
        << "\n\nD2 =\n" << tensor_D2.host_view()
        << "\n\nY =\n" << tensor_Y.host_view()
        << "\n\nY_ref =\n" << tensor_Y_ref.host_view();
    }

    return passed;
  }
};

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

TEST(SM80_Device_GemmWithBroadcast_f16t_f16n_f16t_tensor_op_f16, 128x128_32x3_64x64x32_16x8x16) {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using OpClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
    const int kStages = 3;

    const int batch_count = 1;
    const cutlass::half_t alpha(1);
    const cutlass::half_t beta(1);

    const int M = 1024;
    const int K = 10240;
    const int N = 512;
    cutlass::gemm::GemmCoord problem{M, N, K};

    const int batch_stride_A = 0;
    const int batch_stride_B = 0;
    const int batch_stride_C1 = 0;
    const int batch_stride_C2 = 0;
    const int batch_stride_D = 0;
    const int batch_stride_Vector = 0;
    const int batch_stride_Tensor = 0;

    const int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    const int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    const int64_t ldc1 = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    const int64_t ldc2 = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    const int64_t ldd = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    const int64_t ldv = 0;
    const int64_t ldt = 0;

    TestbedUtils<ElementA, LayoutA, LayoutB, LayoutC> utils;
    utils.initialize(problem);

    //
    // Create reference Gemm
    //
    using GemmRef = cutlass::gemm::device::GemmUniversal<
        ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
         OpClass, ArchTag, ThreadblockShape,  WarpShape, InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ThreadblockSwizzle, kStages>;

    typename GemmRef::Arguments args_ref{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      batch_count,
      {alpha, beta},
      utils.tensor_A.device_data(),
      utils.tensor_B.device_data(),
      utils.tensor_C.device_data(),
      utils.tensor_Y_ref.device_data(),
      batch_stride_A,
      batch_stride_B,
      batch_stride_C1,
      batch_stride_D,
      lda,
      ldb,
      ldv,
      ldd,
    };

    GemmRef gemm_op_ref;
    size_t workspace_size_ref = GemmRef::get_workspace_size(args_ref);
    cutlass::device_memory::allocation<uint8_t> workspace_ref(workspace_size_ref);
    cutlass::Status status = gemm_op_ref.initialize(args_ref, workspace_ref.get());
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);

    status = gemm_op_ref();
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);

    //
    // Create GemmWithBroadcast from single source
    //
    using GemmSingle = cutlass::gemm::device::GemmUniversalWithBroadcast<
        ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
         OpClass, ArchTag, ThreadblockShape,  WarpShape, InstructionShape,
        cutlass::epilogue::thread::LinearCombinationResidualBlock<
            ElementOutput, ElementAccumulator, ElementAccumulator,
            ElementAccumulator, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            cutlass::epilogue::thread::Identity, cutlass::multiplies, cutlass::epilogue::thread::Identity>,
        ThreadblockSwizzle, kStages>;

    typename GemmSingle::Arguments args_single{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      batch_count,
      {alpha, beta},
      utils.tensor_A.device_data(),
      utils.tensor_B.device_data(),
      utils.tensor_D1.device_data(),
      utils.tensor_Y1.device_data(),
      utils.tensor_C.device_data(),
      /* ptr_Tensor = */ nullptr,
      batch_stride_A,
      batch_stride_B,
      batch_stride_C1,
      batch_stride_D,
      batch_stride_Vector,
      batch_stride_Tensor,
      lda,
      ldb,
      ldc1,
      ldd,
      ldv,
      ldt
    };

    GemmSingle gemm_op_single;
    size_t workspace_size_single = GemmSingle::get_workspace_size(args_single);
    cutlass::device_memory::allocation<uint8_t> workspace_single(workspace_size_single);
    status = gemm_op_single.initialize(args_single, workspace_single.get());
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);

    status = gemm_op_single();
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);

    // Compute the broadcast on the reference previously computed and compare results
    utils.tensor_Y_ref.sync_host();
    cutlass::reference::host::TensorMul(utils.tensor_Y_ref.host_view(), utils.tensor_D1.host_view());
    utils.tensor_Y_ref.sync_device();
    utils.compare_reference(problem, utils.tensor_Y_ref, utils.tensor_Y1);

    //
    // Create GemmWithBroadcast from two sources
    //
    using GemmDouble = cutlass::gemm::device::GemmUniversalWithBroadcast<
        ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
         OpClass, ArchTag, ThreadblockShape,  WarpShape, InstructionShape,
        cutlass::epilogue::thread::LinearCombinationResidualBlock<
            ElementOutput, ElementAccumulator, ElementAccumulator,
            ElementAccumulator, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            cutlass::epilogue::thread::Identity, cutlass::multiplies, cutlass::epilogue::thread::Identity, cutlass::plus>,
        ThreadblockSwizzle, kStages>;

    typename GemmDouble::Arguments args_double{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      batch_count,
      {alpha, beta},
      utils.tensor_A.device_data(),
      utils.tensor_B.device_data(),
      utils.tensor_D1.device_data(),
      utils.tensor_D2.device_data(),
      utils.tensor_Y2.device_data(),
      utils.tensor_C.device_data(),
      /* ptr_Tensor = */ nullptr,
      batch_stride_A,
      batch_stride_B,
      batch_stride_C1,
      batch_stride_C2,
      batch_stride_D,
      batch_stride_Vector,
      batch_stride_Tensor,
      lda,
      ldb,
      ldc1,
      ldc2,
      ldd,
      ldv,
      ldt
    };

    GemmDouble gemm_op_double;
    size_t workspace_size_double = GemmDouble::get_workspace_size(args_double);
    cutlass::device_memory::allocation<uint8_t> workspace_double(workspace_size_double);
    status = gemm_op_double.initialize(args_double, workspace_double.get());
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);

    status = gemm_op_double();
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);

    // Compute the broadcast on the reference previously computed and compare results
    utils.tensor_Y_ref.sync_host();
    cutlass::reference::host::TensorAdd(utils.tensor_Y_ref.host_view(), utils.tensor_D2.host_view());
    utils.tensor_Y_ref.sync_device();
    utils.compare_reference(problem, utils.tensor_Y_ref, utils.tensor_Y2);
}

#endif
