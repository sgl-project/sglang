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
    \brief Tests for device-wide GEMV interface
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/gemv.h"
#include "cutlass/gemm/device/gemv.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/gemm_complex.h"

#include "testbed_utils.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {

template <typename Gemv>
class TestbedGemv {
public:

  using ElementA = typename Gemv::ElementA;
  using LayoutA  = typename Gemv::LayoutA;
  using ElementB = typename Gemv::ElementB;
  using ElementC = typename Gemv::ElementC;

  using ElementAccumulator = typename Gemv::ElementAccumulator;
  using ElementCompute = typename Gemv::EpilogueOutputOp::ElementCompute;

  using LayoutV = cutlass::layout::RowMajor;

private:

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutV> tensor_B;
  cutlass::HostTensor<ElementC, LayoutV> tensor_C;
  cutlass::HostTensor<ElementC, LayoutV> tensor_D;
  cutlass::HostTensor<ElementC, LayoutV> reference_D;

public:

  //
  // Methods
  //

  TestbedGemv(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2023
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
      int bits_output = cutlass::sizeof_bits<typename Gemv::ElementC>::value;

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
  void initialize(
    cutlass::MatrixCoord problem_size,
    int32_t batch_count
  ) {

    //
    // Allocate the GEMV workspace
    //

    if(std::is_same<LayoutA, cutlass::layout::ColumnMajor>::value) {
      tensor_A.resize({problem_size.row(), batch_count * problem_size.column()});
    }
    else {
      tensor_A.resize({batch_count * problem_size.row(), problem_size.column()});
    }
    
    tensor_B.resize({batch_count * problem_size.column(), 1});
    tensor_C.resize({batch_count * problem_size.row(), 1});
    tensor_D.resize({batch_count * problem_size.row(), 1});
    reference_D.resize({batch_count * problem_size.row(), 1}, false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 1));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 3));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Gemv::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Gemv::ElementB(1);
    tensor_C.host_view().at({0, 0}) = typename Gemv::ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }  

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::MatrixCoord problem_size, 
    ElementCompute alpha, 
    ElementCompute beta) {

    tensor_D.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);
    
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed) << " mismatched reference";

    if (!passed) {

      std::ofstream file("testbed_universal_errors.txt");

      file
        << "problem: " << problem_size 
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file 
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result
  bool verify(
    cutlass::MatrixCoord problem_size,
    int32_t batch_count,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_D,
    ElementCompute alpha, 
    ElementCompute beta) {

    //
    // Verify
    //

    cutlass::reference::host::GemmComplex<
        typename Gemv::ElementA, typename Gemv::LayoutA,
        typename Gemv::ElementB, LayoutV,
        typename Gemv::ElementC, LayoutV, 
        ElementCompute, ElementAccumulator
    >(
      {problem_size.row(), 1, problem_size.column()},
      alpha,
      tensor_A.host_ref(),
      Gemv::kTransformA,
      tensor_B.host_ref(),
      Gemv::kTransformB,
      beta, 
      tensor_C.host_ref(), 
      reference_D.host_ref(), 
      ElementAccumulator(0),
      batch_count,
      batch_stride_A,
      batch_stride_B,
      batch_stride_C,
      batch_stride_D
    );

    return compare_reference(problem_size, alpha, beta);
  }

  /// Runs one problem size
  bool run(
    cutlass::MatrixCoord problem_size, 
    int32_t batch_count,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_D,
    ElementCompute alpha,
    ElementCompute beta) {

    this->initialize(problem_size, batch_count);

    //
    // Initialize the GEMV operator
    //

    typename Gemv::Arguments arguments{
      problem_size,
      batch_count,
      {alpha, beta},
      tensor_A.device_ref(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_D.device_data(),
      batch_stride_A,
      batch_stride_B,
      batch_stride_C,
      batch_stride_D
    };

    Gemv gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    size_t workspace_size = Gemv::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the GEMV
    //

    status = gemm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //

    bool passed = this->verify(
        problem_size,
        batch_count,
        batch_stride_A,
        batch_stride_B,
        batch_stride_C,
        batch_stride_D,
        alpha,
        beta);
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemv> 
bool TestAllGemv() {

  using ElementCompute = typename Gemv::EpilogueOutputOp::ElementCompute;

  int Batch[] = {
    1, 520, 1314
  };

  int M[] = {
    1, 5, 16
  };

  int K[] = {
    8, 128, 256
  };

  double Alpha[] = {
    1, 1.25
  };

  double Beta[] = {
    0, 1, 1.25
  };

  for (int b : Batch) {
    for (int m : M) {
      for (int k : K) {
        for (double alpha : Alpha) {
          for (double beta : Beta) {

            TestbedGemv<Gemv> testbed;

            if (!testbed.run(
                    {m, k},
                    b,
                    m * k,
                    k,
                    m,
                    m,
                    ElementCompute(alpha),
                    ElementCompute(beta))) {
              return false;
            }
          }
        }
      }
    }
  }

  return true;
}

} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Gemv_f16n_f16_f16_simt_f32, RowMajorA) {

  using ElementInput = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  int const kElementsPerAccess = 8;
  
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementAccumulator>;

  using Gemv = cutlass::gemm::device::Gemv<
      cutlass::gemm::kernel::Gemv<
          ElementInput,           // Element A
          LayoutA,                // Layout A
          ElementInput,           // Element B
          ElementOutput,          // Element C
          ElementAccumulator,     // Element accumulator
          EpilogueOp,             // Output operator
          kElementsPerAccess      // Element access granularity
          >
      >;

  EXPECT_TRUE(test::gemm::TestAllGemv<Gemv>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Gemv_f32n_f32_f32_simt_f32, RowMajorA) {

  using ElementInput = float;
  using ElementOutput = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  int const kElementsPerAccess = 4;
  
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementAccumulator>;

  using Gemv = cutlass::gemm::device::Gemv<
      cutlass::gemm::kernel::Gemv<
          ElementInput,           // Element A
          LayoutA,                // Layout A
          ElementInput,           // Element B
          ElementOutput,          // Element C
          ElementAccumulator,     // Element accumulator
          EpilogueOp,             // Output operator
          kElementsPerAccess      // Element access granularity
          >
      >;

  EXPECT_TRUE(test::gemm::TestAllGemv<Gemv>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Gemv_f64n_f64_f64_simt_f64, RowMajorA) {

  using ElementInput = double;
  using ElementOutput = double;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementAccumulator = double;
  int const kElementsPerAccess = 2;
  
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementAccumulator>;

  using Gemv = cutlass::gemm::device::Gemv<
      cutlass::gemm::kernel::Gemv<
          ElementInput,           // Element A
          LayoutA,                // Layout A
          ElementInput,           // Element B
          ElementOutput,          // Element C
          ElementAccumulator,     // Element accumulator
          EpilogueOp,             // Output operator
          kElementsPerAccess      // Element access granularity
          >
      >;

  EXPECT_TRUE(test::gemm::TestAllGemv<Gemv>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Gemv_f16n_f16_f16_simt_f32, ColumnMajorA) {

  using ElementInput = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator, 
      ElementAccumulator>;

  using Gemv = cutlass::gemm::device::Gemv<
    cutlass::gemm::kernel::Gemv<
        ElementInput,           // Element A
        LayoutA,                // Layout A
        ElementInput,           // Element B
        ElementOutput,          // Element C
        ElementAccumulator,     // Element Accumulator
        EpilogueOp              // Output operator
        >
    >;


  EXPECT_TRUE(test::gemm::TestAllGemv<Gemv>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Gemv_f32n_f32_f32_simt_f32, ColumnMajorA) {

  using ElementInput = float;
  using ElementOutput = float;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator, 
      ElementAccumulator>;

  using Gemv = cutlass::gemm::device::Gemv<
    cutlass::gemm::kernel::Gemv<
        ElementInput,           // Element A
        LayoutA,                // Layout A
        ElementInput,           // Element B
        ElementOutput,          // Element C
        ElementAccumulator,     // Element Accumulator
        EpilogueOp              // Output operator
        >
    >;


  EXPECT_TRUE(test::gemm::TestAllGemv<Gemv>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Gemv_f64n_f64_f64_simt_f64, ColumnMajorA) {

  using ElementInput = double;
  using ElementOutput = double;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementAccumulator = double;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator, 
      ElementAccumulator>;

  using Gemv = cutlass::gemm::device::Gemv<
    cutlass::gemm::kernel::Gemv<
        ElementInput,           // Element A
        LayoutA,                // Layout A
        ElementInput,           // Element B
        ElementOutput,          // Element C
        ElementAccumulator,     // Element Accumulator
        EpilogueOp              // Output operator
        >
    >;


  EXPECT_TRUE(test::gemm::TestAllGemv<Gemv>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
