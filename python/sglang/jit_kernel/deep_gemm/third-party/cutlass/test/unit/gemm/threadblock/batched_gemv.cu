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
    \brief Unit tests for threadblock level GEMV
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/gemm/threadblock/gemv.h"
#include "cutlass/gemm/threadblock/default_gemv_core.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemv, typename LongIndex, typename RefA, typename RefB, typename RefC>
__global__ void batched_gemv_threadblock_test_kernel(
  cutlass::gemm::GemmCoord problem_size,
  LongIndex stride_a,
  LongIndex stride_b,
  LongIndex stride_c,
  RefA ref_A,
  RefB ref_B,
  RefC ref_C
  ) {

  typename Gemv::IteratorA::TensorCoord threadblock_offset_A(0, 0);
  typename Gemv::IteratorB::TensorCoord threadblock_offset_B(0, 0);
  typename Gemv::IteratorB::TensorCoord threadblock_offset_C(0, 0);

  // Move to the right batches for these threads
  ref_A.add_pointer_offset(threadIdx.y * stride_a);
  ref_B.add_pointer_offset(threadIdx.y * stride_b);
  ref_C.add_pointer_offset(threadIdx.y * stride_c);

  // Construct iterators to A and B operands
  typename Gemv::IteratorA::Params params_A(ref_A.layout());
  typename Gemv::IteratorA iterator_A(params_A, ref_A.data(), { problem_size.m(), problem_size.k() }, 0, threadblock_offset_A);
  typename Gemv::IteratorB::Params params_B(ref_B.layout());
  typename Gemv::IteratorB iterator_B(params_B, ref_B.data(), { problem_size.k(), problem_size.n() }, threadIdx.x, threadblock_offset_B);

  Gemv gemv;

  typename Gemv::FragmentC accum;
  accum.clear();

  // Compute threadblock-scoped matrix multiply-add
  gemv(problem_size, accum, iterator_A, iterator_B, accum);

  // IteratorC is PitchLinear<> assumes n() contiguous
  typename Gemv::IteratorC::Params params_C(ref_C.layout());
  typename Gemv::IteratorC iterator_C(params_C, ref_C.data(), { problem_size.m(), problem_size.n() }, threadIdx.x, threadblock_offset_C);
  iterator_C.store(accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Shape_,
         typename ElementAB_,
         typename ElementC_,
         typename LayoutA_,
         typename LayoutB_,
         typename LayoutC_,
         int THREAD_N,
         int THREAD_K,
         int MAX_THREADS_PER_BLOCK=512,
         bool DEBUG=false>
void batched_gemv_threadblock_test(cutlass::gemm::GemmCoord problem_size, int num_batch)
{
  using Shape = Shape_;
  using ElementA = ElementAB_;
  using LayoutA = LayoutA_;
  using ElementB = ElementAB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ThreadShape = cutlass::gemm::GemmShape<1, THREAD_N, THREAD_K>;

  using Core = typename cutlass::gemm::threadblock::DefaultGemvCore<
    Shape,
    ThreadShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC
  >;

  if (DEBUG)
  { 
      num_batch = 1;
  }

  using Mma = cutlass::gemm::threadblock::Gemv<Core>;

  // Create host tensors that will be the backing store for the batches
  // Note that no device memory is initially allocated
  cutlass::HostTensor<ElementA, LayoutA> matrix_A({problem_size.m(), problem_size.k()}, false); 
  cutlass::HostTensor<ElementB, LayoutB> matrix_B({problem_size.k(), problem_size.n()}, false); 
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_computed({problem_size.m(), problem_size.n()}, false); 
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_reference({problem_size.m(), problem_size.n()}, false);

  // Reserve memory for the batch of tensors
  matrix_A.reserve(problem_size.m()*problem_size.k()*num_batch);
  matrix_B.reserve(problem_size.n()*problem_size.k()*num_batch);
  matrix_C_computed.reserve(problem_size.m()*problem_size.n()*num_batch);
  matrix_C_reference.reserve(problem_size.m()*problem_size.n()*num_batch, false);

  // Fill eatch tensor batch
  const int seed = 6834;
  for (int b = 0; b < num_batch; b++)
  {
    if(DEBUG)
    {
      cutlass::reference::host::BlockFillSequential(
        matrix_A.host_data_ptr_offset(b*matrix_A.capacity()), matrix_A.capacity());
      cutlass::reference::host::BlockFillSequential(
        matrix_B.host_data_ptr_offset(b*matrix_B.capacity()), matrix_B.capacity());
    }
    else
    {
      cutlass::reference::host::TensorFillRandomUniform(
        matrix_A.host_view(b*matrix_A.capacity()),
        seed + 1660,
        8,
        -8,
        0
      );

      cutlass::reference::host::TensorFillRandomUniform(
        matrix_B.host_view(b*matrix_B.capacity()),
        seed + 1880,
        8,
        -8,
        0
      );
    }

    cutlass::reference::host::TensorFill(matrix_C_computed.host_view(b*matrix_C_computed.capacity()));
    cutlass::reference::host::TensorFill(matrix_C_reference.host_view(b*matrix_C_reference.capacity()));
  }

  matrix_A.sync_device();
  matrix_B.sync_device();
  matrix_C_computed.sync_device();

  dim3 grid(1, 1);      // only 1 CTA is used
  dim3 block(Shape::kN / THREAD_N, num_batch, 1);

  #if 0
  printf("block dim = %d x %d\n", block.x, block.y);
  #endif

  // Some sanity checks
  EXPECT_TRUE( problem_size.n() % THREAD_N == 0 );
  EXPECT_TRUE( block.x*block.y <= MAX_THREADS_PER_BLOCK );

  test::gemm::threadblock::batched_gemv_threadblock_test_kernel<Mma><<< grid, block >>>(
    problem_size,
    matrix_A.capacity(),
    matrix_B.capacity(),
    matrix_C_computed.capacity(),
    matrix_A.device_ref(),
    matrix_B.device_ref(),
    matrix_C_computed.device_ref()
  );

  cudaError_t result = cudaDeviceSynchronize();
  EXPECT_EQ(result, cudaSuccess) << " kernel error: " << cudaGetErrorString(result);

  matrix_C_computed.sync_host();

  // Compute the batched gemms
  for (int b = 0; b < num_batch; b++)
  {

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC> reference_gemm;

    reference_gemm(
      problem_size.mnk(),
      ElementC(1),
      matrix_A.host_ref(b*matrix_A.capacity()),
      matrix_B.host_ref(b*matrix_B.capacity()),
      ElementC(0),
      matrix_C_reference.host_ref(b*matrix_C_computed.capacity())
    );

    bool passed = cutlass::reference::host::TensorEquals(
                    matrix_C_computed.host_view(b*matrix_C_computed.capacity()), 
                    matrix_C_reference.host_view(b*matrix_C_reference.capacity()));

    EXPECT_TRUE(passed)
    //<< "A:\n" << matrix_A.host_view() << "\n"
    //<< "B:\n" << matrix_B.host_view() << "\n"
      << "Batch: " << b << "\n"
      << "Reference:\n" << matrix_C_reference.host_view(b*matrix_C_reference.capacity()) << "\n"
      << "Computed:\n" << matrix_C_computed.host_view(b*matrix_C_computed.capacity()) << "\n";
  }
}

} // namespace threadblock
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

// A: ColumnMajor
// B: RowMajor
// C: ColumnMajor

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_crc_fp32_fp32_2N_2K) {
  
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 2;
 
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape, float, float, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 5x1x128x128_crc_fp32_fp32_4N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 128, 128);
  const int num_batch = 5;
  const int THREAD_N = 4;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 128, THREAD_K>;
  batched_gemv_threadblock_test<Shape, float, float, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_crc_fp32_fp32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                float, float, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_crc_fp16_fp32_2N_2K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 2;
  
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_crc_fp16_fp32_2N_8K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 8;
 
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_crc_fp16_fp32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_crc_i8_i32_2N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 128, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                int8_t, int32_t, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_crc_i8_i32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                int8_t, int32_t, 
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

// A: RowMajor
// B: ColumnMajor
// C: RowMajor

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcr_fp32_fp32_2N_2K) {
  
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 2;
 
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape, float, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 5x1x128x128_rcr_fp32_fp32_4N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 128, 128);
  const int num_batch = 5;
  const int THREAD_N = 4;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 128, THREAD_K>;
  batched_gemv_threadblock_test<Shape, float, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_rcr_fp32_fp32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                float, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcr_fp16_fp32_2N_2K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 2;
  
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcr_fp16_fp32_2N_8K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 8;
 
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_rcr_fp16_fp32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcr_i8_i32_2N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 128, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                int8_t, int32_t, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_rcr_i8_i32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                int8_t, int32_t, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

// A: RowMajor
// B: ColumnMajor
// C: ColumnMajor

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcc_fp32_fp32_2N_2K) {
  
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 2;
 
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape, float, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 5x1x128x128_rcc_fp32_fp32_4N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 128, 128);
  const int num_batch = 5;
  const int THREAD_N = 4;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 128, THREAD_K>;
  batched_gemv_threadblock_test<Shape, float, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_rcc_fp32_fp32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                float, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcc_fp16_fp32_2N_2K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 2;
  
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcc_fp16_fp32_2N_8K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 8;
 
  using Shape = cutlass::gemm::GemmShape<1, 64, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_rcc_fp16_fp32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                cutlass::half_t, float, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 4x1x64x64_rcc_i8_i32_2N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 64, 64);
  const int num_batch = 4;
  const int THREAD_N = 2;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 128, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                int8_t, int32_t, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}

TEST(SM50_batched_gemv_threadblock, 16x1x17x64_rcc_i8_i32_1N_4K) {
  using namespace test::gemm::threadblock;
  cutlass::gemm::GemmCoord problem_size(1, 17, 64);
  const int num_batch = 16;
  const int THREAD_N = 1;
  const int THREAD_K = 4;

  using Shape = cutlass::gemm::GemmShape<1, 32, THREAD_K>;
  batched_gemv_threadblock_test<Shape,
                                int8_t, int32_t, 
                                cutlass::layout::RowMajor,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::ColumnMajor,
                                THREAD_N, THREAD_K>(problem_size, num_batch);
}
