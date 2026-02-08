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

#pragma once

#include "../../common/cutlass_unit_test.h"

#include "cutlass/core_io.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/gemm/kernel/default_gemv.h"
#include "cutlass/gemm/kernel/gemv_batched_strided.h"

namespace test {
namespace gemm {
namespace kernel {

template<typename ThreadBlockShape_,
        typename ThreadShape_,
        typename ElementAB_,
        typename ElementAccumulator_,
        typename ElementCD_,
        typename LayoutA_,
        typename LayoutB_,
        typename LayoutCD_,
        int THREAD_B = 1, // batch tile size
        bool DEBUG=false>
void batched_gemv_kernel_test(cutlass::gemm::BatchedGemmCoord problem_size,
                              ElementCD_ alpha = ElementCD_(1),
                              ElementCD_ beta = ElementCD_(0),
                              bool perf_test = false,
                              int perf_test_iter = 1)
{
    using ThreadBlockShape = ThreadBlockShape_;
    using ThreadShape = ThreadShape_;
    using ElementA = ElementAB_;
    using LayoutA = LayoutA_;
    using ElementB = ElementAB_;
    using LayoutB = LayoutB_;
    using ElementAccumulator = ElementCD_;
    using ElementCD = ElementCD_;
    using LayoutCD = LayoutCD_;

    using GemvKernel = cutlass::gemm::kernel::DefaultGemv<ThreadBlockShape,
                                                          ThreadShape,
                                                          ElementA,
                                                          LayoutA,
                                                          ElementB,
                                                          LayoutB,
                                                          ElementCD,
                                                          LayoutCD,
                                                          ElementAccumulator>;

    using ThreadBlockGemv = typename GemvKernel::ThreadBlockGemv;
    using ThreadBlockSwizzle = typename GemvKernel::ThreadBlockSwizzle;

    if (DEBUG)
    { 
        problem_size = cutlass::gemm::BatchedGemmCoord(
                        problem_size.m(), problem_size.n(), problem_size.k(), 1);
    }

    // Create host tensors that will be the backing store for the batches
    // Note that no device memory is initially allocated
    cutlass::HostTensor<ElementA, LayoutA> matrix_A({problem_size.m(), problem_size.k()}, false); 
    cutlass::HostTensor<ElementB, LayoutB> matrix_B({problem_size.k(), problem_size.n()}, false); 
    cutlass::HostTensor<ElementCD, LayoutCD> matrix_C_computed({problem_size.m(), problem_size.n()}, false); 
    cutlass::HostTensor<ElementCD, LayoutCD> matrix_C_reference({problem_size.m(), problem_size.n()}, false);

    // Reserve memory for the batch of tensors
    matrix_A.reserve(problem_size.m()*problem_size.k()*problem_size.batch());
    matrix_B.reserve(problem_size.n()*problem_size.k()*problem_size.batch());
    matrix_C_computed.reserve(problem_size.m()*problem_size.n()*problem_size.batch());
    matrix_C_reference.reserve(problem_size.m()*problem_size.n()*problem_size.batch(), false);

    // Fill eatch tensor batch
    const int seed = 9876;
    for (int b = 0; b < problem_size.batch(); b++)
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

    ThreadBlockSwizzle swizzle;

    cutlass::gemm::BatchedGemmCoord tiled_size{ThreadBlockShape::kM,
                                                ThreadBlockShape::kN,
                                                problem_size.k(), // no split-k
                                                DEBUG ? 1 : THREAD_B };

    cutlass::gemm::BatchedGemmCoord tiled_shape = swizzle.get_tiled_shape(problem_size, tiled_size);

    #if 0 
    printf("tiled_size = %d %d %d %d\n", tiled_size.m(), tiled_size.n(), tiled_size.k(), tiled_size.batch());
    printf("tiled_shape = %d %d %d %d\n", tiled_shape.m(), tiled_shape.n(), tiled_shape.k(), tiled_shape.batch());
    #endif

    // No split-k
    EXPECT_EQ(tiled_size.k(), problem_size.k());

    dim3 grid = swizzle.get_grid_shape(tiled_shape);
    dim3 block(tiled_size.n() / ThreadShape::kN, tiled_size.batch(), tiled_size.k() / problem_size.k());

    // Some sanity checks
    EXPECT_TRUE( block.x*block.y*block.z <= 1024 );
    EXPECT_TRUE( block.x <= 1024 );
    EXPECT_TRUE( block.y <= 1024 );
    EXPECT_TRUE( block.z <= 64 );

    #if 0 
    printf("grid dim = %d, %d, %d\n", grid.x, grid.y, grid.z);
    printf("block dim = %d, %d, %d\n", block.x, block.y, block.z);
    #endif

    cudaError_t result;
    cudaEvent_t start_event, end_event;
 
    for (int iter = 0; iter < (perf_test ? (perf_test_iter+1) : 1); ++iter)
    {
        if (perf_test && iter == 1)
        {
            result = cudaEventCreate(&start_event);
            EXPECT_EQ(result, cudaSuccess);
            
            result = cudaEventCreate(&end_event);
            EXPECT_EQ(result, cudaSuccess);
    
            result = cudaEventRecord(start_event);
            EXPECT_EQ(result, cudaSuccess);
        }

        if (beta == ElementCD(0))
        {
            if (alpha == ElementCD(1))
            {
                cutlass::gemm::kernel::GemvBatchedStrided<GemvKernel><<< grid, block >>>(
                    problem_size,
                    matrix_A.device_ref(),
                    matrix_A.capacity(),
                    matrix_B.device_ref(),
                    matrix_B.capacity(),
                    matrix_C_computed.device_ref(),
                    matrix_C_computed.capacity()
                );
            }
            else
            {
                cutlass::gemm::kernel::GemvBatchedStrided<GemvKernel><<< grid, block >>>(
                    problem_size,
                    alpha,
                    matrix_A.device_ref(),
                    matrix_A.capacity(),
                    matrix_B.device_ref(),
                    matrix_B.capacity(),
                    matrix_C_computed.device_ref(),
                    matrix_C_computed.capacity()
                );
            }
        }
        else
        {
            cutlass::gemm::kernel::GemvBatchedStrided<GemvKernel, ElementCD, false><<< grid, block >>>(
                problem_size,
                alpha,
                beta,
                matrix_A.device_ref(),
                matrix_A.capacity(),
                matrix_B.device_ref(),
                matrix_B.capacity(),
                matrix_C_computed.device_ref(),
                matrix_C_computed.capacity(),
                matrix_C_computed.device_ref(),
                matrix_C_computed.capacity()
            );
        }

        if (iter == 0)
        {
            result = cudaGetLastError();
            EXPECT_EQ(result, cudaSuccess) << " kernel error: " << cudaGetErrorString(result);        
        }
    }

    if (perf_test)
    {
        result = cudaEventRecord(end_event);
        EXPECT_EQ(result, cudaSuccess);
    }

    result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess) << " kernel error: " << cudaGetErrorString(result);

    if (perf_test)
    {
        float ms;
        result = cudaEventElapsedTime(&ms, start_event, end_event);
        EXPECT_EQ(result, cudaSuccess);
        
        double flops = (double(problem_size.m()) *
                        double(problem_size.n()) *
                        double(problem_size.k()) *
                        double(problem_size.batch()) * 2); // 2 for MAC
    
        double read_bytes = double(problem_size.batch()) * (sizeof(ElementA)*double(problem_size.m())*double(problem_size.k()) + 
                                                            sizeof(ElementB)*double(problem_size.k())*double(problem_size.n()));

        double write_bytes = double(problem_size.batch()) * (sizeof(ElementCD)*double(problem_size.m())*double(problem_size.n()));

        double avg_runtime = double(ms) / perf_test_iter;
        double gflops_per_sec = flops / 1.0e6 / avg_runtime;
        double read_bandwidth = read_bytes / 1.0e6 / avg_runtime;
        double write_bandwidth = write_bytes / 1.0e6 / avg_runtime;

        std::cout << "\n\nProblem size: "
                  << problem_size.m() 
                  << " x " << problem_size.n()
                  << " x " << problem_size.k()
                  << " x " << problem_size.batch() 
                  << std::endl;

        std::cout << "  GFLOPs:     " << gflops_per_sec << std::endl;
        std::cout << "BW (R/W):     " << read_bandwidth << " / " << write_bandwidth << " GB/sec" << std::endl;
        std::cout << " Runtime:     " << avg_runtime << " ms" << std::endl;
    }
    else
    {
        matrix_C_computed.sync_host();

        // Compute the batched gemms
        for (int b = 0; b < problem_size.batch(); b++)
        {
          cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                         ElementCD, LayoutCD, ElementCD,
                                         ElementCD>
              reference_gemm;

          reference_gemm(
              problem_size.mnk(), alpha,
              matrix_A.host_ref(b * matrix_A.capacity()),
              matrix_B.host_ref(b * matrix_B.capacity()), beta,
              matrix_C_reference.host_ref(b * matrix_C_computed.capacity()));

          bool passed = cutlass::reference::host::TensorEquals(
              matrix_C_computed.host_view(b * matrix_C_computed.capacity()),
              matrix_C_reference.host_view(b * matrix_C_reference.capacity()));

          EXPECT_TRUE(passed)
              //<< "A:\n" << matrix_A.host_view() << "\n"
              //<< "B:\n" << matrix_B.host_view() << "\n"
              << "Batch: " << b << "\n"
              << "Reference:\n"
              << matrix_C_reference.host_view(b * matrix_C_reference.capacity())
              << "\n"
              << "Computed:\n"
              << matrix_C_computed.host_view(b * matrix_C_computed.capacity())
              << "\n";
        }
    }
}

template<typename ThreadBlockShape_,
        typename ThreadShape_,
        typename ElementAB_,
        typename ElementAccumulator_,
        typename ElementCD_,
        typename LayoutA_,
        typename LayoutB_,
        typename LayoutCD_,
        int THREAD_B = 1, // batch tile size
        bool DEBUG=false>
void batched_gemv_kernel_perf_test(cutlass::gemm::BatchedGemmCoord problem_size,
                                   ElementCD_ alpha = ElementCD_(1),
                                   ElementCD_ beta = ElementCD_(0),
                                   int iter = 50)
{
    batched_gemv_kernel_test<ThreadBlockShape_,
                             ThreadShape_,
                             ElementAB_,
                             ElementAccumulator_,
                             ElementCD_,
                             LayoutA_,
                             LayoutB_,
                             LayoutCD_,
                             THREAD_B,
                             DEBUG>(problem_size, alpha, beta, true, iter);
}
    
} // namespace threadblock
} // namespace kernel
} // namespace test
