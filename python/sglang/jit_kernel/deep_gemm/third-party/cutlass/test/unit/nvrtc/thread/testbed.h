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
    \brief Unit tests for thread-level GEMM
*/

#pragma once

#include <iostream>
#include <cstdio>
#include <vector>

#include "cutlass/gemm/thread/mma.h"
#include "../kernel/thread/testbed_kernel.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/trace.h"

#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include <cuda.h>
#include <nvrtc.h>
#include "../cutlass/nvrtc/environment.h"
#include <assert.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace nvrtc {
namespace thread {

#define NVRTC_RETURN_IF_ERROR(api)                    \
  do {                                                \
    nvrtcResult _result = api;                        \
    if (_result != NVRTC_SUCCESS) {                   \
      CUTLASS_TRACE_HOST("Nvrtc error: " << _result); \
      return false;                                   \
    }                                                 \
  } while(0)

inline const char * cuda_source_fmt = R"""(

#include "kernel/thread/contraction.hpp"

using Operator = %s;

extern "C" __global__ void global_entry(__grid_constant__ Operator::Params const params) {
  extern __shared__ char smem[];

  Operator op;
  op(params, smem);
}

)""";

struct TestbedKernel {
  static bool compile(std::string const &kernel, std::vector<const char *> const &opts) {
    int sz = std::snprintf(nullptr, 0, cuda_source_fmt, kernel.c_str());
    std::vector<char> cuda_source(sz + 1);
    std::snprintf(&cuda_source[0], cuda_source.size(), cuda_source_fmt, kernel.c_str());

    nvrtcProgram program;
    NVRTC_RETURN_IF_ERROR(
        nvrtcCreateProgram(
            &program,
            cuda_source.data(),
            nullptr,
            static_cast<int32_t>(cutlass::nvrtc::kCutlassHeaderCount),
            cutlass::nvrtc::kCutlassHeaders,
            cutlass::nvrtc::kCutlassHeaderNames)
    );

    nvrtcResult compile_result = 
        nvrtcCompileProgram(
            program, 
            static_cast<int32_t>(opts.size()), 
            opts.data());

    size_t log_size;
    NVRTC_RETURN_IF_ERROR(
        nvrtcGetProgramLogSize(program, &log_size)
    );

    if (log_size > 1) {
      auto log = std::make_unique<char[]>(log_size);

      NVRTC_RETURN_IF_ERROR(
          nvrtcGetProgramLog(program, log.get())
      );
                
      std::cout << log.get() << std::endl;
    }

    NVRTC_RETURN_IF_ERROR(compile_result);

    NVRTC_RETURN_IF_ERROR(
        nvrtcDestroyProgram(&program)
    );

    return true;
  }
};

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC
>
struct Testbed {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = cutlass::gemm::thread::Mma<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC
  >;

  //
  // Data members
  //

  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutB> tensor_B;
  cutlass::HostTensor<ElementC, LayoutC> tensor_C;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_reference;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  Testbed() {

    tensor_A.reset(cutlass::make_Coord(Shape::kM, Shape::kK));
    tensor_B.reset(cutlass::make_Coord(Shape::kK, Shape::kN));
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_reference.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
  }

  static inline bool check_nvrtc_error(nvrtcResult error) {
    if (error != NVRTC_SUCCESS) {
      std::cerr << "failed to compile ";
      return false;
    }
    return true;
  }

  /// Runs the test
  bool run(std::string const &gemm_traits) {

    //
    // initialize device memory
    //

    cutlass::reference::host::BlockFillSequential(
      tensor_A.host_data(),
      tensor_A.capacity()
    );

    cutlass::reference::host::BlockFillSequential(
      tensor_B.host_data(),
      tensor_B.capacity(),
      ElementB(1),
      ElementB(2)
    );

    cutlass::reference::host::TensorFill(
      tensor_C.host_view(),
      ElementC(0)
    );

    cutlass::reference::host::TensorFill(
      tensor_D_computed.host_view(),
      ElementC(0)
    );

    cutlass::reference::host::TensorFill(
      tensor_D_reference.host_view(),
      ElementC(0)
    );

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D_computed.sync_device();

#if 0
    // launch kernel
    cutlass::gemm::kernel::testbed_kernel<Mma><<< dim3(1, 1), dim3(1, 1, 1) >>>(
      tensor_D_computed.device_data(),
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data());

#else
    // Instantiate gemm_kernel
    nvrtcResult result_nvrtc;
    nvrtcProgram program;
    static char const *src =
        "#include \"cutlass/gemm/thread/mma.h\"\n"
        "#include \"cutlass/gemm/gemm.h\"\n"
        "#include \"cutlass/layout/matrix.h\"\n"
        "#include \"unit/nvrtc/kernel/thread/testbed_kernel.h\"\n"
    ;

    std::string type_name;
#if 0
    // TODO Ideally we'd use nvrtcGetTypeName to determine the type, but it cannot resolve enum symbol names
    // As altername solution we might want to implement to_string<GemmTraits>() to get the traits string.
    nvrtcGetTypeName<typename GemmTraits_>(&type_name);
#else
    type_name = gemm_traits;
#endif

    result_nvrtc = nvrtcCreateProgram(&program,
                                    src,
                                    NULL,
                                    (int)cutlass::nvrtc::kCutlassHeaderCount,
                                    cutlass::nvrtc::kCutlassHeaders,
                                    cutlass::nvrtc::kCutlassHeaderNames);
    check_nvrtc_error(result_nvrtc);

    std::string gemm_kernel_instantiation =
      "test::nvrtc::kernel::thread::testbed_kernel< " + type_name + " >";
    nvrtcAddNameExpression(program, gemm_kernel_instantiation.c_str());

    const char *opts[] = {"--gpu-architecture=compute_75",
                          "--std=c++17",
                          "--include-path=/usr/local/cuda-10.1/include"};

    result_nvrtc = nvrtcCompileProgram(program, 3, opts);
    if (result_nvrtc != NVRTC_SUCCESS) {
      size_t logSize;
      nvrtcGetProgramLogSize(program, &logSize);
      std::vector<char> log(logSize);
      nvrtcGetProgramLog(program, log.data());
      std::cout << "Compile log:" << std::endl << log.data() << std::endl;
    }
    if (!check_nvrtc_error(result_nvrtc)) {
      assert(0);
    }

    // The lowered name is the name of the template instantiation in the generated PTX code.
    char const *gemm_kernel_lowered_name;
    nvrtcGetLoweredName(program, gemm_kernel_instantiation.c_str(), &gemm_kernel_lowered_name);
    if (!check_nvrtc_error(result_nvrtc)) {
      assert(0);
    }

    // Query the size of the genereated PTX so that we can allocate storage and retrieve it afterwards
    size_t ptx_size;
    result_nvrtc = nvrtcGetPTXSize(program, &ptx_size);
    if (!check_nvrtc_error(result_nvrtc)) {
      assert(0);
    }

    std::vector<char> ptx(ptx_size);
    result_nvrtc = nvrtcGetPTX(program, ptx.data());
    if (!check_nvrtc_error(result_nvrtc)) {
      assert(0);
    }

    // we do not need the nvrtc program anymore
    //nvrtcDestroyProgram(&program);

    CUmodule module;
    CUresult result_cuda;
    result_cuda = cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0);
    if (result_cuda != CUDA_SUCCESS) {
      assert(0);
    }

    CUfunction kernel;
    result_cuda = cuModuleGetFunction(&kernel, module, gemm_kernel_lowered_name);
    if (result_cuda != CUDA_SUCCESS) {
      assert(0);
    }

    void* d_a = (void*)tensor_A.device_data();
    void* d_b = (void*)tensor_B.device_data();
    void* d_c = (void*)tensor_C.device_data();
    void* d_d = (void*)tensor_D_computed.device_data();
    void* args[] = { &d_d, &d_a, &d_b, &d_c };

    // CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra
    result_cuda = cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0 /*cudaStreamDefault*/, args, 0);
    if (result_cuda != CUDA_SUCCESS) {
      assert(0);
    } else {
}
#endif

    // verify no errors
    cudaError_t result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cout << "CUDA ERROR: " << cudaGetErrorString(result);
      return false;
    }

    tensor_D_computed.sync_host();

    //
    // Reference implementation
    //

    //tensor_D_reference.fill(tensor_C.host_view());

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC> reference_gemm;

    reference_gemm(
      {Shape::kM, Shape::kN, Shape::kK},
      ElementC(1),
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      ElementC(0),
      tensor_D_reference.host_ref()
    );

    //
    // Verify equivalence
    //

    // compare
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_D_computed.host_view(),
      tensor_D_reference.host_view()
    );

    if(!passed) std::cout
      << "A:\n" << tensor_A.host_view() << "\n\n"
      << "B:\n" << tensor_B.host_view() << "\n\n"
      << "C:\n" << tensor_C.host_view() << "\n\n"
      << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
      << "Computed:\n" << tensor_D_computed.host_view() << std::endl;
    
    std::cout << "passed " << passed << std::endl;
    
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace nvrtc
} // namespace test
