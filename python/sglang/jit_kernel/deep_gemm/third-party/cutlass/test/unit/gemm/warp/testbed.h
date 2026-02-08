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

#include "cutlass/cutlass.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_types.h"
#include "cutlass/subbyte_reference.h"
#include "cutlass/platform/platform.h"
#include "cutlass/arch/arch.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"

namespace test {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test kernel
template <typename Mma, typename ThreadblockShape>
__global__ void kernel(
  typename Mma::ElementC *output_C, 
  typename Mma::ElementA const *input_A,
  typename Mma::ElementB const *input_B,
  typename Mma::ElementC const *input_C,
  int iterations = 1) {

  // Use AlignedBuffer to store trivially copyable objects in unions and __shared__ buffers.
  __shared__ cutlass::AlignedBuffer<
    typename Mma::ElementA, ThreadblockShape::kM * ThreadblockShape::kK> smem_buffer_A;

  __shared__ cutlass::AlignedBuffer<
    typename Mma::ElementB, ThreadblockShape::kN * ThreadblockShape::kK> smem_buffer_B;

  if (threadIdx.x == 0) {
    typename Mma::ElementA *smem_ptr_A = smem_buffer_A.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_A.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementA>::get(smem_ptr_A, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementA>::type>::get(input_A, i);
    }

    typename Mma::ElementB *smem_ptr_B = smem_buffer_B.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_B.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementB>::get(smem_ptr_B, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementB>::type>::get(input_B, i);
    }
  }

  __syncthreads();

  //
  // Construct warp-level matrix product
  //

  using FragmentA = typename Mma::FragmentA;
  using FragmentB = typename Mma::FragmentB;
  using FragmentC = typename Mma::FragmentC;

  typename Mma::LayoutA layout_A = Mma::LayoutA::packed({ThreadblockShape::kM, ThreadblockShape::kK});
  typename Mma::LayoutB layout_B = Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
  typename Mma::LayoutC layout_C = Mma::LayoutC::packed({Mma::Shape::kM, Mma::Shape::kN});

  typename Mma::IteratorA iter_A({smem_buffer_A.data(), layout_A}, cutlass::arch::LaneId());

  typename Mma::IteratorB iter_B({smem_buffer_B.data(), layout_B}, cutlass::arch::LaneId());

  FragmentA frag_A;
  FragmentB frag_B;

  FragmentC accum;

  Mma mma;

  accum.clear();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < iterations; ++iter) {     // place in loop that is not unrolled 

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK;
         k += Mma::Policy::MmaShape::kK) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);

      ++iter_A;
      ++iter_B;

      mma(accum, frag_A, frag_B, accum);
    }
  }
  
  typename Mma::IteratorC iter_C({output_C, layout_C}, cutlass::arch::LaneId());

  iter_C.store(accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Warp-level matrix multiply-accumulate
  typename Mma_,
  /// Size of threadblock-scoped shape used to store SMEM
  typename ThreadblockShape_,
  /// The inner product operation performed by GEMM 
  typename Operator_ = cutlass::arch::OpMultiplyAdd
>
struct Testbed {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = Mma_;
  using ThreadblockShape = ThreadblockShape_;
  using Operator = Operator_;

  using Shape = typename Mma::Shape;
  using ElementA = typename Mma::ElementA;
  using LayoutA = typename Mma::LayoutA;
  using ElementB = typename Mma::ElementB;
  using LayoutB = typename Mma::LayoutB;
  using ElementC = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;

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

    tensor_A.reset(cutlass::make_Coord(ThreadblockShape::kM, ThreadblockShape::kK));
    tensor_B.reset(cutlass::make_Coord(ThreadblockShape::kK, ThreadblockShape::kN));
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_reference.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.major == 9) {
      // NVIDIA Hopper drops support for several data types
      if (
        cutlass::sizeof_bits<ElementA>::value < 8 ||
        cutlass::sizeof_bits<ElementB>::value < 8 ||
        cutlass::sizeof_bits<ElementC>::value < 8) {

        return false;
      }
    }

    return true;
  }


  /// Runs the test
  bool run(
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform) {

    if (!sufficient()) {
      return true;
    }

    //
    // initialize device memory
    //

    if (init_A == cutlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementA>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementA>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;

      cutlass::reference::host::BlockFillRandomUniform(tensor_A.host_data(),
        tensor_A.capacity(), seed, scope_max, scope_min, 0);

    } else if (init_A == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_A.host_data(),
                                                    tensor_A.capacity());
    } else if (init_A == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_A.host_view());
    } else {
      return false;
    }

    if (init_B == cutlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementB>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementB>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;

      cutlass::reference::host::BlockFillRandomUniform(tensor_B.host_data(),
        tensor_B.capacity(), seed, scope_max, scope_min, 0);

    } else if (init_B == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_B.host_data(),
                                                    tensor_B.capacity());
    } else if (init_B == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_B.host_view());
    } else {
      return false;
    }

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

    // launch kernel
    kernel<Mma, ThreadblockShape><<< dim3(1, 1), dim3(32, 1, 1) >>>(
      tensor_D_computed.device_data(),
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data());

    // verify no errors
    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess) << "CUDA ERROR: " << cudaGetErrorString(result);
    if (result != cudaSuccess) {
      return false;
    }

    tensor_D_computed.sync_host();

    //
    // Reference implementation
    //

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC,
                                   Operator>
        reference_gemm;

    reference_gemm(
      {Shape::kM, Shape::kN, ThreadblockShape::kK},
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

    EXPECT_TRUE(passed);

    if (!passed) {

      cutlass::TensorView<ElementA, cutlass::layout::ColumnMajor> tensor_A_physical(
        tensor_A.host_data(), 
        tensor_A.stride()[0], 
        tensor_A.extent());

      cutlass::TensorView<ElementB, cutlass::layout::RowMajor> tensor_B_physical(
        tensor_B.host_data(), 
        tensor_B.stride()[0], 
        tensor_B.extent());

      std::cout <<"cutlass::sizeof_bits<ElementA>::value = "<<cutlass::sizeof_bits<ElementA>::value<<"\n";
      std::cout
        << "A:\n" << tensor_A.host_view() << "\n\n"
        << "A(physical - stride: " << tensor_A.stride()[0] 
        << ", extent: " << tensor_A.extent() << "):\n" << tensor_A_physical << "\n\n";

      std::cout <<"cutlass::sizeof_bits<ElementB>::value = "<<cutlass::sizeof_bits<ElementB>::value<<"\n";
      std::cout
        << "B:\n" << tensor_B.host_view() << "\n\n"
        << "B(physical - stride: " << tensor_B.stride()[0] 
        << ", extent: " << tensor_B.extent() << "):\n" << tensor_B_physical << "\n\n";

      std::cout
        << "C:\n" << tensor_C.host_view() << "\n\n"
        << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
        << "Computed:\n" << tensor_D_computed.host_view() << std::endl;
    }
    
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Warp-level matrix multiply-accumulate
  typename Mma_,
  /// Size of threadblock-scoped shape used to store SMEM
  typename ThreadblockShape_
>
struct TestbedComplex {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = Mma_;
  using ThreadblockShape = ThreadblockShape_;

  using Shape = typename Mma::Shape;
  using ElementA = typename Mma::ElementA;
  using LayoutA = typename Mma::LayoutA;
  using ElementB = typename Mma::ElementB;
  using LayoutB = typename Mma::LayoutB;
  using ElementC = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;

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
  TestbedComplex() {

    tensor_A.reset(cutlass::make_Coord(ThreadblockShape::kM, ThreadblockShape::kK));
    tensor_B.reset(cutlass::make_Coord(ThreadblockShape::kK, ThreadblockShape::kN));
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_reference.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.major == 9) {
      // NVIDIA Hopper drops support for several data types
      if (
        cutlass::sizeof_bits<ElementA>::value < 8 ||
        cutlass::sizeof_bits<ElementB>::value < 8 ||
        cutlass::sizeof_bits<ElementC>::value < 8) {

        return false;
      }
    }

    return true;
  }

  /// Runs the test
  bool run(
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform) {

    if (!sufficient()) {
      return true;
    }

    //
    // initialize device memory
    //

    if (init_A == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(tensor_A.host_view(),
                                                        seed, 8, -8, 0);
    } else if (init_A == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_A.host_data(),
                                                    tensor_A.capacity());
    } else if (init_A == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_A.host_view());
    } else {
      return false;
    }

    if (init_B == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(tensor_B.host_view(),
                                                        seed + 16, 8, -8, 0);
    } else if (init_B == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_B.host_data(),
                                                    tensor_B.capacity());
    } else if (init_B == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_B.host_view());
    } else {
      return false;
    }

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

    // launch kernel
    kernel<Mma, ThreadblockShape><<< dim3(1, 1), dim3(32, 1, 1) >>>(
      tensor_D_computed.device_data(),
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data());

    // verify no errors
    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess) << "CUDA ERROR: " << cudaGetErrorString(result);
    if (result != cudaSuccess) {
      return false;
    }

    tensor_D_computed.sync_host();

    //
    // Reference implementation
    //

    cutlass::reference::host::GemmComplex(
      {Shape::kM, Shape::kN, ThreadblockShape::kK},
      ElementC(1),
      tensor_A.host_ref(),
      Mma::kTransformA,
      tensor_B.host_ref(),
      Mma::kTransformB,
      ElementC(0),
      tensor_C.host_ref(),
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

    EXPECT_TRUE(passed);

    if (!passed) {

      cutlass::TensorView<ElementA, cutlass::layout::ColumnMajor> tensor_A_physical(
        tensor_A.host_data(), 
        tensor_A.stride()[0], 
        tensor_A.extent());

      cutlass::TensorView<ElementB, cutlass::layout::RowMajor> tensor_B_physical(
        tensor_B.host_data(), 
        tensor_B.stride()[0], 
        tensor_B.extent());

      std::cout <<"cutlass::sizeof_bits<ElementA>::value = "<<cutlass::sizeof_bits<ElementA>::value<<"\n";
      std::cout 
        << "A:\n" << tensor_A.host_view() << "\n\n"
        << "A(physical - stride: " << tensor_A.stride()[0] << ", extent: " << tensor_A.extent() << "):\n" << tensor_A_physical << "\n\n";

      std::cout <<"cutlass::sizeof_bits<ElementB>::value = "<<cutlass::sizeof_bits<ElementB>::value<<"\n";
      std::cout
        << "B:\n" << tensor_B.host_view() << "\n\n"
        << "B(physical - stride: " << tensor_B.stride()[0] << ", extent: " << tensor_B.extent() <<"):\n" << tensor_B_physical << "\n\n";

      std::cout
        << "C:\n" << tensor_C.host_view() << "\n\n"
        << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
        << "Computed:\n" << tensor_D_computed.host_view() << std::endl;
    }
    
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test kernel
template <typename Mma, typename ThreadblockShape>
__global__ void kernel_transform(
  typename Mma::ElementC *output_C, 
  typename Mma::ElementA const *input_A,
  typename Mma::ElementB const *input_B,
  typename Mma::ElementC const *input_C,
  int iterations = 1) {

  // Use AlignedBuffer to store trivially copyable objects in unions and __shared__ buffers.
  __shared__ cutlass::AlignedBuffer<
    typename Mma::ElementA, ThreadblockShape::kM * ThreadblockShape::kK> smem_buffer_A;

  __shared__ cutlass::AlignedBuffer<
    typename Mma::ElementB, ThreadblockShape::kN * ThreadblockShape::kK> smem_buffer_B;

  if (threadIdx.x == 0) {
    typename Mma::ElementA *smem_ptr_A = smem_buffer_A.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_A.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementA>::get(smem_ptr_A, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementA>::type>::get(input_A, i);
    }

    typename Mma::ElementB *smem_ptr_B = smem_buffer_B.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_B.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementB>::get(smem_ptr_B, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementB>::type>::get(input_B, i);
    }
  }

  __syncthreads();

  //
  // Construct warp-level matrix product
  //

  using FragmentA = typename Mma::FragmentA;
  using FragmentB = typename Mma::FragmentB;
  using FragmentC = typename Mma::FragmentC;

  using TransformedFragmentA = typename Mma::TransformedFragmentA;
  using TransformedFragmentB = typename Mma::TransformedFragmentB;

  typename Mma::LayoutA layout_A = Mma::LayoutA::packed({ThreadblockShape::kM, ThreadblockShape::kK});
  typename Mma::LayoutB layout_B = Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
  typename Mma::LayoutC layout_C = Mma::LayoutC::packed({Mma::Shape::kM, Mma::Shape::kN});

  typename Mma::IteratorA iter_A({smem_buffer_A.data(), layout_A}, cutlass::arch::LaneId());

  typename Mma::IteratorB iter_B({smem_buffer_B.data(), layout_B}, cutlass::arch::LaneId());

  FragmentA loaded_frag_A;
  FragmentB loaded_frag_B;
  TransformedFragmentA transformed_frag_A;
  TransformedFragmentB transformed_frag_B;

  FragmentC accum;

  Mma mma;

  accum.clear();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < iterations; ++iter) {     // place in loop that is not unrolled 

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK;
         k += Mma::Policy::MmaShape::kK) {
      iter_A.load(loaded_frag_A);
      iter_B.load(loaded_frag_B);

      ++iter_A;
      ++iter_B;

      mma.transform(transformed_frag_A, transformed_frag_B, loaded_frag_A,
                    loaded_frag_B);

      mma(accum, transformed_frag_A, transformed_frag_B, accum);
    }
  }
  
  typename Mma::IteratorC iter_C({output_C, layout_C}, cutlass::arch::LaneId());

  iter_C.store(accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Warp-level matrix multiply-accumulate
  typename Mma_,
  /// Size of threadblock-scoped shape used to store SMEM
  typename ThreadblockShape_,
  /// The innter product operation performed by GEMM 
  typename Operator_ = cutlass::arch::OpMultiplyAdd
>
struct TransformTestbed {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = Mma_;
  using ThreadblockShape = ThreadblockShape_;
  using Operator = Operator_;

  using Shape = typename Mma::Shape;
  using ElementA = typename Mma::ElementA;
  using LayoutA = typename Mma::LayoutA;
  using ElementB = typename Mma::ElementB;
  using LayoutB = typename Mma::LayoutB;
  using ElementC = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;

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
  TransformTestbed() {

    tensor_A.reset(cutlass::make_Coord(ThreadblockShape::kM, ThreadblockShape::kK));
    tensor_B.reset(cutlass::make_Coord(ThreadblockShape::kK, ThreadblockShape::kN));
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_reference.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.major == 9) {
      // NVIDIA Hopper drops support for several data types
      if (
        cutlass::sizeof_bits<ElementA>::value < 8 ||
        cutlass::sizeof_bits<ElementB>::value < 8 ||
        cutlass::sizeof_bits<ElementC>::value < 8) {

        return false;
      }
    }

    return true;
  }

  /// Runs the test
  bool run(
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform) {

    if (!sufficient()) {
      return true;
    }

    //
    // initialize device memory
    //

    if (init_A == cutlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementA>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementA>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(
          tensor_A.host_view(), seed, scope_max, scope_min, 0);
    } else if (init_A == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_A.host_data(),
                                                    tensor_A.capacity());
    } else if (init_A == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_A.host_view());
    } else {
      return false;
    }

    if (init_B == cutlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementB>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementB>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(
          tensor_B.host_view(), seed + 16, scope_max, scope_min, 0);
    } else if (init_B == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_B.host_data(),
                                                    tensor_B.capacity());
    } else if (init_B == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_B.host_view());
    } else {
      return false;
    }

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

    // launch kernel
    kernel_transform<Mma, ThreadblockShape><<<dim3(1, 1), dim3(32, 1, 1)>>>(
        tensor_D_computed.device_data(), tensor_A.device_data(),
        tensor_B.device_data(), tensor_C.device_data());

    // verify no errors
    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess) << "CUDA ERROR: " << cudaGetErrorString(result);
    if (result != cudaSuccess) {
      return false;
    }

    tensor_D_computed.sync_host();

    //
    // Reference implementation
    //

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC,
                                   Operator>
        reference_gemm;

    reference_gemm(
      {Shape::kM, Shape::kN, ThreadblockShape::kK},
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

    EXPECT_TRUE(passed);

    if (!passed) {

      cutlass::TensorView<ElementA, cutlass::layout::ColumnMajor> tensor_A_physical(
        tensor_A.host_data(), 
        tensor_A.stride()[0], 
        tensor_A.extent());

      cutlass::TensorView<ElementB, cutlass::layout::RowMajor> tensor_B_physical(
        tensor_B.host_data(), 
        tensor_B.stride()[0], 
        tensor_B.extent());

      std::cout <<"cutlass::sizeof_bits<ElementA>::value = "<<cutlass::sizeof_bits<ElementA>::value<<"\n";
      std::cout
        << "A:\n" << tensor_A.host_view() << "\n\n"
        << "A(physical - stride: " << tensor_A.stride()[0] << ", extent: " << tensor_A.extent() << "):\n" << tensor_A_physical << "\n\n";

      std::cout <<"cutlass::sizeof_bits<ElementB>::value = "<<cutlass::sizeof_bits<ElementB>::value<<"\n";
      std::cout
        << "B:\n" << tensor_B.host_view() << "\n\n"
        << "B(physical - stride: " << tensor_B.stride()[0] << ", extent: " << tensor_B.extent() << "):\n" << tensor_B_physical << "\n\n";

      std::cout
        << "C:\n" << tensor_C.host_view() << "\n\n"
        << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
        << "Computed:\n" << tensor_D_computed.host_view() << std::endl;
    }
    
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Warp-level matrix multiply-accumulate
  typename Mma_,
  /// Size of threadblock-scoped shape used to store SMEM
  typename ThreadblockShape_
>
struct TransformedTestbedComplex {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = Mma_;
  using ThreadblockShape = ThreadblockShape_;

  using Shape = typename Mma::Shape;
  using ElementA = typename Mma::ElementA;
  using LayoutA = typename Mma::LayoutA;
  using ElementB = typename Mma::ElementB;
  using LayoutB = typename Mma::LayoutB;
  using ElementC = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;

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
  TransformedTestbedComplex() {

    tensor_A.reset(cutlass::make_Coord(ThreadblockShape::kM, ThreadblockShape::kK));
    tensor_B.reset(cutlass::make_Coord(ThreadblockShape::kK, ThreadblockShape::kN));
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_reference.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.major == 9) {
      // NVIDIA Hopper drops support for several data types
      if (
        cutlass::sizeof_bits<ElementA>::value < 8 ||
        cutlass::sizeof_bits<ElementB>::value < 8 ||
        cutlass::sizeof_bits<ElementC>::value < 8) {

        return false;
      }
    }

    return true;
  }

  /// Runs the test
  bool run(
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform) {

    if (!sufficient()) {
      return true;
    }

    //
    // initialize device memory
    //

    if (init_A == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(tensor_A.host_view(),
                                                        seed, 8, -8, 0);
    } else if (init_A == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_A.host_data(),
                                                    tensor_A.capacity());
    } else if (init_A == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_A.host_view());
    } else {
      return false;
    }

    if (init_B == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(tensor_B.host_view(),
                                                        seed + 16, 8, -8, 0);
    } else if (init_B == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_B.host_data(),
                                                    tensor_B.capacity());
    } else if (init_B == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_B.host_view());
    } else {
      return false;
    }

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

    // launch kernel
    kernel_transform<Mma, ThreadblockShape><<< dim3(1, 1), dim3(32, 1, 1) >>>(
      tensor_D_computed.device_data(),
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data());

    // verify no errors
    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess) << "CUDA ERROR: " << cudaGetErrorString(result);
    if (result != cudaSuccess) {
      return false;
    }

    tensor_D_computed.sync_host();

    //
    // Reference implementation
    //

    cutlass::reference::host::GemmComplex(
      {Shape::kM, Shape::kN, ThreadblockShape::kK},
      ElementC(1),
      tensor_A.host_ref(),
      Mma::kTransformA,
      tensor_B.host_ref(),
      Mma::kTransformB,
      ElementC(0),
      tensor_C.host_ref(),
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

    EXPECT_TRUE(passed);

    if (!passed) {

      cutlass::TensorView<ElementA, cutlass::layout::ColumnMajor> tensor_A_physical(
        tensor_A.host_data(), 
        tensor_A.stride()[0], 
        tensor_A.extent());

      cutlass::TensorView<ElementB, cutlass::layout::RowMajor> tensor_B_physical(
        tensor_B.host_data(), 
        tensor_B.stride()[0], 
        tensor_B.extent());

      std::cout <<"cutlass::sizeof_bits<ElementA>::value = "<<cutlass::sizeof_bits<ElementA>::value<<"\n";
      std::cout 
        << "A:\n" << tensor_A.host_view() << "\n\n"
        << "A(physical - stride: " << tensor_A.stride()[0] << ", extent: " << tensor_A.extent() << "):\n" << tensor_A_physical << "\n\n";

      std::cout <<"cutlass::sizeof_bits<ElementB>::value = "<<cutlass::sizeof_bits<ElementB>::value<<"\n";
      std::cout
        << "B:\n" << tensor_B.host_view() << "\n\n"
        << "B(physical - stride: " << tensor_B.stride()[0] << ", extent: " << tensor_B.extent() <<"):\n" << tensor_B_physical << "\n\n";

      std::cout
        << "C:\n" << tensor_C.host_view() << "\n\n"
        << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
        << "Computed:\n" << tensor_D_computed.host_view() << std::endl;
    }
    
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test kernel
template <typename Mma, typename ThreadblockShape>
__global__ void sparse_kernel(
  typename Mma::ElementC *output_C, 
  typename Mma::ElementA const *input_A,
  typename Mma::ElementB const *input_B,
  typename Mma::ElementC const *input_C,
  typename Mma::ElementE const *input_E,
  int iterations = 1) {

  // Use AlignedBuffer to store trivially copyable objects in unions and __shared__ buffers.
  __shared__ cutlass::AlignedBuffer<typename Mma::ElementA,
                                    ThreadblockShape::kM *
                                        ThreadblockShape::kK / Mma::kSparse>
      smem_buffer_A;

  __shared__ cutlass::AlignedBuffer<
    typename Mma::ElementB, ThreadblockShape::kN * ThreadblockShape::kK> smem_buffer_B;

  __shared__ cutlass::AlignedBuffer<
      typename Mma::ElementE, Mma::Shape::kM * Mma::Shape::kK /
                                  Mma::kSparse / Mma::kElementsPerElementE>
      smem_buffer_E;
  
  __syncthreads();

  if (threadIdx.x == 0) {
    typename Mma::ElementA *smem_ptr_A = smem_buffer_A.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_A.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementA>::get(smem_ptr_A, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementA>::type>::get(input_A, i);
    }

    typename Mma::ElementB *smem_ptr_B = smem_buffer_B.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_B.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementB>::get(smem_ptr_B, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementB>::type>::get(input_B, i);
    }

    typename Mma::ElementE *smem_ptr_E = smem_buffer_E.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_E.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementE>::get(smem_ptr_E, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementE>::type>::get(input_E, i);
    }
  }

  __syncthreads();

  //
  // Construct warp-level matrix product
  //

  using FragmentA = typename Mma::FragmentA;
  using FragmentB = typename Mma::FragmentB;
  using FragmentC = typename Mma::FragmentC;
  using FragmentE = typename Mma::FragmentE;

  typename Mma::LayoutA layout_A = Mma::LayoutA::packed(
      {ThreadblockShape::kM, ThreadblockShape::kK / Mma::kSparse});
  typename Mma::LayoutB layout_B =
      Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
  typename Mma::LayoutC layout_C = Mma::LayoutC::packed({Mma::Shape::kM, Mma::Shape::kN});
  typename Mma::LayoutE layout_E =
      Mma::LayoutE::packed({Mma::Shape::kM * Mma::kInterleaved,
                            Mma::Shape::kK / Mma::kSparse /
                                Mma::kElementsPerElementE / Mma::kInterleaved});

  typename Mma::IteratorA iter_A({smem_buffer_A.data(), layout_A}, cutlass::arch::LaneId());

  typename Mma::IteratorB iter_B({smem_buffer_B.data(), layout_B}, cutlass::arch::LaneId());

  typename Mma::IteratorE iter_E({smem_buffer_E.data(), layout_E}, cutlass::arch::LaneId());

  FragmentA frag_A;
  FragmentB frag_B;

  FragmentC accum;

  FragmentE frag_E;

  Mma mma;

  accum.clear();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < iterations; ++iter) {     // place in loop that is not unrolled 

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK;
         k += Mma::Policy::MmaShape::kK) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);
      iter_E.load(frag_E);

      ++iter_A;
      ++iter_B;
      ++iter_E;

      mma(accum, frag_A, frag_B, accum, frag_E);
    }
  }
  
  typename Mma::IteratorC iter_C({output_C, layout_C}, cutlass::arch::LaneId());

  iter_C.store(accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Warp-level matrix multiply-accumulate
  typename Mma_,
  /// Size of threadblock-scoped shape used to store SMEM
  typename ThreadblockShape_,
  /// The innter product operation performed by GEMM 
  typename Operator_ = cutlass::arch::OpMultiplyAdd
>
struct SparseTestbed {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = Mma_;
  using ThreadblockShape = ThreadblockShape_;
  using Operator = Operator_;

  using Shape = typename Mma::Shape;
  using ElementA = typename Mma::ElementA;
  using LayoutA = typename Mma::LayoutA;
  using ElementB = typename Mma::ElementB;
  using LayoutB = typename Mma::LayoutB;
  using ElementC = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;

  static int const Sparse = Mma::kSparse;
  static int const MetaSizeInBits = Mma::kMetaSizeInBits;
  static int const MaxID2 = Mma::kMaxID2;
  static int const Interleaved = Mma::kInterleaved;

  using ElementE = typename Mma::ElementE;

  static int const ElementsPerElementE = Mma::kElementsPerElementE;

  using LayoutE = cutlass::layout::RowMajor;
  using ReorderedLayoutE =
      cutlass::layout::ColumnMajorInterleaved<Interleaved>;

  //
  // Data members
  //

  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementA, LayoutA> tensor_A_uncompressed;
  cutlass::HostTensor<ElementB, LayoutB> tensor_B;
  cutlass::HostTensor<ElementC, LayoutC> tensor_C;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_reference;
  cutlass::HostTensor<ElementE, LayoutE> tensor_E;
  cutlass::HostTensor<ElementE, ReorderedLayoutE> tensor_E_reordered;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  SparseTestbed() {

    tensor_A.reset(cutlass::make_Coord(ThreadblockShape::kM,
                                       ThreadblockShape::kK / Sparse));
    tensor_A_uncompressed.reset(
        cutlass::make_Coord(ThreadblockShape::kM, ThreadblockShape::kK));
    tensor_B.reset(cutlass::make_Coord(ThreadblockShape::kK, ThreadblockShape::kN));
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_reference.reset(cutlass::make_Coord(Shape::kM, Shape::kN), false);
    tensor_E.reset(cutlass::make_Coord(
        Shape::kM, Shape::kK / Sparse / ElementsPerElementE));
    tensor_E_reordered.reset(cutlass::make_Coord(
        Shape::kM, Shape::kK / Sparse / ElementsPerElementE));
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.major == 9) {
      // NVIDIA Hopper drops support for several data types
      if (
        cutlass::sizeof_bits<ElementA>::value < 8 ||
        cutlass::sizeof_bits<ElementB>::value < 8 ||
        cutlass::sizeof_bits<ElementC>::value < 8) {

        return false;
      }
    }

    return true;
  }

  /// Runs the test
  bool run(
      cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_E = cutlass::Distribution::Uniform) {

    if (!sufficient()) {
      return true;
    }

    //
    // initialize device memory
    //

    if (init_A == cutlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementA>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementA>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(
          tensor_A.host_view(), seed, scope_max, scope_min, 0);
    } else if (init_A == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_A.host_data(),
                                                    tensor_A.capacity());
    } else if (init_A == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_A.host_view());
    } else {
      return false;
    }

    if (init_B == cutlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (cutlass::sizeof_bits<ElementB>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (cutlass::sizeof_bits<ElementB>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomUniform(
          tensor_B.host_view(), seed + 16, scope_max, scope_min, 0);
    } else if (init_B == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(tensor_B.host_data(),
                                                    tensor_B.capacity());
    } else if (init_B == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(tensor_B.host_view());
    } else {
      return false;
    }

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

    if (init_E == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomSparseMeta(
          tensor_E.host_view(), seed, MetaSizeInBits);
    } else if (init_E == cutlass::Distribution::Identity) {
      uint32_t content = (MaxID2 == 1) ? 0x44444444 : 0x4444;
      cutlass::reference::host::TensorFill(tensor_E.host_view(),
                                           (ElementE)(content));
    } else {
      return false;
    }

    cutlass::reorder_meta(
        tensor_E_reordered.host_ref(), tensor_E.host_ref(),
        {Shape::kM, Shape::kN, Shape::kK / Sparse / ElementsPerElementE});

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D_computed.sync_device();
    tensor_E_reordered.sync_device();

    // launch kernel
    sparse_kernel<Mma, ThreadblockShape><<< dim3(1, 1), dim3(32, 1, 1) >>>(
      tensor_D_computed.device_data(),
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_E_reordered.device_data());

    // verify no errors
    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess) << "CUDA ERROR: " << cudaGetErrorString(result);
    if (result != cudaSuccess) {
      return false;
    }

    tensor_D_computed.sync_host();

    //
    // Reference implementation
    //
    cutlass::uncompress(tensor_A_uncompressed.host_ref(), tensor_A.host_ref(),
                        tensor_E.host_ref(), Shape::kM, Shape::kK);

    cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC,
                                   Operator>
        reference_gemm;

    reference_gemm(
      {Shape::kM, Shape::kN, ThreadblockShape::kK},
      ElementC(1),
      tensor_A_uncompressed.host_ref(),
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

    EXPECT_TRUE(passed);

    if (!passed) {
      std::cout <<"cutlass::sizeof_bits<ElementA>::value = "<<cutlass::sizeof_bits<ElementA>::value<<"\n";
      std::cout << "A:\n" << tensor_A.host_view() << "\n\n";

      std::cout <<"cutlass::sizeof_bits<ElementB>::value = "<<cutlass::sizeof_bits<ElementB>::value<<"\n";
      std::cout << "B:\n" << tensor_B.host_view() << "\n\n";

      std::cout <<"cutlass::sizeof_bits<ElementB>::value = "<<cutlass::sizeof_bits<ElementE>::value<<"\n";
      std::cout << "E:\n" << tensor_E.host_view() << "\n\n";

      std::cout
        << "C:\n" << tensor_C.host_view() << "\n\n"
        << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
        << "Computed:\n" << tensor_D_computed.host_view() << "\n";
    }
    
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace test
