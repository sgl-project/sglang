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
/*! \file
  \brief Example demonstrating CuTe and CUTLASS 3.x based Ampere convolution forward propagation kernel
      capable of operating on both affine and gather/scatter tensors.

  This example demonstartes a few super cool features of CUTLASS and CuTe. It shows off
  1. A dense conv 3D fprop kernel written as a single file ...
  2. ... that leverages off the shelf CUTLASS collectives to show how custom kernels can use collectives ...
  3. ... and uses the exact same templated kernel to also stamp out a gather/scatter 3D fprop conv ...
  4. ... while getting near peak performance of the Ampere class tensor core on Ampere and Ada GPUs ...
  5. ... by using static cute shapes and strides in case problem shapes are known at compile time.

  Full documentation for this example can be found within the README.md file in this directory.

  Example executions:
  ./59_ampere_gather_scatter_conv
  ./59_ampere_gather_scatter_conv --n=108
  ./59_ampere_gather_scatter_conv --n=4096 --i=1
  ./59_ampere_gather_scatter_conv --n=1080 --i=1000
  ./59_ampere_gather_scatter_conv --n=131072 --i=1000 --no-check
*/

#include <thrust/sequence.h>
#include <thrust/universal_vector.h>

#include "ampere_conv_kernel.h"
#include "gather_tensor.hpp"

#include "cutlass/util/command_line.h"

bool check_cuda_result(cudaError_t code, const char* file, int line) {
  if (code == cudaSuccess) {
    return true;
  }

  std::cerr << "CUDA error at  (" << file << "," << line << ")\n\t" << unsigned(code) << " -- " << cudaGetErrorString(code) << "\n";
  return false;
}

#define CHECK_CUDA(code) (check_cuda_result(code, __FILE__, __LINE__))

using namespace cute;
using example::IndexedGather;
using example::CustomStride;

template<class Operator, class FilterTensor, class ActivationTensor, class OutputTensor>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void kernel_entrypoint(FilterTensor mFlt, ActivationTensor mAct, OutputTensor mOut) {
  extern __shared__ char smem_buf[];
  Operator op;
  op(mFlt, mAct, mOut, smem_buf);
}

int ampere_dense_conv_fprop(
    int num_images,
    float* activations,
    float* filter,
    float* output,
    float* output_ref,
    int num_iterations = 1,
    bool do_ref_check = true) {
  auto D = typename AmpereUnpredicatedFprop::D{};
  auto H = typename AmpereUnpredicatedFprop::H{};
  auto W = typename AmpereUnpredicatedFprop::W{};
  auto Z = typename AmpereUnpredicatedFprop::Z{};
  auto P = typename AmpereUnpredicatedFprop::P{};
  auto Q = typename AmpereUnpredicatedFprop::Q{};
  auto C = typename AmpereUnpredicatedFprop::C{};
  auto K = typename AmpereUnpredicatedFprop::K{};
  auto S = typename AmpereUnpredicatedFprop::S{};
  auto R = typename AmpereUnpredicatedFprop::R{};
  auto T = typename AmpereUnpredicatedFprop::T{};

  int N = num_images; // dynamic
  if (num_images % int(typename AmpereUnpredicatedFprop::Tiler_N{}) != 0) {
    printf("ERROR: Input image count must be evenly divisible by CTA tiler N.\n");
    return 1;
  }

  // Tensor Activation: (n,d,h,w,c)::(?,6,4,4,64):(6144,1536,384,64,1)
  auto activation_layout = make_layout(
    make_shape (make_shape (      N,     D,   H, W), make_shape ( C,   _1{},_1{},_1{})),
    make_stride(make_stride(D*H*W*C, H*W*C, W*C, C), make_stride(_1{}, _0{},_0{},_0{})));

  auto xformed_act_layout = make_layout(
    make_shape (make_shape(N, Z, P, Q),       make_shape ( C,       T,   R, S)),
    make_stride(stride<0>(activation_layout), make_stride(_1{}, H*W*C, W*C, C)));

  // Tensor Filter    : (k,c,s,r,t)::(128,3,3,3,64):(1728,576,192,64,1)
  auto filter_layout = AmpereUnpredicatedFprop::GmemLayoutFlt{};

  // Tensor Output    : (n,z,p,q,k)::(?,4,2,2,128):(2048,1024,512,128,1)
  auto output_layout = make_ordered_layout(
    make_shape( K,   make_shape( N,   Z,   P,   Q)),
    make_tuple(_0{}, make_tuple(_4{},_3{},_2{},_1{})));

  Tensor mActivation = make_tensor(make_gmem_ptr(activations), activation_layout);
  Tensor mXformedAct = make_tensor(make_gmem_ptr(activations), xformed_act_layout);
  Tensor mFilter     = make_tensor(make_gmem_ptr(filter), filter_layout);
  Tensor mOutput     = make_tensor(make_gmem_ptr(output), output_layout); // (K, (N,Z,P,Q))
  Tensor mOutputRef  = make_tensor(make_gmem_ptr(output_ref), output_layout);

  print("xformed act layout ((N,Z,P,Q), (C,T,R,S)) = "); print(xformed_act_layout); print("\n");

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  constexpr size_t smem_size = sizeof(typename AmpereUnpredicatedFprop::SharedStorage);
  Tensor gOutput_mn = zipped_divide(mOutput, typename AmpereUnpredicatedFprop::TilerOut{}); // ((BLK_M, BLK_N), (m', n'))
  dim3 lauch_grid {static_cast<uint32_t>(size<1,1>(gOutput_mn)), static_cast<uint32_t>(size<1,0>(gOutput_mn)), 1};

  CHECK_CUDA(cudaFuncSetAttribute(
    kernel_entrypoint<AmpereUnpredicatedFprop, decltype(mFilter), decltype(mXformedAct), decltype(mOutput)>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_iterations; ++i) {
    kernel_entrypoint<AmpereUnpredicatedFprop, decltype(mFilter), decltype(mXformedAct), decltype(mOutput)>
      <<<lauch_grid, AmpereUnpredicatedFprop::MaxThreadsPerBlock, smem_size>>>(
        mFilter, mXformedAct, mOutput);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds /= float(num_iterations);

  double tflop_count = (2 * double(size<0>(xformed_act_layout)) * double(size(filter_layout))) / double(1e12);
  double tflops = tflop_count / (double(milliseconds) / double(1e3));

  printf("Conv TFLOP count = %f\n", tflop_count);
  printf("Conv dense perf: %fms | TFLOP/s = %f\n", milliseconds, tflops);

  if (do_ref_check) {
    printf("Running host reference check ...\n");
    return fprop_reference(mFilter, mXformedAct, mOutput, mOutputRef);
  }
  else {
    return 0;
  }
}

int ampere_gather_scatter_conv_fprop(
    int num_images,
    float* activations,
    uint32_t *gather_idx_buf,
    float* filter,
    float* output,
    uint32_t *scatter_idx_buf,
    int num_iterations = 1) {
  auto D = typename AmpereUnpredicatedFprop::D{};
  auto H = typename AmpereUnpredicatedFprop::H{};
  auto W = typename AmpereUnpredicatedFprop::W{};
  auto Z = typename AmpereUnpredicatedFprop::Z{};
  auto P = typename AmpereUnpredicatedFprop::P{};
  auto Q = typename AmpereUnpredicatedFprop::Q{};
  auto C = typename AmpereUnpredicatedFprop::C{};
  auto K = typename AmpereUnpredicatedFprop::K{};
  auto S = typename AmpereUnpredicatedFprop::S{};
  auto R = typename AmpereUnpredicatedFprop::R{};
  auto T = typename AmpereUnpredicatedFprop::T{};

  int N = num_images; // dynamic
  if (N % int(typename AmpereUnpredicatedFprop::Tiler_N{}) != 0) {
    printf("ERROR: Input image count must be evenly divisible by CTA tiler N. Got num_images = %d\n", N);
    return 1;
  }

  // Tensor Filter    : (k,c,s,r,t)::(128,3,3,3,64):(1728,576,192,64,1)
  auto filter_layout = AmpereUnpredicatedFprop::GmemLayoutFlt{};

  // Tensor Output    : (n,z,p,q,k)::(?,4,2,2,128):(2048,1024,512,128,1)
  auto output_layout = make_ordered_layout(
    make_shape( K,   make_shape( N,   Z,   P,   Q)),
    make_tuple(_0{}, make_tuple(_4{},_3{},_2{},_1{})));

  // Input gather layout
  // inner_layout(make_coord((nzpq), (csrt))) => (idx_buffer_idx, dense_c_idx)
  auto EG = E<0>{};  // Gather basis     (1,0) (idx_buffer_idx) 
  auto EC = E<1>{};  // Contiguous basis (0,1) (dense_offset)    
  auto xformed_act_logical_inner = make_layout(
    make_shape (make_shape (       N,      Z,    P,  Q), make_shape ( C,      T,    R,  S)),
    make_stride(make_stride(D*H*W*EG, H*W*EG, W*EG, EG), make_stride(EC, H*W*EG, W*EG, EG)));

  // outer_layout(make_coord(idx_buffer_idx, dense_c_idx)) => idx
  // IndexedGather obtains idx by applying (gmem_base_ptr + gather_idx_buf[idx_buffer_idx] + dense_offset)
  auto xformed_act_gather_outer = make_layout(
    make_shape(_1{},_1{}),
    make_stride(CustomStride{IndexedGather{gather_idx_buf}, C}, _1{}));

  // Compose the inner and outer layouts
  // gather_composed(make_coord((nzpq), (csrt))) => idx
  auto xformed_act_composed_layout = composition(
    xformed_act_gather_outer,
    make_arithmetic_tuple(_0{}, _0{}),
    xformed_act_logical_inner);

  // Output scatter layout
  auto out_basis_stride = make_stride(
    E<1>{},
    make_stride(Z*P*Q*E<0>{}, P*Q*E<0>{}, Q*E<0>{}, _1{}*E<0>{})); // -> (crd0, crd1)
  auto out_basis_layout = make_layout(shape(output_layout), out_basis_stride);
  auto out_scatter_layout = make_layout(
    make_shape(_1{},_1{}),
    make_stride(CustomStride{IndexedGather{scatter_idx_buf}, K}, _1{}));
  auto out_composed_layout = composition(
    out_scatter_layout,
    make_arithmetic_tuple(_0{},_0{}),
    out_basis_layout);

  Tensor mXformedActGather = make_tensor(make_gmem_ptr(activations), xformed_act_composed_layout);
  Tensor mFilter = make_tensor(make_gmem_ptr(filter), filter_layout);
  Tensor mOutputScatter = make_tensor(make_gmem_ptr(output), out_composed_layout);  // (K, (N,Z,P,Q))

  Tensor gOutput_mn = zipped_divide(mOutputScatter, typename AmpereUnpredicatedFprop::TilerOut{}); // ((BLK_M, BLK_N), (m', n'))
  dim3 lauch_grid {static_cast<uint32_t>(size<1,1>(gOutput_mn)), static_cast<uint32_t>(size<1,0>(gOutput_mn)), 1};
  constexpr size_t smem_size = sizeof(typename AmpereUnpredicatedFprop::SharedStorage);

  print("xforemed gather layout ((N,Z,P,Q), (C,T,R,S)) = "); print(xformed_act_composed_layout); print("\n");
  print("Output  scatter layout ( K,        (N,Z,P,Q)) = "); print(out_composed_layout);         print("\n");
  print("Filter layout          ( K,        (C,T,R,S)) = "); print(filter_layout);               print("\n");

  CHECK_CUDA(cudaFuncSetAttribute(
    kernel_entrypoint<AmpereUnpredicatedFprop, decltype(mFilter), decltype(mXformedActGather), decltype(mOutputScatter)>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_iterations; ++i) {
    kernel_entrypoint<AmpereUnpredicatedFprop, decltype(mFilter), decltype(mXformedActGather), decltype(mOutputScatter)>
      <<<lauch_grid, AmpereUnpredicatedFprop::MaxThreadsPerBlock, smem_size>>>(
          mFilter, mXformedActGather, mOutputScatter);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds /= float(num_iterations);

  double tflop_count = (2 * double(size<0>(xformed_act_logical_inner)) * double(size(filter_layout))) / double(1e12);
  double tflops = tflop_count / (double(milliseconds) / double(1e3));
  printf("Conv TFLOP count = %f\n", tflop_count);
  printf("Conv gather/scatter perf: %fms | TFLOP/s = %f\n", milliseconds, tflops);

  return 0;
}

int
main(int argc, char const** argv) {
  cutlass::CommandLine cmd(argc, argv);
  std::cout << "Ampere convolution forward propagation kernel supporting both affine and gather/scatter tensors.\n\n";
  if (cmd.check_cmd_line_flag("help")) {
    std::cout
      << "Options:\n"
         "\t--n=<int>    Sets the number of images for the input activation tensor (dataset size). Default = 131072.\n"
         "\t--i=<int>    Sets the benchmarking repetitions. Default = 128.\n"
         "\t--nocheck    If specified, skips the reference check for dense kernel.\n"
         "\t--help       Displays this help message and exits.\n";
    return 0;
  }


  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
  if (props.major < 8) {
    std::cerr << "This example requires an Ampere GPU or newer.\n";
    return 0;
  }

  int num_images = 4320;
  cmd.get_cmd_line_argument("n", num_images, 4320);
  int num_iterations = 128;
  cmd.get_cmd_line_argument("i", num_iterations, 128);
  bool do_host_ref_check = not cmd.check_cmd_line_flag("no-check");

  auto D = typename AmpereUnpredicatedFprop::D{};
  auto H = typename AmpereUnpredicatedFprop::H{};
  auto W = typename AmpereUnpredicatedFprop::W{};
  auto Z = typename AmpereUnpredicatedFprop::Z{};
  auto P = typename AmpereUnpredicatedFprop::P{};
  auto Q = typename AmpereUnpredicatedFprop::Q{};
  auto C = typename AmpereUnpredicatedFprop::C{};
  auto K = typename AmpereUnpredicatedFprop::K{};

  auto activation_layout = make_layout(
    make_shape (make_shape (num_images,     D,   H, W), make_shape ( C,   _1{},_1{},_1{})),
    make_stride(make_stride(   D*H*W*C, H*W*C, W*C, C), make_stride(_1{}, _0{},_0{},_0{})));

  auto filter_layout = typename AmpereUnpredicatedFprop::GmemLayoutFlt{};

  auto output_layout = make_ordered_layout(
    make_shape( K,   make_shape(num_images,   Z,   P,   Q)),
    make_step (_0{}, make_step (      _4{},_3{},_2{},_1{})));

  print("Filter layout     ( K,        (C,T,R,S)) = "); print(filter_layout);     print("\n");
  print("Activation layout ((N,D,H,W), (C,1,1,1)) = "); print(activation_layout); print("\n");
  print("Output layout     ( K,        (N,Z,P,Q)) = "); print(output_layout);     print("\n");

  // allocate tensors
  std::cout << "Allocating tensors ... ";
  thrust::universal_vector<float> activation_data(size_t(cute::size(activation_layout)), float(0));
  thrust::universal_vector<float> filter_data(size_t(cute::size(filter_layout)), float(0));
  thrust::universal_vector<float> output_data(size_t(cute::size(output_layout)), float(0));
  thrust::universal_vector<float> output_data_ref(size_t(cute::size(output_layout)), float(0));
  std::cout << "done.\n";

  // init tensors
  std::cout << "Initializing data ... " << std::flush;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);
  for (std::size_t i = 0; i < size_t(cute::size(activation_layout)); ++i) {
    activation_data[i] = uniform_dist(gen);
  }

  for (std::size_t i = 0; i < size_t(cute::size(filter_layout)); ++i) {
    filter_data[i] = uniform_dist(gen);
  }
  std::cout << "done.\n";

  // set up index buffers for gather/scatter, fill with indireciton indices in reversed order
  std::cout << "Initializing gather/scatter index buffers ... ";
  thrust::universal_vector<uint32_t> gather_idx_buf(size_t(size<0>(activation_layout)));
  thrust::universal_vector<uint32_t> scatter_idx_buf(size_t(size<1>(output_layout)));
  thrust::sequence(gather_idx_buf.rbegin(), gather_idx_buf.rend());
  thrust::sequence(scatter_idx_buf.rbegin(), scatter_idx_buf.rend());
  std::cout << "done.\n";

  // launch dense
  std::cout << "\nRunning dense fprop kernel\n";
  int passed = ampere_dense_conv_fprop(
    num_images,
    activation_data.data().get(),
    filter_data.data().get(),
    output_data.data().get(),
    output_data_ref.data().get(),
    num_iterations,
    do_host_ref_check);

  // launch gather/scatter
  std::cout << "\nRunning gather/scatter fprop kernel\n";
  ampere_gather_scatter_conv_fprop(
    num_images,
    activation_data.data().get(),
    gather_idx_buf.data().get(),
    filter_data.data().get(),
    output_data.data().get(),
    scatter_idx_buf.data().get(),
    num_iterations);

  return passed;
}
