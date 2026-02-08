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

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <iostream>
#include <fstream>

#include "kernel_backward.h"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"


using Arch = cutlass::arch::Sm80;
static constexpr int kMaxK = 128;

template <typename ArchTag, typename Element, int kMaxK>
struct DefaultKernel {
    // Some heuristics to select the best kernel (tested on Sm60, Sm70, Sm80)
    // NOTE: Requires quite a lot of shmem for Sm80+,
    // so might require tweaking those manually for Sm86/Sm89

    static constexpr bool kSupports64x128 =
        ArchTag::kMinComputeCapability >= 80 ||
        (ArchTag::kMinComputeCapability >= 70 &&
        cutlass::sizeof_bits<Element>::value <= 16);
    static constexpr int kBlockSizeI = kSupports64x128 && kMaxK > 64 ? 128 : 64;
    static constexpr bool kIsHalf = cutlass::sizeof_bits<Element>::value <= 16;
    static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;
    static constexpr bool kPreload = kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF;
    static constexpr int kBlockSizeJ = kPreload && kMaxK > 64 ? 128 : 64;

    using Kernel = AttentionBackwardKernel<
        Arch,
        Element,
        true,        // kIsAligned_
        false,       // kApplyDropout_
        kPreload,    // kPreload_
        kBlockSizeI, // kBlockSizeI_,
        kBlockSizeJ, // kBlockSizeJ_,
        kMaxK,       // kMaxK
        false,       // kKeysQueriesAlignedToBlockSize
        true         // kEnableSplitKeys
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
template <typename T> struct TypeName;
template <> struct TypeName<float> { static constexpr const char* Name = "f32"; };
template <> struct TypeName<cutlass::half_t> { static constexpr const char* Name = "f16"; };
template <> struct TypeName<cutlass::bfloat16_t> { static constexpr const char* Name = "b16"; };

void readExpect(std::string const& expected) {
    std::string read;
    std::cin >> read;
    if (read != expected) {
        std::cerr << "FATAL: Read '" << read << "' but expected '" << expected << "'" << std::endl;
        std::exit(1);
    }
}

/// Helpers to read from stdin
template <typename Element>
cutlass::HostTensor<Element, cutlass::layout::RowMajor> readTensorOnDevice(std::string const& expectedName) {
    readExpect("tensor_begin");
    readExpect(std::string(TypeName<Element>::Name) + ":" + expectedName);
    uint64_t len = 0;
    std::cin >> len;
    readExpect("file");
    std::string filename;
    std::cin >> filename;

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor({int64_t(1), int64_t(len / sizeof(Element))});
    uint8_t* data = (uint8_t*)tensor.host_data();

    std::fstream myFile(filename, std::ios::in | std::ios::binary );
    myFile.read((char*)data, len);
    readExpect("tensor_end");
    tensor.sync_device();
    return tensor;
}

int64_t readInt64(std::string const& expectedName) {
    readExpect(expectedName);
    int64_t s = 0;
    std::cin >> s;
    return s;
}

float readFloat(std::string const& expectedName) {
    readExpect(expectedName);
    float s = 0;
    std::cin >> s;
    return s;
}

// Writing
template <typename Element>
void writeTensor(std::string const& name, cutlass::HostTensor<Element, cutlass::layout::RowMajor>& tensor) {
    tensor.sync_host(); // device->host
    size_t u8len = tensor.size() * sizeof(Element);

    // Python is expected to provide a file name to write to
    readExpect("tmpfile");
    std::string tmpfile;
    std::cin >> tmpfile;

    uint8_t* data = (uint8_t*)tensor.host_data();
    std::fstream myFile(tmpfile, std::ios::out | std::ios::binary );
    myFile.write((char*)data, u8len);
    myFile.close();

    std::cout << "tensor_begin " << TypeName<Element>::Name << ":" << name << " ";
    std::cout << u8len << " file " << tmpfile << " tensor_end" << std::endl;
}

void writeInt64(std::string const& name, int64_t value) {
    std::cout << name << " " << value << std::endl;
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
int runKernel() {
    using Kernel = typename DefaultKernel<Arch, Element, kMaxK>::Kernel;

#define READ_I64(NAME) p.NAME = (decltype(p.NAME))readInt64(#NAME)
#define READ_TENSOR_AND_STRIDES_BMH(DT, NAME, NAME_XS) \
    auto storage##NAME = readTensorOnDevice<DT>(#NAME); \
    p.NAME##_ptr = storage##NAME.device_data(); \
    READ_I64(NAME_XS##_strideB); \
    READ_I64(NAME_XS##_strideM); \
    READ_I64(NAME_XS##_strideH);

#define CUDA_CHECK(FN) { \
    auto cudaError = FN; \
    if (cudaError != cudaSuccess) { \
        std::cerr << "FATAL: " #FN " failed: " << cudaGetErrorString(cudaError) << std::endl; \
        return -1; \
    } \
}

    typename Kernel::Params p;
    p.scale = readFloat("scale");
    READ_I64(head_dim);
    READ_I64(head_dim_value);
    READ_I64(num_queries);
    READ_I64(num_keys);
    READ_I64(num_heads);
    READ_I64(custom_mask_type);
    READ_I64(num_batches);
    int64_t repeat_count = readInt64("repeat_count");
    READ_I64(num_splits_key);

    READ_TENSOR_AND_STRIDES_BMH(Element, query, q);
    READ_TENSOR_AND_STRIDES_BMH(Element, key, k);
    READ_TENSOR_AND_STRIDES_BMH(Element, value, v);
    auto lse = readTensorOnDevice<typename Kernel::lse_scalar_t>("logsumexp");
    p.logsumexp_ptr = lse.device_data();
    p.lse_strideB = readInt64("lse_strideB");
    p.lse_strideH = readInt64("lse_strideH");

    // output
    auto stOutput = readTensorOnDevice<Element>("output");
    p.output_ptr = stOutput.device_data();
    READ_I64(o_strideB);
    auto o_strideM = readInt64("o_strideM");
    if (o_strideM != p.o_strideM()) {
        std::cerr << "Invalid `o_strideM`: " << o_strideM << " - expected " << p.o_strideM();
        return 2;
    }
    READ_I64(o_strideH);

    READ_TENSOR_AND_STRIDES_BMH(Element, grad_output, gO);

    auto stDelta = readTensorOnDevice<typename Kernel::accum_t>("delta");
    p.delta_ptr = stDelta.device_data();
    READ_I64(delta_strideB);
    READ_I64(delta_strideH);

    // Allocate workspace
    if (p.workspace_size()) {
        cudaMalloc(&p.workspace, p.workspace_size());
    }

    // Allocate outputs in BMHK format
    p.gQKV_strideM_multiplier = 1;
    p.gQ_strideH = p.head_dim;
    p.gQ_strideB = p.gQ_strideM() * p.num_queries;
    p.gK_strideH = p.head_dim;
    p.gK_strideB = p.gK_strideM() * p.num_keys;
    p.gV_strideH = p.head_dim_value;
    p.gV_strideB = p.gV_strideM() * p.num_keys;

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> gQ({int64_t(1), p.gQ_strideB * p.num_batches});
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> gK({int64_t(1), p.gK_strideB * p.num_batches});
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> gV({int64_t(1), p.gV_strideB * p.num_batches});
    p.grad_query_ptr = gQ.device_data();
    p.grad_key_ptr = gK.device_data();
    p.grad_value_ptr = gV.device_data();

    if (!Kernel::check_supported(p)) {
      std::cerr << "FATAL: Kernel does not support these inputs" << std::endl;
      return 2;
    }

    // Run kernel
    cudaDeviceSynchronize();
    auto kernel_fn = attention_kernel_backward_batched_impl<Kernel>;
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes)));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // Write outputs
    std::cout << "OK ";
    writeTensor("grad_query", gQ);
    writeInt64("gQ_strideB", p.gQ_strideB);
    writeInt64("gQ_strideM", p.gQ_strideM());
    writeInt64("gQ_strideH", p.gQ_strideH);
    writeTensor("grad_key", gK);
    writeInt64("gK_strideB", p.gK_strideB);
    writeInt64("gK_strideM", p.gK_strideM());
    writeInt64("gK_strideH", p.gK_strideH);
    writeTensor("grad_value", gV);
    writeInt64("gV_strideB", p.gV_strideB);
    writeInt64("gV_strideM", p.gV_strideM());
    writeInt64("gV_strideH", p.gV_strideH);

    // Timing
    cudaEvent_t events[2];
    for (auto & event : events) {
      CUDA_CHECK(cudaEventCreate(&event));
    }
    CUDA_CHECK(cudaEventRecord(events[0]));
    for (int i = 0; i < repeat_count; ++i) {
        kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    }
    CUDA_CHECK(cudaEventRecord(events[1]));
    CUDA_CHECK(cudaEventSynchronize(events[1]));
    // Measure elapsed runtime
    float runtime_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

    std::cout << "runtime_ms " << runtime_ms / float(repeat_count) << std::endl;
    return 0;
}

int main() {
    std::ios_base::sync_with_stdio(false);

    std::string dtype;
    std::cin >> dtype;
    std::cerr << "Running kernel with dtype: " << dtype << std::endl;
    if (dtype == "f16") {
        return runKernel<cutlass::half_t>();
    } else if (dtype == "b16") {
        return runKernel<cutlass::bfloat16_t>();
    } else if (dtype == "f32") {
        return runKernel<float>();
    } else {
        std::cerr << "FATAL: Unknown dtype: " << dtype << std::endl;
        return 3;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
