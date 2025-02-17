// References:
// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Extensions/GemmGroupedBatchedEx/cublas_GemmGroupedBatchedEx_example.cu
// https://github.com/zhihu/ZhiLight/blob/main/src/nn/linear/gemm_grouped.cpp

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cublas_v2.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "utils.h"

static void check_group_count(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& weights,
                              const std::vector<torch::Tensor>& outputs) {
  TORCH_CHECK(((inputs.size() == weights.size()) && (inputs.size() == outputs.size())),
              "The group count of inputs, weights and outputs should be the same.");
}

static void check_device_dtype(const torch::Dtype& dtype, const std::vector<torch::Tensor>& tensors) {
  for (const auto& t : tensors) {
    TORCH_CHECK(dtype == t.dtype(), "dtype of all the tensors should be the same");
    TORCH_CHECK(t.is_cuda(), "All tensors should be in Cuda memory");
  }
}

static std::vector<int> get_dims(const std::vector<torch::Tensor>& tensors, int dim) {
  std::vector<int> results;
  for (const auto& t : tensors) {
    TORCH_CHECK(t.dim() == 2, "Should pass in 2D matrices");
    results.push_back(t.size(dim));
  }
  return std::move(results);
}

static std::vector<int> get_strides(const std::vector<torch::Tensor>& tensors, int dim) {
  std::vector<int> results;
  for (const auto& t : tensors) {
    results.push_back(t.stride(dim));
  }
  return std::move(results);
}

static void check_equal(const std::vector<int>& a, const std::vector<int>& b, const std::string& err_msg) {
  for (int i = 0; i < a.size(); ++i) {
    TORCH_CHECK(a[i] == b[i], err_msg);
  }
}

static std::vector<void*> get_tensor_ptrs(const std::vector<torch::Tensor>& tensors) {
  std::vector<void*> ptrs;
  for (auto& t : tensors) {
    ptrs.push_back(t.data_ptr());
  }
  return std::move(ptrs);
}

static torch::Tensor create_ptr_pointer(const std::vector<void*>& ptrs, cudaStream_t stream) {
  auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA);
  torch::Tensor gpu_ptrs = torch::empty({static_cast<int>(ptrs.size())}, options);
  TORCH_CHECK(cudaMemcpyAsync(gpu_ptrs.data_ptr(), ptrs.data(), sizeof(void*) * ptrs.size(), cudaMemcpyHostToDevice,
                              stream) == CUBLAS_STATUS_SUCCESS);
  return gpu_ptrs;
}

// We want compute input @ weight^T in row major
// This is equivalent to computing weight @ input^T in col major
// Cublas only accepts matrix in column major, so this arrangement is needed
void cublas_grouped_gemm(const std::vector<torch::Tensor>& inputs,   // b: (m, k) row major = (k, m) col major
                         const std::vector<torch::Tensor>& weights,  // a: (n, k) row major = (n, k)^T col major
                         const std::vector<torch::Tensor>& outputs,  // c: (m, n) row major = (n, m) col major
                         const torch::Dtype& out_dtype, int64_t cublas_handle, int64_t cuda_stream) {
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16,
              "cublas grouped_gemm can"
              "only be applied to float16 and bfloat16 dtype");

  int group_count = inputs.size();
  check_group_count(inputs, weights, outputs);
  std::vector<int> group_size(group_count, 1);

  // Make sure all tensors are on cuda and use the same dtype
  check_device_dtype(out_dtype, inputs);
  check_device_dtype(out_dtype, weights);
  check_device_dtype(out_dtype, outputs);
  cudaDataType_t cuda_data_type = (out_dtype == torch::kHalf ? CUDA_R_16F : CUDA_R_16BF);

  // Weights should be transposed to (n, k) of column major
  std::vector<cublasOperation_t> transa_array(group_count, CUBLAS_OP_T);
  std::vector<cublasOperation_t> transb_array(group_count, CUBLAS_OP_N);

  // Get dim arrays
  std::vector<int> m_array = get_dims(weights, 0);
  std::vector<int> n_array = get_dims(inputs, 0);
  std::vector<int> k_array = get_dims(inputs, 1);

  // Make sure the dimensions in each group match
  std::vector<int> m_array1 = get_dims(outputs, 1);
  std::vector<int> n_array1 = get_dims(outputs, 0);
  std::vector<int> k_array1 = get_dims(weights, 1);
  check_equal(m_array, m_array1, "sizes don't match on m dimension");
  check_equal(n_array, n_array1, "sizes don't match on n dimension");
  check_equal(k_array, k_array1, "sizes don't match on k dimension");

  // Get leading dimensions
  std::vector<int> lda_array = get_strides(weights, 0);
  std::vector<int> ldb_array = get_strides(inputs, 0);
  std::vector<int> ldc_array = get_strides(outputs, 0);

  // Use default scaling factors
  std::vector<float> alpha_array(group_count, 1);
  std::vector<float> beta_array(group_count, 0);

  std::vector<void*> a_array = get_tensor_ptrs(weights);
  std::vector<void*> b_array = get_tensor_ptrs(inputs);
  std::vector<void*> c_array = get_tensor_ptrs(outputs);

  auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
  auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  // Should allocate tensors for storage of pointers
  torch::Tensor d_a = create_ptr_pointer(a_array, stream);
  torch::Tensor d_b = create_ptr_pointer(b_array, stream);
  torch::Tensor d_c = create_ptr_pointer(c_array, stream);

#if defined CUDA_VERSION && CUDA_VERSION >= 12050
  auto status = cublasGemmGroupedBatchedEx(handle, transa_array.data(), transb_array.data(), m_array.data(),
                                           n_array.data(), k_array.data(), alpha_array.data(), (void**)d_a.data_ptr(),
                                           cuda_data_type, lda_array.data(), (void**)d_b.data_ptr(), cuda_data_type,
                                           ldb_array.data(), beta_array.data(), (void**)d_c.data_ptr(), cuda_data_type,
                                           ldc_array.data(), group_count, group_size.data(), CUBLAS_COMPUTE_32F);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublas grouped gemm failed: ", cublasGetStatusString(status));
  TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "Failed when stream synchronization");
  return;
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "Cublas GroupGemm is not implemented with current compute capability: ", getSMVersion());
}
