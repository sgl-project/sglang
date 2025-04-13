// Adapted from
// https://github.com/flashinfer-ai/flashinfer/blob/55576c626421b5ee7e7ebe74afd26465c8ae863f/csrc/cascade.cu

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <flashinfer/attention/cascade.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void merge_state(
    at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b, at::Tensor v_merged, at::Tensor s_merged) {
  CHECK_INPUT(v_a);
  CHECK_INPUT(s_a);
  CHECK_INPUT(v_b);
  CHECK_INPUT(s_b);
  auto device = v_a.device();
  CHECK_EQ(s_a.device(), device);
  CHECK_EQ(v_b.device(), device);
  CHECK_EQ(s_b.device(), device);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(s_a, s_b);
  CHECK_EQ(v_a.size(0), s_a.size(0));
  CHECK_EQ(v_a.size(1), s_b.size(1));
  unsigned int seq_len = v_a.size(0);
  unsigned int num_heads = v_a.size(1);
  unsigned int head_dim = v_a.size(2);

  const c10::cuda::OptionalCUDAGuard device_guard(v_a.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(v_a.scalar_type(), c_type, [&] {
    cudaError_t status = MergeState(
        static_cast<c_type*>(v_a.data_ptr()),
        static_cast<float*>(s_a.data_ptr()),
        static_cast<c_type*>(v_b.data_ptr()),
        static_cast<float*>(s_b.data_ptr()),
        static_cast<c_type*>(v_merged.data_ptr()),
        static_cast<float*>(s_merged.data_ptr()),
        seq_len,
        num_heads,
        head_dim,
        stream);
    TORCH_CHECK(status == cudaSuccess, "MergeState kernel launch failed: ", cudaGetErrorString(status));
    return true;
  });

  TORCH_CHECK(success, "MergeState kernel launch failed: unsupported data type");
}
