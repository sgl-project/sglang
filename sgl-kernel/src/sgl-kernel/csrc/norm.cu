#include <cstdint>
#include <flashinfer/norm.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
  CHECK_DIM(1, weight);  // weight: (hidden_size)
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  CHECK_EQ(output.size(0), batch_size);
  CHECK_EQ(output.size(1), hidden_size);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::RMSNorm(static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
                                       static_cast<c_type*>(output.data_ptr()), batch_size, hidden_size, eps, stream);
    TORCH_CHECK(status == cudaSuccess, "RMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}
