// DeepSeek-V4.1 vision RoPE (half-split), matching:
//   x1, x2 = x.float().chunk(2, dim=-1)
//   out = cat([x1*cos - x2*sin, x2*cos + x1*sin], -1).to(x.dtype)
//
// Layouts:
//   x   : [num_tokens, num_heads, head_dim] contiguous, head_dim even
//   cos : [num_tokens, head_dim/2] float32 contiguous
//   sin : same as cos

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

constexpr int64_t kBlockSize = 256;

// One thread = one (token, head, pair_index) along the first half of head_dim.
template <typename DType>
__global__ void vision_rope_half_kernel(
    DType* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim) {
  using namespace device;

  const int64_t half = head_dim >> 1;
  const int64_t n = num_tokens * num_heads * half;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  const int64_t token = idx / (num_heads * half);
  const int64_t rem = idx - token * (num_heads * half);
  const int64_t head = rem / half;
  const int64_t i = rem - head * half;

  DType* row = x + (token * num_heads + head) * head_dim;
  const float c = cos[token * half + i];
  const float s = sin[token * half + i];

  const float x1 = cast<fp32_t>(row[i]);
  const float x2 = cast<fp32_t>(row[i + half]);
  row[i] = cast<DType>(x1 * c - x2 * s);
  row[i + half] = cast<DType>(x2 * c + x1 * s);
}

template <typename DType>
void apply_rotary_inplace(tvm::ffi::TensorView x, tvm::ffi::TensorView cos, tvm::ffi::TensorView sin) {
  using namespace host;

  auto T = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"num_heads"};
  auto D = SymbolicSize{"head_dim"};
  auto Half = SymbolicSize{"half_dim"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({T, H, D})  //
      .with_dtype<DType>()
      .with_device(device_)
      .verify(x);
  TensorMatcher({T, Half})  //
      .with_dtype<fp32_t>()
      .with_device(device_)
      .verify(cos)
      .verify(sin);

  const int64_t num_tokens = T.unwrap();
  const int64_t num_heads = H.unwrap();
  const int64_t head_dim = D.unwrap();
  const int64_t half = Half.unwrap();
  CHECK_HOST(head_dim > 0 && (head_dim % 2) == 0) << "head_dim must be positive and even, got " << head_dim;
  CHECK_HOST(half == head_dim / 2) << "cos/sin last dim must be head_dim/2, got half=" << half
                                   << " head_dim=" << head_dim;
  if (num_tokens == 0 || num_heads == 0) {
    return;
  }

  const int64_t n = num_tokens * num_heads * half;
  const int64_t grid_size = div_ceil(n, kBlockSize);
  LaunchKernel(grid_size, kBlockSize, device_.unwrap())(
      vision_rope_half_kernel<DType>,
      static_cast<DType*>(x.data_ptr()),
      static_cast<const float*>(cos.data_ptr()),
      static_cast<const float*>(sin.data_ptr()),
      num_tokens,
      num_heads,
      head_dim);
}

}  // namespace sglang
