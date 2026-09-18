#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {
namespace qwen4_ple {
using B = bf16_t;
SGL_DEVICE void load16(const B* p, float* x) {
  device::AlignedVector<B, 8> low, high;
  low.load(p);
  high.load(p, 1);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    x[i] = float(low[i]);
    x[i + 8] = float(high[i]);
  }
}
SGL_DEVICE float bf(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
SGL_DEVICE float reduce(float x, float* sm) {
  x = device::warp::reduce_sum(x);
  int w = threadIdx.x / 32;
  sm[w] = x;
  __syncthreads();
  if (w == 0) {
    x = device::warp::reduce_sum(threadIdx.x < 5 ? sm[threadIdx.x] : 0.f);
    sm[threadIdx.x] = x;
  }
  __syncthreads();
  x = sm[w];
  __syncthreads();
  return x;
}
SGL_DEVICE void norm(float* x, const B* weight, float* sm) {
  float w[16];
  load16(weight + threadIdx.x * 16, w);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i += 2)
    s += x[i] * x[i] + x[i + 1] * x[i + 1];
  float f = rsqrtf(reduce(s, sm) / 2560.f + 1.e-6f);
#pragma unroll
  for (int i = 0; i < 16; i++)
    x[i] = bf(x[i] * f * (1.f + w[i]));
}
SGL_DEVICE void producer(
    const B* key,
    const B* query,
    const B* value,
    const B* kw,
    const B* qw,
    const B* cw,
    int token,
    int group,
    float* x,
    float* g,
    float* sm) {
  float q[16];
  int64_t base = int64_t(token) * 10240 + group * 2560 + threadIdx.x * 16;
  load16(key + base, x);
  load16(query + base, q);
  norm(x, kw + group * 2560, sm);
  norm(q, qw + group * 2560, sm);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++)
    s += bf(x[i] * q[i]);
  float gate = bf(bf(reduce(s, sm)) * 0.01976423537605237f);
  float magnitude = isnan(gate) ? gate : fmaxf(fabsf(gate), 1.e-6f);
  float root = bf(sqrtf(bf(magnitude)));
  float signed_root = bf(root * (gate > 0 ? 1.f : gate < 0 ? -1.f : 0.f));
  float a = bf(1.f / (1.f + expf(-signed_root)));
  load16(value + token * 2560 + threadIdx.x * 16, g);
#pragma unroll
  for (int i = 0; i < 16; i++) {
    g[i] = bf(a * g[i]);
    x[i] = g[i];
  }
  norm(x, cw + group * 2560, sm);
}

__global__ void qwen4_ple_producer_kernel(
    const B* key, const B* query, const B* value, const B* kw, const B* qw, const B* cw, B* xs, B* gs) {
  __shared__ float sm[32];
  int token = blockIdx.x / 4, group = blockIdx.x % 4;
  float x[16], g[16];
  producer(key, query, value, kw, qw, cw, token, group, x, g, sm);
#pragma unroll
  for (int i = 0; i < 16; i++) {
    int64_t p = int64_t(blockIdx.x) * 2560 + threadIdx.x * 16 + i;
    xs[p] = B(x[i]);
    gs[p] = B(g[i]);
  }
}

inline void produce(
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView query,
    tvm::ffi::TensorView value,
    tvm::ffi::TensorView key_weight,
    tvm::ffi::TensorView query_weight,
    tvm::ffi::TensorView conv_weight,
    tvm::ffi::TensorView normalized,
    tvm::ffi::TensorView gated) {
  using namespace host;
  SymbolicSize tokens{"tokens"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  TensorMatcher({tokens, 10240})
      .with_dtype<B>()
      .with_device(device)
      .verify(key)
      .verify(query)
      .verify(normalized)
      .verify(gated);
  TensorMatcher({tokens, 2560}).with_dtype<B>().with_device(device).verify(value);
  TensorMatcher({10240})
      .with_dtype<B>()
      .with_device(device)
      .verify(key_weight)
      .verify(query_weight)
      .verify(conv_weight);
  for (auto tensor : {key, query, value, key_weight, query_weight, conv_weight}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0)
        << "PLE producer requires 16-byte aligned input";
  }
  CHECK_HOST(tokens.unwrap() <= 2147483647 / 10240) << "PLE token count exceeds indexing range";
  if (tokens.unwrap() == 0) return;
  LaunchKernel(static_cast<uint32_t>(tokens.unwrap() * 4), 160, device.unwrap())(
      qwen4_ple_producer_kernel,
      static_cast<const B*>(key.data_ptr()),
      static_cast<const B*>(query.data_ptr()),
      static_cast<const B*>(value.data_ptr()),
      static_cast<const B*>(key_weight.data_ptr()),
      static_cast<const B*>(query_weight.data_ptr()),
      static_cast<const B*>(conv_weight.data_ptr()),
      static_cast<B*>(normalized.data_ptr()),
      static_cast<B*>(gated.data_ptr()));
}
}  // namespace qwen4_ple
}  // namespace sglang
