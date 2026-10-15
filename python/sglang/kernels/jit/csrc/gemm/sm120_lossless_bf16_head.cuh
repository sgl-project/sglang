// Experimental lossless representation. Dot8 and reduction order are identical
// to the qualified original-BF16 GEMV; every BF16 bit is reconstructed first.
#include "hopper_bf16_gemv.cuh"

namespace sglang {
__device__ __forceinline__ uint16_t restore_bf16(uint32_t code, uint32_t base) {
  return (code & 127) | ((code & 128) << 8) | (((code >> 8) + base) << 7);
}
__device__ __forceinline__ float4 load_lossless8(const uint32_t* data, uint64_t meta, uint32_t lane) {
  uint32_t offset = static_cast<uint32_t>(meta);
  if (!(meta & (1ull << 40))) return __ldcs(reinterpret_cast<const float4*>(data + offset + lane * 4));
  uint32_t base = (meta >> 32) & 255;
  const uint32_t* p = data + offset + lane * 3;
  uint32_t a = __ldcs(p), b = __ldcs(p + 1), c = __ldcs(p + 2);
  uint32_t codes[8] = {
      a & 4095,
      (a >> 12) & 4095,
      (a >> 24) | ((b & 15) << 8),
      (b >> 4) & 4095,
      (b >> 16) & 4095,
      (b >> 28) | ((c & 255) << 4),
      (c >> 8) & 4095,
      c >> 20};
  alignas(16) uint32_t words[4];
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
    words[i] = static_cast<uint32_t>(restore_bf16(codes[2 * i], base)) |
               (static_cast<uint32_t>(restore_bf16(codes[2 * i + 1], base)) << 16);
  return *reinterpret_cast<float4*>(words);
}

template <uint32_t N, uint32_t K>
__global__ __launch_bounds__(256) void lossless_bf16_gemv(
    const bf16_t* __restrict__ x,
    const uint32_t* __restrict__ data,
    const uint64_t* __restrict__ metadata,
    bf16_t* __restrict__ out) {
  __shared__ bf16_t sx[K];
  uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  for (uint32_t i = tid * 8; i < K; i += 256 * 8)
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  __syncthreads();
  uint32_t row = blockIdx.x * 8 + warp;
  if (row >= N) return;
  float acc = 0.f;
  // The original kernel uses these same consecutive eight-value dot products,
  // FP32 += boundaries, and warp shuffle reduction.
  for (uint32_t k = lane * 8; k < K; k += 256) {
    uint64_t meta = metadata[static_cast<size_t>(row) * (K / 256) + k / 256];
    float4 wv = load_lossless8(data, meta, lane);
    float4 xv = *reinterpret_cast<const float4*>(sx + k);
    acc += dot8_f32(wv, xv);
  }
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  if (lane == 0) out[row] = cast<bf16_t>(acc);
}

template <uint32_t N, uint32_t K>
__global__ void lossless_bf16_restore(
    const uint32_t* __restrict__ data, const uint64_t* __restrict__ metadata, bf16_t* __restrict__ out) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * K / 8) return;
  uint64_t meta = metadata[idx / 32];
  *reinterpret_cast<float4*>(out + static_cast<size_t>(idx) * 8) = load_lossless8(data, meta, idx % 32);
}

template <uint32_t N, uint32_t K>
struct LosslessBf16 {
  static_assert(N > 0 && K % 256 == 0 && K * sizeof(bf16_t) <= 48 * 1024);
  static_assert(static_cast<uint64_t>(N) * K < (1ull << 32));
  /// Evaluate the original BF16 GEMV after lossless weight-bit reconstruction.
  static void
  run(tvm::ffi::TensorView x, tvm::ffi::TensorView data, tvm::ffi::TensorView meta, tvm::ffi::TensorView out) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({N * K / 2}).with_dtype<uint32_t>().with_device(dev).verify(data);
    TensorMatcher({1, K}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({N * K / 256}).with_dtype<uint64_t>().with_device(dev).verify(meta);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(dev).verify(out);
    LaunchKernel((N + 7) / 8, 256, dev.unwrap())(
        lossless_bf16_gemv<N, K>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const uint32_t*>(data.data_ptr()),
        static_cast<const uint64_t*>(meta.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
  }
  /// Restore every BF16 bit for loading-time audits, including NaN payloads.
  static void restore(tvm::ffi::TensorView data, tvm::ffi::TensorView meta, tvm::ffi::TensorView out) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({N * K / 2}).with_dtype<uint32_t>().with_device(dev).verify(data);
    TensorMatcher({N * K / 256}).with_dtype<uint64_t>().with_device(dev).verify(meta);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(dev).verify(out);
    LaunchKernel((N * K / 8 + 255) / 256, 256, dev.unwrap())(
        lossless_bf16_restore<N, K>,
        static_cast<const uint32_t*>(data.data_ptr()),
        static_cast<const uint64_t*>(meta.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
  }
};
}  // namespace sglang
