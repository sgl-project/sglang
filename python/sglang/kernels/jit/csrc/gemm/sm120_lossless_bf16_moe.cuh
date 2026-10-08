// Restore original BF16 expert bits and evaluate every selected route.
// Two-stage 128-column async copies hide compressed-weight load latency.
// Raw cp.async and mma expose the needed staging and accumulation order.
// MMA K order and FP32 ReLU-squared precede the production BF16 epilogue.
#include "sm120_lossless_bf16_head.cuh"
namespace sglang {
__device__ __forceinline__ float4 load_lossless_shared(const uint32_t* data, uint64_t meta, uint32_t lane) {
  uint32_t offset = static_cast<uint32_t>(meta);
  if (!(meta & (1ull << 40))) return (*reinterpret_cast<const float4*>(data + offset + lane * 4));
  uint32_t base = (meta >> 32) & 255;
  const uint32_t* p = data + offset + lane * 3;
  uint32_t a = p[0], b = p[1], c = p[2];
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

template <uint32_t N, uint32_t K, bool Packed>
__device__ __forceinline__ void lossless_moe_prefetch(
    uint8_t* dst,
    uint64_t* metadata_shared,
    const bf16_t* raw,
    const uint32_t* data,
    const uint64_t* metadata,
    uint32_t expert,
    uint32_t tile,
    uint32_t stage,
    uint32_t lane) {
  uint32_t col = stage * 128;
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    uint32_t local_row = 2 * j + lane / 16, row = tile + local_row;
    uint64_t meta = 0;
    size_t offset = 0;
    bool compressed = false;
    if (col < K) {
      if constexpr (Packed) {
        meta = metadata[(static_cast<size_t>(expert) * N + row) * (K / 256) + col / 256];
        compressed = (meta & (1ull << 40)) != 0;
        offset = static_cast<uint32_t>(meta) + (col % 256) / 128 * (compressed ? 48 : 64);
      } else
        offset = ((static_cast<size_t>(expert) * N + row) * K + col) / 2;
    }
    if (lane % 16 == 0) metadata_shared[local_row] = meta & 0xffffffff00000000ull;
    uint32_t bytes = col < K && lane % 16 < (compressed ? 12 : 16) ? 16 : 0;
    const uint32_t* source = Packed ? data : reinterpret_cast<const uint32_t*>(raw);
    source += offset + (bytes ? (lane % 16) * 4 : 0);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst + local_row * 272 + (lane % 16) * 16));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(addr), "l"(source), "r"(bytes) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <uint32_t N, uint32_t K, uint32_t XStride, bool FC1, bool Packed>
__global__ __launch_bounds__(64) void lossless_moe_dot(
    const bf16_t* x,
    const bf16_t* raw,
    const uint32_t* data,
    const uint64_t* metadata,
    const int32_t* ids,
    bf16_t* out) {
  __shared__ bf16_t sx[K];
  __shared__ __align__(16) bf16_t ws[2][16][24];
  __shared__ __align__(16) uint8_t packed_ws[2][2][16 * 272];
  __shared__ uint64_t metas[2][2][16];
  uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  uint32_t token = blockIdx.z, slot = blockIdx.y, route = token * 22 + slot;
  uint32_t expert = ids[route], tile = (blockIdx.x * 2 + warp) * 16;
  if constexpr (FC1) {
    if (blockIdx.x == 0)
      for (uint32_t i = tid; i < 128; i += 64)
        out[route * 2816 + 2688 + i] = cast<bf16_t>(0.f);
  }
  const bf16_t* source = x + (FC1 ? token : route) * XStride;
  for (uint32_t i = tid * 8; i < K; i += 64 * 8)
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(source + i);
  __syncthreads();
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  for (uint32_t stage = 0; stage < 2; ++stage)
    lossless_moe_prefetch<N, K, Packed>(
        packed_ws[warp][stage], metas[warp][stage], raw, data, metadata, expert, tile, stage, lane);
  for (uint32_t base = 0; base < K; base += 128) {
    uint32_t buf = (base / 128) % 2;
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (uint32_t inner = 0; inner < 128; inner += 16) {
      uint64_t meta = metas[warp][buf][lane % 16] | ((lane % 16) * 68);
      float4 wv = load_lossless_shared(
          reinterpret_cast<const uint32_t*>(packed_ws[warp][buf]), meta, (inner + (lane / 16) * 8) / 8);
      *reinterpret_cast<float4*>(&ws[warp][lane % 16][(lane / 16) * 8]) = wv;
      __syncwarp();
      uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&ws[warp][lane % 16][(lane / 16) * 8]));
      uint32_t a0, a1, a2, a3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                   : "r"(addr)
                   : "memory");
      uint32_t b0 = *reinterpret_cast<uint32_t*>(sx + base + inner + (lane % 4) * 2);
      uint32_t b1 = *reinterpret_cast<uint32_t*>(sx + base + inner + (lane % 4) * 2 + 8);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      __syncwarp();
    }
    lossless_moe_prefetch<N, K, Packed>(
        packed_ws[warp][buf], metas[warp][buf], raw, data, metadata, expert, tile, base / 128 + 2, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (lane % 4 == 0) {
    uint32_t row = tile + lane / 4;
    if constexpr (FC1) {
      float v0 = fmaxf(d0, 0.f), v1 = fmaxf(d2, 0.f);
      out[route * 2816 + row] = cast<bf16_t>(v0 * v0);
      out[route * 2816 + row + 8] = cast<bf16_t>(v1 * v1);
    } else {
      out[route * N + row] = cast<bf16_t>(d0);
      out[route * N + row + 8] = cast<bf16_t>(d2);
    }
  }
}
__global__ void lossless_moe_reduce(const bf16_t* partial, const float* weights, bf16_t* out, uint32_t M) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * 1024) return;
  uint32_t token = i / 1024, col = i % 1024;
  float value = 0.f;
  for (uint32_t slot = 0; slot < 22; ++slot)
    value += static_cast<float>(partial[(token * 22 + slot) * 1024 + col]) * weights[token * 22 + slot];
  out[i] = cast<bf16_t>(value);
}
__global__ void restore_expert(const uint32_t* payload, const uint64_t* metadata, bf16_t* out, uint32_t values) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= values / 8) return;
  *reinterpret_cast<float4*>(out + static_cast<size_t>(i) * 8) = load_lossless8(payload, metadata[i / 32], i % 32);
}
struct LosslessBf16Moe {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView p1,
      tvm::ffi::TensorView m1,
      tvm::ffi::TensorView p2,
      tvm::ffi::TensorView m2,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView routing,
      tvm::ffi::TensorView z,
      tvm::ffi::TensorView partial,
      tvm::ffi::TensorView out) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto rows = SymbolicSize{"rows"};
    TensorMatcher({rows, 1024}).with_dtype<bf16_t>().with_device(dev).verify(x).verify(out);
    uint32_t M = rows.unwrap();
    CHECK_HOST(M > 0 && M <= 4);
    TensorMatcher({512u * 2688u * 1024u / 2u}).with_dtype<uint32_t>().with_device(dev).verify(p1);
    TensorMatcher({512u * 1024u * 2816u / 2u}).with_dtype<uint32_t>().with_device(dev).verify(p2);
    TensorMatcher({512u * 2688u * 1024u / 256u}).with_dtype<uint64_t>().with_device(dev).verify(m1);
    TensorMatcher({512u * 1024u * 2816u / 256u}).with_dtype<uint64_t>().with_device(dev).verify(m2);
    TensorMatcher({M, 22}).with_dtype<int32_t>().with_device(dev).verify(ids);
    TensorMatcher({M, 22}).with_dtype<float>().with_device(dev).verify(routing);
    TensorMatcher({M * 22, 2816}).with_dtype<bf16_t>().with_device(dev).verify(z);
    TensorMatcher({M * 22, 1024}).with_dtype<bf16_t>().with_device(dev).verify(partial);
    auto device = dev.unwrap();
    LaunchKernel({84, 22, M}, 64, device)(
        lossless_moe_dot<2688, 1024, 1024, true, true>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(nullptr),
        static_cast<const uint32_t*>(p1.data_ptr()),
        static_cast<const uint64_t*>(m1.data_ptr()),
        static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<bf16_t*>(z.data_ptr()));
    LaunchKernel({32, 22, M}, 64, device)(
        lossless_moe_dot<1024, 2816, 2816, false, true>,
        static_cast<const bf16_t*>(z.data_ptr()),
        static_cast<const bf16_t*>(nullptr),
        static_cast<const uint32_t*>(p2.data_ptr()),
        static_cast<const uint64_t*>(m2.data_ptr()),
        static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<bf16_t*>(partial.data_ptr()));
    LaunchKernel((M * 1024 + 255) / 256, 256, device)(
        lossless_moe_reduce,
        static_cast<const bf16_t*>(partial.data_ptr()),
        static_cast<const float*>(routing.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        M);
  }
  /// Restore one expert for a full loading-time audit, including padding.
  static void restore(tvm::ffi::TensorView payload, tvm::ffi::TensorView meta, tvm::ffi::TensorView out) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto count = SymbolicSize{"count"};
    auto n = SymbolicSize{"n"};
    auto k = SymbolicSize{"k"};
    TensorMatcher({count}).with_dtype<uint32_t>().with_device(dev).verify(payload);
    TensorMatcher({n, k}).with_dtype<bf16_t>().with_device(dev).verify(out);
    CHECK_HOST(k.unwrap() % 256 == 0);
    TensorMatcher({n.unwrap() * k.unwrap() / 256}).with_dtype<uint64_t>().with_device(dev).verify(meta);
    uint32_t values = n.unwrap() * k.unwrap();
    LaunchKernel((values / 8 + 255) / 256, 256, dev.unwrap())(
        restore_expert,
        static_cast<const uint32_t*>(payload.data_ptr()),
        static_cast<const uint64_t*>(meta.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        values);
  }
};
}  // namespace sglang
