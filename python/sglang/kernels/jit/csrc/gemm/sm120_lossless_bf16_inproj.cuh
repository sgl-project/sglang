// Lossless M4 Nemotron input projection. Preserve the production two-part
// reduction: round the first K half to BF16 before adding the FP32 second half.
#include "sm120_lossless_bf16_moe.cuh"
namespace sglang {
template <uint32_t N, uint32_t K, bool Packed>
__device__ __forceinline__ void lossless_prefetch_part(
    uint8_t* dst,
    uint64_t* metadata_shared,
    const bf16_t* raw,
    const uint32_t* data,
    const uint64_t* metadata,
    uint32_t expert,
    uint32_t tile,
    uint32_t stage,
    uint32_t end,
    uint32_t lane) {
  uint32_t col = stage * 128;
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    uint32_t local_row = 2 * j + lane / 16, row = tile + local_row;
    uint64_t meta = 0;
    size_t offset = 0;
    bool compressed = false;
    if (col < end) {
      if constexpr (Packed) {
        meta = metadata[(static_cast<size_t>(expert) * N + row) * (K / 256) + col / 256];
        compressed = (meta & (1ull << 40)) != 0;
        offset = static_cast<uint32_t>(meta) + (col % 256) / 128 * (compressed ? 48 : 64);
      } else
        offset = ((static_cast<size_t>(expert) * N + row) * K + col) / 2;
    }
    if (lane % 16 == 0) metadata_shared[local_row] = meta & 0xffffffff00000000ull;
    uint32_t bytes = col < end && lane % 16 < (compressed ? 12 : 16) ? 16 : 0;
    const uint32_t* source = Packed ? data : reinterpret_cast<const uint32_t*>(raw);
    source += offset + (bytes ? (lane % 16) * 4 : 0);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst + local_row * 272 + (lane % 16) * 16));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(addr), "l"(source), "r"(bytes) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <uint32_t N, uint32_t K, uint32_t Split>
__global__ __launch_bounds__(64) void lossless_inproj_tc4(
    const bf16_t* x, const uint32_t* payload, const uint64_t* metadata, bf16_t* out, float* partial) {
  __shared__ __align__(16) uint8_t packed[2][2][16 * 272];
  __shared__ uint64_t metas[2][2][16];
  __shared__ __align__(16) bf16_t ws[2][16][24];
  __shared__ __align__(16) bf16_t sx[4][128];
  uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32, tile = (blockIdx.x * 2 + warp) * 16;
  uint32_t start = blockIdx.y * (K / Split), end = start + K / Split;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  for (uint32_t stage = 0; stage < 2; ++stage)
    lossless_prefetch_part<N, K, true>(
        packed[warp][stage], metas[warp][stage], nullptr, payload, metadata, 0, tile, start / 128 + stage, end, lane);
  for (uint32_t base = start; base < end; base += 128) {
    uint32_t buf = ((base - start) / 128) % 2;
    *reinterpret_cast<float4*>(sx[tid / 16] + (tid % 16) * 8) =
        *reinterpret_cast<const float4*>(x + (tid / 16) * K + base + (tid % 16) * 8);
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (uint32_t inner = 0; inner < 128; inner += 16) {
      uint64_t meta = metas[warp][buf][lane % 16] | ((lane % 16) * 68);
      float4 wv = load_lossless_shared(
          reinterpret_cast<const uint32_t*>(packed[warp][buf]), meta, (inner + (lane / 16) * 8) / 8);
      *reinterpret_cast<float4*>(&ws[warp][lane % 16][(lane / 16) * 8]) = wv;
      __syncwarp();
      uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&ws[warp][lane % 16][(lane / 16) * 8]));
      uint32_t a0, a1, a2, a3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                   : "r"(addr)
                   : "memory");
      uint32_t bt = (lane / 4) % 4;
      uint32_t b0 = *reinterpret_cast<uint32_t*>(sx[bt] + inner + (lane % 4) * 2);
      uint32_t b1 = *reinterpret_cast<uint32_t*>(sx[bt] + inner + (lane % 4) * 2 + 8);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      __syncwarp();
    }
    __syncthreads();
    lossless_prefetch_part<N, K, true>(
        packed[warp][buf], metas[warp][buf], nullptr, payload, metadata, 0, tile, base / 128 + 2, end, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (lane % 4 < 2) {
    uint32_t row = tile + lane / 4, token = 2 * (lane % 4);
    if constexpr (Split == 1) {
      out[token * N + row] = cast<bf16_t>(d0);
      out[(token + 1) * N + row] = cast<bf16_t>(d1);
      out[token * N + row + 8] = cast<bf16_t>(d2);
      out[(token + 1) * N + row + 8] = cast<bf16_t>(d3);
    } else {
      partial += blockIdx.y * 4 * N;
      partial[token * N + row] = d0;
      partial[(token + 1) * N + row] = d1;
      partial[token * N + row + 8] = d2;
      partial[(token + 1) * N + row + 8] = d3;
    }
  }
}
template <uint32_t N, uint32_t Split>
__global__ void reduce_dense(const float* partial, bf16_t* out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 4 * N) return;
  float v = 0.f;
#pragma unroll
  for (uint32_t p = 0; p < Split; ++p)
    v = static_cast<float>(cast<bf16_t>(v + partial[p * 4 * N + i]));
  out[i] = cast<bf16_t>(v);
}
template <uint32_t N, uint32_t K, uint32_t Split>
struct LosslessInprojTc4Split {
  static_assert(N == 18560 && K == 4096 && Split == 2);
  /// Compute four rows using every original BF16 weight and production rounding.
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView p,
      tvm::ffi::TensorView m,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView partial) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({4, K}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({N * K / 2}).with_dtype<uint32_t>().with_device(dev).verify(p);
    TensorMatcher({N * K / 256}).with_dtype<uint64_t>().with_device(dev).verify(m);
    TensorMatcher({4, N}).with_dtype<bf16_t>().with_device(dev).verify(out);
    TensorMatcher({Split, 4, N}).with_dtype<float>().with_device(dev).verify(partial);
    LaunchKernel({N / 32, Split}, 64, x.device())(
        lossless_inproj_tc4<N, K, Split>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const uint32_t*>(p.data_ptr()),
        static_cast<const uint64_t*>(m.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<float*>(partial.data_ptr()));
    if constexpr (Split > 1)
      LaunchKernel((4 * N + 255) / 256, 256, x.device())(
          reduce_dense<N, Split>, static_cast<const float*>(partial.data_ptr()), static_cast<bf16_t*>(out.data_ptr()));
  }
};
}  // namespace sglang
