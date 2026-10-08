// Full-vocabulary M4 BF16 head with original tensor-core accumulation order.
#include "sm120_lossless_bf16_moe.cuh"
namespace sglang {
template <uint32_t N, uint32_t K>
__global__ __launch_bounds__(64) void lossless_head_tc4(
    const bf16_t* x, const uint32_t* payload, const uint64_t* metadata, bf16_t* out) {
  __shared__ __align__(16) uint8_t packed[2][2][16 * 272];
  __shared__ uint64_t metas[2][2][16];
  __shared__ __align__(16) bf16_t ws[2][16][24];
  __shared__ __align__(16) bf16_t sx[4][128];
  uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32, tile = (blockIdx.x * 2 + warp) * 16;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  for (uint32_t stage = 0; stage < 2; ++stage)
    lossless_moe_prefetch<N, K, true>(
        packed[warp][stage], metas[warp][stage], nullptr, payload, metadata, 0, tile, stage, lane);
  for (uint32_t base = 0; base < K; base += 128) {
    uint32_t buf = (base / 128) % 2;
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
    lossless_moe_prefetch<N, K, true>(
        packed[warp][buf], metas[warp][buf], nullptr, payload, metadata, 0, tile, base / 128 + 2, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (lane % 4 < 2) {
    uint32_t row = tile + lane / 4, token = 2 * (lane % 4);
    out[token * N + row] = cast<bf16_t>(d0);
    out[(token + 1) * N + row] = cast<bf16_t>(d1);
    out[token * N + row + 8] = cast<bf16_t>(d2);
    out[(token + 1) * N + row + 8] = cast<bf16_t>(d3);
  }
}
struct LosslessHeadTc4 {
  /// Restore every weight bit and calculate all four full-vocabulary rows.
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView p, tvm::ffi::TensorView m, tvm::ffi::TensorView out) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({4, 4096}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({131072u * 4096u / 2u}).with_dtype<uint32_t>().with_device(dev).verify(p);
    TensorMatcher({131072u * 4096u / 256u}).with_dtype<uint64_t>().with_device(dev).verify(m);
    TensorMatcher({4, 131072}).with_dtype<bf16_t>().with_device(dev).verify(out);
    LaunchKernel(4096, 64, dev.unwrap())(
        lossless_head_tc4<131072, 4096>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const uint32_t*>(p.data_ptr()),
        static_cast<const uint64_t*>(m.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
  }
};
}  // namespace sglang
