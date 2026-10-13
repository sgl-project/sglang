#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// Router-gate logits for decode-sized M: out[m, e] = sum_k x[m, k] * w[e, k]
// in fp32, for a bf16 activation and an fp32 or bf16 gate weight.
//
// The gate is a few MB that every MoE layer reads cold (the experts flush L2
// in between), so the kernel is shaped to have the whole weight in flight at
// once: each block owns kRows expert rows and every thread issues all of its
// 16-byte weight loads for those rows, plus the matching L2-resident
// activation chunks, before consuming any of them. One shuffle +
// shared-memory reduce per block finishes the kRows x M outputs. Summation
// order is fixed by (kRows, kThreads), so every TP rank computes identical
// logits. Staging the activation in shared memory first was measured slower
// (7.65 vs 5.87 us on the 4 MB bf16 BailingMoeV3 gate): the staging pass is a
// serialized L2 round trip before any weight byte is requested.

constexpr uint32_t kGateThreads = 256;

template <typename WT>
struct GateVec;

template <>
struct GateVec<fp32_t> {
  static constexpr uint32_t kElems = 4;  // one 16-byte load
  __device__ __forceinline__ static void load(const fp32_t* p, float (&v)[8]) {
    const float4 f = __ldcs(reinterpret_cast<const float4*>(p));
    v[0] = f.x;
    v[1] = f.y;
    v[2] = f.z;
    v[3] = f.w;
  }
};

template <>
struct GateVec<bf16_t> {
  static constexpr uint32_t kElems = 8;
  __device__ __forceinline__ static void load(const bf16_t* p, float (&v)[8]) {
    const uint4 u = __ldcs(reinterpret_cast<const uint4*>(p));
    const bf16x2_t* h = reinterpret_cast<const bf16x2_t*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto [a, b] = cast<fp32x2_t>(h[i]);
      v[2 * i] = a;
      v[2 * i + 1] = b;
    }
  }
};

template <typename WT, uint32_t kMaxM, uint32_t kRows>
__global__ void __launch_bounds__(kGateThreads) router_gate_matvec_kernel(
    float* __restrict__ out,
    const bf16_t* __restrict__ x,
    const WT* __restrict__ w,
    const uint32_t M,
    const uint32_t K,
    const uint32_t E) {
  constexpr uint32_t kWarps = kGateThreads / 32;
  constexpr uint32_t kVec = GateVec<WT>::kElems;
  __shared__ float sred[kWarps * kRows * kMaxM];

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t row0 = blockIdx.x * kRows;

  const uint32_t chunks = K / kVec;  // host-checked multiple of kGateThreads
  const uint32_t per_thread = chunks / kGateThreads;
  float acc[kRows][kMaxM];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t m = 0; m < kMaxM; ++m) {
      acc[r][m] = 0.0f;
    }
  }
  // Every weight load of the block issued before the first FMA: the loop is
  // fully unrolled for the shipped K, so the compiler hoists them.
  for (uint32_t i = 0; i < per_thread; ++i) {
    const uint32_t k = (tid + i * kGateThreads) * kVec;
    float wv[kRows][8];
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const uint32_t row = min(row0 + r, E - 1);
      GateVec<WT>::load(w + static_cast<size_t>(row) * K + k, wv[r]);
    }
#pragma unroll
    for (uint32_t m = 0; m < kMaxM; ++m) {
      float xv[8];
      // The activation is L2-resident (a few tens of KB); rows past M read
      // row 0 and are discarded at the store, so every lane issues the same
      // loads and nothing waits on a branch.
      const bf16_t* xrow = x + static_cast<size_t>(min(m, M - 1)) * K + k;
      if constexpr (kVec == 8) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(xrow));
        const bf16x2_t* h = reinterpret_cast<const bf16x2_t*>(&u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const auto [a, b] = cast<fp32x2_t>(h[j]);
          xv[2 * j] = a;
          xv[2 * j + 1] = b;
        }
      } else {
        const uint2 u = __ldg(reinterpret_cast<const uint2*>(xrow));
        const bf16x2_t* h = reinterpret_cast<const bf16x2_t*>(&u);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const auto [a, b] = cast<fp32x2_t>(h[j]);
          xv[2 * j] = a;
          xv[2 * j + 1] = b;
        }
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        float p = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < kVec; ++j) {
          p = fmaf(wv[r][j], xv[j], p);
        }
        acc[r][m] += p;
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t m = 0; m < kMaxM; ++m) {
#pragma unroll
      for (uint32_t off = 16; off > 0; off >>= 1) {
        acc[r][m] += __shfl_xor_sync(0xffffffff, acc[r][m], off);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
      for (uint32_t m = 0; m < kMaxM; ++m) {
        sred[(warp * kRows + r) * kMaxM + m] = acc[r][m];
      }
    }
  }
  __syncthreads();
  if (tid < kRows * kMaxM) {
    const uint32_t r = tid / kMaxM;
    const uint32_t m = tid % kMaxM;
    const uint32_t row = row0 + r;
    if (m < M && row < E) {
      float s = 0.0f;
#pragma unroll
      for (uint32_t wi = 0; wi < kWarps; ++wi) {
        s += sred[(wi * kRows + r) * kMaxM + m];
      }
      out[static_cast<size_t>(m) * E + row] = s;
    }
  }
}

template <typename WT, uint32_t kMaxM, uint32_t kRows>
struct RouterGateMatvecKernel {
  static_assert(kMaxM >= 1 && kMaxM <= 8, "kMaxM out of range");
  static_assert(kRows >= 1 && kRows <= 8, "kRows out of range");

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"M"};
    auto E = SymbolicSize{"E"};
    auto K = SymbolicSize{"K"};
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({E, K}).with_dtype<WT>().with_device(device).verify(w);
    TensorMatcher({M, E}).with_dtype<fp32_t>().with_device(device).verify(out);

    const auto m = static_cast<uint32_t>(M.unwrap());
    const auto e = static_cast<uint32_t>(E.unwrap());
    const auto k = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(m >= 1 && m <= kMaxM, "M must be in [1, kMaxM]");
    RuntimeCheck(k % (kGateThreads * GateVec<WT>::kElems) == 0, "K must be a multiple of the block chunk");
    RuntimeCheck(e % kRows == 0, "E must be a multiple of kRows");

    LaunchKernel(e / kRows, kGateThreads, device.unwrap())(
        router_gate_matvec_kernel<WT, kMaxM, kRows>,
        static_cast<float*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const WT*>(w.data_ptr()),
        m,
        k,
        e);
  }
};

}  // namespace sglang
