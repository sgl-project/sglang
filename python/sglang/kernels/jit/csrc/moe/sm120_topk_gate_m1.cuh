#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// Single-token (M=1) fused sigmoid+bias top-k router for wide expert counts.
// One block; selection on biased scores with lowest-expert-id tie-break,
// weights from the unbiased sigmoid, optional renormalize and routed
// scaling — the exact semantics of the generic Triton router, specialized
// for the bs=1 decode hot path.

__device__ __forceinline__ uint32_t f32_sortable(float v) {
  uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <uint32_t N, uint32_t kTopK, bool kRenorm, bool kApplyScale, bool kUsePDL>
__global__ void __launch_bounds__(1024) sm120_topk_gate_m1_kernel(
    float* __restrict__ out_w,         // [kTopK]
    int32_t* __restrict__ out_i,       // [kTopK]
    const float* __restrict__ scores,  // [N] raw router logits (fp32)
    const float* __restrict__ bias,    // [N]
    const float routed_scaling) {
  constexpr uint32_t kThreads = 1024;
  static_assert(N <= kThreads, "one score per thread");
  __shared__ float s_act[N];
  __shared__ float s_biased[N];
  __shared__ unsigned long long s_wmax[kThreads / 32];
  __shared__ float s_sel[kTopK];
  __shared__ int32_t s_idx[kTopK];

  PDLTriggerSecondary<kUsePDL>();
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  if (tid < N) {
    const float sc = scores[tid];
    const float act = 1.0f / (1.0f + expf(-sc));
    s_act[tid] = act;
    const float biased = act + bias[tid];
    s_biased[tid] = isnan(biased) ? -1e30f : biased;
  }
  __syncthreads();

  const uint32_t lane = tid % 32;
  const uint32_t warp = tid / 32;

  bool remaining = tid < N;
  for (uint32_t k = 0; k < kTopK; ++k) {
    // pack (biased value desc, id asc) into u64 and block-max it
    float v = (tid < N) ? s_biased[tid] : -__int_as_float(0x7f800000);
    unsigned long long key = (static_cast<unsigned long long>(f32_sortable(v)) << 32) | (0xFFFFFFFFu - tid);
    key = remaining ? key : 0ull;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
      const unsigned long long other = __shfl_down_sync(0xffffffff, key, off);
      key = other > key ? other : key;
    }
    if (lane == 0) {
      s_wmax[warp] = key;
    }
    __syncthreads();
    if (warp == 0) {
      unsigned long long bkey = (lane < kThreads / 32) ? s_wmax[lane] : 0ull;
#pragma unroll
      for (uint32_t off = 16; off > 0; off >>= 1) {
        const unsigned long long other = __shfl_down_sync(0xffffffff, bkey, off);
        bkey = other > bkey ? other : bkey;
      }
      if (lane == 0) {
        const uint32_t win = 0xFFFFFFFFu - static_cast<uint32_t>(bkey & 0xFFFFFFFFu);
        s_idx[k] = static_cast<int32_t>(win);
        s_sel[k] = s_act[win];
      }
    }
    __syncthreads();
    remaining = remaining && tid != static_cast<uint32_t>(s_idx[k]);
  }

  if (tid == 0) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      sum += s_sel[k];
    }
    const float norm = kRenorm ? (sum > 0.0f ? 1.0f / sum : 1.0f) : 1.0f;
    const float scale = kApplyScale ? routed_scaling : 1.0f;
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      out_w[k] = s_sel[k] * norm * scale;
      out_i[k] = s_idx[k];
    }
  }
}

template <uint32_t N, uint32_t kTopK, bool kRenorm, bool kApplyScale, bool kUsePDL>
struct Sm120TopkGateM1Kernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      double routed_scaling) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<fp32_t>().with_device(device).verify(scores);
    TensorMatcher({N}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({kTopK}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({kTopK}).with_dtype<int32_t>().with_device(device).verify(out_i);
    LaunchKernel(1, 1024, device.unwrap())
        .enable_pdl(kUsePDL)(
            sm120_topk_gate_m1_kernel<N, kTopK, kRenorm, kApplyScale, kUsePDL>,
            static_cast<float*>(out_w.data_ptr()),
            static_cast<int32_t*>(out_i.data_ptr()),
            static_cast<const float*>(scores.data_ptr()),
            static_cast<const float*>(bias.data_ptr()),
            static_cast<float>(routed_scaling));
  }
};

}  // namespace sglang
