// Latency-lean rel_logits projection for SMALL token counts:
// out[t, h, :] = bf16(sum_d fp32(r[t, h, d]) * fp32(proj[d, :])) with an
// optional per-token tau prescale folded in registers (the shipped prescale
// semantics: r*tau rounds to bf16 BEFORE the dot, matching
// {row_scale -> einsum} exactly).
//
// At t=1 the cuBLAS GEMM ([16,16]@[16,1024]) is pure launch + entry overhead
// (~1.6 us for ~64 KB of traffic); this kernel is a no-smem no-sync grid of
// independent 8-wide dots reading proj straight from L2 (32 KB, hot across
// decode steps), so its floor is the launch itself. An earlier smem-staged
// bandwidth-oriented kernel lost to cuBLAS at EVERY size -- this one is only
// dispatched inside its measured small-t band; large t stays on cuBLAS.

#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/type.cuh>   // For bf16_t/fp32_t aliases
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector (16B loads)

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kRpVec = 8;  // bf16x8 = 16 B
constexpr uint32_t kRpBlock = 256;

template <int kDRel, bool kUsePDL, bool kHasTau>
__global__ __launch_bounds__(kRpBlock, 1) void rel_proj_small_t_kernel(
    const bf16_t* __restrict__ r,    // [t, h, kDRel], token rows strided
    const fp32_t* __restrict__ tau,  // [t]; unread when !kHasTau
    const bf16_t* __restrict__ proj, // [kDRel, e] contiguous
    bf16_t* __restrict__ out,        // [t, h, e] contiguous
    const int64_t r_stride_t,        // elems between token rows
    const uint32_t h,
    const uint32_t e,
    const uint32_t t) {
  using namespace device;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t evecs = e / kRpVec;
  const uint32_t total = t * h * evecs;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const uint32_t ev = idx % evecs;
    const uint32_t th = idx / evecs;
    const uint32_t ti = th / h;
    const uint32_t hi = th % h;

    // r[ti, hi, :] once into registers (2x 16B for kDRel=16), tau folded
    // with the prescale rounding (bf16 round before the dot).
    const bf16_t* rrow = r + static_cast<int64_t>(ti) * r_stride_t +
                         static_cast<int64_t>(hi) * kDRel;
    float rv[kDRel];
#pragma unroll
    for (int d = 0; d < kDRel; d += static_cast<int>(kRpVec)) {
      AlignedVector<bf16_t, kRpVec> a;
      a.load(rrow, d / static_cast<int>(kRpVec));
#pragma unroll
      for (int k = 0; k < static_cast<int>(kRpVec); ++k) {
        if constexpr (kHasTau) {
          rv[d + k] = static_cast<float>(
              static_cast<bf16_t>(static_cast<float>(a[k]) * tau[ti]));
        } else {
          rv[d + k] = static_cast<float>(a[k]);
        }
      }
    }

    float acc[kRpVec] = {};
#pragma unroll
    for (int d = 0; d < kDRel; ++d) {
      AlignedVector<bf16_t, kRpVec> p;
      p.load(proj + static_cast<int64_t>(d) * e, ev);
#pragma unroll
      for (int k = 0; k < static_cast<int>(kRpVec); ++k) {
        acc[k] += rv[d] * static_cast<float>(p[k]);
      }
    }

    AlignedVector<bf16_t, kRpVec> o;
#pragma unroll
    for (int k = 0; k < static_cast<int>(kRpVec); ++k) {
      o[k] = static_cast<bf16_t>(acc[k]);
    }
    o.store(out, idx);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <int kDRel, bool kUsePDL>
void rel_proj_small_t(
    tvm::ffi::TensorView r,
    tvm::ffi::TensorView tau,  // numel-0 sentinel = no prescale
    tvm::ffi::TensorView proj,
    tvm::ffi::TensorView out) {
  using namespace host;
  auto T = SymbolicSize{"t"};
  auto H = SymbolicSize{"h"};
  auto D = SymbolicSize{"d_rel"};
  auto E = SymbolicSize{"e"};
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();

  TensorMatcher({T, H, D})
      .with_dtype<bf16_t>()
      .with_device(dev)
      .with_strides({-1, D, 1})
      .verify(r);
  TensorMatcher({D, E}).with_dtype<bf16_t>().with_device(dev).verify(proj);
  TensorMatcher({T, H, E}).with_dtype<bf16_t>().with_device(dev).verify(out);

  const uint32_t t = static_cast<uint32_t>(T.unwrap());
  const uint32_t h = static_cast<uint32_t>(H.unwrap());
  const uint32_t e = static_cast<uint32_t>(E.unwrap());
  RuntimeCheck(D.unwrap() == kDRel, "d_rel must be ", kDRel);
  static_assert(kDRel % static_cast<int>(kRpVec) == 0,
                "d_rel must be a vector multiple (r loads are 16B)");
  RuntimeCheck(e % kRpVec == 0, "e must be a multiple of ", kRpVec);
  RuntimeCheck((r.stride(0) * 2) % 16 == 0, "r token stride must keep 16B alignment");
  RuntimeCheck(std::bit_cast<intptr_t>(r.data_ptr()) % 16 == 0, "r not 16B aligned");
  RuntimeCheck(std::bit_cast<intptr_t>(proj.data_ptr()) % 16 == 0, "proj not 16B aligned");

  const bool has_tau = tau.numel() > 0;
  if (has_tau) {
    TensorMatcher({T}).with_dtype<fp32_t>().with_device(dev).verify(tau);
  }

  const uint32_t total = t * h * (e / kRpVec);
  const uint32_t grid = div_ceil(total, kRpBlock);
  auto launch = [&](auto kernel) {
    LaunchKernel(grid, kRpBlock, dev.unwrap()).enable_pdl(kUsePDL)(
        kernel,
        static_cast<const bf16_t*>(r.data_ptr()),
        has_tau ? static_cast<const fp32_t*>(tau.data_ptr()) : nullptr,
        static_cast<const bf16_t*>(proj.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        r.stride(0), h, e, t);
  };
  if (has_tau) {
    launch(rel_proj_small_t_kernel<kDRel, kUsePDL, true>);
  } else {
    launch(rel_proj_small_t_kernel<kDRel, kUsePDL, false>);
  }
}

}  // namespace
