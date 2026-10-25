#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct Add3Params {
  const bf16_t* __restrict__ a;  // [N] contiguous
  const bf16_t* __restrict__ b;  // [N] contiguous
  const bf16_t* __restrict__ c;  // [N] contiguous
  bf16_t* __restrict__ out;      // [N] contiguous
  int64_t n_vecs;                // N / kVecElems
};

template <bool kUsePDL, bool kPrefetchBC>
__global__ void add3_kernel(const __grid_constant__ Add3Params params) {
  constexpr uint32_t kVecPairs = device::kMaxVecBytes / sizeof(bf16x2_t);
  using vec_t = device::AlignedVector<bf16x2_t, kVecPairs>;

  const int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vid >= params.n_vecs) return;

  // Trigger early, so that the next kernel gets a chance to prefetch.
  device::PDLTriggerSecondary<kUsePDL>();

  vec_t a, b, c;
  if constexpr (kPrefetchBC) {
    b.load(params.b, vid);
    c.load(params.c, vid);
    device::PDLWaitPrimary<kUsePDL>();
    a.load(params.a, vid);
  } else {
    device::PDLWaitPrimary<kUsePDL>();
    a.load(params.a, vid);
    b.load(params.b, vid);
    c.load(params.c, vid);
  }

  vec_t out;
#pragma unroll
  for (uint32_t i = 0; i < kVecPairs; ++i) {
    out[i] = __hadd2(__hadd2(a[i], b[i]), c[i]);
  }
  out.store(params.out, vid);
}

template <bool kUsePDL>
struct Add3Kernel {
  static constexpr int64_t kVecElems = device::kMaxVecBytes / sizeof(bf16_t);

  static void launch(
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView c,
      const tvm::ffi::TensorView out,
      const bool prefetch_bc) {
    using namespace host;

    auto N = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N}).with_dtype<bf16_t>().with_device(device).verify(a).verify(b).verify(c).verify(out);
    const auto numel = N.unwrap();
    RuntimeCheck(numel % kVecElems == 0, "numel must be divisible by the vector width");
    if (numel == 0) return;
    const auto params = Add3Params{
        .a = static_cast<const bf16_t*>(a.data_ptr()),
        .b = static_cast<const bf16_t*>(b.data_ptr()),
        .c = static_cast<const bf16_t*>(c.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .n_vecs = numel / kVecElems,
    };
    const auto num_threads = [&]() -> int64_t {
      for (int64_t n : {128, 256, 512}) {
        if (params.n_vecs <= n * 256) return n;
      }
      return 512;
    }();
    const auto grid = div_ceil(params.n_vecs, num_threads);
    const auto kernel = prefetch_bc ? add3_kernel<kUsePDL, true> : add3_kernel<kUsePDL, false>;
    LaunchKernel(grid, num_threads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
