// Native CUDA candidate for the two SGLang diffusion rotary-embedding kernels,
// built through SGLang's jit_kernel / tvm-ffi stack (NOT torch.utils.cpp_extension).
// Mirrors the launcher/validation style of csrc/diffusion/qknorm_rope.cuh.
//
// v2 — optimized from the prior NCU bound diagnosis (SM/instruction-throughput,
// ~23% DRAM): per-thread runtime div/mod removed via fixed grid geometry + power-of-2
// shift/mask indexing; 128-bit vectorized bf16 loads/stores (AlignedVector<packed,4> =
// 8 bf16); standard cos/sin loaded once per token into shared memory and reused across
// all heads. Numerics preserved: standard does fp32 FMA and rounds only on the bf16
// store; LTX-2 keeps the intermediate (x*cos)->bf16 rounding before the fp32 sin term.
//
// Only the captured production buckets reach these kernels (the Python dispatcher gates
// to standard (1,27030,24,128)/D=128/half=64 and LTX-2 half in {32,64}); both have
// D and half divisible by 8 and half a power of two, so 8-wide vectorization and
// shift/mask indexing are always valid here. Compile flags follow the SGLang jit_kernel
// build (no --use_fast_math).

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kVecBf16 = 8;  // 8 bf16 per 128-bit (16-byte) access

// ---------------------------------------------------------------------------
// Standard adjacent-pair RoPE — one block per token, cos/sin reused via shared mem
// ---------------------------------------------------------------------------
struct StdRopeParams {
  void* __restrict__ out;
  const void* __restrict__ x;
  const float* __restrict__ cos;
  const float* __restrict__ sin;
  int64_t hidden;        // H * D (elements per token row)
  uint32_t num_tokens;   // cos/sin rows (== grid.x for B=1)
  uint32_t D;            // head_size (power of 2 for the gated path)
  uint32_t half;         // D / 2
};

template <typename DType>
__global__ void standard_rope_kernel(const StdRopeParams __grid_constant__ params) {
  using namespace device;
  using Packed = packed_t<DType>;
  using Vec = AlignedVector<Packed, kVecBf16 / 2>;  // 4 bf16x2 = 8 bf16 = 16 bytes

  extern __shared__ float smem[];  // [0,half)=cos, [half,2*half)=sin for this token
  const uint32_t token = blockIdx.x;
  const uint32_t token_for_cos = token % params.num_tokens;  // == token when B==1
  const uint32_t half = params.half;
  const uint32_t dmask = params.D - 1u;  // D is a power of 2

  const float* cos_row = params.cos + static_cast<int64_t>(token_for_cos) * half;
  const float* sin_row = params.sin + static_cast<int64_t>(token_for_cos) * half;
  for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
    smem[i] = cos_row[i];
    smem[half + i] = sin_row[i];
  }
  __syncthreads();

  const int64_t token_base = static_cast<int64_t>(token) * params.hidden;  // element offset
  const void* xrow = pointer::offset(params.x, token_base * static_cast<int64_t>(sizeof(DType)));
  void* orow = pointer::offset(params.out, token_base * static_cast<int64_t>(sizeof(DType)));
  const uint32_t nvec = static_cast<uint32_t>(params.hidden) / kVecBf16;

  for (uint32_t v = threadIdx.x; v < nvec; v += blockDim.x) {
    const uint32_t e = v * kVecBf16;          // element offset within the token row
    const uint32_t i0 = (e & dmask) >> 1u;    // first pair-index of this 8-elem group
    Vec xv = load_as<Vec>(xrow, v);
#pragma unroll
    for (uint32_t k = 0; k < kVecBf16 / 2; ++k) {
      const auto [x1, x2] = cast<fp32x2_t>(xv[k]);  // pair (2(i0+k), 2(i0+k)+1)
      const float c = smem[i0 + k];
      const float s = smem[half + i0 + k];
      const float o1 = fmaf(-x2, s, x1 * c);
      const float o2 = fmaf(x1, s, x2 * c);
      xv[k] = cast<Packed, fp32x2_t>({o1, o2});
    }
    store_as<Vec>(orow, xv, v);
  }
}

template <typename DType>
struct StandardRopeKernel {
  static void run(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView cos,
      tvm::ffi::TensorView sin) {
    using namespace host;

    auto N = SymbolicSize{"N"};
    auto H = SymbolicSize{"H"};
    auto D = SymbolicSize{"D"};
    auto T = SymbolicSize{"T"};
    auto Half = SymbolicSize{"half"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, H, D}).with_dtype<DType>().with_device(device).verify(x).verify(out);
    TensorMatcher({T, Half}).with_dtype<float>().with_device(device).verify(cos).verify(sin);

    const int64_t n = N.unwrap();
    const auto h = static_cast<uint32_t>(H.unwrap());
    const auto d = static_cast<uint32_t>(D.unwrap());
    const auto t = static_cast<uint32_t>(T.unwrap());
    const auto half = static_cast<uint32_t>(Half.unwrap());
    RuntimeCheck(d % 2 == 0 && d == 2 * half, "head_size must be even and equal 2*cos width");
    RuntimeCheck((d & 7u) == 0, "head_size must be divisible by 8 for the vectorized path");
    RuntimeCheck(t != 0 && n % t == 0, "N must be a positive multiple of num_tokens");

    const int64_t hidden = static_cast<int64_t>(h) * d;
    const auto params = StdRopeParams{
        .out = out.data_ptr(),
        .x = x.data_ptr(),
        .cos = static_cast<const float*>(cos.data_ptr()),
        .sin = static_cast<const float*>(sin.data_ptr()),
        .hidden = hidden,
        .num_tokens = t,
        .D = d,
        .half = half,
    };
    const auto grid = static_cast<uint32_t>(n);  // one block per token row
    const std::size_t smem_bytes = static_cast<std::size_t>(2u * half) * sizeof(float);
    LaunchKernel(grid, kThreadsPerBlock, device.unwrap(), smem_bytes)(standard_rope_kernel<DType>, params);
  }
};

// ---------------------------------------------------------------------------
// LTX-2 split-half RoPE — one block per (b,s); vectorize along the contiguous j dim
// ---------------------------------------------------------------------------
struct Ltx2RopeParams {
  void* __restrict__ out;
  const void* __restrict__ x;
  const void* __restrict__ cos;
  const void* __restrict__ sin;
  int64_t x_outer;       // inner = H * D (elements between (b,s) rows of x)
  int64_t cos_stride_b;
  int64_t cos_stride_h;
  int64_t cos_stride_s;  // cos/sin last-dim stride is 1
  int64_t sin_stride_b;
  int64_t sin_stride_h;
  int64_t sin_stride_s;
  uint32_t S;
  uint32_t H;
  uint32_t half;
  uint32_t D;            // 2 * half
};

template <typename DType>
__global__ void ltx2_split_rope_kernel(const Ltx2RopeParams __grid_constant__ params) {
  using namespace device;
  using Packed = packed_t<DType>;
  using Vec = AlignedVector<Packed, kVecBf16 / 2>;  // 8 bf16

  const uint32_t S = params.S;
  const uint32_t b = blockIdx.x / S;   // == 0 when B==1 (one div per block)
  const uint32_t s = blockIdx.x - b * S;
  const uint32_t H = params.H;
  const uint32_t half = params.half;
  const uint32_t vpg = half / kVecBf16;     // vec-groups per head along j (power of 2)
  const uint32_t vpg_mask = vpg - 1u;
  const uint32_t vpg_shift = static_cast<uint32_t>(__ffs(static_cast<int>(vpg)) - 1);
  const uint32_t nvec = H * vpg;

  const int64_t row_base = (static_cast<int64_t>(b) * S + s) * params.x_outer;
  const int64_t c_bs = b * params.cos_stride_b + static_cast<int64_t>(s) * params.cos_stride_s;
  const int64_t s_bs = b * params.sin_stride_b + static_cast<int64_t>(s) * params.sin_stride_s;
  const int64_t esz = static_cast<int64_t>(sizeof(DType));

  for (uint32_t v = threadIdx.x; v < nvec; v += blockDim.x) {
    const uint32_t h = v >> vpg_shift;
    const uint32_t jg = v & vpg_mask;
    const uint32_t j0 = jg * kVecBf16;

    const int64_t xf_base = row_base + static_cast<int64_t>(h) * params.D + j0;  // first half
    const int64_t xs_base = xf_base + half;                                       // second half
    const int64_t c_base = c_bs + static_cast<int64_t>(h) * params.cos_stride_h + j0;
    const int64_t s_base = s_bs + static_cast<int64_t>(h) * params.sin_stride_h + j0;

    Vec xf = load_as<Vec>(pointer::offset(params.x, xf_base * esz), 0);
    Vec xs = load_as<Vec>(pointer::offset(params.x, xs_base * esz), 0);
    Vec cv = load_as<Vec>(pointer::offset(params.cos, c_base * esz), 0);
    Vec sv = load_as<Vec>(pointer::offset(params.sin, s_base * esz), 0);
#pragma unroll
    for (uint32_t k = 0; k < kVecBf16 / 2; ++k) {
      const auto [xf1, xf2] = cast<fp32x2_t>(xf[k]);  // two consecutive, independent j
      const auto [xs1, xs2] = cast<fp32x2_t>(xs[k]);
      const auto [c1, c2] = cast<fp32x2_t>(cv[k]);
      const auto [s1, s2] = cast<fp32x2_t>(sv[k]);
      // Intermediate (x*cos) rounded to DType before the fp32 sin term.
      const float of1 = cast<fp32_t>(cast<DType>(xf1 * c1)) - xs1 * s1;
      const float of2 = cast<fp32_t>(cast<DType>(xf2 * c2)) - xs2 * s2;
      const float og1 = cast<fp32_t>(cast<DType>(xs1 * c1)) + xf1 * s1;
      const float og2 = cast<fp32_t>(cast<DType>(xs2 * c2)) + xf2 * s2;
      xf[k] = cast<Packed, fp32x2_t>({of1, of2});  // reuse loaded vectors for output
      xs[k] = cast<Packed, fp32x2_t>({og1, og2});
    }
    store_as<Vec>(pointer::offset(params.out, xf_base * esz), xf, 0);
    store_as<Vec>(pointer::offset(params.out, xs_base * esz), xs, 0);
  }
}

template <typename DType>
struct Ltx2SplitRopeKernel {
  static void run(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView cos,
      tvm::ffi::TensorView sin) {
    using namespace host;

    auto B = SymbolicSize{"B"};
    auto S = SymbolicSize{"S"};
    auto Inner = SymbolicSize{"inner"};
    auto Hh = SymbolicSize{"num_heads"};
    auto Half = SymbolicSize{"half"};
    auto Sb = SymbolicSize{"cos_stride_b"};
    auto Sh = SymbolicSize{"cos_stride_h"};
    auto Ss = SymbolicSize{"cos_stride_s"};
    auto SbS = SymbolicSize{"sin_stride_b"};
    auto ShS = SymbolicSize{"sin_stride_h"};
    auto SsS = SymbolicSize{"sin_stride_s"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, S, Inner}).with_dtype<DType>().with_device(device).verify(x).verify(out);
    TensorMatcher({B, Hh, S, Half}).with_strides({Sb, Sh, Ss, 1}).with_dtype<DType>().with_device(device).verify(cos);
    TensorMatcher({B, Hh, S, Half}).with_strides({SbS, ShS, SsS, 1}).with_dtype<DType>().with_device(device).verify(sin);

    const int64_t b = B.unwrap();
    const auto s = static_cast<uint32_t>(S.unwrap());
    const int64_t inner = Inner.unwrap();
    const auto hh = static_cast<uint32_t>(Hh.unwrap());
    const auto half = static_cast<uint32_t>(Half.unwrap());
    const auto d = 2u * half;
    RuntimeCheck(inner == static_cast<int64_t>(hh) * d, "inner_dim must equal num_heads * 2 * half");
    RuntimeCheck((half & 7u) == 0, "half must be divisible by 8 for the vectorized path");

    const int64_t cs_b = Sb.has_value() ? Sb.unwrap() : 0;  // size-1 batch stride may be skipped
    const int64_t sn_b = SbS.has_value() ? SbS.unwrap() : 0;
    const auto params = Ltx2RopeParams{
        .out = out.data_ptr(),
        .x = x.data_ptr(),
        .cos = cos.data_ptr(),
        .sin = sin.data_ptr(),
        .x_outer = inner,
        .cos_stride_b = cs_b,
        .cos_stride_h = Sh.unwrap(),
        .cos_stride_s = Ss.unwrap(),
        .sin_stride_b = sn_b,
        .sin_stride_h = ShS.unwrap(),
        .sin_stride_s = SsS.unwrap(),
        .S = s,
        .H = hh,
        .half = half,
        .D = d,
    };
    const auto grid = static_cast<uint32_t>(b * s);  // one block per (b, s)
    LaunchKernel(grid, kThreadsPerBlock, device.unwrap())(ltx2_split_rope_kernel<DType>, params);
  }
};

}  // namespace
