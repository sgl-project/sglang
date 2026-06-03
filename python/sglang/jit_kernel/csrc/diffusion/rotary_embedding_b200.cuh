// Native CUDA kernels for the two SGLang diffusion rotary-embedding entry points,
// built and exported through SGLang's jit_kernel / tvm-ffi stack (header-only,
// templated launcher mirroring csrc/diffusion/qknorm_rope.cuh).
//
//   * StandardRotaryKernel<head_dim, use_pdl, DType>::run(out, x, cos, sin)
//       Adjacent-pair RoPE. out-of-place. x is (rows, heads, head_dim); cos/sin
//       are (tokens, head_dim/2) fp32. Math is done in fp32 then rounded back to
//       DType, matching the Triton baseline (o1 = x1*cos - x2*sin via fma,
//       o2 = x2*cos + x1*sin via fma).
//
//   * Ltx2SplitRotaryKernel<half_dim, use_pdl, DType>::run(out, x, cos, sin)
//       Split-half RoPE. out-of-place. x is (batch, seq, heads*2*half_dim);
//       cos/sin are (batch, heads, seq, half_dim) DType and may be
//       non-contiguous (indexed via the passed strides). The (x*cos) term is
//       rounded to DType BEFORE the fp32 sine term is added, matching the
//       PyTorch addcmul_ op order that the Triton baseline reproduces.

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <type_traits>

namespace kda_diffusion_rotary {

constexpr uint32_t kThreadsPerBlock = 256;

// 16-byte aligned vector for 128-bit (LDG.128/STG.128) coalesced access.
template <typename T, int N>
struct alignas(sizeof(T) * N) AVec {
  T d[N];
};

// ============================================================================
// Standard adjacent-pair RoPE (apply_rotary_embedding, interleaved=False path)
// ============================================================================

struct StandardRotaryParams {
  void* __restrict__ out;        // (rows, heads, head_dim) DType, contiguous
  const void* __restrict__ x;    // (rows, heads, head_dim) DType, contiguous
  const float* __restrict__ cos;  // (tokens, head_dim/2) fp32
  const float* __restrict__ sin;  // (tokens, head_dim/2) fp32
  int64_t x_row_stride;          // elements between consecutive rows (= heads*head_dim)
  int64_t cos_row_stride;        // elements between consecutive token rows (= head_dim/2)
  uint32_t num_rows;             // rows = batch * tokens
  uint32_t num_tokens;           // tokens per batch (cos/sin row index = row % num_tokens)
  uint32_t num_heads;
};

template <int kHeadDim, bool kUsePDL, typename DType>
__global__ void standard_rotary_kernel(const StandardRotaryParams __grid_constant__ params) {
  using namespace device;
  static_assert(kHeadDim % 2 == 0, "head_dim must be even");
  constexpr int kHalf = kHeadDim / 2;

  PDLWaitPrimary<kUsePDL>();

  const uint32_t row = blockIdx.x;
  if (row >= params.num_rows) {
    PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const uint32_t token = (params.num_tokens != 0u) ? (row % params.num_tokens) : row;

  const DType* x_row = static_cast<const DType*>(params.x) + static_cast<int64_t>(row) * params.x_row_stride;
  DType* out_row = static_cast<DType*>(params.out) + static_cast<int64_t>(row) * params.x_row_stride;
  const float* cos_row = params.cos + static_cast<int64_t>(token) * params.cos_row_stride;
  const float* sin_row = params.sin + static_cast<int64_t>(token) * params.cos_row_stride;

  // 128-bit vectorized over contiguous head_dim: each thread owns kPairsPerVec
  // adjacent pairs (2*kPairsPerVec elements) loaded/stored in one transaction.
  // cos/sin (fp32) are loaded directly per thread (vectorized); the small per-token
  // cos/sin row is reused across heads through L2, so no shared-memory barrier is
  // needed -- dropping __syncthreads improves memory-level parallelism.
  constexpr int kPairsPerVec = (kHalf % 4 == 0) ? 4 : ((kHalf % 2 == 0) ? 2 : 1);
  constexpr int kVecPerHead = kHalf / kPairsPerVec;
  using V = AVec<DType, 2 * kPairsPerVec>;
  using CV = AVec<float, kPairsPerVec>;

  const uint32_t total_vec = params.num_heads * static_cast<uint32_t>(kVecPerHead);
  for (uint32_t pv = threadIdx.x; pv < total_vec; pv += blockDim.x) {
    const uint32_t h = pv / static_cast<uint32_t>(kVecPerHead);
    const uint32_t i0 = (pv % static_cast<uint32_t>(kVecPerHead)) * kPairsPerVec;
    const DType* xh = x_row + static_cast<int64_t>(h) * kHeadDim;
    DType* oh = out_row + static_cast<int64_t>(h) * kHeadDim;

    const V v = *reinterpret_cast<const V*>(xh + 2 * i0);
    const CV cvec = *reinterpret_cast<const CV*>(cos_row + i0);
    const CV svec = *reinterpret_cast<const CV*>(sin_row + i0);
    V vo;
#pragma unroll
    for (int k = 0; k < kPairsPerVec; ++k) {
      const float x1 = static_cast<float>(v.d[2 * k]);
      const float x2 = static_cast<float>(v.d[2 * k + 1]);
      const float c = cvec.d[k];
      const float s = svec.d[k];
      vo.d[2 * k] = static_cast<DType>(fmaf(-x2, s, x1 * c));      // x1*cos - x2*sin
      vo.d[2 * k + 1] = static_cast<DType>(fmaf(x1, s, x2 * c));   // x2*cos + x1*sin
    }
    *reinterpret_cast<V*>(oh + 2 * i0) = vo;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kHeadDim, bool kUsePDL, typename DType>
struct StandardRotaryKernel {
  static_assert(kHeadDim <= 1024, "head_dim too large for the shared cos/sin cache");
  static constexpr auto kernel = standard_rotary_kernel<kHeadDim, kUsePDL, DType>;

  static void run(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView cos,
      tvm::ffi::TensorView sin) {
    using namespace host;

    auto N = SymbolicSize{"rows"};
    auto H = SymbolicSize{"heads"};
    auto Xs = SymbolicSize{"x_row_stride"};
    auto T = SymbolicSize{"tokens"};
    auto Cs = SymbolicSize{"cos_row_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, H, kHeadDim}).with_strides({Xs, kHeadDim, 1}).with_dtype<DType>().with_device(device).verify(x);
    TensorMatcher({N, H, kHeadDim}).with_strides({Xs, kHeadDim, 1}).with_dtype<DType>().with_device(device).verify(out);
    TensorMatcher({T, kHeadDim / 2}).with_strides({Cs, 1}).with_dtype<float>().with_device(device).verify(cos).verify(sin);

    const auto params = StandardRotaryParams{
        .out = out.data_ptr(),
        .x = x.data_ptr(),
        .cos = static_cast<const float*>(cos.data_ptr()),
        .sin = static_cast<const float*>(sin.data_ptr()),
        .x_row_stride = static_cast<int64_t>(Xs.unwrap()),
        .cos_row_stride = static_cast<int64_t>(Cs.unwrap()),
        .num_rows = static_cast<uint32_t>(N.unwrap()),
        .num_tokens = static_cast<uint32_t>(T.unwrap()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
    };
    const uint32_t num_blocks = params.num_rows;
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

// ============================================================================
// LTX-2 split-half RoPE (apply_ltx2_split_rotary_emb)
// ============================================================================

struct Ltx2RotaryParams {
  void* __restrict__ out;        // (batch, seq, heads*2*half) DType, contiguous
  const void* __restrict__ x;    // (batch, seq, heads*2*half) DType, contiguous
  const void* __restrict__ cos;  // (batch, heads, seq, half) DType, strided
  const void* __restrict__ sin;  // (batch, heads, seq, half) DType, strided
  int64_t x_batch_stride;
  int64_t x_seq_stride;          // = heads*2*half (inner) for contiguous x
  int64_t cos_b, cos_h, cos_s;   // cos strides (elements); inner stride = 1
  int64_t sin_b, sin_h, sin_s;
  uint32_t batch;
  uint32_t seq_len;
  uint32_t num_heads;
};

template <int kHalf, bool kUsePDL, typename DType>
__global__ void ltx2_split_rotary_kernel(const Ltx2RotaryParams __grid_constant__ params) {
  using namespace device;
  constexpr int kHeadDim = 2 * kHalf;

  PDLWaitPrimary<kUsePDL>();

  const uint32_t row = blockIdx.x;  // row = b*seq_len + s
  const uint32_t total_rows = params.batch * params.seq_len;
  if (row >= total_rows) {
    PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const uint32_t b = row / params.seq_len;
  const uint32_t s = row - b * params.seq_len;

  const DType* x_row =
      static_cast<const DType*>(params.x) + static_cast<int64_t>(b) * params.x_batch_stride + static_cast<int64_t>(s) * params.x_seq_stride;
  DType* out_row =
      static_cast<DType*>(params.out) + static_cast<int64_t>(b) * params.x_batch_stride + static_cast<int64_t>(s) * params.x_seq_stride;
  const DType* cos_base =
      static_cast<const DType*>(params.cos) + static_cast<int64_t>(b) * params.cos_b + static_cast<int64_t>(s) * params.cos_s;
  const DType* sin_base =
      static_cast<const DType*>(params.sin) + static_cast<int64_t>(b) * params.sin_b + static_cast<int64_t>(s) * params.sin_s;

  // 128-bit vectorized over the contiguous inner (half) dimension. The first
  // half, second half, cos and sin rows are each contiguous in `j`, so a whole
  // vector of kVec elements is loaded/stored in one coalesced transaction.
  constexpr int kVec = (kHalf % 8 == 0) ? 8 : ((kHalf % 4 == 0) ? 4 : ((kHalf % 2 == 0) ? 2 : 1));
  constexpr int kVecPerHead = kHalf / kVec;
  using V = AVec<DType, kVec>;

  const uint32_t total_vec = params.num_heads * static_cast<uint32_t>(kVecPerHead);
  for (uint32_t pv = threadIdx.x; pv < total_vec; pv += blockDim.x) {
    const uint32_t h = pv / static_cast<uint32_t>(kVecPerHead);
    const uint32_t j0 = (pv % static_cast<uint32_t>(kVecPerHead)) * kVec;
    const DType* xh = x_row + static_cast<int64_t>(h) * kHeadDim;
    DType* oh = out_row + static_cast<int64_t>(h) * kHeadDim;
    const DType* cos_h_ptr = cos_base + static_cast<int64_t>(h) * params.cos_h;
    const DType* sin_h_ptr = sin_base + static_cast<int64_t>(h) * params.sin_h;

    const V va = *reinterpret_cast<const V*>(xh + j0);          // x_first[j0:j0+kVec]
    const V vb = *reinterpret_cast<const V*>(xh + kHalf + j0);  // x_second[j0:j0+kVec]
    const V vc = *reinterpret_cast<const V*>(cos_h_ptr + j0);
    const V vs = *reinterpret_cast<const V*>(sin_h_ptr + j0);
    V vo1, vo2;
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const float fa = static_cast<float>(va.d[k]);
      const float fb = static_cast<float>(vb.d[k]);
      const float fc = static_cast<float>(vc.d[k]);
      const float fs = static_cast<float>(vs.d[k]);
      // Round (x * cos) to DType BEFORE adding the fp32 sine term (matches the
      // PyTorch addcmul_ op order that the Triton baseline reproduces).
      const float cf1 = static_cast<float>(static_cast<DType>(fa * fc));
      const float cf2 = static_cast<float>(static_cast<DType>(fb * fc));
      vo1.d[k] = static_cast<DType>(cf1 - fb * fs);  // out_first
      vo2.d[k] = static_cast<DType>(cf2 + fa * fs);  // out_second
    }
    *reinterpret_cast<V*>(oh + j0) = vo1;
    *reinterpret_cast<V*>(oh + kHalf + j0) = vo2;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kHalf, bool kUsePDL, typename DType>
struct Ltx2SplitRotaryKernel {
  static constexpr auto kernel = ltx2_split_rotary_kernel<kHalf, kUsePDL, DType>;

  static void run(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView cos,
      tvm::ffi::TensorView sin) {
    using namespace host;
    constexpr int kHeadDim = 2 * kHalf;

    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"seq"};
    auto H = SymbolicSize{"heads"};
    auto Inner = SymbolicSize{"inner"};
    auto Xb = SymbolicSize{"x_batch_stride"};
    auto Xs = SymbolicSize{"x_seq_stride"};
    auto Cb = SymbolicSize{"cos_b"};
    auto Ch = SymbolicSize{"cos_h"};
    auto Cs = SymbolicSize{"cos_s"};
    auto Sb = SymbolicSize{"sin_b"};
    auto Sh = SymbolicSize{"sin_h"};
    auto Ss = SymbolicSize{"sin_s"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    // cos/sin: (batch, heads, seq, half), inner (half) contiguous, head/seq/batch strided.
    TensorMatcher({B, H, S, kHalf}).with_strides({Cb, Ch, Cs, 1}).with_dtype<DType>().with_device(device).verify(cos);
    TensorMatcher({B, H, S, kHalf}).with_strides({Sb, Sh, Ss, 1}).with_dtype<DType>().with_device(device).verify(sin);
    // x/out: (batch, seq, heads*2*half), contiguous inner.
    TensorMatcher({B, S, Inner}).with_strides({Xb, Xs, 1}).with_dtype<DType>().with_device(device).verify(x);
    TensorMatcher({B, S, Inner}).with_strides({Xb, Xs, 1}).with_dtype<DType>().with_device(device).verify(out);

    const auto params = Ltx2RotaryParams{
        .out = out.data_ptr(),
        .x = x.data_ptr(),
        .cos = cos.data_ptr(),
        .sin = sin.data_ptr(),
        .x_batch_stride = static_cast<int64_t>(Xb.unwrap()),
        .x_seq_stride = static_cast<int64_t>(Xs.unwrap()),
        .cos_b = static_cast<int64_t>(Cb.unwrap()),
        .cos_h = static_cast<int64_t>(Ch.unwrap()),
        .cos_s = static_cast<int64_t>(Cs.unwrap()),
        .sin_b = static_cast<int64_t>(Sb.unwrap()),
        .sin_h = static_cast<int64_t>(Sh.unwrap()),
        .sin_s = static_cast<int64_t>(Ss.unwrap()),
        .batch = static_cast<uint32_t>(B.unwrap()),
        .seq_len = static_cast<uint32_t>(S.unwrap()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
    };
    // Match block size to per-row work so threads are not left idle: half32 rows
    // hold only num_heads*(half/kVec) vectors, which is < 256 -> a fixed 256-thread
    // block would idle half its warps and halve effective memory parallelism.
    constexpr int kVec = (kHalf % 8 == 0) ? 8 : ((kHalf % 4 == 0) ? 4 : ((kHalf % 2 == 0) ? 2 : 1));
    const uint32_t work_per_row = params.num_heads * (static_cast<uint32_t>(kHalf) / kVec);
    uint32_t threads = ((work_per_row + 31u) / 32u) * 32u;  // round up to a warp
    threads = (threads == 0u) ? 32u : (threads > kThreadsPerBlock ? kThreadsPerBlock : threads);
    const uint32_t num_blocks = params.batch * params.seq_len;
    LaunchKernel(num_blocks, threads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace kda_diffusion_rotary
