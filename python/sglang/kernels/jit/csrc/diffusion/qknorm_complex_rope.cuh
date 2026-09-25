// Bit-exact 128-wide RMSNorm + complex RoPE for Qwen-Image-2.1 attention prep.
//
// Eager chain reproduced for bf16 activations:
//   RMSNorm(cast_x_before_out_mul=True): var = mean(float(x)^2) in aten's
//   vectorized 128-wide fp32 mean order (lane t owns elements 4t..4t+3, sums
//   them left to right, then a 32-lane shuffle tree with offsets 16,8,4,2,1),
//   inv = rsqrtf(var + eps), n = bf16(x * inv), o = bf16(n * w);
//   complex RoPE on interleaved pairs: out = view_as_real(complex(o) * rope),
//   with the FMA orientation PyTorch's complex64 multiply uses on this GPU
//   (kFuseRealSin, probed on the Python side).
// One launch handles the Q rows and the K rows (source -> destination, in place
// when they alias), copies the cached prefix K/V into rows [0:P] of the
// [B, P+S, H, 128] K/V buffers and, with kCopyV, the raw V rows into [P:P+S].

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace qknorm_complex_rope {

constexpr int kHeadDim = 128;
constexpr int kLaneElems = kHeadDim / device::kWarpThreads;  // 4 bf16 per lane
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarpsPerBlock = kThreads / device::kWarpThreads;
constexpr uint32_t kMaxBlocks = 65535;
constexpr int64_t kVecAlignBytes = 8;

static_assert(kLaneElems == 4, "the aten mean replica assumes four elements per lane");

using RowVec = device::AlignedVector<bf16_t, kLaneElems>;   // 8 bytes
using RopeVec = device::AlignedVector<fp32_t, kLaneElems>;  // cos0 sin0 cos1 sin1

/// Strided [B, T, H, 128] tensor: element (b, t, h, d) lives at ptr + b*batch_stride + t*token_stride + h*128 + d.
struct Rows {
  const bf16_t* ptr;
  int64_t batch_stride;
  int64_t token_stride;
};

struct MutableRows {
  bf16_t* ptr;
  int64_t batch_stride;
  int64_t token_stride;
};

struct Params {
  Rows q_src;
  MutableRows q_dst;
  Rows k_src;
  MutableRows k_dst;  // [B, P+S, H, 128]; token rows land at P + s
  MutableRows v_dst;  // [B, P+S, H, 128]
  Rows v_src;         // read only with kCopyV
  const bf16_t* q_weight;
  const bf16_t* k_weight;
  const bf16_t* k_prefix;  // contiguous [B, P, H, 128]
  const bf16_t* v_prefix;  // contiguous [B, P, H, 128]
  const fp32_t* rope;      // [S, 128] = view_as_real of the complex64 [S, 64] table
  uint32_t batch;
  uint32_t seq;
  uint32_t heads;
  uint32_t prefix;
  uint32_t k_rows_enabled;  // 0 for the single-tensor entry point
  float eps;
};

SGL_DEVICE const bf16_t* row_ptr(const Rows& rows, int64_t b, int64_t t, int64_t h) {
  return rows.ptr + b * rows.batch_stride + t * rows.token_stride + h * kHeadDim;
}

SGL_DEVICE bf16_t* row_ptr(const MutableRows& rows, int64_t b, int64_t t, int64_t h) {
  return rows.ptr + b * rows.batch_stride + t * rows.token_stride + h * kHeadDim;
}

/// \brief Normalize and rotate one 128-wide row; every lane owns 4 consecutive elements.
template <bool kFuseRealSin>
SGL_DEVICE void norm_rope_row(
    const bf16_t* __restrict__ src,
    bf16_t* __restrict__ dst,
    const bf16_t* __restrict__ weight,
    const fp32_t* __restrict__ rope_row,
    float eps,
    int lane) {
  RowVec x;
  RowVec w;
  RopeVec r;
  x.load(src + lane * kLaneElems);
  w.load(weight + lane * kLaneElems);
  r.load(rope_row + lane * kLaneElems);

  float value[kLaneElems];
#pragma unroll
  for (int i = 0; i < kLaneElems; ++i) {
    value[i] = device::cast<fp32_t>(x[i]);
  }
  // aten vectorized mean: four squares combined left to right ...
  float part = __fmul_rn(value[0], value[0]);
  part = __fadd_rn(part, __fmul_rn(value[1], value[1]));
  part = __fadd_rn(part, __fmul_rn(value[2], value[2]));
  part = __fadd_rn(part, __fmul_rn(value[3], value[3]));
  // ... then the 32-lane tree with decreasing offsets (matches shfl_down 16..1 for lane 0).
#pragma unroll
  for (int offset = device::kWarpThreads / 2; offset > 0; offset >>= 1) {
    part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, offset));
  }
  const float variance = __fmul_rn(part, 1.0f / kHeadDim);
  const float inv = rsqrtf(__fadd_rn(variance, eps));

  // cast_x_before_out_mul: round after the normalize, then again after the weight.
  float out_f32[kLaneElems];
#pragma unroll
  for (int i = 0; i < kLaneElems; ++i) {
    const float normalized = device::cast<fp32_t>(device::cast<bf16_t>(__fmul_rn(value[i], inv)));
    out_f32[i] = device::cast<fp32_t>(device::cast<bf16_t>(__fmul_rn(normalized, device::cast<fp32_t>(w[i]))));
  }

  RowVec out;
#pragma unroll
  for (int p = 0; p < kLaneElems / 2; ++p) {
    const float re = out_f32[2 * p];
    const float im = out_f32[2 * p + 1];
    const float cos = r[2 * p];
    const float sin = r[2 * p + 1];
    const float out_re = __fmaf_rn(re, cos, -__fmul_rn(im, sin));
    float out_im;
    if constexpr (kFuseRealSin) {
      out_im = __fmaf_rn(re, sin, __fmul_rn(im, cos));
    } else {
      out_im = __fmaf_rn(im, cos, __fmul_rn(re, sin));
    }
    out[2 * p] = device::cast<bf16_t>(out_re);
    out[2 * p + 1] = device::cast<bf16_t>(out_im);
  }
  out.store(dst + lane * kLaneElems);
}

SGL_DEVICE void copy_row(const bf16_t* __restrict__ src, bf16_t* __restrict__ dst, int lane) {
  RowVec v;
  v.load(src + lane * kLaneElems);
  v.store(dst + lane * kLaneElems);
}

/// Work list: Q rows, K rows, prefix-K copies, prefix-V copies, then (kCopyV) V copies. One warp per row.
template <bool kFuseRealSin, bool kCopyV>
__global__ void kernel(const Params __grid_constant__ params) {
  const int lane = threadIdx.x & (device::kWarpThreads - 1);
  const uint32_t warp = threadIdx.x >> 5;
  const int64_t heads = params.heads;
  const int64_t seq = params.seq;
  const int64_t prefix = params.prefix;
  const int64_t rows_q = static_cast<int64_t>(params.batch) * seq * heads;
  const int64_t rows_k = params.k_rows_enabled ? rows_q : 0;
  const int64_t rows_p = params.k_rows_enabled ? static_cast<int64_t>(params.batch) * prefix * heads : 0;
  const int64_t rows_v = kCopyV ? rows_q : 0;
  const int64_t total = rows_q + rows_k + 2 * rows_p + rows_v;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp; row < total; row += stride) {
    int64_t r = row;
    if (r < rows_q) {
      const int64_t h = r % heads, t = r / heads, s = t % seq, b = t / seq;
      norm_rope_row<kFuseRealSin>(
          row_ptr(params.q_src, b, s, h),
          row_ptr(params.q_dst, b, s, h),
          params.q_weight,
          params.rope + s * kHeadDim,
          params.eps,
          lane);
      continue;
    }
    r -= rows_q;
    if (r < rows_k) {
      const int64_t h = r % heads, t = r / heads, s = t % seq, b = t / seq;
      norm_rope_row<kFuseRealSin>(
          row_ptr(params.k_src, b, s, h),
          row_ptr(params.k_dst, b, prefix + s, h),
          params.k_weight,
          params.rope + s * kHeadDim,
          params.eps,
          lane);
      continue;
    }
    r -= rows_k;
    if (r < rows_p) {
      const int64_t h = r % heads, t = r / heads, p = t % prefix, b = t / prefix;
      copy_row(params.k_prefix + ((b * prefix + p) * heads + h) * kHeadDim, row_ptr(params.k_dst, b, p, h), lane);
      continue;
    }
    r -= rows_p;
    if (r < rows_p) {
      const int64_t h = r % heads, t = r / heads, p = t % prefix, b = t / prefix;
      copy_row(params.v_prefix + ((b * prefix + p) * heads + h) * kHeadDim, row_ptr(params.v_dst, b, p, h), lane);
      continue;
    }
    if constexpr (kCopyV) {
      r -= rows_p;
      const int64_t h = r % heads, t = r / heads, s = t % seq, b = t / seq;
      copy_row(row_ptr(params.v_src, b, s, h), row_ptr(params.v_dst, b, prefix + s, h), lane);
    }
  }
}

template <bool kFuseRealSin, bool kCopyV>
struct Kernel {
  /// \brief Single tensor, out of place: `out = rope(rmsnorm(x))` for a contiguous [B, S, H, 128] bf16 tensor.
  static void
  run(tvm::ffi::TensorView x, tvm::ffi::TensorView out, tvm::ffi::TensorView weight, tvm::ffi::TensorView rope, double eps) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"seq"};
    auto H = SymbolicSize{"heads"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, S, H, kHeadDim})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kVecAlignBytes)
        .verify(x)
        .verify(out);
    TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({S, kHeadDim}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(rope);
    const int64_t batch = B.unwrap(), seq = S.unwrap(), heads = H.unwrap();
    if (batch == 0 || seq == 0 || heads == 0) return;
    const int64_t token_stride = heads * kHeadDim;
    Params params{};
    params.q_src = Rows{static_cast<const bf16_t*>(x.data_ptr()), seq * token_stride, token_stride};
    params.q_dst = MutableRows{static_cast<bf16_t*>(out.data_ptr()), seq * token_stride, token_stride};
    params.q_weight = static_cast<const bf16_t*>(weight.data_ptr());
    params.k_weight = params.q_weight;
    params.rope = static_cast<const fp32_t*>(rope.data_ptr());
    params.batch = static_cast<uint32_t>(batch);
    params.seq = static_cast<uint32_t>(seq);
    params.heads = static_cast<uint32_t>(heads);
    params.prefix = 0;
    params.k_rows_enabled = 0;
    params.eps = static_cast<float>(eps);
    launch(params, batch * seq * heads, device.unwrap());
  }

  /// \brief Q in place, K source -> rows [P:] of k_out, prefix K/V -> rows [0:P] of k_out/v_out,
  /// raw V -> rows [P:] of v_out when kCopyV (v_src is ignored otherwise).
  static void run_pack(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView q_weight,
      tvm::ffi::TensorView k_src,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView v_src,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView k_prefix,
      tvm::ffi::TensorView v_prefix,
      double eps) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"seq"};
    auto H = SymbolicSize{"heads"};
    auto P = SymbolicSize{"prefix"};
    auto PS = SymbolicSize{"prefix_plus_seq"};
    auto Bq = SymbolicSize{"q_batch_stride"}, Tq = SymbolicSize{"q_token_stride"};
    auto Bk = SymbolicSize{"k_src_batch_stride"}, Tk = SymbolicSize{"k_src_token_stride"};
    auto Bko = SymbolicSize{"k_out_batch_stride"}, Tko = SymbolicSize{"k_out_token_stride"};
    auto Bvo = SymbolicSize{"v_out_batch_stride"}, Tvo = SymbolicSize{"v_out_token_stride"};
    auto Bv = SymbolicSize{"v_src_batch_stride"}, Tv = SymbolicSize{"v_src_token_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    // TensorMatcher is not copyable, so verify inside the lambda; the member
    // template call on a dependent expression needs `template`.
    const auto verify_rows =
        [&](const tvm::ffi::TensorView& t, auto& rows, auto& batch_stride, auto& token_stride) {
          TensorMatcher({B, rows, H, kHeadDim})
              .with_strides({batch_stride, token_stride, kHeadDim, 1})
              .template with_dtype<bf16_t>()
              .with_device(device)
              .ensure_alignment(kVecAlignBytes)
              .verify(t);
        };
    verify_rows(q, S, Bq, Tq);
    verify_rows(k_src, S, Bk, Tk);
    verify_rows(k_out, PS, Bko, Tko);
    verify_rows(v_out, PS, Bvo, Tvo);
    if constexpr (kCopyV) {
      verify_rows(v_src, S, Bv, Tv);
    }
    TensorMatcher({B, P, H, kHeadDim})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kVecAlignBytes)
        .verify(k_prefix)
        .verify(v_prefix);
    TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({S, kHeadDim}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(rope);
    const int64_t batch = B.unwrap(), seq = S.unwrap(), heads = H.unwrap(), prefix = P.unwrap();
    CHECK_HOST(PS.unwrap() == prefix + seq) << "k_out/v_out must hold prefix + seq tokens";
    if (batch == 0 || seq == 0 || heads == 0) return;

    Params params{};
    params.q_src = Rows{static_cast<const bf16_t*>(q.data_ptr()), Bq.unwrap(), Tq.unwrap()};
    params.q_dst = MutableRows{static_cast<bf16_t*>(q.data_ptr()), Bq.unwrap(), Tq.unwrap()};
    params.k_src = Rows{static_cast<const bf16_t*>(k_src.data_ptr()), Bk.unwrap(), Tk.unwrap()};
    params.k_dst = MutableRows{static_cast<bf16_t*>(k_out.data_ptr()), Bko.unwrap(), Tko.unwrap()};
    params.v_dst = MutableRows{static_cast<bf16_t*>(v_out.data_ptr()), Bvo.unwrap(), Tvo.unwrap()};
    if constexpr (kCopyV) {
      params.v_src = Rows{static_cast<const bf16_t*>(v_src.data_ptr()), Bv.unwrap(), Tv.unwrap()};
    }
    params.q_weight = static_cast<const bf16_t*>(q_weight.data_ptr());
    params.k_weight = static_cast<const bf16_t*>(k_weight.data_ptr());
    params.k_prefix = static_cast<const bf16_t*>(k_prefix.data_ptr());
    params.v_prefix = static_cast<const bf16_t*>(v_prefix.data_ptr());
    params.rope = static_cast<const fp32_t*>(rope.data_ptr());
    params.batch = static_cast<uint32_t>(batch);
    params.seq = static_cast<uint32_t>(seq);
    params.heads = static_cast<uint32_t>(heads);
    params.prefix = static_cast<uint32_t>(prefix);
    params.k_rows_enabled = 1;
    params.eps = static_cast<float>(eps);
    const int64_t rows = batch * heads * (2 * seq + 2 * prefix + (kCopyV ? seq : 0));
    launch(params, rows, device.unwrap());
  }

 private:
  static void launch(const Params& params, int64_t rows, DLDevice device) {
    using namespace host;
    const int64_t needed = div_ceil(rows, static_cast<int64_t>(kWarpsPerBlock));
    const auto blocks = static_cast<uint32_t>(std::min<int64_t>(needed, kMaxBlocks));
    LaunchKernel(blocks, kThreads, device)(kernel<kFuseRealSin, kCopyV>, params);
  }
};

}  // namespace qknorm_complex_rope

}  // namespace sglang
