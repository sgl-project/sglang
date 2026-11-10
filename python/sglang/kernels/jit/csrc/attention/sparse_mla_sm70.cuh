// SM70 sparse MLA for the NoPE latent. Volta has no bf16 tensor cores, and
// the Triton / TileLang DSA kernels only accept bf16 or fp8. This kernel is
// the fp16 path: K and V are the same 512-d latent, rope tail is 0.
//
// One block owns one query token, up to 8 heads (one warp each) and one
// split of the topk slots. Rows are staged 32 at a time in shared memory and
// every head dots against them; invalid (negative) indices are skipped.
// Softmax is fp32 and online. With one split the block writes the fp16
// output; with several, each writes its (max, sum, unnormalized acc) and
// sparse_mla_sm70_combine_kernel merges them in split order.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace sglang {

constexpr int kSparseMlaSm70Dim = 512;
constexpr int kSparseMlaSm70HeadsPerBlock = 8;
constexpr int kSparseMlaSm70Threads = 256;
constexpr int kSparseMlaSm70Chunk = 32;

struct SparseMlaSm70Params {
  const fp16_t* __restrict__ q;       // [S, H, 512]
  const fp16_t* __restrict__ kv;      // row stride = stride_kv
  const int32_t* __restrict__ indices;  // [S, topk]
  fp16_t* __restrict__ out;           // [S, H, 512]
  fp32_t* __restrict__ part_acc;      // [S, H, splits, 512], splits > 1 only
  fp32_t* __restrict__ part_ml;       // [S, H, splits, 2], splits > 1 only
  int64_t stride_kv;
  int64_t stride_q_s;  // q token and head strides; q may be a transposed view
  int64_t stride_q_h;
  uint32_t H;
  uint32_t topk;
  uint32_t splits;
  uint32_t keys_per_split;
  fp32_t scale;
};

SGL_DEVICE float sparse_mla_sm70_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  return v;
}

template <bool kVec16>
__global__ void __launch_bounds__(kSparseMlaSm70Threads) sparse_mla_sm70_kernel(SparseMlaSm70Params p) {
  const uint32_t seq = blockIdx.x;
  const uint32_t split = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = static_cast<int>(blockIdx.y) * kSparseMlaSm70HeadsPerBlock + warp;
  const bool active = head >= 0 && static_cast<uint32_t>(head) < p.H;

  float qv[16];
  float acc[16];
  if (active) {
    const fp16_t* qrow =
        p.q + static_cast<int64_t>(seq) * p.stride_q_s + static_cast<int64_t>(head) * p.stride_q_h;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const fp16x2_t h2 =
          *reinterpret_cast<const fp16x2_t*>(qrow + (i * 32 + lane) * 2);
      qv[2 * i] = __half2float(h2.x);
      qv[2 * i + 1] = __half2float(h2.y);
      acc[2 * i] = 0.f;
      acc[2 * i + 1] = 0.f;
    }
  }

  float m_i = -INFINITY;
  float l_i = 0.f;
  __shared__ __align__(16) fp16_t s_kv[kSparseMlaSm70Chunk][kSparseMlaSm70Dim];
  __shared__ int32_t s_idx[kSparseMlaSm70Chunk];

  const int32_t* idx_row = p.indices + static_cast<int64_t>(seq) * p.topk;
  const uint32_t t_begin = split * p.keys_per_split;
  const uint32_t t_end = min(p.topk, t_begin + p.keys_per_split);
  for (uint32_t c0 = t_begin; c0 < t_end; c0 += kSparseMlaSm70Chunk) {
    const int n = static_cast<int>(min(static_cast<uint32_t>(kSparseMlaSm70Chunk), t_end - c0));
    if (threadIdx.x < kSparseMlaSm70Chunk) {
      s_idx[threadIdx.x] = static_cast<int>(threadIdx.x) < n ? idx_row[c0 + threadIdx.x] : -1;
    }
    __syncthreads();
    if constexpr (kVec16) {
      constexpr int kPerRow = kSparseMlaSm70Dim / 8;
      for (int e = threadIdx.x; e < kSparseMlaSm70Chunk * kPerRow; e += kSparseMlaSm70Threads) {
        const int r = e / kPerRow;
        const int col = (e % kPerRow) * 8;
        const int32_t idx = s_idx[r];
        if (idx >= 0) {
          *reinterpret_cast<uint4*>(&s_kv[r][col]) =
              *reinterpret_cast<const uint4*>(p.kv + static_cast<int64_t>(idx) * p.stride_kv + col);
        }
      }
    } else {
      constexpr int kPerRow = kSparseMlaSm70Dim / 2;
      for (int e = threadIdx.x; e < kSparseMlaSm70Chunk * kPerRow; e += kSparseMlaSm70Threads) {
        const int r = e / kPerRow;
        const int col = (e % kPerRow) * 2;
        const int32_t idx = s_idx[r];
        if (idx >= 0) {
          *reinterpret_cast<fp16x2_t*>(&s_kv[r][col]) =
              *reinterpret_cast<const fp16x2_t*>(p.kv + static_cast<int64_t>(idx) * p.stride_kv + col);
        }
      }
    }
    __syncthreads();
    if (active) {
      for (int r = 0; r < n; ++r) {
        if (s_idx[r] < 0) {
          continue;
        }
        const fp16_t* row = s_kv[r];
        float partial = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const fp16x2_t h2 = *reinterpret_cast<const fp16x2_t*>(row + (i * 32 + lane) * 2);
          const float x = __half2float(h2.x);
          const float y = __half2float(h2.y);
          partial += qv[2 * i] * x + qv[2 * i + 1] * y;
        }
        const float score = sparse_mla_sm70_warp_sum(partial) * p.scale;
        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float prob = expf(score - m_new);
        l_i = l_i * alpha + prob;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const fp16x2_t h2 = *reinterpret_cast<const fp16x2_t*>(row + (i * 32 + lane) * 2);
          acc[2 * i] = acc[2 * i] * alpha + prob * __half2float(h2.x);
          acc[2 * i + 1] = acc[2 * i + 1] * alpha + prob * __half2float(h2.y);
        }
        m_i = m_new;
      }
    }
    __syncthreads();
  }

  if (!active) {
    return;
  }
  const int64_t sh = static_cast<int64_t>(seq) * p.H + head;
  if (p.splits > 1) {
    const int64_t slot = sh * p.splits + split;
    fp32_t* pacc = p.part_acc + slot * kSparseMlaSm70Dim;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<float2*>(pacc + (i * 32 + lane) * 2) = make_float2(acc[2 * i], acc[2 * i + 1]);
    }
    if (lane == 0) {
      p.part_ml[slot * 2] = m_i;
      p.part_ml[slot * 2 + 1] = l_i;
    }
    return;
  }
  const float inv = (l_i == 0.f) ? 0.f : (1.f / l_i);
  fp16_t* orow = p.out + sh * kSparseMlaSm70Dim;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    fp16x2_t h2;
    h2.x = __float2half_rn(acc[2 * i] * inv);
    h2.y = __float2half_rn(acc[2 * i + 1] * inv);
    *reinterpret_cast<fp16x2_t*>(orow + (i * 32 + lane) * 2) = h2;
  }
}

// Grid (S, H). Each thread owns two of the 512 dims.
__global__ void __launch_bounds__(kSparseMlaSm70Threads) sparse_mla_sm70_combine_kernel(SparseMlaSm70Params p) {
  const int64_t sh = static_cast<int64_t>(blockIdx.x) * p.H + blockIdx.y;
  const fp32_t* ml = p.part_ml + sh * p.splits * 2;
  const fp32_t* pacc = p.part_acc + sh * p.splits * kSparseMlaSm70Dim;
  float m = -INFINITY;
  for (uint32_t s = 0; s < p.splits; ++s) m = fmaxf(m, ml[2 * s]);
  float l = 0.f;
  float a0 = 0.f;
  float a1 = 0.f;
  const int d = threadIdx.x * 2;
  if (m != -INFINITY) {
    for (uint32_t s = 0; s < p.splits; ++s) {
      const float ms = ml[2 * s];
      if (ms == -INFINITY) {
        continue;
      }
      const float w = expf(ms - m);
      l += ml[2 * s + 1] * w;
      const float2 v = *reinterpret_cast<const float2*>(pacc + static_cast<int64_t>(s) * kSparseMlaSm70Dim + d);
      a0 += v.x * w;
      a1 += v.y * w;
    }
  }
  const float inv = (l == 0.f) ? 0.f : (1.f / l);
  fp16x2_t h2;
  h2.x = __float2half_rn(a0 * inv);
  h2.y = __float2half_rn(a1 * inv);
  *reinterpret_cast<fp16x2_t*>(p.out + sh * kSparseMlaSm70Dim + d) = h2;
}

struct SparseMlaSm70Kernel {

  /// \brief Sparse attention over indexer-selected latent rows.
  /// \param q `[S, H, 512]` fp16
  /// \param kv `[N, 512]` or `[N, 1, 512]` fp16. Index `i` selects row `i`.
  /// \param indices `[S, topk]` int32. Negative entries are masked out.
  /// \param out `[S, H, 512]` fp16
  /// \param part_acc fp32 workspace, at least `S * H * splits * 512` when splits > 1
  /// \param part_ml fp32 workspace, at least `S * H * splits * 2` when splits > 1
  /// \param sm_scale Softmax scale applied to the query-key dot.
  /// \param splits Blocks per (token, head tile) over the topk slots; 1 writes `out` directly.
  static void run(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView part_acc,
      const tvm::ffi::TensorView part_ml,
      double sm_scale,
      int64_t splits) {
    using namespace host;

    auto S_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"heads"};
    auto Dq_ = SymbolicSize{"q_dim"};
    auto Topk_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S_, H_, Dq_})
        .with_dtype<fp16_t>()
        .with_device(device)
        .with_strides({-1, -1, 1})
        .verify(q);
    TensorMatcher({S_, H_, Dq_})
        .with_dtype<fp16_t>()
        .with_device(device)
        .with_strides({-1, -1, 1})
        .verify(out);
    TensorMatcher({S_, Topk_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(indices);

    const auto S = S_.unwrap();
    const auto H = static_cast<uint32_t>(H_.unwrap());
    const auto topk = static_cast<uint32_t>(Topk_.unwrap());
    RuntimeCheck(Dq_.unwrap() == kSparseMlaSm70Dim, "SM70 sparse MLA is specialized for latent dim 512");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0 && q.stride(0) % 2 == 0 && q.stride(1) % 2 == 0,
        "q rows must be fp16x2-aligned");
    RuntimeCheck(
        out.stride(0) == static_cast<int64_t>(H) * kSparseMlaSm70Dim && out.stride(1) == kSparseMlaSm70Dim,
        "out must be contiguous [S, H, 512]");
    RuntimeCheck(indices.stride(0) == static_cast<int64_t>(topk), "indices must be contiguous [S, topk]");

    int64_t stride_kv = 0;
    if (kv.ndim() == 2) {
      auto N_ = SymbolicSize{"kv_rows"};
      auto Dk_ = SymbolicSize{"kv_dim"};
      TensorMatcher({N_, Dk_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, 1}).verify(kv);
      RuntimeCheck(Dk_.unwrap() == kSparseMlaSm70Dim, "kv last dim must be 512");
      stride_kv = kv.stride(0);
    } else if (kv.ndim() == 3) {
      auto N_ = SymbolicSize{"kv_rows"};
      auto P_ = SymbolicSize{"page"};
      auto Dk_ = SymbolicSize{"kv_dim"};
      TensorMatcher({N_, P_, Dk_})
          .with_dtype<fp16_t>()
          .with_device(device)
          .with_strides({-1, -1, 1})
          .verify(kv);
      RuntimeCheck(P_.unwrap() == 1 && Dk_.unwrap() == kSparseMlaSm70Dim, "kv must be [N, 1, 512]");
      stride_kv = kv.stride(0);
    } else {
      RuntimeCheck(false, "kv must be [N, 512] or [N, 1, 512]");
    }
    RuntimeCheck(stride_kv > 0 && (stride_kv % 2) == 0, "kv row stride must be a positive even number of fp16 elements");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(kv.data_ptr()) % 4 == 0, "kv base must be 4-byte aligned for half2 loads");

    if (S == 0 || H == 0) {
      return;
    }

    RuntimeCheck(splits >= 1 && static_cast<uint64_t>(splits) <= std::max<uint32_t>(topk, 1), "splits must be in [1, topk]");
    const uint32_t n_splits = static_cast<uint32_t>(splits);
    if (n_splits > 1) {
      const int64_t slots = S * static_cast<int64_t>(H) * n_splits;
      RuntimeCheck(
          part_acc.is_contiguous() && part_acc.dtype().code == kDLFloat && part_acc.dtype().bits == 32 &&
              part_acc.numel() >= slots * kSparseMlaSm70Dim,
          "part_acc must be contiguous fp32 [S, H, splits, 512]");
      RuntimeCheck(
          part_ml.is_contiguous() && part_ml.dtype().code == kDLFloat && part_ml.dtype().bits == 32 &&
              part_ml.numel() >= slots * 2,
          "part_ml must be contiguous fp32 [S, H, splits, 2]");
    }

    const SparseMlaSm70Params params{
        .q = static_cast<const fp16_t*>(q.data_ptr()),
        .kv = static_cast<const fp16_t*>(kv.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .out = static_cast<fp16_t*>(out.data_ptr()),
        .part_acc = n_splits > 1 ? static_cast<fp32_t*>(part_acc.data_ptr()) : nullptr,
        .part_ml = n_splits > 1 ? static_cast<fp32_t*>(part_ml.data_ptr()) : nullptr,
        .stride_kv = stride_kv,
        .stride_q_s = q.stride(0),
        .stride_q_h = q.stride(1),
        .H = H,
        .topk = topk,
        .splits = n_splits,
        .keys_per_split = (topk + n_splits - 1) / n_splits,
        .scale = static_cast<fp32_t>(sm_scale),
    };

    const uint32_t head_tiles =
        (H + kSparseMlaSm70HeadsPerBlock - 1) / kSparseMlaSm70HeadsPerBlock;
    dim3 grid(static_cast<uint32_t>(S), head_tiles, n_splits);
    const bool vec16 =
        stride_kv % 8 == 0 && reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0;
    if (vec16) {
      LaunchKernel(grid, kSparseMlaSm70Threads, device.unwrap())(sparse_mla_sm70_kernel<true>, params);
    } else {
      LaunchKernel(grid, kSparseMlaSm70Threads, device.unwrap())(sparse_mla_sm70_kernel<false>, params);
    }
    if (n_splits > 1) {
      LaunchKernel(dim3(static_cast<uint32_t>(S), H), kSparseMlaSm70Threads, device.unwrap())(
          sparse_mla_sm70_combine_kernel, params);
    }
  }
};

}  // namespace sglang
