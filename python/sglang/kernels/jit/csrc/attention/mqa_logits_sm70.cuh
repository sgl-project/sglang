// SM70 indexer score. DeepGEMM's fp8 MQA logits need Hopper. The math is the
// same on Volta, done in fp32 after a software E4M3FN decode:
//   score(q, k) = sum_h relu(q[h] . k) * weight[h] * k_scale
// Keys outside [start, end) are -inf when clean_logits is set, else 0.
// Paged keys use the fused page layout: 64 packed fp8 tokens, then 64 fp32
// scales (8448 bytes per page).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace sglang {

constexpr int kMqaSm70MaxHeads = 64;
constexpr int kMqaSm70Dim = 128;
constexpr int kMqaSm70Tile = 128;
constexpr int kMqaSm70PageTokens = 64;
constexpr int kMqaSm70PageBytes = kMqaSm70PageTokens * (kMqaSm70Dim + 4);
constexpr int kMqaSm70ScaleOffset = kMqaSm70PageTokens * kMqaSm70Dim;

SGL_DEVICE float mqa_sm70_e4m3_to_float(uint8_t x) {
  const uint32_t sign = static_cast<uint32_t>(x & 0x80) << 24;
  const int exp = (x >> 3) & 0xF;
  const int mant = x & 0x7;
  if ((x & 0x7F) == 0x7F) {
    return __int_as_float(0x7FC00000 | sign);
  }
  if (exp == 0) {
    // Subnormal: mant * 2^-9. Zero stays zero.
    const float mag = static_cast<float>(mant) * 0x1p-9f;
    return __int_as_float(__float_as_int(mag) | sign);
  }
  // Normal: (1 + mant/8) * 2^(exp - 7). fp32 exponent bias is 127.
  const uint32_t bits = sign | (static_cast<uint32_t>(exp + 120) << 23) | (static_cast<uint32_t>(mant) << 20);
  return __uint_as_float(bits);
}

SGL_DEVICE float mqa_sm70_score_key(
    const float* q_s, const float* w_s, int heads, int lane, const uint8_t* k_ptr, float k_scale) {
  float acc = 0.f;
  for (int h = 0; h < heads; ++h) {
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      dot += q_s[h * kMqaSm70Dim + d] * mqa_sm70_e4m3_to_float(k_ptr[d]);
    }
    dot = device::warp::reduce_sum<32>(dot);
    acc += fmaxf(dot, 0.f) * w_s[h];
  }
  return acc * k_scale;
}

struct MqaSm70RaggedParams {
  const uint8_t* __restrict__ q;  // [M, H, 128]
  const uint8_t* __restrict__ k;  // [N, 128]
  const float* __restrict__ k_scale;
  const float* __restrict__ weights;  // [M, H]
  const int32_t* __restrict__ starts;
  const int32_t* __restrict__ ends;
  float* __restrict__ out;  // [M, N]
  int32_t heads;
  int32_t n;
  float masked;
  int32_t queries;
};

__global__ void mqa_sm70_ragged_kernel(const MqaSm70RaggedParams params) {
  const int m = static_cast<int>(blockIdx.y);
  const int k0 = static_cast<int>(blockIdx.x) * kMqaSm70Tile;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int heads = params.heads;
  const int n = params.n;
  if (k0 >= n) {
    return;
  }
  const int k1 = k0 + kMqaSm70Tile < n ? k0 + kMqaSm70Tile : n;

  __shared__ float q_s[kMqaSm70MaxHeads * kMqaSm70Dim];
  __shared__ float w_s[kMqaSm70MaxHeads];
  const uint8_t* q_row = params.q + static_cast<int64_t>(m) * heads * kMqaSm70Dim;
  for (int i = tid; i < heads * kMqaSm70Dim; i += blockDim.x) {
    q_s[i] = mqa_sm70_e4m3_to_float(q_row[i]);
  }
  for (int h = tid; h < heads; h += blockDim.x) {
    w_s[h] = params.weights[static_cast<int64_t>(m) * heads + h];
  }
  __syncthreads();

  int start = params.starts[m];
  int end = params.ends[m];
  if (start < 0) {
    start = 0;
  }
  if (end > n) {
    end = n;
  }
  if (end < start) {
    end = start;
  }

  for (int base = k0; base < k1; base += 4) {
    const int key = base + warp;
    const bool in_tile = key < k1;
    const bool scored = in_tile && key >= start && key < end;
    float value = params.masked;
    if (scored) {
      value = mqa_sm70_score_key(
          q_s, w_s, heads, lane, params.k + static_cast<int64_t>(key) * kMqaSm70Dim, params.k_scale[key]);
    }
    if (in_tile && lane == 0) {
      params.out[static_cast<int64_t>(m) * n + key] = value;
    }
  }
}

// Tensor-core ragged scores for 32 heads: per block 4 queries (128 q rows) times
// 128-key tiles, q.k on HMMA with fp32 accumulation, relu-weighted head sum in fp32.
constexpr int kMqaTcHeads = 32;
constexpr int kMqaTcQueries = 4;
constexpr int kMqaTcRows = kMqaTcQueries * kMqaTcHeads;
constexpr int kMqaTcKeys = 128;
constexpr int kMqaTcWarps = 8;
constexpr int kMqaTcLd = kMqaSm70Dim + 8;  // halves; the pad spreads wmma row loads over banks
// Arbitrary; enough key tiles per block to amortise the q decode.
constexpr int kMqaTcTilesPerBlock = 8;

struct MqaSm70TcSmem {
  __half q[kMqaTcRows][kMqaTcLd];
  __half k[kMqaTcKeys][kMqaTcLd];
  float scratch[kMqaTcWarps][16 * 16];
  float w[kMqaTcQueries][kMqaTcHeads];
  int start[kMqaTcQueries];
  int end[kMqaTcQueries];
};

// Four E4M3FN bytes to fp16, exact: an E4M3 byte's exponent and mantissa bits placed
// at fp16 bits 13..7 read as the value times 2^-8, subnormals included. NaN codes give 0.
SGL_DEVICE void mqa_sm70_e4m3x4_to_half(uint32_t w, __half2& lo, __half2& hi) {
  w &= ~__vcmpeq4(w & 0x7F7F7F7Fu, 0x7F7F7F7Fu);
  const uint32_t t_lo = __byte_perm(w, 0, 0x1404);
  const uint32_t t_hi = __byte_perm(w, 0, 0x3424);
  const uint32_t h_lo = (t_lo & 0x80008000u) | ((t_lo >> 1) & 0x3F803F80u);
  const uint32_t h_hi = (t_hi & 0x80008000u) | ((t_hi >> 1) & 0x3F803F80u);
  const __half2 k256 = __float2half2_rn(256.f);
  lo = __hmul2(*reinterpret_cast<const __half2*>(&h_lo), k256);
  hi = __hmul2(*reinterpret_cast<const __half2*>(&h_hi), k256);
}

// 16 E4M3 bytes to 16 halves at dst (32-byte aligned).
SGL_DEVICE void mqa_sm70_store_e4m3x16(__half* dst, uint4 v) {
  __half2 h[8];
  mqa_sm70_e4m3x4_to_half(v.x, h[0], h[1]);
  mqa_sm70_e4m3x4_to_half(v.y, h[2], h[3]);
  mqa_sm70_e4m3x4_to_half(v.z, h[4], h[5]);
  mqa_sm70_e4m3x4_to_half(v.w, h[6], h[7]);
  reinterpret_cast<uint4*>(dst)[0] = *reinterpret_cast<const uint4*>(&h[0]);
  reinterpret_cast<uint4*>(dst)[1] = *reinterpret_cast<const uint4*>(&h[4]);
}

// One query's 32 q rows against 16 * kCols keys of a k tile, both fp16 rows of
// kMqaTcLd halves. Lanes 0-15 get part[j] for key 16 * j + lane: sum_h relu(q[h].k) * w[h].
template <int kCols>
SGL_DEVICE void mqa_sm70_tc_scores(
    const __half* q_rows, const __half* k_rows, const float* w, float* scratch, int lane, float (&part)[kCols]) {
  using namespace nvcuda;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][kCols];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < kCols; ++j) {
      wmma::fill_fragment(c[i][j], 0.f);
    }
  }
#pragma unroll
  for (int d = 0; d < kMqaSm70Dim; d += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
    wmma::load_matrix_sync(a[0], q_rows + d, kMqaTcLd);
    wmma::load_matrix_sync(a[1], q_rows + 16 * kMqaTcLd + d, kMqaTcLd);
#pragma unroll
    for (int j = 0; j < kCols; ++j) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::load_matrix_sync(b, k_rows + j * 16 * kMqaTcLd + d, kMqaTcLd);
      wmma::mma_sync(c[0][j], a[0], b, c[0][j]);
      wmma::mma_sync(c[1][j], a[1], b, c[1][j]);
    }
  }
  // Lanes 0-15 sum the even heads, 16-31 the odd ones, then one shuffle adds the
  // halves; the order is fixed, so a row's score does not depend on its neighbours.
  const int col = lane & 15;
  const int parity = lane >> 4;
#pragma unroll
  for (int j = 0; j < kCols; ++j) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      wmma::store_matrix_sync(scratch, c[i][j], 16, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int row = 2 * r + parity;
        acc += fmaxf(scratch[row * 16 + col], 0.f) * w[i * 16 + row];
      }
      __syncwarp();
    }
    part[j] = acc + __shfl_xor_sync(0xffffffffu, acc, 16);
  }
}

__global__ void __launch_bounds__(kMqaTcWarps * 32, 1) mqa_sm70_ragged_tc_kernel(const MqaSm70RaggedParams params) {
  extern __shared__ __align__(16) unsigned char mqa_tc_smem_raw[];
  MqaSm70TcSmem& s = *reinterpret_cast<MqaSm70TcSmem*>(mqa_tc_smem_raw);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int n = params.n;
  const int m0 = static_cast<int>(blockIdx.y) * kMqaTcQueries;
  const int m_count = static_cast<int>(params.queries) - m0 < kMqaTcQueries ? static_cast<int>(params.queries) - m0
                                                                            : kMqaTcQueries;

  // q rows of this block's queries; rows of missing queries are zero.
  for (int i = tid; i < kMqaTcRows * kMqaSm70Dim / 16; i += blockDim.x) {
    const int row = i / (kMqaSm70Dim / 16);
    const int col = (i % (kMqaSm70Dim / 16)) * 16;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row / kMqaTcHeads < m_count) {
      v = *reinterpret_cast<const uint4*>(params.q + (static_cast<int64_t>(m0) * kMqaTcHeads + row) * kMqaSm70Dim + col);
    }
    mqa_sm70_store_e4m3x16(&s.q[row][col], v);
  }
  for (int i = tid; i < kMqaTcRows; i += blockDim.x) {
    const int qi = i / kMqaTcHeads;
    s.w[qi][i % kMqaTcHeads] = qi < m_count ? params.weights[static_cast<int64_t>(m0) * kMqaTcHeads + i] : 0.f;
  }
  if (tid < kMqaTcQueries) {
    int start = 0, end = 0;
    if (tid < m_count) {
      start = max(params.starts[m0 + tid], 0);
      end = min(params.ends[m0 + tid], n);
      end = max(end, start);
    }
    s.start[tid] = start;
    s.end[tid] = end;
  }

  const int qi = warp >> 1;
  const int key_half = (warp & 1) * 64;
  const int tile0 = static_cast<int>(blockIdx.x) * kMqaTcTilesPerBlock;
  for (int t = 0; t < kMqaTcTilesPerBlock; ++t) {
    const int k0 = (tile0 + t) * kMqaTcKeys;
    if (k0 >= n) {
      break;
    }
    __syncthreads();  // q/w/start/end ready on the first pass; the previous k tile consumed after
    bool any = false;
#pragma unroll
    for (int i = 0; i < kMqaTcQueries; ++i) {
      any |= s.start[i] < k0 + kMqaTcKeys && s.end[i] > k0 && s.start[i] < s.end[i];
    }
    if (any) {
      for (int i = tid; i < kMqaTcKeys * kMqaSm70Dim / 16; i += blockDim.x) {
        const int key = i / (kMqaSm70Dim / 16);
        const int col = (i % (kMqaSm70Dim / 16)) * 16;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (k0 + key < n) {
          v = __ldg(reinterpret_cast<const uint4*>(params.k + static_cast<int64_t>(k0 + key) * kMqaSm70Dim + col));
        }
        mqa_sm70_store_e4m3x16(&s.k[key][col], v);
      }
      __syncthreads();
    }

    float part[4];
    if (any) {
      mqa_sm70_tc_scores<4>(&s.q[qi * kMqaTcHeads][0], &s.k[key_half][0], s.w[qi], s.scratch[warp], lane, part);
    }
    if (qi < m_count && lane < 16) {
      const int64_t m = m0 + qi;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int key = k0 + key_half + j * 16 + lane;
        if (key < n) {
          const bool scored = any && key >= s.start[qi] && key < s.end[qi];
          params.out[m * n + key] = scored ? part[j] * params.k_scale[key] : params.masked;
        }
      }
    }
  }
}

struct MqaSm70PagedParams {
  const uint8_t* __restrict__ q;  // [B, H, 128]
  const uint8_t* __restrict__ kv;  // [pages, 8448]
  const float* __restrict__ weights;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;  // [B, max_pages]
  float* __restrict__ out;                 // [B, max_seq_len]
  int32_t heads;
  int32_t max_pages;
  int32_t max_seq_len;
  int64_t page_stride;
};

__global__ void mqa_sm70_paged_kernel(const MqaSm70PagedParams params) {
  const int b = static_cast<int>(blockIdx.y);
  const int p0 = static_cast<int>(blockIdx.x) * kMqaSm70Tile;
  const int seq_len = params.seq_lens[b];
  const int limit = seq_len < params.max_seq_len ? seq_len : params.max_seq_len;
  if (p0 >= limit) {
    return;
  }
  const int p1 = p0 + kMqaSm70Tile < limit ? p0 + kMqaSm70Tile : limit;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int heads = params.heads;

  __shared__ float q_s[kMqaSm70MaxHeads * kMqaSm70Dim];
  __shared__ float w_s[kMqaSm70MaxHeads];
  const uint8_t* q_row = params.q + static_cast<int64_t>(b) * heads * kMqaSm70Dim;
  for (int i = tid; i < heads * kMqaSm70Dim; i += blockDim.x) {
    q_s[i] = mqa_sm70_e4m3_to_float(q_row[i]);
  }
  for (int h = tid; h < heads; h += blockDim.x) {
    w_s[h] = params.weights[static_cast<int64_t>(b) * heads + h];
  }
  __syncthreads();

  for (int base = p0; base < p1; base += 4) {
    const int p = base + warp;
    const bool in_tile = p < p1;
    float value = 0.f;
    if (in_tile) {
      const int page = params.page_table[static_cast<int64_t>(b) * params.page_stride + (p >> 6)];
      if (page >= 0) {
        const uint8_t* page_ptr = params.kv + static_cast<int64_t>(page) * kMqaSm70PageBytes;
        const int off = p & 63;
        const float scale = *reinterpret_cast<const float*>(page_ptr + kMqaSm70ScaleOffset + off * 4);
        value = mqa_sm70_score_key(q_s, w_s, heads, lane, page_ptr + off * kMqaSm70Dim, scale);
      }
    }
    if (in_tile && lane == 0) {
      params.out[static_cast<int64_t>(b) * params.max_seq_len + p] = value;
    }
  }
}

struct MqaSm70PagedTcSmem {
  __half q[kMqaTcHeads][kMqaTcLd];
  __half k[kMqaSm70PageTokens][kMqaTcLd];
  float scratch[kMqaSm70PageTokens / 16][16 * 16];
  float w[kMqaTcHeads];
};

// Tensor-core paged scores for 32 heads: one block per (row, 64-token page).
__global__ void __launch_bounds__(kMqaSm70PageTokens / 16 * 32) mqa_sm70_paged_tc_kernel(const MqaSm70PagedParams params) {
  __shared__ __align__(16) unsigned char smem_raw[sizeof(MqaSm70PagedTcSmem)];
  MqaSm70PagedTcSmem& s = *reinterpret_cast<MqaSm70PagedTcSmem*>(smem_raw);
  const int b = static_cast<int>(blockIdx.y);
  const int p0 = static_cast<int>(blockIdx.x) * kMqaSm70PageTokens;
  const int seq_len = params.seq_lens[b];
  const int limit = seq_len < params.max_seq_len ? seq_len : params.max_seq_len;
  if (p0 >= limit) {
    return;
  }
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  float* out = params.out + static_cast<int64_t>(b) * params.max_seq_len;
  const int page = params.page_table[static_cast<int64_t>(b) * params.page_stride + blockIdx.x];
  if (page < 0) {
    if (p0 + tid < limit && tid < kMqaSm70PageTokens) {
      out[p0 + tid] = 0.f;
    }
    return;
  }
  const uint8_t* page_ptr = params.kv + static_cast<int64_t>(page) * kMqaSm70PageBytes;
  const uint8_t* q_row = params.q + static_cast<int64_t>(b) * kMqaTcHeads * kMqaSm70Dim;
  for (int i = tid; i < kMqaTcHeads * kMqaSm70Dim / 16; i += blockDim.x) {
    const int row = i / (kMqaSm70Dim / 16);
    const int col = (i % (kMqaSm70Dim / 16)) * 16;
    mqa_sm70_store_e4m3x16(&s.q[row][col], *reinterpret_cast<const uint4*>(q_row + row * kMqaSm70Dim + col));
  }
  for (int i = tid; i < kMqaSm70PageTokens * kMqaSm70Dim / 16; i += blockDim.x) {
    const int key = i / (kMqaSm70Dim / 16);
    const int col = (i % (kMqaSm70Dim / 16)) * 16;
    mqa_sm70_store_e4m3x16(&s.k[key][col], __ldg(reinterpret_cast<const uint4*>(page_ptr + key * kMqaSm70Dim + col)));
  }
  if (tid < kMqaTcHeads) {
    s.w[tid] = params.weights[static_cast<int64_t>(b) * kMqaTcHeads + tid];
  }
  __syncthreads();

  float part[1];
  mqa_sm70_tc_scores<1>(&s.q[0][0], &s.k[warp * 16][0], s.w, s.scratch[warp], lane, part);
  const int off = warp * 16 + lane;
  if (lane < 16 && p0 + off < limit) {
    const float scale = *reinterpret_cast<const float*>(page_ptr + kMqaSm70ScaleOffset + off * 4);
    out[p0 + off] = part[0] * scale;
  }
}

struct MqaLogitsSm70Kernel {
  static constexpr auto ragged_kernel = mqa_sm70_ragged_kernel;
  static constexpr auto ragged_tc_kernel = mqa_sm70_ragged_tc_kernel;
  static constexpr auto paged_kernel = mqa_sm70_paged_kernel;
  static constexpr auto paged_tc_kernel = mqa_sm70_paged_tc_kernel;

  /// \brief Ragged fp8 MQA logits. `q` and `k` are E4M3FN bytes.
  static void ragged(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView k_scale,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView starts,
      const tvm::ffi::TensorView ends,
      const tvm::ffi::TensorView out,
      double masked) {
    using namespace host;
    auto M_ = SymbolicSize{"queries"};
    auto H_ = SymbolicSize{"heads"};
    auto D_ = SymbolicSize{"dim"};
    auto N_ = SymbolicSize{"keys"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, H_, D_}).with_dtype<uint8_t>().with_device(device).verify(q);
    TensorMatcher({N_, D_}).with_dtype<uint8_t>().with_device(device).verify(k);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).verify(k_scale);
    TensorMatcher({M_, H_}).with_dtype<fp32_t>().with_device(device).verify(weights);
    TensorMatcher({M_}).with_dtype<int32_t>().with_device(device).verify(starts);
    TensorMatcher({M_}).with_dtype<int32_t>().with_device(device).verify(ends);
    TensorMatcher({M_, N_}).with_dtype<fp32_t>().with_device(device).verify(out);

    const auto M = static_cast<int>(M_.unwrap());
    const auto H = static_cast<int>(H_.unwrap());
    const auto N = static_cast<int>(N_.unwrap());
    RuntimeCheck(D_.unwrap() == kMqaSm70Dim, "SM70 MQA logits are specialized for head dim 128");
    RuntimeCheck(H > 0 && H <= kMqaSm70MaxHeads, "SM70 MQA logits support 1..64 heads");
    RuntimeCheck(
        q.stride(0) == static_cast<int64_t>(H) * kMqaSm70Dim && q.stride(1) == kMqaSm70Dim && q.stride(2) == 1,
        "q must be contiguous [M, H, 128]");
    RuntimeCheck(k.stride(0) == kMqaSm70Dim && k.stride(1) == 1, "k must be contiguous [N, 128]");
    RuntimeCheck(weights.stride(0) == H && weights.stride(1) == 1, "weights must be contiguous [M, H]");
    RuntimeCheck(out.stride(0) == N && out.stride(1) == 1, "out must be contiguous [M, N]");
    if (M == 0 || N == 0) {
      return;
    }

    const MqaSm70RaggedParams params{
        .q = static_cast<const uint8_t*>(q.data_ptr()),
        .k = static_cast<const uint8_t*>(k.data_ptr()),
        .k_scale = static_cast<const float*>(k_scale.data_ptr()),
        .weights = static_cast<const float*>(weights.data_ptr()),
        .starts = static_cast<const int32_t*>(starts.data_ptr()),
        .ends = static_cast<const int32_t*>(ends.data_ptr()),
        .out = static_cast<float*>(out.data_ptr()),
        .heads = H,
        .n = N,
        .masked = static_cast<float>(masked),
        .queries = M,
    };
    if (H == kMqaTcHeads) {
      constexpr int smem = sizeof(MqaSm70TcSmem);
      static const bool smem_set = [] {
        return cudaFuncSetAttribute(ragged_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess;
      }();
      RuntimeCheck(smem_set, "SM70 tensor-core MQA logits could not reserve shared memory");
      const int span = kMqaTcKeys * kMqaTcTilesPerBlock;
      const dim3 grid((N + span - 1) / span, (M + kMqaTcQueries - 1) / kMqaTcQueries);
      LaunchKernel(grid, kMqaTcWarps * 32, device.unwrap(), smem)(ragged_tc_kernel, params);
      return;
    }
    const int tiles = static_cast<int>((N + kMqaSm70Tile - 1) / kMqaSm70Tile);
    LaunchKernel(dim3(tiles, M), 128, device.unwrap())(ragged_kernel, params);
  }

  /// \brief Paged fp8 MQA logits. `kv` is `[pages, 8448]` uint8, tokens then scales.
  static void paged(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto B_ = SymbolicSize{"batch"};
    auto H_ = SymbolicSize{"heads"};
    auto D_ = SymbolicSize{"dim"};
    auto Pages_ = SymbolicSize{"pages"};
    auto PageBytes_ = SymbolicSize{"page_bytes"};
    auto MaxPages_ = SymbolicSize{"max_pages"};
    auto MaxLen_ = SymbolicSize{"max_seq_len"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B_, H_, D_}).with_dtype<uint8_t>().with_device(device).verify(q);
    TensorMatcher({Pages_, PageBytes_}).with_dtype<uint8_t>().with_device(device).verify(kv);
    TensorMatcher({B_, H_}).with_dtype<fp32_t>().with_device(device).verify(weights);
    TensorMatcher({B_}).with_dtype<int32_t>().with_device(device).verify(seq_lens);
    TensorMatcher({B_, MaxPages_}).with_dtype<int32_t>().with_device(device).verify(page_table);
    TensorMatcher({B_, MaxLen_}).with_dtype<fp32_t>().with_device(device).verify(out);

    const auto B = static_cast<int>(B_.unwrap());
    const auto H = static_cast<int>(H_.unwrap());
    const auto max_len = static_cast<int>(MaxLen_.unwrap());
    RuntimeCheck(D_.unwrap() == kMqaSm70Dim, "SM70 paged MQA logits are specialized for head dim 128");
    RuntimeCheck(PageBytes_.unwrap() == kMqaSm70PageBytes, "paged KV page must be 64 * 132 bytes");
    RuntimeCheck(H > 0 && H <= kMqaSm70MaxHeads, "SM70 paged MQA logits support 1..64 heads");
    RuntimeCheck(
        q.stride(0) == static_cast<int64_t>(H) * kMqaSm70Dim && q.stride(1) == kMqaSm70Dim && q.stride(2) == 1,
        "q must be contiguous [B, H, 128]");
    RuntimeCheck(kv.stride(1) == 1, "kv pages must be contiguous");
    RuntimeCheck(weights.stride(0) == H && weights.stride(1) == 1, "weights must be contiguous [B, H]");
    RuntimeCheck(page_table.stride(1) == 1, "page_table must be contiguous in the page axis");
    RuntimeCheck(out.stride(0) == max_len && out.stride(1) == 1, "out must be contiguous [B, max_seq_len]");
    if (B == 0 || max_len == 0) {
      return;
    }

    const MqaSm70PagedParams params{
        .q = static_cast<const uint8_t*>(q.data_ptr()),
        .kv = static_cast<const uint8_t*>(kv.data_ptr()),
        .weights = static_cast<const float*>(weights.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .out = static_cast<float*>(out.data_ptr()),
        .heads = H,
        .max_pages = static_cast<int>(MaxPages_.unwrap()),
        .max_seq_len = max_len,
        .page_stride = page_table.stride(0),
    };
    if (H == kMqaTcHeads) {
      const int pages = (max_len + kMqaSm70PageTokens - 1) / kMqaSm70PageTokens;
      LaunchKernel(dim3(pages, B), kMqaSm70PageTokens / 16 * 32, device.unwrap())(paged_tc_kernel, params);
      return;
    }
    const int tiles = (max_len + kMqaSm70Tile - 1) / kMqaSm70Tile;
    LaunchKernel(dim3(tiles, B), 128, device.unwrap())(paged_kernel, params);
  }
};

}  // namespace sglang
