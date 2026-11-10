// SPDX-License-Identifier: Apache-2.0
// Indexer K-pool decode update and q block-fp8 quant on Volta. Both reproduce
// the torch fallbacks in kpool_sm70.py / act_quant bitwise: precise expf and
// division, left-to-right pool sums, software bf16 and e4m3fn rounding.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::kpool_sm70 {

inline constexpr int kDim = 128;
inline constexpr int kWarp = 32;
inline constexpr float kFp8Max = 448.0f;
inline constexpr float kHadamardScale = 0.08838834764831845f;  // 128 ** -0.5

SGL_DEVICE uint32_t bits(float f) {
  return __float_as_uint(f);
}

// c10::BFloat16 round-to-nearest-even.
SGL_DEVICE float round_bf16(float f) {
  uint32_t b = bits(f);
  if (f != f) {
    return __uint_as_float(0x7FC00000u);
  }
  b += 0x7FFFu + ((b >> 16) & 1u);
  return __uint_as_float(b & 0xFFFF0000u);
}

SGL_DEVICE uint16_t to_bf16_bits(float f) {
  return static_cast<uint16_t>(bits(round_bf16(f)) >> 16);
}

SGL_DEVICE float from_bf16_bits(uint16_t b) {
  return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

// torch/headeronly/util/Float8_e4m3fn.h fp8e4m3fn_from_fp32_value.
SGL_DEVICE uint8_t to_e4m3fn(float f) {
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
  uint32_t f_bits = bits(f);
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  uint8_t result;
  if (f_bits >= fp8_max) {
    result = 0x7f;
  } else if (f_bits < (UINT32_C(121) << 23)) {
    f_bits = bits(__fadd_rn(__uint_as_float(f_bits), __uint_as_float(denorm_mask)));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    const uint32_t mant_odd = (f_bits >> 20) & 1u;
    f_bits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFFu;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }
  return result | static_cast<uint8_t>(sign >> 24);
}

// torch max/clamp propagate NaN; fmaxf would drop it.
SGL_DEVICE float max_nan(float a, float b) {
  return (a != a || a > b) ? a : b;
}

SGL_DEVICE float clamp_nan(float x, float lo, float hi) {
  x = x < lo ? lo : x;
  return x > hi ? hi : x;
}

// torch: ldexp(1, frexp exponent of amax/448, minus one on an exact power of two);
// it divides by a Python scalar as a multiply by its fp32 reciprocal.
SGL_DEVICE float quant_scale(float amax, bool round_scale) {
  const float ratio = __fmul_rn(amax, 1.0f / kFp8Max);
  if (!round_scale) {
    return ratio;
  }
  const float r = ratio < 1.17549435e-38f ? 1.17549435e-38f : ratio;
  if (r != r) {
    return r;
  }
  const uint32_t b = bits(r);
  const int biased = static_cast<int>((b >> 23) & 0xFFu);
  const int pow2 = (b & 0x7FFFFFu) == 0 ? biased - 127 : biased - 126;
  return __uint_as_float(static_cast<uint32_t>(pow2 + 127) << 23);
}

SGL_DEVICE float warp_max_nan(float v) {
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    v = max_nan(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

SGL_DEVICE int64_t load_index(const void* ptr, bool is64, int64_t i) {
  return is64 ? static_cast<const int64_t*>(ptr)[i] : static_cast<const int32_t*>(ptr)[i];
}

template <typename T>
SGL_DEVICE float to_float(T v) {
  if constexpr (std::is_same_v<T, bf16_t>) {
    return from_bf16_bits(reinterpret_cast<const uint16_t&>(v));
  } else {
    return static_cast<float>(v);
  }
}

struct DecodeParams {
  uint8_t* __restrict__ buf;  // [pages, page_bytes]: slots x 128 fp8, then slots fp32 scales
  int64_t page_bytes;
  uint16_t* __restrict__ tail_k;  // bf16 [req_pool, tail, 128]
  uint16_t* __restrict__ tail_score;
  int64_t tail_k_stride0, tail_k_stride1, tail_s_stride0, tail_s_stride1;
  int64_t req_pool_size;
  int64_t tail_size;
  const void* __restrict__ key;  // [n, 128]
  int64_t key_stride;
  const void* __restrict__ score;
  int64_t score_stride;
  const float* __restrict__ ape;  // [pool, 128]
  int64_t ape_stride;
  const int32_t* __restrict__ block_tables;
  int64_t bt_stride0, bt_stride1, bt_cols;
  const void* req_pool_indices;
  const void* positions;
  const void* seq_lens;
  const void* out_cache_loc;
  bool req_is64, pos_is64, seq_is64, loc_is64;
  int64_t slots_per_page;
  bool round_scale;
};

// Pool softmax (two-pass, left-to-right), bf16 round, Hadamard, block fp8.
// Thread d owns channel d; every thread of the block must call it.
template <int kPool>
SGL_DEVICE void compress_pool(
    const float (&k)[kPool],
    float (&s)[kPool],
    const float* __restrict__ ape,
    int64_t ape_stride,
    int d,
    float* s_x,
    float* s_max,
    bool round_scale,
    float& q,
    float& scale) {
  float peak = -INFINITY;
#pragma unroll
  for (int i = 0; i < kPool; ++i) {
    s[i] = __fadd_rn(s[i], ape[i * ape_stride + d]);
    peak = max_nan(peak, s[i]);
  }
  float denom = 0.0f;
  float num = 0.0f;
#pragma unroll
  for (int i = 0; i < kPool; ++i) {
    const float w = expf(__fsub_rn(s[i], peak));
    denom = i == 0 ? w : __fadd_rn(denom, w);
    const float kw = __fmul_rn(k[i], w);
    num = i == 0 ? kw : __fadd_rn(num, kw);
  }
  float x = round_bf16(__fdiv_rn(num, denom));

#pragma unroll
  for (int span = 1; span < kWarp; span <<= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, x, span);
    x = (d & span) ? __fsub_rn(other, x) : __fadd_rn(x, other);
  }
#pragma unroll
  for (int span = kWarp; span < kDim; span <<= 1) {
    s_x[d] = x;
    __syncthreads();
    const float other = s_x[d ^ span];
    __syncthreads();
    x = (d & span) ? __fsub_rn(other, x) : __fadd_rn(x, other);
  }
  x = round_bf16(__fmul_rn(x, kHadamardScale));

  const float wmax = warp_max_nan(fabsf(x));
  if (d % kWarp == 0) {
    s_max[d / kWarp] = wmax;
  }
  __syncthreads();
  float amax = s_max[0];
#pragma unroll
  for (int w = 1; w < kDim / kWarp; ++w) {
    amax = max_nan(amax, s_max[w]);
  }
  amax = amax < 1e-4f ? 1e-4f : amax;
  scale = quant_scale(amax, round_scale);
  q = clamp_nan(__fdiv_rn(x, scale), -kFp8Max, kFp8Max);
  // s_max is reused by the next call.
  __syncthreads();
}

// One block of 128 threads per token row, thread d owns channel d.
template <typename T, int kPool>
__global__ __launch_bounds__(kDim) void decode_update_kernel(const DecodeParams p) {
  __shared__ float s_x[kDim];
  __shared__ float s_max[kDim / kWarp];
  const int64_t row = blockIdx.x;
  const int d = static_cast<int>(threadIdx.x);

  const int64_t req_raw = load_index(p.req_pool_indices, p.req_is64, row);
  const int64_t pos = load_index(p.positions, p.pos_is64, row);
  const int64_t seq_len = load_index(p.seq_lens, p.seq_is64, row);
  const int64_t cache_loc = load_index(p.out_cache_loc, p.loc_is64, row);
  const bool valid = req_raw >= 0 && req_raw < p.req_pool_size && cache_loc != 0 && pos >= 0 && pos < seq_len;
  const int64_t req = req_raw < 0 ? 0 : (req_raw >= p.req_pool_size ? p.req_pool_size - 1 : req_raw);
  const int64_t safe_pos = pos < 0 ? 0 : pos;
  const int64_t slot = safe_pos % kPool;

  const float key = to_float(static_cast<const T*>(p.key)[row * p.key_stride + d]);
  const float score = to_float(static_cast<const T*>(p.score)[row * p.score_stride + d]);

  if (valid && slot == kPool - 1) {
    const int64_t start = safe_pos - slot;
    float k[kPool];
    float s[kPool];
#pragma unroll
    for (int i = 0; i < kPool - 1; ++i) {
      const int64_t phys = (start + i) % p.tail_size;
      k[i] = from_bf16_bits(p.tail_k[req * p.tail_k_stride0 + phys * p.tail_k_stride1 + d]);
      s[i] = from_bf16_bits(p.tail_score[req * p.tail_s_stride0 + phys * p.tail_s_stride1 + d]);
    }
    k[kPool - 1] = key;
    s[kPool - 1] = score;
    float q, scale;
    compress_pool<kPool>(k, s, p.ape, p.ape_stride, d, s_x, s_max, p.round_scale, q, scale);

    const int64_t pool_id = safe_pos / kPool;
    int64_t col = (pool_id / p.slots_per_page) * kPool;
    col = col < 0 ? 0 : (col > p.bt_cols - 1 ? p.bt_cols - 1 : col);
    const int64_t page = p.block_tables[row * p.bt_stride0 + col * p.bt_stride1];
    const int64_t page_slot = pool_id % p.slots_per_page;
    if (page >= 0) {
      uint8_t* page_ptr = p.buf + page * p.page_bytes;
      page_ptr[page_slot * kDim + d] = to_e4m3fn(q);
      if (d == 0) {
        reinterpret_cast<float*>(page_ptr + p.slots_per_page * kDim)[page_slot] = scale;
      }
    }
  }

  if (valid) {
    const int64_t phys = safe_pos % p.tail_size;
    p.tail_k[req * p.tail_k_stride0 + phys * p.tail_k_stride1 + d] = to_bf16_bits(key);
    p.tail_score[req * p.tail_s_stride0 + phys * p.tail_s_stride1 + d] = to_bf16_bits(score);
  }
}

struct VerifyParams {
  uint8_t* __restrict__ buf;
  int64_t page_bytes;
  uint16_t* __restrict__ tail_k;  // bf16 [req_pool, tail, 128]
  uint16_t* __restrict__ tail_score;
  int64_t tail_k_stride0, tail_k_stride1, tail_s_stride0, tail_s_stride1;
  int64_t tail_size;
  const void* __restrict__ key;  // [bs * n, 128]
  int64_t key_stride;
  const void* __restrict__ score;
  int64_t score_stride;
  const float* __restrict__ ape;
  int64_t ape_stride;
  const void* req_pool_indices;  // [bs]
  const void* write_start;
  const void* tail_logical_start;
  const int64_t* __restrict__ write_loc;  // [bs, max_closed]
  int64_t write_loc_stride;
  int64_t max_closed;
  const void* out_cache_loc;  // [bs * n]
  const void* effective_n;  // [bs] or null
  bool req_is64, ws_is64, tls_is64, loc_is64, eff_is64;
  int64_t n;
  int64_t slots_per_page;
  bool round_scale;
};

// Target verify: roll n draft keys into the tail, then compress every pool
// they close. Each pool is compressed exactly as the decode step on its last
// slot would: older slots read back from the bf16 tail, the closing slot at
// the input precision. One block per request, thread d owns channel d.
template <typename T, int kPool>
__global__ __launch_bounds__(kDim) void verify_write_kernel(const VerifyParams p) {
  __shared__ float s_x[kDim];
  __shared__ float s_max[kDim / kWarp];
  const int64_t b = blockIdx.x;
  const int d = static_cast<int>(threadIdx.x);
  if (load_index(p.out_cache_loc, p.loc_is64, b * p.n) == 0) {
    return;
  }
  const int64_t req = load_index(p.req_pool_indices, p.req_is64, b);
  const int64_t write_start = load_index(p.write_start, p.ws_is64, b);
  const T* key = static_cast<const T*>(p.key);
  const T* score = static_cast<const T*>(p.score);

  for (int64_t i = 0; i < p.n; ++i) {
    const int64_t row = b * p.n + i;
    const int64_t phys = (write_start + i) % p.tail_size;
    p.tail_k[req * p.tail_k_stride0 + phys * p.tail_k_stride1 + d] = to_bf16_bits(to_float(key[row * p.key_stride + d]));
    p.tail_score[req * p.tail_s_stride0 + phys * p.tail_s_stride1 + d] =
        to_bf16_bits(to_float(score[row * p.score_stride + d]));
  }

  const int64_t gate_n = p.effective_n != nullptr ? load_index(p.effective_n, p.eff_is64, b) : p.n;
  const int64_t n_pool = (write_start + gate_n) / kPool - write_start / kPool;
  const int64_t base0 = load_index(p.tail_logical_start, p.tls_is64, b);
  for (int64_t c = 0; c < n_pool && c < p.max_closed; ++c) {
    const int64_t base = base0 + c * kPool;
    float k[kPool];
    float s[kPool];
#pragma unroll
    for (int i = 0; i < kPool - 1; ++i) {
      const int64_t phys = (base + i) % p.tail_size;
      k[i] = from_bf16_bits(p.tail_k[req * p.tail_k_stride0 + phys * p.tail_k_stride1 + d]);
      s[i] = from_bf16_bits(p.tail_score[req * p.tail_s_stride0 + phys * p.tail_s_stride1 + d]);
    }
    const int64_t last = b * p.n + (base + kPool - 1 - write_start);
    k[kPool - 1] = to_float(key[last * p.key_stride + d]);
    s[kPool - 1] = to_float(score[last * p.score_stride + d]);
    float q, scale;
    compress_pool<kPool>(k, s, p.ape, p.ape_stride, d, s_x, s_max, p.round_scale, q, scale);

    const int64_t loc = p.write_loc[b * p.write_loc_stride + c];
    uint8_t* page_ptr = p.buf + (loc / p.slots_per_page) * p.page_bytes;
    const int64_t page_slot = loc % p.slots_per_page;
    page_ptr[page_slot * kDim + d] = to_e4m3fn(q);
    if (d == 0) {
      reinterpret_cast<float*>(page_ptr + p.slots_per_page * kDim)[page_slot] = scale;
    }
  }
}

// One warp per 128-wide group, four channels per lane.
template <typename T>
__global__ __launch_bounds__(128) void act_quant_kernel(
    uint8_t* __restrict__ y, float* __restrict__ s, const T* __restrict__ x, int64_t groups, bool round_scale) {
  const int64_t g = static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (g >= groups) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x % kWarp);
  float v[kDim / kWarp];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kDim / kWarp; ++j) {
    v[j] = to_float(x[g * kDim + j * kWarp + lane]);
    amax = j == 0 ? fabsf(v[j]) : max_nan(amax, fabsf(v[j]));
  }
  amax = warp_max_nan(amax);
  amax = amax < 1e-4f ? 1e-4f : amax;
  const float scale = quant_scale(amax, round_scale);
#pragma unroll
  for (int j = 0; j < kDim / kWarp; ++j) {
    y[g * kDim + j * kWarp + lane] = to_e4m3fn(clamp_nan(__fdiv_rn(v[j], scale), -kFp8Max, kFp8Max));
  }
  if (lane == 0) {
    s[g] = scale;
  }
}

inline bool is_int64(const tvm::ffi::TensorView& t) {
  host::RuntimeCheck(t.dtype().code == kDLInt && (t.dtype().bits == 32 || t.dtype().bits == 64),
                     "index tensors must be int32 or int64");
  return t.dtype().bits == 64;
}

template <typename T>
void launch_decode(const DecodeParams& p, int64_t rows, int64_t pool, DLDevice device) {
  switch (pool) {
    case 2:
      host::LaunchKernel(rows, kDim, device)(decode_update_kernel<T, 2>, p);
      break;
    case 4:
      host::LaunchKernel(rows, kDim, device)(decode_update_kernel<T, 4>, p);
      break;
    case 8:
      host::LaunchKernel(rows, kDim, device)(decode_update_kernel<T, 8>, p);
      break;
    default:
      host::RuntimeCheck(false, "kpool size must be 2, 4 or 8");
  }
}

/// \brief Decode-step k-pool update: roll the current key into the request
///        tail and, on a pool's last slot, write the compressed fp8 key.
/// \param buf uint8 [pages, page_bytes], index-K cache of one layer.
/// \param tail_k bf16 [req_pool, tail, 128]; tail_score likewise.
/// \param key fp16/bf16 [n, 128]; score the same dtype and shape.
/// \param ape fp32 [pool, 128].
/// \param block_tables int32 [>= n, cols]; the index tensors int32 or int64 [>= n].
void decode_update(tvm::ffi::TensorView buf,
                   tvm::ffi::TensorView tail_k,
                   tvm::ffi::TensorView tail_score,
                   tvm::ffi::TensorView key,
                   tvm::ffi::TensorView score,
                   tvm::ffi::TensorView ape,
                   tvm::ffi::TensorView block_tables,
                   tvm::ffi::TensorView req_pool_indices,
                   tvm::ffi::TensorView positions,
                   tvm::ffi::TensorView seq_lens,
                   tvm::ffi::TensorView out_cache_loc,
                   int64_t slots_per_page,
                   bool round_scale) {
  using namespace host;
  SymbolicSize n = {"rows"};
  SymbolicSize pool = {"pool"};
  SymbolicSize pages = {"pages"};
  SymbolicSize page_bytes = {"page_bytes"};
  SymbolicSize reqs = {"req_pool"};
  SymbolicSize tail = {"tail"};
  SymbolicSize cols = {"cols"};
  SymbolicSize bt_rows = {"bt_rows"};
  SymbolicDType dtype;
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({pages, page_bytes}).with_dtype<uint8_t>().with_strides({-1, 1}).with_device(device_).verify(buf);
  TensorMatcher({reqs, tail, kDim}).with_dtype<bf16_t>().with_strides({-1, -1, 1}).with_device(device_).verify(tail_k);
  TensorMatcher({reqs, tail, kDim})
      .with_dtype<bf16_t>()
      .with_strides({-1, -1, 1})
      .with_device(device_)
      .verify(tail_score);
  TensorMatcher({n, kDim}).with_dtype<fp16_t, bf16_t>(dtype).with_strides({-1, 1}).with_device(device_).verify(key);
  TensorMatcher({n, kDim}).with_dtype<fp16_t, bf16_t>(dtype).with_strides({-1, 1}).with_device(device_).verify(score);
  TensorMatcher({pool, kDim}).with_dtype<fp32_t>().with_strides({-1, 1}).with_device(device_).verify(ape);
  TensorMatcher({bt_rows, cols}).with_dtype<int32_t>().with_device(device_).verify(block_tables);
  const int64_t rows = n.unwrap();
  RuntimeCheck(req_pool_indices.size(0) >= rows && positions.size(0) >= rows && seq_lens.size(0) >= rows &&
                   out_cache_loc.size(0) >= rows && bt_rows.unwrap() >= rows,
               "index tensors shorter than the key rows");
  RuntimeCheck(tail.unwrap() >= pool.unwrap(), "tail must hold a full pool");
  RuntimeCheck(page_bytes.unwrap() >= slots_per_page * (kDim + 4) && page_bytes.unwrap() % 4 == 0,
               "page too small for slots_per_page");
  if (rows == 0) {
    return;
  }
  const DecodeParams p{
      .buf = static_cast<uint8_t*>(buf.data_ptr()),
      .page_bytes = buf.stride(0),
      .tail_k = static_cast<uint16_t*>(tail_k.data_ptr()),
      .tail_score = static_cast<uint16_t*>(tail_score.data_ptr()),
      .tail_k_stride0 = tail_k.stride(0),
      .tail_k_stride1 = tail_k.stride(1),
      .tail_s_stride0 = tail_score.stride(0),
      .tail_s_stride1 = tail_score.stride(1),
      .req_pool_size = reqs.unwrap(),
      .tail_size = tail.unwrap(),
      .key = key.data_ptr(),
      .key_stride = key.stride(0),
      .score = score.data_ptr(),
      .score_stride = score.stride(0),
      .ape = static_cast<const float*>(ape.data_ptr()),
      .ape_stride = ape.stride(0),
      .block_tables = static_cast<const int32_t*>(block_tables.data_ptr()),
      .bt_stride0 = block_tables.stride(0),
      .bt_stride1 = block_tables.stride(1),
      .bt_cols = cols.unwrap(),
      .req_pool_indices = req_pool_indices.data_ptr(),
      .positions = positions.data_ptr(),
      .seq_lens = seq_lens.data_ptr(),
      .out_cache_loc = out_cache_loc.data_ptr(),
      .req_is64 = is_int64(req_pool_indices),
      .pos_is64 = is_int64(positions),
      .seq_is64 = is_int64(seq_lens),
      .loc_is64 = is_int64(out_cache_loc),
      .slots_per_page = slots_per_page,
      .round_scale = round_scale,
  };
  if (dtype.is_type<fp16_t>()) {
    launch_decode<fp16_t>(p, rows, pool.unwrap(), device_.unwrap());
  } else {
    launch_decode<bf16_t>(p, rows, pool.unwrap(), device_.unwrap());
  }
}

template <typename T>
void launch_verify(const VerifyParams& p, int64_t bs, int64_t pool, DLDevice device) {
  switch (pool) {
    case 2:
      host::LaunchKernel(bs, kDim, device)(verify_write_kernel<T, 2>, p);
      break;
    case 4:
      host::LaunchKernel(bs, kDim, device)(verify_write_kernel<T, 4>, p);
      break;
    case 8:
      host::LaunchKernel(bs, kDim, device)(verify_write_kernel<T, 8>, p);
      break;
    default:
      host::RuntimeCheck(false, "kpool size must be 2, 4 or 8");
  }
}

/// \brief Target-verify k-pool write for n draft tokens per request.
/// \param key fp16/bf16 [bs * n, 128]; score the same dtype and shape.
/// \param write_loc int64 [bs, max_closed]; req_pool_indices, write_start,
///        tail_logical_start int32 or int64 [bs]; out_cache_loc [bs * n].
/// \param effective_n int32/int64 [bs], or an empty tensor for n.
void verify_write(tvm::ffi::TensorView buf,
                  tvm::ffi::TensorView tail_k,
                  tvm::ffi::TensorView tail_score,
                  tvm::ffi::TensorView key,
                  tvm::ffi::TensorView score,
                  tvm::ffi::TensorView ape,
                  tvm::ffi::TensorView req_pool_indices,
                  tvm::ffi::TensorView write_start,
                  tvm::ffi::TensorView tail_logical_start,
                  tvm::ffi::TensorView write_loc,
                  tvm::ffi::TensorView out_cache_loc,
                  tvm::ffi::TensorView effective_n,
                  int64_t n,
                  int64_t slots_per_page,
                  bool round_scale) {
  using namespace host;
  SymbolicSize rows = {"rows"};
  SymbolicSize pool = {"pool"};
  SymbolicSize pages = {"pages"};
  SymbolicSize page_bytes = {"page_bytes"};
  SymbolicSize reqs = {"req_pool"};
  SymbolicSize tail = {"tail"};
  SymbolicSize bs = {"bs"};
  SymbolicSize closed = {"max_closed"};
  SymbolicDType dtype;
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({pages, page_bytes}).with_dtype<uint8_t>().with_strides({-1, 1}).with_device(device_).verify(buf);
  TensorMatcher({reqs, tail, kDim}).with_dtype<bf16_t>().with_strides({-1, -1, 1}).with_device(device_).verify(tail_k);
  TensorMatcher({reqs, tail, kDim})
      .with_dtype<bf16_t>()
      .with_strides({-1, -1, 1})
      .with_device(device_)
      .verify(tail_score);
  TensorMatcher({rows, kDim}).with_dtype<fp16_t, bf16_t>(dtype).with_strides({-1, 1}).with_device(device_).verify(key);
  TensorMatcher({rows, kDim}).with_dtype<fp16_t, bf16_t>(dtype).with_strides({-1, 1}).with_device(device_).verify(score);
  TensorMatcher({pool, kDim}).with_dtype<fp32_t>().with_strides({-1, 1}).with_device(device_).verify(ape);
  TensorMatcher({bs, closed}).with_dtype<int64_t>().with_strides({-1, 1}).with_device(device_).verify(write_loc);
  RuntimeCheck(n > 0 && rows.unwrap() == bs.unwrap() * n, "key rows must be bs * n");
  RuntimeCheck(req_pool_indices.size(0) >= bs.unwrap() && write_start.size(0) >= bs.unwrap() &&
                   tail_logical_start.size(0) >= bs.unwrap() && out_cache_loc.size(0) >= rows.unwrap(),
               "index tensors shorter than the batch");
  RuntimeCheck(effective_n.numel() == 0 || effective_n.size(0) >= bs.unwrap(), "effective_n shorter than the batch");
  RuntimeCheck(tail.unwrap() >= pool.unwrap() + n - 1, "tail must hold a pool plus the drafts");
  RuntimeCheck(page_bytes.unwrap() >= slots_per_page * (kDim + 4) && page_bytes.unwrap() % 4 == 0,
               "page too small for slots_per_page");
  if (bs.unwrap() == 0) {
    return;
  }
  const VerifyParams p{
      .buf = static_cast<uint8_t*>(buf.data_ptr()),
      .page_bytes = buf.stride(0),
      .tail_k = static_cast<uint16_t*>(tail_k.data_ptr()),
      .tail_score = static_cast<uint16_t*>(tail_score.data_ptr()),
      .tail_k_stride0 = tail_k.stride(0),
      .tail_k_stride1 = tail_k.stride(1),
      .tail_s_stride0 = tail_score.stride(0),
      .tail_s_stride1 = tail_score.stride(1),
      .tail_size = tail.unwrap(),
      .key = key.data_ptr(),
      .key_stride = key.stride(0),
      .score = score.data_ptr(),
      .score_stride = score.stride(0),
      .ape = static_cast<const float*>(ape.data_ptr()),
      .ape_stride = ape.stride(0),
      .req_pool_indices = req_pool_indices.data_ptr(),
      .write_start = write_start.data_ptr(),
      .tail_logical_start = tail_logical_start.data_ptr(),
      .write_loc = static_cast<const int64_t*>(write_loc.data_ptr()),
      .write_loc_stride = write_loc.stride(0),
      .max_closed = closed.unwrap(),
      .out_cache_loc = out_cache_loc.data_ptr(),
      .effective_n = effective_n.numel() == 0 ? nullptr : effective_n.data_ptr(),
      .req_is64 = is_int64(req_pool_indices),
      .ws_is64 = is_int64(write_start),
      .tls_is64 = is_int64(tail_logical_start),
      .loc_is64 = is_int64(out_cache_loc),
      .eff_is64 = effective_n.numel() != 0 && is_int64(effective_n),
      .n = n,
      .slots_per_page = slots_per_page,
      .round_scale = round_scale,
  };
  if (dtype.is_type<fp16_t>()) {
    launch_verify<fp16_t>(p, bs.unwrap(), pool.unwrap(), device_.unwrap());
  } else {
    launch_verify<bf16_t>(p, bs.unwrap(), pool.unwrap(), device_.unwrap());
  }
}

/// \brief Block fp8 quant with 128-wide groups.
/// \param y uint8 (e4m3fn bytes) [..., K]; s fp32 [..., K / 128]; x fp16/bf16 [..., K], contiguous.
void act_quant(tvm::ffi::TensorView y, tvm::ffi::TensorView s, tvm::ffi::TensorView x, bool round_scale) {
  using namespace host;
  RuntimeCheck(x.is_contiguous() && y.is_contiguous() && s.is_contiguous(), "act_quant needs contiguous tensors");
  RuntimeCheck(x.numel() % kDim == 0 && y.numel() == x.numel() && s.numel() == x.numel() / kDim,
               "act_quant shape mismatch");
  RuntimeCheck(y.dtype().bits == 8 && s.dtype().code == kDLFloat && s.dtype().bits == 32, "act_quant dtypes");
  const int64_t groups = x.numel() / kDim;
  if (groups == 0) {
    return;
  }
  constexpr int kGroupsPerBlock = 4;
  const auto blocks = static_cast<uint32_t>((groups + kGroupsPerBlock - 1) / kGroupsPerBlock);
  const DLDevice device = x.device();
  auto* yp = static_cast<uint8_t*>(y.data_ptr());
  auto* sp = static_cast<float*>(s.data_ptr());
  if (x.dtype().code == kDLFloat && x.dtype().bits == 16) {
    LaunchKernel(blocks, kGroupsPerBlock * kWarp, device)(
        act_quant_kernel<fp16_t>, yp, sp, static_cast<const fp16_t*>(x.data_ptr()), groups, round_scale);
  } else {
    RuntimeCheck(x.dtype().code == kDLBfloat && x.dtype().bits == 16, "act_quant x must be fp16 or bf16");
    LaunchKernel(blocks, kGroupsPerBlock * kWarp, device)(
        act_quant_kernel<bf16_t>, yp, sp, static_cast<const bf16_t*>(x.data_ptr()), groups, round_scale);
  }
}

}  // namespace sglang::kpool_sm70
