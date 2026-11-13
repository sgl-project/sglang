// SPDX-License-Identifier: Apache-2.0
// Short convolution over decode / MTP verify / small extend rows, equal to the
// Triton _sconv_kernel in sconv.py: optional dynamic weights, routed top-k row
// gather, conv cache shift (decode) or per-row conv windows (verify), and the
// residual-update epilogues.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

namespace sglang::inkling2_sconv {

constexpr int kVec = 4;
constexpr int kThreads = 128;

using bf16 = __nv_bfloat16;

// kVec contiguous elements of T, kept in their storage type.
template <typename T>
struct Vec {
  using Raw = std::
      conditional_t<sizeof(T) * kVec % 16 == 0, uint4, std::conditional_t<sizeof(T) * kVec % 8 == 0, uint2, uint32_t>>;
  static constexpr int kRaw = sizeof(T) * kVec / sizeof(Raw);
  Raw raw[kRaw];
  __device__ __forceinline__ void load(T const* p) {
#pragma unroll
    for (int i = 0; i < kRaw; ++i)
      raw[i] = reinterpret_cast<Raw const*>(p)[i];
  }
  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < kRaw; ++i)
      raw[i] = Raw{};
  }
  __device__ __forceinline__ float operator[](int i) const {
    T v = reinterpret_cast<T const*>(raw)[i];
    if constexpr (sizeof(T) == 2)
      return __bfloat162float(v);
    else
      return v;
  }
  __device__ __forceinline__ void set(float const (&v)[kVec]) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      if constexpr (sizeof(T) == 2)
        reinterpret_cast<T*>(raw)[i] = __float2bfloat16_rn(v[i]);
      else
        reinterpret_cast<T*>(raw)[i] = v[i];
    }
  }
  __device__ __forceinline__ void store(T* p) const {
#pragma unroll
    for (int i = 0; i < kRaw; ++i)
      reinterpret_cast<Raw*>(p)[i] = raw[i];
  }
};

__device__ __forceinline__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// v rounded to T, as Triton's .to(T).to(float32).
template <typename T>
__device__ __forceinline__ float round_to(float v) {
  if constexpr (std::is_same_v<T, bf16>)
    return round_bf16(v);
  else
    return v;
}

__device__ __forceinline__ float to_f(float v) {
  return v;
}
__device__ __forceinline__ float to_f(bf16 v) {
  return __bfloat162float(v);
}

struct Params {
  void const* x;
  int64_t sx;
  void const* weight;
  int64_t sw0, sw1;
  bf16 const* dynamic;
  void* cache;
  int64_t sc0, sc1;
  int64_t const* indices;
  bool const* mask;
  int64_t const* cu;
  int32_t const* seq;
  bf16* y;
  bool const* track_mask;
  int64_t const* track_indices;
  int32_t const* src2dst;
  void const* partial;
  void const* attention;
  void* out;
  void* window;
  int64_t swb, swt, sww;
  int d, head_dim;
  float scale;
  bool packed_weight;  // static [D, W] weights, row-major and 16-byte aligned
};

template <int TOP_K, typename TX>
struct Gather {
  Vec<TX> rows[TOP_K];
  __device__ __forceinline__ void load(Params const& p, int r, int col) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
      rows[k].load(static_cast<TX const*>(p.x) + int64_t(p.src2dst[r * TOP_K + k]) * p.sx + col);
  }
  // bf16(sum of the rows in top-k order), as post_reorder rounds it (FP32 rows:
  // the Triton kernel's ROUND_BF16).
  __device__ __forceinline__ float operator[](int i) const {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
      acc = __fadd_rn(acc, rows[k][i]);
    return round_bf16(acc);
  }
};

// EPILOGUE 0 stores y; 1 stores partial + attention + y; 2 attention + y.
template <
    int W,
    int TOP_K,
    bool DYNAMIC,
    bool DECODE,
    bool WINDOW,
    bool RESIDUAL,
    int EPILOGUE,
    typename TX,
    typename TW,
    typename TC,
    typename TA,
    typename TO>
__global__ void __launch_bounds__(kThreads) sconv_kernel(Params p) {
  // Decode and MTP verify read their metadata, static weights, and conv cache
  // before waiting on the previous kernel, which writes none of them.
  constexpr bool kEarly = DECODE || WINDOW;
  if constexpr (!kEarly) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }
  int const row = blockIdx.x;
  int const col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  if (col >= p.d) return;
  int const seq = DECODE ? row : p.seq[row];
  int const bos = DECODE ? row : int(p.cu[seq]);
  int64_t const slot = p.indices[seq];
  bool const in_pool = p.mask[seq];
  bool const has_cache = in_pool && slot > 0;
  TC* cache = static_cast<TC*>(p.cache);
  TC* slot_cache = cache + slot * p.sc0 + col;

  Vec<TC> lag_cache[W];
#pragma unroll
  for (int lag = 1; lag < W; ++lag) {
    int const src = row - lag;
    if (src < bos && (WINDOW ? in_pool : has_cache))
      lag_cache[lag].load(slot_cache + int64_t(src - bos + W - 1) * p.sc1);
    else
      lag_cache[lag].zero();
  }

  float weight[W][kVec];
  TW const* w = static_cast<TW const*>(p.weight);
  if constexpr (!DYNAMIC) {
    if (p.packed_weight) {
#pragma unroll
      for (int chunk = 0; chunk < W; ++chunk) {
        Vec<TW> v;
        v.load(w + int64_t(col) * W + chunk * kVec);
#pragma unroll
        for (int j = 0; j < kVec; ++j)
          weight[(chunk * kVec + j) % W][(chunk * kVec + j) / W] = round_to<TX>(v[j]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i)
#pragma unroll
        for (int tap = 0; tap < W; ++tap)
          weight[tap][i] = round_to<TX>(to_f(w[int64_t(col + i) * p.sw0 + tap * p.sw1]));
    }
  }

  if constexpr (kEarly) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }
  constexpr bool kGather = TOP_K > 0;
  constexpr int kRowLoads = DECODE ? 1 : W;
  using Row = std::conditional_t<kGather, Gather<kGather ? TOP_K : 1, TX>, Vec<TX>>;
  Row xs[kRowLoads];
#pragma unroll
  for (int lag = 0; lag < kRowLoads; ++lag) {
    int const src = max(row - lag, 0);
    if constexpr (kGather)
      xs[lag].load(p, src, col);
    else
      xs[lag].load(static_cast<TX const*>(p.x) + int64_t(src) * p.sx + col);
  }
  if constexpr (DYNAMIC) {
#pragma unroll
    for (int lag = 0; lag < W; ++lag) {
      Vec<TW> v;
      v.load(w + lag * p.sw0 + col);
      float dyn = __bfloat162float(p.dynamic[(int64_t(row) * (p.d / p.head_dim) + col / p.head_dim) * W + lag]);
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        weight[lag][i] = __fadd_rn(v[i], dyn);
    }
  }

  float current[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    current[i] = xs[0][i];

  float acc[kVec];
#pragma unroll
  for (int tap = 0; tap < W; ++tap) {
    int const lag = DYNAMIC ? tap : W - 1 - tap;
    bool const from_x = row - lag >= bos;
    float value[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      if (lag == 0 || (kGather && DECODE && from_x))
        value[i] = current[i];
      else if (!from_x)
        value[i] = has_cache ? lag_cache[lag][i] : 0.0f;
      else
        value[i] = xs[kRowLoads > 1 ? lag : 0][i];
    }
    if constexpr (WINDOW) {
      if (lag < W - 1 && in_pool) {
        Vec<TC> raw;
        if (!from_x && !has_cache) {
          raw = lag_cache[lag];
        } else {
          raw.set(value);
        }
        raw.store(static_cast<TC*>(p.window) + seq * p.swb + (row - bos) * p.swt + (W - 2 - lag) * p.sww + col);
      }
    }
    float const* wt = weight[DYNAMIC ? lag : tap];
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      acc[i] =
          (DYNAMIC && tap == 0) ? __fmul_rn(value[i], wt[i]) : __fmaf_rn(value[i], wt[i], tap == 0 ? 0.0f : acc[i]);
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    if constexpr (RESIDUAL) acc[i] = __fadd_rn(acc[i], current[i]);
    acc[i] = __fmul_rn(acc[i], p.scale);
  }
  if constexpr (EPILOGUE == 0) {
    Vec<bf16> y;
    y.set(acc);
    y.store(p.y + int64_t(row) * p.d + col);
  } else {
    Vec<TA> attention;
    attention.load(static_cast<TA const*>(p.attention) + int64_t(row) * p.d + col);
    float total[kVec];
    if constexpr (EPILOGUE == 1) {
      Vec<TO> partial;
      partial.load(static_cast<TO const*>(p.partial) + int64_t(row) * p.d + col);
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        total[i] = __fadd_rn(__fadd_rn(__fadd_rn(0.0f, partial[i]), attention[i]), round_to<TX>(acc[i]));
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        total[i] = __fadd_rn(attention[i], round_to<TX>(acc[i]));
    }
    Vec<TO> out;
    out.set(total);
    out.store(static_cast<TO*>(p.out) + int64_t(row) * p.d + col);
  }
  if constexpr (DECODE) {
    if (has_cache) {
      bool const track = p.track_mask != nullptr && p.track_mask[seq];
      TC* dest = track ? cache + p.track_indices[seq] * p.sc0 + col : nullptr;
#pragma unroll
      for (int tap = 0; tap < W - 1; ++tap) {
        Vec<TC> value;
        if (tap == W - 2)
          value.set(current);
        else
          value = lag_cache[W - 2 - tap];
        value.store(slot_cache + tap * p.sc1);
        if (track) value.store(dest + tap * p.sc1);
      }
    }
  }
}

template <
    int W,
    int TOP_K,
    bool DYNAMIC,
    bool DECODE,
    bool WINDOW,
    bool RESIDUAL,
    int EPILOGUE,
    typename TX,
    typename TW,
    typename TC,
    typename TA,
    typename TO>
void run(
    tvm::ffi::TensorView x,
    int64_t rows,
    int64_t sx,
    int64_t weight,
    int64_t sw0,
    int64_t sw1,
    int64_t dynamic,
    int64_t cache,
    int64_t sc0,
    int64_t sc1,
    int64_t indices,
    int64_t mask,
    int64_t cu,
    int64_t seq,
    int64_t y,
    int64_t track_mask,
    int64_t track_indices,
    int64_t src2dst,
    int64_t partial,
    int64_t attention,
    int64_t out,
    int64_t window,
    int64_t swb,
    int64_t swt,
    int64_t sww,
    int64_t d,
    int64_t head_dim,
    double scale) {
  host::RuntimeCheck(d % kVec == 0 && sx % kVec == 0, "sconv rows must be 8-element aligned");
  Params p{
      x.data_ptr(),
      sx,
      reinterpret_cast<void const*>(weight),
      sw0,
      sw1,
      reinterpret_cast<bf16 const*>(dynamic),
      reinterpret_cast<void*>(cache),
      sc0,
      sc1,
      reinterpret_cast<int64_t const*>(indices),
      reinterpret_cast<bool const*>(mask),
      reinterpret_cast<int64_t const*>(cu),
      reinterpret_cast<int32_t const*>(seq),
      reinterpret_cast<bf16*>(y),
      reinterpret_cast<bool const*>(track_mask),
      reinterpret_cast<int64_t const*>(track_indices),
      reinterpret_cast<int32_t const*>(src2dst),
      reinterpret_cast<void const*>(partial),
      reinterpret_cast<void const*>(attention),
      reinterpret_cast<void*>(out),
      reinterpret_cast<void*>(window),
      swb,
      swt,
      sww,
      int(d),
      int(head_dim),
      float(scale),
      !DYNAMIC && sw0 == W && sw1 == 1 && weight % 16 == 0};
  int const groups = int(d / kVec);
  host::LaunchKernel(dim3(rows, (groups + kThreads - 1) / kThreads), kThreads, x.device())
      .enable_pdl(true)(sconv_kernel<W, TOP_K, DYNAMIC, DECODE, WINDOW, RESIDUAL, EPILOGUE, TX, TW, TC, TA, TO>, p);
}

}  // namespace sglang::inkling2_sconv
