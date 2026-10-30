#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

// Temperature softmax, optionally top-p renormalized (FlashInfer's pivot rule: the
// smallest probability whose mass at or below it reaches 1 - p, ties kept), with each
// row split over many blocks: a step's few rows of a 129K vocabulary on one block each
// leave all but a handful of CUs idle.
//
// Launches over (chunk, row) blocks:
//   stats   per chunk: max of x / T and the sum of exp(x / T - max); zeroes the row's
//           histograms, counters and state
//   hist k  (top-p, k = 0..2) p = exp(x / T - M) / S of the elements matching the
//           pivot's digits so far, a mass histogram of the next digit (11, 10, 10 bits
//           of the float's bit pattern) in 2^-48 fixed point, exact in any order; the
//           row's last block to finish takes the digit and writes the row's state
//   kept    (top-p) per chunk: the mass at or above the pivot; the last block sums them
//   write   p, or p / kept mass at or above the pivot and 0 below it
// Every block takes M and S from the chunks' stats in chunk order, so the result does
// not depend on scheduling.

struct SoftmaxTopPParams {
  const float* __restrict__ logits;
  const float* __restrict__ temps;   // one per request
  const float* __restrict__ top_ps;  // one per request, or null: softmax only
  float* __restrict__ out;
  float* __restrict__ stats;              // [rows, chunks, 2]: max, sum
  float* __restrict__ kept;               // [rows, chunks]
  unsigned long long* __restrict__ work;  // [rows, kRowWords]: histograms, state, counters
  int64_t logits_stride;
  int64_t out_stride;
  uint32_t vocab;
  uint32_t rows_per_request;
  uint32_t chunks;
};

namespace softmax_top_p {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kChunk = 4096;
// the pivot's bit pattern, high digits first: (shift, bits, histogram offset) a pass
__host__ __device__ constexpr int shift_of(int pass) {
  return pass == 0 ? 20 : pass == 1 ? 10 : 0;
}
__host__ __device__ constexpr int bits_of(int pass) {
  return pass == 0 ? 11 : pass == 1 ? 10 : 10;
}
__host__ __device__ constexpr uint32_t hist_offset(int pass) {
  return pass == 0 ? 0 : pass == 1 ? 2048 : 2048 + 1024;
}
constexpr uint32_t kHistWords = 2048 + 1024 + 1024;
// a row's state past its histograms; all zero is the start: no digits, nothing below,
// pivot 0 (top-p 1 keeps every token), crossed
constexpr uint32_t kPrefix = kHistWords;       // the pivot's digits so far
constexpr uint32_t kBelow = kHistWords + 1;    // mass below them
constexpr uint32_t kMissed = kHistWords + 2;   // 1: the row's mass never reached 1 - p
constexpr uint32_t kPivot = kHistWords + 3;    // float bits
constexpr uint32_t kTotal = kHistWords + 4;    // float bits: the kept mass
constexpr uint32_t kCounter = kHistWords + 5;  // 4: hist 0..2 and kept, blocks done
constexpr uint32_t kRowWords = kHistWords + 9;

// Hand-offs between a row's blocks: device-scope atomics and atomic loads, which bypass
// the per-XCD L2s. A device-scope fence would write back the whole L2 of each block.
#if defined(__HIP_PLATFORM_AMD__)
template <typename T>
__device__ __forceinline__ T dev_load(const T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
template <typename T>
__device__ __forceinline__ void dev_store(T* ptr, T v) {
  __hip_atomic_store(ptr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
// this thread's atomics and stores performed
__device__ __forceinline__ void drain() {
  __builtin_amdgcn_s_waitcnt(0);
}
#else
template <typename T>
__device__ __forceinline__ T dev_load(const T* ptr) {
  return *static_cast<const volatile T*>(ptr);
}
template <typename T>
__device__ __forceinline__ void dev_store(T* ptr, T v) {
  *static_cast<volatile T*>(ptr) = v;
}
__device__ __forceinline__ void drain() {
  __threadfence();
}
#endif

__device__ __forceinline__ unsigned long long fixed_mass(float x) {
  return static_cast<unsigned long long>(x * 0x1p48f);
}

__device__ __forceinline__ float block_max(float* buf, float v) {
  buf[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t off = kThreads / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + off]);
    __syncthreads();
  }
  const float r = buf[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float block_sum(float* buf, float v) {
  buf[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t off = kThreads / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
    __syncthreads();
  }
  const float r = buf[0];
  __syncthreads();
  return r;
}

struct RowState {
  float inv_t;  // 1 / temperature
  float max;    // of x / T over the row
  float sum;    // of exp(x / T - max) over the row
};

// the row's max and sum from every chunk's, in chunk order
__device__ __forceinline__ RowState row_state(const SoftmaxTopPParams& p, uint32_t row) {
  RowState s;
  s.inv_t = 1.0f / p.temps[row / p.rows_per_request];
  const float* st = p.stats + static_cast<int64_t>(row) * p.chunks * 2;
  float m = -INFINITY;
  for (uint32_t c = 0; c < p.chunks; ++c)
    m = fmaxf(m, st[2 * c]);
  float sum = 0.0f;
  for (uint32_t c = 0; c < p.chunks; ++c)
    sum += st[2 * c + 1] * __expf(st[2 * c] - m);
  s.max = m;
  s.sum = sum;
  return s;
}

__device__ __forceinline__ float prob(const RowState& s, float x) {
  return __expf(x * s.inv_t - s.max) / s.sum;
}

__device__ __forceinline__ unsigned long long row_target(const SoftmaxTopPParams& p, uint32_t row) {
  const float floor_mass = 1.0f - p.top_ps[row / p.rows_per_request];
  return floor_mass > 0.0f ? fixed_mass(floor_mass) : 0ull;
}

// true on the row's last block to arrive at ``counter``: every other block's atomics
// and device stores were performed before it counted
__device__ __forceinline__ bool last_block(unsigned long long* counter, uint32_t chunks, bool* s_last) {
  drain();
  __syncthreads();
  if (threadIdx.x == 0) *s_last = atomicAdd(counter, 1ull) == chunks - 1;
  __syncthreads();
  return *s_last;
}

// the first bin, ascending, where ``below`` plus the bins so far reaches ``target``
// (bins if none), and the mass below it
template <int kPass>
__device__ void pick_digit(
    const unsigned long long* h,
    unsigned long long below,
    unsigned long long target,
    unsigned long long* buf,
    uint32_t* s_digit,
    uint32_t* digit_out,
    unsigned long long* below_out,
    unsigned long long* total_out = nullptr) {
  constexpr uint32_t bins = 1u << bits_of(kPass);
  constexpr uint32_t per = bins / kThreads > 0 ? bins / kThreads : 1u;
  const bool owner = threadIdx.x * per < bins;
  // every load issued before any is used: one round trip, not ``per``
  unsigned long long v[per];
#pragma unroll
  for (uint32_t j = 0; j < per; ++j)
    v[j] = owner ? dev_load(&h[threadIdx.x * per + j]) : 0ull;
  unsigned long long local = 0;
#pragma unroll
  for (uint32_t j = 0; j < per; ++j)
    local += v[j];
  buf[threadIdx.x] = local;
  if (threadIdx.x == 0) *s_digit = bins;
  __syncthreads();
  for (uint32_t off = 1; off < kThreads; off <<= 1) {
    const unsigned long long add = threadIdx.x >= off ? buf[threadIdx.x - off] : 0ull;
    __syncthreads();
    buf[threadIdx.x] += add;
    __syncthreads();
  }
  const unsigned long long excl = buf[threadIdx.x] - local;
  // every bin's mass (an inclusive scan's last), before the scan buffer is reused
  if (total_out != nullptr) *total_out = buf[kThreads - 1];
  unsigned long long run = below + excl;
  if (owner) {
    for (uint32_t j = 0; j < per; ++j) {
      if (run + v[j] >= target) {
        atomicMin(s_digit, threadIdx.x * per + j);  // the smallest owner's bins come first
        break;
      }
      run += v[j];
    }
  }
  __syncthreads();
  const uint32_t digit = *s_digit;
  if (digit < bins && threadIdx.x == digit / per) {
    *digit_out = digit;
    *below_out = run;  // this thread stopped at the digit: the mass below it
  }
  if (digit >= bins && threadIdx.x == 0) *digit_out = bins;
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads) stats_kernel(const __grid_constant__ SoftmaxTopPParams p) {
  __shared__ float buf[kThreads];
  const uint32_t row = blockIdx.y, chunk = blockIdx.x;
  const float inv_t = 1.0f / p.temps[row / p.rows_per_request];
  const float* x = p.logits + static_cast<int64_t>(row) * p.logits_stride;
  const uint32_t lo = chunk * kChunk, hi = min(lo + kChunk, p.vocab);
  float m = -INFINITY;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads)
    m = fmaxf(m, x[i] * inv_t);
  m = block_max(buf, m);
  float s = 0.0f;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads)
    s += __expf(x[i] * inv_t - m);
  s = block_sum(buf, s);
  if (threadIdx.x == 0) {
    p.stats[(static_cast<int64_t>(row) * p.chunks + chunk) * 2] = m;
    p.stats[(static_cast<int64_t>(row) * p.chunks + chunk) * 2 + 1] = s;
  }
  if (p.top_ps != nullptr) {
    unsigned long long* w = p.work + static_cast<int64_t>(row) * kRowWords;
    for (uint32_t i = chunk * kThreads + threadIdx.x; i < kRowWords; i += p.chunks * kThreads)
      w[i] = 0;
  }
}

template <int kPass>
__global__ void __launch_bounds__(kThreads) hist_kernel(const __grid_constant__ SoftmaxTopPParams p) {
  constexpr uint32_t bins = 1u << bits_of(kPass);
  constexpr int shift = shift_of(kPass);
  __shared__ unsigned long long lhist[bins];
  __shared__ unsigned long long buf[kThreads];
  __shared__ uint32_t s_digit, s_digit_out;
  __shared__ unsigned long long s_below_out;
  __shared__ bool s_last;
  const uint32_t row = blockIdx.y, chunk = blockIdx.x;
  const unsigned long long target = row_target(p, row);
  if (target == 0) return;  // top-p 1: every token kept
  unsigned long long* w = p.work + static_cast<int64_t>(row) * kRowWords;
  if (w[kMissed] != 0) return;
  const RowState s = row_state(p, row);
  const uint32_t prefix = static_cast<uint32_t>(w[kPrefix]);
  for (uint32_t i = threadIdx.x; i < bins; i += kThreads)
    lhist[i] = 0;
  __syncthreads();
  const float* x = p.logits + static_cast<int64_t>(row) * p.logits_stride;
  const uint32_t lo = chunk * kChunk, hi = min(lo + kChunk, p.vocab);
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
    const float pr = prob(s, x[i]);
    const uint32_t b = __float_as_uint(pr);
    if ((b >> (shift + bits_of(kPass))) == prefix) atomicAdd(&lhist[(b >> shift) & (bins - 1)], fixed_mass(pr));
  }
  __syncthreads();
  unsigned long long* gh = w + hist_offset(kPass);
  for (uint32_t i = threadIdx.x; i < bins; i += kThreads) {
    if (lhist[i] != 0) atomicAdd(&gh[i], lhist[i]);
  }
  if (!last_block(&w[kCounter + kPass], p.chunks, &s_last)) return;
  pick_digit<kPass>(gh, w[kBelow], target, buf, &s_digit, &s_digit_out, &s_below_out);
  if (threadIdx.x == 0) {
    if (s_digit_out >= bins) {
      // the row's mass never reached 1 - p (rounding): keep the largest, exp(0) / S
      w[kMissed] = 1;
      w[kPivot] = __float_as_uint(1.0f / s.sum);
    } else {
      const uint32_t next = (prefix << bits_of(kPass)) | s_digit_out;
      w[kPrefix] = next;
      w[kBelow] = s_below_out;
      if (kPass == 2) w[kPivot] = next;  // 31 bits: the pivot's own pattern
    }
  }
}

__global__ void __launch_bounds__(kThreads) kept_kernel(const __grid_constant__ SoftmaxTopPParams p) {
  __shared__ float fbuf[kThreads];
  __shared__ bool s_last;
  const uint32_t row = blockIdx.y, chunk = blockIdx.x;
  const RowState s = row_state(p, row);
  unsigned long long* w = p.work + static_cast<int64_t>(row) * kRowWords;
  const float pivot = __uint_as_float(static_cast<uint32_t>(w[kPivot]));
  const float* x = p.logits + static_cast<int64_t>(row) * p.logits_stride;
  const uint32_t lo = chunk * kChunk, hi = min(lo + kChunk, p.vocab);
  float k = 0.0f;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
    const float pr = prob(s, x[i]);
    k += pr >= pivot ? pr : 0.0f;
  }
  k = block_sum(fbuf, k);
  float* kp = p.kept + static_cast<int64_t>(row) * p.chunks;
  if (threadIdx.x == 0) dev_store(&kp[chunk], k);
  if (!last_block(&w[kCounter + 3], p.chunks, &s_last)) return;
  // the chunks' masses, a thread each, summed in a fixed tree
  float part = 0.0f;
  for (uint32_t c = threadIdx.x; c < p.chunks; c += kThreads)
    part += dev_load(&kp[c]);
  const float total = block_sum(fbuf, part);
  if (threadIdx.x == 0) w[kTotal] = __float_as_uint(total);
}

__global__ void __launch_bounds__(kThreads) write_kernel(const __grid_constant__ SoftmaxTopPParams p) {
  const uint32_t row = blockIdx.y, chunk = blockIdx.x;
  const RowState s = row_state(p, row);
  float pivot = 0.0f, total = 1.0f;
  if (p.top_ps != nullptr) {
    const unsigned long long* w = p.work + static_cast<int64_t>(row) * kRowWords;
    pivot = __uint_as_float(static_cast<uint32_t>(w[kPivot]));
    total = __uint_as_float(static_cast<uint32_t>(w[kTotal]));
  }
  const float* x = p.logits + static_cast<int64_t>(row) * p.logits_stride;
  float* o = p.out + static_cast<int64_t>(row) * p.out_stride;
  const uint32_t lo = chunk * kChunk, hi = min(lo + kChunk, p.vocab);
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
    const float pr = prob(s, x[i]);
    o[i] = p.top_ps == nullptr ? pr : (pr >= pivot ? pr / total : 0.0f);
  }
}

// ---- one launch: every block of a row resident, the passes split by the row's barriers
//
// The same passes, a barrier in place of each launch boundary: a row's blocks count in,
// then wait for the count (device-scope atomics). Every block then reads the row's
// stats, histogram or kept masses itself, so no block hands a result on. The last
// block out zeroes the row's histograms and counters for the next launch.
constexpr uint32_t kBarrier = kHistWords + 9;  // 5 barriers' counts, then the exits
constexpr uint32_t kFusedRowWords = kBarrier + 6;

__device__ __forceinline__ void row_barrier(unsigned long long* counter, uint32_t chunks) {
  drain();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(counter, 1ull);
    while (dev_load(counter) < chunks)
      __builtin_amdgcn_s_sleep(1);
  }
  __syncthreads();
}

// the row's max and sum from every chunk's (``row_state`` over this launch's stats):
// a thread a chunk, every load in flight at once, summed in a fixed tree
__device__ __forceinline__ RowState row_state_dev(const SoftmaxTopPParams& p, uint32_t row, float* buf) {
  RowState s;
  s.inv_t = 1.0f / p.temps[row / p.rows_per_request];
  const float* st = p.stats + static_cast<int64_t>(row) * p.chunks * 2;
  const bool mine = threadIdx.x < p.chunks;
  const float cm = mine ? dev_load(&st[2 * threadIdx.x]) : -INFINITY;
  const float cs = mine ? dev_load(&st[2 * threadIdx.x + 1]) : 0.0f;
  s.max = block_max(buf, cm);
  s.sum = block_sum(buf, mine ? cs * __expf(cm - s.max) : 0.0f);
  return s;
}

template <int kPass>
__device__ void fused_hist(
    const SoftmaxTopPParams& p,
    const RowState& s,
    unsigned long long* w,
    const float* x,
    uint32_t lo,
    uint32_t hi,
    uint32_t& prefix,
    unsigned long long& below,
    unsigned long long target,
    bool& missed,
    unsigned long long* lhist,
    unsigned long long* buf,
    uint32_t* s_digit,
    uint32_t* s_digit_out,
    unsigned long long* s_below_out,
    unsigned long long* total = nullptr) {
  constexpr uint32_t bins = 1u << bits_of(kPass);
  constexpr int shift = shift_of(kPass);
  for (uint32_t i = threadIdx.x; i < bins; i += kThreads)
    lhist[i] = 0;
  __syncthreads();
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
    const float pr = prob(s, x[i]);
    const uint32_t b = __float_as_uint(pr);
    if ((b >> (shift + bits_of(kPass))) == prefix) atomicAdd(&lhist[(b >> shift) & (bins - 1)], fixed_mass(pr));
  }
  __syncthreads();
  unsigned long long* gh = w + hist_offset(kPass);
  for (uint32_t i = threadIdx.x; i < bins; i += kThreads)
    if (lhist[i] != 0) atomicAdd(&gh[i], lhist[i]);
  row_barrier(&w[kBarrier + 1 + kPass], p.chunks);
  pick_digit<kPass>(gh, below, target, buf, s_digit, s_digit_out, s_below_out, total);
  if (*s_digit_out >= bins) {
    missed = true;
  } else {
    prefix = (prefix << bits_of(kPass)) | *s_digit_out;
    below = *s_below_out;
  }
}

__global__ void __launch_bounds__(kThreads) fused_kernel(const __grid_constant__ SoftmaxTopPParams p) {
  static_assert(bits_of(0) >= bits_of(1) && bits_of(0) >= bits_of(2), "lhist holds pass 0's bins");
  __shared__ unsigned long long lhist[1u << bits_of(0)];
  __shared__ unsigned long long buf[kThreads];
  __shared__ float fbuf[kThreads];
  __shared__ uint32_t s_digit, s_digit_out;
  __shared__ unsigned long long s_below_out;
  const uint32_t row = blockIdx.y, chunk = blockIdx.x;
  unsigned long long* w = p.work + static_cast<int64_t>(row) * kFusedRowWords;
  const float* x = p.logits + static_cast<int64_t>(row) * p.logits_stride;
  const uint32_t lo = chunk * kChunk, hi = min(lo + kChunk, p.vocab);
  // stats
  {
    const float inv_t = 1.0f / p.temps[row / p.rows_per_request];
    float m = -INFINITY;
    for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads)
      m = fmaxf(m, x[i] * inv_t);
    m = block_max(fbuf, m);
    float sm = 0.0f;
    for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads)
      sm += __expf(x[i] * inv_t - m);
    sm = block_sum(fbuf, sm);
    if (threadIdx.x == 0) {
      dev_store(&p.stats[(static_cast<int64_t>(row) * p.chunks + chunk) * 2], m);
      dev_store(&p.stats[(static_cast<int64_t>(row) * p.chunks + chunk) * 2 + 1], sm);
    }
  }
  row_barrier(&w[kBarrier], p.chunks);
  const RowState s = row_state_dev(p, row, fbuf);
  float pivot = 0.0f, total = 1.0f;
  if (p.top_ps != nullptr) {
    const unsigned long long target = row_target(p, row);
    if (target != 0) {
      uint32_t prefix = 0;
      unsigned long long below = 0, row_mass = 0;
      bool missed = false;
      fused_hist<0>(
          p,
          s,
          w,
          x,
          lo,
          hi,
          prefix,
          below,
          target,
          missed,
          lhist,
          buf,
          &s_digit,
          &s_digit_out,
          &s_below_out,
          &row_mass);
      if (!missed)
        fused_hist<1>(
            p, s, w, x, lo, hi, prefix, below, target, missed, lhist, buf, &s_digit, &s_digit_out, &s_below_out);
      if (!missed)
        fused_hist<2>(
            p, s, w, x, lo, hi, prefix, below, target, missed, lhist, buf, &s_digit, &s_digit_out, &s_below_out);
      // missed (rounding): keep the largest, exp(0) / S
      pivot = missed ? 1.0f / s.sum : __uint_as_float(prefix);
      if (!missed) {
        // the pivot is a whole bit pattern: the mass at or above it is the row's less
        // the mass below it, both exact in the histograms' fixed point
        total = static_cast<float>(row_mass - below) * 0x1p-48f;
      } else {
        float k = 0.0f;
        for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
          const float pr = prob(s, x[i]);
          k += pr >= pivot ? pr : 0.0f;
        }
        k = block_sum(fbuf, k);
        float* kp = p.kept + static_cast<int64_t>(row) * p.chunks;
        if (threadIdx.x == 0) dev_store(&kp[chunk], k);
        row_barrier(&w[kBarrier + 4], p.chunks);
        // the chunks' masses, a thread each, summed in a fixed tree
        float part = 0.0f;
        for (uint32_t c = threadIdx.x; c < p.chunks; c += kThreads)
          part += dev_load(&kp[c]);
        total = block_sum(fbuf, part);
      }
    }
  }
  float* o = p.out + static_cast<int64_t>(row) * p.out_stride;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
    const float pr = prob(s, x[i]);
    o[i] = p.top_ps == nullptr ? pr : (pr >= pivot ? pr / total : 0.0f);
  }
  // the last block out: every block has passed every barrier, so the row's words are free
  __shared__ bool s_last;
  if (threadIdx.x == 0) s_last = atomicAdd(&w[kBarrier + 5], 1ull) == p.chunks - 1;
  __syncthreads();
  if (s_last) {
    for (uint32_t i = threadIdx.x; i < kFusedRowWords; i += kThreads)
      dev_store(&w[i], 0ull);
  }
}

}  // namespace softmax_top_p

struct SoftmaxTopPKernel {
  // ``top_ps`` empty: the temperature softmax alone
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView temps,
      const tvm::ffi::TensorView top_ps,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView stats,
      const tvm::ffi::TensorView kept,
      const tvm::ffi::TensorView work,
      const int64_t rows_per_request,
      const int64_t fused) {
    using namespace host;
    using namespace softmax_top_p;
    auto rows = SymbolicSize{"rows"};
    auto vocab = SymbolicSize{"vocab"};
    auto requests = SymbolicSize{"requests"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({rows, vocab}).with_strides({-1, 1}).with_dtype<float>().with_device(device).verify(logits);
    TensorMatcher({requests}).with_strides({1}).with_dtype<float>().with_device(device).verify(temps);
    TensorMatcher({rows, vocab}).with_strides({-1, 1}).with_dtype<float>().with_device(device).verify(out);
    const auto n = static_cast<uint32_t>(rows.unwrap());
    if (n == 0) return;
    const auto v = static_cast<uint32_t>(vocab.unwrap());
    const uint32_t chunks = (v + kChunk - 1) / kChunk;
    CHECK_HOST(n == static_cast<uint32_t>(requests.unwrap() * rows_per_request));
    CHECK_HOST(stats.numel() >= static_cast<int64_t>(n) * chunks * 2);
    CHECK_HOST(kept.numel() >= static_cast<int64_t>(n) * chunks);
    const bool with_top_p = top_ps.numel() > 0;
    if (with_top_p) {
      TensorMatcher({requests}).with_strides({1}).with_dtype<float>().with_device(device).verify(top_ps);
    }
    if (fused != 0) {
      // the caller's zeroed words, which the launch leaves zeroed
      CHECK_HOST(work.numel() >= static_cast<int64_t>(n) * kFusedRowWords);
    } else if (with_top_p) {
      CHECK_HOST(work.numel() >= static_cast<int64_t>(n) * kRowWords);
    }
    const auto params = SoftmaxTopPParams{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .temps = static_cast<const float*>(temps.data_ptr()),
        .top_ps = with_top_p ? static_cast<const float*>(top_ps.data_ptr()) : nullptr,
        .out = static_cast<float*>(out.data_ptr()),
        .stats = static_cast<float*>(stats.data_ptr()),
        .kept = static_cast<float*>(kept.data_ptr()),
        .work = static_cast<unsigned long long*>(work.data_ptr()),
        .logits_stride = logits.stride(0),
        .out_stride = out.stride(0),
        .vocab = v,
        .rows_per_request = static_cast<uint32_t>(rows_per_request),
        .chunks = chunks,
    };
    const dim3 grid(chunks, n);
    const auto dev = device.unwrap();
    if (fused != 0) {
      // every block resident (the caller checks the grid fits the CUs)
      LaunchKernel(grid, kThreads, dev)(fused_kernel, params);
      return;
    }
    LaunchKernel(grid, kThreads, dev)(stats_kernel, params);
    if (with_top_p) {
      LaunchKernel(grid, kThreads, dev)(hist_kernel<0>, params);
      LaunchKernel(grid, kThreads, dev)(hist_kernel<1>, params);
      LaunchKernel(grid, kThreads, dev)(hist_kernel<2>, params);
      LaunchKernel(grid, kThreads, dev)(kept_kernel, params);
    }
    LaunchKernel(grid, kThreads, dev)(write_kernel, params);
  }
};

}  // namespace sglang
