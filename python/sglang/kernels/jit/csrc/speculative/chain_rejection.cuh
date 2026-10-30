#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

// Classic chain rejection sampling (``speculative_sampling_classic_kernel``'s chain path)
// with each request's vocabulary over many blocks, in one launch. Every block of a
// request redoes the short accept chain; each sums its chunk of the final row's residual
// max(p - q, 0) (p alone when every draft was accepted), then the request's blocks
// meet at a barrier (device-scope atomics, every block resident); each block then takes
// the chunks' sums, and the chunk where the cumulative mass passes u * total finds its
// token. The last block out zeroes the request's counters for the next launch.

struct ChainRejectionParams {
  int32_t* __restrict__ predicts;
  int32_t* __restrict__ accept_index;      // [bs, slots]
  int32_t* __restrict__ accept_token_num;  // [bs]
  const int64_t* __restrict__ candidates;  // [bs, slots]
  const int64_t* __restrict__ retrive_index;
  const float* __restrict__ uniform;        // [bs, >= slots - 1]
  const float* __restrict__ uniform_final;  // [bs]
  const float* __restrict__ target_probs;   // [bs, slots, vocab]
  const float* __restrict__ draft_probs;    // [bs, slots - 1, vocab]
  float* __restrict__ chunk_mass;           // [bs, chunks]
  unsigned long long* __restrict__ work;    // [bs, 2]: barrier, exits (zero between launches)
  int64_t cand_stride, idx_stride, acc_stride, uni_stride;
  int64_t tp_b, tp_s, dp_b, dp_s;
  uint32_t slots;
  uint32_t vocab;
  uint32_t chunks;
};

namespace chain_rejection {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kChunk = 8192;
constexpr uint32_t kPer = kChunk / kThreads;

#if defined(__HIP_PLATFORM_AMD__)
template <typename T>
__device__ __forceinline__ T dev_load(const T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
template <typename T>
__device__ __forceinline__ void dev_store(T* ptr, T v) {
  __hip_atomic_store(ptr, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void drain() {
  __builtin_amdgcn_s_waitcnt(0);
}
__device__ __forceinline__ void pause() {
  __builtin_amdgcn_s_sleep(1);
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
__device__ __forceinline__ void pause() {}
#endif

// inclusive scan of ``v`` over the block in thread order (a fixed tree)
__device__ __forceinline__ float block_scan(float* buf, float v) {
  buf[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t off = 1; off < kThreads; off <<= 1) {
    const float add = threadIdx.x >= off ? buf[threadIdx.x - off] : 0.0f;
    __syncthreads();
    buf[threadIdx.x] += add;
    __syncthreads();
  }
  const float r = buf[threadIdx.x];
  __syncthreads();
  return r;
}

__global__ void __launch_bounds__(kThreads) kernel(const __grid_constant__ ChainRejectionParams p) {
  __shared__ float buf[kThreads];
  __shared__ uint32_t s_accept, s_token;
  __shared__ float s_base;
  __shared__ int64_t s_last_idx;
  __shared__ bool s_last;
  const uint32_t b = blockIdx.y, chunk = blockIdx.x;
  const int64_t* cand = p.candidates + b * p.cand_stride;
  const int64_t* ridx = p.retrive_index + b * p.idx_stride;
  // the accept chain, the same on every block: step s reads row s - 1 until the first
  // reject, so thread s loads its step's test in parallel, then thread 0 walks them
  __shared__ bool s_ok[kThreads];
  const uint32_t step = threadIdx.x;
  if (step >= 1 && step < p.slots) {
    const int64_t tok = cand[step];
    const float pv = p.target_probs[b * p.tp_b + (step - 1) * p.tp_s + tok];
    const float qv = p.draft_probs[b * p.dp_b + (step - 1) * p.dp_s + tok];
    const float coin = p.uniform[b * p.uni_stride + step - 1];
    // a proposal token must have positive probability under its draft distribution
    s_ok[step] = (qv > 0.0f) & (qv <= 1.0f) & (coin * qv < pv);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t n = 0;
    int64_t last = ridx[0];
    for (uint32_t step = 1; step < p.slots; ++step) {
      const int64_t tok = cand[step];
      if (!s_ok[step]) break;
      if (chunk == 0) {
        p.predicts[last] = static_cast<int32_t>(tok);
        last = ridx[step];
        p.accept_index[b * p.acc_stride + step] = static_cast<int32_t>(last);
      } else {
        last = ridx[step];
      }
      ++n;
    }
    if (chunk == 0) {
      p.accept_index[b * p.acc_stride] = static_cast<int32_t>(ridx[0]);
      p.accept_token_num[b] = static_cast<int32_t>(n);
    }
    s_accept = n;
    s_last_idx = last;
    s_token = p.vocab;
  }
  __syncthreads();
  const uint32_t row = s_accept;
  const bool all_accepted = row == p.slots - 1;
  const float* pr = p.target_probs + b * p.tp_b + row * p.tp_s;
  // every draft accepted: the bonus row's p (draft_probs has no row there)
  const float* qr = all_accepted ? nullptr : p.draft_probs + b * p.dp_b + row * p.dp_s;
  // the chunk's residuals into LDS and its mass: thread t loads four entries at
  // 4 t + 1024 j (coalesced, every load in flight), the hit block rereads them below
  // a pad word a kPer run: a thread's run at stride kPer + 1, free of bank conflicts
  __shared__ float vals[kChunk + kChunk / kPer];
  const uint32_t c0 = chunk * kChunk;
  constexpr uint32_t kVec = 4, kRounds = kChunk / (kThreads * kVec);
  float4 pv[kRounds], qv[kRounds];
#pragma unroll
  for (uint32_t j = 0; j < kRounds; ++j) {
    const uint32_t i = c0 + (threadIdx.x + j * kThreads) * kVec;
    // vocab % 4 == 0 (checked on the host), so a vector is wholly in or out
    pv[j] = i < p.vocab ? *reinterpret_cast<const float4*>(pr + i) : make_float4(0, 0, 0, 0);
    qv[j] = (i < p.vocab && !all_accepted) ? *reinterpret_cast<const float4*>(qr + i) : make_float4(0, 0, 0, 0);
  }
  float part = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kRounds; ++j) {
    const float ps[4] = {pv[j].x, pv[j].y, pv[j].z, pv[j].w};
    const float qs[4] = {qv[j].x, qv[j].y, qv[j].z, qv[j].w};
#pragma unroll
    for (uint32_t e = 0; e < kVec; ++e) {
      float v = ps[e];
      if (!all_accepted) {
        // a non-probability q (NaN, inf, negative) counts as 0: the residual is p
        const float q = (qs[e] >= 0.0f) & (qs[e] <= 1.0f) ? qs[e] : 0.0f;
        v = fmaxf(v - q, 0.0f);
      }
      const uint32_t i = (threadIdx.x + j * kThreads) * kVec + e;
      vals[i + i / kPer] = v;
      part += v;
    }
  }
  const float chunk_total = block_scan(buf, part);
  float* cm = p.chunk_mass + static_cast<int64_t>(b) * p.chunks;
  if (threadIdx.x == kThreads - 1) dev_store(&cm[chunk], chunk_total);
  unsigned long long* w = p.work + static_cast<int64_t>(b) * 2;
  drain();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&w[0], 1ull);
    while (dev_load(&w[0]) < p.chunks)
      pause();
  }
  __syncthreads();
  // the chunks' masses in chunk order: the total and this chunk's mass before it
  const float c_mass = threadIdx.x < p.chunks ? dev_load(&cm[threadIdx.x]) : 0.0f;
  const float c_incl = block_scan(buf, c_mass);
  const float total = buf[kThreads - 1 < p.chunks ? kThreads - 1 : p.chunks - 1];
  const float target = p.uniform_final[b] * total;
  // the first chunk whose cumulative mass passes the target
  if (threadIdx.x < p.chunks && c_incl > target && c_incl - c_mass <= target) {
    atomicMin(&s_token, threadIdx.x);
    s_base = c_incl - c_mass;  // the mass before the chunk (one chunk crosses)
  }
  __syncthreads();
  const uint32_t hit = s_token;
  __syncthreads();
  if (threadIdx.x == 0) s_token = p.vocab;
  __syncthreads();
  if (hit == chunk) {
    // thread t walks entries kPer t .. in order, from the mass before them
    float mine = 0.0f;
    for (uint32_t j = 0; j < kPer; ++j)
      mine += vals[threadIdx.x * (kPer + 1) + j];
    float run = s_base + (block_scan(buf, mine) - mine);
    bool found = false;
    for (uint32_t j = 0; j < kPer; ++j) {
      run += vals[threadIdx.x * (kPer + 1) + j];
      if (!found && run > target) {
        atomicMin(&s_token, c0 + threadIdx.x * kPer + j);
        found = true;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      // rounding can leave the crossing just past this chunk's last entry
      const uint32_t tok = s_token < p.vocab ? s_token : min(c0 + kChunk, p.vocab) - 1;
      p.predicts[s_last_idx] = static_cast<int32_t>(tok);
    }
  } else if (hit >= p.chunks && chunk == 0 && threadIdx.x == 0) {
    // the residual's mass is 0 (p == q, numerically): the classic kernel's last token
    p.predicts[s_last_idx] = static_cast<int32_t>(p.vocab - 1);
  }
  // the last block out: every block is past the barrier, so the counters are free
  if (threadIdx.x == 0) s_last = atomicAdd(&w[1], 1ull) == p.chunks - 1;
  __syncthreads();
  if (s_last && threadIdx.x < 2) dev_store(&w[threadIdx.x], 0ull);
}

}  // namespace chain_rejection

struct ChainRejectionKernel {
  static void
  run(const tvm::ffi::TensorView predicts,
      const tvm::ffi::TensorView accept_index,
      const tvm::ffi::TensorView accept_token_num,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView retrive_index,
      const tvm::ffi::TensorView uniform,
      const tvm::ffi::TensorView uniform_final,
      const tvm::ffi::TensorView target_probs,
      const tvm::ffi::TensorView draft_probs,
      const tvm::ffi::TensorView chunk_mass,
      const tvm::ffi::TensorView work) {
    using namespace host;
    using namespace chain_rejection;
    auto bs = SymbolicSize{"bs"};
    auto slots = SymbolicSize{"slots"};
    auto vocab = SymbolicSize{"vocab"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({bs, slots}).with_strides({-1, 1}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({bs, slots}).with_strides({-1, 1}).with_dtype<int64_t>().with_device(device).verify(retrive_index);
    TensorMatcher({bs, slots}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device).verify(accept_index);
    TensorMatcher({bs}).with_strides({1}).with_dtype<int32_t>().with_device(device).verify(accept_token_num);
    TensorMatcher({bs}).with_strides({1}).with_dtype<float>().with_device(device).verify(uniform_final);
    TensorMatcher({bs, slots, vocab})
        .with_strides({-1, -1, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(target_probs);
    const auto n = static_cast<uint32_t>(bs.unwrap());
    if (n == 0) return;
    const auto s = static_cast<uint32_t>(slots.unwrap());
    const auto v = static_cast<uint32_t>(vocab.unwrap());
    const uint32_t chunks = (v + kChunk - 1) / kChunk;
    CHECK_HOST(chunks <= kThreads && s <= kThreads && v % 4 == 0);
    // 16-byte rows: the residual pass loads four entries at a time
    CHECK_HOST(target_probs.stride(0) % 4 == 0 && target_probs.stride(1) % 4 == 0);
    CHECK_HOST(draft_probs.stride(0) % 4 == 0 && draft_probs.stride(1) % 4 == 0);
    CHECK_HOST(reinterpret_cast<uintptr_t>(target_probs.data_ptr()) % 16 == 0);
    CHECK_HOST(reinterpret_cast<uintptr_t>(draft_probs.data_ptr()) % 16 == 0);
    CHECK_HOST(draft_probs.size(0) == n && draft_probs.size(1) >= s - 1 && draft_probs.size(2) == v);
    CHECK_HOST(draft_probs.stride(2) == 1);
    CHECK_HOST(uniform.size(0) == n && uniform.size(1) >= s - 1 && uniform.stride(1) == 1);
    CHECK_HOST(chunk_mass.numel() >= static_cast<int64_t>(n) * chunks);
    CHECK_HOST(work.numel() >= static_cast<int64_t>(n) * 2);
    const auto params = ChainRejectionParams{
        .predicts = static_cast<int32_t*>(predicts.data_ptr()),
        .accept_index = static_cast<int32_t*>(accept_index.data_ptr()),
        .accept_token_num = static_cast<int32_t*>(accept_token_num.data_ptr()),
        .candidates = static_cast<const int64_t*>(candidates.data_ptr()),
        .retrive_index = static_cast<const int64_t*>(retrive_index.data_ptr()),
        .uniform = static_cast<const float*>(uniform.data_ptr()),
        .uniform_final = static_cast<const float*>(uniform_final.data_ptr()),
        .target_probs = static_cast<const float*>(target_probs.data_ptr()),
        .draft_probs = static_cast<const float*>(draft_probs.data_ptr()),
        .chunk_mass = static_cast<float*>(chunk_mass.data_ptr()),
        .work = static_cast<unsigned long long*>(work.data_ptr()),
        .cand_stride = candidates.stride(0),
        .idx_stride = retrive_index.stride(0),
        .acc_stride = accept_index.stride(0),
        .uni_stride = uniform.stride(0),
        .tp_b = target_probs.stride(0),
        .tp_s = target_probs.stride(1),
        .dp_b = draft_probs.stride(0),
        .dp_s = draft_probs.stride(1),
        .slots = s,
        .vocab = v,
        .chunks = chunks,
    };
    LaunchKernel(dim3(chunks, n), kThreads, device.unwrap())(kernel, params);
  }
};

}  // namespace sglang
