#pragma once

// All-reduce for 4 GPUs on two PCIe switches (ranks 0-1 and 2-3 each share one), for
// decode-sized messages, where NCCL's 6-hop ring is bound by per-hop latency.
//
// Three steps, each a store into the receiver's IPC buffer and one release flag:
//   1. reduce-scatter within the pair: send the partner the half it owns, add mine;
//   2. exchange the pair sums with the same-position GPU of the other pair, add;
//   3. all-gather within the pair.
// For the smallest messages a one-shot variant does it in two: the whole tile within the
// pair, then the whole pair sum across.
//
// Each element is (r0 + r1) + (r2 + r3), added in fp32. By default (wide) the pair sums
// cross the switches in fp32 and only the final sum is rounded to bf16, once (the host
// doubles the segments for it). Without wide, each pair sum is rounded to bf16 too.
// In both variants the order is the same for any message size and block count, so the
// result is the same on all ranks and a row's bits do not depend on what else was in
// the message (NCCL's ring order does). a + b == b + a in IEEE arithmetic, so the two
// GPUs that compute a sum agree.
//
// Block b of every rank works on the same tiles and only talks to block b of its peers:
// no grid-wide barrier. Each block keeps a 64-bit epoch counter in device memory (CUDA
// graph safe); a flag holds the epoch of its last write, and buffers alternate by epoch
// parity. Every iteration a rank receives from its partner and its cross peer before it
// finishes, and sends to both only after it finished the previous one, so a sender can
// be at most one epoch ahead of a receiver: it never overwrites an unread slot.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

namespace pcie_pair {

constexpr int kStages = 3;

__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

// All threads' stores to the peer are done; one release publishes them (it is
// cumulative over the CTA barrier). A separate __threadfence_system() costs ~5 us/step.
__device__ __forceinline__ void signal(uint64_t* flag, uint64_t epoch) {
  __syncthreads();
  if (threadIdx.x == 0) st_release_sys(flag, epoch);
}

__device__ __forceinline__ void wait(const uint64_t* flag, uint64_t epoch) {
  if (threadIdx.x == 0) {
    while (ld_acquire_sys(flag) < epoch) {
    }
  }
  __syncthreads();
}

// 8 bf16 lanes: fp32 add, round to nearest even.
__device__ __forceinline__ uint4 add8(uint4 a, uint4 b) {
  uint4 r;
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
  __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 fa = __bfloat1622float2(pa[i]);
    float2 fb = __bfloat1622float2(pb[i]);
    pr[i] = __floats2bfloat162_rn(__fadd_rn(fa.x, fb.x), __fadd_rn(fa.y, fb.y));
  }
  return r;
}

// 8 bf16 lanes summed in fp32, kept in fp32 (two uint4 of float4).
__device__ __forceinline__ void sum8_f32(uint4 a, uint4 b, uint4& lo, uint4& hi) {
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
  float* f[2] = {reinterpret_cast<float*>(&lo), reinterpret_cast<float*>(&hi)};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 fa = __bfloat1622float2(pa[i]);
    float2 fb = __bfloat1622float2(pb[i]);
    f[i >> 1][(i & 1) * 2] = __fadd_rn(fa.x, fb.x);
    f[i >> 1][(i & 1) * 2 + 1] = __fadd_rn(fa.y, fb.y);
  }
}

// Two fp32 octets added, rounded once to bf16.
__device__ __forceinline__ uint4 add8_f32(uint4 alo, uint4 ahi, uint4 blo, uint4 bhi) {
  uint4 r;
  const float* fa[2] = {reinterpret_cast<const float*>(&alo), reinterpret_cast<const float*>(&ahi)};
  const float* fb[2] = {reinterpret_cast<const float*>(&blo), reinterpret_cast<const float*>(&bhi)};
  __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int h = i >> 1, k = (i & 1) * 2;
    pr[i] = __floats2bfloat162_rn(__fadd_rn(fa[h][k], fb[h][k]), __fadd_rn(fa[h][k + 1], fb[h][k + 1]));
  }
  return r;
}

struct Args {
  const uint4* in;
  uint4* out;
  int64_t n_vec;  // message length in uint4 (8 bf16)
  char* self_buf;
  char* partner_buf;
  char* cross_buf;
  uint64_t* counters;  // per block, local memory
  int pos;             // position in the pair: 0 owns the first half of each tile
  int max_blocks;
  int64_t half_cap;  // uint4 per half tile per block per iteration
  int64_t data_off;  // bytes from buffer base to the data area
  int one_shot;
  int wide;  // pair sums cross in fp32 (two planes of uint4)
};

__device__ __forceinline__ uint4* segment(const Args& a, char* buf, int slot, int stage) {
  int64_t idx = ((int64_t)slot * kStages + stage) * a.max_blocks + blockIdx.x;
  return reinterpret_cast<uint4*>(buf + a.data_off + idx * a.half_cap * 16);
}

__device__ __forceinline__ uint64_t* flag(const Args& a, char* buf, int stage) {
  return reinterpret_cast<uint64_t*>(buf) + (int64_t)stage * a.max_blocks + blockIdx.x;
}

__global__ void __launch_bounds__(1024) pair_allreduce_kernel(Args a) {
  const int64_t G = gridDim.x, b = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  uint64_t epoch = a.counters[b];
  // Wide: stage 2 holds fp32 as two planes (lanes 0-3, lanes 4-7) of `cap` uint4 each, so
  // every warp store to the peer is contiguous (interleaving them halves PCIe efficiency).
  const int64_t cap = a.wide ? a.half_cap / 2 : a.half_cap;

  if (a.one_shot) {
    // The block's whole tile fits in one segment (checked on the host).
    ++epoch;
    const int slot = epoch & 1;
    const int64_t t = (a.n_vec + G - 1) / G;
    const int64_t t0 = min(b * t, a.n_vec), n = min(t0 + t, a.n_vec) - t0;
    uint4* dst = segment(a, a.partner_buf, slot, 0);
    for (int64_t i = tid; i < n; i += nt)
      dst[i] = a.in[t0 + i];
    signal(flag(a, a.partner_buf, 0), epoch);
    wait(flag(a, a.self_buf, 0), epoch);
    const uint4* rcv0 = segment(a, a.self_buf, slot, 0);
    dst = segment(a, a.cross_buf, slot, 1);
    for (int64_t i = tid; i < n; i += nt) {
      if (a.wide) {
        uint4 lo, hi;
        sum8_f32(a.in[t0 + i], __ldcg(rcv0 + i), lo, hi);  // r0 + r1 on both GPUs of a pair
        dst[i] = lo;
        dst[cap + i] = hi;
      } else {
        uint4 p = add8(a.in[t0 + i], __ldcg(rcv0 + i));
        a.out[t0 + i] = p;
        dst[i] = p;
      }
    }
    signal(flag(a, a.cross_buf, 1), epoch);
    wait(flag(a, a.self_buf, 1), epoch);
    const uint4* rcv = segment(a, a.self_buf, slot, 1);
    for (int64_t i = tid; i < n; i += nt) {
      if (a.wide) {
        uint4 lo, hi;  // recomputed: the same bits as sent, the received slot is still live
        sum8_f32(a.in[t0 + i], __ldcg(rcv0 + i), lo, hi);
        a.out[t0 + i] = add8_f32(lo, hi, __ldcg(rcv + i), __ldcg(rcv + cap + i));
      } else {
        a.out[t0 + i] = add8(a.out[t0 + i], __ldcg(rcv + i));
      }
    }
    if (tid == 0) a.counters[b] = epoch;
    return;
  }

  const int64_t iter_vec = G * 2 * cap;
  for (int64_t base = 0; base < a.n_vec; base += iter_vec) {
    ++epoch;
    const int slot = epoch & 1;
    const int64_t end = min(base + iter_vec, a.n_vec);
    // Same tiles on every rank: G tiles of two halves of h vectors, clipped at the end.
    const int64_t h = (end - base + 2 * G - 1) / (2 * G);
    const int64_t a0 = min(base + b * 2 * h, end), a1 = min(a0 + h, end);
    const int64_t b0 = a1, b1 = min(a0 + 2 * h, end);
    const int64_t my0 = a.pos == 0 ? a0 : b0, my_n = a.pos == 0 ? a1 - a0 : b1 - b0;
    const int64_t ot0 = a.pos == 0 ? b0 : a0, ot_n = a.pos == 0 ? b1 - b0 : a1 - a0;

    // 1. Reduce-scatter within the pair.
    uint4* dst = segment(a, a.partner_buf, slot, 0);
    for (int64_t i = tid; i < ot_n; i += nt)
      dst[i] = a.in[ot0 + i];
    signal(flag(a, a.partner_buf, 0), epoch);
    wait(flag(a, a.self_buf, 0), epoch);

    const uint4* rcv0 = segment(a, a.self_buf, slot, 0);
    dst = segment(a, a.cross_buf, slot, 1);
    for (int64_t i = tid; i < my_n; i += nt) {
      if (a.wide) {
        uint4 lo, hi;
        sum8_f32(a.in[my0 + i], __ldcg(rcv0 + i), lo, hi);
        dst[i] = lo;
        dst[cap + i] = hi;
      } else {
        uint4 p = add8(a.in[my0 + i], __ldcg(rcv0 + i));
        a.out[my0 + i] = p;
        dst[i] = p;
      }
    }
    // 2. Exchange pair sums across the switches.
    signal(flag(a, a.cross_buf, 1), epoch);
    wait(flag(a, a.self_buf, 1), epoch);

    const uint4* rcv = segment(a, a.self_buf, slot, 1);
    dst = segment(a, a.partner_buf, slot, 2);
    for (int64_t i = tid; i < my_n; i += nt) {
      uint4 o;
      if (a.wide) {
        uint4 lo, hi;
        sum8_f32(a.in[my0 + i], __ldcg(rcv0 + i), lo, hi);
        o = add8_f32(lo, hi, __ldcg(rcv + i), __ldcg(rcv + cap + i));
      } else {
        o = add8(a.out[my0 + i], __ldcg(rcv + i));
      }
      a.out[my0 + i] = o;
      dst[i] = o;
    }
    // 3. All-gather within the pair.
    signal(flag(a, a.partner_buf, 2), epoch);
    wait(flag(a, a.self_buf, 2), epoch);

    rcv = segment(a, a.self_buf, slot, 2);
    for (int64_t i = tid; i < ot_n; i += nt)
      a.out[ot0 + i] = __ldcg(rcv + i);
    __syncthreads();
  }
  if (tid == 0) a.counters[b] = epoch;
}

}  // namespace pcie_pair

/// Host entry. `input`/`output` are flat bf16 tensors of the same length (a multiple
/// of 8); `counters` holds one int64 epoch per block, local to this rank. The three
/// buffer pointers are this rank's IPC buffer and its partner's and cross peer's
/// mappings of theirs (see pcie_pair_ar.py for the layout).
struct PciePairAllReduceKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView counters,
      int64_t self_buf,
      int64_t partner_buf,
      int64_t cross_buf,
      int64_t pos,
      int64_t max_blocks,
      int64_t half_cap,
      int64_t data_off,
      int64_t blocks,
      int64_t threads,
      bool one_shot,
      bool wide) {
    using namespace host;
    SymbolicSize N = {"num_elements"};
    SymbolicSize B = {"num_counters"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<bf16_t>().with_device(device_).verify(input).verify(output);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(counters);
    const int64_t n = N.unwrap();
    RuntimeCheck(n % 8 == 0, "message must be a multiple of 8 bf16, got ", n);
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
        "input and output must be 16-byte aligned");
    RuntimeCheck(blocks >= 1 && blocks <= max_blocks && B.unwrap() >= max_blocks, "bad block count ", blocks);
    RuntimeCheck(!wide || half_cap % 2 == 0, "wide mode needs an even half_cap");
    if (n == 0) return;  // same on every rank, so the counters stay in step
    pcie_pair::Args a;
    a.in = static_cast<const uint4*>(input.data_ptr());
    a.out = static_cast<uint4*>(output.data_ptr());
    a.n_vec = n / 8;
    a.self_buf = reinterpret_cast<char*>(self_buf);
    a.partner_buf = reinterpret_cast<char*>(partner_buf);
    a.cross_buf = reinterpret_cast<char*>(cross_buf);
    a.counters = static_cast<uint64_t*>(counters.data_ptr());
    a.pos = static_cast<int>(pos);
    a.max_blocks = static_cast<int>(max_blocks);
    a.half_cap = half_cap;
    a.data_off = data_off;
    a.one_shot = one_shot;
    a.wide = wide;
    RuntimeCheck(
        !one_shot || (a.n_vec + blocks - 1) / blocks <= (wide ? half_cap / 2 : half_cap), "one-shot tile too big");
    LaunchKernel(static_cast<uint32_t>(blocks), static_cast<uint32_t>(threads), device_.unwrap())(
        pcie_pair::pair_allreduce_kernel, a);
  }
};

}  // namespace sglang
