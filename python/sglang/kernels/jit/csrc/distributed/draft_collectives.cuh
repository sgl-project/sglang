// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// One-shot collectives for the DFLASH draft: a BF16 all-reduce and a packed top-k gather.
//
// Both push epoch-tagged 64-bit records to every peer's mailbox with st.relaxed.sys and poll their own mailbox with
// ld.relaxed.sys; a record carries its payload and its tag in one 64-bit word, so there is no sentinel, no re-arm and
// no fence. A mailbox is [2 parities][kSlots ranks][slot_bytes]; every call writes every record of its parity half,
// so a stale record is always two calls old and its tag differs from the current one.
//
// Every mailbox poll is bounded: a thread whose records are still missing after the deadline fail-stops the context before
// it writes an output, advances its epoch counter or releases dependents, so a late peer can never feed a partial result.
// One instance serves one stream at a time: two calls in flight on one mailbox would read the same epoch.
//
// kSlots ranks run on kSlots / kRpg processes (one GPU each): process `gpu` hosts ranks gpu + q * num_gpus for
// q < kRpg. Serving uses kRpg = 1; kRpg > 1 lets one node stand in for a wider group in tests and benchmarks.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/array.h>

#include <cuda_bf16.h>

#include <cstdint>

namespace sglang::draft_collectives {

inline constexpr uint32_t kMaxGpus = 16;
// 64 fragments per CTA: an 8 x 7168 reduce runs on 112 CTAs (tuned on the target GPU).
inline constexpr uint32_t kArThreads = 64;
inline constexpr uint32_t kGatherThreads = 256;
// Gather record: fp32 value bits [0, 32), token id [32, 56), tag [56, 64).
inline constexpr uint32_t kGatherIdBits = 24;

struct Mailbox {
  uint64_t peers[kMaxGpus];  // mailbox base of each process, as mapped in this process
  uint32_t* counters;        // local, one per CTA; each call advances every entry by one
  uint64_t slot_bytes;       // stride between rank slots; a parity half is kSlots slots
  uint32_t num_gpus;
  uint32_t gpu;
  uint64_t deadline_ns;  // > 0; a poll that outlives it fail-stops the context
};

struct ArParams {
  const uint4* input;  // [kRpg][num_frags] fragments of 8 BF16
  uint4* output;       // [num_frags]
  uint32_t num_frags;
  Mailbox mb;
};

struct GatherParams {
  const uint32_t* vals;  // [kRpg][num_rows * k] fp32 bits
  const int64_t* ids;    // [kRpg][num_rows * k]
  float* out_vals;       // [num_rows][kSlots * k]
  int64_t* out_ids;      // [num_rows][kSlots * k]
  uint32_t num_rows;
  uint32_t k;
  Mailbox mb;
};

SGL_DEVICE void st_relaxed_sys_v2u64(uint64_t addr, uint64_t a, uint64_t b) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(addr), "l"(a), "l"(b) : "memory");
}

SGL_DEVICE void st_relaxed_sys_u64(uint64_t addr, uint64_t a) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(a) : "memory");
}

SGL_DEVICE ulonglong2 ld_relaxed_sys_v2u64(uint64_t addr) {
  ulonglong2 v;
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(addr) : "memory");
  return v;
}

SGL_DEVICE uint64_t ld_relaxed_sys_u64(uint64_t addr) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
  return v;
}

SGL_DEVICE uint64_t global_timer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// A store to address 0, then trap: the context's sticky error aborts the step on the host.
SGL_DEVICE void fail_stop() {
  asm volatile("{ .reg .u64 a; mov.u64 a, 0; st.volatile.global.u32 [a], 0; trap; }" ::: "memory");
}

// Called once per poll round; fail-stops once the deadline has passed, timed from the 1024th round.
SGL_DEVICE void check_deadline(const Mailbox& mb, uint64_t& start, uint32_t& polls) {
  if ((++polls & 1023u) != 0) return;
  const uint64_t now = global_timer_ns();
  if (start == 0) {
    start = now;
  } else if (now - start > mb.deadline_ns) {
    fail_stop();
  }
}

SGL_DEVICE uint64_t slot_base(const Mailbox& mb, uint32_t gpu, uint32_t parity, uint32_t slots, uint32_t rank) {
  return mb.peers[gpu] + (static_cast<uint64_t>(parity) * slots + rank) * mb.slot_bytes;
}

// The shipped 1shot_push turns an all-zero 32-bit word into 0x00008000 (low BF16 -0) before it is summed;
// doing the same keeps the sums bitwise equal to it, signed zeros included.
SGL_DEVICE uint32_t lamport_zero_word(uint32_t w) {
  return w == 0 ? 0x00008000u : w;
}

// {3 x bf16, u16 tag} records of one 8-column fragment (4 BF16x2 words): rec0 = cols 0-2, rec1 = cols 3-5,
// rec2 = cols 6-7 and a zero pad.
SGL_DEVICE void pack3(const uint4& w, uint32_t tag, uint64_t& r0, uint64_t& r1, uint64_t& r2) {
  const uint64_t t = static_cast<uint64_t>(tag & 0xffffu) << 48;
  const uint64_t c0 = w.x & 0xffffu, c1 = w.x >> 16, c2 = w.y & 0xffffu, c3 = w.y >> 16;
  const uint64_t c4 = w.z & 0xffffu, c5 = w.z >> 16, c6 = w.w & 0xffffu, c7 = w.w >> 16;
  r0 = t | c0 | c1 << 16 | c2 << 32;
  r1 = t | c3 | c4 << 16 | c5 << 32;
  r2 = t | c6 | c7 << 16;
}

SGL_DEVICE uint4 unpack3(uint64_t r0, uint64_t r1, uint64_t r2) {
  uint4 w;
  w.x = static_cast<uint32_t>(r0 & 0xffffffffull);
  w.y = static_cast<uint32_t>(r0 >> 32 & 0xffffu) | static_cast<uint32_t>(r1 & 0xffffu) << 16;
  w.z = static_cast<uint32_t>(r1 >> 16 & 0xffffffffull);
  w.w = static_cast<uint32_t>(r2 & 0xffffffffull);
  return w;
}

SGL_DEVICE float2 bf16x2_to_float2(uint32_t w) {
  __nv_bfloat162 v;
  *reinterpret_cast<uint32_t*>(&v) = w;
  return __bfloat1622float2(v);
}

SGL_DEVICE uint32_t float2_to_bf16x2(float2 f) {
  const __nv_bfloat162 v = __float22bfloat162_rn(f);
  return *reinterpret_cast<const uint32_t*>(&v);
}

/// Rank-order FP32 sum over kSlots ranks of one 8-column fragment, rounded once to BF16 (the shipped 1shot_push
/// reduction). Fragment f of a slot: rec0/rec1 at byte 16 f, rec2 at byte 16 num_frags + 8 f.
template <uint32_t kSlots, uint32_t kRpg, bool kUsePDL>
__global__ __launch_bounds__(kArThreads) void all_reduce_pk3_kernel(const __grid_constant__ ArParams p) {
  using namespace device;
  const auto& mb = p.mb;
  const uint32_t frag = blockIdx.x * kArThreads + threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t epoch = mb.counters[blockIdx.x];
  const uint32_t parity = epoch & 1u, tag = (epoch + 1u) & 0xffffu;
  const uint64_t a_off = static_cast<uint64_t>(frag) * 16, b_off = static_cast<uint64_t>(p.num_frags) * 16 + frag * 8ull;

  if (frag < p.num_frags) {
#pragma unroll
    for (uint32_t q = 0; q < kRpg; ++q) {
      uint4 w = p.input[static_cast<uint64_t>(q) * p.num_frags + frag];
      w = make_uint4(lamport_zero_word(w.x), lamport_zero_word(w.y), lamport_zero_word(w.z), lamport_zero_word(w.w));
      uint64_t r0, r1, r2;
      pack3(w, tag, r0, r1, r2);
      const uint32_t rank = mb.gpu + q * mb.num_gpus;
      for (uint32_t g = 0; g < mb.num_gpus; ++g) {
        const uint64_t base = slot_base(mb, g, parity, kSlots, rank);
        st_relaxed_sys_v2u64(base + a_off, r0, r1);
        st_relaxed_sys_u64(base + b_off, r2);
      }
    }

    ulonglong2 ra[kSlots];
    uint64_t rb[kSlots];
    const uint64_t t = static_cast<uint64_t>(tag);
    uint32_t pending = 0;
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      const uint64_t base = slot_base(mb, mb.gpu, parity, kSlots, s);
      ra[s] = ld_relaxed_sys_v2u64(base + a_off);
      rb[s] = ld_relaxed_sys_u64(base + b_off);
    }
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      if ((ra[s].x >> 48) != t || (ra[s].y >> 48) != t || (rb[s] >> 48) != t) pending |= 1u << s;
    }
    uint64_t start = 0;
    uint32_t polls = 0;
    while (pending != 0) {
      check_deadline(mb, start, polls);
#pragma unroll
      for (uint32_t s = 0; s < kSlots; ++s) {
        if (pending >> s & 1u) {
          const uint64_t base = slot_base(mb, mb.gpu, parity, kSlots, s);
          ra[s] = ld_relaxed_sys_v2u64(base + a_off);
          rb[s] = ld_relaxed_sys_u64(base + b_off);
        }
      }
#pragma unroll
      for (uint32_t s = 0; s < kSlots; ++s) {
        if ((pending >> s & 1u) && (ra[s].x >> 48) == t && (ra[s].y >> 48) == t && (rb[s] >> 48) == t) {
          pending &= ~(1u << s);
        }
      }
    }

    float2 acc[4];
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      const uint4 w = unpack3(ra[s].x, ra[s].y, rb[s]);
      const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const float2 f = bf16x2_to_float2(words[j]);
        acc[j] = s == 0 ? f : make_float2(acc[j].x + f.x, acc[j].y + f.y);
      }
    }
    p.output[frag] = make_uint4(
        float2_to_bf16x2(acc[0]), float2_to_bf16x2(acc[1]), float2_to_bf16x2(acc[2]), float2_to_bf16x2(acc[3]));
  }

  // The epoch advances before any thread releases dependents, so no later grid can start ahead of it.
  __syncthreads();
  if (threadIdx.x == 0) mb.counters[blockIdx.x] = epoch + 1u;
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();
}

/// out[row][r * k + j] = rank r's (val, id) at [row][j]: the layout of an all-gather along the last dim.
template <uint32_t kSlots, uint32_t kRpg, bool kUsePDL>
__global__ __launch_bounds__(kGatherThreads) void topk_gather_kernel(const __grid_constant__ GatherParams p) {
  using namespace device;
  const auto& mb = p.mb;
  const uint32_t num_rec = p.num_rows * p.k;
  const uint32_t i = blockIdx.x * kGatherThreads + threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t epoch = mb.counters[blockIdx.x];
  const uint32_t parity = epoch & 1u;
  const uint64_t tag = static_cast<uint64_t>((epoch + 1u) & 0xffu);
  constexpr uint64_t kIdMask = (1ull << kGatherIdBits) - 1;

  // kSlots * num_rec threads both produce (rank q's record e to process g) and consume (slot s, record e).
  if (i < kSlots * num_rec) {
    const uint32_t per_q = num_rec * mb.num_gpus;
    const uint32_t q = i / per_q, g = i % per_q / num_rec, e = i % num_rec;
    const uint64_t src = static_cast<uint64_t>(q) * num_rec + e;
    const uint64_t rec = p.vals[src] | (static_cast<uint64_t>(p.ids[src]) & kIdMask) << 32 | tag << 56;
    const uint32_t rank = mb.gpu + q * mb.num_gpus;
    st_relaxed_sys_u64(slot_base(mb, g, parity, kSlots, rank) + e * 8ull, rec);

    const uint32_t s = i / num_rec, row = e / p.k, j = e % p.k;
    const uint64_t addr = slot_base(mb, mb.gpu, parity, kSlots, s) + e * 8ull;
    uint64_t got = ld_relaxed_sys_u64(addr);
    uint64_t start = 0;
    uint32_t polls = 0;
    while ((got >> 56) != tag) {
      check_deadline(mb, start, polls);
      got = ld_relaxed_sys_u64(addr);
    }
    const uint64_t out = static_cast<uint64_t>(row) * kSlots * p.k + static_cast<uint64_t>(s) * p.k + j;
    p.out_vals[out] = __uint_as_float(static_cast<uint32_t>(got));
    p.out_ids[out] = static_cast<int64_t>(got >> 32 & kIdMask);
  }

  // The epoch advances before any thread releases dependents, so no later grid can start ahead of it.
  __syncthreads();
  if (threadIdx.x == 0) mb.counters[blockIdx.x] = epoch + 1u;
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();
}

inline Mailbox make_mailbox(
    const tvm::ffi::Array<int64_t>& peers,
    tvm::ffi::TensorView counters,
    int64_t gpu,
    int64_t slot_bytes,
    int64_t deadline_ns) {
  using namespace host;
  RuntimeCheck(peers.size() >= 1 && peers.size() <= kMaxGpus, "peers: ", peers.size());
  RuntimeCheck(gpu >= 0 && gpu < static_cast<int64_t>(peers.size()), "gpu: ", gpu);
  RuntimeCheck(slot_bytes > 0 && slot_bytes % 16 == 0, "slot_bytes: ", slot_bytes);
  RuntimeCheck(deadline_ns > 0, "deadline_ns must be positive: ", deadline_ns);
  Mailbox mb{};
  for (size_t g = 0; g < peers.size(); ++g) {
    mb.peers[g] = static_cast<uint64_t>(peers[g]);
  }
  mb.counters = static_cast<uint32_t*>(counters.data_ptr());
  mb.slot_bytes = static_cast<uint64_t>(slot_bytes);
  mb.num_gpus = static_cast<uint32_t>(peers.size());
  mb.gpu = static_cast<uint32_t>(gpu);
  mb.deadline_ns = static_cast<uint64_t>(deadline_ns);
  return mb;
}

template <uint32_t kSlots, uint32_t kRpg, bool kUsePDL>
struct DraftCollectives {
  static_assert(kSlots % kRpg == 0 && kSlots / kRpg <= kMaxGpus && kSlots <= 32);

  /// \brief Rank-order one-shot all-reduce of BF16 rows; `output = sum over ranks of input`.
  /// \param input [kRpg, rows, hidden] BF16, contiguous; kRpg partials hosted by this process.
  /// \param output [rows, hidden] BF16.
  /// \param counters int32 [ceil(rows * hidden / 8 / 128)], zeroed once; the launch grid is its length.
  /// \param peers Mailbox base of each process (kSlots / kRpg entries), each >= 2 * kSlots * slot_bytes bytes.
  /// \param slot_bytes Must be >= 3 * rows * hidden bytes (24 bytes per 8 columns).
  /// \param deadline_ns Positive; a rank still missing a peer's record after it fail-stops (trap).
  static void all_reduce(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView counters,
      tvm::ffi::Array<int64_t> peers,
      int64_t gpu,
      int64_t slot_bytes,
      int64_t deadline_ns) {
    using namespace host;
    RuntimeCheck(peers.size() * kRpg == kSlots, "peers ", peers.size(), " x kRpg ", kRpg, " != kSlots ", kSlots);
    const int64_t numel = output.numel();
    RuntimeCheck(numel % 8 == 0 && input.numel() == numel * kRpg, "numel: ", input.numel(), " vs ", numel);
    const uint32_t num_frags = static_cast<uint32_t>(numel / 8);
    RuntimeCheck(slot_bytes >= static_cast<int64_t>(num_frags) * 24, "slot_bytes ", slot_bytes, " < ", num_frags * 24);
    const uint32_t grid = div_ceil(num_frags, kArThreads);
    RuntimeCheck(counters.numel() == grid, "counters ", counters.numel(), " != grid ", grid);
    const auto params = ArParams{
        .input = static_cast<const uint4*>(input.data_ptr()),
        .output = static_cast<uint4*>(output.data_ptr()),
        .num_frags = num_frags,
        .mb = make_mailbox(peers, counters, gpu, slot_bytes, deadline_ns),
    };
    LaunchKernel(grid, kArThreads, output.device()).enable_pdl(kUsePDL)(all_reduce_pk3_kernel<kSlots, kRpg, kUsePDL>,
                                                                      params);
  }

  /// \brief Packed all-gather of per-rank top-k (fp32 value, token id < 2^24) along the last dim.
  /// \param vals [kRpg, rows, k] fp32; ids [kRpg, rows, k] int64.
  /// \param out_vals [rows, kSlots * k] fp32; out_ids [rows, kSlots * k] int64.
  /// \param counters int32 [ceil(kSlots * rows * k / 256)], zeroed once; the launch grid is its length.
  /// \param slot_bytes Must be >= 8 * rows * k.
  /// \param deadline_ns Positive; a rank still missing a peer's record after it fail-stops (trap).
  static void topk_gather(
      tvm::ffi::TensorView vals,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView out_vals,
      tvm::ffi::TensorView out_ids,
      tvm::ffi::TensorView counters,
      tvm::ffi::Array<int64_t> peers,
      int64_t gpu,
      int64_t slot_bytes,
      int64_t deadline_ns) {
    using namespace host;
    RuntimeCheck(peers.size() * kRpg == kSlots, "peers ", peers.size(), " x kRpg ", kRpg, " != kSlots ", kSlots);
    const int64_t rows = out_vals.size(0), k = out_vals.size(1) / kSlots;
    RuntimeCheck(out_vals.size(1) == k * kSlots && out_ids.numel() == rows * k * kSlots, "out shape");
    RuntimeCheck(vals.numel() == rows * k * kRpg && ids.numel() == rows * k * kRpg, "input numel");
    RuntimeCheck(slot_bytes >= rows * k * 8, "slot_bytes ", slot_bytes, " < ", rows * k * 8);
    const uint32_t grid = div_ceil(static_cast<uint32_t>(kSlots * rows * k), kGatherThreads);
    RuntimeCheck(counters.numel() == grid, "counters ", counters.numel(), " != grid ", grid);
    const auto params = GatherParams{
        .vals = static_cast<const uint32_t*>(vals.data_ptr()),
        .ids = static_cast<const int64_t*>(ids.data_ptr()),
        .out_vals = static_cast<float*>(out_vals.data_ptr()),
        .out_ids = static_cast<int64_t*>(out_ids.data_ptr()),
        .num_rows = static_cast<uint32_t>(rows),
        .k = static_cast<uint32_t>(k),
        .mb = make_mailbox(peers, counters, gpu, slot_bytes, deadline_ns),
    };
    LaunchKernel(grid, kGatherThreads, out_vals.device()).enable_pdl(kUsePDL)(topk_gather_kernel<kSlots, kRpg, kUsePDL>,
                                                                           params);
  }
};

}  // namespace sglang::draft_collectives
