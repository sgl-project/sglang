// One kernel for the DFLASH verify commit of a decode step, over the CustomAllReduceV2 push plane (the phase and
// slot protocol of csrc/muse_spark/all_gather.cuh):
//   1. (optional) rank 0's sampled accept lengths and bonus tokens are delivered to every rank (the decision
//      broadcast), through the custom all-reduce push plane: rank 0 stores (accept_len + 1, bonus + 1) words into
//      slot (dst = peer, src = 0) of every peer's push workspace for the current phase, and every other rank stores
//      one non-zero 16-byte marker into slot (dst = peer, src = rank) of every peer. Every rank then polls its own
//      slots until every expected word from every peer is non-zero (the plane's empty marker is zero), takes rank 0's
//      values and restores the zeros. As in SGLang's push collectives, every rank waits for every peer in each call,
//      so no rank can start a call more than one call ahead of a peer, and the two phase halves are never reused
//      before the peer has consumed them;
//   2. the committed block: out_tokens[i] = candidates[i, 1:] followed by 0, with the bonus token written at
//      position accept_len[i]; commit_lens = accept_len + 1.
// Steps 1-2 are what SGLang runs as separate operations (group broadcast, commit); the arithmetic is integer copies
// only.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace dflash_commit {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxPolls = 1u << 28;

__device__ __noinline__ void poll_bound_exceeded() {
  __trap();
}

template <uint32_t kWorld>
struct Params {
  device::distributed::PushWorkSpace<kWorld> ws;
  uint32_t num_counters;
  uint32_t rank;
  bool exchange;
  int bs;
  int block;
  const int64_t* candidates;  // [bs, block]
  int32_t* accept_len;        // [bs], in/out
  int64_t* bonus;             // [bs], in/out
  int64_t* out_tokens;        // [bs, block]
  int32_t* commit_lens;       // [bs]
};

template <uint32_t kWorld, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void commit_kernel(const __grid_constant__ Params<kWorld> p) {
  device::PDLWaitPrimary<kUsePDL>();
  __shared__ int32_t s_acc[kThreads * 4];
  __shared__ int64_t s_bonus[kThreads * 4];
  const int tid = threadIdx.x;
  uint32_t phase = 0;
  if (p.exchange) {
    phase = p.ws.counter[0].get() & 1;
    const int nvec = (p.bs + 1) / 2;
    if (p.rank == 0) {
      for (int v = tid; v < nvec; v += kThreads) {
        const int i0 = 2 * v, i1 = 2 * v + 1;
        uint4 w;
        w.x = static_cast<uint32_t>(p.accept_len[i0] + 1);
        w.y = static_cast<uint32_t>(p.bonus[i0] + 1);
        w.z = i1 < p.bs ? static_cast<uint32_t>(p.accept_len[i1] + 1) : 1u;
        w.w = i1 < p.bs ? static_cast<uint32_t>(p.bonus[i1] + 1) : 1u;
#pragma unroll
        for (uint32_t d = 1; d < kWorld; ++d) {
          uint8_t* slot = p.ws.workspaces[d] + (0 + phase * kWorld) * p.ws.slot_bytes;
          device::ptx::st_relaxed_16B(w, slot, v);
        }
        s_acc[i0] = p.accept_len[i0];
        s_bonus[i0] = p.bonus[i0];
        if (i1 < p.bs) {
          s_acc[i1] = p.accept_len[i1];
          s_bonus[i1] = p.bonus[i1];
        }
      }
    } else if (tid == 0) {
      const uint4 marker = make_uint4(1u, 1u, 1u, 1u);
#pragma unroll
      for (uint32_t d = 0; d < kWorld; ++d) {
        if (d == p.rank) continue;
        uint8_t* slot = p.ws.workspaces[d] + (p.rank + phase * kWorld) * p.ws.slot_bytes;
        device::ptx::st_relaxed_16B(marker, slot, 0);
      }
    }
    // own slots: rank 0's data (nvec vectors) and one marker from every other peer
#pragma unroll
    for (uint32_t s = 0; s < kWorld; ++s) {
      if (s == p.rank) continue;
      const int n = s == 0 ? nvec : 1;
      uint8_t* slot = p.ws.workspaces[p.rank] + (s + phase * kWorld) * p.ws.slot_bytes;
      for (int v = tid; v < n; v += kThreads) {
        uint4 w;
        uint32_t polls = 0;
        do {
          device::ptx::ld_relaxed_16B(w, slot, v);
          if (w.x != 0u && w.y != 0u && w.z != 0u && w.w != 0u) break;
          if (++polls == kMaxPolls) [[unlikely]]
            poll_bound_exceeded();
        } while (true);
        const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
        device::ptx::st_global_16B(zero, slot, v);
        if (s == 0) {
          const int i0 = 2 * v, i1 = 2 * v + 1;
          s_acc[i0] = static_cast<int32_t>(w.x) - 1;
          s_bonus[i0] = static_cast<int64_t>(w.y) - 1;
          if (i1 < p.bs) {
            s_acc[i1] = static_cast<int32_t>(w.z) - 1;
            s_bonus[i1] = static_cast<int64_t>(w.w) - 1;
          }
        }
      }
    }
  } else {
    for (int i = tid; i < p.bs; i += kThreads) {
      s_acc[i] = p.accept_len[i];
      s_bonus[i] = p.bonus[i];
    }
  }
  __syncthreads();
  for (int e = tid; e < p.bs * p.block; e += kThreads) {
    const int i = e / p.block, j = e % p.block;
    const int a = s_acc[i];
    int64_t tok = j < p.block - 1 ? p.candidates[i * p.block + j + 1] : 0;
    if (j == a) tok = s_bonus[i];
    p.out_tokens[e] = tok;
  }
  for (int i = tid; i < p.bs; i += kThreads) {
    p.accept_len[i] = s_acc[i];
    p.commit_lens[i] = s_acc[i] + 1;
    p.bonus[i] = s_bonus[i];
  }
  device::PDLTriggerSecondary<kUsePDL>();
  if (p.exchange) {
    __syncthreads();
    for (uint32_t c = tid; c < p.num_counters; c += kThreads)
      p.ws.counter[c].set(phase ^ 1);
  }
}

}  // namespace dflash_commit

using host::distributed::CommunicatorRef;

template <uint32_t kWorld, bool kUsePDL>
struct DflashCommit {
  using TensorView = tvm::ffi::TensorView;
  using P = dflash_commit::Params<kWorld>;

  static void launch(
      P& p,
      TensorView candidates,
      TensorView accept_len,
      TensorView bonus,
      TensorView out_tokens,
      TensorView commit_lens) {
    using namespace host;
    auto B = SymbolicSize{"bs"};
    auto K = SymbolicSize{"block"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, K}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({B, K}).with_dtype<int64_t>().with_device(device).verify(out_tokens);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(accept_len);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device).verify(bonus);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(commit_lens);
    RuntimeCheck(candidates.stride(0) == K.unwrap() && out_tokens.stride(0) == K.unwrap(), "rows must be contiguous");
    RuntimeCheck(B.unwrap() <= int64_t(dflash_commit::kThreads) * 4, "batch above the kernel's capacity");
    p.bs = static_cast<int>(B.unwrap());
    p.block = static_cast<int>(K.unwrap());
    p.candidates = static_cast<const int64_t*>(candidates.data_ptr());
    p.accept_len = static_cast<int32_t*>(accept_len.data_ptr());
    p.bonus = static_cast<int64_t*>(bonus.data_ptr());
    p.out_tokens = static_cast<int64_t*>(out_tokens.data_ptr());
    p.commit_lens = static_cast<int32_t*>(commit_lens.data_ptr());
    if (p.bs == 0) return;
    LaunchKernel(1, dflash_commit::kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(dflash_commit::commit_kernel<kWorld, kUsePDL>, p);
  }

  // with the decision broadcast over the push plane (every rank of the group calls it)
  static void
  run(CommunicatorRef ref,
      TensorView candidates,
      TensorView accept_len,
      TensorView bonus,
      TensorView out_tokens,
      TensorView commit_lens) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();
    P p{};
    RuntimeCheck(push.world_size == kWorld, "the kernel is compiled for TP", kWorld);
    RuntimeCheck(int64_t((candidates.size(0) + 1) / 2) * 16 <= int64_t(push.slot_bytes), "batch above the push slot");
    p.ws = push.get_workspace<kWorld>(int64_t((candidates.size(0) + 1) / 2) * 16);
    p.num_counters = push.num_blocks;
    p.rank = push.rank;
    p.exchange = true;
    launch(p, candidates, accept_len, bonus, out_tokens, commit_lens);
  }

  // without the broadcast (single rank, or the decision is not synchronized)
  static void run_local(
      TensorView candidates, TensorView accept_len, TensorView bonus, TensorView out_tokens, TensorView commit_lens) {
    P p{};
    p.exchange = false;
    launch(p, candidates, accept_len, bonus, out_tokens, commit_lens);
  }
};

}  // namespace sglang
