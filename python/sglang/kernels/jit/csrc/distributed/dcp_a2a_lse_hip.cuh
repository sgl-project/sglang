// Wave64 ROCm XGMI push all-to-all + fp32 LSE combine for DCP MLA decode.
//
// Contract:
//   cp_attn_out: [B, world * H_per_rank, D] bf16/fp16 contiguous (16B aligned)
//   cp_attn_lse: [B, world * H_per_rank] fp32 contiguous
//   out:         [B, H_per_rank, D] bf16/fp16 contiguous (16B aligned)
//
// Scratch lives in the uncached IPC meta buffer tail of a dedicated
// aiter::CustomAllreduce (2 epochs x world x max_slots x (D/8 + 1) uint4s).
// Each (src, slot) holds D/8 16-byte payload vectors plus one 16-byte header
// {lse_bits, flag, 0, flag}. Deadlock freedom follows from in-order workgroup
// dispatch and identical per-slot stride across all ranks (each workgroup
// index only waits on peers' matching slot).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "../kimi_k3/comm/ll_hip.cuh"
#include <cmath>
#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, Signal

#ifndef USE_ROCM
#error "distributed/dcp_a2a_lse_hip.cuh is a wave64 (ROCm) kernel"
#endif

namespace sglang {
namespace dcp_a2a_lse_hip {

using k3_ll::ll_load;
using k3_ll::ll_store;
using k3_ll::llx_v4u;
using k3_ll::next_flag;

inline constexpr int kWave = 64;

SGL_DEVICE void nt_store_u4(uint4* dst, const uint4& v) {
  const llx_v4u vec = {v.x, v.y, v.z, v.w};
  __builtin_nontemporal_store(vec, reinterpret_cast<llx_v4u*>(dst));
}

SGL_DEVICE uint4 nt_load_u4(const uint4* src) {
  const llx_v4u vec = __builtin_nontemporal_load(reinterpret_cast<const llx_v4u*>(src));
  return make_uint4(vec[0], vec[1], vec[2], vec[3]);
}

template <typename T>
SGL_DEVICE void unpack8(const uint4& packed, float (&dst)[8]);

template <>
SGL_DEVICE void unpack8<__hip_bfloat16>(const uint4& packed, float (&dst)[8]) {
  const uint32_t w[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __hip_bfloat162 pair = *reinterpret_cast<const __hip_bfloat162*>(&w[i]);
    const float2 f = __bfloat1622float2(pair);
    dst[2 * i] = f.x;
    dst[2 * i + 1] = f.y;
  }
}

template <>
SGL_DEVICE void unpack8<__half>(const uint4& packed, float (&dst)[8]) {
  const uint32_t w[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 pair = *reinterpret_cast<const __half2*>(&w[i]);
    const float2 f = __half22float2(pair);
    dst[2 * i] = f.x;
    dst[2 * i + 1] = f.y;
  }
}

template <typename T>
SGL_DEVICE uint4 pack8(const float (&src)[8]);

template <>
SGL_DEVICE uint4 pack8<__hip_bfloat16>(const float (&src)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __hip_bfloat162 pair = __float22bfloat162_rn(make_float2(src[2 * i], src[2 * i + 1]));
    w[i] = *reinterpret_cast<const uint32_t*>(&pair);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

template <>
SGL_DEVICE uint4 pack8<__half>(const float (&src)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 pair = __float22half2_rn(make_float2(src[2 * i], src[2 * i + 1]));
    w[i] = *reinterpret_cast<const uint32_t*>(&pair);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

SGL_DEVICE float sanitize_lse(float v) {
  return (isnan(v) || v == INFINITY) ? -INFINITY : v;
}

template <int kWorld>
struct Params {
  const uint4* __restrict__ in_out;  // [B, kWorld * H_per_rank, D / 8]
  const float* __restrict__ in_lse;  // [B, kWorld * H_per_rank]
  uint4* __restrict__ out;           // [B, H_per_rank, D / 8]
  uint4* scratch[kWorld];
  const uint4* self_scratch;
  int32_t* __restrict__ epochs;
  int64_t epoch_vecs;
  int64_t max_slots;
  int64_t num_slots;
  int32_t h_per_rank;
  int32_t rank;
};

template <int kWorld, int kDim, bool kIsBaseE, typename T>
__global__ __launch_bounds__(kWave, 4) void dcp_a2a_lse_kernel(const Params<kWorld> p) {
  static_assert(kDim == 128 || kDim == 512, "kDim must be 128 or 512");
  constexpr int kDataVecs = kDim / 8;
  constexpr int kVecsPerSlot = kDataVecs + 1;
  constexpr int kLseLane = (kDim == 128) ? 16 : 0;

  const int tid = threadIdx.x;
  const int h_total = kWorld * p.h_per_rank;
  const bool data_active = (kDim == 512) || (tid < kDataVecs);

  for (int64_t slot = static_cast<int64_t>(blockIdx.x); slot < p.num_slots; slot += static_cast<int64_t>(gridDim.x)) {
    const int64_t b = slot / p.h_per_rank;
    const int64_t hl = slot - b * p.h_per_rank;
    const uint32_t flag = next_flag(p.epochs, static_cast<int>(slot));
    const int64_t epoch_off = static_cast<int64_t>(flag & 1u) * p.epoch_vecs;

    // Load local inputs before issuing remote stores so HBM loads do not wait behind store acks.
    uint4 send_data[kWorld];
    uint32_t send_lse[kWorld];
#pragma unroll
    for (int peer = 0; peer < kWorld; ++peer) {
      const int64_t h_idx = b * h_total + peer * p.h_per_rank + hl;
      if (data_active) {
        send_data[peer] = p.in_out[h_idx * kDataVecs + tid];
      }
      if (tid == kLseLane) {
        send_lse[peer] = __float_as_uint(p.in_lse[h_idx]);
      }
    }

    const int64_t my_src_off = epoch_off + (static_cast<int64_t>(p.rank) * p.max_slots + slot) * kVecsPerSlot;
    if (data_active) {
#pragma unroll
      for (int peer = 0; peer < kWorld; ++peer) {
        if (peer == p.rank) continue;
        nt_store_u4(p.scratch[peer] + my_src_off + tid, send_data[peer]);
      }
    }

    // Correctness relies on aiter::CustomAllreduce allocating sg_.signals with
    // hipDeviceMallocUncached (bypasses L1/L2); s_waitcnt(0) waits for all remote
    // payload store acks without issuing whole-L2 buffer_wbl2/buffer_inv flushes.
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0);
    asm volatile("" ::: "memory");

    if (tid == kLseLane) {
#pragma unroll
      for (int peer = 0; peer < kWorld; ++peer) {
        if (peer == p.rank) continue;
        ll_store(p.scratch[peer] + my_src_off + kDataVecs, send_lse[peer], 0u, flag);
      }
    }

    uint4 recv_data[kWorld];
    float recv_lse[kWorld];
#pragma unroll
    for (int src = 0; src < kWorld; ++src) {
      uint32_t lse_bits = 0u;
      if (src == p.rank) {
        recv_data[src] = send_data[src];
        if (tid == kLseLane) lse_bits = send_lse[src];
      } else {
        const uint4* inbox =
            p.self_scratch + epoch_off + (static_cast<int64_t>(src) * p.max_slots + slot) * kVecsPerSlot;
        if (tid == kLseLane) {
          uint4 hdr;
          do {
            hdr = ll_load(inbox + kDataVecs);
          } while (hdr.y != flag || hdr.w != flag);
          lse_bits = hdr.x;
        }
        asm volatile("" ::: "memory");
        if (data_active) {
          recv_data[src] = nt_load_u4(inbox + tid);
        }
      }
      const uint32_t bcast = static_cast<uint32_t>(__shfl(static_cast<int>(lse_bits), kLseLane, kWave));
      recv_lse[src] = sanitize_lse(__uint_as_float(bcast));
    }

    float lse_max = recv_lse[0];
#pragma unroll
    for (int src = 1; src < kWorld; ++src) {
      lse_max = fmaxf(lse_max, recv_lse[src]);
    }
    if (lse_max == -INFINITY) lse_max = 0.0f;

    float weights[kWorld];
    float weight_sum = 0.0f;
#pragma unroll
    for (int src = 0; src < kWorld; ++src) {
      float w = 0.0f;
      if (recv_lse[src] != -INFINITY) {
        const float centered = recv_lse[src] - lse_max;
        w = kIsBaseE ? expf(centered) : exp2f(centered);
      }
      weights[src] = w;
      weight_sum += w;
    }
    const float inv_w = 1.0f / fmaxf(weight_sum, 1.0f);

    if (data_active) {
      float acc[8] = {};
#pragma unroll
      for (int src = 0; src < kWorld; ++src) {
        float vals[8];
        unpack8<T>(recv_data[src], vals);
        const float w = weights[src];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          acc[e] = fmaf(vals[e], w, acc[e]);
        }
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        acc[e] *= inv_w;
      }
      p.out[slot * kDataVecs + tid] = pack8<T>(acc);
    }

    asm volatile("" ::: "memory");
    if (tid == 0) {
      p.epochs[slot] = static_cast<int32_t>(flag);
    }
  }
}

template <int kWorld, int kDim, typename T>
struct DcpA2ALseKernel {
  /**
   * @brief Exchange DCP head partials over XGMI IPC scratch and merge with fp32 LSE weights.
   *
   * @param cp_attn_out  [B, kWorld * H_local, kDim] contiguous 16B-aligned partial outputs.
   * @param cp_attn_lse  [B, kWorld * H_local] contiguous fp32 partial LSE values.
   * @param out          [B, H_local, kDim] contiguous 16B-aligned merged output.
   * @param comm         Opaque pointer to aiter::CustomAllreduce owning uncached IPC scratch.
   * @param epochs       [max_slots] rank-local int32 epoch table.
   * @param max_slots    Allocated slot capacity (max_rows * H_local).
   * @param max_grid     Workgroup cap (runtime CU count * 4) for persistent slot iteration.
   * @param is_lse_base_on_e  True for natural-log LSE, false for base-2 LSE.
   */
  static void run(
      const tvm::ffi::TensorView cp_attn_out,
      const tvm::ffi::TensorView cp_attn_lse,
      tvm::ffi::TensorView out,
      int64_t comm,
      tvm::ffi::TensorView epochs,
      int64_t max_slots,
      int64_t max_grid,
      bool is_lse_base_on_e) {
    using namespace host;
    SymbolicSize B{"B"}, H{"H"}, H_local{"H_local"};
    SymbolicDevice device;
    device.set_options<kDLGPU>();

    TensorMatcher({B, H, kDim})
        .with_dtype<T>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(cp_attn_out);
    TensorMatcher({B, H}).with_dtype<float>().with_device(device).verify(cp_attn_lse);
    TensorMatcher({B, H_local, kDim})
        .with_dtype<T>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(out);
    TensorMatcher({max_slots}).with_dtype<int32_t>().with_device(device).verify(epochs);

    const int64_t b = B.unwrap();
    const int64_t h = H.unwrap();
    const int64_t hl = H_local.unwrap();
    const int64_t num_slots = b * hl;
    CHECK_HOST(h == kWorld * hl) << "H (" << h << ") must equal world_size (" << kWorld << ") * H_local (" << hl << ")";
    CHECK_HOST(b >= 1 && num_slots <= max_slots) << "num_slots (" << num_slots << ") out of range [1, " << max_slots << "]";
    CHECK_HOST(max_grid >= 1) << "max_grid must be positive, got " << max_grid;

    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    CHECK_HOST(communicator->world_size_ == kWorld)
        << "communicator world_size (" << communicator->world_size_ << ") != " << kWorld;

    constexpr int64_t kVecsPerSlot = (kDim / 8) + 1;
    Params<kWorld> params{
        .in_out = static_cast<const uint4*>(cp_attn_out.data_ptr()),
        .in_lse = static_cast<const float*>(cp_attn_lse.data_ptr()),
        .out = static_cast<uint4*>(out.data_ptr()),
        .scratch = {},
        .self_scratch = nullptr,
        .epochs = static_cast<int32_t*>(epochs.data_ptr()),
        .epoch_vecs = static_cast<int64_t>(kWorld) * max_slots * kVecsPerSlot,
        .max_slots = max_slots,
        .num_slots = num_slots,
        .h_per_rank = static_cast<int32_t>(hl),
        .rank = communicator->rank_,
    };
    for (int i = 0; i < kWorld; ++i) {
      params.scratch[i] =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(communicator->sg_.signals[i]) + sizeof(aiter::Signal));
    }
    params.self_scratch = params.scratch[communicator->rank_];

    const unsigned int grid_x = static_cast<unsigned int>(num_slots < max_grid ? num_slots : max_grid);
    const dim3 grid(grid_x, 1u, 1u);
    const DLDevice dev = device.unwrap();
    if (is_lse_base_on_e) {
      LaunchKernel(grid, kWave, dev)(dcp_a2a_lse_kernel<kWorld, kDim, true, T>, params);
    } else {
      LaunchKernel(grid, kWave, dev)(dcp_a2a_lse_kernel<kWorld, kDim, false, T>, params);
    }
  }
};

}  // namespace dcp_a2a_lse_hip
}  // namespace sglang
