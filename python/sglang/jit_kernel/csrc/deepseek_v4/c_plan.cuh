#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tuple.h>

#include <cstdint>
#include <limits>

namespace host::compress {

constexpr auto kDLUInt8 = DLDataType{.code = kDLUInt, .bits = 8, .lanes = 1};

using PlanC = CompressPlan;
using PlanW = WritePlan;
using PlanD = DecodePlan;

using RID_T = int64_t;
using R2T_T = int32_t;
using F2S_T = int64_t;
using IDX_T = int64_t;

/// NOTE: for the internal use, we pack the ragged and batch id, since both not exceed 65536
SGL_DEVICE __host__ PlanW pack_w(uint32_t ragged_id, uint32_t batch_id, int32_t seq_len) {
  return {static_cast<uint32_t>(ragged_id | batch_id << 16), seq_len};
}

/// NOTE: for the internal use, we pack the ragged and batch id, since both not exceed 65536
SGL_DEVICE uint2 unpack_w(PlanW plan) {
  return {static_cast<uint16_t>(plan.ragged_id), static_cast<uint16_t>(plan.ragged_id >> 16)};
}

struct Prefill0Params {
  PlanC* plan_c;
  PlanW* plan_w;
  const IDX_T* seq_lens_ptr;     // [batch_size]
  const IDX_T* extend_lens_ptr;  // [batch_size]
  uint32_t batch_size;
  uint32_t num_q_tokens;
  int32_t compress_ratio;
  int32_t swa_page_size;
};

struct Prefill1Params {
  PlanC* plan_c;
  PlanW* plan_w;
  const RID_T* rid_ptr;  // [batch_size]
  const R2T_T* r2t_ptr;  // [num_reqs, stride_r2t]
  const F2S_T* f2s_ptr;  // [num_swa_slots]
  int64_t stride_r2t;
  uint32_t num_c;
  uint32_t num_w;
  uint32_t num_c_padded;
  uint32_t num_w_padded;
  uint32_t num_work;
  int32_t swa_page_size;
  int32_t ring_size;
  int32_t compress_ratio;
};

struct DecodeParams {
  PlanD* plan_d;
  const RID_T* rid_ptr;  // [batch_size]
  const R2T_T* r2t_ptr;  // [num_reqs, stride_r2t]
  const F2S_T* f2s_ptr;  // [num_swa_slots]
  const IDX_T* seq_ptr;  // [batch_size]
  int64_t stride_r2t;
  uint32_t batch_size;
  int32_t swa_page_size;
  int32_t ring_size;
  int32_t compress_ratio;
};

struct Prefill1ParamsLegacy {
  PlanC* plan_c;
  PlanW* plan_w;
  const RID_T* rid_ptr;  // [batch_size]
  uint32_t num_c;
  uint32_t num_w;
  uint32_t num_c_padded;
  uint32_t num_w_padded;
  uint32_t num_work;
  int32_t compress_ratio;
};

struct DecodeParamsLegacy {
  PlanD* plan_d;
  const RID_T* rid_ptr;  // [batch_size]
  const IDX_T* seq_ptr;  // [batch_size]
  uint32_t batch_size;
  int32_t compress_ratio;
};

inline constexpr uint32_t kMaxPrefillBatchSize = 1024;

inline constexpr int32_t kMaxMTPDraftTokens = 4;

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
  static_assert(device::kWarpThreads == 32);
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

/// Warp-wide max/min for integer types. `device::warp::reduce_max` routes through
/// `dtype_trait<T>::max` which is only specialized for FP types.
SGL_DEVICE uint32_t warp_reduce_max_u32(uint32_t val) {
#pragma unroll
  for (uint32_t mask = 16; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, mask, 32));
  }
  return val;
}

SGL_DEVICE uint32_t warp_reduce_min_u32(uint32_t val) {
#pragma unroll
  for (uint32_t mask = 16; mask > 0; mask >>= 1) {
    val = min(val, __shfl_xor_sync(0xFFFFFFFF, val, mask, 32));
  }
  return val;
}

__global__ __launch_bounds__(1024, 1)  //
    void plan_compress_prefill_kernel0(const Prefill0Params params) {
  using namespace device;
  const auto tx = threadIdx.x;
  const auto block_size = kMaxPrefillBatchSize;
  constexpr auto kNumWarps = kMaxPrefillBatchSize / kWarpThreads;
  const auto cr = params.compress_ratio;
  const auto sps = params.swa_page_size;
  const bool is_overlap = (cr == 4);
  const int32_t window_size = cr * (is_overlap ? 2 : 1);

  alignas(128) __shared__ uint32_t counter_c;
  alignas(128) __shared__ uint32_t counter_w;
  __shared__ int32_t s_seq_len[kMaxPrefillBatchSize];
  __shared__ int32_t s_prefix_len[kMaxPrefillBatchSize];
  __shared__ uint32_t warp_max[kNumWarps];
  __shared__ uint32_t warp_min[kNumWarps];
  __shared__ uint32_t s_max_extend;
  __shared__ uint32_t s_min_extend;

  const auto lane_id = tx % kWarpThreads;
  const auto warp_id = tx / kWarpThreads;

  // === Stage A: load per-batch fields, init shared scratch ===
  int32_t seq_len = 0, extend_len = 0, prefix_len = 0;
  if (tx < params.batch_size) {
    seq_len = static_cast<int32_t>(params.seq_lens_ptr[tx]);
    extend_len = static_cast<int32_t>(params.extend_lens_ptr[tx]);
    prefix_len = seq_len - extend_len;
    s_seq_len[tx] = seq_len;
    s_prefix_len[tx] = prefix_len;
  }
  if (tx == 0) {
    counter_c = 0;
    counter_w = 0;
  }
  if (tx < kNumWarps) {
    warp_max[tx] = 0;
    warp_min[tx] = 0xFFFFFFFFu;
  }

  // === Stage B: min/max(extend_len) for MTP-uniform detection ===
  // For min, treat threads outside `batch_size` as +inf so they don't pull the min down.
  const uint32_t e_for_max = static_cast<uint32_t>(extend_len);
  const uint32_t e_for_min = (tx < params.batch_size) ? e_for_max : 0xFFFFFFFFu;
  warp_max[warp_id] = warp_reduce_max_u32(e_for_max);
  warp_min[warp_id] = warp_reduce_min_u32(e_for_min);
  __syncthreads();
  if (warp_id == 0) {
    s_max_extend = warp_reduce_max_u32(warp_max[lane_id]);
    s_min_extend = warp_reduce_min_u32(warp_min[lane_id]);
  }
  __syncthreads();

  const auto num_q = params.num_q_tokens;
  // MTP-uniform: every batch shares the same small extend_len `E`, so we can decompose
  // a global token id `k` into (batch_id, j) = (k / E, k % E) and skip the per-batch loop.
  const bool is_mtp_extend = (s_min_extend == s_max_extend) && (s_max_extend > 0) && (s_max_extend <= 32);

  // === Stage C: emit valid plans, slot allocation via shared-mem atomicAdd ===
  if (is_mtp_extend) {
    // Path 1: token-driven. Each global token id maps to exactly one (batch_id, j).
    const uint32_t E = s_max_extend;
    for (uint32_t k = tx; k < num_q; k += block_size) {
      const uint32_t batch_id = k / E;
      const uint32_t j = k % E;
      const int32_t pl = s_prefix_len[batch_id];
      const int32_t sl = s_seq_len[batch_id];
      const int32_t position = pl + static_cast<int32_t>(j);
      const uint32_t ragged_id = k;

      if ((position + 1) % cr == 0) {
        const int32_t buffer_len = window_size - min(static_cast<int32_t>(j) + 1, window_size);
        const uint32_t out_idx = atomicAdd(&counter_c, 1u);
        params.plan_c[out_idx] = {
            .seq_len = static_cast<uint32_t>(position + 1),
            .ragged_id = static_cast<uint16_t>(ragged_id),
            .buffer_len = static_cast<uint16_t>(buffer_len),
            .read_page_0 = -1,
            .read_page_1 = static_cast<int32_t>(batch_id),
        };
      }

      // w-event: A-region tail (position >= first_w_pos) plus, for c4, the swa_page
      // boundary band so the overlap page is fully populated when prefix-matching
      // resumes from a page edge. Pull `first_w_pos` back to also cover the trailing
      // `kMaxMTPDraftTokens` positions so MTP rollback always has the draft tokens.
      const int32_t last_c_pos = (sl / cr) * cr;
      const int32_t first_w_pos = min(last_c_pos - (is_overlap ? cr : 0), sl - kMaxMTPDraftTokens);
      bool do_write = position >= first_w_pos;
      if (!do_write && is_overlap) do_write = (position % sps) >= (sps - cr);
      if (do_write) {
        const uint32_t out_idx = atomicAdd(&counter_w, 1u);
        params.plan_w[out_idx] = pack_w(ragged_id, batch_id, position + 1);
      }
    }
  } else {
    // Path 2: general prefill (long extend_len). Iterate batches in an outer loop;
    // the whole block sweeps each batch's tokens in parallel.
    uint32_t base_e = 0;
    for (uint32_t batch_id = 0; batch_id < params.batch_size; ++batch_id) {
      const int32_t pl = s_prefix_len[batch_id];
      const int32_t sl = s_seq_len[batch_id];
      const int32_t el = sl - pl;
      const int32_t last_c_pos = (sl / cr) * cr;
      // Include the trailing `kMaxMTPDraftTokens` positions in the always-write tail
      // so MTP rollback can recover them; see kMaxMTPDraftTokens definition.
      const int32_t first_w_pos = min(last_c_pos - (is_overlap ? cr : 0), sl - kMaxMTPDraftTokens);

      for (int32_t j = static_cast<int32_t>(tx); j < el; j += static_cast<int32_t>(block_size)) {
        const int32_t position = pl + j;
        const uint32_t ragged_id = base_e + static_cast<uint32_t>(j);

        if ((position + 1) % cr == 0) {
          const int32_t buffer_len = window_size - min(j + 1, window_size);
          const uint32_t out_idx = atomicAdd(&counter_c, 1u);
          params.plan_c[out_idx] = {
              .seq_len = static_cast<uint32_t>(position + 1),
              .ragged_id = static_cast<uint16_t>(ragged_id),
              .buffer_len = static_cast<uint16_t>(buffer_len),
              .read_page_0 = -1,
              .read_page_1 = static_cast<int32_t>(batch_id),
          };
        }

        bool do_write = position >= first_w_pos;
        if (!do_write && is_overlap) do_write = (position % sps) >= (sps - cr);
        if (do_write) {
          const uint32_t out_idx = atomicAdd(&counter_w, 1u);
          params.plan_w[out_idx] = pack_w(ragged_id, static_cast<uint32_t>(batch_id), position + 1);
        }
      }
      base_e += static_cast<uint32_t>(el);
    }
  }
  __syncthreads();

  // === Stage D: pad [counter_c, num_q) / [counter_w, num_q) with invalid ===
  const auto total_c = counter_c;
  const auto total_w = counter_w;
  for (uint32_t k = total_c + tx; k < num_q; k += block_size) {
    params.plan_c[k] = PlanC::invalid();
  }
  for (uint32_t k = total_w + tx; k < num_q; k += block_size) {
    params.plan_w[k] = PlanW::invalid();
  }
}

/// NOTE: stage 1
__global__ void plan_compress_prefill_kernel_1(const Prefill1Params params) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.num_work) return;
  auto plan_c = idx < params.num_c ? params.plan_c[idx] : PlanC::invalid();
  auto plan_w = idx < params.num_w ? params.plan_w[idx] : PlanW::invalid();

  const auto compute_loc = [&](int32_t swa_loc) {
    const auto swa_page = swa_loc / params.swa_page_size;
    const auto ring_offset = swa_loc % params.ring_size;
    return swa_page * params.ring_size + ring_offset;
  };

  if (!plan_c.is_invalid()) {  // 1. in bound. 2. not masked
    const auto batch_id = plan_c.read_page_1;
    const auto rid = params.rid_ptr[batch_id];
    const auto mapping = params.r2t_ptr + rid * params.stride_r2t;
    // `seq_len` should be ratio-aligned here
    const auto position_1 = static_cast<int32_t>(plan_c.seq_len - 1);
    // only used for c4, harmless for c128
    const auto position_0 = max(position_1 - params.compress_ratio, 0);
    const auto raw_loc_0 = mapping[position_0];
    const auto raw_loc_1 = mapping[position_1];
    const auto swa_loc_0 = params.f2s_ptr[raw_loc_0];
    const auto swa_loc_1 = params.f2s_ptr[raw_loc_1];
    plan_c.read_page_0 = compute_loc(swa_loc_0) / params.compress_ratio;
    plan_c.read_page_1 = compute_loc(swa_loc_1) / params.compress_ratio;
    params.plan_c[idx] = plan_c;
  } else if (idx < params.num_c_padded) {
    params.plan_c[idx] = PlanC::invalid();
  }

  if (!plan_w.is_invalid()) {  // 1. in bound. 2. not masked
    const auto [ragged_id, batch_id] = unpack_w(plan_w);
    const auto rid = params.rid_ptr[batch_id];
    const auto mapping = params.r2t_ptr + rid * params.stride_r2t;
    // `seq_len` (`write_loc`) may not be aligned here
    const auto position = static_cast<int32_t>(plan_w.write_loc - 1);
    const auto raw_loc = mapping[position];
    const auto swa_loc = params.f2s_ptr[raw_loc];
    plan_w.ragged_id = ragged_id;
    plan_w.write_loc = compute_loc(swa_loc);
    params.plan_w[idx] = plan_w;
  } else if (idx < params.num_w_padded) {
    params.plan_w[idx] = PlanW::invalid();
  }
}

__global__ void plan_compress_decode_kernel(const DecodeParams params) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.batch_size) return;
  const auto rid = params.rid_ptr[idx];
  const auto mapping = params.r2t_ptr + rid * params.stride_r2t;
  const auto compute_loc = [&](int32_t swa_loc) {
    const auto swa_page = swa_loc / params.swa_page_size;
    const auto ring_offset = swa_loc % params.ring_size;
    return swa_page * params.ring_size + ring_offset;
  };
  const auto seq_len = static_cast<int32_t>(params.seq_ptr[idx]);
  const auto position_1 = static_cast<int32_t>(seq_len - 1);
  const auto position_0 = max(position_1 - params.compress_ratio, 0);
  const auto raw_loc_0 = mapping[position_0];
  const auto raw_loc_1 = mapping[position_1];
  const auto swa_loc_0 = params.f2s_ptr[raw_loc_0];
  const auto swa_loc_1 = params.f2s_ptr[raw_loc_1];
  const auto write_loc = compute_loc(swa_loc_1);
  const auto read_page_0 = compute_loc(swa_loc_0) / params.compress_ratio;
  const auto read_page_1 = write_loc / params.compress_ratio;
  params.plan_d[idx] = {
      .seq_len = static_cast<uint32_t>(seq_len),
      .write_loc = write_loc,
      .read_page_0 = read_page_0,
      .read_page_1 = read_page_1,
  };
}

__global__ void plan_compress_prefill_legacy_kernel(const Prefill1ParamsLegacy params) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.num_work) return;
  auto plan_c = idx < params.num_c ? params.plan_c[idx] : PlanC::invalid();
  auto plan_w = idx < params.num_w ? params.plan_w[idx] : PlanW::invalid();

  /// Per-request ring buffer slot translation:
  /// - c4:   page = rid * 2 + (position / 4) % 2; slot = page * 4 + position % 4
  /// - c128: page = rid;                          slot = rid * 128 + position % 128
  const auto legacy_compute_page = [&](int32_t rid, int32_t position) {
    if (params.compress_ratio == 4) return rid * 2 + ((position / 4) & 1);
    return rid;  // c128
  };
  const auto legacy_compute_loc = [&](int32_t rid, int32_t position) {
    const auto remainder = position % params.compress_ratio;
    return legacy_compute_page(rid, position) * params.compress_ratio + remainder;
  };

  if (!plan_c.is_invalid()) {
    const auto batch_id = plan_c.read_page_1;
    const auto rid = static_cast<int32_t>(params.rid_ptr[batch_id]);
    // `seq_len` is ratio-aligned for compress events
    const auto position_1 = static_cast<int32_t>(plan_c.seq_len) - 1;
    const auto position_0 = max(position_1 - params.compress_ratio, 0);
    plan_c.read_page_0 = legacy_compute_page(rid, position_0);
    plan_c.read_page_1 = legacy_compute_page(rid, position_1);
    params.plan_c[idx] = plan_c;
  } else if (idx < params.num_c_padded) {
    params.plan_c[idx] = PlanC::invalid();
  }

  if (!plan_w.is_invalid()) {
    const auto [ragged_id, batch_id] = unpack_w(plan_w);
    const auto rid = static_cast<int32_t>(params.rid_ptr[batch_id]);
    // `write_loc` carries (position + 1) at this stage; may not be ratio-aligned
    const auto position = static_cast<int32_t>(plan_w.write_loc) - 1;
    plan_w.ragged_id = ragged_id;
    plan_w.write_loc = legacy_compute_loc(rid, position);
    params.plan_w[idx] = plan_w;
  } else if (idx < params.num_w_padded) {
    params.plan_w[idx] = PlanW::invalid();
  }
}

__global__ void plan_compress_decode_legacy_kernel(const DecodeParamsLegacy params) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.batch_size) return;
  /// Per-request ring buffer slot translation:
  /// - c4:   page = rid * 2 + (position / 4) % 2; slot = page * 4 + position % 4
  /// - c128: page = rid;                          slot = rid * 128 + position % 128
  const auto legacy_compute_page = [&](int32_t rid, int32_t position) {
    if (params.compress_ratio == 4) return rid * 2 + ((position / 4) & 1);
    return rid;  // c128
  };
  const auto legacy_compute_loc = [&](int32_t rid, int32_t position) {
    const auto remainder = position % params.compress_ratio;
    return legacy_compute_page(rid, position) * params.compress_ratio + remainder;
  };
  const auto rid = static_cast<int32_t>(params.rid_ptr[idx]);
  const auto seq_len = static_cast<int32_t>(params.seq_ptr[idx]);
  const auto position_1 = seq_len - 1;
  const auto position_0 = max(position_1 - params.compress_ratio, 0);
  const auto write_loc = legacy_compute_loc(rid, position_1);
  const auto read_page_0 = legacy_compute_page(rid, position_0);
  const auto read_page_1 = legacy_compute_page(rid, position_1);
  params.plan_d[idx] = {
      .seq_len = static_cast<uint32_t>(seq_len),
      .write_loc = write_loc,
      .read_page_0 = read_page_0,
      .read_page_1 = read_page_1,
  };
}

using PrefillPlan = tvm::ffi::Tuple<tvm::ffi::Tensor, tvm::ffi::Tensor>;

/**
 * \brief Build c4/c128 prefill plan tensors. CPU-resident.
 * Inputs (all CPU-resident):
 * @param req_pool_indices  `[batch_size]` int64_t
 * @param req_to_token      `[num_reqs, max_tokens_per_req]` int64_t
 * @param full_to_swa       `[num_swa_slots]` int64_t
 * @param seq_lens          `[batch_size]` int64
 * @param extend_lens       `[batch_size]` int64
 * @param compress_plan     `[num_q_tokens, 16]` uint8 (output)
 * @param write_plan        `[num_q_tokens,  8]` uint8 (output)
 * @param compress_ratio 4 for c4, 128 for c128
 * @param use_cuda_graph Whether the plans will be used with cuda graph (affects padding)
 * @return (compress plan tensor, write plan tensor)
 */
inline PrefillPlan plan_compress_prefill(
    const tvm::ffi::TensorView req_pool_indices,  // GPU
    const tvm::ffi::TensorView req_to_token,      // GPU
    const tvm::ffi::TensorView full_to_swa,       // GPU
    const tvm::ffi::TensorView seq_lens,          // CPU/GPU
    const tvm::ffi::TensorView extend_lens,       // CPU/GPU
    const tvm::ffi::TensorView pin_buffer,        // CPU
    const uint32_t num_q_tokens,
    const int32_t compress_ratio,
    const int32_t swa_page_size,
    const int32_t ring_size,
    const bool use_cuda_graph) {
  auto B = SymbolicSize{"batch_size"};
  auto N = SymbolicSize{"num_q_tokens"};
  auto cpu_or_gpu = SymbolicDevice{};
  auto device_ = SymbolicDevice{};
  cpu_or_gpu.set_options<kDLCPU, kDLCUDA>();
  device_.set_options<kDLCUDA>();

  TensorMatcher({B})  //
      .with_dtype<RID_T>()
      .with_device(device_)
      .verify(req_pool_indices);
  TensorMatcher({-1, -1})  //
      .with_dtype<R2T_T>()
      .with_device(device_)
      .verify(req_to_token);
  TensorMatcher({-1})  //
      .with_dtype<F2S_T>()
      .with_device(device_)
      .verify(full_to_swa);
  TensorMatcher({B})  //
      .with_dtype<IDX_T>()
      .with_device(cpu_or_gpu)
      .verify(seq_lens)
      .verify(extend_lens);
  TensorMatcher({-1})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCPU>()
      .verify(pin_buffer);

  const bool is_overlap = (compress_ratio == 4);
  const int32_t window_size = compress_ratio * (is_overlap ? 2 : 1);

  const auto seq_ptr = static_cast<const IDX_T*>(seq_lens.data_ptr());
  const auto ext_ptr = static_cast<const IDX_T*>(extend_lens.data_ptr());
  const auto rid_ptr = static_cast<const RID_T*>(req_pool_indices.data_ptr());
  const auto r2t_ptr = static_cast<const R2T_T*>(req_to_token.data_ptr());
  const auto f2s_ptr = static_cast<const F2S_T*>(full_to_swa.data_ptr());

  const auto batch_size = static_cast<uint32_t>(B.unwrap());
  constexpr auto kMaxTokens = static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
  RuntimeCheck(compress_ratio == 4 || compress_ratio == 128);
  RuntimeCheck(batch_size <= num_q_tokens && num_q_tokens <= kMaxTokens);
  // `swa_page_size` >= `ring_size` >= `compress_ratio`
  RuntimeCheck(swa_page_size % ring_size == 0 && ring_size % compress_ratio == 0);

  const auto device = device_.unwrap();
  const auto stream = LaunchKernel::resolve_device(device);
  if (cpu_or_gpu.unwrap().device_type == kDLCUDA) {
    // GPU input path: kernel0 builds the (CPU-loop-equivalent) plan metadata directly
    // on device, padding to num_q_tokens with invalid; kernel_1 then finalizes the
    // SWA-translated read/write locations. Used for MTP / cuda-graph capture where
    // a host sync would be expensive.
    RuntimeCheck(batch_size <= kMaxPrefillBatchSize, "GPU plan only support batch size up to ", kMaxPrefillBatchSize);
    auto C = ffi::empty({num_q_tokens, sizeof(PlanC)}, kDLUInt8, device);
    auto W = ffi::empty({num_q_tokens, sizeof(PlanW)}, kDLUInt8, device);
    const auto params0 = Prefill0Params{
        .plan_c = static_cast<PlanC*>(C.data_ptr()),
        .plan_w = static_cast<PlanW*>(W.data_ptr()),
        .seq_lens_ptr = seq_ptr,
        .extend_lens_ptr = ext_ptr,
        .batch_size = batch_size,
        .num_q_tokens = num_q_tokens,
        .compress_ratio = compress_ratio,
        .swa_page_size = swa_page_size,
    };
    LaunchKernel(1, kMaxPrefillBatchSize, device)(plan_compress_prefill_kernel0, params0);
    // kernel_1 sees the already-padded buffers, so num_c == num_w == num_padded == num_q_tokens.
    const auto params1 = Prefill1Params{
        .plan_c = static_cast<PlanC*>(C.data_ptr()),
        .plan_w = static_cast<PlanW*>(W.data_ptr()),
        .rid_ptr = rid_ptr,
        .r2t_ptr = r2t_ptr,
        .f2s_ptr = f2s_ptr,
        .stride_r2t = req_to_token.size(1),
        .num_c = num_q_tokens,
        .num_w = num_q_tokens,
        .num_c_padded = num_q_tokens,
        .num_w_padded = num_q_tokens,
        .num_work = num_q_tokens,
        .swa_page_size = swa_page_size,
        .ring_size = ring_size,
        .compress_ratio = compress_ratio,
    };
    const auto block_size_1 = 256;
    const auto num_blocks_1 = div_ceil(params1.num_work, block_size_1);
    LaunchKernel(num_blocks_1, block_size_1, device)(plan_compress_prefill_kernel_1, params1);
    return PrefillPlan{std::move(C), std::move(W)};
  }

  // CPU input path: only here do we need the pinned scratch buffer.
  const auto pin_buffer_bytes = static_cast<size_t>(pin_buffer.numel()) * sizeof(uint8_t);
  RuntimeCheck(pin_buffer_bytes >= num_q_tokens * (sizeof(PlanC) + sizeof(PlanW)));
  const auto plan_c_ptr = reinterpret_cast<PlanC*>(pin_buffer.data_ptr());
  const auto plan_w_ptr = reinterpret_cast<PlanW*>(plan_c_ptr + num_q_tokens);

  uint32_t counter = 0;
  uint32_t counter_c = 0;
  uint32_t counter_w = 0;

  const auto should_compress = [=](int32_t position) { return (position + 1) % compress_ratio == 0; };
  for (const auto i : irange(batch_size)) {
    const int32_t seq_len = seq_ptr[i];
    const int32_t extend_len = ext_ptr[i];
    const int32_t prefix_len = seq_len - extend_len;
    const int32_t last_c_pos = seq_len / compress_ratio * compress_ratio;
    // Pull `first_w_pos` back so the trailing `kMaxMTPDraftTokens` positions always
    // fall into the always-write tail; without this, c128 with seq_len % 128 == 0
    // (or c4 with a tiny extend) would drop the MTP draft tokens on rollback.
    const int32_t first_w_pos = std::min(last_c_pos - (is_overlap ? compress_ratio : 0), seq_len - kMaxMTPDraftTokens);
    RuntimeCheck(0 < extend_len && extend_len <= seq_len);
    const auto should_write = [=](int32_t position) {
      if (position >= first_w_pos) return true;
      // Write the last `compress_ratio` positions of every swa_page so the
      // overlap page is fully populated when prefix-matching resumes from a
      // swa_page boundary.
      return is_overlap && position % swa_page_size >= (swa_page_size - compress_ratio);
    };
    for (const auto j : irange(extend_len)) {
      const int32_t position = prefix_len + j;
      const int32_t ragged_id = counter + j;
      if (should_compress(position)) {
        const auto buffer_len = window_size - std::min(j + 1, window_size);
        plan_c_ptr[counter_c++] = {
            .seq_len = static_cast<uint32_t>(position + 1),
            .ragged_id = static_cast<uint16_t>(ragged_id),
            .buffer_len = static_cast<uint16_t>(buffer_len),
            // to be filled by kernel
            .read_page_0 = -1,
            .read_page_1 = static_cast<int32_t>(i),
        };
      }
      if (should_write(position)) {
        plan_w_ptr[counter_w++] = pack_w(ragged_id, i, position + 1);
      }
    }
    counter += extend_len;
  }
  RuntimeCheck(counter == num_q_tokens);

  const auto copy_to_device = [stream](void* cuda_ptr, auto* host_ptr, size_t count) {
    const auto size_bytes = count * sizeof(*host_ptr);
    RuntimeDeviceCheck(cudaMemcpyAsync(cuda_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice, stream));
  };
  const auto num_c_padded = use_cuda_graph ? num_q_tokens : counter_c;
  const auto num_w_padded = use_cuda_graph ? num_q_tokens : counter_w;
  auto C = ffi::empty({num_c_padded, sizeof(PlanC)}, kDLUInt8, device);
  auto W = ffi::empty({num_w_padded, sizeof(PlanW)}, kDLUInt8, device);
  copy_to_device(C.data_ptr(), plan_c_ptr, counter_c);
  copy_to_device(W.data_ptr(), plan_w_ptr, counter_w);
  const auto params = Prefill1Params{
      .plan_c = static_cast<PlanC*>(C.data_ptr()),
      .plan_w = static_cast<PlanW*>(W.data_ptr()),
      .rid_ptr = rid_ptr,
      .r2t_ptr = r2t_ptr,
      .f2s_ptr = f2s_ptr,
      .stride_r2t = req_to_token.size(1),
      .num_c = counter_c,
      .num_w = counter_w,
      .num_c_padded = num_c_padded,
      .num_w_padded = num_w_padded,
      .num_work = std::max(num_c_padded, num_w_padded),
      .swa_page_size = swa_page_size,
      .ring_size = ring_size,
      .compress_ratio = compress_ratio,
  };
  const auto block_size = 256;
  const auto num_blocks = div_ceil(params.num_work, block_size);
  LaunchKernel(num_blocks, block_size, device)(plan_compress_prefill_kernel_1, params);
  return PrefillPlan{std::move(C), std::move(W)};
}

inline tvm::ffi::Tensor plan_compress_decode(
    const tvm::ffi::TensorView req_pool_indices,  // GPU
    const tvm::ffi::TensorView req_to_token,      // GPU
    const tvm::ffi::TensorView full_to_swa,       // GPU
    const tvm::ffi::TensorView seq_lens,          // CPU/GPU
    const int32_t compress_ratio,
    const int32_t swa_page_size,
    const int32_t ring_size) {
  auto B = SymbolicSize{"batch_size"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({B})  //
      .with_dtype<RID_T>()
      .with_device(device_)
      .verify(req_pool_indices);
  TensorMatcher({-1, -1})  //
      .with_dtype<R2T_T>()
      .with_device(device_)
      .verify(req_to_token);
  TensorMatcher({-1})  //
      .with_dtype<F2S_T>()
      .with_device(device_)
      .verify(full_to_swa);
  TensorMatcher({B})  //
      .with_dtype<IDX_T>()
      .with_device(device_)
      .verify(seq_lens);

  const auto batch_size = static_cast<uint32_t>(B.unwrap());
  const auto device = device_.unwrap();
  auto D = ffi::empty({batch_size, sizeof(PlanD)}, kDLUInt8, device);
  const auto params = DecodeParams{
      .plan_d = static_cast<PlanD*>(D.data_ptr()),
      .rid_ptr = static_cast<const RID_T*>(req_pool_indices.data_ptr()),
      .r2t_ptr = static_cast<const R2T_T*>(req_to_token.data_ptr()),
      .f2s_ptr = static_cast<const F2S_T*>(full_to_swa.data_ptr()),
      .seq_ptr = static_cast<const IDX_T*>(seq_lens.data_ptr()),
      .stride_r2t = req_to_token.size(1),
      .batch_size = batch_size,
      .swa_page_size = swa_page_size,
      .ring_size = ring_size,
      .compress_ratio = compress_ratio,
  };
  const auto block_size = 256;
  const auto num_blocks = div_ceil(batch_size, block_size);
  LaunchKernel(num_blocks, block_size, device)(plan_compress_decode_kernel, params);
  return D;
}

/**
 * \brief Build c4/c128 prefill plan tensors for the legacy non-paged ring
 * buffer. Uses only `req_pool_indices` to derive ring slots:
 *   - c4 (overlap):  each request occupies 2 contiguous pages (8 token slots)
 *   - c128:          each request occupies 1 page (128 token slots)
 *
 * Inputs:
 * @param req_pool_indices  `[batch_size]` int64 (GPU)
 * @param seq_lens          `[batch_size]` int64 (CPU)
 * @param extend_lens       `[batch_size]` int64 (CPU)
 * @param pin_buffer        pinned scratch (CPU uint8)
 * @return (compress plan tensor, write plan tensor)
 */
inline PrefillPlan plan_compress_prefill_legacy(
    const tvm::ffi::TensorView req_pool_indices,  // GPU
    const tvm::ffi::TensorView seq_lens,          // CPU
    const tvm::ffi::TensorView extend_lens,       // CPU
    const tvm::ffi::TensorView pin_buffer,        // CPU
    const uint32_t num_q_tokens,
    const int32_t compress_ratio,
    const bool use_cuda_graph) {
  auto B = SymbolicSize{"batch_size"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({B})  //
      .with_dtype<RID_T>()
      .with_device(device_)
      .verify(req_pool_indices);
  TensorMatcher({B})  //
      .with_dtype<IDX_T>()
      .with_device<kDLCPU>()
      .verify(seq_lens)
      .verify(extend_lens);
  TensorMatcher({-1})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCPU>()
      .verify(pin_buffer);

  const auto pin_buffer_bytes = static_cast<size_t>(pin_buffer.numel()) * sizeof(uint8_t);
  RuntimeCheck(pin_buffer_bytes >= num_q_tokens * (sizeof(PlanC) + sizeof(PlanW)));
  const auto plan_c_ptr = reinterpret_cast<PlanC*>(pin_buffer.data_ptr());
  const auto plan_w_ptr = reinterpret_cast<PlanW*>(plan_c_ptr + num_q_tokens);

  const bool is_overlap = (compress_ratio == 4);
  const auto seq_ptr = static_cast<const IDX_T*>(seq_lens.data_ptr());
  const auto ext_ptr = static_cast<const IDX_T*>(extend_lens.data_ptr());
  const auto rid_ptr = static_cast<const RID_T*>(req_pool_indices.data_ptr());

  const auto window_size = compress_ratio * (is_overlap ? 2 : 1);
  const auto batch_size = static_cast<uint32_t>(B.unwrap());
  constexpr auto kMaxTokens = static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
  RuntimeCheck(compress_ratio == 4 || compress_ratio == 128);
  RuntimeCheck(batch_size <= num_q_tokens && num_q_tokens <= kMaxTokens);

  uint32_t counter = 0;
  uint32_t counter_c = 0;
  uint32_t counter_w = 0;
  const auto should_compress = [=](int32_t position) { return (position + 1) % compress_ratio == 0; };
  for (const auto i : irange(batch_size)) {
    const int32_t seq_len = seq_ptr[i];
    const int32_t extend_len = ext_ptr[i];
    const int32_t prefix_len = seq_len - extend_len;
    const int32_t last_c_pos = seq_len / compress_ratio * compress_ratio;
    const int32_t first_w_pos = last_c_pos - (is_overlap ? compress_ratio : 0);
    RuntimeCheck(0 < extend_len && extend_len <= seq_len);
    const auto should_write = [=](int32_t position) { return position >= first_w_pos; };
    for (const auto j : irange(extend_len)) {
      const int32_t position = prefix_len + j;
      const int32_t ragged_id = counter + j;
      if (should_compress(position)) {
        const auto buffer_len = window_size - std::min(j + 1, window_size);
        plan_c_ptr[counter_c++] = {
            .seq_len = static_cast<uint32_t>(position + 1),
            .ragged_id = static_cast<uint16_t>(ragged_id),
            .buffer_len = static_cast<uint16_t>(buffer_len),
            // to be filled by kernel
            .read_page_0 = -1,
            .read_page_1 = static_cast<int32_t>(i),
        };
      }
      if (should_write(position)) {
        plan_w_ptr[counter_w++] = pack_w(ragged_id, i, position + 1);
      }
    }
    counter += extend_len;
  }
  RuntimeCheck(counter == num_q_tokens);

  const auto device = device_.unwrap();
  const auto stream = LaunchKernel::resolve_device(device);
  const auto copy_to_device = [stream](void* cuda_ptr, auto* host_ptr, size_t count) {
    const auto size_bytes = count * sizeof(*host_ptr);
    RuntimeDeviceCheck(cudaMemcpyAsync(cuda_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice, stream));
  };
  const auto num_c_padded = use_cuda_graph ? num_q_tokens : counter_c;
  const auto num_w_padded = use_cuda_graph ? num_q_tokens : counter_w;
  auto C = ffi::empty({num_c_padded, sizeof(PlanC)}, kDLUInt8, device);
  auto W = ffi::empty({num_w_padded, sizeof(PlanW)}, kDLUInt8, device);
  copy_to_device(C.data_ptr(), plan_c_ptr, counter_c);
  copy_to_device(W.data_ptr(), plan_w_ptr, counter_w);
  const auto params = Prefill1ParamsLegacy{
      .plan_c = static_cast<PlanC*>(C.data_ptr()),
      .plan_w = static_cast<PlanW*>(W.data_ptr()),
      .rid_ptr = rid_ptr,
      .num_c = counter_c,
      .num_w = counter_w,
      .num_c_padded = num_c_padded,
      .num_w_padded = num_w_padded,
      .num_work = std::max(num_c_padded, num_w_padded),
      .compress_ratio = compress_ratio,
  };
  const auto block_size = 256;
  const auto num_blocks = div_ceil(params.num_work, block_size);
  if (num_blocks > 0) {
    LaunchKernel(num_blocks, block_size, device)(plan_compress_prefill_legacy_kernel, params);
  }
  return PrefillPlan{std::move(C), std::move(W)};
}

inline tvm::ffi::Tensor plan_compress_decode_legacy(
    const tvm::ffi::TensorView req_pool_indices,  // GPU
    const tvm::ffi::TensorView seq_lens,          // GPU
    const int32_t compress_ratio) {
  auto B = SymbolicSize{"batch_size"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({B})  //
      .with_dtype<RID_T>()
      .with_device(device_)
      .verify(req_pool_indices);
  TensorMatcher({B})  //
      .with_dtype<IDX_T>()
      .with_device(device_)
      .verify(seq_lens);
  RuntimeCheck(compress_ratio == 4 || compress_ratio == 128);

  const auto batch_size = static_cast<uint32_t>(B.unwrap());
  const auto device = device_.unwrap();
  auto D = ffi::empty({batch_size, sizeof(PlanD)}, kDLUInt8, device);
  const auto params = DecodeParamsLegacy{
      .plan_d = static_cast<PlanD*>(D.data_ptr()),
      .rid_ptr = static_cast<const RID_T*>(req_pool_indices.data_ptr()),
      .seq_ptr = static_cast<const IDX_T*>(seq_lens.data_ptr()),
      .batch_size = batch_size,
      .compress_ratio = compress_ratio,
  };
  const auto block_size = 256;
  const auto num_blocks = div_ceil(batch_size, block_size);
  LaunchKernel(num_blocks, block_size, device)(plan_compress_decode_legacy_kernel, params);
  return D;
}

}  // namespace host::compress

using namespace host::compress;  // expose binding
