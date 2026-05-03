/******************************************************************************
 * Fused RoPE + Hadamard for the NSA Lightning Indexer
 *
 *   Replaces the chain
 *     apply_rope_with_cos_sin_cache_inplace(q_rope, k_rope, ...)   // 1 launch
 *     hadamard_transform(query, scale=1/sqrt(head_dim))            // 1 launch
 *     hadamard_transform(key,   scale=1/sqrt(head_dim))            // 1 launch
 *   with a single in-place kernel that:
 *     - reads `query` and `key` once (full head_dim columns),
 *     - applies neox-style RoPE on the rope-half (first kRopeDim dims) per head
 *       using `cos_sin_cache[positions[token]]`,
 *     - applies a kHeadDim-wide Hadamard butterfly on the full per-head row,
 *     - stores back in place with the orthogonal scale 1 / sqrt(kHeadDim).
 *
 *   Per-thread layout (one CTA covers `kBlockSize / kWorkThreads` workers, where
 *   each worker = one (token, head) pair):
 *     - kWorkThreads = kHeadDim / kNElts  (e.g. 16 for kHeadDim=128 bf16)
 *     - kNElts       = 16 / sizeof(DType) (8 for bf16/fp16; 4 for fp32)
 *     - kLogNElts    = log2(kNElts)
 *     - kLogWorkThreads = log2(kWorkThreads)
 *
 *   Hadamard butterfly composition (reuses fast-hadamard-transform helpers):
 *     - kLogNElts in-thread stages           (`hadamard_mult_thread`)
 *     - kLogWorkThreads warp-shuffle stages  (`hadamard_mult_warp`)
 *     - Total stages = log2(kHeadDim).
 *
 *   RoPE pairing inside the worker:
 *     - threads [0, kRopeXThreads)            hold q_x  = head[0          : kRopeDim/2)
 *     - threads [kRopeXThreads, kRopeYThreads)hold q_y  = head[kRopeDim/2 : kRopeDim)
 *     - threads [kRopeYThreads, kWorkThreads) hold the nope half (no RoPE)
 *     - pair partner = lane ^ kRopeXThreads (warp-shuffle within width=kWorkThreads)
 *
 *   Architecture gating:
 *     - The kernel uses warp shuffles + 128-bit vec loads only; portable from
 *       SM70+. PDL hooks are gated by kUsePDL, which the Python wrapper sets
 *       from is_arch_support_pdl() (SM90+, covers SM100/SM103).
 ******************************************************************************/
#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDType, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/runtime.cuh>  // For runtime::get_blocks_per_sm, get_sm_count
#include <sgl_kernel/type.cuh>     // For dtype_trait, fp16_t, bf16_t, fp32_t, packed_t, cast
#include <sgl_kernel/utils.cuh>    // For SGL_DEVICE, LaunchKernel, PDLWait/Trigger
#include <sgl_kernel/vec.cuh>      // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "fast_hadamard_transform_common.h"  // For hadamard_mult_thread, hadamard_mult_warp, cilog2
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>

namespace {

struct FusedRopeHadamardParams {
  void* __restrict__ q_ptr;
  // NOTE: pre-offset by `-num_qo_heads * head_stride_bytes` so that index math
  // can use a unified head_id range [0, num_qo_heads + num_kv_heads).
  void* __restrict__ k_ptr;
  const void* __restrict__ cos_sin_cache_ptr;
  const void* __restrict__ positions;
  int64_t q_stride_bytes;     // byte stride between tokens for q
  int64_t k_stride_bytes;     // byte stride between tokens for k
  int64_t head_stride_bytes;  // byte stride between heads (same for q and k by validation)
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  float had_scale;  // 1 / sqrt(kHeadDim)
};

constexpr uint32_t kBlockSize = 128;

template <
    bool kIsNeox,
    int kHeadDim,
    int kRopeDim,
    bool kUsePDL,
    typename DType,
    typename IdType>
__global__ __launch_bounds__(kBlockSize)  //
    void fused_rope_hadamard_kernel(const __grid_constant__ FusedRopeHadamardParams params) {
  using namespace device;

  // ---- compile-time geometry ------------------------------------------------
  static_assert(sizeof(DType) == 2, "fused_rope_hadamard: only fp16/bf16 supported in this version");

  constexpr int kNElts = 16 / sizeof(DType);             // 8 for bf16/fp16
  constexpr int kWorkThreads = kHeadDim / kNElts;        // 16 for head_dim=128
  constexpr int kLogNElts = cilog2(kNElts);              // 3
  constexpr int kLogWorkThreads = cilog2(kWorkThreads);  // 4
  constexpr int kNChunks = 1;
  constexpr int kWorkersPerBlock = kBlockSize / kWorkThreads;

  static_assert(kHeadDim > 0 && (kHeadDim & (kHeadDim - 1)) == 0, "kHeadDim must be a power of 2");
  static_assert(kRopeDim > 0 && kRopeDim <= kHeadDim, "kRopeDim must be in (0, kHeadDim]");
  static_assert(
      kRopeDim % (2 * kNElts) == 0,
      "kRopeDim must be divisible by 2 * kNElts so the rope/nope split aligns to vector lanes");
  static_assert(
      kWorkThreads >= 4 && kWorkThreads <= 32,
      "kWorkThreads in [4, 32]; larger head_dim needs cross-warp Hadamard staging");
  static_assert(kBlockSize % kWorkThreads == 0);
  static_assert((1 << kLogNElts) == kNElts);
  static_assert((1 << kLogWorkThreads) == kWorkThreads);

  // RoPE rope-half partitioning (within a worker):
  constexpr int kRopeHalf = kRopeDim / 2;            // dims [0, kRopeHalf) and [kRopeHalf, kRopeDim)
  constexpr int kRopeXThreads = kRopeHalf / kNElts;  // first  kRopeHalf dims live in these lanes
  constexpr int kRopeYThreads =
      kRopeDim / kNElts;  // second kRopeHalf dims live in lanes [kRopeXThreads, kRopeYThreads)

  // ---- worker / lane identity ----------------------------------------------
  const uint32_t lane_id = threadIdx.x % kWorkThreads;
  const uint32_t worker_in_block = threadIdx.x / kWorkThreads;
  const uint32_t start_worker_id = blockIdx.x * kWorkersPerBlock + worker_in_block;
  const uint32_t total_workers = (params.num_qo_heads + params.num_kv_heads) * params.num_tokens;
  const uint32_t worker_stride = gridDim.x * kWorkersPerBlock;

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t work_id = start_worker_id; work_id < total_workers; work_id += worker_stride) {
    const uint32_t num_q_and_k_heads = params.num_qo_heads + params.num_kv_heads;
    const uint32_t token_id = work_id / num_q_and_k_heads;
    const uint32_t head_id = work_id % num_q_and_k_heads;
    const bool load_q = head_id < params.num_qo_heads;

    // Compute base row pointer. params.k_ptr has been pre-offset by
    // `-num_qo_heads * head_stride_bytes` on the host so the kernel can index
    // both q and k with the same `head_id` running over [0, num_q_and_k_heads).
    void* row_ptr = pointer::offset(
        load_q ? params.q_ptr : params.k_ptr,
        token_id * (load_q ? params.q_stride_bytes : params.k_stride_bytes),
        head_id * params.head_stride_bytes);

    // ---- Step 1: vec-load 128-bit per thread, cast to fp32 -----------------
    using vec_t = AlignedVector<DType, kNElts>;
    float x_vals[kNChunks][kNElts];
    {
      auto v = load_as<vec_t>(row_ptr, lane_id);
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[0][i] = static_cast<float>(v[i]);
      }
    }

    // ---- Step 2: in-register RoPE on the rope-half -------------------------
    //
    //   Two layouts (selected at compile time by kIsNeox):
    //
    //   Neox (split-half pairing): pair (head[i], head[i + kRopeHalf]) for
    //     i in [0, kRopeHalf). Lanes [0, kRopeXThreads) hold q_x; lanes
    //     [kRopeXThreads, kRopeYThreads) hold q_y. Pair partner via
    //     __shfl_xor_sync(_, _, kRopeXThreads, kWorkThreads). Both x-lane and
    //     y-lane in a pair load the SAME (cos[i], sin[i]).
    //
    //   Non-neox (interleaved pairing): pair (head[2i], head[2i+1]) for
    //     i in [0, kRopeHalf). Each lane holds kPairsPerThread = kNElts/2
    //     full pairs in its own 8 elements; no cross-lane shuffle is needed.
    //     Lanes [0, kRopeYThreads) participate; lane t loads cos[t * kPairs ..]
    //     and sin[t * kPairs ..] from the cos / sin halves of cos_sin_cache.
    {
      const auto pos = static_cast<const IdType*>(params.positions)[token_id];
      const float* cos_cache = static_cast<const float*>(params.cos_sin_cache_ptr);
      const float* sin_cache = cos_cache + kRopeHalf;
      const int64_t pos_offset = static_cast<int64_t>(pos) * static_cast<int64_t>(kRopeDim);

      if constexpr (kIsNeox) {
        float cos_vals[kNElts] = {0};
        float sin_vals[kNElts] = {0};
        if (lane_id < kRopeYThreads) {
          const uint32_t lane_in_rope = lane_id & (kRopeXThreads - 1);  // 0..kRopeXThreads-1
          const int64_t base = pos_offset + static_cast<int64_t>(lane_in_rope) * kNElts;
#pragma unroll
          for (int i = 0; i < kNElts; ++i) {
            cos_vals[i] = cos_cache[base + i];
            sin_vals[i] = sin_cache[base + i];
          }
        }

        // Width = kWorkThreads keeps shuffles confined to one worker's lanes
        // when multiple workers share a warp (kBlockSize / kWorkThreads
        // workers/warp). All kWorkThreads lanes call shfl uniformly.
#pragma unroll
        for (int i = 0; i < kNElts; ++i) {
          const float my_val = x_vals[0][i];
          const float pair_val = __shfl_xor_sync(0xFFFFFFFFu, my_val, kRopeXThreads, kWorkThreads);
          if (lane_id < kRopeXThreads) {
            // q_x lane: my=x, pair=y → new_x = x*cos − y*sin
            x_vals[0][i] = my_val * cos_vals[i] - pair_val * sin_vals[i];
          } else if (lane_id < kRopeYThreads) {
            // q_y lane: my=y, pair=x → new_y = x*sin + y*cos = pair*sin + my*cos
            x_vals[0][i] = pair_val * sin_vals[i] + my_val * cos_vals[i];
          }
          // lane_id >= kRopeYThreads: nope, leave x_vals unchanged.
        }
      } else {
        // Non-neox: pairs are adjacent inside each lane, all in registers.
        constexpr int kPairsPerThread = kNElts / 2;  // 4 for kNElts=8
        static_assert(kNElts % 2 == 0, "non-neox requires kNElts to be even");
        if (lane_id < kRopeYThreads) {
          const int64_t base = pos_offset + static_cast<int64_t>(lane_id) * kPairsPerThread;
          float cos_vals[kPairsPerThread];
          float sin_vals[kPairsPerThread];
#pragma unroll
          for (int m = 0; m < kPairsPerThread; ++m) {
            cos_vals[m] = cos_cache[base + m];
            sin_vals[m] = sin_cache[base + m];
          }
#pragma unroll
          for (int m = 0; m < kPairsPerThread; ++m) {
            const float x = x_vals[0][2 * m];
            const float y = x_vals[0][2 * m + 1];
            x_vals[0][2 * m] = x * cos_vals[m] - y * sin_vals[m];
            x_vals[0][2 * m + 1] = x * sin_vals[m] + y * cos_vals[m];
          }
        }
        // lane_id >= kRopeYThreads: nope, leave x_vals unchanged.
      }
    }

    // ---- Step 3: in-register kHeadDim-wide Hadamard butterfly --------------
    //   kLogNElts in-thread stages, then kLogWorkThreads warp-shuffle stages.
    //   At kHeadDim=128 / kWorkThreads=16 there are no cross-warp stages.
    //   Width inside `hadamard_mult_warp` is kWorkThreads (matches our worker).
    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWorkThreads, 0, kNChunks, kNElts>(x_vals);

    // ---- Step 4: cast back to DType, apply scale, store in place -----------
    {
      vec_t v;
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        v[i] = static_cast<DType>(x_vals[0][i] * params.had_scale);
      }
      store_as<vec_t>(row_ptr, v, lane_id);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kIsNeox, int kHeadDim, int kRopeDim, bool kUsePDL, typename DType>
struct FusedRopeHadamardKernel {
  static constexpr int kNElts = 16 / sizeof(DType);
  static constexpr int kWorkThreads = kHeadDim / kNElts;

  template <typename IdType>
  static constexpr auto _kernel = fused_rope_hadamard_kernel<kIsNeox, kHeadDim, kRopeDim, kUsePDL, DType, IdType>;

  static auto get_num_sm(DLDevice device) {
    static const auto kNumSM = host::runtime::get_sm_count(device.device_id);
    return kNumSM;
  }

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto Hd = SymbolicSize{"head_dim"};
    auto Rd = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_token_stride"};
    auto Dk = SymbolicSize{"k_token_stride"};
    auto Dh = SymbolicSize{"head_stride"};
    Hd.set_value(kHeadDim);
    Rd.set_value(kRopeDim);

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto id_type = SymbolicDType{};

    // q : [N, Q, kHeadDim], in-place
    TensorMatcher({N, Q, Hd}).with_strides({Dq, Dh, 1}).with_dtype<DType>().with_device(device).verify(q);
    // k : [N, K, kHeadDim], in-place; head stride must match q's
    TensorMatcher({N, K, Hd}).with_strides({Dk, Dh, 1}).with_dtype<DType>().with_device(device).verify(k);
    // cos_sin_cache : [max_pos, kRopeDim], fp32; first kRopeDim/2 = cos, second = sin
    TensorMatcher({-1, Rd}).with_dtype<float>().with_device(device).verify(cos_sin_cache);
    // positions : [N], int32 or int64
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap()) * sizeof(DType);
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap()) * sizeof(DType);
    const auto head_stride_bytes = static_cast<int64_t>(Dh.unwrap()) * sizeof(DType);

    // Pre-offset k pointer so the kernel can index q/k uniformly via head_id ∈ [0, Q+K).
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;

    FusedRopeHadamardParams params;
    std::memset(&params, 0, sizeof(params));
    params.q_ptr = const_cast<void*>(q.data_ptr());
    params.k_ptr = pointer::offset(const_cast<void*>(k.data_ptr()), -k_offset);
    params.cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    params.positions = positions.data_ptr();
    params.q_stride_bytes = q_stride_bytes;
    params.k_stride_bytes = k_stride_bytes;
    params.head_stride_bytes = head_stride_bytes;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.num_tokens = num_tokens;
    params.had_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    const DLDevice dev = device.unwrap();
    const auto is_int32 = id_type.is_type<int32_t>();
    const auto kernel = is_int32 ? _kernel<int32_t> : _kernel<int64_t>;

    // Persistent grid sized to occupancy × SM count, capped by needed workers.
    static const uint32_t kOccupancy[2] = {
        runtime::get_blocks_per_sm(_kernel<int32_t>, kBlockSize),
        runtime::get_blocks_per_sm(_kernel<int64_t>, kBlockSize),
    };
    const uint32_t num_sm = get_num_sm(dev);
    const uint32_t max_blocks = num_sm * kOccupancy[is_int32 ? 0 : 1];
    const uint32_t total_workers = (num_qo_heads + num_kv_heads) * num_tokens;
    constexpr uint32_t kWorkersPerBlock = kBlockSize / kWorkThreads;
    const uint32_t needed_blocks = div_ceil(total_workers, kWorkersPerBlock);
    const uint32_t num_blocks = std::min(max_blocks, needed_blocks);

    LaunchKernel(num_blocks, kBlockSize, dev)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
