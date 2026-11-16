/**
 * Fused per-head RMS norm + sigmoid gate for the MuseSpark v1.2 attention output.
 *
 * x: [num_tokens * num_heads, head_dim], contiguous, row = (token, head).
 * y: same shape and dtype as x.
 * weight: [head_dim], the already-resolved effective weight (zero-centered gamma
 *         is folded in on the Python side).
 * gate: [num_tokens, num_groups, heads_per_group * head_dim] with arbitrary
 *       strides on the first two dims and a contiguous innermost dim; a flat
 *       [num_tokens, num_heads * head_dim] gate is the num_groups == 1 case.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/activation.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

struct NormGateParams {
  const void* __restrict__ x;
  const void* __restrict__ gate;
  const void* __restrict__ weight;
  void* __restrict__ y;
  int64_t gate_stride_token;  // in elements
  int64_t gate_stride_group;  // in elements
  uint32_t num_tokens;
  uint32_t num_heads;
  uint32_t heads_per_group;
  float eps;
};

// One kVecBytes vector per thread; a head_dim row spans kThreadsPerRow lanes.
template <typename T, int64_t kHeadDim, int32_t kVecBytes>
struct NormGateTrait {
  using Packed = packed_t<T>;
  static constexpr int64_t kPack = kVecBytes / static_cast<int64_t>(sizeof(Packed));
  static constexpr int64_t kElemsPerThread = kPack * 2;
  static_assert(kVecBytes <= static_cast<int32_t>(device::kMaxVecBytes), "vector wider than this arch loads");
  static_assert(kHeadDim % kElemsPerThread == 0 && kHeadDim / kElemsPerThread <= 32, "a row must fit in one warp");
  static constexpr int64_t kThreadsPerRow = kHeadDim / kElemsPerThread;
  static constexpr int64_t kRowsPerWarp = device::kWarpThreads / kThreadsPerRow;
  using Storage = device::AlignedVector<Packed, kPack>;
};

template <
    typename T,
    int64_t kHeadDim,
    int32_t kVecBytes,
    int32_t kWarpsPerBlock,
    bool kHasNorm,
    bool kHasWeight,
    bool kHasGate,
    bool kSingleGroup,
    bool kExact,
    bool kUsePDL>
// Worth ~5%; unconstrained, ptxas buys registers this kernel does not need.
__global__ __launch_bounds__(
    kWarpsPerBlock* device::kWarpThreads,
    1024 /
        (kWarpsPerBlock * device::kWarpThreads)) void norm_gate_kernel(const __grid_constant__ NormGateParams params) {
  using namespace device;
  using Trait = NormGateTrait<T, kHeadDim, kVecBytes>;
  using Storage = typename Trait::Storage;
  using Packed = typename Trait::Packed;

  constexpr int64_t kPack = Trait::kPack;
  constexpr int64_t kThreadsPerRow = Trait::kThreadsPerRow;
  constexpr int64_t kRowsPerBlock = Trait::kRowsPerWarp * kWarpsPerBlock;

  const uint32_t lane = threadIdx.x % kThreadsPerRow;
  const uint32_t head = blockIdx.x * kRowsPerBlock + threadIdx.x / kThreadsPerRow;
  const uint32_t num_heads = params.num_heads;
  // Clamped, not branched: the warp reduction below needs every lane present.
  const uint32_t safe_head = (kExact || head < num_heads) ? head : num_heads - 1;

  const auto x_row = static_cast<const T*>(params.x) + static_cast<int64_t>(safe_head) * kHeadDim;
  const auto y_row = static_cast<T*>(params.y) + static_cast<int64_t>(safe_head) * kHeadDim;
  const int64_t row_pitch = static_cast<int64_t>(num_heads) * kHeadDim;

  int64_t gate_head_offset = 0;
  if constexpr (kHasGate) {
    if constexpr (kSingleGroup) {
      gate_head_offset = static_cast<int64_t>(safe_head) * kHeadDim;
    } else {
      const uint32_t group = safe_head / params.heads_per_group;
      const uint32_t head_in_group = safe_head - group * params.heads_per_group;
      gate_head_offset = group * params.gate_stride_group + static_cast<int64_t>(head_in_group) * kHeadDim;
    }
  }
  const auto gate_row = kHasGate ? (static_cast<const T*>(params.gate) + gate_head_offset) : 0;

  PDLWaitPrimary<kUsePDL>();

  Storage wv, gv, yv;

  for (uint32_t token = blockIdx.y; kExact ? token == blockIdx.y : token < params.num_tokens; token += gridDim.y) {
    Storage xv;
    xv.load(assume_global(x_row + token * row_pitch), lane);

    float sum_of_squares = 0.0f;
    if constexpr (kHasNorm) {
#pragma unroll
      for (int32_t p = 0; p < kPack; ++p) {
        const auto v = cast<fp32x2_t>(xv[p]);
        sum_of_squares = fmaf(v.x, v.x, sum_of_squares);
        sum_of_squares = fmaf(v.y, v.y, sum_of_squares);
      }
    }
    if constexpr (kHasGate) {
      gv.load(assume_global(gate_row + token * params.gate_stride_token), lane);
    }
    // ---- PDL: Trigger next kernel here to increase overlap
    PDLTriggerSecondary<kUsePDL>();
    float rstd = 0.0f;
    if constexpr (kHasNorm) {
      sum_of_squares = warp::reduce_sum<kThreadsPerRow>(sum_of_squares);
      rstd = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

      if constexpr (kHasWeight) {
        wv.load(assume_global(params.weight), lane);
      }
    }
#pragma unroll
    for (int32_t p = 0; p < kPack; ++p) {
      auto acc = cast<fp32x2_t>(xv[p]);
      if constexpr (kHasNorm) {
        acc.x *= rstd;
        acc.y *= rstd;
        if constexpr (kHasWeight) {
          const auto w = cast<fp32x2_t>(wv[p]);
          acc.x *= w.x;
          acc.y *= w.y;
        }
      }
      if constexpr (kHasGate) {
        const auto g = cast<fp32x2_t>(gv[p]);
        acc.x *= activation::sigmoid(g.x);
        acc.y *= activation::sigmoid(g.y);
      }
      yv[p] = cast<Packed>(acc);
    }

    if (kExact || head < num_heads) {
      yv.store(y_row + token * row_pitch, lane);
    }
  }
}

struct NormGateConfig {
  int32_t vec_bytes;
  int32_t warps_per_block;
};

// One module per (layer shape, flags). The (vector width, block size) choice is
// constexpr per (desired width, 32B alignment); run() picks one by input size and alignment.
template <
    typename T,
    int64_t kHeadDim,
    uint32_t kNumHeads,
    bool kHasNorm,
    bool kHasWeight,
    bool kHasGate,
    bool kSingleGroup,
    bool kUsePDL>
struct NormGateKernel {
  static_assert(kHasNorm || !kHasWeight, "a weight only scales the normed row");
  static constexpr int64_t kWideBytes = 32;
  static constexpr uint32_t kMaxGridY = 65535;

  // Rows (heads) per block for a config, or 0 when it cannot run.
  static constexpr uint32_t rows_per_block(NormGateConfig cfg, bool vec32_ok) {
    if (cfg.vec_bytes > static_cast<int32_t>(device::kMaxVecBytes)) return 0;
    if (cfg.vec_bytes > 16 && !vec32_ok) return 0;
    // The kernel loads packed pairs (packed_t<T>).
    const int64_t elems_per_thread = cfg.vec_bytes / static_cast<int64_t>(sizeof(packed_t<T>)) * 2;
    if (elems_per_thread == 0 || kHeadDim % elems_per_thread != 0) return 0;
    const int64_t threads_per_row = kHeadDim / elems_per_thread;
    if (threads_per_row > device::kWarpThreads) return 0;
    return static_cast<uint32_t>(device::kWarpThreads / threads_per_row) * cfg.warps_per_block;
  }

  // The largest-vector, near-target block that divides the head axis evenly.
  static constexpr NormGateConfig select_config(int32_t desired_vec, bool vec32_ok) {
    constexpr int32_t kVecBytes[] = {4, 8, 16, 32};
    constexpr int32_t kWarps[] = {1, 2, 4, 8};
    // 32B saturates bandwidth with half the threads; the others want a full block.
    const uint32_t target_threads = desired_vec == 32 ? 128 : 256;
    NormGateConfig best{0, 0}, fallback{0, 0};
    uint64_t best_key = 0;
    for (const auto vec_bytes : kVecBytes) {
      for (const auto warps : kWarps) {
        const NormGateConfig cfg{vec_bytes, warps};
        const uint32_t rows = rows_per_block(cfg, vec32_ok);
        if (rows == 0) continue;
        if (fallback.vec_bytes == 0) fallback = cfg;
        if (kNumHeads % rows != 0) continue;
        const uint32_t threads = warps * device::kWarpThreads;
        const uint32_t distance = threads > target_threads ? threads - target_threads : target_threads - threads;
        const uint64_t key = (uint64_t{threads >= 128} << 40) | (uint64_t{vec_bytes == desired_vec} << 32) |
                             (uint64_t{1024 - std::min<uint32_t>(distance, 1023)} << 8) |
                             static_cast<uint64_t>(vec_bytes);
        if (key > best_key) {
          best = cfg;
          best_key = key;
        }
      }
    }
    // No block divides the head axis: fall back to anything that runs.
    return best.vec_bytes != 0 ? best : fallback;
  }

  // Indexed by [desired vector width 8 / 16 / 32][32B-aligned].
  static constexpr NormGateConfig kChoices[3][2] = {
      {select_config(8, false), select_config(8, true)},
      {select_config(16, false), select_config(16, true)},
      {select_config(32, false), select_config(32, true)},
  };
  static_assert(kChoices[0][0].vec_bytes != 0, "no norm_gate config covers this head_dim");

  template <NormGateConfig kCfg, bool kExact>
  static void launch(const NormGateParams& params, DLDevice device) {
    using Trait = NormGateTrait<T, kHeadDim, kCfg.vec_bytes>;
    constexpr uint32_t kRowsPerBlock = Trait::kRowsPerWarp * kCfg.warps_per_block;
    constexpr auto kernel = norm_gate_kernel<
        T,
        kHeadDim,
        kCfg.vec_bytes,
        kCfg.warps_per_block,
        kHasNorm,
        kHasWeight,
        kHasGate,
        kSingleGroup,
        kExact,
        kUsePDL>;
    // gridDim.y is capped by the hardware limit; the non-exact kernel strides over tokens.
    const uint32_t grid_x = host::div_ceil(kNumHeads, kRowsPerBlock);
    const uint32_t grid_y = std::min(params.num_tokens, kMaxGridY);
    host::LaunchKernel(dim3(grid_x, grid_y), kCfg.warps_per_block * device::kWarpThreads, device)
        .enable_pdl(kUsePDL)(kernel, params);
  }

  template <int kDesired, bool kVec32>
  static void launch_choice(const NormGateParams& params, DLDevice device) {
    constexpr auto kCfg = kChoices[kDesired][kVec32];
    // One token per block and a grid that never over-covers the head axis: drops
    // the grid-stride backedge, the safe_head clamp and the store predicate.
    if constexpr (kNumHeads % rows_per_block(kCfg, kVec32) == 0) {
      if (params.num_tokens <= kMaxGridY) return launch<kCfg, true>(params, device);
    }
    launch<kCfg, false>(params, device);
  }

  template <int kDesired>
  static void dispatch_alignment(const NormGateParams& params, DLDevice device, bool vec32_ok) {
    if (vec32_ok) return launch_choice<kDesired, true>(params, device);
    launch_choice<kDesired, false>(params, device);
  }

  static void
  run(const tvm::ffi::TensorView x,                           // [num_tokens * num_heads, head_dim]
      const tvm::ffi::Optional<tvm::ffi::TensorView> gate,    // [num_tokens, num_groups, group_width]
      const tvm::ffi::Optional<tvm::ffi::TensorView> weight,  // [head_dim]
      const tvm::ffi::TensorView y,                           // [num_tokens * num_heads, head_dim]
      double eps) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({-1, kHeadDim})  //
        .with_dtype<T>()
        .with_device(device)
        .verify(x)
        .verify(y);
    CHECK_HOST(weight.has_value() == kHasWeight);
    if constexpr (kHasWeight) {
      TensorMatcher({kHeadDim})  //
          .with_dtype<T>()
          .with_device(device)
          .verify(weight.value());
    }
    const auto num_rows = static_cast<uint64_t>(x.size(0));
    CHECK_HOST(num_rows % kNumHeads == 0);
    const auto num_tokens = static_cast<uint32_t>(num_rows / kNumHeads);

    auto params = NormGateParams{
        .x = x.data_ptr(),
        .gate = nullptr,
        .weight = kHasWeight ? weight.value().data_ptr() : nullptr,
        .y = y.data_ptr(),
        .gate_stride_token = 0,
        .gate_stride_group = 0,
        .num_tokens = num_tokens,
        .num_heads = kNumHeads,
        .heads_per_group = kNumHeads,
        .eps = static_cast<float>(eps),
    };

    // Every address the kernel vectorizes over, OR-ed: its lowest set bit is the alignment.
    uint64_t address_bits = kHeadDim * sizeof(T) | reinterpret_cast<uintptr_t>(params.x) |
                            reinterpret_cast<uintptr_t>(params.y) | reinterpret_cast<uintptr_t>(params.weight);

    CHECK_HOST(gate.has_value() == kHasGate);
    if constexpr (kHasGate) {
      const auto& g = gate.value();
      TensorMatcher({num_tokens, kSingleGroup ? 1 : -1, -1})  //
          .with_strides({-1, -1, 1})
          .with_dtype<T>()
          .with_device(device)
          .verify(g);
      const auto num_groups = g.size(1);
      const auto group_width = g.size(2);
      CHECK_HOST(group_width % kHeadDim == 0 && num_groups * group_width == kNumHeads * kHeadDim);
      params.gate = g.data_ptr();
      params.gate_stride_token = g.stride(0);
      params.gate_stride_group = g.stride(1);
      params.heads_per_group = static_cast<uint32_t>(group_width / kHeadDim);
      address_bits |= reinterpret_cast<uintptr_t>(params.gate) | params.gate_stride_token * sizeof(T) |
                      params.gate_stride_group * sizeof(T);
    }

    if (num_tokens == 0) return;

    // The vector width follows the thread count the whole input needs:
    // 8B up to num_sm * 256 threads, 16B up to num_sm * 2048, then 32B.
    const auto dl_device = device.unwrap();
    const uint64_t num_sm = runtime::get_sm_count(dl_device.device_id);
    const uint64_t num_bytes = uint64_t{num_tokens} * kNumHeads * kHeadDim * sizeof(T);
    const int desired = num_bytes / 8 <= num_sm * 256 ? 0 : num_bytes / 16 <= num_sm * 2048 ? 1 : 2;
    const bool vec32_ok = address_bits % kWideBytes == 0;
    // The narrow choices load at most 16B; check that they are aligned to their width.
    CHECK_HOST(address_bits % kChoices[desired][vec32_ok].vec_bytes == 0)
        << "norm_gate: inputs are not aligned to " << kChoices[desired][vec32_ok].vec_bytes << " bytes";
    if (desired == 0) return dispatch_alignment<0>(params, dl_device, vec32_ok);
    if (desired == 1) return dispatch_alignment<1>(params, dl_device, vec32_ok);
    dispatch_alignment<2>(params, dl_device, vec32_ok);
  }
};

}  // namespace sglang
