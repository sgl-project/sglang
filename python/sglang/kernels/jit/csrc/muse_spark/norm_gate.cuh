/**
 * \brief Fused per-head RMS norm + sigmoid gate for the MuseSpark v1.2 attention output.
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

/// \brief sigmoid on an approximate reciprocal (MUFU.RCP). `1.0f / y` lowers
/// to div.rn.f32, whose out-of-line slow path costs a CALL per element; the
/// correctly-rounded `__frcp_rn` measured slower still. Error stays within the
/// 1 bf16 ulp the exp approximation already carries.
SGL_DEVICE float sigmoid_gate(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

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

/// \brief One kVecBytes vector per thread; a head_dim row spans kThreadsPerRow lanes.
template <typename T, int64_t kHeadDim, int32_t kVecBytes>
struct NormGateTrait {
  using Packed = packed_t<T>;
  static constexpr int64_t kPack = kVecBytes / static_cast<int64_t>(sizeof(Packed));
  static constexpr int64_t kElemsPerThread = kPack * 2;
  static constexpr bool kValid = kVecBytes <= static_cast<int32_t>(device::kMaxVecBytes) &&
                                 kHeadDim % kElemsPerThread == 0 && kHeadDim / kElemsPerThread <= 32;
  // Keep the dependent types well-formed even for a rejected (vector, head_dim) pair.
  static constexpr int64_t kThreadsPerRow = kValid ? kHeadDim / kElemsPerThread : 1;
  static constexpr int64_t kRowsPerWarp = device::kWarpThreads / kThreadsPerRow;
  using Storage = device::AlignedVector<Packed, kValid ? kPack : 1>;
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
    xv.load(x_row + token * row_pitch, lane);

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
      gv.load(gate_row + token * params.gate_stride_token, lane);
    }
    // ---- PDL: Trigger next kernel here to increase overlap
    PDLTriggerSecondary<kUsePDL>();
    float rstd = 0.0f;
    if constexpr (kHasNorm) {
      sum_of_squares = warp::reduce_sum<kThreadsPerRow>(sum_of_squares);
      rstd = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

      if constexpr (kHasWeight) {
        wv.load(params.weight, lane);
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
        acc.x *= sigmoid_gate(g.x);
        acc.y *= sigmoid_gate(g.y);
      }
      yv[p] = cast<Packed>(acc);
    }

    if (kExact || head < num_heads) {
      yv.store(y_row + token * row_pitch, lane);
    }
  }
}

#define SGL_NORM_GATE_DISPATCH_BOOL(cond, kFlag, ...) \
  [&] {                                               \
    if (cond) {                                       \
      constexpr bool kFlag = true;                    \
      return (__VA_ARGS__)();                         \
    } else {                                          \
      constexpr bool kFlag = false;                   \
      return (__VA_ARGS__)();                         \
    }                                                 \
  }()

struct NormGateConfig {
  int32_t vec_bytes;
  int32_t warps_per_block;
};

template <typename T, int64_t kHeadDim, bool kUsePDL>
struct NormGateKernel {
  static constexpr NormGateConfig kConfigs[] = {
      {4, 1},
      {4, 2},
      {4, 4},
      {4, 8},
      {8, 1},
      {8, 2},
      {8, 4},
      {8, 8},
      {16, 1},
      {16, 2},
      {16, 4},
      {16, 8},
      {32, 1},
      {32, 2},
      {32, 4},
      {32, 8},
  };
  static constexpr int32_t kNumConfigs = sizeof(kConfigs) / sizeof(NormGateConfig);

  template <int32_t kConfig, bool kHasNorm, bool kHasWeight, bool kHasGate, bool kSingleGroup, bool kExact>
  static void launch(const NormGateParams& params, DLDevice device) {
    using Trait = NormGateTrait<T, kHeadDim, kConfigs[kConfig].vec_bytes>;
    if constexpr (Trait::kValid) {
      constexpr auto kCfg = kConfigs[kConfig];
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

      constexpr uint32_t kRowsPerBlock = Trait::kRowsPerWarp * kCfg.warps_per_block;
      const uint32_t grid_x = host::div_ceil(params.num_heads, kRowsPerBlock);
      // gridDim.y is capped by the hardware limit; the kernel strides over tokens.
      const uint32_t grid_y = std::min<uint32_t>(params.num_tokens, 65535u);
      host::LaunchKernel(dim3(grid_x, grid_y), kCfg.warps_per_block * device::kWarpThreads, device)
          .enable_pdl(kUsePDL)(kernel, params);
    }
  }

  template <int32_t kConfig, bool kHasNorm, bool kHasWeight, bool kHasGate, bool kSingleGroup, bool kExact>
  static void dispatch_config(const NormGateParams& params, DLDevice device, int32_t config) {
    if constexpr (kConfig < kNumConfigs) {
      if (config == kConfig) {
        return launch<kConfig, kHasNorm, kHasWeight, kHasGate, kSingleGroup, kExact>(params, device);
      }
      return dispatch_config<kConfig + 1, kHasNorm, kHasWeight, kHasGate, kSingleGroup, kExact>(params, device, config);
    }
  }

  static int64_t elems_per_thread(const NormGateConfig& cfg) {
    return cfg.vec_bytes / static_cast<int64_t>(sizeof(packed_t<T>)) * 2;
  }

  /// \brief Rows per block for a config, or 0 when it cannot run on this input.
  static uint32_t runtime_rows_per_block(const NormGateConfig& cfg, bool vec32_ok) {
    if (cfg.vec_bytes > static_cast<int32_t>(device::kMaxVecBytes)) return 0;
    if (cfg.vec_bytes > 16 && !vec32_ok) return 0;
    if (kHeadDim % elems_per_thread(cfg) != 0) return 0;
    const int64_t threads_per_row = kHeadDim / elems_per_thread(cfg);
    if (threads_per_row > device::kWarpThreads) return 0;
    return static_cast<uint32_t>(device::kWarpThreads / threads_per_row) * cfg.warps_per_block;
  }

  /// \brief Vector width from the total thread count, then the largest block
  /// that still divides the head axis evenly.
  static int32_t select_config(uint32_t num_tokens, uint32_t num_heads, bool vec32_ok, int32_t device_id) {
    static const uint32_t num_sm = host::runtime::get_sm_count(device_id);
    const uint64_t num_rows = static_cast<uint64_t>(num_tokens) * num_heads;
    const auto threads_for = [&](int32_t vec_bytes) {
      return num_rows * static_cast<uint64_t>(kHeadDim * sizeof(T)) / static_cast<uint64_t>(vec_bytes);
    };
    int32_t desired_vec = 8;
    if (threads_for(8) > static_cast<uint64_t>(num_sm) * 256) desired_vec = 16;
    if (desired_vec == 16 && threads_for(16) > static_cast<uint64_t>(num_sm) * 2048) desired_vec = 32;
    // 32B saturates bandwidth with half the threads; the others want a full block.
    const uint32_t target_threads = desired_vec == 32 ? 128 : 256;

    int32_t best = -1;
    uint64_t best_key = 0;
    for (int32_t config = 0; config < kNumConfigs; ++config) {
      const NormGateConfig& cfg = kConfigs[config];
      const uint32_t rows = runtime_rows_per_block(cfg, vec32_ok);
      if (rows == 0 || num_heads % rows != 0) continue;
      const uint32_t threads = cfg.warps_per_block * device::kWarpThreads;
      const uint32_t distance = threads > target_threads ? threads - target_threads : target_threads - threads;
      const uint64_t key = (static_cast<uint64_t>(threads >= 128) << 40) |
                           (static_cast<uint64_t>(cfg.vec_bytes == desired_vec) << 32) |
                           (static_cast<uint64_t>(1024 - std::min<uint32_t>(distance, 1023)) << 8) |
                           static_cast<uint64_t>(cfg.vec_bytes);
      if (best < 0 || key > best_key) {
        best = config;
        best_key = key;
      }
    }
    if (best < 0) {
      // No block divides the head axis; fall back to anything that runs.
      for (int32_t config = 0; config < kNumConfigs; ++config) {
        if (runtime_rows_per_block(kConfigs[config], vec32_ok) > 0) return config;
      }
    }
    return best;
  }

  static void
  run(const tvm::ffi::TensorView x,                           // [num_tokens * num_heads, head_dim]
      const tvm::ffi::Optional<tvm::ffi::TensorView> gate,    // [num_tokens, num_groups, group_width]
      const tvm::ffi::Optional<tvm::ffi::TensorView> weight,  // [head_dim]
      const tvm::ffi::TensorView y,                           // [num_tokens * num_heads, head_dim]
      int64_t num_heads,
      double eps,
      bool has_norm,
      int64_t config_override) {
    using namespace host;

    auto N = SymbolicSize{"num_rows"};
    auto D = SymbolicSize{"head_dim"};
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({D, 1}).with_dtype<T>().with_device(device).verify(x).verify(y);
    if (weight.has_value()) {
      TensorMatcher({D}).with_strides({1}).with_dtype<T>().with_device(device).verify(weight.value());
    }

    const auto num_rows = static_cast<uint64_t>(N.unwrap());
    RuntimeCheck(num_heads > 0 && num_rows % static_cast<uint64_t>(num_heads) == 0);
    const auto num_tokens = static_cast<uint32_t>(num_rows / static_cast<uint64_t>(num_heads));

    auto params = NormGateParams{
        .x = x.data_ptr(),
        .gate = nullptr,
        .weight = weight.has_value() ? weight.value().data_ptr() : nullptr,
        .y = y.data_ptr(),
        .gate_stride_token = 0,
        .gate_stride_group = 0,
        .num_tokens = num_tokens,
        .num_heads = static_cast<uint32_t>(num_heads),
        .heads_per_group = static_cast<uint32_t>(num_heads),
        .eps = static_cast<float>(eps),
    };

    constexpr int64_t kWideBytes = 32;
    const auto aligned = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % kWideBytes == 0; };
    const auto aligned_elems = [](int64_t elems) { return elems * sizeof(T) % kWideBytes == 0; };
    bool vec32_ok = kHeadDim * sizeof(T) % kWideBytes == 0 && aligned(params.x) && aligned(params.y);

    bool single_group = true;
    if (gate.has_value()) {
      auto Tk = SymbolicSize{"num_tokens"};
      auto G = SymbolicSize{"num_groups"};
      auto W = SymbolicSize{"group_width"};
      auto S0 = SymbolicSize{"gate_stride_token"};
      auto S1 = SymbolicSize{"gate_stride_group"};
      Tk.set_value(num_tokens);
      TensorMatcher({Tk, G, W}).with_strides({S0, S1, 1}).with_dtype<T>().with_device(device).verify(gate.value());
      const auto num_groups = static_cast<int64_t>(G.unwrap());
      const auto group_width = static_cast<int64_t>(W.unwrap());
      RuntimeCheck(group_width % kHeadDim == 0 && num_groups * group_width == num_heads * kHeadDim);
      params.gate = gate.value().data_ptr();
      params.gate_stride_token = S0.unwrap();
      params.gate_stride_group = S1.unwrap();
      params.heads_per_group = static_cast<uint32_t>(group_width / kHeadDim);
      single_group = num_groups == 1;
      vec32_ok = vec32_ok && aligned(params.gate) && aligned_elems(params.gate_stride_token) &&
                 aligned_elems(params.gate_stride_group);
    }
    if (weight.has_value()) {
      vec32_ok = vec32_ok && aligned(params.weight);
    }

    if (num_tokens == 0) return;

    const auto dl_device = device.unwrap();
    int32_t config = select_config(num_tokens, params.num_heads, vec32_ok, dl_device.device_id);
    // dispatch_config matches on equality, so a -1 here would fall through every
    // config and leave y unwritten rather than failing.
    RuntimeCheck(config >= 0, "no norm_gate config covers this head_dim");
    if (config_override >= 0) {
      config = static_cast<int32_t>(config_override);
      RuntimeCheck(config < kNumConfigs && runtime_rows_per_block(kConfigs[config], vec32_ok) > 0);
    }
    const bool has_weight = has_norm && weight.has_value();
    const bool has_gate = gate.has_value();
    // One token per block and a grid that never over-covers the head axis: drops
    // the grid-stride backedge, the safe_head clamp and the store predicate.
    const uint32_t rows_per_block = runtime_rows_per_block(kConfigs[config], vec32_ok);
    const bool exact = num_tokens <= 65535 && params.num_heads % rows_per_block == 0;

    SGL_NORM_GATE_DISPATCH_BOOL(has_norm, kHasNorm, [&] {
      SGL_NORM_GATE_DISPATCH_BOOL(has_weight, kHasWeight, [&] {
        SGL_NORM_GATE_DISPATCH_BOOL(has_gate, kHasGate, [&] {
          SGL_NORM_GATE_DISPATCH_BOOL(single_group, kSingleGroup, [&] {
            SGL_NORM_GATE_DISPATCH_BOOL(exact, kExact, [&] {
              dispatch_config<0, kHasNorm, kHasWeight, kHasGate, kSingleGroup, kExact>(params, dl_device, config);
            });
          });
        });
      });
    });
  }
};

}  // namespace sglang
