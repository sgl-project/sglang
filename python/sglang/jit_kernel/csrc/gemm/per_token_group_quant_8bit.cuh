#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

// Optimized per-token-group quant to FP8-e4m3 (or int8), with optional
// column-major UE8M0 (int32-packed) scale. Memory-bound rewrite of the AOT
// `per_token_group_quant_8bit` (sgl-kernel): the previous JIT clone read the
// input TWICE and used 16 threads/group with only group/8 active. This version:
//   * loads each group once into registers (single 128-bit load per thread),
//   * uses exactly kGroupSize/kVec threads per group (no idle lanes),
//   * sub-warp shuffle-reduces the absmax over those lanes (no shared memory),
//   * launches warp-aligned, ~256-thread blocks for high occupancy / latency
//     hiding (the AOT kernel's 1-warp blocks left HBM ~80% idle at prefill).
// The UE8M0 path reuses the dsv4 cast_to_ue8m0/inv_scale_ue8m0 primitives and is
// byte-identical to `sgl_per_token_group_quant_8bit_v2` (both ceil-round the
// scale and store the biased exponent byte). ROCm portability comes from the
// portable `warp::reduce_max<kThreadsPerGroup>` used directly in the kernel
// (gfx942-safe; no separate GroupReduceMax helper needed).

template <bool kUE8M0>
using scale_packed_t_t = std::conditional_t<kUE8M0, uint32_t, float>;
template <bool kUE8M0>
using scale_element_t_t = std::conditional_t<kUE8M0, uint8_t, float>;

struct PerTokenGroupQuantParams {
  const void* __restrict__ input;
  void* __restrict__ output_q;
  void* __restrict__ output_s;
  int64_t num_groups;      // total groups = num_tokens * num_groups_per_row
  int num_groups_per_row;  // hidden / group_size
  int groups_per_block;    // groups handled by one CTA
  int scale_stride;        // output_s.stride(1), in scale_packed_t elements
  float eps;
  float min_8bit;
  float max_8bit;
};

// kGroupSize columns per group; kThreadsPerGroup threads cover one group, each
// issuing kNumVec coalesced 128-bit loads. Lane loads are interleaved
// (chunk v at element (v*kThreadsPerGroup + lane)*kVec) so consecutive lanes hit
// consecutive 16B addresses -> fully coalesced even for kNumVec > 1.
template <typename T, typename DST, int64_t kGroupSize, int kThreadsPerGroup, bool kColMajor, bool kUE8M0, bool kUsePDL>
__global__ __launch_bounds__(256, 8) void per_token_group_quant_8bit_kernel(const PerTokenGroupQuantParams params) {
  using namespace device;
  namespace math = device::math;

  constexpr uint32_t kVec = 16u / sizeof(T);  // 8 for bf16/fp16
  constexpr uint32_t kElemsPerThread = kGroupSize / kThreadsPerGroup;
  constexpr uint32_t kNumVec = kElemsPerThread / kVec;  // 128-bit loads per thread
  static_assert(kGroupSize % (kThreadsPerGroup * kVec) == 0, "bad tiling");
  static_assert(
      kThreadsPerGroup >= 1 && kThreadsPerGroup <= 32 && (kThreadsPerGroup & (kThreadsPerGroup - 1)) == 0,
      "threads-per-group must be a pow2 <= 32");

  using InVec = AlignedVector<T, kVec>;
  using scale_packed_t = scale_packed_t_t<kUE8M0>;
  using scale_element_t = scale_element_t_t<kUE8M0>;

  const int local_group = threadIdx.x / kThreadsPerGroup;
  const int lane = threadIdx.x % kThreadsPerGroup;
  const int64_t global_group = static_cast<int64_t>(blockIdx.x) * params.groups_per_block + local_group;

  PDLWaitPrimary<kUsePDL>();
  if (global_group >= params.num_groups) {
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const T* gin = static_cast<const T*>(params.input) + global_group * kGroupSize;
  DST* gout = static_cast<DST*>(params.output_q) + global_group * kGroupSize;

  // Load kNumVec interleaved 128-bit chunks into registers.
  float vals[kElemsPerThread];
  float local_absmax = params.eps;
#pragma unroll
  for (uint32_t v = 0; v < kNumVec; ++v) {
    InVec in_vec;
    in_vec.load(gin + (v * kThreadsPerGroup + lane) * kVec, 0);
#pragma unroll
    for (uint32_t j = 0; j < kVec; ++j) {
      const float val = static_cast<float>(in_vec[j]);
      vals[v * kVec + j] = val;
      local_absmax = math::max(local_absmax, math::abs(val));
    }
  }
  if constexpr (kThreadsPerGroup > 1) {
    local_absmax = warp::reduce_max<kThreadsPerGroup>(local_absmax);
  }

  // Scale (byte-identical to sgl_per_token_group_quant_8bit_v2).
  const float kMaxInv = 1.0f / params.max_8bit;
  float inv_scale;  // multiply input by this to quantize
  scale_element_t scale_store;
  if constexpr (kUE8M0) {
    const int32_t exp = cast_to_ue8m0(local_absmax * kMaxInv);
    inv_scale = inv_scale_ue8m0(exp);
    scale_store = static_cast<uint8_t>(exp);
  } else {
    const float scale_inv = local_absmax * kMaxInv;  // stored scale
    inv_scale = params.max_8bit / local_absmax;      // quant multiplier
    scale_store = scale_inv;
  }

  // Quantize from registers and store kNumVec interleaved chunks.
#pragma unroll
  for (uint32_t v = 0; v < kNumVec; ++v) {
    DST* o = gout + (v * kThreadsPerGroup + lane) * kVec;
    if constexpr (std::is_same_v<DST, fp8_e4m3_t>) {
      AlignedVector<fp8x2_e4m3_t, kVec / 2> out_vec;
#pragma unroll
      for (uint32_t j = 0; j < kVec / 2; ++j) {
        out_vec[j] = pack_fp8(vals[v * kVec + 2 * j] * inv_scale, vals[v * kVec + 2 * j + 1] * inv_scale);
      }
      out_vec.store(o, 0);
    } else {
      AlignedVector<DST, kVec> out_vec;
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j) {
        const float q = math::min(math::max(vals[v * kVec + j] * inv_scale, params.min_8bit), params.max_8bit);
        out_vec[j] = static_cast<DST>(q);
      }
      out_vec.store(o, 0);
    }
  }

  // One scale write per group (the leading lane).
  if (lane == 0) {
    scale_element_t* scale_out;
    if constexpr (kColMajor) {
      constexpr int kPack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
      const int row = static_cast<int>(global_group / params.num_groups_per_row);    // token
      const int col_u = static_cast<int>(global_group % params.num_groups_per_row);  // group in row
      const int col = col_u / kPack;
      const int pack = col_u % kPack;
      scale_out = reinterpret_cast<scale_element_t*>(params.output_s) +
                  (static_cast<int64_t>(col) * params.scale_stride * kPack + static_cast<int64_t>(row) * kPack + pack);
    } else {
      static_assert(!kUE8M0, "non-column-major UE8M0 is unsupported");
      scale_out = static_cast<scale_element_t*>(params.output_s) + global_group;
    }
    *scale_out = scale_store;
  }

  PDLTriggerSecondary<kUsePDL>();
}

// Threads cooperating on one group. Heuristic: ~8 elems/thread for small groups
// (one 128-bit load) and 16 elems/thread for group=128 (two loads), capped at 8
// threads/group to keep the sub-warp reduction (and register pressure) small.
template <int64_t kGroupSize, typename T>
constexpr int threads_per_group() {
  constexpr int kVec = 16 / sizeof(T);                  // 8 for bf16/fp16
  int tpg = static_cast<int>(kGroupSize) / (2 * kVec);  // ~16 elems/thread (2 vecs)
  if (tpg > 8) tpg = 8;
  if (tpg < 1) tpg = 1;
  return tpg;
}

constexpr int kBlockThreads = 256;

template <int64_t kGroupSize, typename T>
inline int pick_groups_per_block() {
  int gpb = kBlockThreads / threads_per_group<kGroupSize, T>();
  if (gpb < 1) gpb = 1;
  return gpb;
}

template <typename T, typename DST, int64_t kGroupSize, bool kColMajor, bool kUE8M0, bool kUsePDL>
void launch_quant(const PerTokenGroupQuantParams& base, int64_t num_groups, int groups_per_block, DLDevice device) {
  using namespace host;
  constexpr int kThreadsPerGroup = threads_per_group<kGroupSize, T>();
  const int num_threads = groups_per_block * kThreadsPerGroup;
  const int64_t num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  PerTokenGroupQuantParams params = base;
  params.groups_per_block = groups_per_block;
  constexpr auto kernel =
      per_token_group_quant_8bit_kernel<T, DST, kGroupSize, kThreadsPerGroup, kColMajor, kUE8M0, kUsePDL>;
  LaunchKernel(static_cast<uint32_t>(num_blocks), static_cast<uint32_t>(num_threads), device)
      .enable_pdl(kUsePDL)(kernel, params);
}

template <typename DType, typename OutType, int64_t kGroupSize, bool kUsePDL>
void per_token_group_quant_8bit(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output_q,
    tvm::ffi::TensorView output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0) {
  using namespace host;
  static_assert(
      kGroupSize == 16 || kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128,
      "group_size template arg must be 16/32/64/128");

  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"hidden_dim"};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, K}).with_dtype<DType>().with_device(device).verify(input);
  TensorMatcher({M, K}).with_dtype<OutType>().with_device(device).verify(output_q);

  RuntimeCheck(group_size == kGroupSize, "group_size does not match compiled template");

  const int64_t num_tokens = M.unwrap();
  const int64_t hidden_dim = K.unwrap();
  const int64_t num_groups_per_row = hidden_dim / kGroupSize;
  const int64_t num_groups = num_tokens * num_groups_per_row;
  if (num_groups == 0) return;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_stride = static_cast<int>(output_s.stride(1));

  PerTokenGroupQuantParams base{};
  base.input = input.data_ptr();
  base.output_q = output_q.data_ptr();
  base.output_s = output_s.data_ptr();
  base.num_groups = num_groups;
  base.num_groups_per_row = static_cast<int>(num_groups_per_row);
  base.scale_stride = scale_stride;
  base.eps = static_cast<float>(eps);
  base.min_8bit = static_cast<float>(min_8bit);
  base.max_8bit = static_cast<float>(max_8bit);

  const auto dev = input.device();
  const int gpb = pick_groups_per_block<kGroupSize, DType>();

  // Runtime selection between compile-time-instantiated scale-layout variants
  // (the group size itself is a template arg, supplied from Python).
  if (is_column_major) {
    if (scale_ue8m0) {
      launch_quant<DType, OutType, kGroupSize, true, true, kUsePDL>(base, num_groups, gpb, dev);
    } else {
      launch_quant<DType, OutType, kGroupSize, true, false, kUsePDL>(base, num_groups, gpb, dev);
    }
  } else {
    RuntimeCheck(!scale_ue8m0, "row-major UE8M0 unsupported");
    launch_quant<DType, OutType, kGroupSize, false, false, kUsePDL>(base, num_groups, gpb, dev);
  }
}

}  // namespace
