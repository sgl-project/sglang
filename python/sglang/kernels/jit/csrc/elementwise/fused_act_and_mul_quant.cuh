#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

// ============================================================================
// Activation function definitions (same as activation.cuh).
// Duplicated here to keep each JIT module independently compilable.
// ============================================================================

enum class ActivationKind : uint32_t {
  kSiLU,
  kGELU,
  kGELUTanh,
};

template <ActivationKind kAct>
SGL_DEVICE float apply_activation_f32(float x) {
  if constexpr (kAct == ActivationKind::kSiLU) {
    return x / (1.0f + expf(-x));
  } else if constexpr (kAct == ActivationKind::kGELU) {
    constexpr float kSqrt1Over2 = 0.7071067811865475f;
    return x * (0.5f * (1.0f + erff(x * kSqrt1Over2)));
  } else if constexpr (kAct == ActivationKind::kGELUTanh) {
    constexpr float kAlpha = 0.044715f;
    constexpr float kBeta = 0.7978845608028654f;
    const float cdf = 0.5f * (1.0f + tanhf(kBeta * (x + kAlpha * x * x * x)));
    return x * cdf;
  } else {
    static_assert(host::dependent_false_v<decltype(kAct)>, "unsupported activation kind");
    return 0.0f;
  }
}

// ============================================================================
// Fused activation + per-token-group FP8 quantization kernel
// ============================================================================

struct ActivationQuantParams {
  const void* __restrict__ input;        // [num_tokens, hidden_dim * 2]
  void* __restrict__ output_q;           // [num_tokens, hidden_dim], fp8_e4m3
  void* __restrict__ output_scale;       // [num_tokens, hidden_dim / group_size], float32
  uint32_t hidden_dim;                   // output hidden dimension (= input_width / 2)
  uint32_t num_tokens;
  uint32_t group_size;                   // quantization group size (128)
  const int32_t* __restrict__ expert_ids;
  uint32_t expert_step;
};

/// Fused kernel: act(gate) * up  →  per-group FP8 quantize  →  write (output_q, output_scale).
///
/// Non-persistent grid: each thread handles exactly one vec (8 elements),
/// grid = ceil(num_tokens * num_vecs_per_token / blockSize).
/// 16 adjacent threads form a quantization group (128 elements) and cooperate on absmax reduce.
template <typename T, ActivationKind kAct, bool kUsePDL, bool kFilterExpert, bool kScaleUE8M0>
__global__ void act_and_mul_quant_kernel(const __grid_constant__ ActivationQuantParams params) {
  using namespace device;
  using fp8_t = fp8_e4m3_t;

  constexpr uint32_t kVecSize = kMaxVecBytes / sizeof(T);  // 8 for bf16/fp16
  using vec_t = AlignedVector<T, kVecSize>;

  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecSize;  // 16
  constexpr float kFP8E4M3Max = 448.0f;
  constexpr float kEps = 1e-8f;

  static_assert(kThreadsPerGroup == 16, "scale store assumes 16 threads per group");

  const uint32_t hidden_dim = params.hidden_dim;
  const uint32_t num_tokens = params.num_tokens;
  const uint32_t num_vecs = hidden_dim / kVecSize;
  const uint32_t num_groups = hidden_dim / kGroupSize;

  // Global thread id → (token_id, vec_offset_within_token)
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t token_id = tid / num_vecs;
  if (token_id >= num_tokens) return;

  if constexpr (kFilterExpert) {
    if (params.expert_ids[token_id / params.expert_step] == -1) return;
  }

  const uint32_t vec_offset = tid % num_vecs;
  const uint32_t icol = vec_offset * kVecSize;
  const uint32_t lane_id = threadIdx.x % 32;

  PDLWaitPrimary<kUsePDL>();

  // Row pointers
  fp8_t* const out_row = static_cast<fp8_t*>(params.output_q) + token_id * hidden_dim;
  float* const scale_row = static_cast<float*>(params.output_scale) + token_id * num_groups;

  // Vectorized load gate and up
  const uint32_t gate_offset = token_id * (num_vecs * 2) + vec_offset;
  const vec_t gate_vec = load_as<vec_t>(params.input, gate_offset);
  const vec_t up_vec = load_as<vec_t>(params.input, gate_offset + num_vecs);

  // Activation + truncate to T precision + local absmax
  float vals[kVecSize];
  float local_max = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    const float act_f32 = apply_activation_f32<kAct>(cast<fp32_t>(gate_vec[i])) * cast<fp32_t>(up_vec[i]);
    vals[i] = cast<fp32_t>(cast<T>(act_f32));
    local_max = fmaxf(local_max, fabsf(vals[i]));
  }

  // Half-warp reduce (16 threads = one quant group)
#pragma unroll
  for (uint32_t mask = kThreadsPerGroup / 2; mask > 0; mask >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
  }

  // Scale computation
  float scale;
  if constexpr (kScaleUE8M0) {
    const float raw = fmaxf(local_max, kEps) / kFP8E4M3Max;
    const uint32_t bits = __float_as_uint(raw);
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
    scale = __uint_as_float(static_cast<uint32_t>(exp) << 23);
  } else {
    scale = local_max / kFP8E4M3Max;
  }
  const float inv_scale = 1.0f / (scale + kEps);

  // Quantize and vectorized store FP8 (8 bytes = 1x STG.64)
  {
    const float4 f4_0 = make_float4(
        vals[0] * inv_scale, vals[1] * inv_scale,
        vals[2] * inv_scale, vals[3] * inv_scale);
    const float4 f4_1 = make_float4(
        vals[4] * inv_scale, vals[5] * inv_scale,
        vals[6] * inv_scale, vals[7] * inv_scale);
    const __nv_fp8x4_e4m3 pack0(f4_0);
    const __nv_fp8x4_e4m3 pack1(f4_1);
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t*>(&pack0);
    packed.y = *reinterpret_cast<const uint32_t*>(&pack1);
    *reinterpret_cast<uint2*>(out_row + icol) = packed;
  }

  // Store scale (one per group; lane 0 and lane 16 each own one group)
  if (lane_id == 0 || lane_id == 16) {
    scale_row[icol / kGroupSize] = scale;
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ============================================================================
// Launch infrastructure
// ============================================================================

template <typename T, bool kUsePDL>
struct ActivationQuantKernel {
  static constexpr uint32_t kBlockSize = 256u;
  static constexpr uint32_t kElementsPerThread = device::kMaxVecBytes / sizeof(T);  // 8

  using kernel_fn_t = decltype(&act_and_mul_quant_kernel<T, ActivationKind::kSiLU, kUsePDL, false, false>);

  template <ActivationKind kAct, bool kFilterExpert, bool kScaleUE8M0>
  static constexpr kernel_fn_t quant_kernel = act_and_mul_quant_kernel<T, kAct, kUsePDL, kFilterExpert, kScaleUE8M0>;

  template <bool kFilterExpert, bool kScaleUE8M0>
  static kernel_fn_t select_kernel(const std::string& type) {
    using namespace host;
    if (type == "silu") return quant_kernel<ActivationKind::kSiLU, kFilterExpert, kScaleUE8M0>;
    if (type == "gelu") return quant_kernel<ActivationKind::kGELU, kFilterExpert, kScaleUE8M0>;
    if (type == "gelu_tanh") return quant_kernel<ActivationKind::kGELUTanh, kFilterExpert, kScaleUE8M0>;
    Panic("unsupported activation type: ", type);
    return nullptr;
  }

  static void launch(
      const tvm::ffi::TensorView& input,
      const tvm::ffi::TensorView& output_q,
      const tvm::ffi::TensorView& output_scale,
      const std::string& type,
      int64_t group_size,
      bool scale_ue8m0,
      const int32_t* expert_ids,
      uint32_t expert_step) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D_in = SymbolicSize{"input_width"};
    auto D_out = SymbolicSize{"output_width"};
    auto D_scale = SymbolicSize{"scale_width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D_in}).with_dtype<T>().with_device(device_).verify(input);
    TensorMatcher({N, D_out}).with_device(device_).verify(output_q);
    TensorMatcher({N, D_scale}).with_device(device_).verify(output_scale);

    const uint32_t hidden_dim = static_cast<uint32_t>(D_out.unwrap());
    const uint32_t num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto device = device_.unwrap();
    if (num_tokens == 0) return;

    RuntimeCheck(hidden_dim * 2 == D_in.unwrap(), "input width must be 2 * output width");
    RuntimeCheck(group_size == 128, "only group_size=128 is supported");
    RuntimeCheck(hidden_dim % group_size == 0, "hidden_dim must be divisible by group_size");
    RuntimeCheck(D_scale.unwrap() == static_cast<int64_t>(hidden_dim / group_size),
                 "scale width must equal hidden_dim / group_size");

    // Select kernel
    kernel_fn_t kernel = nullptr;
    if (expert_ids != nullptr) {
      kernel = scale_ue8m0 ? select_kernel<true, true>(type) : select_kernel<true, false>(type);
    } else {
      kernel = scale_ue8m0 ? select_kernel<false, true>(type) : select_kernel<false, false>(type);
    }

    // Non-persistent grid: one thread per vec, grid = ceil(total_vecs / blockSize)
    const uint32_t num_vecs_per_token = hidden_dim / kElementsPerThread;
    const uint32_t total_items = num_tokens * num_vecs_per_token;
    const uint32_t num_blocks = div_ceil(total_items, kBlockSize);

    const auto params = ActivationQuantParams{
        .input = input.data_ptr(),
        .output_q = output_q.data_ptr(),
        .output_scale = output_scale.data_ptr(),
        .hidden_dim = hidden_dim,
        .num_tokens = num_tokens,
        .group_size = static_cast<uint32_t>(group_size),
        .expert_ids = expert_ids,
        .expert_step = expert_step,
    };

    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }

  static void run_activation_quant(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output_q,
      const tvm::ffi::TensorView output_scale,
      std::string type,
      int64_t group_size,
      bool scale_ue8m0) {
    launch(input, output_q, output_scale, type, group_size, scale_ue8m0,
           /*expert_ids=*/nullptr, /*expert_step=*/1);
  }

  static void run_activation_quant_filtered(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output_q,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView expert_ids,
      int64_t expert_step,
      std::string type,
      int64_t group_size,
      bool scale_ue8m0) {
    using namespace host;
    RuntimeCheck(is_type<int32_t>(expert_ids.dtype()), "expert_ids must have dtype int32");
    RuntimeCheck(expert_step >= 1, "expert_step must be positive");
    launch(input, output_q, output_scale, type, group_size, scale_ue8m0,
           static_cast<const int32_t*>(expert_ids.data_ptr()), static_cast<uint32_t>(expert_step));
  }
};

}  // namespace
