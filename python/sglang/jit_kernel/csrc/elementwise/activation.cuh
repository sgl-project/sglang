#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

enum class ActivationKind : uint32_t {
  kSiLU,
  kGELU,
  kGELUTanh,
};

template <typename T, ActivationKind kAct>
SGL_DEVICE T apply_activation(T x) {
  const float x_f32 = device::cast<fp32_t>(x);
  float y_f32 = 0.0f;

  if constexpr (kAct == ActivationKind::kSiLU) {
    y_f32 = x_f32 / (1.0f + expf(-x_f32));
  } else if constexpr (kAct == ActivationKind::kGELU) {
    constexpr auto kSqrt1Over2 = 0.7071067811865475f;
    y_f32 = x_f32 * (0.5f * (1.0f + erff(x_f32 * kSqrt1Over2)));
  } else if constexpr (kAct == ActivationKind::kGELUTanh) {
    constexpr auto kGeluTanhAlpha = 0.044715f;
    constexpr auto kGeluTanhBeta = 0.7978845608028654f;
    const float x_cube = x_f32 * x_f32 * x_f32;
    const float cdf = 0.5f * (1.0f + tanhf(kGeluTanhBeta * (x_f32 + kGeluTanhAlpha * x_cube)));
    y_f32 = x_f32 * cdf;
  } else {
    static_assert(host::dependent_false_v<T>, "unsupported activation kind");
  }

  return device::cast<T>(y_f32);
}

struct ActivationParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  uint32_t hidden_dim;
  uint32_t num_tokens;
};

template <typename T, ActivationKind kAct, bool kUsePDL>
__global__ void act_and_mul_kernel(const __grid_constant__ ActivationParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
  const auto num_vecs = params.hidden_dim / kVecSize;  // per token
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / num_vecs;

  if (token_id >= params.num_tokens) return;
  const auto offset = tid % num_vecs;
  const auto input_offset = token_id * (num_vecs * 2) + offset;
  const auto output_offset = tid;
  PDLWaitPrimary<kUsePDL>();
  const auto gate = device::load_as<vec_t>(params.input, input_offset);
  const auto up = device::load_as<vec_t>(params.input, input_offset + num_vecs);
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    out[i] = apply_activation<T, kAct>(gate[i]) * up[i];
  }
  device::store_as<vec_t>(params.out, out, output_offset);
  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct ActivationKernel {
  static constexpr auto kVecSize = device::kMaxVecBytes / sizeof(T);
  static constexpr auto kBlockSize = 256u;

  template <ActivationKind kAct>
  static constexpr auto activation_kernel = act_and_mul_kernel<T, kAct, kUsePDL>;

  static_assert(device::kMaxVecBytes % sizeof(T) == 0, "unsupported data type");
  static void run_activation(const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D_in = SymbolicSize{"input_width"};
    auto D_out = SymbolicSize{"output_width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D_out})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(out);
    TensorMatcher({N, D_in})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(input);

    const auto hidden_size = D_out.unwrap();
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto device = device_.unwrap();
    RuntimeCheck(hidden_size * 2 == D_in.unwrap(), "invalid activation dimension");
    RuntimeCheck(hidden_size % kVecSize == 0, "hidden size must be divisible by vector size");
    const auto kernel = [&]() -> decltype(activation_kernel<ActivationKind::kSiLU>) {
      if (type == "silu") {
        return activation_kernel<ActivationKind::kSiLU>;
      } else if (type == "gelu") {
        return activation_kernel<ActivationKind::kGELU>;
      } else if (type == "gelu_tanh") {
        return activation_kernel<ActivationKind::kGELUTanh>;
      } else {
        Panic("unsupported activation type: ", type);
      }
      return nullptr;
    }();
    // only get once to avoid overhead
    const auto num_total_items = num_tokens * (hidden_size / kVecSize);
    RuntimeCheck(num_total_items <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");
    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_total_items), kBlockSize);
    const auto params = ActivationParams{
        .input = input.data_ptr(),
        .out = out.data_ptr(),
        .hidden_dim = hidden_size,
        .num_tokens = num_tokens,
    };
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
