#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For dtype_trait, bf16_t, fp32_t, cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace {

// ----------------------------------------------------------------
// SiTU (SoftCap-GLU) activation:
//   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
//   up_out   = linear_beta * tanh(up / linear_beta)
//   output   = gate_out * up_out
//
// Input: bf16 tensor [N, 2*D] (gate = [:, :D], up = [:, D:])
// Output: bf16 tensor [N, D]
// ----------------------------------------------------------------

struct SituAndMulParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  float beta;
  float inv_beta;
  float linear_beta;
  float inv_linear_beta;
  uint32_t hidden_dim;  // D (output width, half of input last dim)
  uint32_t num_tokens;
  uint32_t stride_in_vecs;  // input row stride in vector units (2*D/vec if dense)
};

template <typename T, bool kHasLinearBeta, bool kUsePDL>
__global__ void situ_and_mul_kernel(const __grid_constant__ SituAndMulParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;

  const auto num_vecs = params.hidden_dim / kVecSize;  // per token
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / num_vecs;

  if (token_id >= params.num_tokens) return;

  const auto offset = tid % num_vecs;
  // Input rows may be strided (e.g. a slice of a wider fused-GEMM output);
  // within a row: gate = [0..D-1], up = [D..2D-1].
  const auto input_offset = static_cast<uint64_t>(token_id) * params.stride_in_vecs + offset;
  const auto output_offset = tid;

  PDLWaitPrimary<kUsePDL>();

  const auto gate = load_as<vec_t>(params.input, input_offset);
  const auto up = load_as<vec_t>(params.input, input_offset + num_vecs);

  PDLTriggerSecondary<kUsePDL>();

  const float beta = params.beta;
  const float inv_beta = params.inv_beta;
  const float linear_beta = params.linear_beta;
  const float inv_linear_beta = params.inv_linear_beta;

  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const float g = cast<fp32_t>(gate[i]);
    const float u = cast<fp32_t>(up[i]);

    // gate_out = beta * tanh(g / beta) * sigmoid(g)
    const float gate_out = beta * tanhf(g * inv_beta) * (1.0f / (1.0f + __expf(-g)));

    // up_out = linear_beta * tanh(u / linear_beta) if has_linear_beta, else u
    float up_out;
    if constexpr (kHasLinearBeta) {
      up_out = linear_beta * tanhf(u * inv_linear_beta);
    } else {
      up_out = u;
    }

    out[i] = cast<T>(gate_out * up_out);
  }

  store_as<vec_t>(params.out, out, output_offset);
}

// ----------------------------------------------------------------
// Host launcher
// ----------------------------------------------------------------

template <typename T, bool kUsePDL>
struct SituAndMulKernel {
  static constexpr auto kVecSize = device::kMaxVecBytes / sizeof(T);
  static constexpr auto kBlockSize = 256u;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView out,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
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
        .with_strides({-1, 1})
        .verify(input);

    const auto hidden_size = static_cast<uint32_t>(D_out.unwrap());
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto device = device_.unwrap();

    if (num_tokens == 0) return;
    RuntimeCheck(hidden_size * 2 == D_in.unwrap(), "invalid activation dimension: D_out * 2 != D_in");
    RuntimeCheck(hidden_size % kVecSize == 0, "hidden size must be divisible by vector size");
    RuntimeCheck(input.stride(0) % kVecSize == 0, "input row stride must be divisible by vector size");

    const auto num_total_items = num_tokens * (hidden_size / kVecSize);
    RuntimeCheck(num_total_items <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");

    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_total_items), kBlockSize);
    const float beta_f = static_cast<float>(beta);
    const float linear_beta_f = static_cast<float>(linear_beta);

    const auto params = SituAndMulParams{
        .input = input.data_ptr(),
        .out = out.data_ptr(),
        .beta = beta_f,
        .inv_beta = 1.0f / beta_f,
        .linear_beta = linear_beta_f,
        .inv_linear_beta = linear_beta_f != 0.0f ? 1.0f / linear_beta_f : 0.0f,
        .hidden_dim = hidden_size,
        .num_tokens = num_tokens,
        .stride_in_vecs = static_cast<uint32_t>(input.stride(0) / kVecSize),
    };

    if (has_linear_beta) {
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(situ_and_mul_kernel<T, true, kUsePDL>, params);
    } else {
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(situ_and_mul_kernel<T, false, kUsePDL>, params);
    }
  }
};

}  // namespace
