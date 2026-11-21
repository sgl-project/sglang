#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/activation.cuh>
#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

namespace sglang {

constexpr uint32_t kSiluQuantWarpSize = 32;
constexpr uint32_t kSiluQuantTokensPerCTA = 8;

template <typename T, int kVecSize, bool kFastMath>
SGL_DEVICE float compute_silu_mul_vec(
    const device::AlignedVector<T, kVecSize>& gate_vec,
    const device::AlignedVector<T, kVecSize>& up_vec,
    device::AlignedVector<T, kVecSize>& act_vec) {
  using namespace device;
  float max_value = 0.0f;
#pragma unroll
  for (int j = 0; j < kVecSize; ++j) {
    const float gate_f32 = cast<fp32_t>(gate_vec[j]);
    const float up_f32 = cast<fp32_t>(up_vec[j]);
    const T act_t = cast<T>(activation::silu<kFastMath>(gate_f32) * up_f32);
    act_vec[j] = act_t;
    max_value = math::max(max_value, math::abs(static_cast<float>(act_t)));
  }
  return max_value;
}

template <typename T, int kVecSize>
SGL_DEVICE void quantize_fp8_vec(
    const device::AlignedVector<T, kVecSize>& act_vec,
    float scale_inv,
    device::AlignedVector<fp8_e4m3_t, kVecSize>& output_vec) {
  using namespace device;
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  static_assert(kVecSize % 2 == 0, "kVecSize must be even for packed fp8x2 conversion");
  auto* out_x2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_vec[0]);
#pragma unroll
  for (int j = 0; j < kVecSize; j += 2) {
    const float2 vals =
        make_float2(static_cast<float>(act_vec[j]) * scale_inv, static_cast<float>(act_vec[j + 1]) * scale_inv);
    out_x2[j / 2] = __nv_cvt_float2_to_fp8x2(vals, __NV_SATFINITE, __NV_E4M3);
  }
#else
#pragma unroll
  for (int j = 0; j < kVecSize; ++j) {
    const float value = static_cast<float>(act_vec[j]) * scale_inv;
    output_vec[j] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
  }
#endif
}

/** \brief Single-pass register-cached warp-per-token fused SiLU+mul+per-token FP8 quant. */
template <typename T, int kVecSize, int kMaxRegVecs, bool kUsePDL, bool kFastMath>
__global__
__launch_bounds__(kSiluQuantTokensPerCTA* kSiluQuantWarpSize) void silu_and_mul_per_token_quant_fp8_warp_reg_kernel(
    const T* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    uint32_t hidden_dim,
    uint32_t num_tokens) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const uint32_t warp_id = threadIdx.x / kSiluQuantWarpSize;
  const uint32_t lane_id = threadIdx.x % kSiluQuantWarpSize;
  const uint32_t token_id = blockIdx.x * kSiluQuantTokensPerCTA + warp_id;
  if (token_id >= num_tokens) {
    return;
  }

  PDLWaitPrimary<kUsePDL>();

  const int64_t row_offset = static_cast<int64_t>(token_id) * static_cast<int64_t>(hidden_dim);
  const T* token_gate = input + row_offset * 2;
  const T* token_up = token_gate + hidden_dim;
  fp8_e4m3_t* token_output = output_q + row_offset;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  input_vec_t act_reg[kMaxRegVecs];
  float max_value = 0.0f;
#pragma unroll
  for (int r = 0; r < kMaxRegVecs; ++r) {
    const uint32_t idx = lane_id + static_cast<uint32_t>(r) * kSiluQuantWarpSize;
    if (idx < num_vecs) {
      input_vec_t gate_vec, up_vec;
      gate_vec.load(token_gate, idx);
      up_vec.load(token_up, idx);
      max_value = math::max(max_value, compute_silu_mul_vec<T, kVecSize, kFastMath>(gate_vec, up_vec, act_reg[r]));
    }
  }

  const float scale = warp::reduce_max(max_value) / math::FP8_E4M3_MAX;
  if (lane_id == 0) {
    output_s[token_id] = scale;
  }
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

#pragma unroll
  for (int r = 0; r < kMaxRegVecs; ++r) {
    const uint32_t idx = lane_id + static_cast<uint32_t>(r) * kSiluQuantWarpSize;
    if (idx < num_vecs) {
      output_vec_t output_vec;
      quantize_fp8_vec<T, kVecSize>(act_reg[r], scale_inv, output_vec);
      output_vec.store(token_output, idx);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

/** \brief Single-pass register-cached CTA-per-token fused SiLU+mul+per-token FP8 quant. */
template <typename T, int kVecSize, uint32_t kBlockSize, int kMaxRegVecs, bool kUsePDL, bool kFastMath>
__global__ __launch_bounds__(kBlockSize) void silu_and_mul_per_token_quant_fp8_cta_reg_kernel(
    const T* __restrict__ input, fp8_e4m3_t* __restrict__ output_q, float* __restrict__ output_s, uint32_t hidden_dim) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  PDLWaitPrimary<kUsePDL>();

  const uint32_t token_id = blockIdx.x;
  const int64_t row_offset = static_cast<int64_t>(token_id) * static_cast<int64_t>(hidden_dim);
  const T* token_gate = input + row_offset * 2;
  const T* token_up = token_gate + hidden_dim;
  fp8_e4m3_t* token_output = output_q + row_offset;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  input_vec_t act_reg[kMaxRegVecs];
  float max_value = 0.0f;
#pragma unroll
  for (int r = 0; r < kMaxRegVecs; ++r) {
    const uint32_t idx = threadIdx.x + static_cast<uint32_t>(r) * kBlockSize;
    if (idx < num_vecs) {
      input_vec_t gate_vec, up_vec;
      gate_vec.load(token_gate, idx);
      up_vec.load(token_up, idx);
      max_value = math::max(max_value, compute_silu_mul_vec<T, kVecSize, kFastMath>(gate_vec, up_vec, act_reg[r]));
    }
  }

  __shared__ float reduction_smem[kSiluQuantWarpSize];
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  const float scale = reduction_smem[0] / math::FP8_E4M3_MAX;
  if (threadIdx.x == 0) {
    output_s[token_id] = scale;
  }
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

#pragma unroll
  for (int r = 0; r < kMaxRegVecs; ++r) {
    const uint32_t idx = threadIdx.x + static_cast<uint32_t>(r) * kBlockSize;
    if (idx < num_vecs) {
      output_vec_t output_vec;
      quantize_fp8_vec<T, kVecSize>(act_reg[r], scale_inv, output_vec);
      output_vec.store(token_output, idx);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

/** \brief Single-pass shared-memory-cached CTA-per-token fused SiLU+mul+per-token FP8 quant.
 *
 * For large hidden_dim (e.g. 14336..36864), caching the rounded bf16/fp16
 * activations in dynamic shared memory keeps per-thread register usage low
 * (~32 regs) so multiple CTAs can co-reside on each SM and hide barrier/HBM
 * latency.
 */
template <typename T, int kVecSize, uint32_t kBlockSize, bool kUsePDL, bool kFastMath>
__global__ __launch_bounds__(kBlockSize) void silu_and_mul_per_token_quant_fp8_cta_smem_kernel(
    const T* __restrict__ input, fp8_e4m3_t* __restrict__ output_q, float* __restrict__ output_s, uint32_t hidden_dim) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;
  extern __shared__ __align__(32) uint8_t smem_buf[];
  T* act_smem = reinterpret_cast<T*>(smem_buf);

  PDLWaitPrimary<kUsePDL>();

  const uint32_t token_id = blockIdx.x;
  const int64_t row_offset = static_cast<int64_t>(token_id) * static_cast<int64_t>(hidden_dim);
  const T* token_gate = input + row_offset * 2;
  const T* token_up = token_gate + hidden_dim;
  fp8_e4m3_t* token_output = output_q + row_offset;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  float max_value = 0.0f;
  for (uint32_t idx = threadIdx.x; idx < num_vecs; idx += kBlockSize) {
    input_vec_t gate_vec, up_vec, act_vec;
    gate_vec.load(token_gate, idx);
    up_vec.load(token_up, idx);
    max_value = math::max(max_value, compute_silu_mul_vec<T, kVecSize, kFastMath>(gate_vec, up_vec, act_vec));
    act_vec.store(act_smem, idx);
  }

  __shared__ float reduction_smem[kSiluQuantWarpSize];
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  const float scale = reduction_smem[0] / math::FP8_E4M3_MAX;
  if (threadIdx.x == 0) {
    output_s[token_id] = scale;
  }
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

  for (uint32_t idx = threadIdx.x; idx < num_vecs; idx += kBlockSize) {
    input_vec_t act_vec;
    output_vec_t output_vec;
    act_vec.load(act_smem, idx);
    quantize_fp8_vec<T, kVecSize>(act_vec, scale_inv, output_vec);
    output_vec.store(token_output, idx);
  }

  PDLTriggerSecondary<kUsePDL>();
}

/** \brief Two-pass CTA-per-token fallback for very large hidden_dim without register spilling. */
template <typename T, int kVecSize, uint32_t kBlockSize, bool kUsePDL, bool kFastMath>
__global__ __launch_bounds__(kBlockSize) void silu_and_mul_per_token_quant_fp8_cta_2pass_kernel(
    const T* __restrict__ input, fp8_e4m3_t* __restrict__ output_q, float* __restrict__ output_s, uint32_t hidden_dim) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  PDLWaitPrimary<kUsePDL>();

  const uint32_t token_id = blockIdx.x;
  const int64_t row_offset = static_cast<int64_t>(token_id) * static_cast<int64_t>(hidden_dim);
  const T* token_gate = input + row_offset * 2;
  const T* token_up = token_gate + hidden_dim;
  fp8_e4m3_t* token_output = output_q + row_offset;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  float max_value = 0.0f;
  for (uint32_t idx = threadIdx.x; idx < num_vecs; idx += kBlockSize) {
    input_vec_t gate_vec, up_vec, act_vec;
    gate_vec.load(token_gate, idx);
    up_vec.load(token_up, idx);
    max_value = math::max(max_value, compute_silu_mul_vec<T, kVecSize, kFastMath>(gate_vec, up_vec, act_vec));
  }

  __shared__ float reduction_smem[kSiluQuantWarpSize];
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  const float scale = reduction_smem[0] / math::FP8_E4M3_MAX;
  if (threadIdx.x == 0) {
    output_s[token_id] = scale;
  }
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

  for (uint32_t idx = threadIdx.x; idx < num_vecs; idx += kBlockSize) {
    input_vec_t gate_vec, up_vec, act_vec;
    output_vec_t output_vec;
    gate_vec.load(token_gate, idx);
    up_vec.load(token_up, idx);
    compute_silu_mul_vec<T, kVecSize, kFastMath>(gate_vec, up_vec, act_vec);
    quantize_fp8_vec<T, kVecSize>(act_vec, scale_inv, output_vec);
    output_vec.store(token_output, idx);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, int kVecSize, bool kUsePDL, bool kFastMath>
void launch_silu_and_mul_per_token_quant_fp8(
    DLDevice device, const T* input, fp8_e4m3_t* output_q, float* output_s, uint32_t hidden_dim, uint32_t num_tokens) {
  const uint32_t num_vecs = hidden_dim / kVecSize;
  const uint32_t sm_count = host::runtime::get_sm_count(device.device_id);

  // Dispatch to the tightest register-cache tier to avoid unused register pressure
  // and preserve high SM occupancy across common MLP widths (e.g. 4096..28672).
  constexpr int kWarpRegVecs = 64 / kVecSize;
  constexpr uint32_t kWarpMaxVecs = kSiluQuantWarpSize * kWarpRegVecs;

  constexpr int kMaxRegVecs256 = 32 / kVecSize;
  if (num_vecs <= kWarpMaxVecs && num_tokens >= sm_count * 2 * kSiluQuantTokensPerCTA) {
    const uint32_t grid = host::div_ceil(num_tokens, kSiluQuantTokensPerCTA);
    host::LaunchKernel(grid, kSiluQuantTokensPerCTA * kSiluQuantWarpSize, device)
        .enable_pdl(kUsePDL)(
            silu_and_mul_per_token_quant_fp8_warp_reg_kernel<T, kVecSize, kWarpRegVecs, kUsePDL, kFastMath>,
            input,
            output_q,
            output_s,
            hidden_dim,
            num_tokens);
  } else if (num_vecs <= 256u * 2) {
    host::LaunchKernel(num_tokens, 256, device)
        .enable_pdl(kUsePDL)(
            silu_and_mul_per_token_quant_fp8_cta_reg_kernel<T, kVecSize, 256, 2, kUsePDL, kFastMath>,
            input,
            output_q,
            output_s,
            hidden_dim);
  } else if (kMaxRegVecs256 >= 4 && num_vecs <= 256u * 4) {
    host::LaunchKernel(num_tokens, 256, device)
        .enable_pdl(kUsePDL)(
            silu_and_mul_per_token_quant_fp8_cta_reg_kernel<T, kVecSize, 256, 4, kUsePDL, kFastMath>,
            input,
            output_q,
            output_s,
            hidden_dim);
  } else {
    constexpr std::size_t kStaticSmemReserve = 2048;
    const std::size_t act_smem_bytes = static_cast<std::size_t>(hidden_dim) * sizeof(T);
    const std::size_t max_optin_smem = host::runtime::get_max_smem_per_block(device.device_id, /*opt_in=*/true);
    // Prefer dynamic SMEM caching whenever >= 2 CTAs/SM fit in SMEM so multiple
    // tokens co-reside per SM (~32 regs/thread) and overlap HBM traffic across barriers.
    if (max_optin_smem > kStaticSmemReserve && (act_smem_bytes + kStaticSmemReserve) * 2 <= max_optin_smem) {
      const std::size_t max_dyn_smem = max_optin_smem - kStaticSmemReserve;
      // Use 512 threads for small batches (lower single-token latency), for
      // kVecSize == 16 (32 regs vs 40 regs at 256 threads), or when SMEM caps
      // resident CTAs/SM to <= 4 (512 * 4 = 2048 threads/SM = 100% occupancy).
      const bool use_512 = (kVecSize == 16) || (num_tokens < sm_count * 2) ||
                           ((act_smem_bytes + kStaticSmemReserve) * 5 > max_optin_smem);
      if (use_512) {
        constexpr auto kernel = silu_and_mul_per_token_quant_fp8_cta_smem_kernel<T, kVecSize, 512, kUsePDL, kFastMath>;
        host::runtime::set_max_dynamic_smem_per_device<kernel>(device, max_dyn_smem);
        host::LaunchKernel(num_tokens, 512, device, act_smem_bytes)
            .enable_pdl(kUsePDL)(kernel, input, output_q, output_s, hidden_dim);
      } else {
        constexpr auto kernel = silu_and_mul_per_token_quant_fp8_cta_smem_kernel<T, kVecSize, 256, kUsePDL, kFastMath>;
        host::runtime::set_max_dynamic_smem_per_device<kernel>(device, max_dyn_smem);
        host::LaunchKernel(num_tokens, 256, device, act_smem_bytes)
            .enable_pdl(kUsePDL)(kernel, input, output_q, output_s, hidden_dim);
      }
    } else if (num_vecs <= 256u * 6) {
      host::LaunchKernel(num_tokens, 256, device)
          .enable_pdl(kUsePDL)(
              silu_and_mul_per_token_quant_fp8_cta_reg_kernel<T, kVecSize, 256, 6, kUsePDL, kFastMath>,
              input,
              output_q,
              output_s,
              hidden_dim);
    } else if (num_vecs <= 256u * 8) {
      host::LaunchKernel(num_tokens, 256, device)
          .enable_pdl(kUsePDL)(
              silu_and_mul_per_token_quant_fp8_cta_reg_kernel<T, kVecSize, 256, 8, kUsePDL, kFastMath>,
              input,
              output_q,
              output_s,
              hidden_dim);
    } else if (num_vecs <= 512u * 5) {
      host::LaunchKernel(num_tokens, 512, device)
          .enable_pdl(kUsePDL)(
              silu_and_mul_per_token_quant_fp8_cta_reg_kernel<T, kVecSize, 512, 5, kUsePDL, kFastMath>,
              input,
              output_q,
              output_s,
              hidden_dim);
    } else if (kVecSize <= 8 && num_vecs <= 1024u * 4) {
      host::LaunchKernel(num_tokens, 1024, device)
          .enable_pdl(kUsePDL)(
              silu_and_mul_per_token_quant_fp8_cta_reg_kernel<T, kVecSize, 1024, 4, kUsePDL, kFastMath>,
              input,
              output_q,
              output_s,
              hidden_dim);
    } else {
      host::LaunchKernel(num_tokens, 512, device)
          .enable_pdl(kUsePDL)(
              silu_and_mul_per_token_quant_fp8_cta_2pass_kernel<T, kVecSize, 512, kUsePDL, kFastMath>,
              input,
              output_q,
              output_s,
              hidden_dim);
    }
  }
}

/** \brief Validate and launch fused SiLU-and-mul + dynamic per-token FP8 E4M3 quantization. */
template <typename T, bool kUsePDL, bool kFastMath>
void silu_and_mul_per_token_quant_fp8(
    tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
  using namespace host;
  static_assert(sizeof(T) == 2, "silu_and_mul_per_token_quant_fp8 supports fp16/bf16 only");

  auto M = SymbolicSize{"num_tokens"};
  auto K2 = SymbolicSize{"gate_up_dim"};
  auto K = SymbolicSize{"hidden_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLGPU>();

  TensorMatcher({M, K2}).with_dtype<T>().with_device(device).verify(input);
  TensorMatcher({M, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
  TensorMatcher({M, 1}).with_dtype<float>().with_device(device).verify(output_s);

  CHECK_HOST(M.unwrap() > 0) << "silu_and_mul_per_token_quant_fp8: num_tokens must be positive";
  CHECK_HOST(K.unwrap() > 0 && K2.unwrap() == K.unwrap() * 2)
      << "silu_and_mul_per_token_quant_fp8: input last dim (" << K2.unwrap() << ") must be 2 * output last dim ("
      << K.unwrap() << ")";
  CHECK_HOST(K.unwrap() % 4 == 0) << "silu_and_mul_per_token_quant_fp8: hidden_dim must be divisible by 4, got "
                                  << K.unwrap();
  CHECK_HOST(M.unwrap() <= UINT32_MAX && K.unwrap() <= UINT32_MAX)
      << "silu_and_mul_per_token_quant_fp8: dimensions exceed uint32 indexing";

  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden_dim = static_cast<uint32_t>(K.unwrap());
  const auto* input_ptr = static_cast<const T*>(input.data_ptr());
  auto* output_q_ptr = static_cast<fp8_e4m3_t*>(output_q.data_ptr());
  auto* output_s_ptr = static_cast<float*>(output_s.data_ptr());

  const auto in_addr = reinterpret_cast<uintptr_t>(input_ptr);
  const auto out_addr = reinterpret_cast<uintptr_t>(output_q_ptr);

  if constexpr (device::kMaxVecBytes >= 32) {
    if (hidden_dim % 16 == 0 && (in_addr % 32 == 0) && (out_addr % 16 == 0)) {
      launch_silu_and_mul_per_token_quant_fp8<T, 16, kUsePDL, kFastMath>(
          device.unwrap(), input_ptr, output_q_ptr, output_s_ptr, hidden_dim, num_tokens);
      return;
    }
  }
  if (hidden_dim % 8 == 0 && (in_addr % 16 == 0) && (out_addr % 8 == 0)) {
    launch_silu_and_mul_per_token_quant_fp8<T, 8, kUsePDL, kFastMath>(
        device.unwrap(), input_ptr, output_q_ptr, output_s_ptr, hidden_dim, num_tokens);
  } else {
    CHECK_HOST((in_addr % 8 == 0) && (out_addr % 4 == 0))
        << "silu_and_mul_per_token_quant_fp8: input/output pointers must be 8B/4B aligned";
    launch_silu_and_mul_per_token_quant_fp8<T, 4, kUsePDL, kFastMath>(
        device.unwrap(), input_ptr, output_q_ptr, output_s_ptr, hidden_dim, num_tokens);
  }
}

}  // namespace sglang
