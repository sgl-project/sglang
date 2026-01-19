#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/impl/norm.cuh>
#include <sgl_kernel/impl/norm_fusion.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cuda_fp16.h>

namespace {

using host::norm::NormEnum;
using host::norm_fusion::IndexEnum;

template <typename T>
using Vec4 = device::AlignedVector<T, 4>;

template <int V>
struct ItemPerThreadTag {
  static constexpr int value = V;
};

template <typename T, int64_t kDim, NormEnum norm_enum, IndexEnum scale_index_enum, IndexEnum shift_index_enum>
__global__ void norm_fused_scale_shift_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    const int S,
    const int F,
    bool affine,
    float eps) {
  using namespace device;
  using namespace device::norm;
  using PackedT = packed_t<T>;
  using Storage = StorageType<T, kDim>;

  constexpr bool kUseCTA = host::norm::should_use_cta<T, kDim>();
  // Storage size: for CTA norm it's 4, for warp norm it's kDim / (2 * 32)
  constexpr int kStorageSize = kUseCTA ? 4 : (kDim / (2 * kWarpThreads));
  constexpr int kPackedPerRow = kDim / 2;

  __shared__ float smem_buffer[kSmemBufferSize];

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int b_id = bidx / S;
  const int s_id = bidx % S;

  // Compute offsets
  const int row_packed_offset = bidx * kPackedPerRow;
  const int thread_packed_offset = row_packed_offset + tidx * kStorageSize;
  const int gamma_packed_offset = tidx * kStorageSize;  // gamma/beta broadcast across rows

  const int scale_row = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id);
  const int shift_row = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id);

  // ============ Step 1: Load data ============
  const auto* input_packed = reinterpret_cast<const PackedT*>(input);
  const auto* gamma_packed = reinterpret_cast<const PackedT*>(gamma);
  const auto* beta_packed = reinterpret_cast<const PackedT*>(beta);

  Storage input_vec, gamma_vec, beta_vec;
  input_vec.load(input_packed + thread_packed_offset);

  if (affine) {
    gamma_vec.load(gamma_packed + gamma_packed_offset);
    if constexpr (norm_enum == NormEnum::LayerNorm) {
      beta_vec.load(beta_packed + gamma_packed_offset);
    } else {
      // RMSNorm doesn't use beta, fill with zeros
#pragma unroll
      for (int i = 0; i < kStorageSize; ++i) {
        beta_vec[i] = cast<PackedT, fp32x2_t>({0.0f, 0.0f});
      }
    }
  } else {
    // Default values: gamma=1, beta=0
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      gamma_vec[i] = cast<PackedT, fp32x2_t>({1.0f, 1.0f});
      beta_vec[i] = cast<PackedT, fp32x2_t>({0.0f, 0.0f});
    }
  }

  // ============ Step 2: Apply norm using norm template ============
  Storage norm_output;
  if constexpr (kUseCTA) {
    norm_output = apply_norm_cta<norm_enum, kDim>(input_vec, gamma_vec, beta_vec, eps, smem_buffer);
  } else {
    norm_output = apply_norm_warp<norm_enum, kDim>(input_vec, gamma_vec, beta_vec, eps);
  }

  // ============ Step 3: Load scale/shift and apply ============
  const auto* scale_packed = reinterpret_cast<const PackedT*>(scale);
  const auto* shift_packed = reinterpret_cast<const PackedT*>(shift);

  Storage scale_vec, shift_vec;

  // Load scale based on IndexEnum
  if constexpr (scale_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(scale[0]);
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      scale_vec[i] = cast<PackedT, fp32x2_t>({s, s});
    }
  } else {
    const int scale_packed_offset = scale_row * kPackedPerRow + tidx * kStorageSize;
    scale_vec.load(scale_packed + scale_packed_offset);
  }

  // Load shift based on IndexEnum
  if constexpr (shift_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(shift[0]);
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      shift_vec[i] = cast<PackedT, fp32x2_t>({s, s});
    }
  } else {
    const int shift_packed_offset = shift_row * kPackedPerRow + tidx * kStorageSize;
    shift_vec.load(shift_packed + shift_packed_offset);
  }

  // Apply: output = norm * (1 + scale) + shift
  Storage output_vec;
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto norm_fp32 = cast<fp32x2_t>(norm_output[i]);
    auto scale_fp32 = cast<fp32x2_t>(scale_vec[i]);
    auto shift_fp32 = cast<fp32x2_t>(shift_vec[i]);

    float out_x = norm_fp32.x * (1.0f + scale_fp32.x) + shift_fp32.x;
    float out_y = norm_fp32.y * (1.0f + scale_fp32.y) + shift_fp32.y;

    output_vec[i] = cast<PackedT, fp32x2_t>({out_x, out_y});
  }

  // ============ Step 4: Store output ============
  auto* output_packed = reinterpret_cast<PackedT*>(output);
  output_vec.store(output_packed + thread_packed_offset);
}

template <NormEnum norm_enum, typename T, IndexEnum scale_index_enum, IndexEnum shift_index_enum, int64_t kDim>
void fused_norm_scale_shift(
    tvm::ffi::TensorView out,
    const tvm::ffi::TensorView x,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gamma_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> beta_opt,
    const tvm::ffi::TensorView scale,
    const tvm::ffi::TensorView shift,
    double eps) {
  using namespace host;

  static_assert(
      std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>, "Only support fp16, bf16 for norm template version");
  static_assert(
      norm_enum == NormEnum::LayerNorm || norm_enum == NormEnum::RMSNorm, "norm_enum must be layernorm or rmsnorm.");
  static_assert(host::norm::is_config_supported<T, kDim>(), "Unsupported norm configuration for kDim");

  host::norm_fusion::Matcher<T> checker;
  checker.template match<IndexEnum::NoBroadcast>(out);
  checker.template match<IndexEnum::NoBroadcast>(x);
  checker.template match<scale_index_enum>(scale);
  checker.template match<shift_index_enum>(shift);
  bool affine = gamma_opt.has_value();
  if (affine) {
    checker.template match<IndexEnum::BroadcastBS>(gamma_opt.value());
    if (beta_opt.has_value()) {
      checker.template match<IndexEnum::BroadcastBS>(beta_opt.value());
    }
  }

  const auto B = checker.B_.unwrap();
  const auto S = checker.S_.unwrap();
  const auto F = checker.has_value_F ? checker.F_.unwrap() : 0;
  const auto D = checker.D_.unwrap();

  // Verify D matches kDim
  RuntimeCheck(D == kDim, "Tensor dimension D must match template kDim");

  // Compute thread configuration based on kDim
  constexpr bool kUseCTA = host::norm::should_use_cta<T, kDim>();
  constexpr uint32_t kThreads = kUseCTA ? host::norm::get_cta_threads<T, kDim>() : device::kWarpThreads;

  dim3 grid(B * S);
  dim3 block(kThreads);

  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;

  // Launch kernel
  LaunchKernel(grid, block, x.device())(
      norm_fused_scale_shift_kernel<T, kDim, norm_enum, scale_index_enum, shift_index_enum>,
      (T*)out.data_ptr(),
      (const T*)x.data_ptr(),
      (const T*)gamma_ptr,
      (const T*)beta_ptr,
      (const T*)scale.data_ptr(),
      (const T*)shift.data_ptr(),
      S,
      F,
      affine,
      static_cast<float>(eps));
}

template <
    typename T,
    int64_t kDim,
    NormEnum norm_enum,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum,
    IndexEnum gate_index_enum>
__global__ void norm_fused_res_gate_scale_shift_kernel(
    T* __restrict__ output,
    T* __restrict__ residual_out,
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    const T* __restrict__ gate,
    const int S,
    const int F,
    bool affine,
    float eps) {
  using namespace device;
  using namespace device::norm;
  using PackedT = packed_t<T>;
  using Storage = StorageType<T, kDim>;

  constexpr bool kUseCTA = host::norm::should_use_cta<T, kDim>();
  // Storage size: for CTA norm it's 4, for warp norm it's kDim / (2 * 32)
  constexpr int kStorageSize = kUseCTA ? 4 : (kDim / (2 * kWarpThreads));
  constexpr int kPackedPerRow = kDim / 2;

  // ============ Setup ============
  __shared__ float smem_buffer[kSmemBufferSize];

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int b_id = bidx / S;
  const int s_id = bidx % S;

  // Compute offsets
  const int row_packed_offset = bidx * kPackedPerRow;
  const int thread_packed_offset = row_packed_offset + tidx * kStorageSize;
  const int gamma_packed_offset = tidx * kStorageSize;

  const int scale_row = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id);
  const int shift_row = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id);
  const int gate_row = norm_fusion::get_offset<gate_index_enum>(S, F, b_id, s_id);

  // ============ Step 1: Load x, residual, gate and compute residual + x * gate ============
  const auto* x_packed = reinterpret_cast<const PackedT*>(x);
  const auto* residual_packed = reinterpret_cast<const PackedT*>(residual);
  const auto* gate_packed = reinterpret_cast<const PackedT*>(gate);
  const auto* gamma_packed = reinterpret_cast<const PackedT*>(gamma);
  const auto* beta_packed = reinterpret_cast<const PackedT*>(beta);

  Storage input_vec, gamma_vec, beta_vec;

  // Load and compute residual + x * gate
  Storage x_vec, r_vec, g_vec;
  x_vec.load(x_packed + thread_packed_offset);
  r_vec.load(residual_packed + thread_packed_offset);

  // Load gate based on IndexEnum
  if constexpr (gate_index_enum == IndexEnum::NotATensor) {
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      g_vec[i] = cast<PackedT, fp32x2_t>({1.0f, 1.0f});
    }
  } else if constexpr (gate_index_enum == IndexEnum::Scalar) {
    float g = static_cast<float>(gate[0]);
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      g_vec[i] = cast<PackedT, fp32x2_t>({g, g});
    }
  } else {
    const int gate_packed_offset = gate_row * kPackedPerRow + tidx * kStorageSize;
    g_vec.load(gate_packed + gate_packed_offset);
  }

  // Compute: input = residual + x * gate
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto x_fp32 = cast<fp32x2_t>(x_vec[i]);
    auto r_fp32 = cast<fp32x2_t>(r_vec[i]);
    auto g_fp32 = cast<fp32x2_t>(g_vec[i]);

    float sum_x = r_fp32.x + x_fp32.x * g_fp32.x;
    float sum_y = r_fp32.y + x_fp32.y * g_fp32.y;

    input_vec[i] = cast<PackedT, fp32x2_t>({sum_x, sum_y});
  }

  // Store residual_out if needed
  if (residual_out != nullptr) {
    auto* residual_out_packed = reinterpret_cast<PackedT*>(residual_out);
    input_vec.store(residual_out_packed + thread_packed_offset);
  }

  // ============ Step 2: Load gamma/beta ============
  if (affine) {
    gamma_vec.load(gamma_packed + gamma_packed_offset);
    if constexpr (norm_enum == NormEnum::LayerNorm) {
      beta_vec.load(beta_packed + gamma_packed_offset);
    } else {
#pragma unroll
      for (int i = 0; i < kStorageSize; ++i) {
        beta_vec[i] = cast<PackedT, fp32x2_t>({0.0f, 0.0f});
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      gamma_vec[i] = cast<PackedT, fp32x2_t>({1.0f, 1.0f});
      beta_vec[i] = cast<PackedT, fp32x2_t>({0.0f, 0.0f});
    }
  }

  // ============ Step 3: Apply norm using norm template ============
  Storage norm_output;
  if constexpr (kUseCTA) {
    norm_output = apply_norm_cta<norm_enum, kDim>(input_vec, gamma_vec, beta_vec, eps, smem_buffer);
  } else {
    norm_output = apply_norm_warp<norm_enum, kDim>(input_vec, gamma_vec, beta_vec, eps);
  }

  // ============ Step 4: Load scale/shift and apply ============
  const auto* scale_packed = reinterpret_cast<const PackedT*>(scale);
  const auto* shift_packed = reinterpret_cast<const PackedT*>(shift);

  Storage scale_vec, shift_vec;

  // Load scale
  if constexpr (scale_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(scale[0]);
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      scale_vec[i] = cast<PackedT, fp32x2_t>({s, s});
    }
  } else {
    const int scale_packed_offset = scale_row * kPackedPerRow + tidx * kStorageSize;
    scale_vec.load(scale_packed + scale_packed_offset);
  }

  // Load shift
  if constexpr (shift_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(shift[0]);
#pragma unroll
    for (int i = 0; i < kStorageSize; ++i) {
      shift_vec[i] = cast<PackedT, fp32x2_t>({s, s});
    }
  } else {
    const int shift_packed_offset = shift_row * kPackedPerRow + tidx * kStorageSize;
    shift_vec.load(shift_packed + shift_packed_offset);
  }

  // Apply: output = norm * (1 + scale) + shift
  Storage output_vec;
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto norm_fp32 = cast<fp32x2_t>(norm_output[i]);
    auto scale_fp32 = cast<fp32x2_t>(scale_vec[i]);
    auto shift_fp32 = cast<fp32x2_t>(shift_vec[i]);

    float out_x = norm_fp32.x * (1.0f + scale_fp32.x) + shift_fp32.x;
    float out_y = norm_fp32.y * (1.0f + scale_fp32.y) + shift_fp32.y;

    output_vec[i] = cast<PackedT, fp32x2_t>({out_x, out_y});
  }

  // ============ Step 5: Store output ============
  auto* output_packed = reinterpret_cast<PackedT*>(output);
  output_vec.store(output_packed + thread_packed_offset);
}

template <
    NormEnum norm_enum,
    typename T,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum,
    IndexEnum gate_index_enum,
    int64_t kDim>
void fused_scale_residual_norm_scale_shift(
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView residual_out,
    const tvm::ffi::TensorView residual,
    const tvm::ffi::TensorView x,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gate_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gamma_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> beta_opt,
    const tvm::ffi::TensorView scale,
    const tvm::ffi::TensorView shift,
    double eps) {
  using namespace host;

  static_assert(
      std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>, "Only support fp16, bf16 for norm template version");
  static_assert(
      norm_enum == NormEnum::LayerNorm || norm_enum == NormEnum::RMSNorm, "norm_enum must be layernorm or rmsnorm.");
  static_assert(host::norm::is_config_supported<T, kDim>(), "Unsupported norm configuration for kDim");

  norm_fusion::Matcher<T> checker;
  checker.template match<IndexEnum::NoBroadcast>(y);
  checker.template match<IndexEnum::NoBroadcast>(residual_out);
  checker.template match<IndexEnum::NoBroadcast>(x);
  checker.template match<IndexEnum::NoBroadcast>(residual);
  checker.template match<scale_index_enum>(scale);
  checker.template match<shift_index_enum>(shift);
  if (gate_opt.has_value()) {
    checker.template match<gate_index_enum>(gate_opt.value());
  }
  bool affine = gamma_opt.has_value();
  if (affine) {
    checker.template match<IndexEnum::BroadcastBS>(gamma_opt.value());
    if (beta_opt.has_value()) {
      checker.template match<IndexEnum::BroadcastBS>(beta_opt.value());
    }
  }

  const auto B = checker.B_.unwrap();
  const auto S = checker.S_.unwrap();
  const auto F = checker.has_value_F ? checker.F_.unwrap() : 0;
  const auto D = checker.D_.unwrap();

  // Verify D matches kDim
  RuntimeCheck(D == kDim, "Tensor dimension D must match template kDim");

  // Compute thread configuration based on kDim
  constexpr bool kUseCTA = host::norm::should_use_cta<T, kDim>();
  constexpr uint32_t kThreads = kUseCTA ? host::norm::get_cta_threads<T, kDim>() : device::kWarpThreads;

  dim3 grid(B * S);
  dim3 block(kThreads);

  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;
  auto gate_ptr = gate_opt.has_value() ? gate_opt.value().data_ptr() : nullptr;

  // Launch kernel
  LaunchKernel(grid, block, x.device())(
      norm_fused_res_gate_scale_shift_kernel<T, kDim, norm_enum, scale_index_enum, shift_index_enum, gate_index_enum>,
      (T*)y.data_ptr(),
      (T*)residual_out.data_ptr(),
      (const T*)x.data_ptr(),
      (const T*)residual.data_ptr(),
      (const T*)gamma_ptr,
      (const T*)beta_ptr,
      (const T*)scale.data_ptr(),
      (const T*)shift.data_ptr(),
      (const T*)gate_ptr,
      S,
      F,
      affine,
      static_cast<float>(eps));
}

}  // namespace
