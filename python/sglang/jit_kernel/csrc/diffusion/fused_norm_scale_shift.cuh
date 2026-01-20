#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/impl/norm.cuh>
#include <sgl_kernel/impl/norm_fusion.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cuda_fp16.h>

namespace {

using host::norm::NormEnum;
using host::norm_fusion::IndexEnum;

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
  using Storage = norm::StorageType<T, kDim>;

  constexpr int kStorageSize = 4;
  __shared__ float smem_buffer[kSmemBufferSize];
  const int bidx = blockIdx.x;
  const int b_id = bidx / S, s_id = bidx % S;
  const auto gmem = tile::Memory<Storage>::cta();

  // Compute offsets
  const int scale_row = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id);
  const int shift_row = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id);

  // ============ Step 1: normed = norm(input) * gamma + beta ============
  Storage beta_vec;
  const auto input_vec = gmem.load(input + bidx * kDim);
  const auto gamma_vec = affine ? gmem.load(gamma) : Storage(cast<PackedT, fp32x2_t>({1.0f, 1.0f}));
  if constexpr (norm_enum == NormEnum::LayerNorm)
    beta_vec = affine ? gmem.load(beta) : Storage(cast<PackedT, fp32x2_t>({0.0f, 0.0f}));
  else
    beta_vec = Storage(cast<PackedT, fp32x2_t>({0.0f, 0.0f}));
  const auto norm_output = apply_norm_cta<norm_enum, kDim>(input_vec, gamma_vec, beta_vec, eps, smem_buffer);
  // ============ Step 2: output = normed * (1 + scale) + shift ============
  Storage scale_vec, shift_vec;
  if constexpr (scale_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(scale[0]);
    scale_vec.fill(cast<PackedT, fp32x2_t>({s, s}));
  } else {
    scale_vec = gmem.load(scale + scale_row * kDim);
  }
  if constexpr (shift_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(shift[0]);
    shift_vec.fill(cast<PackedT, fp32x2_t>({s, s}));
  } else {
    shift_vec = gmem.load(shift + shift_row * kDim);
  }
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
  gmem.store(output + bidx * kDim, output_vec);
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
  RuntimeCheck(D == kDim, "Tensor dimension D must match template kDim");

  // Compute thread configuration based on kDim
  constexpr uint32_t kThreads = host::norm::get_cta_threads<T, kDim>();

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
  using Storage = norm::StorageType<T, kDim>;
  // ============ Setup ============
  __shared__ float smem_buffer[kSmemBufferSize];
  const int bidx = blockIdx.x;
  const int b_id = bidx / S, s_id = bidx % S;
  const auto gmem = tile::Memory<Storage>::cta();
  constexpr int kStorageSize = 4;

  // Compute offsets
  const int scale_row = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id);
  const int shift_row = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id);
  const int gate_row = norm_fusion::get_offset<gate_index_enum>(S, F, b_id, s_id);

  // ============ Step 1: normed = norm(residual + x * gate) * gamma + beta ============
  const auto x_vec = gmem.load(x + bidx * kDim);
  const auto r_vec = gmem.load(residual + bidx * kDim);
  Storage g_vec;
  if constexpr (gate_index_enum == IndexEnum::NotATensor) {
    g_vec.fill(cast<PackedT, fp32x2_t>({1.0f, 1.0f}));
  } else if constexpr (gate_index_enum == IndexEnum::Scalar) {
    float g = static_cast<float>(gate[0]);
    g_vec.fill(cast<PackedT, fp32x2_t>({g, g}));
  } else {
    g_vec = gmem.load(gate + gate_row * kDim);
  }
  Storage gated;
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto x_fp32 = cast<fp32x2_t>(x_vec[i]);
    auto r_fp32 = cast<fp32x2_t>(r_vec[i]);
    auto g_fp32 = cast<fp32x2_t>(g_vec[i]);
    float sum_x = r_fp32.x + x_fp32.x * g_fp32.x;
    float sum_y = r_fp32.y + x_fp32.y * g_fp32.y;
    gated[i] = cast<PackedT, fp32x2_t>({sum_x, sum_y});
  }
  if (residual_out != nullptr) {
    gmem.store(residual_out + bidx * kDim, gated);
  }
  Storage beta_vec;
  const auto gamma_vec = affine ? gmem.load(gamma) : Storage(cast<PackedT, fp32x2_t>({1.0f, 1.0f}));
  if constexpr (norm_enum == NormEnum::LayerNorm)
    beta_vec = affine ? gmem.load(beta) : Storage(cast<PackedT, fp32x2_t>({0.0f, 0.0f}));
  else
    beta_vec = Storage(cast<PackedT, fp32x2_t>({0.0f, 0.0f}));
  const auto normed = apply_norm_cta<norm_enum, kDim>(gated, gamma_vec, beta_vec, eps, smem_buffer);

  // ============ Step 2: output = normed * (1 + scale) + shift ============
  Storage scale_vec, shift_vec;
  if constexpr (scale_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(scale[0]);
    scale_vec.fill(cast<PackedT, fp32x2_t>({s, s}));
  } else {
    scale_vec = gmem.load(scale + scale_row * kDim);
  }
  if constexpr (shift_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(shift[0]);
    shift_vec.fill(cast<PackedT, fp32x2_t>({s, s}));
  } else {
    shift_vec = gmem.load(shift + shift_row * kDim);
  }
  Storage output_vec;
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto norm_fp32 = cast<fp32x2_t>(normed[i]);
    auto scale_fp32 = cast<fp32x2_t>(scale_vec[i]);
    auto shift_fp32 = cast<fp32x2_t>(shift_vec[i]);
    float out_x = norm_fp32.x * (1.0f + scale_fp32.x) + shift_fp32.x;
    float out_y = norm_fp32.y * (1.0f + scale_fp32.y) + shift_fp32.y;
    output_vec[i] = cast<PackedT, fp32x2_t>({out_x, out_y});
  }
  gmem.store(output + bidx * kDim, output_vec);
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
  RuntimeCheck(D == kDim, "Tensor dimension D must match template kDim");

  // Compute thread configuration based on kDim
  constexpr uint32_t kThreads = host::norm::get_cta_threads<T, kDim>();

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
