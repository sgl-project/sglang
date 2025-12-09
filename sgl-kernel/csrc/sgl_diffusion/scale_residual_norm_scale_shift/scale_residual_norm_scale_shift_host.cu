

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/SmallVector.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

#include "kernel_welford.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DIM(x, n) TORCH_CHECK((x).dim() == (n), #x " must have " #n " dimensions")

namespace {

template <typename DType>
struct BroadcastParam {
  at::Tensor tensor;
  BroadcastDesc<DType> desc;
};

bool tensor_aligned_for_vectorized_load(const at::Tensor& t) {
  auto dt = t.scalar_type();
  uintptr_t addr = reinterpret_cast<uintptr_t>(t.data_ptr());
  if (dt == at::kFloat) {
    return addr % 16 == 0;
  }
  if (dt == at::kHalf || dt == at::kBFloat16) {
    return addr % 8 == 0;
  }
  return false;
}

bool optional_tensor_aligned_for_vectorized_load(const c10::optional<at::Tensor>& t_opt) {
  if (!t_opt.has_value()) return true;
  return tensor_aligned_for_vectorized_load(t_opt.value());
}

template <typename DType>
BroadcastParam<DType> prepare_scale_shift_tensor(
    const at::Tensor& tensor, int64_t B, int64_t S, int64_t D, const at::TensorOptions& options) {
  BroadcastParam<DType> param;
  TORCH_CHECK(tensor.defined(), "Tensor must be defined.");
  auto t = tensor.to(options.dtype());
  param.desc.frame_len = S;

  const int64_t ndim = t.dim();
  if (ndim == 0) {
    // (scalar) -> layout(shape=(1), stride=(0))
    DType value = t.item<DType>();
    param.desc.stride_b = -1;
    param.tensor = t;
    param.desc.union_value.value = value;
  } else if (ndim == 1) {
    // (1) -> layout(shape=(1), stride=(0))
    TORCH_CHECK(t.size(0) == 1, "Expected shape [1] for broadcast tensor.");
    DType value = t[0].item<DType>();
    param.desc.stride_b = -1;
    param.tensor = t;
    param.desc.union_value.value = value;
  } else if (ndim == 2) {
    // (B,D) -> layout(shape=(B,1,D), stride=(stride_B,0,1))
    // (1,D) -> layout(shape=(1,1,D), stride=(0,0,1))
    TORCH_CHECK(t.size(1) == D, "Trailing dim must match hidden size.");
    TORCH_CHECK(t.size(0) == B || t.size(0) == 1, "Leading dim must be batch size or 1.");
    param.desc.stride_b = t.size(0) == B ? S : 0;
    param.desc.frame_len = S;
    t = t.reshape({t.size(0), 1, D});
    t = t.contiguous();
    param.tensor = t;
    param.desc.union_value.ptr = t.data_ptr<DType>();
  } else if (ndim == 3) {
    // (B,S,D)
    // (B,1,D)
    // (1,S,D)
    // (1,1,D)
    TORCH_CHECK(t.size(2) == D, "Trailing dim must match hidden size.");
    TORCH_CHECK(t.size(0) == B || t.size(0) == 1, "Leading dim must be batch size or 1.");
    TORCH_CHECK(t.size(1) == S || t.size(1) == 1, "Middle dim must be sequence length or 1.");
    param.desc.stride_b = t.size(0) == B ? S : 0;
    param.desc.frame_len = S / t.size(1);
    t = t.contiguous();
    param.tensor = t;
    param.desc.union_value.ptr = t.data_ptr<DType>();
  } else if (ndim == 4) {
    // (B,F,1,D) -> (B,F,D)
    TORCH_CHECK(t.size(2) == 1 && t.size(3) == D, "Expected [B,F,1,D] for frame broadcast.");
    TORCH_CHECK(t.size(0) == B || t.size(0) == 1, "Leading dim must be batch size or 1.");
    auto num_frames = t.size(1);
    TORCH_CHECK(S % num_frames == 0, "Sequence length must be divisible by num_frames.");
    t = t.reshape({t.size(0), num_frames, D});
    param.desc.stride_b = S;
    param.desc.frame_len = S / num_frames;
    t = t.contiguous();
    param.tensor = t;
    param.desc.union_value.ptr = t.data_ptr<DType>();
  } else {
    TORCH_CHECK(false, "Unsupported rank for broadcast tensor.");
  }
  return param;
}

struct GateParam {
  at::Tensor storage;
  int frame_len;
};

template <typename DType>
GateParam prepare_gate(
    const c10::optional<at::Tensor>& gate_opt, int64_t B, int64_t S, int64_t D, const at::TensorOptions& options) {
  GateParam gate_param;
  int64_t num_frames = 1;
  at::Tensor gate_prepared;
  if (gate_opt.has_value() && gate_opt.value().defined()) {
    const auto& gate = gate_opt.value();
    CHECK_CUDA(gate);
    if (gate.dim() == 2) {
      TORCH_CHECK(gate.size(1) == D, "2D-gate hidden size mismatch");
      TORCH_CHECK(gate.size(0) == 1, "2D-gate tensor must be [1,D]");
      gate_prepared = gate.contiguous().to(options.dtype()).view({1, D});
      gate_param.frame_len = -2;
    } else if (gate.dim() == 3) {
      TORCH_CHECK(gate.size(0) == B, "3D-gate batch size mismatch");
      TORCH_CHECK(gate.size(2) == D, "3D-gate hidden size mismatch");
      TORCH_CHECK(gate.size(1) == 1, "3D-gate tensor must be [B,1,D]");
      gate_prepared = gate.contiguous().to(options.dtype()).view({B, 1, D});
      gate_param.frame_len = S / num_frames;
    } else if (gate.dim() == 4) {
      TORCH_CHECK(gate.size(0) == B, "4D-gate batch size mismatch");
      TORCH_CHECK(gate.size(3) == D, "4D-gate hidden size mismatch");
      TORCH_CHECK(gate.size(2) == 1, "4D-gate tensor must be [B,F,1,D]");
      num_frames = gate.size(1);
      TORCH_CHECK(S % num_frames == 0, "sequence length must be divisible by num_frames");
      gate_prepared = gate.contiguous().to(options.dtype()).view({B, num_frames, 1, D});
      gate_param.frame_len = S / num_frames;
    } else {
      TORCH_CHECK(false, "gate tensor must be rank 3 or 4");
    }
    gate_param.storage = gate_prepared;
  } else {
    gate_param.frame_len = -1;
  }
  return gate_param;
}

struct NormParams {
  at::Tensor weight;
  bool has_weight_tensor;
  at::Tensor bias;
  bool has_bias_tensor;
};

template <typename ParamDType>
NormParams prepare_norm_params(
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t D,
    const at::TensorOptions& options) {
  NormParams params;
  if (weight_opt.has_value() && weight_opt.value().defined()) {
    const auto& norm_weight = weight_opt.value();
    CHECK_CUDA(norm_weight);
    TORCH_CHECK(norm_weight.numel() == D, "norm_weight must have length D");
    params.weight = norm_weight.contiguous().to(options.dtype());
    params.has_weight_tensor = true;
  } else {
    params.has_weight_tensor = false;
  }
  if (bias_opt.has_value() && bias_opt.value().defined()) {
    const auto& norm_bias = bias_opt.value();
    CHECK_CUDA(norm_bias);
    TORCH_CHECK(norm_bias.numel() == D, "norm_bias must have length D");
    params.bias = norm_bias.contiguous().to(options.dtype());
    params.has_bias_tensor = true;
  } else {
    params.has_bias_tensor = false;
  }
  return params;
}

template <typename DType, typename ParamDType, NormType norm_type, bool is_d_aligned>
void launch_fused(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    DType* residual,
    DType* x,
    DType* gate,
    const ParamDType* w,
    const ParamDType* b,
    BroadcastDesc<DType> shift_desc,
    BroadcastDesc<DType> scale_desc,
    double eps,
    DType* modulated,
    DType* residual_output,
    int B,
    int S,
    int D,
    int frame_len,
    bool is_warp_reduce,
    bool has_weight_tensor,
    bool has_bias_tensor) {
  scale_residual_norm_scale_shift_kernel<DType, ParamDType, norm_type, is_d_aligned><<<grid, block, 0, stream>>>(
      residual,
      x,
      gate,
      w,
      b,
      shift_desc,
      scale_desc,
      eps,
      modulated,
      residual_output,
      B,
      S,
      D,
      frame_len,
      is_warp_reduce,
      has_weight_tensor,
      has_bias_tensor);
}

template <typename DType, typename ParamDType>
using LauncherFn = void (*)(
    dim3,
    dim3,
    cudaStream_t,
    DType*,
    DType*,
    DType*,
    const ParamDType*,
    const ParamDType*,
    BroadcastDesc<DType>,
    BroadcastDesc<DType>,
    double,
    DType*,
    DType*,
    int,
    int,
    int,
    int,
    bool,
    bool,
    bool);

template <typename DType, typename ParamDType>
static constexpr LauncherFn<DType, ParamDType> DISPATCH_TABLE[2][2] = {
    {&launch_fused<DType, ParamDType, NormType::LayerNorm, false>,
     &launch_fused<DType, ParamDType, NormType::LayerNorm, true>},
    {&launch_fused<DType, ParamDType, NormType::RMSNorm, false>,
     &launch_fused<DType, ParamDType, NormType::RMSNorm, true>}};
}  // namespace

/*==========================================================================*
 *  Public entry point invoked from Python. It validates inputs, prepares   *
 *  all broadcast buffers (gate/norm/scale/shift), and dispatches the CUDA  *
 *  kernel that fuses gate + normalization + scale/shift.                   *
 *==========================================================================*/
std::tuple<at::Tensor, at::Tensor> scale_residual_norm_scale_shift(
    const at::Tensor& residual,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& gate_opt,
    const c10::optional<at::Tensor>& norm_weight_opt,
    const c10::optional<at::Tensor>& norm_bias_opt,
    const at::Tensor& shift,
    const at::Tensor& scale,
    double eps,
    bool use_rms_norm) {
  // --- basic input validation ---
  CHECK_CUDA(residual);
  CHECK_CUDA(x);
  CHECK_CUDA(shift);
  CHECK_CUDA(scale);
  TORCH_CHECK(residual.dim() == 3, "residual must be [B, S, D]");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x must match residual shape");

  const auto B = residual.size(0);
  const auto S = residual.size(1);
  const auto D = residual.size(2);
  auto orig_dtype = residual.dtype();

  c10::SmallVector<at::ScalarType, 6> activation_types = {
      residual.scalar_type(), x.scalar_type(), scale.scalar_type(), shift.scalar_type()};
  if (gate_opt.has_value() && gate_opt.value().defined()) {
    activation_types.push_back(gate_opt.value().scalar_type());
  }
  auto activation_scalar = activation_types.front();
  bool has_mixed_activation = false;
  for (const auto& st : activation_types) {
    if (st != activation_scalar) {
      has_mixed_activation = true;
      break;
    }
  }
  if (has_mixed_activation) {
    activation_scalar = at::ScalarType::Float;
  }
  auto act_opts = residual.options().dtype(activation_scalar);
  auto cast_activation = [&](const at::Tensor& t) {
    if (t.scalar_type() == activation_scalar) {
      return t.contiguous();
    }
    return t.to(act_opts).contiguous();
  };
  auto residual_f = cast_activation(residual);
  auto x_f = cast_activation(x);
  auto modulated = at::empty_like(residual_f);
  auto residual_output = at::empty_like(residual_f);

  auto param_scalar = at::ScalarType::Float;
  bool param_scalar_set = false;
  auto set_param_scalar = [&](const at::Tensor& t, const char* name) {
    CHECK_CUDA(t);
    if (!param_scalar_set) {
      param_scalar = t.scalar_type();
      param_scalar_set = true;
    } else {
      TORCH_CHECK(t.scalar_type() == param_scalar, name, " dtype must match other norm parameters.");
    }
  };
  if (norm_weight_opt.has_value() && norm_weight_opt.value().defined()) {
    set_param_scalar(norm_weight_opt.value(), "norm_weight");
  }
  if (norm_bias_opt.has_value() && norm_bias_opt.value().defined()) {
    set_param_scalar(norm_bias_opt.value(), "norm_bias");
  }
  auto act_opts_const = act_opts;

  bool is_warp_reduce = D <= CTA_REDUCE_THRESHOLD;
  bool is_d_aligned = D % 4 == 0 && tensor_aligned_for_vectorized_load(residual) &&
                      tensor_aligned_for_vectorized_load(x) && optional_tensor_aligned_for_vectorized_load(gate_opt) &&
                      optional_tensor_aligned_for_vectorized_load(norm_weight_opt) &&
                      optional_tensor_aligned_for_vectorized_load(norm_bias_opt) &&
                      tensor_aligned_for_vectorized_load(shift) && tensor_aligned_for_vectorized_load(scale);
  dim3 block(THREADS_PER_CTA);
  uint32_t cta_per_grid = is_warp_reduce ? (B * S + WARP_PER_CTA - 1) / WARP_PER_CTA : B * S;
  dim3 grid(dim3(cta_per_grid, 1, 1));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto param_opts = torch::TensorOptions().device(residual.device()).dtype(param_scalar);

  auto dispatch_activation = [&](auto dtype_tag) {
    using DType = decltype(dtype_tag);
    auto gate_param = prepare_gate<DType>(gate_opt, B, S, D, act_opts_const);
    bool has_gate_tensor = gate_param.frame_len != -1;
    auto shift_param = prepare_scale_shift_tensor<DType>(shift, B, S, D, act_opts_const);
    auto scale_param = prepare_scale_shift_tensor<DType>(scale, B, S, D, act_opts_const);

    auto dispatch_param = [&](auto param_tag) {
      using ParamDType = decltype(param_tag);
      auto norm_params = prepare_norm_params<ParamDType>(norm_weight_opt, norm_bias_opt, D, param_opts);
      auto launcher = DISPATCH_TABLE<DType, ParamDType>[use_rms_norm][is_d_aligned];
      launcher(
          grid,
          block,
          stream,
          residual_f.data_ptr<DType>(),
          x_f.data_ptr<DType>(),
          has_gate_tensor ? gate_param.storage.template data_ptr<DType>() : nullptr,
          norm_params.has_weight_tensor ? norm_params.weight.template data_ptr<ParamDType>() : nullptr,
          norm_params.has_bias_tensor ? norm_params.bias.template data_ptr<ParamDType>() : nullptr,
          shift_param.desc,
          scale_param.desc,
          eps,
          modulated.data_ptr<DType>(),
          residual_output.data_ptr<DType>(),
          B,
          S,
          D,
          static_cast<int>(gate_param.frame_len),
          is_warp_reduce,
          norm_params.has_weight_tensor,
          norm_params.has_bias_tensor);
    };

    switch (param_scalar) {
      case at::ScalarType::Float:
        dispatch_param(float{});
        break;
      case at::ScalarType::Half:
        dispatch_param(at::Half{});
        break;
      case at::ScalarType::BFloat16:
        dispatch_param(at::BFloat16{});
        break;
      default:
        TORCH_CHECK(false, "Unsupported parameter dtype for fused kernel.");
    }
  };

  switch (activation_scalar) {
    case at::ScalarType::Float:
      dispatch_activation(float{});
      break;
    case at::ScalarType::Half:
      dispatch_activation(at::Half{});
      break;
    case at::ScalarType::BFloat16:
      dispatch_activation(at::BFloat16{});
      break;
    default:
      TORCH_CHECK(false, "Unsupported activation dtype for fused kernel.");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

  return {modulated.to(orig_dtype), residual_output.to(orig_dtype)};
}
