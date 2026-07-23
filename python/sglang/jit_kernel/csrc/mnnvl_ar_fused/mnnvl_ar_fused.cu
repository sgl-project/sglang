// TVM-FFI binding for the mnnvl_ar_fused port. Mirrors flashinfer 0.6.12's
// csrc/trtllm_mnnvl_allreduce.cu binding (see
// baseline/upstream_ref/flashinfer/csrc_trtllm_mnnvl_allreduce.cu) with the
// kernel include swapped to the workspace-owned verbatim port. The exported
// symbol name and full signature are kept identical so both sides are driven
// through the same ABI by the harness and the serving hook.
// mnnvl_ar_fused_opt.cuh includes the verbatim mnnvl_ar_fused.cuh first; the
// verbatim header has no include guard (upstream includes it exactly once),
// so it must not be included directly here as well.
#include "mnnvl_ar_fused_opt.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_mnnvl_allreduce;

using tvm::ffi::Optional;

#define DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(dtype, c_type, ...)             \
  [&] {                                                                             \
    switch (encode_dlpack_dtype(dtype)) {                                           \
      case float32_code: {                                                          \
        using c_type = float;                                                       \
        return __VA_ARGS__();                                                       \
      }                                                                             \
      case float16_code: {                                                          \
        using c_type = half;                                                        \
        return __VA_ARGS__();                                                       \
      }                                                                             \
      case bfloat16_code: {                                                         \
        using c_type = __nv_bfloat16;                                               \
        return __VA_ARGS__();                                                       \
      }                                                                             \
      default:                                                                      \
        TVM_FFI_LOG_AND_THROW(NotImplementedError)                                  \
            << "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE."; \
    }                                                                               \
  }()

void trtllm_mnnvl_allreduce_fusion(
    TensorView input,
    int64_t multicast_buffer_ptr,
    int64_t buffer_ptrs_dev,
    int64_t buffer_ptr_local,
    TensorView buffer_flags_mnnvl,
    int64_t nranks,
    int64_t rank,
    bool rmsnorm_fusion,
    bool launch_with_pdl,
    bool use_oneshot,
    TensorView output,
    Optional<TensorView> residual_out,
    Optional<TensorView> residual_in,
    Optional<TensorView> gamma,
    Optional<double> epsilon,
    Optional<double> weight_bias) {
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(input.dtype(), c_type, [&] {
    // Extract parameters from tensors
    int64_t num_tokens = input.size(0);
    int64_t token_dim = input.size(1);

    // Validate input parameters
    TVM_FFI_ICHECK_EQ(token_dim % (sizeof(float4) / sizeof(c_type)), 0)
        << "token_dim must be divisible by " << sizeof(float4) / sizeof(c_type);
    TVM_FFI_ICHECK(output.size(0) == input.size(0) && output.size(1) == input.size(1))
        << "output shape mismatch: expected (" << input.size(0) << ", " << input.size(1) << ") but got ("
        << output.size(0) << ", " << output.size(1) << ")";
    TVM_FFI_ICHECK(nranks >= 2 && nranks <= 64) << "nranks must be between 2 and 64, got " << nranks;
    TVM_FFI_ICHECK(rank >= 0 && rank < nranks) << "rank must be between 0 and nranks-1, got " << rank;
    TVM_FFI_ICHECK(
        (residual_in.has_value() && residual_out.has_value() && gamma.has_value() && epsilon.has_value()) ||
        !rmsnorm_fusion)
        << "residual_in, residual_out, gamma, and epsilon must be provided if rmsnorm_fusion is "
           "true";

    if (rmsnorm_fusion) {
      TVM_FFI_ICHECK(residual_in.value().size(0) == num_tokens && residual_in.value().size(1) == token_dim)
          << "residual_in shape mismatch: expected (" << input.size(0) << ", " << input.size(1) << ") but got ("
          << residual_in.value().size(0) << ", " << residual_in.value().size(1) << ")";
      TVM_FFI_ICHECK(residual_out.value().size(0) == num_tokens && residual_out.value().size(1) == token_dim)
          << "residual_out shape mismatch: expected (" << input.size(0) << ", " << input.size(1) << ") but got ("
          << residual_out.value().size(0) << ", " << residual_out.value().size(1) << ")";
      TVM_FFI_ICHECK(gamma.value().size(0) == token_dim) << "gamma must have the same shape as token dimension ("
                                                         << token_dim << ") but got (" << gamma.value().size(0) << ")";
    }

    // Create the parameters struct
    AllReduceFusionParams params;

    // Aux Information
    params.nRanks = nranks;
    params.rank = rank;
    params.numTokens = num_tokens;
    params.tokenDim = token_dim;
    params.bufferPtrsDev = reinterpret_cast<void**>(buffer_ptrs_dev);
    params.bufferPtrLocal = reinterpret_cast<void*>(buffer_ptr_local);
    params.multicastPtr = reinterpret_cast<void*>(multicast_buffer_ptr);
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags_mnnvl.data_ptr());
    params.rmsNormFusion = rmsnorm_fusion;
    params.launchWithPdl = launch_with_pdl;

    // input data
    params.input = const_cast<void const*>(input.data_ptr());
    params.residualIn = residual_in.has_value() ? const_cast<void const*>(residual_in.value().data_ptr()) : nullptr;
    params.gamma = gamma.has_value() ? const_cast<void const*>(gamma.value().data_ptr()) : nullptr;
    params.epsilon = epsilon.has_value() ? epsilon.value() : 1e-5;
    params.weightBias = weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;

    // output data
    params.output = const_cast<void*>(output.data_ptr());
    params.residualOut = residual_out.has_value() ? const_cast<void*>(residual_out.value().data_ptr()) : nullptr;
    params.stream = stream;

    cudaError_t status;
    if (use_oneshot) {
      status = oneshotAllreduceFusionDispatch<c_type>(params);
    } else {
      status = twoshotAllreduceFusionDispatch<c_type>(params);
    }
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "trtllm_mnnvl_allreduce_fusion failed with error code " << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_allreduce_fusion, trtllm_mnnvl_allreduce_fusion);

// bs=1 specialized entry: identical signature/validation; routes the frozen
// shapes to the constant-instantiated kernel and everything else to the
// generic verbatim dispatch (see mnnvl_ar_fused_opt.cuh).
void mnnvl_ar_fused_opt(
    TensorView input,
    int64_t multicast_buffer_ptr,
    int64_t buffer_ptrs_dev,
    int64_t buffer_ptr_local,
    TensorView buffer_flags_mnnvl,
    int64_t nranks,
    int64_t rank,
    bool rmsnorm_fusion,
    bool launch_with_pdl,
    bool use_oneshot,
    TensorView output,
    Optional<TensorView> residual_out,
    Optional<TensorView> residual_in,
    Optional<TensorView> gamma,
    Optional<double> epsilon,
    Optional<double> weight_bias) {
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(input.dtype(), c_type, [&] {
    int64_t num_tokens = input.size(0);
    int64_t token_dim = input.size(1);
    TVM_FFI_ICHECK_EQ(token_dim % (sizeof(float4) / sizeof(c_type)), 0)
        << "token_dim must be divisible by " << sizeof(float4) / sizeof(c_type);
    TVM_FFI_ICHECK(nranks >= 2 && nranks <= 64) << "nranks must be between 2 and 64, got " << nranks;
    TVM_FFI_ICHECK(rank >= 0 && rank < nranks) << "rank must be between 0 and nranks-1, got " << rank;
    TVM_FFI_ICHECK(use_oneshot) << "specialized entry supports the oneshot path only";

    AllReduceFusionParams params;
    params.nRanks = nranks;
    params.rank = rank;
    params.numTokens = num_tokens;
    params.tokenDim = token_dim;
    params.bufferPtrsDev = reinterpret_cast<void**>(buffer_ptrs_dev);
    params.bufferPtrLocal = reinterpret_cast<void*>(buffer_ptr_local);
    params.multicastPtr = reinterpret_cast<void*>(multicast_buffer_ptr);
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags_mnnvl.data_ptr());
    params.rmsNormFusion = rmsnorm_fusion;
    params.launchWithPdl = launch_with_pdl;
    params.input = const_cast<void const*>(input.data_ptr());
    params.residualIn = residual_in.has_value() ? const_cast<void const*>(residual_in.value().data_ptr()) : nullptr;
    params.gamma = gamma.has_value() ? const_cast<void const*>(gamma.value().data_ptr()) : nullptr;
    params.epsilon = epsilon.has_value() ? epsilon.value() : 1e-5;
    params.weightBias = weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;
    params.output = const_cast<void*>(output.data_ptr());
    params.residualOut = residual_out.has_value() ? const_cast<void*>(residual_out.value().data_ptr()) : nullptr;
    params.stream = stream;

    cudaError_t status = oneshotArFusedConstDispatch<c_type>(params);
    TVM_FFI_ICHECK(status == cudaSuccess) << "mnnvl_ar_fused_opt failed with error code " << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mnnvl_ar_fused_opt, mnnvl_ar_fused_opt);

// Experimental block-arrival variant (P1 fence/flag-granularity route
// evaluation only; not a serving path).
void mnnvl_ar_fused_opt_ba(
    TensorView input,
    int64_t multicast_buffer_ptr,
    int64_t buffer_ptrs_dev,
    int64_t buffer_ptr_local,
    TensorView buffer_flags_mnnvl,
    int64_t nranks,
    int64_t rank,
    bool rmsnorm_fusion,
    bool launch_with_pdl,
    bool use_oneshot,
    TensorView output,
    Optional<TensorView> residual_out,
    Optional<TensorView> residual_in,
    Optional<TensorView> gamma,
    Optional<double> epsilon,
    Optional<double> weight_bias) {
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(input.dtype(), c_type, [&] {
    TVM_FFI_ICHECK(use_oneshot) << "variant supports the oneshot path only";
    AllReduceFusionParams params;
    params.nRanks = nranks;
    params.rank = rank;
    params.numTokens = input.size(0);
    params.tokenDim = input.size(1);
    params.bufferPtrsDev = reinterpret_cast<void**>(buffer_ptrs_dev);
    params.bufferPtrLocal = reinterpret_cast<void*>(buffer_ptr_local);
    params.multicastPtr = reinterpret_cast<void*>(multicast_buffer_ptr);
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags_mnnvl.data_ptr());
    params.rmsNormFusion = rmsnorm_fusion;
    params.launchWithPdl = launch_with_pdl;
    params.input = const_cast<void const*>(input.data_ptr());
    params.residualIn = residual_in.has_value() ? const_cast<void const*>(residual_in.value().data_ptr()) : nullptr;
    params.gamma = gamma.has_value() ? const_cast<void const*>(gamma.value().data_ptr()) : nullptr;
    params.epsilon = epsilon.has_value() ? epsilon.value() : 1e-5;
    params.weightBias = weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;
    params.output = const_cast<void*>(output.data_ptr());
    params.residualOut = residual_out.has_value() ? const_cast<void*>(residual_out.value().data_ptr()) : nullptr;
    params.stream = stream;

    cudaError_t status = oneshotArFusedConstBaDispatch<c_type>(params);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "mnnvl_ar_fused_opt_ba failed with error code " << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mnnvl_ar_fused_opt_ba, mnnvl_ar_fused_opt_ba);
