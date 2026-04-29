/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
// clang-format off
// JIT version adapted from sgl-kernel/csrc/mamba/causal_conv1d.cu, which is in turn
// adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu
// and https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_update.cu

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, RuntimeDeviceCheck

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include <cstring>

#include "causal_conv1d_common.cuh"

namespace {

// ---------------------------------------------------------------------------
// Forward kernel
// ---------------------------------------------------------------------------

template <int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct CausalConv1dFwdKernelTraits {
  using input_t = input_t_;
  using weight_t = weight_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kWidth = kWidth_;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static_assert(kWidth <= kNElts);
  static constexpr bool kIsVecLoad = kIsVecLoad_;
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
  static constexpr int kSmemIOSize = kIsVecLoad
      ? 0
      : conv_custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
  static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
  static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_fwd_kernel(ConvParamsBase params) {
  constexpr int kWidth = Ktraits::kWidth;
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;
  using weight_t = typename Ktraits::weight_t;

  extern __shared__ char smem_[];
  auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
  auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
  auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
  auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_ + Ktraits::kSmemIOSize);

  const bool kVarlen = params.query_start_loc_ptr != nullptr;
  const int tidx = threadIdx.x;
  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y;
  const int* query_start_loc = kVarlen ? reinterpret_cast<int*>(params.query_start_loc_ptr) : nullptr;
  const int sequence_start_index = kVarlen ? query_start_loc[batch_id] : batch_id;
  const int seqlen = kVarlen ? query_start_loc[batch_id + 1] - sequence_start_index : params.seqlen;

  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + sequence_start_index * params.x_batch_stride
      + channel_id * params.x_c_stride;
  weight_t* weight = reinterpret_cast<weight_t*>(params.weight_ptr) + channel_id * params.weight_c_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + sequence_start_index * params.out_batch_stride
      + channel_id * params.out_c_stride;
  float bias_val = params.bias_ptr == nullptr ? 0.f
                                              : float(reinterpret_cast<weight_t*>(params.bias_ptr)[channel_id]);

  bool has_initial_state = params.has_initial_state_ptr == nullptr
      ? false
      : reinterpret_cast<bool*>(params.has_initial_state_ptr)[batch_id];

  int* cache_indices =
      params.cache_indices_ptr == nullptr ? nullptr : reinterpret_cast<int*>(params.cache_indices_ptr);
  int cache_index = cache_indices == nullptr ? batch_id : cache_indices[batch_id];
  if (cache_index == params.pad_slot_id) {
    return;
  }
  input_t* conv_states = params.conv_states_ptr == nullptr
      ? nullptr
      : reinterpret_cast<input_t*>(params.conv_states_ptr) + cache_index * params.conv_states_batch_stride
          + channel_id * params.conv_states_c_stride;

  if (tidx == 0) {
    input_t initial_state[kNElts] = {0};
    if (has_initial_state) {
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        initial_state[kNElts - 1 - (kWidth - 2) + w] = conv_states[w];
      }
    }
    smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t*>(initial_state)[0];
  }

  float weight_vals[kWidth];
#pragma unroll
  for (int i = 0; i < kWidth; ++i) {
    weight_vals[i] = float(weight[i * params.weight_width_stride]);
  }

  constexpr int kChunkSize = kNThreads * kNElts;
  const int n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    input_t x_vals_load[2 * kNElts] = {0};
    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockLoadVecT(smem_load_vec)
          .Load(
              reinterpret_cast<vec_t*>(x),
              *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]),
              (seqlen - chunk * kChunkSize) / kNElts);
    } else {
      __syncthreads();
      typename Ktraits::BlockLoadT(smem_load)
          .Load(
              x,
              *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]),
              seqlen - chunk * kChunkSize);
    }
    x += kChunkSize;
    __syncthreads();
    if (tidx < kNThreads - 1) {
      smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
    }
    __syncthreads();
    reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
    __syncthreads();
    if (tidx == kNThreads - 1) {
      smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
    }

    float x_vals[2 * kNElts];
#pragma unroll
    for (int i = 0; i < 2 * kNElts; ++i) {
      x_vals[i] = float(x_vals_load[i]);
    }

    float out_vals[kNElts];
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      out_vals[i] = bias_val;
#pragma unroll
      for (int w = 0; w < kWidth; ++w) {
        out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
      }
    }

    if (params.silu_activation) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
      }
    }

    input_t out_vals_store[kNElts];
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      out_vals_store[i] = out_vals[i];
    }
    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockStoreVecT(smem_store_vec)
          .Store(
              reinterpret_cast<vec_t*>(out),
              reinterpret_cast<vec_t (&)[1]>(out_vals_store),
              (seqlen - chunk * kChunkSize) / kNElts);
    } else {
      typename Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, seqlen - chunk * kChunkSize);
    }
    out += kChunkSize;

    int final_state_position = ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize);
    if (conv_states != nullptr && final_state_position < 0 && seqlen > kWidth) {
      input_t vals_load[kNElts] = {0};
      if ((chunk == n_chunks - 2) && (tidx == kNThreads - 1)) {
        reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[kNThreads - 1];
#pragma unroll
        for (int w = 0; w < -final_state_position; ++w) {
          conv_states[w] = vals_load[kNElts + final_state_position + w];
        }
      }
      if ((chunk == n_chunks - 1) && tidx == 0) {
        reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[0];
        for (int w = -final_state_position; w < kWidth - 1; ++w) {
          conv_states[w] = vals_load[w + final_state_position];
        }
        return;
      }
    }
  }
  int last_thread = ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize) / kNElts;
  if (conv_states != nullptr && tidx == last_thread) {
    input_t x_vals_load[kNElts * 2] = {0};
    if (last_thread == 0 && seqlen < kWidth) {
      reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[0];
      const int offset = seqlen - (kWidth - 1);
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        if ((w - seqlen) >= 0 && has_initial_state) {
          conv_states[w - seqlen] = conv_states[w];
        } else if ((w - seqlen) >= 0 && !has_initial_state) {
          conv_states[w - seqlen] = input_t(0.0f);
        }
      }
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        if (offset + w >= 0) conv_states[w] = x_vals_load[offset + w];
      }
    } else {
      const int offset = ((seqlen - (kWidth - 1)) % (kNElts));
      if ((offset + kWidth - 2) >= kNElts && (last_thread + 1 < kNThreads)) {
        reinterpret_cast<vec_t*>(x_vals_load)[1] = smem_exchange[last_thread + 1];
      }
      reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[last_thread];
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        conv_states[w] = x_vals_load[offset + w];
      }
    }
  }
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
inline void causal_conv1d_fwd_launch(ConvParamsBase& params, cudaStream_t stream) {
  static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
  const bool kVarlen = params.query_start_loc_ptr != nullptr;
  CONV_BOOL_SWITCH(params.seqlen % kNElts == 0 && !kVarlen, kIsVecLoad, [&] {
    using Ktraits = CausalConv1dFwdKernelTraits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch, params.dim);

    auto kernel = &causal_conv1d_fwd_kernel<Ktraits>;

    if (kSmemSize >= 48 * 1024) {
#ifndef USE_ROCM
      host::RuntimeDeviceCheck(
          ::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
      host::RuntimeDeviceCheck(
          ::cudaFuncSetAttribute((void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#endif
    }
    host::LaunchKernel(grid, dim3(Ktraits::kNThreads), stream, kSmemSize)(kernel, params);
    host::RuntimeDeviceCheck();
  });
}

template <typename input_t>
inline void causal_conv1d_fwd_cuda(ConvParamsBase& params, cudaStream_t stream) {
  using weight_t = input_t;
  if (params.width == 2) {
    causal_conv1d_fwd_launch<128, 2, input_t, weight_t>(params, stream);
  } else if (params.width == 3) {
    causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream);
  } else if (params.width == 4) {
    causal_conv1d_fwd_launch<128, 4, input_t, weight_t>(params, stream);
  }
}

// ---------------------------------------------------------------------------
// Update kernel
// ---------------------------------------------------------------------------

template <int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct CausalConv1dUpdateKernelTraits {
  using input_t = input_t_;
  using weight_t = weight_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kWidth = kWidth_;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
};

template <typename Ktraits, bool kIsCircularBuffer>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_update_kernel(ConvParamsBase params) {
  constexpr int kWidth = Ktraits::kWidth;
  constexpr int kNThreads = Ktraits::kNThreads;
  using input_t = typename Ktraits::input_t;
  using weight_t = typename Ktraits::weight_t;

  const int tidx = threadIdx.x;
  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y * kNThreads + tidx;
  if (channel_id >= params.dim) return;

  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride
      + channel_id * params.x_c_stride;

  const int conv_state_batch_coord =
      params.conv_state_indices_ptr == nullptr ? batch_id : params.conv_state_indices_ptr[batch_id];
  if (conv_state_batch_coord == params.pad_slot_id) {
    return;
  }
  input_t* conv_state = reinterpret_cast<input_t*>(params.conv_state_ptr)
      + conv_state_batch_coord * params.conv_state_batch_stride
      + channel_id * params.conv_state_c_stride;

  weight_t* weight = reinterpret_cast<weight_t*>(params.weight_ptr) + channel_id * params.weight_c_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride
      + channel_id * params.out_c_stride;
  float bias_val = params.bias_ptr == nullptr ? 0.f
                                              : float(reinterpret_cast<weight_t*>(params.bias_ptr)[channel_id]);

  int state_len = params.conv_state_len;
  int advance_len = params.seqlen;
  int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
  int update_idx = cache_seqlen - (kWidth - 1);
  update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

  float weight_vals[kWidth] = {0};
#pragma unroll
  for (int i = 0; i < kWidth; ++i) {
    weight_vals[i] = float(weight[i * params.weight_width_stride]);
  }

  float x_vals[kWidth] = {0};
  if constexpr (!kIsCircularBuffer) {
#pragma unroll 2
    for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
      conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
    }
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i) {
      input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
      if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
        conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
      }
      x_vals[i] = float(state_val);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kWidth - 1;
         ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1) {
      input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
      x_vals[i] = float(state_val);
    }
  }
#pragma unroll 2
  for (int i = 0; i < params.seqlen; ++i) {
    input_t x_val = x[i * params.x_l_stride];
    if constexpr (!kIsCircularBuffer) {
      if (i < advance_len && state_len - advance_len + i >= 0) {
        conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
      }
    } else {
      conv_state[update_idx * params.conv_state_l_stride] = x_val;
      ++update_idx;
      update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
    }
    x_vals[kWidth - 1] = float(x_val);
    float out_val = bias_val;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      out_val += weight_vals[j] * x_vals[j];
    }
    if (params.silu_activation) {
      out_val = out_val / (1 + expf(-out_val));
    }
    out[i * params.out_l_stride] = input_t(out_val);
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i) {
      x_vals[i] = x_vals[i + 1];
    }
  }
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
inline void causal_conv1d_update_launch(ConvParamsBase& params, cudaStream_t stream) {
  using Ktraits = CausalConv1dUpdateKernelTraits<kNThreads, kWidth, input_t, weight_t>;
  dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
  auto kernel = params.cache_seqlens == nullptr
      ? &causal_conv1d_update_kernel<Ktraits, false>
      : &causal_conv1d_update_kernel<Ktraits, true>;
  host::LaunchKernel(grid, dim3(Ktraits::kNThreads), stream)(kernel, params);
  host::RuntimeDeviceCheck();
}

template <typename input_t>
inline void causal_conv1d_update_cuda(ConvParamsBase& params, cudaStream_t stream) {
  using weight_t = input_t;
  if (params.width == 2) {
    causal_conv1d_update_launch<64, 2, input_t, weight_t>(params, stream);
  } else if (params.width == 3) {
    causal_conv1d_update_launch<64, 3, input_t, weight_t>(params, stream);
  } else if (params.width == 4) {
    causal_conv1d_update_launch<64, 4, input_t, weight_t>(params, stream);
  }
}

// ---------------------------------------------------------------------------
// Host-side entry points (TVM-FFI)
// ---------------------------------------------------------------------------

inline void set_conv_params_fwd(
    ConvParamsBase& params,
    int64_t batch,
    int64_t dim,
    int64_t seqlen,
    int64_t width,
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& out,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias,
    bool silu_activation,
    int64_t pad_slot_id,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& query_start_loc,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& cache_indices,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& has_initial_state) {
  std::memset(&params, 0, sizeof(params));

  params.batch = static_cast<int>(batch);
  params.dim = static_cast<int>(dim);
  params.seqlen = static_cast<int>(seqlen);
  params.width = static_cast<int>(width);
  params.pad_slot_id = pad_slot_id;
  params.silu_activation = silu_activation;

  params.x_ptr = const_cast<void*>(x.data_ptr());
  params.weight_ptr = const_cast<void*>(weight.data_ptr());
  params.bias_ptr = bias.has_value() ? const_cast<void*>(bias.value().data_ptr()) : nullptr;
  params.out_ptr = const_cast<void*>(out.data_ptr());

  params.query_start_loc_ptr =
      query_start_loc.has_value() ? const_cast<void*>(query_start_loc.value().data_ptr()) : nullptr;
  params.cache_indices_ptr =
      cache_indices.has_value() ? const_cast<void*>(cache_indices.value().data_ptr()) : nullptr;
  params.has_initial_state_ptr =
      has_initial_state.has_value() ? const_cast<void*>(has_initial_state.value().data_ptr()) : nullptr;

  const bool varlen = params.query_start_loc_ptr != nullptr;
  params.x_batch_stride = static_cast<uint32_t>(x.stride(varlen ? 1 : 0));
  params.x_c_stride = static_cast<uint32_t>(x.stride(varlen ? 0 : 1));
  // Innermost stride along the seqlen axis: stride(1) for varlen 2D, stride(2) for non-varlen 3D.
  params.x_l_stride = static_cast<uint32_t>(varlen ? x.stride(1) : x.stride(2));
  params.weight_c_stride = static_cast<uint32_t>(weight.stride(0));
  params.weight_width_stride = static_cast<uint32_t>(weight.stride(1));
  params.out_batch_stride = static_cast<uint32_t>(out.stride(varlen ? 1 : 0));
  params.out_c_stride = static_cast<uint32_t>(out.stride(varlen ? 0 : 1));
  params.out_l_stride = static_cast<uint32_t>(varlen ? out.stride(1) : out.stride(2));
}

template <typename DType>
void run_causal_conv1d_fwd(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    tvm::ffi::Optional<tvm::ffi::TensorView> conv_states,
    tvm::ffi::Optional<tvm::ffi::TensorView> query_start_loc,
    tvm::ffi::Optional<tvm::ffi::TensorView> cache_indices,
    tvm::ffi::Optional<tvm::ffi::TensorView> has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id) {
  using namespace host;

  const bool varlen = query_start_loc.has_value();
  const int64_t batch_size = varlen ? query_start_loc.value().size(0) - 1 : x.size(0);
  const int64_t dim = varlen ? x.size(0) : x.size(1);
  const int64_t seqlen = varlen ? x.size(1) : x.size(2);
  const int64_t width = weight.size(-1);

  RuntimeCheck(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");
  RuntimeCheck(weight.size(0) == dim, "weight first dim must match input channels");

  ConvParamsBase params;
  set_conv_params_fwd(
      params,
      batch_size,
      dim,
      seqlen,
      width,
      x,
      weight,
      x,  // out aliases x (in-place)
      bias,
      silu_activation,
      pad_slot_id,
      query_start_loc,
      cache_indices,
      has_initial_state);

  if (conv_states.has_value()) {
    auto cs = conv_states.value();
    params.conv_states_ptr = const_cast<void*>(cs.data_ptr());
    params.conv_states_batch_stride = static_cast<uint32_t>(cs.stride(0));
    params.conv_states_c_stride = static_cast<uint32_t>(cs.stride(-2));
    params.conv_states_l_stride = static_cast<uint32_t>(cs.stride(-1));
  } else {
    params.conv_states_ptr = nullptr;
  }

  cudaStream_t stream = LaunchKernel::resolve_device(x.device());
  causal_conv1d_fwd_cuda<DType>(params, stream);
}

template <typename DType>
void run_causal_conv1d_update(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView conv_state,
    tvm::ffi::TensorView weight,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    bool silu_activation,
    tvm::ffi::Optional<tvm::ffi::TensorView> cache_seqlens,
    tvm::ffi::Optional<tvm::ffi::TensorView> conv_state_indices,
    int64_t pad_slot_id) {
  using namespace host;

  const int64_t batch_size = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t seqlen = x.size(2);
  const int64_t width = weight.size(-1);
  const int64_t conv_state_len = conv_state.size(2);

  RuntimeCheck(conv_state_len >= width - 1, "conv_state_len must be >= width - 1");
  RuntimeCheck(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");
  RuntimeCheck(weight.size(0) == dim, "weight first dim must match input channels");

  ConvParamsBase params;
  set_conv_params_fwd(
      params,
      batch_size,
      dim,
      seqlen,
      width,
      x,
      weight,
      x,  // out aliases x (in-place)
      bias,
      silu_activation,
      pad_slot_id,
      /*query_start_loc=*/{},
      /*cache_indices=*/{},
      /*has_initial_state=*/{});

  params.conv_state_ptr = const_cast<void*>(conv_state.data_ptr());
  params.conv_state_len = static_cast<int>(conv_state_len);
  params.conv_state_batch_stride = static_cast<uint32_t>(conv_state.stride(0));
  params.conv_state_c_stride = static_cast<uint32_t>(conv_state.stride(1));
  params.conv_state_l_stride = static_cast<uint32_t>(conv_state.stride(2));

  if (cache_seqlens.has_value()) {
    params.cache_seqlens = static_cast<int32_t*>(const_cast<void*>(cache_seqlens.value().data_ptr()));
  } else {
    params.cache_seqlens = nullptr;
  }

  if (conv_state_indices.has_value()) {
    params.conv_state_indices_ptr =
        static_cast<int32_t*>(const_cast<void*>(conv_state_indices.value().data_ptr()));
  } else {
    params.conv_state_indices_ptr = nullptr;
  }

  cudaStream_t stream = LaunchKernel::resolve_device(x.device());
  causal_conv1d_update_cuda<DType>(params, stream);
}

template <typename DType>
struct CausalConv1dFwdKernel {
  static void run(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView weight,
      tvm::ffi::Optional<tvm::ffi::TensorView> bias,
      tvm::ffi::Optional<tvm::ffi::TensorView> conv_states,
      tvm::ffi::Optional<tvm::ffi::TensorView> query_start_loc,
      tvm::ffi::Optional<tvm::ffi::TensorView> cache_indices,
      tvm::ffi::Optional<tvm::ffi::TensorView> has_initial_state,
      bool silu_activation,
      int64_t pad_slot_id) {
    run_causal_conv1d_fwd<DType>(
        x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state, silu_activation, pad_slot_id);
  }
};

template <typename DType>
struct CausalConv1dUpdateKernel {
  static void run(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView conv_state,
      tvm::ffi::TensorView weight,
      tvm::ffi::Optional<tvm::ffi::TensorView> bias,
      bool silu_activation,
      tvm::ffi::Optional<tvm::ffi::TensorView> cache_seqlens,
      tvm::ffi::Optional<tvm::ffi::TensorView> conv_state_indices,
      int64_t pad_slot_id) {
    run_causal_conv1d_update<DType>(
        x, conv_state, weight, bias, silu_activation, cache_seqlens, conv_state_indices, pad_slot_id);
  }
};

}  // namespace
