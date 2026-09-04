/// \file causal_conv1d.cuh
/// \brief Depthwise causal conv1d: prefill (`causal_conv1d_fwd`) and decode
///        (`causal_conv1d_update`).
///
/// Adapted from
/// https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu
/// and
/// https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_update.cu
///
/// The device kernels are carried over unchanged from the AOT implementation, so
/// results stay bit-identical; only the host launchers are adapted to the tvm-ffi
/// `TensorView` API.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil

#include <sgl_kernel/type.cuh>   // For DTypeTrait, fp16_t / bf16_t / fp32_t
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

/// Nested so these generic names cannot collide with the unrelated
/// `causal_conv1d` of the Inkling short-conv kernels.
namespace mamba_conv {

/// \brief Runtime parameters shared by the prefill and decode kernels.
///
/// The subset of the AOT `ConvParamsBase` these kernels read. Strides are in
/// elements, not bytes, and stay `uint32_t` as they were there.
struct MambaConvParams {
  using index_t = uint32_t;

  int32_t batch;
  int32_t dim;
  int32_t seqlen;
  int32_t width;
  int64_t pad_slot_id;
  bool silu_activation;

  index_t x_batch_stride;
  index_t x_c_stride;
  index_t x_l_stride;
  index_t weight_c_stride;
  index_t weight_width_stride;
  index_t out_batch_stride;
  index_t out_c_stride;
  index_t out_l_stride;

  int32_t conv_state_len;
  index_t conv_state_batch_stride;
  index_t conv_state_c_stride;
  index_t conv_state_l_stride;

  // Common data pointers.
  void* __restrict__ x_ptr;
  void* __restrict__ weight_ptr;
  void* __restrict__ bias_ptr;
  void* __restrict__ out_ptr;

  void* __restrict__ conv_state_ptr;
  void* __restrict__ query_start_loc_ptr;
  void* __restrict__ has_initial_state_ptr;
  void* __restrict__ cache_indices_ptr;
  const int32_t* __restrict__ cache_seqlens;

  // For the continuous batching case: the conv state for the current batch does
  // not need to be a contiguous tensor.
  const int32_t* __restrict__ conv_state_indices_ptr;

  void* conv_states_ptr;
  index_t conv_states_batch_stride;
  index_t conv_states_c_stride;
  index_t conv_states_l_stride;
};

/// \brief The unsigned integer type occupying exactly `kBytes` bytes.
template <int kBytes>
struct BytesToType {};

template <>
struct BytesToType<16> {
  using Type = uint4;
};
template <>
struct BytesToType<8> {
  using Type = uint64_t;
};
template <>
struct BytesToType<4> {
  using Type = uint32_t;
};
template <>
struct BytesToType<2> {
  using Type = uint16_t;
};

/// \brief Convert a stored element to fp32 through the dtype-specific intrinsic.
template <typename T>
SGL_DEVICE fp32_t to_float(const T& value) {
  return DTypeTrait<fp32_t>::from(value);
}

/// \brief Convert an fp32 accumulator back to the stored element type.
template <typename T>
SGL_DEVICE T from_float(fp32_t value) {
  return DTypeTrait<T>::from(value);
}

/// \brief Zero every element of a stored-element array.
template <typename T, int kN>
SGL_DEVICE void zero_fill(T (&values)[kN]) {
#pragma unroll
  for (int i = 0; i < kN; ++i) {
    values[i] = from_float<T>(0.0f);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Prefill
////////////////////////////////////////////////////////////////////////////////

/// \brief Compile-time configuration of the prefill kernel.
///
/// \tparam kNThreads_  Threads per CTA.
/// \tparam kWidth_     Convolution width (2..4).
/// \tparam kIsVecLoad_ Whether the chunk can be loaded/stored as whole vectors.
/// \tparam T           Element type: fp16_t | bf16_t | fp32_t.
template <int kNThreads_, int kWidth_, bool kIsVecLoad_, typename T>
struct CausalConv1dFwdTraits {
  using input_t = T;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kWidth = kWidth_;
  static constexpr int kNBytes = sizeof(T);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static_assert(kWidth <= kNElts);
  static constexpr bool kIsVecLoad = kIsVecLoad_;
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  using BlockLoadT = cub::BlockLoad<T, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::BlockStore<T, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
  static constexpr int kSmemIOSize =
      kIsVecLoad ? 0
                 : static_cast<int>(
                       std::max(sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)));
  static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
  static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

/// \brief One CTA per (sequence, channel): convolve a channel over its sequence
///        and write back the trailing `kWidth - 1` taps as the new conv state.
template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_fwd_kernel(MambaConvParams params) {
  constexpr int kWidth = Ktraits::kWidth;
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  // Shared memory.
  extern __shared__ char smem_[];
  auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
  auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
  auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
  auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_ + Ktraits::kSmemIOSize);

  const bool kVarlen = params.query_start_loc_ptr != nullptr;
  const int32_t tidx = threadIdx.x;
  const int32_t batch_id = blockIdx.x;
  const int32_t channel_id = blockIdx.y;
  const int32_t* query_start_loc = kVarlen ? reinterpret_cast<const int32_t*>(params.query_start_loc_ptr) : nullptr;
  const int32_t sequence_start_index = kVarlen ? query_start_loc[batch_id] : batch_id;
  const int32_t seqlen = kVarlen ? query_start_loc[batch_id + 1] - sequence_start_index : params.seqlen;

  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + sequence_start_index * params.x_batch_stride +
               channel_id * params.x_c_stride;
  const input_t* weight = reinterpret_cast<const input_t*>(params.weight_ptr) + channel_id * params.weight_c_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + sequence_start_index * params.out_batch_stride +
                 channel_id * params.out_c_stride;
  const float bias_val =
      params.bias_ptr == nullptr ? 0.f : to_float(reinterpret_cast<const input_t*>(params.bias_ptr)[channel_id]);

  const bool has_initial_state = params.has_initial_state_ptr == nullptr
                                     ? false
                                     : reinterpret_cast<const bool*>(params.has_initial_state_ptr)[batch_id];

  const int32_t* cache_indices =
      params.cache_indices_ptr == nullptr ? nullptr : reinterpret_cast<const int32_t*>(params.cache_indices_ptr);
  const int32_t cache_index = cache_indices == nullptr ? batch_id : cache_indices[batch_id];
  // cache_index == params.pad_slot_id is defined as padding, so we exit early.
  if (cache_index == params.pad_slot_id) {
    return;
  }
  input_t* conv_states = params.conv_states_ptr == nullptr ? nullptr
                                                           : reinterpret_cast<input_t*>(params.conv_states_ptr) +
                                                                 cache_index * params.conv_states_batch_stride +
                                                                 channel_id * params.conv_states_c_stride;

  // Thread 0 will load the last elements of the previous chunk, so we initialize those to 0.
  if (tidx == 0) {
    input_t initial_state[kNElts];
    zero_fill(initial_state);
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
    weight_vals[i] = to_float(weight[i * params.weight_width_stride]);
  }

  constexpr int kChunkSize = kNThreads * kNElts;
  const int32_t n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
  for (int32_t chunk = 0; chunk < n_chunks; ++chunk) {
    input_t x_vals_load[2 * kNElts];
    zero_fill(x_vals_load);
    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockLoadVecT(smem_load_vec)
          .Load(
              reinterpret_cast<vec_t*>(x),
              *reinterpret_cast<vec_t(*)[1]>(&x_vals_load[kNElts]),
              (seqlen - chunk * kChunkSize) / kNElts);
    } else {
      __syncthreads();
      typename Ktraits::BlockLoadT(smem_load).Load(
          x, *reinterpret_cast<input_t(*)[kNElts]>(&x_vals_load[kNElts]), seqlen - chunk * kChunkSize);
    }
    x += kChunkSize;
    __syncthreads();
    // Thread kNThreads - 1 doesn't write yet, so that thread 0 can read
    // the last elements of the previous chunk.
    if (tidx < kNThreads - 1) {
      smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
    }
    __syncthreads();
    reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
    __syncthreads();
    // Now thread kNThreads - 1 can write the last elements of the current chunk.
    if (tidx == kNThreads - 1) {
      smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
    }

    float x_vals[2 * kNElts];
#pragma unroll
    for (int i = 0; i < 2 * kNElts; ++i) {
      x_vals[i] = to_float(x_vals_load[i]);
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
      out_vals_store[i] = from_float<input_t>(out_vals[i]);
    }
    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockStoreVecT(smem_store_vec)
          .Store(
              reinterpret_cast<vec_t*>(out),
              reinterpret_cast<vec_t(&)[1]>(out_vals_store),
              (seqlen - chunk * kChunkSize) / kNElts);
    } else {
      typename Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, seqlen - chunk * kChunkSize);
    }
    out += kChunkSize;

    const int32_t final_state_position = ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize);
    // In case the final state is separated between the last "smem_exchange" and
    // the one before it (chunk = n_chunks - 1 and chunk = n_chunks - 2),
    // (which occurs when `final_state_position` is a non-positive index)
    // we load the correct data from smem_exchange from both chunks, the last
    // chunk iteration and the one before it.
    if (conv_states != nullptr && final_state_position < 0 && seqlen > kWidth) {
      input_t vals_load[kNElts];
      zero_fill(vals_load);
      if ((chunk == n_chunks - 2) && (tidx == kNThreads - 1)) {
        // chunk = n_chunks - 2, a segment of the final state sits in the last index
        reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[kNThreads - 1];
#pragma unroll
        for (int w = 0; w < -final_state_position; ++w) {
          conv_states[w] = vals_load[kNElts + final_state_position + w];
        }
      }
      if ((chunk == n_chunks - 1) && tidx == 0) {
        // chunk = n_chunks - 1, the second segment of the final state first positions
        reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[0];
        for (int w = -final_state_position; w < kWidth - 1; ++w) {
          conv_states[w] = vals_load[w + final_state_position];
        }
        return;
      }
    }
  }
  // Final state is stored in the smem_exchange last token slot,
  // in case seqlen < kWidth, we would need to take the final state from the
  // initial state which is stored in conv_states
  // in case seqlen > kWidth, we would need to load the last kWidth - 1 data
  // and load it into conv_state accordingly
  const int32_t last_thread = ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize) / kNElts;
  if (conv_states != nullptr && tidx == last_thread) {
    input_t x_vals_load[kNElts * 2];
    zero_fill(x_vals_load);
    // in case we are on the first kWidth tokens
    if (last_thread == 0 && seqlen < kWidth) {
      // Need to take the initial state
      reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[0];
      const int32_t offset = seqlen - (kWidth - 1);
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        // pad the existing state
        if ((w - seqlen) >= 0 && has_initial_state) {
          conv_states[w - seqlen] = conv_states[w];
        } else if ((w - seqlen) >= 0 && !has_initial_state) {
          conv_states[w - seqlen] = from_float<input_t>(0.0f);
        }
      }
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        if (offset + w >= 0) {
          conv_states[w] = x_vals_load[offset + w];
        }
      }
    } else {
      // in case the final state is in between the threads data
      const int32_t offset = ((seqlen - (kWidth - 1)) % (kNElts));
      if ((offset + kWidth - 2) >= kNElts && (last_thread + 1 < kNThreads)) {
        // In case last_thread == kNThreads - 1, accessing last_thread + 1 will result in an
        // illegal access error on H100.
        // Therefore, we access last_thread + 1 only if the final state data sits there.
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

template <int kNThreads, int kWidth, typename T>
void causal_conv1d_fwd_launch(const MambaConvParams& params, DLDevice device) {
  static constexpr int kNElts = sizeof(T) == 4 ? 4 : 8;
  const bool is_varlen = params.query_start_loc_ptr != nullptr;
  const bool is_vec_load = params.seqlen % kNElts == 0 && !is_varlen;

  const dim3 grid(params.batch, params.dim);
  if (is_vec_load) {
    using Ktraits = CausalConv1dFwdTraits<kNThreads, kWidth, true, T>;
    // The AOT launcher raised the dynamic-smem cap past 48 KB; these traits stay
    // near 4 KB, so the default limit is enough.
    static_assert(Ktraits::kSmemSize < 48 * 1024);
    host::LaunchKernel(grid, kNThreads, device, Ktraits::kSmemSize)(causal_conv1d_fwd_kernel<Ktraits>, params);
  } else {
    using Ktraits = CausalConv1dFwdTraits<kNThreads, kWidth, false, T>;
    static_assert(Ktraits::kSmemSize < 48 * 1024);
    host::LaunchKernel(grid, kNThreads, device, Ktraits::kSmemSize)(causal_conv1d_fwd_kernel<Ktraits>, params);
  }
}

template <typename T>
void causal_conv1d_fwd_cuda(const MambaConvParams& params, DLDevice device) {
  switch (params.width) {
    case 2:
      return causal_conv1d_fwd_launch<128, 2, T>(params, device);
    case 3:
      return causal_conv1d_fwd_launch<128, 3, T>(params, device);
    case 4:
      return causal_conv1d_fwd_launch<128, 4, T>(params, device);
    default:
      host::Panic("causal_conv1d_fwd: width must be between 2 and 4, got ", params.width);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Decode
////////////////////////////////////////////////////////////////////////////////

/// \brief One thread per (sequence, channel): advance the conv state by
///        `seqlen` tokens and emit the convolution over the sliding window.
template <int kNThreads, int kWidth, bool kIsCircularBuffer, typename T>
__global__ __launch_bounds__(kNThreads) void causal_conv1d_update_kernel(MambaConvParams params) {
  using input_t = T;

  const int32_t tidx = threadIdx.x;
  const int32_t batch_id = blockIdx.x;
  const int32_t channel_id = blockIdx.y * kNThreads + tidx;
  if (channel_id >= params.dim) return;

  const input_t* x = reinterpret_cast<const input_t*>(params.x_ptr) + batch_id * params.x_batch_stride +
                     channel_id * params.x_c_stride;

  // If params.conv_state_indices_ptr is set, the conv state is gathered from the conv state
  // tensor along the batch axis. Otherwise, the conv state coordinate is the same as the batch id.
  const int32_t conv_state_batch_coord =
      params.conv_state_indices_ptr == nullptr ? batch_id : params.conv_state_indices_ptr[batch_id];
  // conv_state_batch_coord == params.pad_slot_id is defined as padding so we exit early.
  if (conv_state_batch_coord == params.pad_slot_id) {
    return;
  }
  input_t* conv_state = reinterpret_cast<input_t*>(params.conv_state_ptr) +
                        conv_state_batch_coord * params.conv_state_batch_stride +
                        channel_id * params.conv_state_c_stride;

  const input_t* weight = reinterpret_cast<const input_t*>(params.weight_ptr) + channel_id * params.weight_c_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride +
                 channel_id * params.out_c_stride;
  const float bias_val =
      params.bias_ptr == nullptr ? 0.f : to_float(reinterpret_cast<const input_t*>(params.bias_ptr)[channel_id]);

  const int32_t state_len = params.conv_state_len;
  const int32_t advance_len = params.seqlen;
  const int32_t cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
  int32_t update_idx = cache_seqlen - (kWidth - 1);
  update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

  float weight_vals[kWidth] = {0};
#pragma unroll
  for (int i = 0; i < kWidth; ++i) {
    weight_vals[i] = to_float(weight[i * params.weight_width_stride]);
  }

  float x_vals[kWidth] = {0};
  if constexpr (!kIsCircularBuffer) {
#pragma unroll 2
    for (int32_t i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
      conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
    }
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i) {
      const input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
      if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
        conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
      }
      x_vals[i] = to_float(state_val);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kWidth - 1;
         ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1) {
      const input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
      x_vals[i] = to_float(state_val);
    }
  }
#pragma unroll 2
  for (int32_t i = 0; i < params.seqlen; ++i) {
    const input_t x_val = x[i * params.x_l_stride];
    if constexpr (!kIsCircularBuffer) {
      if (i < advance_len && state_len - advance_len + i >= 0) {
        conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
      }
    } else {
      conv_state[update_idx * params.conv_state_l_stride] = x_val;
      ++update_idx;
      update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
    }
    x_vals[kWidth - 1] = to_float(x_val);
    float out_val = bias_val;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      out_val += weight_vals[j] * x_vals[j];
    }
    if (params.silu_activation) {
      out_val = out_val / (1 + expf(-out_val));
    }
    out[i * params.out_l_stride] = from_float<input_t>(out_val);
    // Shift the input buffer by 1
#pragma unroll
    for (int k = 0; k < kWidth - 1; ++k) {
      x_vals[k] = x_vals[k + 1];
    }
  }
}

template <int kNThreads, int kWidth, typename T>
void causal_conv1d_update_launch(const MambaConvParams& params, DLDevice device) {
  const dim3 grid(params.batch, host::div_ceil(params.dim, kNThreads));
  if (params.cache_seqlens == nullptr) {
    host::LaunchKernel(grid, kNThreads, device)(causal_conv1d_update_kernel<kNThreads, kWidth, false, T>, params);
  } else {
    host::LaunchKernel(grid, kNThreads, device)(causal_conv1d_update_kernel<kNThreads, kWidth, true, T>, params);
  }
}

template <typename T>
void causal_conv1d_update_cuda(const MambaConvParams& params, DLDevice device) {
  switch (params.width) {
    case 2:
      return causal_conv1d_update_launch<64, 2, T>(params, device);
    case 3:
      return causal_conv1d_update_launch<64, 3, T>(params, device);
    case 4:
      return causal_conv1d_update_launch<64, 4, T>(params, device);
    default:
      host::Panic("causal_conv1d_update: width must be between 2 and 4, got ", params.width);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host entry points
////////////////////////////////////////////////////////////////////////////////

/// \brief Depthwise causal conv1d over whole sequences, in place on `x`.
///
/// \tparam T                 Element type: fp16_t | bf16_t | fp32_t.
/// \param x                  `(batch, dim, seqlen)`, or `(dim, cu_seqlen)` when
///                           `query_start_loc` is given. Overwritten with the output.
/// \param weight             `(dim, width)`, `width` in 2..4.
/// \param bias               Optional `(dim,)`.
/// \param conv_states        Optional `(num_slots, dim, state_len)`; the trailing
///                           `width - 1` taps of each sequence are written back.
/// \param query_start_loc    Optional int32 `(batch + 1,)` varlen cumulative lengths.
/// \param cache_indices      Optional int32 `(batch,)` conv-state slot per sequence.
/// \param has_initial_state  Optional bool `(batch,)`; whether to seed from `conv_states`.
/// \param silu_activation    Whether to apply SiLU to the output.
/// \param pad_slot_id        Sequences whose cache index equals this are skipped.
template <typename T>
void causal_conv1d_fwd(
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
  auto batch_sym = SymbolicSize{"batch"};
  auto dim_sym = SymbolicSize{"dim"};
  auto seqlen_sym = SymbolicSize{"seqlen"};
  auto width_sym = SymbolicSize{"width"};
  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLCUDA>();

  // Only the innermost stride is pinned: the Python wrapper makes `x` unit-stride
  // in its last dimension, everything else is carried through as a stride.
  if (varlen) {
    TensorMatcher({dim_sym, seqlen_sym}).with_strides({-1, 1}).with_dtype<T>().with_device(device_sym).verify(x);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_sym).verify(query_start_loc.value());
    batch_sym.set_value(query_start_loc.value().size(0) - 1);
  } else {
    TensorMatcher({batch_sym, dim_sym, seqlen_sym})
        .with_strides({-1, -1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .verify(x);
  }
  TensorMatcher({dim_sym, width_sym}).with_strides({-1, -1}).with_dtype<T>().with_device(device_sym).verify(weight);

  const int64_t batch = batch_sym.unwrap();
  const int64_t dim = dim_sym.unwrap();
  const int64_t seqlen = seqlen_sym.unwrap();
  const int64_t width = width_sym.unwrap();
  CHECK_HOST(batch > 0) << "causal_conv1d_fwd: batch must be positive, got " << batch;
  CHECK_HOST(width >= 2 && width <= 4) << "causal_conv1d only supports width between 2 and 4, got " << width;

  if (bias.has_value()) {
    TensorMatcher({dim_sym}).with_strides({1}).with_dtype<T>().with_device(device_sym).verify(bias.value());
  }
  if (cache_indices.has_value()) {
    TensorMatcher({batch_sym}).with_dtype<int32_t>().with_device(device_sym).verify(cache_indices.value());
  }
  if (has_initial_state.has_value()) {
    const auto& initial_state_mask = has_initial_state.value();
    TensorMatcher({batch_sym}).with_device(device_sym).verify(initial_state_mask);
    // Read as `const bool*` by the kernel. `kDLBool` has no C++ trait here, so
    // `.with_dtype<bool>()` is unavailable -- check the code directly.
    CHECK_HOST(initial_state_mask.dtype().code == kDLBool && initial_state_mask.dtype().bits == 8)
        << "causal_conv1d_fwd: has_initial_state must be a bool tensor, got dtype code "
        << static_cast<int32_t>(initial_state_mask.dtype().code) << " with "
        << static_cast<int32_t>(initial_state_mask.dtype().bits) << " bits";
  }

  // `out` aliases `x`: this op is in-place and callers rely on that.
  auto params = MambaConvParams{};
  params.batch = static_cast<int32_t>(batch);
  params.dim = static_cast<int32_t>(dim);
  params.seqlen = static_cast<int32_t>(seqlen);
  params.width = static_cast<int32_t>(width);
  params.pad_slot_id = pad_slot_id;
  params.silu_activation = silu_activation;
  params.x_ptr = x.data_ptr();
  params.weight_ptr = weight.data_ptr();
  params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
  params.out_ptr = x.data_ptr();
  params.query_start_loc_ptr = varlen ? query_start_loc.value().data_ptr() : nullptr;
  params.cache_indices_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr() : nullptr;
  params.has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr() : nullptr;
  // In the varlen layout `x` is (dim, cu_seqlen): the "batch" axis is the token
  // axis, so the token stride doubles as the batch stride.
  params.x_batch_stride = static_cast<uint32_t>(x.stride(varlen ? 1 : 0));
  params.x_c_stride = static_cast<uint32_t>(x.stride(varlen ? 0 : 1));
  params.x_l_stride = static_cast<uint32_t>(x.stride(varlen ? 1 : 2));
  params.weight_c_stride = static_cast<uint32_t>(weight.stride(0));
  params.weight_width_stride = static_cast<uint32_t>(weight.stride(1));
  params.out_batch_stride = params.x_batch_stride;
  params.out_c_stride = params.x_c_stride;
  params.out_l_stride = params.x_l_stride;

  if (conv_states.has_value()) {
    const auto& states = conv_states.value();
    TensorMatcher({-1, dim_sym, -1}).with_strides({-1, -1, -1}).with_dtype<T>().with_device(device_sym).verify(states);
    params.conv_states_ptr = states.data_ptr();
    params.conv_states_batch_stride = static_cast<uint32_t>(states.stride(0));
    params.conv_states_c_stride = static_cast<uint32_t>(states.stride(1));
    params.conv_states_l_stride = static_cast<uint32_t>(states.stride(2));
  } else {
    params.conv_states_ptr = nullptr;
  }

  causal_conv1d_fwd_cuda<T>(params, device_sym.unwrap());
}

/// \brief Single-step (decode) depthwise causal conv1d, in place on `x`.
///
/// \tparam T                   Element type: fp16_t | bf16_t | fp32_t.
/// \param x                    `(batch, dim, seqlen)`, overwritten with the output.
/// \param conv_state           `(num_entries, dim, state_len)`, `state_len >= width - 1`,
///                             advanced in place.
/// \param weight               `(dim, width)`, `width` in 2..4.
/// \param bias                 Optional `(dim,)`.
/// \param silu_activation      Whether to apply SiLU to the output.
/// \param cache_seqlens        Optional int32 `(batch,)`; when given, `conv_state` is
///                             treated as a circular buffer starting at
///                             `cache_seqlens % state_len`.
/// \param conv_state_indices   Optional int32 `(batch,)` conv-state slot per sequence.
/// \param pad_slot_id          Sequences whose state index equals this are skipped.
template <typename T>
void causal_conv1d_update(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView conv_state,
    tvm::ffi::TensorView weight,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    bool silu_activation,
    tvm::ffi::Optional<tvm::ffi::TensorView> cache_seqlens,
    tvm::ffi::Optional<tvm::ffi::TensorView> conv_state_indices,
    int64_t pad_slot_id) {
  using namespace host;

  auto batch_sym = SymbolicSize{"batch"};
  auto dim_sym = SymbolicSize{"dim"};
  auto seqlen_sym = SymbolicSize{"seqlen"};
  auto width_sym = SymbolicSize{"width"};
  auto state_len_sym = SymbolicSize{"state_len"};
  auto entries_sym = SymbolicSize{"conv_state_entries"};
  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLCUDA>();

  TensorMatcher({batch_sym, dim_sym, seqlen_sym})
      .with_strides({-1, -1, -1})
      .with_dtype<T>()
      .with_device(device_sym)
      .verify(x);
  TensorMatcher({dim_sym, width_sym}).with_strides({-1, -1}).with_dtype<T>().with_device(device_sym).verify(weight);
  // Gathered decode indexes `conv_state` by slot, so its leading dimension is
  // the pool size rather than the batch size.
  if (conv_state_indices.has_value()) {
    TensorMatcher({batch_sym})
        .with_strides({1})
        .with_dtype<int32_t>()
        .with_device(device_sym)
        .verify(conv_state_indices.value());
  } else {
    entries_sym.set_value(batch_sym.unwrap());
  }
  TensorMatcher({entries_sym, dim_sym, state_len_sym})
      .with_strides({-1, -1, -1})
      .with_dtype<T>()
      .with_device(device_sym)
      .verify(conv_state);

  const int64_t width = width_sym.unwrap();
  const int64_t state_len = state_len_sym.unwrap();
  CHECK_HOST(width >= 2 && width <= 4) << "causal_conv1d only supports width between 2 and 4, got " << width;
  CHECK_HOST(state_len >= width - 1) << "causal_conv1d_update: conv_state length " << state_len
                                     << " is shorter than width - 1 = " << width - 1;

  if (bias.has_value()) {
    TensorMatcher({dim_sym}).with_strides({1}).with_dtype<T>().with_device(device_sym).verify(bias.value());
  }

  auto params = MambaConvParams{};
  params.batch = static_cast<int32_t>(batch_sym.unwrap());
  params.dim = static_cast<int32_t>(dim_sym.unwrap());
  params.seqlen = static_cast<int32_t>(seqlen_sym.unwrap());
  params.width = static_cast<int32_t>(width);
  params.pad_slot_id = pad_slot_id;
  params.silu_activation = silu_activation;
  params.x_ptr = x.data_ptr();
  params.weight_ptr = weight.data_ptr();
  params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
  params.out_ptr = x.data_ptr();
  params.x_batch_stride = static_cast<uint32_t>(x.stride(0));
  params.x_c_stride = static_cast<uint32_t>(x.stride(1));
  params.x_l_stride = static_cast<uint32_t>(x.stride(2));
  params.weight_c_stride = static_cast<uint32_t>(weight.stride(0));
  params.weight_width_stride = static_cast<uint32_t>(weight.stride(1));
  params.out_batch_stride = params.x_batch_stride;
  params.out_c_stride = params.x_c_stride;
  params.out_l_stride = params.x_l_stride;

  params.conv_state_ptr = conv_state.data_ptr();
  params.conv_state_len = static_cast<int32_t>(state_len);
  params.conv_state_batch_stride = static_cast<uint32_t>(conv_state.stride(0));
  params.conv_state_c_stride = static_cast<uint32_t>(conv_state.stride(1));
  params.conv_state_l_stride = static_cast<uint32_t>(conv_state.stride(2));

  if (cache_seqlens.has_value()) {
    TensorMatcher({batch_sym})
        .with_strides({1})
        .with_dtype<int32_t>()
        .with_device(device_sym)
        .verify(cache_seqlens.value());
    params.cache_seqlens = static_cast<const int32_t*>(cache_seqlens.value().data_ptr());
  } else {
    params.cache_seqlens = nullptr;
  }
  params.conv_state_indices_ptr =
      conv_state_indices.has_value() ? static_cast<const int32_t*>(conv_state_indices.value().data_ptr()) : nullptr;

  causal_conv1d_update_cuda<T>(params, device_sym.unwrap());
}

}  // namespace mamba_conv

using mamba_conv::causal_conv1d_fwd;
using mamba_conv::causal_conv1d_update;

}  // namespace sglang
