#ifndef SCALE_RESIDUAL_NORM_SCALE_SHIFT_KERNEL_H
#define SCALE_RESIDUAL_NORM_SCALE_SHIFT_KERNEL_H

#include <cuda_runtime.h>

struct WelfordValue {
  float mean = 0.0f, m2 = 0.0f;
  int count = 0;
};

template <typename DType>
union PtrValUnion {
  const DType* ptr;
  DType value;
};

template <typename DType>
struct BroadcastDesc {
  PtrValUnion<DType> union_value;
  int32_t stride_b;
  int32_t frame_len;
};

constexpr int THREADS_PER_WARP = 32;
constexpr int THREADS_PER_CTA = 128;
constexpr int WARP_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;
constexpr int64_t CTA_REDUCE_THRESHOLD = 1024;

enum NormType : int {
  LayerNorm = 0,
  RMSNorm = 1,
};

// Load 4 elements from global memory and convert to RegT (float).
template <typename PtrTy, typename RegT>
__inline__ __device__ void load4_cast(const PtrTy* ptr, RegT v[4]) {
  using Raw = std::conditional_t<std::is_same_v<PtrTy, float>, float4, ushort4>;
  Raw raw = *reinterpret_cast<const Raw*>(ptr);
  const PtrTy* t4 = reinterpret_cast<const PtrTy*>(&raw);
  v[0] = static_cast<RegT>(t4[0]);
  v[1] = static_cast<RegT>(t4[1]);
  v[2] = static_cast<RegT>(t4[2]);
  v[3] = static_cast<RegT>(t4[3]);
}

// Store 4 float values back to memory as PtrTy, using vectorized write.
template <typename PtrTy, typename RegTy>
__inline__ __device__ void store4_cast(PtrTy* ptr, const RegTy v[4]) {
  using Raw = std::conditional_t<std::is_same_v<PtrTy, float>, float4, ushort4>;
  PtrTy cast_v[4];
  cast_v[0] = static_cast<PtrTy>(v[0]);
  cast_v[1] = static_cast<PtrTy>(v[1]);
  cast_v[2] = static_cast<PtrTy>(v[2]);
  cast_v[3] = static_cast<PtrTy>(v[3]);
  Raw raw = *reinterpret_cast<const Raw*>(cast_v);
  *reinterpret_cast<Raw*>(ptr) = raw;
}

template <NormType norm_type>
__inline__ __device__ WelfordValue compute_scale_residual(float x, float g, float r, WelfordValue welf, float& out) {
  out = fmaf(x, g, r);
  if constexpr (norm_type == LayerNorm) {
    welf.count += 1;
    float delta = out - welf.mean;
    welf.mean = welf.mean + delta / welf.count;
    float delta2 = (out - welf.mean);
    welf.m2 = fmaf(delta, delta2, welf.m2);
  } else {
    welf.count += 1;
    float delta = out * out - welf.mean;
    welf.mean = welf.mean + delta / welf.count;
  }
  return welf;
}

// Vectorized path of (x*gate + residual), computing 4 elements per thread.
template <typename DType, typename ParamDType, NormType norm_type>
__inline__ __device__ WelfordValue scale_residual_aligned(
    WelfordValue welf,
    const DType* x,
    const DType* gate,
    const DType* residual,
    DType* residual_output,
    bool is_warp_reduce,
    bool has_gate_tensor,
    int D,
    uint32_t thr_id,
    uint32_t lane_id) {
  uint32_t idx = (is_warp_reduce ? lane_id : thr_id) * 4;
  uint32_t stride = (is_warp_reduce ? THREADS_PER_WARP : THREADS_PER_CTA) * 4;

  while (idx + 3 < D) {
    float x_i[4], gate_i[4], residual_i[4];
    load4_cast<DType, float>(x + idx, x_i);
    if (has_gate_tensor) {
      load4_cast<DType, float>(gate + idx, gate_i);
    } else {
      gate_i[0] = gate_i[1] = gate_i[2] = gate_i[3] = 1.0f;
    }
    load4_cast<DType, float>(residual + idx, residual_i);
    float resi_out[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      welf = compute_scale_residual<norm_type>(x_i[j], gate_i[j], residual_i[j], welf, resi_out[j]);
    }
    store4_cast<DType, float>(residual_output + idx, resi_out);
    idx += stride;
  }
  return welf;
}

// Scalar fallback path for residual = x * gate + residual.
template <typename DType, typename ParamDType, NormType norm_type>
__inline__ __device__ WelfordValue scale_residual_general(
    WelfordValue welf,
    const DType* x,
    const DType* gate,
    const DType* residual,
    DType* residual_output,
    bool is_warp_reduce,
    bool has_gate_tensor,
    int D,
    uint32_t thr_id,
    uint32_t lane_id) {
  uint32_t idx = is_warp_reduce ? lane_id : thr_id;
  uint32_t stride = is_warp_reduce ? THREADS_PER_WARP : THREADS_PER_CTA;
  while (idx < D) {
    float resi_out;
    float gate_v = has_gate_tensor ? static_cast<float>(gate[idx]) : 1.0f;
    welf = compute_scale_residual<norm_type>(
        static_cast<float>(x[idx]), gate_v, static_cast<float>(residual[idx]), welf, resi_out);
    residual_output[idx] = static_cast<DType>(resi_out);
    idx += stride;
  }
  return welf;
}

// Warp-level mean reduction using shuffle instructions.
template <NormType norm_type, int thread_group_width = THREADS_PER_WARP>
__inline__ __device__ WelfordValue warp_reduce(WelfordValue welf) {
#pragma unroll
  for (int offset = thread_group_width >> 1; offset > 0; offset >>= 1) {
    if constexpr (norm_type == LayerNorm) {
      float other_mean = __shfl_down_sync(0xffffffff, welf.mean, offset);
      float other_m2 = __shfl_down_sync(0xffffffff, welf.m2, offset);
      int other_count = __shfl_down_sync(0xffffffff, welf.count, offset);
      if (other_count == 0) continue;
      float total = welf.count + other_count;
      float delta = other_mean - welf.mean;
      float rate = other_count / total;
      welf.mean = fmaf(delta, rate, welf.mean);
      welf.m2 = fmaf(delta * delta, welf.count * rate, welf.m2 + other_m2);
      welf.count = total;
    } else {
      float other_mean = __shfl_down_sync(0xffffffff, welf.mean, offset);
      int other_count = __shfl_down_sync(0xffffffff, welf.count, offset);
      if (other_count == 0) continue;
      float total = welf.count + other_count;
      float delta = other_mean - welf.mean;
      welf.mean = fmaf(delta, other_count / total, welf.mean);
      welf.count = total;
    }
  }
  return welf;
}

// CTA-level reduction for LayerNorm/RMSNorm.
template <NormType norm_type>
__inline__ __device__ void cta_reduce(
    int lane,
    int warp,
    WelfordValue welf,
    int D,
    float eps,
    float* __restrict__ shm_mean,
    float* __restrict__ shm_m2,
    int* __restrict__ shm_count) {
  if (lane == 0) {
    shm_mean[warp] = welf.mean;
    shm_m2[warp] = welf.m2;
    shm_count[warp] = welf.count;
  }
  __syncthreads();

  if (warp == 0) {
    welf.mean = (lane < WARP_PER_CTA) ? shm_mean[lane] : 0;
    welf.m2 = (lane < WARP_PER_CTA) ? shm_m2[lane] : 0;
    welf.count = (lane < WARP_PER_CTA) ? shm_count[lane] : 0;
    welf = warp_reduce<norm_type, WARP_PER_CTA>(welf);
  }

  if (warp == 0 && lane == 0) {
    if constexpr (norm_type == LayerNorm) {
      shm_mean[0] = welf.mean;
      shm_m2[0] = rsqrtf(welf.m2 / D + eps);
    } else {
      shm_mean[0] = rsqrtf(welf.mean + eps);
    }
  }
}

// Vectorized path for norm (LayerNorm/RMSNorm) + scale/shift modulation.
template <typename DType, typename ParamDType, NormType norm_type>
__inline__ __device__ void norm_scale_shift_aligned(
    const DType* residual_output,
    const ParamDType* norm_weight,
    const ParamDType* norm_bias,
    PtrValUnion<DType> shift_union,
    PtrValUnion<DType> scale_union,
    DType* modulated,
    float mean,
    float inv,
    bool is_warp_reduce,
    bool is_scale_shift_tensor,
    bool has_weight_tensor,
    bool has_bias_tensor,
    int D,
    uint32_t thr_id,
    uint32_t lane_id) {
  uint32_t idx = (is_warp_reduce ? lane_id : thr_id) * 4;
  uint32_t stride = (is_warp_reduce ? THREADS_PER_WARP : THREADS_PER_CTA) * 4;
  while (idx + 3 < D) {
    float resi_out_i[4], weight_i[4], bias_i[4], scale_i[4], shift_i[4];
    float mod_i[4];
    load4_cast<DType, float>(residual_output + idx, resi_out_i);
    if (has_weight_tensor) {
      load4_cast<ParamDType, float>(norm_weight + idx, weight_i);
    } else {
      weight_i[0] = weight_i[1] = weight_i[2] = weight_i[3] = 1.0f;
    }
    if (has_bias_tensor) {
      load4_cast<ParamDType, float>(norm_bias + idx, bias_i);
    } else {
      bias_i[0] = bias_i[1] = bias_i[2] = bias_i[3] = 0.0f;
    }
    if (is_scale_shift_tensor) {
      load4_cast<DType, float>(scale_union.ptr + idx, scale_i);
      load4_cast<DType, float>(shift_union.ptr + idx, shift_i);
    } else {
      scale_i[0] = scale_i[1] = scale_i[2] = scale_i[3] = scale_union.value;
      shift_i[0] = shift_i[1] = shift_i[2] = shift_i[3] = shift_union.value;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float norm_x;
      if constexpr (norm_type == LayerNorm) {
        norm_x = (resi_out_i[j] - mean) * inv;
        norm_x = fmaf(weight_i[j], norm_x, bias_i[j]);
      } else if constexpr (norm_type == RMSNorm) {
        norm_x = weight_i[j] * resi_out_i[j] * inv;
      }
      // 3. Modulate
      mod_i[j] = fmaf(norm_x, (1.0f + scale_i[j]), shift_i[j]);
    }
    store4_cast<DType, float>(modulated + idx, mod_i);
    idx += stride;
  }
}

// Scalar fallback path for norm + scale/shift, used when D is unaligned.
template <typename DType, typename ParamDType, NormType norm_type>
__inline__ __device__ void norm_scale_shift_general(
    const DType* residual_output,
    const ParamDType* norm_weight,
    const ParamDType* norm_bias,
    PtrValUnion<DType> shift_union,
    PtrValUnion<DType> scale_union,
    DType* modulated,
    float mean,
    float inv,
    bool is_warp_reduce,
    bool is_scale_shift_tensor,
    bool has_weight_tensor,
    bool has_bias_tensor,
    int D,
    uint32_t thr_id,
    uint32_t lane_id) {
  uint32_t idx = is_warp_reduce ? lane_id : thr_id;
  uint32_t stride = is_warp_reduce ? THREADS_PER_WARP : THREADS_PER_CTA;
  while (idx < D) {
    float resi_out = static_cast<float>(residual_output[idx]);
    float norm_weight_v = has_weight_tensor ? static_cast<float>(norm_weight[idx]) : 1.0f;
    float norm_x;
    if constexpr (norm_type == LayerNorm) {
      float norm_bias_v = has_bias_tensor ? static_cast<float>(norm_bias[idx]) : 0.0f;
      norm_x = (resi_out - mean) * inv;
      norm_x = fmaf(norm_weight_v, norm_x, norm_bias_v);
    } else if constexpr (norm_type == RMSNorm) {
      norm_x = norm_weight_v * resi_out * inv;
    }
    // 3. Modulate
    float scale_value = is_scale_shift_tensor ? scale_union.ptr[idx] : scale_union.value;
    float shift_value = is_scale_shift_tensor ? shift_union.ptr[idx] : shift_union.value;
    float mod = fmaf(norm_x, (1.0f + scale_value), shift_value);
    modulated[idx] = static_cast<DType>(mod);
    idx += stride;
  }
}

/**
 * @brief ScaleResidualNormScaleShift.
 */
template <typename DType, typename ParamDType, NormType norm_type, bool is_d_aligned>
__global__ __launch_bounds__(THREADS_PER_CTA) void scale_residual_norm_scale_shift_kernel(
    const DType* residual,
    const DType* x,
    const DType* gate,
    const ParamDType* norm_weight,
    const ParamDType* norm_bias,
    BroadcastDesc<DType> shift_desc,
    BroadcastDesc<DType> scale_desc,
    double eps,
    DType* modulated,
    DType* residual_output,
    int B,
    int S,
    int D,
    int gate_frame_len,
    bool is_warp_reduce,
    bool has_weight_tensor,
    bool has_bias_tensor) {
  uint32_t cta_id = blockIdx.x;
  uint32_t thr_id = threadIdx.x;
  uint32_t lane_id = thr_id & 31;
  uint32_t warp_id = thr_id >> 5;

  // Pointer Offset
  int64_t tile_id = is_warp_reduce ? cta_id * WARP_PER_CTA + warp_id : cta_id;
  if (tile_id >= B * S) return;

  residual += tile_id * D;
  x += tile_id * D;
  bool has_gate_tensor = gate_frame_len != -1;
  if (has_gate_tensor && gate_frame_len != -2) {
    gate += tile_id / gate_frame_len * D;
  }
  bool is_scale_shift_tensor = scale_desc.stride_b != -1;
  if (is_scale_shift_tensor) {
    const int64_t batch_idx = tile_id / S;
    const int64_t seq_idx = tile_id % S;
    shift_desc.union_value.ptr += (batch_idx * shift_desc.stride_b + seq_idx) / shift_desc.frame_len * D;
    scale_desc.union_value.ptr += (batch_idx * scale_desc.stride_b + seq_idx) / scale_desc.frame_len * D;
  }
  modulated += tile_id * D;
  residual_output += tile_id * D;

  // Scale & Residual
  WelfordValue welf;
  if constexpr (is_d_aligned) {
    welf = scale_residual_aligned<DType, ParamDType, norm_type>(
        welf, x, gate, residual, residual_output, is_warp_reduce, has_gate_tensor, D, thr_id, lane_id);
  } else {
    welf = scale_residual_general<DType, ParamDType, norm_type>(
        welf, x, gate, residual, residual_output, is_warp_reduce, has_gate_tensor, D, thr_id, lane_id);
  }

  // Reduce
  __shared__ float shm_mean[WARP_PER_CTA];  // mean of {LayerNorm: x, RMSNorm: x^2}
  __shared__ float shm_m2[WARP_PER_CTA];
  __shared__ int shm_count[WARP_PER_CTA];
  welf = warp_reduce<norm_type>(welf);
  float mean = 0.0f, inv;
  if (is_warp_reduce) {
    if constexpr (norm_type == LayerNorm) {
      welf.mean = __shfl_sync(0xffffffff, welf.mean, 0);
      welf.m2 = __shfl_sync(0xffffffff, welf.m2, 0);
      mean = welf.mean;
      inv = rsqrtf(welf.m2 / D + eps);
    } else {
      welf.mean = __shfl_sync(0xffffffff, welf.mean, 0);
      inv = rsqrtf(welf.mean + eps);
    }
  } else {
    cta_reduce<norm_type>(lane_id, warp_id, welf, D, eps, shm_mean, shm_m2, shm_count);
    __syncthreads();
    if constexpr (norm_type == LayerNorm) {
      mean = shm_mean[0];
      inv = shm_m2[0];
    } else {
      inv = shm_mean[0];
    }
  }

  // Norm & Modulate
  if constexpr (is_d_aligned) {
    norm_scale_shift_aligned<DType, ParamDType, norm_type>(
        residual_output,
        norm_weight,
        norm_bias,
        shift_desc.union_value,
        scale_desc.union_value,
        modulated,
        mean,
        inv,
        is_warp_reduce,
        is_scale_shift_tensor,
        has_weight_tensor,
        has_bias_tensor,
        D,
        thr_id,
        lane_id);
  } else {
    norm_scale_shift_general<DType, ParamDType, norm_type>(
        residual_output,
        norm_weight,
        norm_bias,
        shift_desc.union_value,
        scale_desc.union_value,
        modulated,
        mean,
        inv,
        is_warp_reduce,
        is_scale_shift_tensor,
        has_weight_tensor,
        has_bias_tensor,
        D,
        thr_id,
        lane_id);
  }
}

#endif
