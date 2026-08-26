// Fused QSA (Qwen4-Exp sparse attention) indexer-prep kernels.
//
// qsa_index_q_prep_kernel replaces, per token, the eager chain
//   split -> GemmaRMSNorm(index q) -> MRoPE(index q)
//   -> set_qsa_key_state_buffer(raw token k) -> set_qsa_rope_position_buffer
// with a single kernel launch. q rows are normalised per (token, head) and
// rotated with the model's (multimodal) RoPE; the raw k row and the RoPE
// coordinates are written to the indexer state buffers.
//
// qsa_index_k_compress_kernel replaces, per completed compress group,
//   gather group -> fp32 mean -> round to storage dtype -> GemmaRMSNorm
//   -> MRoPE(group-start position) -> set_qsa_compressed_k_buffer
// with one warp per group.
//
// Numerics follow the eager chain step by step so results are bit-identical:
//   - norm: fp32 sum of squares, rsqrt(mean + eps), x * nf * (1 + w) rounded
//     once to the storage dtype (matches sgl_kernel gemma_rmsnorm);
//   - rope: cos/sin are rounded to the storage dtype first, every elementary
//     multiply/add is rounded to the storage dtype, matching PyTorch's
//     opmath (fp32) + per-op rounding on 2-byte tensors;
//   - mean: fp32 sequential accumulation over the group, scaled by 1/ratio,
//     rounded to the storage dtype before the norm reads it.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/// \brief Round a float to the storage dtype and back (one eager aten step).
/// The inline cvt keeps nvcc from folding the round-trip away: every eager
/// bf16/fp16 aten op rounds its result, and this kernel must round likewise.
template <typename T>
SGL_DEVICE float eager_round(float x) {
  return static_cast<float>(DTypeTrait<T>::from(x));
}

template <>
SGL_DEVICE float eager_round<bf16_t>(float x) {
#ifndef USE_ROCM
  uint16_t u;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(u) : "f"(x));
  return __bfloat162float(__ushort_as_bfloat16(u));
#else
  return static_cast<float>(DTypeTrait<bf16_t>::from(x));
#endif
}

template <>
SGL_DEVICE float eager_round<fp16_t>(float x) {
#ifndef USE_ROCM
  uint16_t u;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(u) : "f"(x));
  return __ushort_as_half(u);
#else
  return static_cast<float>(DTypeTrait<fp16_t>::from(x));
#endif
}

/**
 * \brief Apply (M)RoPE to a normed row staged in shared memory.
 *
 * Reproduces the eager `get_cos_sin_with_position` + `apply_rotary_emb` chain:
 * `axis_map[i]` selects which position axis feeds pair index i (plain RoPE
 * uses all-zero maps; Qwen interleaved/sectioned MRoPE maps are built on the
 * host). The cos/sin cache row is [cos(half), sin(half)] of width rotary_dim.
 *
 * \tparam T          Storage element type: bf16_t | fp16_t.
 * \tparam kHeadDim   Compile-time head dimension (multiple of 32).
 * \tparam kIsNeox    true -> NeoX pairing (d, d+half); false -> GPT-J (2i, 2i+1).
 * \param smem_row    Normed row [kHeadDim], one warp cooperates.
 * \param out_row     Destination row [kHeadDim].
 * \param cos_sin_cache [num_positions, rotary_dim] fp32 cache.
 * \param axis_map    [rotary_dim/2] int32 position-axis selector per pair.
 * \param pos         Resolved per-axis positions for this token (>= 3 entries).
 * \param rotary_dim  Rotated prefix length; tail dims pass through.
 */
template <typename T, int kHeadDim, bool kIsNeox>
SGL_DEVICE void qsa_mrope_apply(
    const T* __restrict__ smem_row,
    T* __restrict__ out_row,
    const float* __restrict__ cos_sin_cache,
    const int32_t* __restrict__ axis_map,
    const int64_t* pos,
    const int32_t rotary_dim) {
  using namespace device;
  constexpr int kPerLane = kHeadDim / kWarpThreads;
  using vec_t = AlignedVector<T, kPerLane>;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const int32_t half = rotary_dim / 2;

  vec_t ov;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const int32_t d = static_cast<int32_t>(lane * kPerLane) + i;
    T o;
    if constexpr (kIsNeox) {
      if (d < half) {
        const int32_t p = d + half;
        const float* row = cos_sin_cache + pos[axis_map[d]] * rotary_dim;
        const float c = eager_round<T>(row[d]);
        const float s = eager_round<T>(row[half + d]);
        const float nd = static_cast<float>(smem_row[d]);
        const float np = static_cast<float>(smem_row[p]);
        o = DTypeTrait<T>::from(eager_round<T>(nd * c) - eager_round<T>(np * s));
      } else if (d < rotary_dim) {
        const int32_t p = d - half;
        const float* row = cos_sin_cache + pos[axis_map[p]] * rotary_dim;
        const float c = eager_round<T>(row[p]);
        const float s = eager_round<T>(row[half + p]);
        const float nd = static_cast<float>(smem_row[d]);
        const float np = static_cast<float>(smem_row[p]);
        o = DTypeTrait<T>::from(eager_round<T>(nd * c) + eager_round<T>(np * s));
      } else {
        o = smem_row[d];
      }
    } else {
      if (d < rotary_dim) {
        const int32_t p = d / 2;
        const float* row = cos_sin_cache + pos[axis_map[p]] * rotary_dim;
        const float c = eager_round<T>(row[p]);
        const float s = eager_round<T>(row[half + p]);
        const int32_t q = (d % 2 == 0) ? d + 1 : d - 1;
        const float nd = static_cast<float>(smem_row[d]);
        const float nq = static_cast<float>(smem_row[q]);
        const float t1 = eager_round<T>(nd * c);
        const float t2 = eager_round<T>(nq * s);
        o = DTypeTrait<T>::from((d % 2 == 0) ? t1 - t2 : t1 + t2);
      } else {
        o = smem_row[d];
      }
    }
    ov[i] = o;
  }
  ov.store(out_row, lane);  // offset is in vector units
}

/**
 * \brief Gemma RMSNorm of one row into shared memory (warp-cooperative).
 *
 * out = x * rsqrt(mean(x^2) + eps) * (1 + w), fp32 math with one rounding.
 * The sum-of-squares uses flashinfer RMSNormKernel's exact partial layout
 * (vec_size = 8 for 2-byte dtypes, thread t sums elements [8t, 8t+8), inactive
 * lanes contribute 0, xor butterfly over the warp) so results stay bit-equal
 * to the eager sgl_kernel gemma_rmsnorm this replaces.
 */
template <typename T, int kHeadDim>
SGL_DEVICE void qsa_gemma_norm_row(
    const T* __restrict__ x_row,
    const T* __restrict__ weight,
    const float eps,
    T* __restrict__ smem_row) {
  using namespace device;
  constexpr int kPerLane = kHeadDim / kWarpThreads;
  using vec_t = AlignedVector<T, kPerLane>;
  const uint32_t lane = threadIdx.x % kWarpThreads;

  vec_t xv, wv;
  xv.load(x_row, lane);  // offset is in vector units
  wv.load(weight, lane);

  float xf[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    xf[i] = static_cast<float>(xv[i]);
  }

  static_assert(kHeadDim % 8 == 0);
  constexpr uint32_t kNormThreads = kHeadDim / 8;
  float ss = 0.0f;
  if (lane < kNormThreads) {
    AlignedVector<T, 4> va, vb;
    va.load(x_row, lane * 2);      // elements [8*lane, 8*lane+4)
    vb.load(x_row, lane * 2 + 1);  // elements [8*lane+4, 8*lane+8)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float f = static_cast<float>(va[i]);
      ss += f * f;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float f = static_cast<float>(vb[i]);
      ss += f * f;
    }
  }
  ss = warp::reduce_sum(ss);
  const float nf = math::rsqrt(ss / kHeadDim + eps);

#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const float wf = static_cast<float>(wv[i]);
    smem_row[lane * kPerLane + i] = DTypeTrait<T>::from(xf[i] * nf * (1.0f + wf));
  }
  __syncwarp();
}

struct QsaIndexQPrepParams {
  const void* qk;                  // [tokens, (num_q_heads + 1) * kHeadDim]
  void* q_out;                     // [tokens, q_heads_padded, kHeadDim]
  const void* weight;              // [kHeadDim]
  const float* cos_sin_cache;      // [positions_capacity, rotary_dim]
  const int32_t* axis_map;         // [rotary_dim / 2]
  const int64_t* positions;        // [num_axes, tokens] (row stride may exceed tokens)
  const int64_t* cache_loc;        // [tokens]
  void* key_state_buffer;          // [slots, kHeadDim]
  int64_t* rope_position_buffer;   // [slots, 3]
  int64_t positions_stride;
  int32_t num_axes;
  int32_t num_q_heads;
  int32_t q_heads_padded;
  int32_t rotary_dim;
  float eps;
};

/**
 * \brief Per-token fused index-Q prep: gemma norm + MRoPE for every query
 * head, zero-fill of padded heads, raw token-K store and RoPE-position store.
 * One CTA (4 warps) per token; one warp per query head.
 */
template <typename T, int kHeadDim, bool kIsNeox, bool kUsePDL>
__global__ __launch_bounds__(128) void qsa_index_q_prep_kernel(
    const QsaIndexQPrepParams __grid_constant__ params) {
  using namespace device;
  constexpr int kPerLane = kHeadDim / kWarpThreads;
  using vec_t = AlignedVector<T, kPerLane>;
  const uint32_t token = blockIdx.x;
  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  __shared__ T smem_rows[4][kHeadDim];

  device::PDLWaitPrimary<kUsePDL>();

  const int64_t qk_row =
      static_cast<int64_t>(token) * (params.num_q_heads + 1) * kHeadDim;
  const int64_t loc = params.cache_loc[token];
  int64_t pos[3];
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    const int64_t ax = a < params.num_axes ? a : 0;
    pos[a] = params.positions[ax * params.positions_stride + token];
  }

  for (int32_t h = static_cast<int32_t>(warp); h < params.q_heads_padded;
       h += 4) {
    T* out_row =
        static_cast<T*>(params.q_out) +
        (static_cast<int64_t>(token) * params.q_heads_padded + h) * kHeadDim;
    if (h < params.num_q_heads) {
      const T* x_row = static_cast<const T*>(params.qk) + qk_row + h * kHeadDim;
      qsa_gemma_norm_row<T, kHeadDim>(
          x_row, static_cast<const T*>(params.weight), params.eps,
          smem_rows[warp]);
      qsa_mrope_apply<T, kHeadDim, kIsNeox>(
          smem_rows[warp], out_row, params.cos_sin_cache, params.axis_map, pos,
          params.rotary_dim);
    } else {
      vec_t zv;
      zv.fill(DTypeTrait<T>::from(0.0f));
      zv.store(out_row, lane);  // offset is in vector units
    }
  }

  // Raw token-K and RoPE coordinates are stored for every token, whether or
  // not the token completes a compression group.
  if (warp == 0) {
    vec_t kv;
    kv.load(
        static_cast<const T*>(params.qk) + qk_row + params.num_q_heads * kHeadDim,
        lane);  // offset is in vector units
    kv.store(static_cast<T*>(params.key_state_buffer) + loc * kHeadDim,
             lane);
  }
  if (warp == 1 && lane < 3) {
    params.rope_position_buffer[loc * 3 + lane] = pos[lane];
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

struct QsaIndexKCompressParams {
  const void* key_state_buffer;      // [slots, kHeadDim]
  const int32_t* group_locs;         // [groups, compress_ratio]
  const int64_t* rope_position_buffer;  // [slots, 3]
  const float* cos_sin_cache;        // [positions_capacity, rotary_dim]
  const int32_t* axis_map;           // [rotary_dim / 2]
  const void* weight;                // [kHeadDim]
  const int32_t* write_locs;         // [groups]
  void* compressed_k_buffer;         // [compressed_slots, kHeadDim]
  int32_t compress_ratio;
  int32_t rotary_dim;
  int32_t num_groups;
  float eps;
};

/**
 * \brief Per-group compressed-K prep: fp32 mean over the group, gemma norm,
 * MRoPE at the group-start position, store into the compressed cache.
 * One warp per group.
 */
template <typename T, int kHeadDim, bool kIsNeox, bool kUsePDL>
__global__ __launch_bounds__(128) void qsa_index_k_compress_kernel(
    const QsaIndexKCompressParams __grid_constant__ params) {
  using namespace device;
  constexpr int kPerLane = kHeadDim / kWarpThreads;
  using vec_t = AlignedVector<T, kPerLane>;
  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t group = blockIdx.x * 4 + warp;
  if (group >= static_cast<uint32_t>(params.num_groups)) {
    return;
  }
  __shared__ T smem_rows[4][kHeadDim];

  device::PDLWaitPrimary<kUsePDL>();

  const int32_t* locs = params.group_locs + group * params.compress_ratio;
  const int32_t loc0 = locs[0];

  // fp32 mean over the group, rounded to the storage dtype exactly like
  // average_pool_qsa_keys (float().mean(dim=1).to(dtype)).
  float mf[kPerLane];
  {
    float acc[kPerLane];
    for (int32_t r = 0; r < params.compress_ratio; ++r) {
      vec_t v;
      v.load(
          static_cast<const T*>(params.key_state_buffer) +
              static_cast<int64_t>(locs[r]) * kHeadDim,
          lane);  // offset is in vector units
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        const float f = static_cast<float>(v[i]);
        acc[i] = (r == 0) ? f : acc[i] + f;
      }
    }
    const float inv_ratio = 1.0f / static_cast<float>(params.compress_ratio);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const T m = DTypeTrait<T>::from(acc[i] * inv_ratio);
      mf[i] = static_cast<float>(m);
      smem_rows[warp][lane * kPerLane + i] = m;
    }
    __syncwarp();
    // Sum of squares in flashinfer RMSNormKernel's exact partial layout
    // (thread t sums elements [8t, 8t+8), inactive lanes contribute 0), so
    // the result stays bit-equal to the eager k_layernorm this replaces.
    static_assert(kHeadDim % 8 == 0);
    float ss = 0.0f;
    if (lane < kHeadDim / 8) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = static_cast<float>(smem_rows[warp][lane * 8 + j]);
        ss += f * f;
      }
    }
    ss = warp::reduce_sum(ss);
    const float nf = math::rsqrt(ss / kHeadDim + params.eps);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const float wf = static_cast<float>(
          static_cast<const T*>(params.weight)[lane * kPerLane + i]);
      smem_rows[warp][lane * kPerLane + i] =
          DTypeTrait<T>::from(mf[i] * nf * (1.0f + wf));
    }
    __syncwarp();
  }

  int64_t pos[3];
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    pos[a] = params.rope_position_buffer[static_cast<int64_t>(loc0) * 3 + a];
  }

  T* out_row = static_cast<T*>(params.compressed_k_buffer) +
               static_cast<int64_t>(params.write_locs[group]) * kHeadDim;
  qsa_mrope_apply<T, kHeadDim, kIsNeox>(
      smem_rows[warp], out_row, params.cos_sin_cache, params.axis_map, pos,
      params.rotary_dim);

  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate inputs and launch `qsa_index_q_prep_kernel` (one CTA per token).
 *
 * \tparam T         Element type: bf16_t | fp16_t.
 * \tparam kHeadDim  Index head dimension: 64 | 128 | 256.
 * \tparam kIsNeox   RoPE pairing style.
 * \tparam kUsePDL   Whether to launch with PDL enabled.
 */
template <typename T, int kHeadDim, bool kIsNeox, bool kUsePDL>
void qsa_index_q_prep(
    tvm::ffi::TensorView qk,
    tvm::ffi::TensorView q_out,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView cos_sin_cache,
    tvm::ffi::TensorView axis_map,
    tvm::ffi::TensorView positions,
    int64_t num_axes,
    tvm::ffi::TensorView cache_loc,
    tvm::ffi::TensorView key_state_buffer,
    tvm::ffi::TensorView rope_position_buffer,
    int64_t num_q_heads,
    int64_t rotary_dim,
    float eps) {
  using namespace host;
  auto tokens = SymbolicSize{"tokens"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  constexpr int64_t D = kHeadDim;

  TensorMatcher({tokens, (num_q_heads + 1) * D})
      .with_dtype<T>()
      .with_device(device)
      .verify(qk);
  auto heads_padded = SymbolicSize{"heads_padded"};
  TensorMatcher({tokens, heads_padded, D})
      .with_dtype<T>()
      .with_device(device)
      .verify(q_out);
  TensorMatcher({D}).with_dtype<T>().with_device(device).verify(weight);
  auto cache_rows = SymbolicSize{"cos_sin_cache_rows"};
  TensorMatcher({cache_rows, rotary_dim})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(cos_sin_cache);
  TensorMatcher({rotary_dim / 2})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(axis_map);
  TensorMatcher({num_axes, tokens})
      .with_dtype<int64_t>()
      .with_device(device)
      .with_strides({-1, 1})
      .verify(positions);
  TensorMatcher({tokens}).with_dtype<int64_t>().with_device(device).verify(
      cache_loc);
  auto slots = SymbolicSize{"state_slots"};
  TensorMatcher({slots, D}).with_dtype<T>().with_device(device).verify(
      key_state_buffer);
  TensorMatcher({slots, 3})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(rope_position_buffer);

  const int64_t num_tokens = tokens.unwrap();
  const int64_t q_heads_padded = heads_padded.unwrap();
  CHECK_HOST(num_tokens > 0) << "qsa_index_q_prep: no tokens";
  CHECK_HOST(num_axes == 1 || num_axes == 3)
      << "qsa_index_q_prep: positions must have 1 or 3 axes, got " << num_axes;
  CHECK_HOST(q_heads_padded >= num_q_heads)
      << "qsa_index_q_prep: padded heads " << q_heads_padded
      << " < num_q_heads " << num_q_heads;
  CHECK_HOST(rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= D)
      << "qsa_index_q_prep: invalid rotary_dim " << rotary_dim;

  const auto params = QsaIndexQPrepParams{
      .qk = qk.data_ptr(),
      .q_out = q_out.data_ptr(),
      .weight = weight.data_ptr(),
      .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
      .axis_map = static_cast<const int32_t*>(axis_map.data_ptr()),
      .positions = static_cast<const int64_t*>(positions.data_ptr()),
      .cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr()),
      .key_state_buffer = key_state_buffer.data_ptr(),
      .rope_position_buffer =
          static_cast<int64_t*>(rope_position_buffer.data_ptr()),
      .positions_stride = positions.stride(0),
      .num_axes = static_cast<int32_t>(num_axes),
      .num_q_heads = static_cast<int32_t>(num_q_heads),
      .q_heads_padded = static_cast<int32_t>(q_heads_padded),
      .rotary_dim = static_cast<int32_t>(rotary_dim),
      .eps = eps,
  };
  LaunchKernel(static_cast<uint32_t>(num_tokens), 128, device.unwrap())
      .enable_pdl(kUsePDL)(
          qsa_index_q_prep_kernel<T, kHeadDim, kIsNeox, kUsePDL>, params);
}

/**
 * \brief Validate inputs and launch `qsa_index_k_compress_kernel` (one warp per group).
 */
template <typename T, int kHeadDim, bool kIsNeox, bool kUsePDL>
void qsa_index_k_compress(
    tvm::ffi::TensorView key_state_buffer,
    tvm::ffi::TensorView group_locs,
    tvm::ffi::TensorView rope_position_buffer,
    tvm::ffi::TensorView cos_sin_cache,
    tvm::ffi::TensorView axis_map,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView write_locs,
    tvm::ffi::TensorView compressed_k_buffer,
    int64_t compress_ratio,
    int64_t rotary_dim,
    float eps) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  constexpr int64_t D = kHeadDim;

  auto slots = SymbolicSize{"state_slots"};
  TensorMatcher({slots, D}).with_dtype<T>().with_device(device).verify(
      key_state_buffer);
  auto groups = SymbolicSize{"groups"};
  TensorMatcher({groups, compress_ratio})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(group_locs);
  TensorMatcher({slots, 3})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(rope_position_buffer);
  auto cache_rows = SymbolicSize{"cos_sin_cache_rows"};
  TensorMatcher({cache_rows, rotary_dim})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(cos_sin_cache);
  TensorMatcher({rotary_dim / 2})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(axis_map);
  TensorMatcher({D}).with_dtype<T>().with_device(device).verify(weight);
  TensorMatcher({groups}).with_dtype<int32_t>().with_device(device).verify(
      write_locs);
  auto compressed_slots = SymbolicSize{"compressed_slots"};
  TensorMatcher({compressed_slots, D})
      .with_dtype<T>()
      .with_device(device)
      .verify(compressed_k_buffer);

  const int64_t num_groups = groups.unwrap();
  CHECK_HOST(num_groups > 0) << "qsa_index_k_compress: no groups";
  CHECK_HOST(compress_ratio > 0 && compress_ratio <= 16)
      << "qsa_index_k_compress: invalid compress_ratio " << compress_ratio;
  CHECK_HOST(rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= D)
      << "qsa_index_k_compress: invalid rotary_dim " << rotary_dim;

  const auto params = QsaIndexKCompressParams{
      .key_state_buffer = key_state_buffer.data_ptr(),
      .group_locs = static_cast<const int32_t*>(group_locs.data_ptr()),
      .rope_position_buffer =
          static_cast<const int64_t*>(rope_position_buffer.data_ptr()),
      .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
      .axis_map = static_cast<const int32_t*>(axis_map.data_ptr()),
      .weight = weight.data_ptr(),
      .write_locs = static_cast<const int32_t*>(write_locs.data_ptr()),
      .compressed_k_buffer = compressed_k_buffer.data_ptr(),
      .compress_ratio = static_cast<int32_t>(compress_ratio),
      .rotary_dim = static_cast<int32_t>(rotary_dim),
      .num_groups = static_cast<int32_t>(num_groups),
      .eps = eps,
  };
  LaunchKernel(
      static_cast<uint32_t>(div_ceil(num_groups, 4)), 128, device.unwrap())
      .enable_pdl(kUsePDL)(
          qsa_index_k_compress_kernel<T, kHeadDim, kIsNeox, kUsePDL>, params);
}

}  // namespace sglang
