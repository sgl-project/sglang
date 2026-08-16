// MXFP4 fused decode attention kernel for sm86 (M2a: correctness prototype).
//
// One CTA handles one (request, kv_head) pair and computes attention for the
// GQA group of qo_heads sharing that kv_head. KV is read directly from the
// packed fp4 pool (data + E8M0 scale) and dequantized into fp16 SMEM, so the
// decode step never materializes bf16 KV — this is what removes the workspace
// round-trip that made the M1 path >2x slower than bf16 flashinfer.
//
// Structure follows flashinfer's BatchDecodeWithPagedKVCacheKernel (vendored
// under flashinfer_vendored/): online softmax state, shfl reduce, fp16 SMEM
// tiles. Fixed config for Qwen3: head_dim=128, vec_size=8, bdx=16, bdy=4
// (GQA group), bdz=2, tile_size_per_bdx=1, 2-stage double buffering.
//
// Layout of the fp4 pool (see mxfp4_kv.cu):
//   data  [S, H, D/2]  uint8   <- packed E2M1, block16
//   scale [S, H, D/16] uint8   <- E8M0
// kv_indices[i] = pool slot of the i-th kv token (flashinfer index order).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cooperative_groups.h>

#include "flashinfer_vendored/flashinfer/math.cuh"
#include "flashinfer_vendored/flashinfer/utils.cuh"
#include "flashinfer_vendored/flashinfer/vec_dtypes.cuh"
#include "flashinfer_vendored/flashinfer/attention/state.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kVecSize = 8;
constexpr uint32_t kBdx = kHeadDim / kVecSize;  // 16 threads along head_dim
constexpr uint32_t kBdy = 4;                    // GQA group size (32/8)
constexpr uint32_t kBdz = 2;                    // pipeline depth
constexpr uint32_t kTile = 1;                   // tokens per (bdx,bdy,bdz) tile
constexpr uint32_t kStages = 2;
constexpr uint32_t kNumThreads = kBdx * kBdy * kBdz;  // 128

struct Mxfp4DecodeParams {
  const __nv_bfloat16* q;   // [batch, qo_heads, head_dim]
  const uint8_t* k_data;    // [S, H, 64]
  const uint8_t* k_scale;   // [S, H, 8]
  const uint8_t* v_data;
  const uint8_t* v_scale;
  const int* kv_indices;    // [n] pool slot per kv token
  const int* kv_indptr;     // [batch+1]
  __nv_bfloat16* o;         // [batch, qo_heads, head_dim]
  float* lse;               // [batch, qo_heads] (log2 lse)
  int n;                    // total kv tokens
  int num_qo_heads;
  int num_kv_heads;
  float sm_scale;           // attention scale (pre-softmax)
};

// E2M1 positive magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
__constant__ float c_e2m1_mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ---------------------------------------------------------------------------
// fp4 KV -> fp16 SMEM tile.
// Each thread loads 8 elements (4 packed bytes) + the shared E8M0 scale byte
// (one per block32 = 4 threads), dequantizes in registers, stores fp16.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_fp4_tile(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const int* __restrict__ kv_indices, uint32_t token_idx, uint32_t kv_head,
    uint32_t kv_heads, __half* __restrict__ smem_dst, uint32_t tx) {
  const int slot = kv_indices[token_idx];
  const long long row = (long long)slot * kv_heads + kv_head;
  // data row = slot*H + head, 64 bytes: thread tx reads 4B at tx*4.
  const uint32_t packed = *reinterpret_cast<const uint32_t*>(
      data + row * (kHeadDim / 2) + tx * 4);
  // E8M0 scale: one byte per 32 elements (block32), 4 threads share one.
  const uint8_t s = scale[row * (kHeadDim / 32) + tx / 4];
  const float sscale = __int_as_float((uint32_t)s << 23);
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    const uint8_t b = (packed >> ((i / 2) * 8)) & 0xFF;
    const uint8_t v = (i & 1) ? (b >> 4) : (b & 0xF);
    float val = c_e2m1_mag[v & 0x7] * sscale;
    if (v & 0x8) val = -val;
    smem_dst[i] = __float2half_rn(val);
  }
}

// ---------------------------------------------------------------------------
// QK dot product (flashinfer CUDA-cores style) + online softmax updates.
// ---------------------------------------------------------------------------
template <uint32_t tile_size>
__device__ __forceinline__ void compute_qk(
    const Mxfp4DecodeParams& params, const __half* __restrict__ k_smem,
    const vec_t<float, kVecSize>& q_vec, uint32_t iter_base, uint32_t chunk_size,
    float* __restrict__ s, state_t<kVecSize>& st, uint32_t tx, uint32_t tz) {
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, kVecSize> k_vec;
    k_vec.cast_load(k_smem + (j * kBdx + tx) * kVecSize);
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = kBdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    // exp2 domain: exp(s * scale) == exp2(s * scale * log2e)
    s[j] *= params.sm_scale * 1.4426950408889634f;
    const bool valid = (iter_base + tz * tile_size + j) < chunk_size;
    s[j] = valid ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }
  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    st.o[i] *= o_scale;
  }
}

// ---------------------------------------------------------------------------
// PV accumulation: o += p * v  (v from SMEM, p = s[j])
// ---------------------------------------------------------------------------
template <uint32_t tile_size>
__device__ __forceinline__ void update_local_state(
    const __half* __restrict__ v_smem, const float* __restrict__ s,
    state_t<kVecSize>& st, uint32_t tx) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, kVecSize> v_vec;
    v_vec.cast_load(v_smem + j * kHeadDim + tx * kVecSize);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      st.o[i] += s[j] * v_vec[i];
    }
  }
}

// Merge the bdz per-tz states (o/m/d through SMEM), flashinfer sync_state.
__device__ __forceinline__ void sync_states(state_t<kVecSize>& st, float* smem_o,
                                            float* smem_md, uint32_t tx, uint32_t ty,
                                            uint32_t tz) {
  auto block = cg::this_thread_block();
  st.o.store(smem_o + (tz * kBdy + ty) * kHeadDim + tx * kVecSize);
  smem_md[(tz * kBdy + ty) * 2] = st.m;
  smem_md[(tz * kBdy + ty) * 2 + 1] = st.d;
  block.sync();
  st.init();
#pragma unroll
  for (uint32_t j = 0; j < kBdz; ++j) {
    float mz = smem_md[(j * kBdy + ty) * 2], dz = smem_md[(j * kBdy + ty) * 2 + 1];
    vec_t<float, kVecSize> oz;
    oz.load(smem_o + (j * kBdy + ty) * kHeadDim + tx * kVecSize);
    st.merge(oz, mz, dz);
  }
}

__global__ void mxfp4_decode_fused_kernel(const Mxfp4DecodeParams params) {
  extern __shared__ uint8_t smem[];
  const uint32_t bx = blockIdx.x;  // request
  const uint32_t by = blockIdx.y;  // kv_head
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t tz = threadIdx.z;

  const uint32_t chunk_start = params.kv_indptr[bx];
  const uint32_t chunk_end = params.kv_indptr[bx + 1];
  const uint32_t chunk_size = chunk_end - chunk_start;

  // SMEM layout: 2 stages of K and V, each [bdz, bdy, tile, head_dim] fp16,
  // plus float merge area for sync_states (o and m/d per (tz, ty)).
  __half* k_smem = reinterpret_cast<__half*>(smem);
  __half* v_smem = k_smem + kStages * kBdz * kBdy * kTile * kHeadDim;
  const uint32_t smem_md_off = kStages * kBdz * kBdy * kTile * kHeadDim;
  float* smem_o =
      reinterpret_cast<float*>(smem + smem_md_off * 2 * sizeof(__half));
  float* smem_md = smem_o + kBdz * kBdy * kHeadDim;

  // q load (bf16 -> float)
  const uint32_t qo_head = by * kBdy + ty;
  vec_t<float, kVecSize> q_vec;
  q_vec.cast_load(params.q + (bx * params.num_qo_heads + qo_head) * kHeadDim + tx * kVecSize);

  state_t<kVecSize> st;
  float s[kBdy * kTile];

  // 2-stage pipeline: preload tile 0..1, then loop.
  uint32_t stage_idx = 0;
  const uint32_t num_iters = (chunk_size + kBdy * kBdz * kTile - 1) / (kBdy * kBdz * kTile);
  if (num_iters == 0) {
    // empty chunk: emit zero output / -inf lse
    if (tz == 0) {
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) st.o[i] = 0.f;
      st.o.cast_store(params.o + (bx * params.num_qo_heads + qo_head) * kHeadDim + tx * kVecSize);
      if (params.lse != nullptr) {
        params.lse[bx * params.num_qo_heads + qo_head] = -math::inf;
      }
    }
    return;
  }

  const uint32_t token_base = chunk_start;
#pragma unroll
  for (uint32_t iter = 0; iter < kStages; ++iter) {
    if (iter < num_iters) {
      const uint32_t tile_token = iter * kBdy * kBdz * kTile + (tz * kBdy + ty) * kTile;
      load_fp4_tile(params.k_data, params.k_scale, params.kv_indices, token_base + tile_token, by,
                    params.num_kv_heads,
                    k_smem + (stage_idx * kBdz + tz) * kBdy * kTile * kHeadDim +
                        (ty * kTile) * kHeadDim + tx * kVecSize,
                    tx);
      load_fp4_tile(params.v_data, params.v_scale, params.kv_indices, token_base + tile_token, by,
                    params.num_kv_heads,
                    v_smem + (stage_idx * kBdz + tz) * kBdy * kTile * kHeadDim +
                        (ty * kTile) * kHeadDim + tx * kVecSize,
                    tx);
    }
    stage_idx = (stage_idx + 1) % kStages;
  }
  __syncthreads();

  for (uint32_t iter = 0; iter < num_iters; ++iter) {
    const uint32_t stage = stage_idx;
    compute_qk<kBdy * kTile>(params, k_smem + (stage * kBdz + tz) * kBdy * kTile * kHeadDim,
                             q_vec, iter * kBdy * kBdz * kTile, chunk_size, s, st, tx, tz);
    __syncthreads();
    update_local_state<kBdy * kTile>(
        v_smem + (stage * kBdz + tz) * kBdy * kTile * kHeadDim, s, st, tx);
    __syncthreads();

    // prefetch next tile
    const uint32_t next_iter = iter + kStages;
    if (next_iter < num_iters) {
      const uint32_t tile_token =
          next_iter * kBdy * kBdz * kTile + (tz * kBdy + ty) * kTile;
      load_fp4_tile(params.k_data, params.k_scale, params.kv_indices, token_base + tile_token, by,
                    params.num_kv_heads,
                    k_smem + (stage * kBdz + tz) * kBdy * kTile * kHeadDim +
                        (ty * kTile) * kHeadDim + tx * kVecSize,
                    tx);
      load_fp4_tile(params.v_data, params.v_scale, params.kv_indices, token_base + tile_token, by,
                    params.num_kv_heads,
                    v_smem + (stage * kBdz + tz) * kBdy * kTile * kHeadDim +
                        (ty * kTile) * kHeadDim + tx * kVecSize,
                    tx);
    }
    __syncthreads();
    stage_idx = (stage_idx + 1) % kStages;
  }

  // finalize: merge bdz partial states, normalize by d, store o and lse
  sync_states(st, smem_o, smem_md, tx, ty, tz);
  st.normalize();
  if (tz == 0) {
    st.o.cast_store(params.o + (bx * params.num_qo_heads + qo_head) * kHeadDim + tx * kVecSize);
    if (params.lse != nullptr) {
      params.lse[bx * params.num_qo_heads + qo_head] = st.get_lse();
    }
  }
}

}  // namespace flashinfer
