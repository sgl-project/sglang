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
constexpr uint32_t kTile = 2;                   // tokens per (bdx,bdy,bdz) tile
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
  int debug_lane;
};

// E2M1 positive magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
__constant__ float c_e2m1_mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
// Same magnitudes as fp16 bit patterns (sign bit ORed in per value).
__constant__ uint16_t c_e2m1_half[8] = {0x0000, 0x3800, 0x3C00, 0x3E00,
                                        0x4000, 0x4200, 0x4400, 0x4600};

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
#pragma unroll
      for (uint32_t t = 0; t < kTile; ++t) {
        const uint32_t tile_token =
            iter * kBdy * kBdz * kTile + (tz * kBdy + ty) * kTile + t;
        load_fp4_tile(params.k_data, params.k_scale, params.kv_indices, token_base + tile_token, by,
                      params.num_kv_heads,
                      k_smem + (stage_idx * kBdz + tz) * kBdy * kTile * kHeadDim +
                          (ty * kTile + t) * kHeadDim + tx * kVecSize,
                      tx);
        load_fp4_tile(params.v_data, params.v_scale, params.kv_indices, token_base + tile_token, by,
                      params.num_kv_heads,
                      v_smem + (stage_idx * kBdz + tz) * kBdy * kTile * kHeadDim +
                          (ty * kTile + t) * kHeadDim + tx * kVecSize,
                      tx);
      }
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
#pragma unroll
      for (uint32_t t = 0; t < kTile; ++t) {
        const uint32_t tile_token =
            next_iter * kBdy * kBdz * kTile + (tz * kBdy + ty) * kTile + t;
        load_fp4_tile(params.k_data, params.k_scale, params.kv_indices, token_base + tile_token, by,
                      params.num_kv_heads,
                      k_smem + (stage * kBdz + tz) * kBdy * kTile * kHeadDim +
                          (ty * kTile + t) * kHeadDim + tx * kVecSize,
                      tx);
        load_fp4_tile(params.v_data, params.v_scale, params.kv_indices, token_base + tile_token, by,
                      params.num_kv_heads,
                      v_smem + (stage * kBdz + tz) * kBdy * kTile * kHeadDim +
                          (ty * kTile + t) * kHeadDim + tx * kVecSize,
                      tx);
      }
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


namespace flashinfer {

// ---------------------------------------------------------------------------
// M2b v2: tensor-core (mma.m16n8k16) variant, ldmatrix edition.
//
// One (request, kv_head) per CTA, block(32,4): the 4 warps SPLIT THE KV TOKENS
// (warp w takes 16-token tiles w, w+4, w+8, ...), each warp keeps an online
// softmax state for the whole GQA group, and the 4 partial states are merged
// in SMEM at the end (same split-KV structure as the scalar kernel's bdz).
//
// Orientation (m = qo heads, n = kv tokens, k = head_dim):
//   QK: S[h][t] = sum_d Q[h][d] K[t][d]
//       A = Q  [16h x 16d]  <- ldmatrix.x2 non-trans on q_smem (rows 8-15 are
//                               zero, so a2a3/a6a7 stay constant zero)
//       B = K^T             <- ldmatrix.x4 non-trans on k_smem: a natural
//                               [t][d] tile loaded non-trans IS the col-major
//                               B of K^T (one x4 covers both 8-token n-tiles
//                               of one 16-dim k-block).
//   softmax on the C fragments (each row = one head; its 16 tokens live in
//   the 4 tig lanes x {c0,c1} x 2 n-tiles, reduced with shfl_xor 1,2).
//   PV: O[h][d] = sum_t P[h][t] V[t][d]
//       A = P  <- the QK C fragments reused in place (p in a0a1/a4a5, zero
//                 in the padded head rows a2a3/a6a7) - no shuffles at all.
//       B = V  <- ldmatrix.x2.trans on v_smem (trans cancels the natural
//                 [t][d] layout so the fragment is V as col-major B).
//   O stays in C fragments; each output head/dim is owned by exactly one
//   lane, so the epilogue writes straight to global.
//
// SMEM tiles use the 128B swizzle phys(r,u) = r*16 + (u ^ (r%8)) in 16B
// units (row = token, 16 units = 128 halves) so STS.128 and both ldmatrix
// variants are bank-conflict-free.
//
// Global loads are prefetched into registers one tile ahead, so the LDG
// latency hides under the current tile's mma work (no cp.async needed).
// ---------------------------------------------------------------------------

namespace mma_v2 {

constexpr uint32_t kTT = 16;               // tokens per tile
constexpr uint32_t kUnits = kHeadDim / 8;  // 16 x 16B units per row
constexpr uint32_t kQSmem = 16 * kHeadDim * sizeof(__half);         // 4096
constexpr uint32_t kTileBytes = kTT * kHeadDim * sizeof(__half);    // 8192
constexpr uint32_t kKVSmem = 4 * 2 * kTileBytes;                    // 32768
constexpr uint32_t kOPart = 4 * kBdy * kHeadDim * sizeof(float);    // 8192
constexpr uint32_t kMDPart = 4 * kBdy * 2 * sizeof(float);          // 128

// Byte offset of 16B unit (row r, unit u) inside a swizzled tile.
__device__ __forceinline__ uint32_t swz(uint32_t r, uint32_t u) {
  return (r * kUnits + (u ^ (r & 7))) * 16;
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&R)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Dequant one packed u32 (8 E2M1 nibbles sharing one E8M0 scale) into one
// 16B SMEM unit of 8 halves (dims 8u..8u+8 of some token row).
__device__ __forceinline__ uint4 dequant_unit(uint32_t packed, float scale) {
  uint4 out;
  uint16_t* h = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
  for (uint32_t n = 0; n < 8; ++n) {
    const uint32_t nib = (packed >> (n * 4)) & 0xF;
    const float v = c_e2m1_mag[nib & 7] * scale;
    h[n] = __half_as_ushort(__float2half_rn((nib & 8) ? -v : v));
  }
  return out;
}

// Per-lane global prefetch state for one 16-token tile: lane covers token
// r = lane>>1, dims [dh*64, dh*64+64) (dh = lane&1) = 32B packed per tensor
// plus 2 scale bytes per tensor (one per 32 dims).
struct TilePrefetch {
  uint4 kd[2], vd[2];
  uint32_t ks, vs;
};

}  // namespace mma_v2

__global__ void __launch_bounds__(128, 3) mxfp4_decode_fused_mma_kernel(
    const Mxfp4DecodeParams params) {
  using namespace mma_v2;
  extern __shared__ uint8_t smem[];
  const uint32_t bx = blockIdx.x;    // request
  const uint32_t by = blockIdx.y;    // kv_head
  const uint32_t wid = threadIdx.y;  // warp = token chunk
  const uint32_t lane = threadIdx.x;
  const uint32_t g = lane >> 2;   // head row of this lane's fragments
  const uint32_t tig = lane & 3;  // position inside the 4-lane group

  const uint32_t chunk_start = params.kv_indptr[bx];
  const uint32_t chunk_end = params.kv_indptr[bx + 1];
  const uint32_t chunk_size = chunk_end - chunk_start;

  uint8_t* k_tile = smem + kQSmem + (wid * 2 + 0) * kTileBytes;
  uint8_t* v_tile = smem + kQSmem + (wid * 2 + 1) * kTileBytes;
  float* o_part = reinterpret_cast<float*>(smem + kQSmem + kKVSmem);
  float* md_part = o_part + 4 * kBdy * kHeadDim;
  const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t k_addr = smem_base + kQSmem + (wid * 2 + 0) * kTileBytes;
  const uint32_t v_addr = smem_base + kQSmem + (wid * 2 + 1) * kTileBytes;

  // ---- fill Q: 16 rows (4 real heads + 12 zero rows), swizzled ----
  {
    const uint32_t tid = wid * 32 + lane;
    const uint32_t r = tid >> 3;        // 16 rows x 8 threads
    const uint32_t u0 = (tid & 7) * 2;  // 2 units (8 dims each) per thread
    const __nv_bfloat16* qsrc =
        params.q + ((bx * params.num_qo_heads + by * kBdy) + (r < kBdy ? r : 0)) * kHeadDim;
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      const uint32_t u = u0 + j;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (r < kBdy) {  // dims [8u, 8u+8): 8 bf16 (16B) -> 8 halves (16B)
        const uint4 qb = *reinterpret_cast<const uint4*>(qsrc + u * 8);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&qb);
        uint16_t* h = reinterpret_cast<uint16_t*>(&val);
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i)
          h[i] = __half_as_ushort(__float2half_rn(__bfloat162float(b[i])));
      }
      *reinterpret_cast<uint4*>(smem + swz(r, u)) = val;
    }
  }
  __syncthreads();

  // ---- Q fragments (once): a0a1 = Q[g][16kb+2tig,+1], a4a5 = +8 dims ----
  uint32_t q_frag[8][2];
  {
    const uint32_t qs = smem_base;
#pragma unroll
    for (uint32_t kb = 0; kb < 8; ++kb) {
      uint32_t r0, r1;
      // matrix0: rows 0-7 @ unit 2kb (a0a1), matrix1: rows 0-7 @ unit 2kb+1 (a4a5)
      ldmatrix_x2(r0, r1, qs + swz(lane & 7, 2 * kb + (lane >> 3)));
      q_frag[kb][0] = r0;
      q_frag[kb][1] = r1;
    }
  }

  const uint32_t num_tiles = (chunk_size + kTT - 1) / kTT;
  const uint32_t my_tiles = (num_tiles > wid) ? (num_tiles - wid + 3) / 4 : 0;

  // ---- empty chunk: zero output, -inf lse ----
  if (num_tiles == 0) {
    if (wid == 0 && lane < 16) {
      const uint32_t qo_head = by * kBdy + g;
#pragma unroll
      for (uint32_t dt = 0; dt < 16; ++dt) {
        params.o[(bx * params.num_qo_heads + qo_head) * kHeadDim + 8 * dt + 2 * tig] =
            __float2bfloat16_rn(0.f);
        params.o[(bx * params.num_qo_heads + qo_head) * kHeadDim + 8 * dt + 2 * tig + 1] =
            __float2bfloat16_rn(0.f);
      }
      if (params.lse != nullptr && tig == 0) {
        params.lse[bx * params.num_qo_heads + qo_head] = -math::inf;
      }
    }
    return;
  }

  // ---- tile fetch: LDG into registers (predicated on token validity) ----
  auto prefetch_tile = [&](uint32_t tile_idx, TilePrefetch& pf) {
    const uint32_t r = lane >> 1;
    const uint32_t dh = lane & 1;
    const uint32_t token = tile_idx * kTT + r;
    const bool ok = token < chunk_size;
    const long long row =
        ok ? (long long)params.kv_indices[chunk_start + token] * params.num_kv_heads + by : 0;
    const uint8_t* kd = params.k_data + row * (kHeadDim / 2) + dh * 32;
    const uint8_t* vd = params.v_data + row * (kHeadDim / 2) + dh * 32;
    const uint8_t* ksc = params.k_scale + row * (kHeadDim / 32) + dh * 2;
    const uint8_t* vsc = params.v_scale + row * (kHeadDim / 32) + dh * 2;
    pf.kd[0] = ok ? *reinterpret_cast<const uint4*>(kd) : make_uint4(0, 0, 0, 0);
    pf.kd[1] = ok ? *reinterpret_cast<const uint4*>(kd + 16) : make_uint4(0, 0, 0, 0);
    pf.vd[0] = ok ? *reinterpret_cast<const uint4*>(vd) : make_uint4(0, 0, 0, 0);
    pf.vd[1] = ok ? *reinterpret_cast<const uint4*>(vd + 16) : make_uint4(0, 0, 0, 0);
    pf.ks = ok ? *reinterpret_cast<const uint16_t*>(ksc) : 0;
    pf.vs = ok ? *reinterpret_cast<const uint16_t*>(vsc) : 0;
  };

  // ---- dequant the prefetched regs into this warp's swizzled K/V tiles ----
  auto store_tile = [&](const TilePrefetch& pf) {
    const uint32_t r = lane >> 1;
    const uint32_t dh = lane & 1;
    const uint32_t u_base = dh * 8;  // unit of dims [dh*64, +64)
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {  // uint4 i: dims [dh*64+32i, +32)
      const float ksc = __int_as_float((uint32_t)((pf.ks >> (8 * i)) & 0xFF) << 23);
      const float vsc = __int_as_float((uint32_t)((pf.vs >> (8 * i)) & 0xFF) << 23);
      const uint32_t kw[4] = {pf.kd[i].x, pf.kd[i].y, pf.kd[i].z, pf.kd[i].w};
      const uint32_t vw[4] = {pf.vd[i].x, pf.vd[i].y, pf.vd[i].z, pf.vd[i].w};
#pragma unroll
      for (uint32_t w = 0; w < 4; ++w) {  // u32 w: dims +8w
        const uint32_t u = u_base + 4 * i + w;
        *reinterpret_cast<uint4*>(k_tile + swz(r, u)) = dequant_unit(kw[w], ksc);
        *reinterpret_cast<uint4*>(v_tile + swz(r, u)) = dequant_unit(vw[w], vsc);
      }
    }
  };

  // ---- main loop ----
  float m_state = -math::inf, d_state = 0.f;
  float o_frag[16][4];  // C frags: c0,c1 = O[g][8dt+2tig, +1] (c2,c3 padded)
#pragma unroll
  for (uint32_t dt = 0; dt < 16; ++dt)
    for (uint32_t i = 0; i < 4; ++i) o_frag[dt][i] = 0.f;

  const float s_scale = params.sm_scale * 1.4426950408889634f;
  TilePrefetch pf, pf_next;

  if (my_tiles > 0) {
    prefetch_tile(wid, pf);
    store_tile(pf);
    __syncwarp();
  }

  for (uint32_t it = 0; it < my_tiles; ++it) {
    const uint32_t tile_idx = wid + it * 4;
    const uint32_t next_it = it + 1;
    if (next_it < my_tiles) prefetch_tile(wid + next_it * 4, pf_next);

    // ---- QK: 16 mma -> s_frag[2 n-tiles][4], rows = heads, cols = tokens ----
    float s_frag[2][4];
#pragma unroll
    for (uint32_t nt = 0; nt < 2; ++nt)
      for (uint32_t i = 0; i < 4; ++i) s_frag[nt][i] = 0.f;
#pragma unroll
    for (uint32_t kb = 0; kb < 8; ++kb) {
      uint32_t kbR[4];
      {
        // x4: M0 = K[t0-7][u 2kb] (nt0 b0b1), M1 = K[t8-15][u 2kb] (nt1 b0b1),
        //     M2 = K[t0-7][u 2kb+1] (nt0 b2b3), M3 = K[t8-15][u 2kb+1] (nt1 b2b3)
        const uint32_t m = lane >> 3, r = lane & 7;
        const uint32_t row = (m & 1) ? 8 + r : r;
        const uint32_t u = 2 * kb + (m >> 1);
        ldmatrix_x4(kbR, k_addr + swz(row, u));
      }
      const uint32_t a4[4] = {q_frag[kb][0], 0u, q_frag[kb][1], 0u};
      const uint32_t b_nt0[2] = {kbR[0], kbR[2]};
      const uint32_t b_nt1[2] = {kbR[1], kbR[3]};
      mma_16816(s_frag[0], a4, b_nt0);
      mma_16816(s_frag[1], a4, b_nt1);
    }

    // ---- online softmax over this tile's 16 tokens (rows = heads) ----
    float s0 = s_frag[0][0] * s_scale, s1 = s_frag[0][1] * s_scale;
    float s2 = s_frag[1][0] * s_scale, s3 = s_frag[1][1] * s_scale;
    const uint32_t tok0 = tile_idx * kTT + 2 * tig;
    if (tok0 >= chunk_size) s0 = -math::inf;
    if (tok0 + 1 >= chunk_size) s1 = -math::inf;
    if (tok0 + 8 >= chunk_size) s2 = -math::inf;
    if (tok0 + 9 >= chunk_size) s3 = -math::inf;
    float m_tile = max(max(s0, s1), max(s2, s3));
    m_tile = max(m_tile, math::shfl_xor_sync(m_tile, 1));
    m_tile = max(m_tile, math::shfl_xor_sync(m_tile, 2));
    const float m_new = max(m_state, m_tile);
    const float o_scale = math::ptx_exp2(m_state - m_new);
    const __half2 pa01 = __floats2half2_rn(math::ptx_exp2(s0 - m_new),
                                           math::ptx_exp2(s1 - m_new));
    const __half2 pa23 = __floats2half2_rn(math::ptx_exp2(s2 - m_new),
                                           math::ptx_exp2(s3 - m_new));
    // Accumulate d from the same fp16-rounded p values the PV mma uses, so
    // numerator and denominator stay consistent (o = sum p16*v, d = sum p16).
    float d_part = __half2float(__low2half(pa01)) + __half2float(__high2half(pa01)) +
                   __half2float(__low2half(pa23)) + __half2float(__high2half(pa23));
    d_part += math::shfl_xor_sync(d_part, 1);
    d_part += math::shfl_xor_sync(d_part, 2);
    // p0..p3 are already in the m_new scale, so d_part needs no rescale.
    d_state = d_state * o_scale + d_part;
    m_state = m_new;
#pragma unroll
    for (uint32_t dt = 0; dt < 16; ++dt) {
      o_frag[dt][0] *= o_scale;
      o_frag[dt][1] *= o_scale;
    }

    // ---- PV: 16 mma; A = P fragments in place, B = V via ldmatrix.trans ----
    const uint32_t pA[4] = {*reinterpret_cast<const uint32_t*>(&pa01), 0u,
                            *reinterpret_cast<const uint32_t*>(&pa23), 0u};
#pragma unroll
    for (uint32_t dt = 0; dt < 16; ++dt) {
      uint32_t r0, r1;
      {
        // x2.trans: matrix0 = V rows 0-7 @ unit dt (b0b1), matrix1 = rows 8-15
        // (addresses from lanes 0-15; rows 8-15 come from lanes 8-15 directly)
        ldmatrix_x2_trans(r0, r1, v_addr + swz(lane & 15, dt));
      }
      const uint32_t b[2] = {r0, r1};
      mma_16816(o_frag[dt], pA, b);
    }

    // ---- dequant+store the prefetched next tile ----
    if (next_it < my_tiles) {
      store_tile(pf_next);
      pf = pf_next;
    }
    __syncwarp();
  }

  // ---- merge the 4 warps' partial states ----
  if (g < kBdy) {  // only head rows 0..3 are real (lanes 0-15)
#pragma unroll
    for (uint32_t dt = 0; dt < 16; ++dt) {
      o_part[(wid * kBdy + g) * kHeadDim + 8 * dt + 2 * tig] = o_frag[dt][0];
      o_part[(wid * kBdy + g) * kHeadDim + 8 * dt + 2 * tig + 1] = o_frag[dt][1];
    }
    if (tig == 0) {
      md_part[(wid * kBdy + g) * 2] = m_state;
      md_part[(wid * kBdy + g) * 2 + 1] = d_state;
    }
  }
  __syncthreads();

  if (wid == 0 && lane < 16) {
    float m_max = -math::inf;
#pragma unroll
    for (uint32_t w = 0; w < 4; ++w) m_max = max(m_max, md_part[(w * kBdy + g) * 2]);
    float d_tot = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < 4; ++w)
      d_tot += md_part[(w * kBdy + g) * 2 + 1] * math::ptx_exp2(md_part[(w * kBdy + g) * 2] - m_max);
    const float inv_d = 1.f / d_tot;
    const uint32_t qo_head = by * kBdy + g;
    float scale_w[4];
#pragma unroll
    for (uint32_t w = 0; w < 4; ++w)
      scale_w[w] = math::ptx_exp2(md_part[(w * kBdy + g) * 2] - m_max);
#pragma unroll
    for (uint32_t dt = 0; dt < 16; ++dt) {
      float o0 = 0.f, o1 = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < 4; ++w) {
        o0 += o_part[(w * kBdy + g) * kHeadDim + 8 * dt + 2 * tig] * scale_w[w];
        o1 += o_part[(w * kBdy + g) * kHeadDim + 8 * dt + 2 * tig + 1] * scale_w[w];
      }
      params.o[(bx * params.num_qo_heads + qo_head) * kHeadDim + 8 * dt + 2 * tig] =
          __float2bfloat16_rn(o0 * inv_d);
      params.o[(bx * params.num_qo_heads + qo_head) * kHeadDim + 8 * dt + 2 * tig + 1] =
          __float2bfloat16_rn(o1 * inv_d);
    }
    if (params.lse != nullptr && tig == 0) {
      params.lse[bx * params.num_qo_heads + qo_head] = m_max + __log2f(d_tot);
    }
  }
}

}  // namespace flashinfer
