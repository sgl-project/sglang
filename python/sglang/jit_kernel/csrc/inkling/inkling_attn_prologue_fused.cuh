// Fused attention prologues for Inkling: after the qkvr projection, ONE
// kernel does {k_sconv + v_sconv + per-head q/k RMSNorm + the KV-cache
// store}, in three variants -- TARGET-VERIFY (fixed q tokens/seq + both
// save_intermediate_conv_windows), DECODE (conv from the working cache +
// in-block shift-update + track), and EXTEND (varlen sequences via si/cu +
// a tiny trailing conv-cache-update/track kernel). Replaces 2x causal_conv1d
// + save_windows or update_sconv_cache + fused qk-norm + the backend's
// set_kv_buffer scatter (attention then runs with save_kv_cache=False).
// rel_logits_proj overlaps on the alt stream.
//
// Layout: ONE BLOCK PER TOKEN; one 16B vec (8 channels) per thread. Lane
// roles by vec index: [0, Dq/8) q-norm lanes, then Dkv/8 k lanes, then Dkv/8
// v lanes. head_dim=128 -> a head is 16 CONTIGUOUS lanes, reduced with
// width-16 warp shuffles (Dq/8 and Dkv/8 are multiples of 16, so head groups
// never straddle warps). The convs read cross-token taps directly from the
// (strided) qkvr tensor; per-seq prefixes from the read-only conv caches; the
// per-position windows go to the intermediate buffers exactly like
// save_intermediate_conv_windows (raw copies, no gating). PAD sequences
// (cache_indices == -1) skip prefix/window IO but still emit outputs and the
// KV store (mirroring the unfused path, which stores pad rows too).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace {

constexpr int kPadSlot = -1;
constexpr uint32_t kVecElems = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kHeadLanes = kHeadDim / kVecElems;  // 16
constexpr uint32_t kMXFP8Block = 32;
constexpr uint32_t kMXFP8BlockLanes = kMXFP8Block / kVecElems;  // 4
constexpr float kE4M3Max = 448.0f;

// Per-head (16-lane) sum-reduce -> rsqrt(mean(ss)+eps), broadcast to all lanes
// of the group. A head is 16 CONTIGUOUS lanes on a 16-aligned boundary (Dq/8
// and Dkv/8 are multiples of 16), and the q / k / v roles also start on
// 16-aligned boundaries -- so each 16-lane half-warp is a single (role, head)
// group whose lanes all reach (or all skip) this reduction. The mask names
// EXACTLY that half-warp (not the whole warp), so it stays valid even when the
// k and v roles split a warp (odd num_tp_kv_heads, e.g. 1 KV head/rank) or the
// q/k boundary falls mid-warp -- an xor butterfly then leaves every lane in the
// group with the full sum (no cross-role shuffle, no exited-lane in the mask).
__device__ __forceinline__ float head_rmsnorm_inv(float ss, float eps) {
  const unsigned hmask = 0xFFFFu << (threadIdx.x & 16u);  // this thread's 16-lane group
#pragma unroll
  for (int off = 8; off > 0; off >>= 1)
    ss += __shfl_xor_sync(hmask, ss, off, 16);
  return rsqrtf(ss / static_cast<float>(kHeadDim) + eps);
}

__device__ __forceinline__ uint8_t mxfp8_scale_byte(float local_amax, float* descale) {
  const unsigned qmask = 0xFu << (threadIdx.x & 28u);
#pragma unroll
  for (int off = 1; off < static_cast<int>(kMXFP8BlockLanes); off <<= 1) {
    local_amax = fmaxf(local_amax, __shfl_xor_sync(qmask, local_amax, off, kMXFP8BlockLanes));
  }
  const float amax = fmaxf(local_amax, 1.0e-30f);
  float scale_biased = ceilf(log2f(amax / kE4M3Max)) + 127.0f;
  scale_biased = fminf(fmaxf(scale_biased, 0.0f), 254.0f);
  *descale = exp2f(scale_biased - 127.0f);
  return static_cast<uint8_t>(scale_biased);
}

__device__ __forceinline__ void store_mxfp8_vec(const float (&x)[kVecElems], void* dst, uint8_t* sf, uint32_t c) {
  float local_amax = 0.0f;
#pragma unroll
  for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
    local_amax = fmaxf(local_amax, fabsf(x[j]));
  }
  float descale;
  const uint8_t sf_byte = mxfp8_scale_byte(local_amax, &descale);
  if ((c & (kMXFP8Block - 1)) == 0) *sf = sf_byte;

  union {
    __nv_fp8x2_e4m3 fp8x2[4];
    uint2 raw;
  } u;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float x0 = fminf(fmaxf(x[2 * j] / descale, -kE4M3Max), kE4M3Max);
    const float x1 = fminf(fmaxf(x[2 * j + 1] / descale, -kE4M3Max), kE4M3Max);
    u.fp8x2[j] = __nv_fp8x2_e4m3(make_float2(x0, x1));
  }
  *reinterpret_cast<uint2*>(static_cast<uint8_t*>(dst) + c) = u.raw;
}

struct AttnPrologueParams {
  const void* __restrict__ qkvr;  // [T, row_stride] packed projection output
  // sconv (verify) per K/V path
  const void* __restrict__ k_cache;  // [pool, W-1, Dkv]
  const void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;  // int32 [B] (PAD == -1)
  const void* __restrict__ cache_mask;     // bool  [B]
  const void* __restrict__ k_weight;       // [Dkv, W]
  const void* __restrict__ v_weight;
  void* __restrict__ k_inter;  // [max_bs, q, W-1, Dkv]
  void* __restrict__ v_inter;
  // norms
  const void* __restrict__ q_gamma;  // [head_dim]
  const void* __restrict__ k_gamma;  // [head_dim]
  const void* __restrict__ log_tau;  // fp32 [T] per-token q scale (null -> off)
  float eps;
  // outputs
  void* __restrict__ q_out;      // [T, Dq]
  void* __restrict__ k_out;      // [T, Dkv]
  void* __restrict__ v_out;      // [T, Dkv]
  const void* __restrict__ loc;  // int64 [T] KV slots
  void* __restrict__ k_buf;      // [slots, Hkv, head_dim]
  void* __restrict__ v_buf;
  void* __restrict__ sfq;  // uint8 [T, Hq, head_dim/32] when USE_MXFP8
  void* __restrict__ sfk;  // uint8 BlockScaledBasicChunk layout
  void* __restrict__ sfv;
  int64_t qkvr_stride_t;
  int64_t q_off;  // elem offsets of the q/k/v slices within a qkvr row
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t inter_stride_b;
  int64_t inter_stride_t;
  int64_t inter_stride_w;
  int64_t kv_buf_stride;  // elems per KV slot row (= Hkv * head_dim)
  uint32_t T;
  uint32_t q;  // draft_token_num
  uint32_t dq;
  uint32_t dkv;
  uint32_t page_size;
};

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_STORE, bool USE_MXFP8,
          bool USE_PDL>
__global__ __launch_bounds__(1024, 1) void inkling_attn_prologue_kernel(const __grid_constant__ AttnPrologueParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>);
  static_assert(!USE_MXFP8 || DO_STORE, "MXFP8 prologue quantization owns the KV store");
  constexpr int W1 = W - 1;
  constexpr uint32_t SF = kHeadDim / kMXFP8Block;
  const uint32_t t = blockIdx.x;
  const uint32_t seq = t / p.q;
  const int bos = static_cast<int>(seq * p.q);
  const uint32_t tq = t - seq * p.q;
  const uint32_t nq = p.dq / kVecElems;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t vi = threadIdx.x;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t row = static_cast<int64_t>(t) * p.qkvr_stride_t;

  const int ci = static_cast<const int32_t*>(p.cache_indices)[seq];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;
  const float cm = (valid && static_cast<const bool*>(p.cache_mask)[seq]) ? 1.0f : 0.0f;

  // PDL: as in the decode kernel, the immediately-preceding qkvr GEMM only
  // produces qkvr -- gammas, tau, conv weights/cache prefix and metadata are
  // prefetched before PDLWaitPrimary; every qkvr read stays behind the wait.
  if (vi < nq) {
    // ---------------- q path: per-head RMSNorm only ----------------
    const uint32_t c = vi * kVecElems;
    const uint4 gqraw = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.q_gamma) + (c % kHeadDim));
    const auto* gq = reinterpret_cast<const __nv_bfloat16*>(&gqraw);
    const bool do_tau = p.log_tau != nullptr;
    float tau = 0.0f;
    if (do_tau) tau = static_cast<const float*>(p.log_tau)[t];
    device::PDLWaitPrimary<USE_PDL>();
    const uint4 raw = *reinterpret_cast<const uint4*>(base + row + p.q_off + c);
    float x[kVecElems];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw)[j]);
      ss += x[j] * x[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(x[2 * j] * inv * __bfloat162float(gq[2 * j]),
                                   x[2 * j + 1] * inv * __bfloat162float(gq[2 * j + 1]));
    }
    if (do_tau) {
      // Fused log-scaling tau: multiply the bf16-ROUNDED normed q (matching
      // the unfused {norm kernel -> apply_log_scaling_tau} rounding exactly);
      // on the MXFP8 path this scales BEFORE quantization.
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        o[j] = __floats2bfloat162_rn(f.x * tau, f.y * tau);
      }
    }
    if constexpr (USE_MXFP8) {
      float q_quant[kVecElems];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        q_quant[2 * j] = f.x;
        q_quant[2 * j + 1] = f.y;
      }
      const uint32_t sf_idx = static_cast<uint32_t>(t) * (p.dq / kMXFP8Block) + c / kMXFP8Block;
      store_mxfp8_vec(
          q_quant,
          static_cast<uint8_t*>(p.q_out) + static_cast<int64_t>(t) * p.dq,
          static_cast<uint8_t*>(p.sfq) + sf_idx,
          c);
    } else {
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.q_out) + static_cast<int64_t>(t) * p.dq + c) =
          *reinterpret_cast<const uint4*>(o);
    }
    device::PDLTriggerSecondary<USE_PDL>();
    return;
  }
  if (vi >= nq + 2 * nkv) return;

  // ---------------- k / v paths: conv + save_windows (+ k norm) + store ----
  const bool is_k = vi < nq + nkv;
  const uint32_t ch = (is_k ? vi - nq : vi - nq - nkv) * kVecElems;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  const auto* cp = static_cast<const __nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(is_k ? p.k_weight : p.v_weight);
  auto* ip = static_cast<__nv_bfloat16*>(is_k ? p.k_inter : p.v_inter);
  const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + ch;

  uint4 pref[W1];
  __nv_bfloat16 wt[kVecElems][W];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    pref[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t wrow = static_cast<int64_t>(ch + j) * p.weight_stride_d;
    if constexpr (W == 4) {
      if (p.weight_stride_d == W) {
        *reinterpret_cast<uint2*>(wt[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
        continue;
      }
    }
#pragma unroll
    for (int w = 0; w < W; ++w)
      wt[j][w] = wp[wrow + w];
  }

  uint4 gkraw = make_uint4(0, 0, 0, 0);
  if (is_k) {
    gkraw = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.k_gamma) + (ch % kHeadDim));
  }
  device::PDLWaitPrimary<USE_PDL>();
  // In-seq neighbor rows (pre-conv x straight from qkvr) + own row.
  const uint4 xcur = *reinterpret_cast<const uint4*>(base + row + x_off + ch);
  uint4 xn[W1];
#pragma unroll
  for (int j = 1; j <= W1; ++j) {
    const int n = static_cast<int>(t) - j;
    if (n >= bos) {
      xn[j - 1] = *reinterpret_cast<const uint4*>(base + static_cast<int64_t>(n) * p.qkvr_stride_t + x_off + ch);
    }
  }

  float y[kVecElems];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xcur)[j]);
    float acc = 0.0f;
#pragma unroll
    for (int iw = 0; iw < W1; ++iw) {
      const int shifted = static_cast<int>(t) - W1 + iw;
      float tap = 0.0f;
      if (shifted >= bos) {
        tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xn[W1 - 1 - iw])[j]);
      } else {
        const int prefix_pos = shifted - bos + W1;
        if (prefix_pos >= 0) {
          tap = cm * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&pref[prefix_pos])[j]);
        }
      }
      acc += tap * __bfloat162float(wt[j][iw]);
    }
    acc += xj * __bfloat162float(wt[j][W1]);
    if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
    if constexpr (USE_RESIDUAL) acc += xj;
    y[j] = acc;
  }

  if (valid) {  // save_intermediate_conv_windows (raw copies)
    auto* op = ip + static_cast<int64_t>(seq) * p.inter_stride_b + static_cast<int64_t>(tq) * p.inter_stride_t + ch;
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const int position = static_cast<int>(tq) + 1 + w;
      uint4 val;
      if (position < W1) {
        val = pref[position];
      } else {
        const int g = bos + position - W1;
        val = (g == static_cast<int>(t)) ? xcur : xn[t - g - 1];
      }
      *reinterpret_cast<uint4*>(op + w * p.inter_stride_w) = val;
    }
  }

  __nv_bfloat162 o[4];
  if (is_k) {
    // per-head RMSNorm on the conv output (16-lane groups). Round to bf16
    // FIRST: the unfused pipeline writes the conv output to memory as bf16
    // before the norm kernel reads it back.
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __bfloat162float(__float2bfloat16_rn(y[j]));
      ss += y[j] * y[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gk = reinterpret_cast<const __nv_bfloat16*>(&gkraw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          y[2 * j] * inv * __bfloat162float(gk[2 * j]), y[2 * j + 1] * inv * __bfloat162float(gk[2 * j + 1]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
  }
  const uint4 ov = *reinterpret_cast<const uint4*>(o);
  auto* out = static_cast<__nv_bfloat16*>(is_k ? p.k_out : p.v_out);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * p.dkv + ch) = ov;
  // Fused KV store (DO_STORE). Only used for full-attention layers writing a
  // plain bf16 [slots, Hkv, head_dim] pool indexed directly by out_cache_loc;
  // SWA/local layers keep the backend store (swa_out_cache_loc + its own pool),
  // so the caller passes DO_STORE=false and save_kv_cache=True there.
  if (DO_STORE) {
    const int64_t kv_slot = static_cast<const int64_t*>(p.loc)[t];
    if (kv_slot >= 0) {  // SWA full->swa translation can yield -1 sentinels
      if constexpr (USE_MXFP8) {
        float xo[kVecElems];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(o[j]);
          xo[2 * j] = f.x;
          xo[2 * j + 1] = f.y;
        }
        auto* buf = static_cast<uint8_t*>(is_k ? p.k_buf : p.v_buf);
        auto* sfb = static_cast<uint8_t*>(is_k ? p.sfk : p.sfv);
        const int64_t po = kv_slot % static_cast<int64_t>(p.page_size);
        const int64_t sf_base = ((kv_slot / static_cast<int64_t>(p.page_size)) * (p.dkv / kHeadDim) + ch / kHeadDim) *
                                    (kMXFP8Block * (p.page_size / kMXFP8Block) * SF) +
                                (po % kMXFP8Block) * ((p.page_size / kMXFP8Block) * SF) + (po / kMXFP8Block) * SF +
                                (ch % kHeadDim) / kMXFP8Block;
        store_mxfp8_vec(xo, buf + kv_slot * p.kv_buf_stride, sfb + sf_base, ch);
      } else {
        auto* buf = static_cast<__nv_bfloat16*>(is_k ? p.k_buf : p.v_buf);
        *reinterpret_cast<uint4*>(buf + kv_slot * p.kv_buf_stride + ch) = ov;
      }
    }
  }
  device::PDLTriggerSecondary<USE_PDL>();
}

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool USE_MXFP8, bool USE_PDL>
struct AttnPrologueKernel {
  static void
  run(tvm::ffi::TensorView qkvr,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_weight,
      tvm::ffi::TensorView k_inter,
      tvm::ffi::TensorView v_inter,
      tvm::ffi::TensorView q_gamma,
      tvm::ffi::TensorView k_gamma,
      double eps,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_buf,
      tvm::ffi::TensorView v_buf,
      tvm::ffi::TensorView sfq,
      tvm::ffi::TensorView sfk,
      tvm::ffi::TensorView sfv,
      int64_t q_off,
      int64_t k_off,
      int64_t v_off,
      int64_t q_num,
      int64_t do_store,
      int64_t page_size,
      tvm::ffi::TensorView log_tau) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(qkvr.size(0));
    const uint32_t B = static_cast<uint32_t>(cache_indices.size(0));
    const uint32_t dq = static_cast<uint32_t>(q_out.size(1));
    const uint32_t dkv = static_cast<uint32_t>(k_out.size(1));
    RuntimeCheck(q_num > 0 && T == B * static_cast<uint32_t>(q_num), "T != B*q");
    RuntimeCheck(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dims % head_dim");
    RuntimeCheck((dq / kVecElems) % kHeadLanes == 0, "q lanes must tile heads");
    RuntimeCheck(
        qkvr.stride(1) == 1 && qkvr.stride(0) % kVecElems == 0, "qkvr must be row-major with 16B-aligned rows");
    RuntimeCheck(
        q_off % kVecElems == 0 && k_off % kVecElems == 0 && v_off % kVecElems == 0,
        "slice offsets must be 16B aligned");
    RuntimeCheck(k_buf.stride(0) == v_buf.stride(0), "kv buf stride mismatch");
    RuntimeCheck(k_cache.stride(2) == 1 && v_cache.stride(2) == 1, "conv caches must be channel-contiguous");
    RuntimeCheck(
        k_inter.stride(3) == 1 && v_inter.stride(3) == 1 && k_inter.stride(0) == v_inter.stride(0) &&
            k_inter.stride(1) == v_inter.stride(1) && k_inter.stride(2) == v_inter.stride(2),
        "inter buffers must be channel-contiguous with equal strides");
    const uint32_t lanes = dq / kVecElems + 2 * (dkv / kVecElems);
    RuntimeCheck(lanes <= 1024, "token lanes must fit one block");
    if constexpr (USE_MXFP8) {
      RuntimeCheck(do_store, "MXFP8 fused prologue requires do_store=True");
      RuntimeCheck(dq % kMXFP8Block == 0 && dkv % kMXFP8Block == 0, "MXFP8 dims must tile 32-element scale blocks");
      RuntimeCheck(page_size > 0 && page_size % kMXFP8Block == 0, "MXFP8 page size must tile 32-token scale blocks");
      RuntimeCheck(is_type<fp8_e4m3_t>(q_out.dtype()), "MXFP8 q_out must be fp8_e4m3");
      RuntimeCheck(
          is_type<fp8_e4m3_t>(k_buf.dtype()) && is_type<fp8_e4m3_t>(v_buf.dtype()),
          "MXFP8 KV buffers must be fp8_e4m3");
      RuntimeCheck(
          is_type<uint8_t>(sfq.dtype()) && is_type<uint8_t>(sfk.dtype()) && is_type<uint8_t>(sfv.dtype()),
          "MXFP8 scale buffers must be passed as uint8 views");
      RuntimeCheck(q_out.stride(1) == 1 && q_out.stride(0) == dq, "MXFP8 q_out must be contiguous");
      RuntimeCheck(
          sfq.stride(2) == 1 && sfq.stride(1) == kHeadDim / kMXFP8Block, "MXFP8 sfq must be contiguous [T, Hq, 4]");
      const int64_t hkv = dkv / kHeadDim;
      const int64_t sf_dim = kHeadDim / kMXFP8Block;
      const int64_t page_chunks = page_size / kMXFP8Block;
      RuntimeCheck(sfk.ndim() == 5 && sfv.ndim() == 5, "MXFP8 SFK/SFV must be 5D interleaved");
      RuntimeCheck(
          sfk.size(1) == hkv && sfv.size(1) == hkv && sfk.size(2) == kMXFP8Block &&
              sfv.size(2) == kMXFP8Block && sfk.size(3) == page_chunks && sfv.size(3) == page_chunks &&
              sfk.size(4) == sf_dim && sfv.size(4) == sf_dim,
          "MXFP8 SFK/SFV must use [pages, Hkv, 32, page/32, 4] layout");
      RuntimeCheck(
          sfk.stride(4) == 1 && sfv.stride(4) == 1 && sfk.stride(3) == sf_dim && sfv.stride(3) == sf_dim &&
              sfk.stride(2) == page_chunks * sf_dim && sfv.stride(2) == page_chunks * sf_dim &&
              sfk.stride(1) == kMXFP8Block * page_chunks * sf_dim &&
              sfv.stride(1) == kMXFP8Block * page_chunks * sf_dim,
          "MXFP8 SFK/SFV must be contiguous BlockScaledBasicChunk layout");
      RuntimeCheck(k_buf.stride(0) % kMXFP8Block == 0, "MXFP8 kv buf row alignment");
    } else {
      RuntimeCheck(k_buf.stride(0) % kVecElems == 0, "kv buf rows must be 16B aligned");
      RuntimeCheck(is_type<DType>(q_out.dtype()), "q_out dtype mismatch");
    }

    const bool do_tau = log_tau.numel() > 0;
    if (do_tau) {
      RuntimeCheck(is_type<fp32_t>(log_tau.dtype()), "log_tau must be fp32");
      RuntimeCheck(log_tau.IsContiguous(), "log_tau must be contiguous");
      RuntimeCheck(log_tau.numel() >= qkvr.size(0), "log_tau smaller than T");
    }
    const auto params = AttnPrologueParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .v_weight = v_weight.data_ptr(),
        .k_inter = k_inter.data_ptr(),
        .v_inter = v_inter.data_ptr(),
        .q_gamma = q_gamma.data_ptr(),
        .k_gamma = k_gamma.data_ptr(),
        .log_tau = do_tau ? log_tau.data_ptr() : nullptr,
        .eps = static_cast<float>(eps),
        .q_out = q_out.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .loc = loc.data_ptr(),
        .k_buf = k_buf.data_ptr(),
        .v_buf = v_buf.data_ptr(),
        .sfq = USE_MXFP8 ? sfq.data_ptr() : nullptr,
        .sfk = USE_MXFP8 ? sfk.data_ptr() : nullptr,
        .sfv = USE_MXFP8 ? sfv.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .q_off = q_off,
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .weight_stride_d = k_weight.stride(0),
        .inter_stride_b = k_inter.stride(0),
        .inter_stride_t = k_inter.stride(1),
        .inter_stride_w = k_inter.stride(2),
        .kv_buf_stride = k_buf.stride(0),
        .T = T,
        .q = static_cast<uint32_t>(q_num),
        .dq = dq,
        .dkv = dkv,
        .page_size = static_cast<uint32_t>(page_size),
    };
    RuntimeCheck(
        k_cache.stride(0) == v_cache.stride(0) && k_cache.stride(1) == v_cache.stride(1) &&
            k_weight.stride(0) == v_weight.stride(0),
        "k/v cache+weight strides must match");
    const uint32_t block = div_ceil(lanes, 32u) * 32u;
    const auto kernel =
        do_store ? inkling_attn_prologue_kernel<DType, W, USE_SILU, USE_RESIDUAL, true, USE_MXFP8, USE_PDL>
                 : inkling_attn_prologue_kernel<DType, W, USE_SILU, USE_RESIDUAL, false, false, USE_PDL>;
    LaunchKernel(dim3{T}, dim3{block}, qkvr.device()).enable_pdl(USE_PDL)(kernel, params);
  }
};

// ---------------------------------------------------------------------------
// DECODE variant: {k/v decode-conv + conv-cache shift-update (+ track) +
// qk-norm + KV store}. Decode is one token per sequence, so the conv taps come
// from the working conv cache (W-1 history + current token) -- no cross-token
// reads, no AR, no barrier: every token is an independent block. Mirrors
// fused_decode_update.cuh's conv + shift-update + prefix-cache track-copy, then
// adds the per-head q/k RMSNorm and the bf16 KV-cache store (so attention runs
// save_kv_cache=False). Replaces 2x fused_decode_update + qk-norm + set_kv_buffer.
struct AttnPrologueDecodeParams {
  const void* __restrict__ qkvr;
  void* __restrict__ k_cache;  // [pool, W-1, Dkv], in-place shift-update
  void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;  // int32 [T]  per-token slot (PAD == -1)
  const void* __restrict__ cache_mask;     // bool  [T]  per-token history gate
  const void* __restrict__ k_weight;       // [Dkv, W]
  const void* __restrict__ v_weight;
  const void* __restrict__ track_mask;     // bool  [T]  (DO_TRACK)
  const void* __restrict__ track_indices;  // int64 [T]  (DO_TRACK)
  const void* __restrict__ q_gamma;
  const void* __restrict__ k_gamma;
  const void* __restrict__ log_tau;  // fp32 [T] per-token q scale (null -> off)
  float eps;
  void* __restrict__ q_out;
  void* __restrict__ k_out;
  void* __restrict__ v_out;
  const void* __restrict__ loc;
  void* __restrict__ k_buf;
  void* __restrict__ v_buf;
  void* __restrict__ sfq;  // uint8 [T, Hq, head_dim/32] when USE_MXFP8
  void* __restrict__ sfk;  // uint8 BlockScaledBasicChunk layout
  void* __restrict__ sfv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t track_idx_stride;
  int64_t kv_buf_stride;
  uint32_t T;
  uint32_t dq;
  uint32_t dkv;
  uint32_t page_size;
};

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_TRACK, bool DO_STORE, bool USE_MXFP8,
          bool USE_PDL>
__global__ __launch_bounds__(1024, 1) void inkling_attn_prologue_decode_kernel(
    const __grid_constant__ AttnPrologueDecodeParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>);
  static_assert(!USE_MXFP8 || DO_STORE, "MXFP8 decode prologue quantization owns the KV store");
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t nq = p.dq / kVecElems;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t vi = threadIdx.x;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t row = static_cast<int64_t>(t) * p.qkvr_stride_t;

  // PDL: the ONLY input the immediately-preceding kernel (the qkvr
  // projection GEMM) produces is qkvr itself. Gammas, tau, conv weights,
  // metadata and the conv-cache history (last written a full step earlier)
  // are prefetched BEFORE PDLWaitPrimary so their latency hides under the
  // primary's tail; every qkvr read stays behind the wait.
  if (vi < nq) {
    // -------- q path: per-head RMSNorm only --------
    const uint32_t c = vi * kVecElems;
    const uint4 gqraw = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.q_gamma) + (c % kHeadDim));
    const auto* gq = reinterpret_cast<const __nv_bfloat16*>(&gqraw);
    const bool do_tau = p.log_tau != nullptr;
    float tau = 0.0f;
    if (do_tau) tau = static_cast<const float*>(p.log_tau)[t];
    device::PDLWaitPrimary<USE_PDL>();
    const uint4 raw = *reinterpret_cast<const uint4*>(base + row + p.q_off + c);
    float x[kVecElems];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw)[j]);
      ss += x[j] * x[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(x[2 * j] * inv * __bfloat162float(gq[2 * j]),
                                   x[2 * j + 1] * inv * __bfloat162float(gq[2 * j + 1]));
    }
    if (do_tau) {
      // Fused log-scaling tau: multiply the bf16-ROUNDED normed q (matching
      // the unfused {norm kernel -> apply_log_scaling_tau} rounding exactly);
      // on the MXFP8 path this scales BEFORE quantization.
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        o[j] = __floats2bfloat162_rn(f.x * tau, f.y * tau);
      }
    }
    if constexpr (USE_MXFP8) {
      float q_quant[kVecElems];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        q_quant[2 * j] = f.x;
        q_quant[2 * j + 1] = f.y;
      }
      const uint32_t sf_idx = static_cast<uint32_t>(t) * (p.dq / kMXFP8Block) + c / kMXFP8Block;
      store_mxfp8_vec(
          q_quant,
          static_cast<uint8_t*>(p.q_out) + static_cast<int64_t>(t) * p.dq,
          static_cast<uint8_t*>(p.sfq) + sf_idx,
          c);
    } else {
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.q_out) +
                                static_cast<int64_t>(t) * p.dq + c) =
          *reinterpret_cast<const uint4*>(o);
    }
    device::PDLTriggerSecondary<USE_PDL>();
    return;
  }
  if (vi >= nq + 2 * nkv) return;

  // -------- k / v paths: decode-conv + cache update (+ track) (+ k-norm) + store --------
  const bool is_k = vi < nq + nkv;
  const uint32_t ch = (is_k ? vi - nq : vi - nq - nkv) * kVecElems;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  auto* cp = static_cast<__nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(is_k ? p.k_weight : p.v_weight);

  const int ci = static_cast<const int32_t*>(p.cache_indices)[t];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;  // PAD lanes still emit y/store, never write conv cache
  const float cm = static_cast<const bool*>(p.cache_mask)[t] ? 1.0f : 0.0f;
  const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + ch;

  // History taps + current token -> registers BEFORE any cache write (RAW-safe).
  uint4 hist[W1];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    hist[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
  }
  __nv_bfloat16 wt[kVecElems][W];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t wrow = static_cast<int64_t>(ch + j) * p.weight_stride_d;
    if constexpr (W == 4) {
      if (p.weight_stride_d == W) {
        *reinterpret_cast<uint2*>(wt[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
        continue;
      }
    }
#pragma unroll
    for (int w = 0; w < W; ++w) wt[j][w] = wp[wrow + w];
  }
  uint4 gkraw = make_uint4(0, 0, 0, 0);
  if (is_k) {
    gkraw = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.k_gamma) + (ch % kHeadDim));
  }
  device::PDLWaitPrimary<USE_PDL>();
  const uint4 xv = *reinterpret_cast<const uint4*>(base + row + x_off + ch);

  // conv (fused_decode_update semantics): W-1 cached taps (cm-gated) + current.
  float y[kVecElems];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xv)[j]);
    float acc = 0.0f;
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const float h = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&hist[w])[j]);
      acc += h * cm * __bfloat162float(wt[j][w]);
    }
    acc += xj * __bfloat162float(wt[j][W1]);
    if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
    if constexpr (USE_RESIDUAL) acc += xj;
    y[j] = acc;
  }

  // cache shift-update (valid lanes): new[iw] = (iw<W-2)?(cm?hist[iw+1]:0):xv.
  if (valid) {
    int64_t track_base = 0;
    bool do_tr = false;
    if constexpr (DO_TRACK) {
      do_tr = static_cast<const bool*>(p.track_mask)[t];
      if (do_tr) {
        const int64_t tslot = static_cast<const int64_t*>(
            p.track_indices)[static_cast<int64_t>(t) * p.track_idx_stride];
        track_base = tslot * p.cache_stride_slot + ch;
      }
    }
    const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const uint4 nv = (w < W1 - 1) ? ((cm != 0.0f) ? hist[w + 1] : zero) : xv;
      *reinterpret_cast<uint4*>(&cp[cache_base + w * p.cache_stride_w]) = nv;
      if constexpr (DO_TRACK) {
        if (do_tr) *reinterpret_cast<uint4*>(&cp[track_base + w * p.cache_stride_w]) = nv;
      }
    }
  }

  // k-norm (round to bf16 first, matching the unfused HBM round trip); v passes through.
  __nv_bfloat162 o[4];
  if (is_k) {
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __bfloat162float(__float2bfloat16_rn(y[j]));
      ss += y[j] * y[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gk = reinterpret_cast<const __nv_bfloat16*>(&gkraw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(y[2 * j] * inv * __bfloat162float(gk[2 * j]),
                                   y[2 * j + 1] * inv * __bfloat162float(gk[2 * j + 1]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
  }
  const uint4 ov = *reinterpret_cast<const uint4*>(o);
  auto* out = static_cast<__nv_bfloat16*>(is_k ? p.k_out : p.v_out);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * p.dkv + ch) = ov;
  if (DO_STORE && valid) {
    const int64_t kv_slot = static_cast<const int64_t*>(p.loc)[t];
    if (kv_slot >= 0) {
      if constexpr (USE_MXFP8) {
        float xo[kVecElems];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(o[j]);
          xo[2 * j] = f.x;
          xo[2 * j + 1] = f.y;
        }
        auto* buf = static_cast<uint8_t*>(is_k ? p.k_buf : p.v_buf);
        auto* sfb = static_cast<uint8_t*>(is_k ? p.sfk : p.sfv);
        const int64_t po = kv_slot % static_cast<int64_t>(p.page_size);
        const int64_t sf_base = ((kv_slot / static_cast<int64_t>(p.page_size)) * (p.dkv / kHeadDim) + ch / kHeadDim) *
                                    (kMXFP8Block * (p.page_size / kMXFP8Block) * (kHeadDim / kMXFP8Block)) +
                                (po % kMXFP8Block) * ((p.page_size / kMXFP8Block) * (kHeadDim / kMXFP8Block)) +
                                (po / kMXFP8Block) * (kHeadDim / kMXFP8Block) + (ch % kHeadDim) / kMXFP8Block;
        store_mxfp8_vec(xo, buf + kv_slot * p.kv_buf_stride, sfb + sf_base, ch);
      } else {
        auto* buf = static_cast<__nv_bfloat16*>(is_k ? p.k_buf : p.v_buf);
        *reinterpret_cast<uint4*>(buf + kv_slot * p.kv_buf_stride + ch) = ov;
      }
    }
  }
  device::PDLTriggerSecondary<USE_PDL>();
}

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool USE_MXFP8, bool USE_PDL>
struct AttnPrologueDecodeKernel {
  static void
  run(tvm::ffi::TensorView qkvr,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_weight,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_indices,
      tvm::ffi::TensorView q_gamma,
      tvm::ffi::TensorView k_gamma,
      double eps,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_buf,
      tvm::ffi::TensorView v_buf,
      tvm::ffi::TensorView sfq,
      tvm::ffi::TensorView sfk,
      tvm::ffi::TensorView sfv,
      int64_t q_off,
      int64_t k_off,
      int64_t v_off,
      int64_t do_track,
      int64_t do_store,
      int64_t page_size,
      tvm::ffi::TensorView log_tau) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(qkvr.size(0));
    const uint32_t dq = static_cast<uint32_t>(q_out.size(1));
    const uint32_t dkv = static_cast<uint32_t>(k_out.size(1));
    RuntimeCheck(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dims % head_dim");
    RuntimeCheck((dq / kVecElems) % kHeadLanes == 0, "q lanes must tile heads");
    RuntimeCheck(qkvr.stride(1) == 1 && qkvr.stride(0) % kVecElems == 0,
                 "qkvr must be row-major with 16B-aligned rows");
    RuntimeCheck(q_off % kVecElems == 0 && k_off % kVecElems == 0 &&
                     v_off % kVecElems == 0, "slice offsets must be 16B aligned");
    RuntimeCheck(k_cache.stride(2) == 1 && v_cache.stride(2) == 1,
                 "conv caches must be channel-contiguous");
    RuntimeCheck(k_cache.stride(0) == v_cache.stride(0) &&
                     k_cache.stride(1) == v_cache.stride(1) &&
                     k_weight.stride(0) == v_weight.stride(0),
                 "k/v cache+weight strides must match");
    const uint32_t lanes = dq / kVecElems + 2 * (dkv / kVecElems);
    RuntimeCheck(lanes <= 1024, "token lanes must fit one block");
    RuntimeCheck(k_buf.stride(0) == v_buf.stride(0), "kv buf stride mismatch");
    if constexpr (USE_MXFP8) {
      RuntimeCheck(do_store, "MXFP8 fused decode prologue requires do_store=True");
      RuntimeCheck(dq % kMXFP8Block == 0 && dkv % kMXFP8Block == 0, "MXFP8 dims must tile 32-element scale blocks");
      RuntimeCheck(page_size > 0 && page_size % kMXFP8Block == 0, "MXFP8 page size must tile 32-token scale blocks");
      RuntimeCheck(is_type<fp8_e4m3_t>(q_out.dtype()), "MXFP8 q_out must be fp8_e4m3");
      RuntimeCheck(
          is_type<fp8_e4m3_t>(k_buf.dtype()) && is_type<fp8_e4m3_t>(v_buf.dtype()),
          "MXFP8 KV buffers must be fp8_e4m3");
      RuntimeCheck(
          is_type<uint8_t>(sfq.dtype()) && is_type<uint8_t>(sfk.dtype()) && is_type<uint8_t>(sfv.dtype()),
          "MXFP8 scale buffers must be passed as uint8 views");
      RuntimeCheck(q_out.stride(1) == 1 && q_out.stride(0) == dq, "MXFP8 q_out must be contiguous");
      RuntimeCheck(
          sfq.stride(2) == 1 && sfq.stride(1) == kHeadDim / kMXFP8Block, "MXFP8 sfq must be contiguous [T, Hq, 4]");
      const int64_t hkv = dkv / kHeadDim;
      const int64_t sf_dim = kHeadDim / kMXFP8Block;
      const int64_t page_chunks = page_size / kMXFP8Block;
      RuntimeCheck(sfk.ndim() == 5 && sfv.ndim() == 5, "MXFP8 SFK/SFV must be 5D interleaved");
      RuntimeCheck(
          sfk.size(1) == hkv && sfv.size(1) == hkv && sfk.size(2) == kMXFP8Block &&
              sfv.size(2) == kMXFP8Block && sfk.size(3) == page_chunks && sfv.size(3) == page_chunks &&
              sfk.size(4) == sf_dim && sfv.size(4) == sf_dim,
          "MXFP8 SFK/SFV must use [pages, Hkv, 32, page/32, 4] layout");
      RuntimeCheck(
          sfk.stride(4) == 1 && sfv.stride(4) == 1 && sfk.stride(3) == sf_dim && sfv.stride(3) == sf_dim &&
              sfk.stride(2) == page_chunks * sf_dim && sfv.stride(2) == page_chunks * sf_dim &&
              sfk.stride(1) == kMXFP8Block * page_chunks * sf_dim &&
              sfv.stride(1) == kMXFP8Block * page_chunks * sf_dim,
          "MXFP8 SFK/SFV must be contiguous BlockScaledBasicChunk layout");
      RuntimeCheck(k_buf.stride(0) % kMXFP8Block == 0, "MXFP8 kv buf row alignment");
    } else {
      RuntimeCheck(k_buf.stride(0) % kVecElems == 0, "kv buf rows must be 16B aligned");
      RuntimeCheck(is_type<DType>(q_out.dtype()), "q_out dtype mismatch");
    }

    const bool do_tau = log_tau.numel() > 0;
    if (do_tau) {
      RuntimeCheck(is_type<fp32_t>(log_tau.dtype()), "log_tau must be fp32");
      RuntimeCheck(log_tau.IsContiguous(), "log_tau must be contiguous");
      RuntimeCheck(log_tau.numel() >= qkvr.size(0), "log_tau smaller than T");
    }
    const auto params = AttnPrologueDecodeParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .v_weight = v_weight.data_ptr(),
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_indices = do_track ? track_indices.data_ptr() : nullptr,
        .q_gamma = q_gamma.data_ptr(),
        .k_gamma = k_gamma.data_ptr(),
        .log_tau = do_tau ? log_tau.data_ptr() : nullptr,
        .eps = static_cast<float>(eps),
        .q_out = q_out.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .loc = loc.data_ptr(),
        .k_buf = k_buf.data_ptr(),
        .v_buf = v_buf.data_ptr(),
        .sfq = USE_MXFP8 ? sfq.data_ptr() : nullptr,
        .sfk = USE_MXFP8 ? sfk.data_ptr() : nullptr,
        .sfv = USE_MXFP8 ? sfv.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .q_off = q_off,
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .weight_stride_d = k_weight.stride(0),
        .track_idx_stride = do_track ? track_indices.stride(0) : 0,
        .kv_buf_stride = k_buf.stride(0),
        .T = T,
        .dq = dq,
        .dkv = dkv,
        .page_size = static_cast<uint32_t>(page_size),
    };
    const uint32_t block = div_ceil(lanes, 32u) * 32u;
    auto pick = [&](auto tr, auto st, auto mx) {
      return inkling_attn_prologue_decode_kernel<DType, W, USE_SILU, USE_RESIDUAL,
                                             decltype(tr)::value, decltype(st)::value, decltype(mx)::value,
                                             USE_PDL>;
    };
    const bool tr = do_track != 0, st = do_store != 0;
    const auto kernel =
        USE_MXFP8
            ? (tr ? pick(std::true_type{}, std::true_type{}, std::true_type{})
                  : pick(std::false_type{}, std::true_type{}, std::true_type{}))
            : (tr ? (st ? pick(std::true_type{}, std::true_type{}, std::false_type{})
                       : pick(std::true_type{}, std::false_type{}, std::false_type{}))
                  : (st ? pick(std::false_type{}, std::true_type{}, std::false_type{})
                        : pick(std::false_type{}, std::false_type{}, std::false_type{})));
    LaunchKernel(dim3{T}, dim3{block}, qkvr.device()).enable_pdl(USE_PDL)(kernel, params);
  }
};

// ---------------------------------------------------------------------------
// EXTEND variant: {k/v extend-conv + per-head q/k RMSNorm + KV store} in the
// main kernel (one block per token, varlen sequences via si/cu -- the verify
// kernel's dataflow with the fixed q-per-seq mapping generalized), plus a
// TINY trailing kernel for the conv-cache update + prefix-cache track. The
// update cannot ride the main kernel: early tokens of a sequence read the OLD
// cache prefix rows from other blocks while the seq-end block would overwrite
// them, and with grid = T (up to 16K blocks) there is no co-residency for a
// grid barrier. The trailing kernel is B*(W-1) rows over both caches --
// microseconds -- and both launches are on one stream, so ordering is free.
// Replaces {2x causal_conv1d + apply_qk_norm + 2x update_sconv_cache (+track)
// + the backend KV store}.
struct AttnPrologueExtendParams {
  const void* __restrict__ qkvr;  // [T, row_stride] packed projection output
  const void* __restrict__ k_cache;  // [pool, W-1, Dkv] (read-only here)
  const void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;  // int32 [B] (PAD == -1)
  const void* __restrict__ cache_mask;     // bool  [B] has_init & valid
  const void* __restrict__ cu;             // int64 [B+1] query_start_loc
  const void* __restrict__ si;             // int32 [T] token -> sequence
  const void* __restrict__ k_weight;       // [Dkv, W]
  const void* __restrict__ v_weight;
  const void* __restrict__ q_gamma;  // [head_dim]
  const void* __restrict__ k_gamma;  // [head_dim]
  const void* __restrict__ log_tau;  // fp32 [T] per-token q scale (null -> off)
  float eps;
  void* __restrict__ q_out;      // [T, Dq]
  void* __restrict__ k_out;      // [T, Dkv]
  void* __restrict__ v_out;      // [T, Dkv]
  const void* __restrict__ loc;  // int64 [T] KV slots
  void* __restrict__ k_buf;      // [slots, Hkv, head_dim]
  void* __restrict__ v_buf;
  void* __restrict__ sfq;  // uint8 [T, Hq, head_dim/32] when USE_MXFP8
  void* __restrict__ sfk;  // uint8 BlockScaledBasicChunk layout
  void* __restrict__ sfv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t kv_buf_stride;
  uint32_t T;
  uint32_t dq;
  uint32_t dkv;
  uint32_t page_size;
};

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_STORE, bool USE_MXFP8,
          bool USE_PDL>
__global__ __launch_bounds__(1024, 1) void inkling_attn_prologue_extend_kernel(
    const __grid_constant__ AttnPrologueExtendParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>);
  static_assert(!USE_MXFP8 || DO_STORE, "MXFP8 prologue quantization owns the KV store");
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t seq = static_cast<uint32_t>(static_cast<const int32_t*>(p.si)[t]);
  const int bos = static_cast<int>(static_cast<const int64_t*>(p.cu)[seq]);
  const uint32_t nq = p.dq / kVecElems;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t vi = threadIdx.x;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t row = static_cast<int64_t>(t) * p.qkvr_stride_t;

  const int ci = static_cast<const int32_t*>(p.cache_indices)[seq];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;
  const float cm = static_cast<const bool*>(p.cache_mask)[seq] ? 1.0f : 0.0f;

  // PDL + load restructure (same as the decode/verify kernels): the
  // immediately-preceding qkvr GEMM only produces qkvr; gammas, tau, conv
  // weights/prefix and the si/cu/ci/cm metadata are prefetched before
  // PDLWaitPrimary, every qkvr read stays behind it. The TRAILING
  // kv_conv_update launch stays non-PDL: it overwrites cache rows this
  // kernel reads, so it must keep full completion ordering.
  if (vi < nq) {
    // ---------------- q path: per-head RMSNorm only ----------------
    const uint32_t c = vi * kVecElems;
    const uint4 gqraw = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.q_gamma) + (c % kHeadDim));
    const auto* gq = reinterpret_cast<const __nv_bfloat16*>(&gqraw);
    const bool do_tau = p.log_tau != nullptr;
    float tau = 0.0f;
    if (do_tau) tau = static_cast<const float*>(p.log_tau)[t];
    device::PDLWaitPrimary<USE_PDL>();
    const uint4 raw = *reinterpret_cast<const uint4*>(base + row + p.q_off + c);
    float x[kVecElems];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw)[j]);
      ss += x[j] * x[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(x[2 * j] * inv * __bfloat162float(gq[2 * j]),
                                   x[2 * j + 1] * inv * __bfloat162float(gq[2 * j + 1]));
    }
    if (do_tau) {
      // Fused log-scaling tau: multiply the bf16-ROUNDED normed q (matching
      // the unfused {norm kernel -> apply_log_scaling_tau} rounding exactly);
      // on the MXFP8 path this scales BEFORE quantization.
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        o[j] = __floats2bfloat162_rn(f.x * tau, f.y * tau);
      }
    }
    if constexpr (USE_MXFP8) {
      float q_quant[kVecElems];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        q_quant[2 * j] = f.x;
        q_quant[2 * j + 1] = f.y;
      }
      const uint32_t sf_idx = static_cast<uint32_t>(t) * (p.dq / kMXFP8Block) + c / kMXFP8Block;
      store_mxfp8_vec(
          q_quant,
          static_cast<uint8_t*>(p.q_out) + static_cast<int64_t>(t) * p.dq,
          static_cast<uint8_t*>(p.sfq) + sf_idx,
          c);
    } else {
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.q_out) + static_cast<int64_t>(t) * p.dq + c) =
          *reinterpret_cast<const uint4*>(o);
    }
    device::PDLTriggerSecondary<USE_PDL>();
    return;
  }
  if (vi >= nq + 2 * nkv) return;

  // ---------------- k / v paths: varlen conv (+ k norm) + store ----------
  const bool is_k = vi < nq + nkv;
  const uint32_t ch = (is_k ? vi - nq : vi - nq - nkv) * kVecElems;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  const auto* cp = static_cast<const __nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(is_k ? p.k_weight : p.v_weight);
  const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + ch;

  uint4 pref[W1];
  __nv_bfloat16 wt[kVecElems][W];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    pref[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t wrow = static_cast<int64_t>(ch + j) * p.weight_stride_d;
    if constexpr (W == 4) {
      if (p.weight_stride_d == W) {
        *reinterpret_cast<uint2*>(wt[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
        continue;
      }
    }
#pragma unroll
    for (int w = 0; w < W; ++w)
      wt[j][w] = wp[wrow + w];
  }

  uint4 gkraw = make_uint4(0, 0, 0, 0);
  if (is_k) {
    gkraw = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.k_gamma) + (ch % kHeadDim));
  }
  device::PDLWaitPrimary<USE_PDL>();
  // In-seq neighbor rows (pre-conv x straight from qkvr) + own row.
  const uint4 xcur = *reinterpret_cast<const uint4*>(base + row + x_off + ch);
  uint4 xn[W1];
#pragma unroll
  for (int j = 1; j <= W1; ++j) {
    const int n = static_cast<int>(t) - j;
    if (n >= bos) {
      xn[j - 1] = *reinterpret_cast<const uint4*>(base + static_cast<int64_t>(n) * p.qkvr_stride_t + x_off + ch);
    }
  }

  float y[kVecElems];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xcur)[j]);
    float acc = 0.0f;
#pragma unroll
    for (int iw = 0; iw < W1; ++iw) {
      const int shifted = static_cast<int>(t) - W1 + iw;
      float tap = 0.0f;
      if (shifted >= bos) {
        tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xn[W1 - 1 - iw])[j]);
      } else {
        const int prefix_pos = shifted - bos + W1;
        if (prefix_pos >= 0) {
          tap = cm * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&pref[prefix_pos])[j]);
        }
      }
      acc += tap * __bfloat162float(wt[j][iw]);
    }
    acc += xj * __bfloat162float(wt[j][W1]);
    if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
    if constexpr (USE_RESIDUAL) acc += xj;
    y[j] = acc;
  }

  __nv_bfloat162 o[4];
  if (is_k) {
    // per-head RMSNorm on the conv output (16-lane groups). Round to bf16
    // FIRST: the unfused pipeline writes the conv output to memory as bf16
    // before the norm kernel reads it back.
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __bfloat162float(__float2bfloat16_rn(y[j]));
      ss += y[j] * y[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gk = reinterpret_cast<const __nv_bfloat16*>(&gkraw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          y[2 * j] * inv * __bfloat162float(gk[2 * j]), y[2 * j + 1] * inv * __bfloat162float(gk[2 * j + 1]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
  }
  const uint4 ov = *reinterpret_cast<const uint4*>(o);
  auto* out = static_cast<__nv_bfloat16*>(is_k ? p.k_out : p.v_out);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * p.dkv + ch) = ov;
  if (DO_STORE) {
    const int64_t kv_slot = static_cast<const int64_t*>(p.loc)[t];
    if (kv_slot >= 0) {  // SWA full->swa translation can yield -1 sentinels
      if constexpr (USE_MXFP8) {
        float xo[kVecElems];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(o[j]);
          xo[2 * j] = f.x;
          xo[2 * j + 1] = f.y;
        }
        auto* buf = static_cast<uint8_t*>(is_k ? p.k_buf : p.v_buf);
        auto* sfb = static_cast<uint8_t*>(is_k ? p.sfk : p.sfv);
        const int64_t po = kv_slot % static_cast<int64_t>(p.page_size);
        constexpr uint32_t SF = kHeadDim / kMXFP8Block;
        const int64_t sf_base = ((kv_slot / static_cast<int64_t>(p.page_size)) * (p.dkv / kHeadDim) + ch / kHeadDim) *
                                    (kMXFP8Block * (p.page_size / kMXFP8Block) * SF) +
                                (po % kMXFP8Block) * ((p.page_size / kMXFP8Block) * SF) + (po / kMXFP8Block) * SF +
                                (ch % kHeadDim) / kMXFP8Block;
        store_mxfp8_vec(xo, buf + kv_slot * p.kv_buf_stride, sfb + sf_base, ch);
      } else {
        auto* buf = static_cast<__nv_bfloat16*>(is_k ? p.k_buf : p.v_buf);
        *reinterpret_cast<uint4*>(buf + kv_slot * p.kv_buf_stride + ch) = ov;
      }
    }
  }
  device::PDLTriggerSecondary<USE_PDL>();
}

// Trailing conv-cache update + prefix-cache track for BOTH k/v caches. One
// thread owns all W-1 rows of one (sequence, role, channel-vec) triple --
// RAW-safe load-all-then-store, mirroring update_sconv_cache (and the
// AR-sconv kernel's phase 3, with the pre-conv x read straight from qkvr).
struct KvConvUpdateParams {
  const void* __restrict__ qkvr;
  void* __restrict__ k_cache;  // [pool, W-1, Dkv], in-place
  void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;  // int32 [B] (PAD == -1)
  const void* __restrict__ has_init;       // bool  [B]
  const void* __restrict__ cu;             // int64 [B+1]
  const void* __restrict__ track_rows;     // int64 [B, W-1] (DO_TRACK)
  const void* __restrict__ track_mask;     // bool  [B]      (DO_TRACK)
  const void* __restrict__ track_dst;      // int64 [B]      (DO_TRACK)
  int64_t qkvr_stride_t;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t track_dst_stride;
  uint32_t B;
  uint32_t dkv;
};

template <int W, bool DO_TRACK>
__global__ void inkling_kv_conv_update_kernel(const __grid_constant__ KvConvUpdateParams p) {
  constexpr int W1 = W - 1;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t items = p.B * 2u * nkv;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= items) return;
  const uint32_t b = idx / (2u * nkv);
  const uint32_t r = idx % (2u * nkv);
  const bool is_k = r < nkv;
  const uint32_t ch = (is_k ? r : r - nkv) * kVecElems;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  auto* cp = static_cast<__nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* cu = static_cast<const int64_t*>(p.cu);
  const int slot = static_cast<const int32_t*>(p.cache_indices)[b];
  const int64_t qlen = cu[b + 1] - cu[b];
  if (slot != kPadSlot && qlen > 0) {
    const bool hs = static_cast<const bool*>(p.has_init)[b];
    const int64_t cb = static_cast<int64_t>(slot) * p.cache_stride_slot + ch;
    uint4 old_reg[W1];
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      old_reg[w] = *reinterpret_cast<const uint4*>(&cp[cb + w * p.cache_stride_w]);
    }
    const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      uint4 nv;
      if (qlen >= W1 - w) {
        const int64_t row = cu[b + 1] - W1 + w;
        nv = *reinterpret_cast<const uint4*>(base + row * p.qkvr_stride_t + x_off + ch);
      } else {
        uint4 shift = zero;
#pragma unroll
        for (int src = 0; src < W1; ++src) {
          if (src == w + qlen) shift = old_reg[src];
        }
        nv = hs ? shift : zero;
      }
      *reinterpret_cast<uint4*>(&cp[cb + w * p.cache_stride_w]) = nv;
    }
  }
  if constexpr (DO_TRACK) {
    if (static_cast<const bool*>(p.track_mask)[b]) {
      const int64_t dst = static_cast<const int64_t*>(
          p.track_dst)[static_cast<int64_t>(b) * p.track_dst_stride];
      const int64_t db = dst * p.cache_stride_slot + ch;
      const auto* trows = static_cast<const int64_t*>(p.track_rows);
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        const int64_t row = trows[static_cast<int64_t>(b) * W1 + w];
        *reinterpret_cast<uint4*>(&cp[db + w * p.cache_stride_w]) =
            *reinterpret_cast<const uint4*>(base + row * p.qkvr_stride_t + x_off + ch);
      }
    }
  }
}

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool USE_MXFP8, bool USE_PDL>
struct AttnPrologueExtendKernel {
  static void
  run(tvm::ffi::TensorView qkvr,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView cache_indices,  // int32 [B] raw slots (PAD == -1)
      tvm::ffi::TensorView cache_mask,     // bool  [B] has_init & valid
      tvm::ffi::TensorView has_init,       // bool  [B]
      tvm::ffi::TensorView cu,             // int64 [B+1]
      tvm::ffi::TensorView si,             // int32 [T]
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_weight,
      tvm::ffi::TensorView track_rows,  // int64 [B, W-1] (numel 0 -> no track)
      tvm::ffi::TensorView track_mask,  // bool [B]
      tvm::ffi::TensorView track_dst,   // int64 [B] (possibly strided)
      tvm::ffi::TensorView q_gamma,
      tvm::ffi::TensorView k_gamma,
      double eps,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_buf,
      tvm::ffi::TensorView v_buf,
      tvm::ffi::TensorView sfq,
      tvm::ffi::TensorView sfk,
      tvm::ffi::TensorView sfv,
      int64_t q_off,
      int64_t k_off,
      int64_t v_off,
      int64_t do_store,
      int64_t page_size,
      int64_t do_cache_update,
      tvm::ffi::TensorView log_tau) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(qkvr.size(0));
    const uint32_t B = static_cast<uint32_t>(cache_indices.size(0));
    const uint32_t dq = static_cast<uint32_t>(q_out.size(1));
    const uint32_t dkv = static_cast<uint32_t>(k_out.size(1));
    RuntimeCheck(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dims % head_dim");
    RuntimeCheck((dq / kVecElems) % kHeadLanes == 0, "q lanes must tile heads");
    RuntimeCheck(qkvr.stride(1) == 1 && qkvr.stride(0) % kVecElems == 0,
                 "qkvr must be row-major with 16B-aligned rows");
    RuntimeCheck(q_off % kVecElems == 0 && k_off % kVecElems == 0 &&
                     v_off % kVecElems == 0, "slice offsets must be 16B aligned");
    RuntimeCheck(si.size(0) >= T, "si must cover T tokens");
    RuntimeCheck(cu.size(0) == B + 1, "cu must be [B+1]");
    RuntimeCheck(cache_mask.size(0) == B && has_init.size(0) == B, "per-seq arrays must be [B]");
    RuntimeCheck(k_cache.stride(2) == 1 && v_cache.stride(2) == 1,
                 "conv caches must be channel-contiguous");
    RuntimeCheck(k_cache.stride(0) == v_cache.stride(0) &&
                     k_cache.stride(1) == v_cache.stride(1) &&
                     k_weight.stride(0) == v_weight.stride(0),
                 "k/v cache+weight strides must match");
    const uint32_t lanes = dq / kVecElems + 2 * (dkv / kVecElems);
    RuntimeCheck(lanes <= 1024, "token lanes must fit one block");
    RuntimeCheck(k_buf.stride(0) == v_buf.stride(0), "kv buf stride mismatch");
    const bool do_track = track_mask.numel() > 0;
    if (do_track) {
      RuntimeCheck(track_rows.numel() > 0, "extend track needs gather rows");
    }
    if constexpr (USE_MXFP8) {
      RuntimeCheck(do_store, "MXFP8 fused extend prologue requires do_store=True");
      RuntimeCheck(dq % kMXFP8Block == 0 && dkv % kMXFP8Block == 0, "MXFP8 dims must tile 32-element scale blocks");
      RuntimeCheck(page_size > 0 && page_size % kMXFP8Block == 0, "MXFP8 page size must tile 32-token scale blocks");
      RuntimeCheck(is_type<fp8_e4m3_t>(q_out.dtype()), "MXFP8 q_out must be fp8_e4m3");
      RuntimeCheck(
          is_type<fp8_e4m3_t>(k_buf.dtype()) && is_type<fp8_e4m3_t>(v_buf.dtype()),
          "MXFP8 KV buffers must be fp8_e4m3");
      RuntimeCheck(
          is_type<uint8_t>(sfq.dtype()) && is_type<uint8_t>(sfk.dtype()) && is_type<uint8_t>(sfv.dtype()),
          "MXFP8 scale buffers must be passed as uint8 views");
      RuntimeCheck(q_out.stride(1) == 1 && q_out.stride(0) == dq, "MXFP8 q_out must be contiguous");
      RuntimeCheck(
          sfq.stride(2) == 1 && sfq.stride(1) == kHeadDim / kMXFP8Block, "MXFP8 sfq must be contiguous [T, Hq, 4]");
      const int64_t hkv = dkv / kHeadDim;
      const int64_t sf_dim = kHeadDim / kMXFP8Block;
      const int64_t page_chunks = page_size / kMXFP8Block;
      RuntimeCheck(sfk.ndim() == 5 && sfv.ndim() == 5, "MXFP8 SFK/SFV must be 5D interleaved");
      RuntimeCheck(
          sfk.size(1) == hkv && sfv.size(1) == hkv && sfk.size(2) == kMXFP8Block &&
              sfv.size(2) == kMXFP8Block && sfk.size(3) == page_chunks && sfv.size(3) == page_chunks &&
              sfk.size(4) == sf_dim && sfv.size(4) == sf_dim,
          "MXFP8 SFK/SFV must use [pages, Hkv, 32, page/32, 4] layout");
      RuntimeCheck(
          sfk.stride(4) == 1 && sfv.stride(4) == 1 && sfk.stride(3) == sf_dim && sfv.stride(3) == sf_dim &&
              sfk.stride(2) == page_chunks * sf_dim && sfv.stride(2) == page_chunks * sf_dim &&
              sfk.stride(1) == kMXFP8Block * page_chunks * sf_dim &&
              sfv.stride(1) == kMXFP8Block * page_chunks * sf_dim,
          "MXFP8 SFK/SFV must be contiguous BlockScaledBasicChunk layout");
      RuntimeCheck(k_buf.stride(0) % kMXFP8Block == 0, "MXFP8 kv buf row alignment");
    } else {
      RuntimeCheck(k_buf.stride(0) % kVecElems == 0, "kv buf rows must be 16B aligned");
      RuntimeCheck(is_type<DType>(q_out.dtype()), "q_out dtype mismatch");
    }
    if (T == 0) return;

    const bool do_tau = log_tau.numel() > 0;
    if (do_tau) {
      RuntimeCheck(is_type<fp32_t>(log_tau.dtype()), "log_tau must be fp32");
      RuntimeCheck(log_tau.IsContiguous(), "log_tau must be contiguous");
      RuntimeCheck(log_tau.numel() >= qkvr.size(0), "log_tau smaller than T");
    }
    const auto params = AttnPrologueExtendParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .cu = cu.data_ptr(),
        .si = si.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .v_weight = v_weight.data_ptr(),
        .q_gamma = q_gamma.data_ptr(),
        .k_gamma = k_gamma.data_ptr(),
        .log_tau = do_tau ? log_tau.data_ptr() : nullptr,
        .eps = static_cast<float>(eps),
        .q_out = q_out.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .loc = loc.data_ptr(),
        .k_buf = k_buf.data_ptr(),
        .v_buf = v_buf.data_ptr(),
        .sfq = USE_MXFP8 ? sfq.data_ptr() : nullptr,
        .sfk = USE_MXFP8 ? sfk.data_ptr() : nullptr,
        .sfv = USE_MXFP8 ? sfv.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .q_off = q_off,
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .weight_stride_d = k_weight.stride(0),
        .kv_buf_stride = k_buf.stride(0),
        .T = T,
        .dq = dq,
        .dkv = dkv,
        .page_size = static_cast<uint32_t>(page_size),
    };
    const uint32_t block = div_ceil(lanes, 32u) * 32u;
    const auto kernel = do_store
        ? inkling_attn_prologue_extend_kernel<DType, W, USE_SILU, USE_RESIDUAL, true, USE_MXFP8, USE_PDL>
        : inkling_attn_prologue_extend_kernel<DType, W, USE_SILU, USE_RESIDUAL, false, false, USE_PDL>;
    LaunchKernel(dim3{T}, dim3{block}, qkvr.device()).enable_pdl(USE_PDL)(kernel, params);

    // DRAFT_EXTEND_V2 passes do_cache_update=false: its conv state must
    // reflect only num_accept_tokens, so the caller runs the accept-gated
    // update (_update_sconv_cache_for_draft_extend) instead of this
    // seq-end-window trailing kernel.
    if (!do_cache_update) return;
    const auto uparams = KvConvUpdateParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .has_init = has_init.data_ptr(),
        .cu = cu.data_ptr(),
        .track_rows = do_track ? track_rows.data_ptr() : nullptr,
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_dst = do_track ? track_dst.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .track_dst_stride = do_track ? track_dst.stride(0) : 0,
        .B = B,
        .dkv = dkv,
    };
    const uint32_t uitems = B * 2u * (dkv / kVecElems);
    const uint32_t ublock = 256;
    const uint32_t ugrid = div_ceil(uitems, ublock);
    const auto ukernel = do_track ? inkling_kv_conv_update_kernel<W, true>
                                  : inkling_kv_conv_update_kernel<W, false>;
    LaunchKernel(dim3{ugrid}, dim3{ublock}, qkvr.device())(ukernel, uparams);
  }
};

}  // namespace
