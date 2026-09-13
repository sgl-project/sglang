// GLM-5.2 DSA indexer, the logits kernel: paged MQA FP8 logits.  Raw HIP for
// gfx950.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int i32x8;
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int i32x4;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float f32x4;

namespace dsa_logits {

#define PAGE_TOK 64
#define HD 128
#define TOK_STRIDE 132
#define PAGE_BYTES (PAGE_TOK * TOK_STRIDE) // 8448
#define K_BYTES (PAGE_TOK * HD)            // 8192

union V8 {
  i32x8 v;
  i32x4 h[2];
};

__device__ __forceinline__ f32x4 mfma128(i32x8 a, i32x8 b, f32x4 c) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127,
                                                          0, 127);
}

// wave64 lane exchanges on the VALU instead of through LDS:
//   xor32 <- v_permlane64_b32 (lane l reads lane l^32)
//   xor16 <- v_permlanex16_b32 with identity selectors (lane l reads lane l^16)
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int i32x2;
__device__ __forceinline__ float xor32(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane32_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 32) ? r[0] : r[1]);
}
__device__ __forceinline__ float xor16(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane16_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 16) ? r[0] : r[1]);
}

// The top-k stage's histogram, built here while this kernel waits on KV loads.
// Its three invariants are enforced rather than described: the shared key
// (order_bin_fast), the npages clamp, and do_hist with its topk TORCH_CHECK.
#define LG_CBITS 6
#define LG_CBINS (1 << LG_CBITS)

// Fused paged-MQA FP8 logits + in-loop fine histogram + coarse summary.  The
// histogram is what lets the top-k stage skip a separate scoring pass.

// order_key16(x) >> LOWB in 4 VALU ops after the cvt, with no branch and no
// v_cndmask.  Bit-identical to topk_transform.cu's order_key16(x) >> LOWB for
// every finite x and for +/-0.
template <int LOWB>
__device__ __forceinline__ uint32_t order_bin_fast(float x) {
  const uint32_t h = (uint32_t)__half_as_ushort(__float2half_rn(x));
  const uint32_t m = (uint32_t)((int32_t)__float_as_int(x) >> 31);
  return (h >> LOWB) ^ (0x8000u >> LOWB) ^ (m & (0x7fffu >> LOWB));
}

template <int WARPS, int HB>
__global__ __launch_bounds__(WARPS * 64) void logits_hist_m(
    const uint8_t *__restrict__ q, const uint8_t *__restrict__ kv,
    const float *__restrict__ wgt, const int *__restrict__ seqlens,
    const int *__restrict__ ptable, float *__restrict__ out,
    unsigned int *__restrict__ ghist, int heads, int max_pages, int out_stride,
    int topk) {
  constexpr int NBIN = 1 << HB;
  constexpr int LOWB = 16 - HB;
  constexpr int GHS = NBIN + LG_CBINS;
  constexpr int NTH = WARPS * 64;
  constexpr int PERT = NBIN / NTH; // GLOBAL bins per thread (coarse map)
  constexpr int CSH = HB - LG_CBITS;
  constexpr int GRP = NTH / LG_CBINS;
  static_assert(NBIN >= 2 && (NBIN % 2) == 0, "paired flush needs even bins");
  static_assert(NBIN % NTH == 0 && NTH >= LG_CBINS, "");
  static_assert(PERT <= LG_CBINS && (LG_CBINS % PERT) == 0,
                "a thread's contiguous run must sit inside one coarse bin");
  __shared__ unsigned int s_hist[NBIN];

  const int row = blockIdx.y;
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int seqlen = seqlens[row];
  // seqlen is device data, so no TORCH_CHECK can bound it.  Without the clamp
  // an oversized row walks into the NEXT row's page table -- valid memory,
  // plausible page ids, confidently wrong logits.
  const int npages_raw = (seqlen + PAGE_TOK - 1) / PAGE_TOK;
  // Bound the walk by BOTH the page table and `out`.  The host only requires
  // that `out` cover whole pages -- a captured table is as wide as the graph
  // while the row is only as long as seqlens says -- so a table wider than
  // `out` is legal, and without the second term an oversized seqlen writes
  // past the end of the row.
  const int pages_pt = max_pages;
  const int pages_out = out_stride / PAGE_TOK;
  const int npages_cap = pages_pt < pages_out ? pages_pt : pages_out;
  const int npages = npages_raw < npages_cap ? npages_raw : npages_cap;
  const int g = lane >> 4, c = lane & 15;
  // Nothing to rank when the row is shorter than topk: everything is selected,
  // so the histogram, its two barriers and the epilogue all compile out.
  //
  // This predicate MUST use the same clamped length the paired top-k kernel
  // applies (it clamps row_ends by the page-table width before its own
  // row_len <= TOPK early return).  A row binned here but skipped there is
  // never zeroed, and the histogram stays dirty for the life of the process.
  const int seqlen_eff =
      seqlen < npages_cap * PAGE_TOK ? seqlen : npages_cap * PAGE_TOK;
  const bool do_hist = (seqlen_eff > topk);

  if (do_hist) {
    for (int i = threadIdx.x; i < NBIN; i += NTH)
      s_hist[i] = 0u;
  }

  const uint8_t *qp =
      q + (size_t)row * (size_t)(heads * HD) + (size_t)c * HD + g * 32;
  V8 qa0, qa1;
  qa0.h[0] = *(const i32x4 *)(qp);
  qa0.h[1] = *(const i32x4 *)(qp + 16);
  qa1.h[0] = *(const i32x4 *)(qp + 16 * HD);
  qa1.h[1] = *(const i32x4 *)(qp + 16 * HD + 16);
  const float *wp = wgt + (size_t)row * heads + g * 4;
  f32x4 w0 = *(const f32x4 *)(wp);
  f32x4 w1 = *(const f32x4 *)(wp + 16);

  const int nwaves = gridDim.x * WARPS;
  const int wid = blockIdx.x * WARPS + warp;
  const int koff = g * 512 + c * 16;

  if (do_hist)
    __syncthreads();

  unsigned int *__restrict__ gh = ghist + (size_t)row * GHS;

  const int *__restrict__ pt = ptable + (size_t)row * max_pages;

  for (int p = wid; p < npages; p += nwaves) {
    const int phys = pt[p];
    const uint8_t *base = kv + (size_t)phys * PAGE_BYTES;
    const uint8_t *kb = base + koff;
    V8 b[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      b[tt].h[0] = *(const i32x4 *)(kb + tt * 2048);
      b[tt].h[1] = *(const i32x4 *)(kb + tt * 2048 + 256);
    }
    const float ks = ((const float *)(base + K_BYTES))[lane];
    float s[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const f32x4 z = {0.f, 0.f, 0.f, 0.f};
      f32x4 a0 = mfma128(qa0.v, b[tt].v, z);
      f32x4 a1 = mfma128(qa1.v, b[tt].v, z);
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a0[i], 0.f), w0[i], acc);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a1[i], 0.f), w1[i], acc);
      s[tt] = acc;
    }
    const bool hi = (g & 2) != 0;
    float A0 = (hi ? s[2] : s[0]) + xor32(hi ? s[0] : s[2], lane);
    float A1 = (hi ? s[3] : s[1]) + xor32(hi ? s[1] : s[3], lane);
    const bool odd = (g & 1) != 0;
    float T = (odd ? A1 : A0) + xor16(odd ? A0 : A1, lane);
    const int pos = p * PAGE_TOK + lane;
    const bool live = pos < seqlen;
    const float v = T * ks;
    out[(size_t)row * out_stride + pos] = live ? v : -INFINITY;
    if (do_hist && live) {
      const uint32_t bb = order_bin_fast<LOWB>(v);
      atomicAdd(&s_hist[bb], 1u);
    }
  }

  if (do_hist) {
    __syncthreads();

    // ---- coarse summary, mapped over the GLOBAL bin index ----
    // Thread tx owns global bins [tx*PERT, tx*PERT+PERT), which lies inside
    // coarse bin (tx*PERT)>>CSH.  Needs no barrier of its own.
    const int g0 = (int)threadIdx.x * PERT;
    unsigned int acc = 0u;
#pragma unroll
    for (int j = 0; j < PERT; ++j)
      acc += s_hist[g0 + j];
#pragma unroll
    for (int o = GRP >> 1; o >= 1; o >>= 1)
      acc += __shfl_down(acc, o, 64);
    if ((threadIdx.x & (GRP - 1)) == 0 && acc)
      atomicAdd(&gh[NBIN + (g0 >> CSH)], acc);

    for (int i = threadIdx.x; i < NBIN / 2; i += NTH) {
      const unsigned long long vv =
          (unsigned long long)s_hist[2 * i] |
          ((unsigned long long)s_hist[2 * i + 1] << 32);
      if (vv)
        atomicAdd((unsigned long long *)(gh + 2 * i), vv);
    }
  }
}

// One kernel ships: logits_hist_m<8, 12>.  WARPS and HB are template arguments,
// so there is nothing to select and nothing to validate at runtime.
static void logits_hist(at::Tensor q, at::Tensor kv, at::Tensor weights,
                        at::Tensor seqlens, at::Tensor page_table,
                        at::Tensor out, at::Tensor ghist,
                        int64_t blocks_per_row, int64_t topk) {
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(weights.is_contiguous() && weights.scalar_type() == at::kFloat,
              "weights must be contiguous fp32");
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.stride(1) == 1,
              "logits out must be fp32 with unit row stride");
  TORCH_CHECK(kv.is_contiguous(), "kv cache must be contiguous");
  TORCH_CHECK(page_table.is_contiguous() &&
                  page_table.scalar_type() == at::kInt,
              "page_table_64 must be contiguous int32");
  TORCH_CHECK(
      seqlens.scalar_type() == at::kInt && seqlens.is_contiguous() &&
          seqlens.numel() >= q.size(0),
      "seqlens must be a contiguous int32 tensor with one entry per row");
  // A captured table is as wide as the graph while the row is as long as
  // seqlens says, so it may exceed `out`; what must hold is that `out` covers
  // whole pages.
  TORCH_CHECK(out.size(1) % PAGE_TOK == 0,
              "out row must be a whole number of pages, got ", out.size(1));
  // PAGE_TOK and TOK_STRIDE are compiled in; a cache laid out differently is
  // read as garbage rather than refused.
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == PAGE_TOK * TOK_STRIDE,
              "kv cache rows must be PAGE_TOK * TOK_STRIDE = ",
              PAGE_TOK * TOK_STRIDE, ", got ", kv.size(1));
  TORCH_CHECK(ghist.scalar_type() == at::kInt, "ghist must be int32");
  // topk_transform.cu hardcodes TOPK, and the two must agree on which rows
  // get binned: a row binned here but skipped there is never zeroed, and the
  // histogram stays dirty for the life of the process.
  TORCH_CHECK(topk == 2048, "the paired top-k kernel is built for k=2048, got ",
              topk);
  // The launch below is logits_hist_m<8, HOST_HB>, so the row stride the kernel
  // writes is fixed here too.  topk_transform.cu checks the same width from its
  // side; without this one a mismatch overruns into the next row's counters.
  constexpr int HOST_HB = 12;
  constexpr int64_t HOST_GHS = (1 << HOST_HB) + LG_CBINS;
  TORCH_CHECK(ghist.numel() == q.size(0) * HOST_GHS, "ghist must be [rows, ",
              HOST_GHS, "], got ", ghist.numel(), " elements for ", q.size(0),
              " rows");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 32 && q.size(2) == HD,
              "q must be [rows, 32, 128], got ", q.sizes());
  TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fnuz ||
                  q.scalar_type() == at::kFloat8_e4m3fn,
              "q must be fp8 e4m3; it is read as raw bytes");
  TORCH_CHECK(kv.scalar_type() == q.scalar_type(),
              "kv cache must have the same fp8 dtype as q");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == q.size(0) &&
                  weights.size(1) == q.size(1),
              "weights must be [rows, heads]");
  TORCH_CHECK(q.size(0) == out.size(0) && q.size(0) == page_table.size(0),
              "rows must match across q / logits / page_table");

  hipLaunchKernelGGL(
      (logits_hist_m<8, HOST_HB>), dim3((int)blocks_per_row, (int)q.size(0)),
      dim3(8 * 64), 0, at::cuda::getCurrentCUDAStream().stream(),
      (const uint8_t *)q.data_ptr(), (const uint8_t *)kv.data_ptr(),
      weights.data_ptr<float>(), seqlens.data_ptr<int>(),
      page_table.data_ptr<int>(), out.data_ptr<float>(),
      (unsigned int *)ghist.data_ptr(), (int)q.size(1), (int)page_table.size(1),
      (int)out.stride(0), (int)topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dsa_logits

using dsa_logits::logits_hist;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("logits_hist", &logits_hist,
        "paged MQA fp8 logits + fine/coarse histogram (gfx950)");
}
