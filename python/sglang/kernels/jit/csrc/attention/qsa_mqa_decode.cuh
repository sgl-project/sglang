// QSA indexer decode scoring: the logits of the TileLang paged decode kernels in
// srt/layers/attention/qsa/mqa.py, bit for bit, with pipelined key gathers.
//
// Each logit runs the instructions TileLang emits for those kernels, in the same order:
//   - one 64-key group, one warp per 16 keys: 8 x mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32
//     over head_dim 128 in ascending k (A = 16 keys x 16 dims, B = 8 query heads x 16 dims);
//   - max(score, 0); per key (0 + c(2t)) + c(2t + 1) over the thread's two heads, then two
//     shfl_xor adds (lane distance 2, then 1);
//   - logit = sum / scale, built with --use_fast_math like TileLang's TL_ENABLE_FAST_MATH.
// Only the data movement differs: the R query rows that share a page-table row are staged in
// shared memory once per CTA, each CTA strides over many key groups of its request with
// double-buffered cp.async gathers, and the R rows reuse one load of the A fragments.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace qsa_mqa_decode_detail {

constexpr int kThreads = 128;                   // 4 warps, one per 16 keys of a group
constexpr int kGroup = 64;                      // keys per group (the GEMM M)
constexpr int kHeads = 8;                       // query heads incl. zero padding (the GEMM N)
constexpr int kDim = 128;                       // head_dim (the GEMM K)
constexpr int kPage = 16;                       // compressed keys per page
constexpr int kRowBytes = kDim * 2;             // one bf16 key or query-head row
constexpr int kKBufBytes = kGroup * kRowBytes;  // one key group

struct Params {
  const uint16_t* __restrict__ q;            // [rows, 8, 128] bf16
  const uint16_t* __restrict__ kcache;       // [pages, 16, 1, 128] bf16
  const int32_t* __restrict__ page_table;    // [rows, max_pages]
  const int32_t* __restrict__ context_lens;  // [rows]
  float* __restrict__ logits;                // [rows, max_model_len]
  float scale;
  int32_t rows;
  int32_t max_pages;
  int32_t pages;
  int32_t max_model_len;
  int32_t min_scored_len;
  int64_t pt_stride;
  int64_t logits_stride;
};

/// \brief Byte offset of 16-byte chunk `chunk` of row `row` in a [rows][256 B] tile. The XOR
/// swizzle puts the eight rows of one ldmatrix 8x8 matrix in distinct bank groups.
SGL_DEVICE uint32_t swizzle(int row, int chunk) {
  return static_cast<uint32_t>(row * kRowBytes + ((chunk ^ (row & 7)) << 4));
}

SGL_DEVICE void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int src_bytes = valid ? 16 : 0;  // 0 zero-fills the destination
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}

SGL_DEVICE void ldmatrix_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

SGL_DEVICE void ldmatrix_x2(uint32_t addr, uint32_t (&r)[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

SGL_DEVICE void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/// \brief Issue the cp.async gathers of key group `group` (4 pages of 16 keys) into `kbuf`.
/// Pages at or past `len` are skipped, as in TileLang (their positions are never stored).
SGL_DEVICE void gather_group(const Params& p, const int32_t* __restrict__ pt_row, int group, int len, uint32_t kbuf) {
#pragma unroll
  for (int it = 0; it < kKBufBytes / 16 / kThreads; ++it) {
    const int c = it * kThreads + threadIdx.x;  // 16-byte chunk of the group
    const int key = c >> 4;
    const int chunk = c & 15;
    const int sub_page = key / kPage;
    if (group * kGroup + sub_page * kPage < len) {
      const int col = group * (kGroup / kPage) + sub_page;
      const int page = col < p.max_pages ? pt_row[col] : -1;
      const bool valid = 0 <= page && page < p.pages;
      const int64_t key_row = static_cast<int64_t>(valid ? page : 0) * kPage + key % kPage;
      cp_async16(kbuf + swizzle(key, chunk), p.kcache + key_row * kDim + chunk * 8, valid);
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int R, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void qsa_mqa_decode_kernel(const Params __grid_constant__ p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t kbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));  // 2 key buffers
  const uint32_t qbase = kbase + 2 * kKBufBytes;                                 // then R x 8 heads
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row0 = blockIdx.x * R;
  // The R rows share page prefixes and have increasing lengths: the last row's page-table row
  // and length cover them all.
  const int table_row = ::min(row0 + R - 1, p.rows - 1);

  device::PDLWaitPrimary<kUsePDL>();
  int len[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    len[r] = row0 + r < p.rows ? p.context_lens[row0 + r] : 0;
  }
  const int table_len = p.context_lens[table_row];
  const int num_groups = table_len > p.min_scored_len ? (table_len + kGroup - 1) / kGroup : 0;
  int group = blockIdx.y;
  if (group >= num_groups) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t* __restrict__ pt_row = p.page_table + static_cast<int64_t>(table_row) * p.pt_stride;

  for (int c = tid; c < R * kHeads * (kRowBytes / 16); c += kThreads) {
    const int r = c / (kHeads * 16);
    const int head = (c / 16) % kHeads;
    const int chunk = c & 15;
    const bool valid = row0 + r < p.rows;
    const int64_t q_row = static_cast<int64_t>(valid ? row0 + r : 0) * kHeads + head;
    cp_async16(qbase + r * (kHeads * kRowBytes) + swizzle(head, chunk), p.q + q_row * kDim + chunk * 8, valid);
  }
  gather_group(p, pt_row, group, table_len, kbase);  // one commit group: queries + first keys

  for (int buf = 0; group < num_groups; group += gridDim.y, buf ^= 1) {
    const int next = group + gridDim.y;
    if (next < num_groups) {
      gather_group(p, pt_row, next, table_len, kbase + (buf ^ 1) * kKBufBytes);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();
    const uint32_t kbuf = kbase + buf * kKBufBytes;
    uint32_t a[8][4];
#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {  // this warp's 16 keys, dims [16 ki, 16 ki + 16)
      ldmatrix_x4(kbuf + swizzle(warp * 16 + (lane & 15), ki * 2 + (lane >> 4)), a[ki]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (len[r] <= p.min_scored_len || group * kGroup >= len[r]) continue;  // uniform over the CTA
      float scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const uint32_t qbuf = qbase + r * (kHeads * kRowBytes);
#pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        uint32_t b[2];
        ldmatrix_x2(qbuf + swizzle(lane & 7, ki * 2 + ((lane >> 3) & 1)), b);
        mma_bf16_16816(scores, a[ki], b);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        scores[i] = ::max(scores[i], 0.0f);
      }
      float reduced[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        reduced[i] = (0.0f + scores[i * 2]) + scores[i * 2 + 1];
        reduced[i] = reduced[i] + __shfl_xor_sync(0xffffffffu, reduced[i], 2);
        reduced[i] = reduced[i] + __shfl_xor_sync(0xffffffffu, reduced[i], 1);
      }
      if (lane % 4 == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int pos = group * kGroup + warp * 16 + i * 8 + (lane >> 2);
          if (pos < len[r] && pos < p.max_model_len && row0 + r < p.rows) {
            p.logits[static_cast<int64_t>(row0 + r) * p.logits_stride + pos] = reduced[i] / p.scale;
          }
        }
      }
    }
    __syncthreads();  // the gather issued in the next iteration overwrites the other buffer
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_mqa_decode_detail

/**
 * \brief QSA indexer decode logits (see the file comment).
 *
 * \tparam R       consecutive query rows per CTA; they read the last row's page-table row
 * \tparam kUsePDL launch with programmatic dependent launch
 */
template <int R, bool kUsePDL>
struct QsaMqaDecode {
  static constexpr auto kernel = qsa_mqa_decode_detail::qsa_mqa_decode_kernel<R, kUsePDL>;
  static constexpr int kSmem =
      2 * qsa_mqa_decode_detail::kKBufBytes + R * qsa_mqa_decode_detail::kHeads * qsa_mqa_decode_detail::kRowBytes;
  static_assert(1 <= R && kSmem <= 48 * 1024, "R must keep the dynamic shared memory within the 48 KB default");

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView kcache,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView context_lens,
      const tvm::ffi::TensorView logits,
      double scale,
      int64_t min_scored_len,
      int64_t ctas_per_request) {
    using namespace host;
    namespace d = qsa_mqa_decode_detail;
    auto N = SymbolicSize{"rows"};
    auto P = SymbolicSize{"pages"};
    auto MP = SymbolicSize{"max_pages"};
    auto PS = SymbolicSize{"page_table_stride"};
    auto L = SymbolicSize{"max_model_len"};
    auto LS = SymbolicSize{"logits_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({N, d::kHeads, d::kDim}).with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({P, d::kPage, 1, d::kDim}).with_dtype<bf16_t>().with_device(device).verify(kcache);
    TensorMatcher({N, MP}).with_strides({PS, 1}).with_dtype<int32_t>().with_device(device).verify(page_table);
    TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(context_lens);
    TensorMatcher({N, L}).with_strides({LS, 1}).with_dtype<fp32_t>().with_device(device).verify(logits);
    const int64_t rows = N.unwrap();
    CHECK_HOST(rows % R == 0) << "qsa_mqa_decode: rows " << rows << " not a multiple of " << R;
    CHECK_HOST(ctas_per_request >= 1 && ctas_per_request <= 65535)
        << "qsa_mqa_decode: ctas_per_request " << ctas_per_request;
    if (rows == 0) return;
    const auto params = d::Params{
        .q = static_cast<const uint16_t*>(q.data_ptr()),
        .kcache = static_cast<const uint16_t*>(kcache.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .context_lens = static_cast<const int32_t*>(context_lens.data_ptr()),
        .logits = static_cast<float*>(logits.data_ptr()),
        .scale = static_cast<float>(scale),
        .rows = static_cast<int32_t>(rows),
        .max_pages = static_cast<int32_t>(MP.unwrap()),
        .pages = static_cast<int32_t>(P.unwrap()),
        .max_model_len = static_cast<int32_t>(L.unwrap()),
        .min_scored_len = static_cast<int32_t>(min_scored_len),
        .pt_stride = PS.unwrap(),
        .logits_stride = LS.unwrap(),
    };
    const dim3 grid(static_cast<uint32_t>(rows / R), static_cast<uint32_t>(ctas_per_request));
    LaunchKernel(grid, d::kThreads, device.unwrap(), kSmem).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
