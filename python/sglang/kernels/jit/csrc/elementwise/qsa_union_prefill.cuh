// Token unions for the QSA union prefill attention (opt-in, SGLANG_QSA_UNION_PREFILL; see
// srt/layers/attention/qsa/union_prefill.py).
//
// The query rows of each request are cut into groups of kRows consecutive rows. For each group this
// kernel lists every context token that any of its rows attends -- exactly the entries the stock chunk
// prefill kernel reads: columns [0, row_limit) of the row's indices that are >= 0, row_limit being
// min(topk, ceil16(min(topk, visible))) -- once each, in position order, as rows of the packed K/V,
// and for every 128-token tile of that list a bitmask per group row of the tiles' tokens that row
// attends. The union comes from a bitmap of the request's context in shared memory: one atomicOr per
// selected entry, a block-wide prefix of the word popcounts, then each token's rank.
//
// Outputs, per group g (one CTA each):
//   tok[g, 0:nt)                    packed K/V row of each union token; [ntok, ceil128(ntok)) holds oob,
//                                   a row past the descriptor that TMA fills with zeros
//   mask[g, tile, j, 0:4]           bit c of word c / 32 set when group row j attends the tile's token c
//   meta[g] = {row0, rows, ntok, tiles}
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace qsa_union_prefill_detail {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kTile = 128;

struct Params {
  const int32_t* __restrict__ indices;  // [R, topk], row stride idx_stride; positions relative to the request
  const int64_t* __restrict__ cu_q;     // [B + 1] query rows
  const int32_t* __restrict__ kv_lens;  // [B] context length (prefix + extend)
  const int64_t* __restrict__ cu_k;     // [B + 1] first packed K/V row of each request
  int32_t* __restrict__ tok;            // [G, nt]
  uint32_t* __restrict__ mask;          // [G, nt / 128, kRows, 4]
  int32_t* __restrict__ meta;           // [G, 4]
  int64_t idx_stride;
  int32_t topk;
  int32_t batch;
  int32_t groups;
  int32_t nt;     // per-group capacity: a multiple of 128, >= min(kRows * topk, words * 32)
  int32_t words;  // bitmap words in shared memory: >= ceil(max kv_len / 32)
  int32_t oob;    // the zero-filled row
};

template <int kRows>
__global__ void __launch_bounds__(kThreads) qsa_union_prefill_build_kernel(const Params p) {
  extern __shared__ uint32_t smem[];
  uint32_t* bitmap = smem;                 // [words]
  uint32_t* prefix = smem + p.words;       // [words]
  uint32_t* mwords = prefix + p.words;     // [nt / 128, kRows, 4]
  __shared__ int s_b, s_row0, s_rows, s_qrel0, s_qlen, s_kvlen;
  __shared__ int64_t s_kstart;
  __shared__ uint32_t s_warp[kWarps];
  __shared__ uint32_t s_total;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = blockIdx.x;
  if (tid == 0) {
    int found = -1, acc = 0;
    for (int b = 0; b < p.batch; ++b) {
      const int64_t q0 = p.cu_q[b], q1 = p.cu_q[b + 1];
      const int qlen = static_cast<int>(q1 - q0);
      const int ng = (qlen + kRows - 1) / kRows;
      if (g < acc + ng) {
        found = b;
        s_qrel0 = (g - acc) * kRows;
        s_row0 = static_cast<int>(q0) + s_qrel0;
        s_rows = min(kRows, qlen - s_qrel0);
        s_qlen = qlen;
        s_kvlen = p.kv_lens[b];
        s_kstart = p.cu_k[b];
        break;
      }
      acc += ng;
    }
    s_b = found;
  }
  __syncthreads();
  int32_t* meta = p.meta + static_cast<int64_t>(g) * 4;
  if (s_b < 0) {  // more CTAs than groups: an empty group
    if (tid == 0) meta[0] = meta[1] = meta[2] = meta[3] = 0;
    return;
  }
  const int row0 = s_row0, rows = s_rows, qrel0 = s_qrel0, qlen = s_qlen, kv_len = s_kvlen;
  const int64_t k_start = s_kstart;
  const int words = min(p.words, (kv_len + 31) >> 5);
  const int tile_cap = p.nt / kTile;

  for (int w = tid; w < words; w += kThreads) bitmap[w] = 0u;
  for (int w = tid; w < tile_cap * kRows * 4; w += kThreads) mwords[w] = 0u;
  __syncthreads();

  // 1. the union: one bit per attended context position
  for (int j = 0; j < rows; ++j) {
    const int visible = qrel0 + j + kv_len - qlen + 1;
    const int seen = min(p.topk, visible);
    const int limit = min(p.topk, (seen + 15) / 16 * 16);
    const int32_t* idx = p.indices + static_cast<int64_t>(row0 + j) * p.idx_stride;
    for (int e = tid; e < limit; e += kThreads) {
      const int t = idx[e];
      if (t >= 0 && t < (words << 5)) atomicOr(&bitmap[t >> 5], 1u << (t & 31));
    }
  }
  __syncthreads();

  // 2. exclusive prefix of the word popcounts (each thread owns a run of consecutive words)
  const int per = (words + kThreads - 1) / kThreads;
  const int w0 = min(words, tid * per), w1 = min(words, w0 + per);
  uint32_t sum = 0;
  for (int w = w0; w < w1; ++w) sum += __popc(bitmap[w]);
  uint32_t incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s_warp[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    const uint32_t v = lane < kWarps ? s_warp[lane] : 0u;
    uint32_t vi = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t u = __shfl_up_sync(0xffffffffu, vi, o);
      if (lane >= o) vi += u;
    }
    if (lane < kWarps) s_warp[lane] = vi - v;
    if (lane == 31) s_total = vi;
  }
  __syncthreads();
  uint32_t run = s_warp[warp] + incl - sum;
  for (int w = w0; w < w1; ++w) {
    prefix[w] = run;
    run += __popc(bitmap[w]);
  }
  __syncthreads();
  const int ntok = static_cast<int>(s_total);
  const int tiles = (ntok + kTile - 1) / kTile;

  // 3. the union in position order, as packed K/V rows; the last tile's tail reads the zero row
  int32_t* tok = p.tok + static_cast<int64_t>(g) * p.nt;
  for (int w = w0; w < w1; ++w) {
    uint32_t bits = bitmap[w];
    uint32_t pos = prefix[w];
    while (bits) {
      const int bit = __ffs(bits) - 1;
      tok[pos++] = static_cast<int32_t>(k_start + (static_cast<int64_t>(w) << 5) + bit);
      bits &= bits - 1u;
    }
  }
  for (int i = ntok + tid; i < tiles * kTile; i += kThreads) tok[i] = p.oob;

  // 4. per tile and group row, the tile's tokens that row attends
  for (int j = 0; j < rows; ++j) {
    const int visible = qrel0 + j + kv_len - qlen + 1;
    const int seen = min(p.topk, visible);
    const int limit = min(p.topk, (seen + 15) / 16 * 16);
    const int32_t* idx = p.indices + static_cast<int64_t>(row0 + j) * p.idx_stride;
    for (int e = tid; e < limit; e += kThreads) {
      const int t = idx[e];
      if (t >= 0 && t < (words << 5)) {
        const int w = t >> 5;
        const uint32_t below = (1u << (t & 31)) - 1u;
        const uint32_t rank = prefix[w] + __popc(bitmap[w] & below);
        atomicOr(&mwords[((rank >> 7) * kRows + j) * 4 + ((rank >> 5) & 3)], 1u << (rank & 31));
      }
    }
  }
  __syncthreads();
  uint32_t* mask = p.mask + static_cast<int64_t>(g) * tile_cap * kRows * 4;
  for (int i = tid; i < tiles * kRows * 4; i += kThreads) mask[i] = mwords[i];
  if (tid == 0) {
    meta[0] = row0;
    meta[1] = rows;
    meta[2] = ntok;
    meta[3] = tiles;
  }
}

}  // namespace qsa_union_prefill_detail

template <int kRows>
struct QsaUnionPrefillBuild {
  static constexpr auto kernel = qsa_union_prefill_detail::qsa_union_prefill_build_kernel<kRows>;

  static void
  run(const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView cu_q,
      const tvm::ffi::TensorView kv_lens,
      const tvm::ffi::TensorView cu_k,
      const tvm::ffi::TensorView tok,
      const tvm::ffi::TensorView mask,
      const tvm::ffi::TensorView meta,
      const int64_t groups,
      const int64_t nt,
      const int64_t words,
      const int64_t oob) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"index_stride"};
    auto B = SymbolicSize{"batch"};
    auto B1 = SymbolicSize{"batch_plus_one"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, K}).with_strides({S, 1}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({B1}).with_dtype<int64_t>().with_device(device).verify(cu_q);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(kv_lens);
    TensorMatcher({B1}).with_dtype<int64_t>().with_device(device).verify(cu_k);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(tok);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(mask);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(meta);
    RuntimeCheck(B1.unwrap() == B.unwrap() + 1, "cu_q must have batch + 1 entries");
    // a group's union holds at most its rows' entries and at most the context's positions (< words * 32)
    RuntimeCheck(nt % qsa_union_prefill_detail::kTile == 0 && nt >= std::min<int64_t>(kRows * K.unwrap(), words * 32),
                 "bad nt");
    RuntimeCheck(tok.size(0) >= groups * nt, "tok too small");
    RuntimeCheck(mask.size(0) >= groups * (nt / qsa_union_prefill_detail::kTile) * kRows * 4, "mask too small");
    RuntimeCheck(meta.size(0) >= groups * 4, "meta too small");
    if (groups == 0) return;

    const auto params = qsa_union_prefill_detail::Params{
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .cu_q = static_cast<const int64_t*>(cu_q.data_ptr()),
        .kv_lens = static_cast<const int32_t*>(kv_lens.data_ptr()),
        .cu_k = static_cast<const int64_t*>(cu_k.data_ptr()),
        .tok = static_cast<int32_t*>(tok.data_ptr()),
        .mask = static_cast<uint32_t*>(mask.data_ptr()),
        .meta = static_cast<int32_t*>(meta.data_ptr()),
        .idx_stride = S.unwrap(),
        .topk = static_cast<int32_t>(K.unwrap()),
        .batch = static_cast<int32_t>(B.unwrap()),
        .groups = static_cast<int32_t>(groups),
        .nt = static_cast<int32_t>(nt),
        .words = static_cast<int32_t>(words),
        .oob = static_cast<int32_t>(oob),
    };
    const size_t smem = sizeof(uint32_t) * (2 * static_cast<size_t>(words) +
                                            static_cast<size_t>(nt / qsa_union_prefill_detail::kTile) * kRows * 4);
    runtime::set_max_dynamic_smem(kernel, smem);
    LaunchKernel(static_cast<uint32_t>(groups), qsa_union_prefill_detail::kThreads, device.unwrap(), smem)(
        kernel, params);
  }
};

}  // namespace sglang
