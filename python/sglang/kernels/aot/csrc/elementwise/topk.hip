/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// ROCm implementation of the DSA indexer top-k ops. Same three entry points, same
// semantics and same tensor contracts as csrc/elementwise/topk.cu, which the ROCm
// build takes this file instead of; see include/hip/dsa_topk_coop.cuh for what the
// kernel does differently and why.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/util/Exception.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <cstdlib>
#include <optional>

#include "hip/dsa_topk_coop.cuh"

namespace {

using namespace sgl::dsa_topk;

// The k every caller of these ops uses (DeepSeek-V3.2 and GLM-5.2 index_topk). It is
// a template parameter so the emit bounds and the -1 padding stay compile-time.
constexpr uint32_t kTopK = 2048;

// 12-bit bins (16 KB) plus 4096 ties (32 KB) is 48 KB of LDS, inside the 64 KB static
// ceiling that gfx942 imposes. The bin count is what keeps the tie set small enough to
// fit: see the precision discussion in dsa_topk_coop.cuh.
constexpr uint32_t kHistBits = 12;
constexpr uint32_t kTieCap = 4096;
constexpr uint32_t kBlock = 1024;

using Ws = CoopMbWorkspace<kHistBits, kTieCap>;

// Blocks per row, 0 meaning "use the one-block kernel".
//
// Splitting a row buys two things, both of which only matter when one block per row
// leaves the device idle: it divides the LDS histogram contention that clustered
// logits create, and it gives a row more than one block's worth of memory parallelism.
// It costs three extra launches and a second read of the row in scatter, so once the
// batch alone fills the machine it is a straight loss.
//
// The two gates below are both needed because the terms move independently. The cost
// is close to fixed, while the benefit grows with row length, since that is what sets
// how much contention there is to divide. Measured on MI355X under graph replay (us,
// split against one-block, clustered / diffuse scores, page_size=1 transform):
//   len  32768  b1   57.7/30.7 vs  57.8/26.1     b64   63.4/33.5 vs  58.2/26.5
//   len  49152  b1   68.9/31.0 vs  77.2/31.6     b64   78.2/36.2 vs  78.5/32.3
//   len  65536  b1   81.5/31.1 vs  98.3/37.2     b64   95.1/41.3 vs  99.7/39.7
//   len 100500  b1  102.6/32.1 vs 135.5/46.6     b64  123.7/49.5 vs 143.0/54.7
// Below 64K the win turns into a loss, so short rows stay on the one-block kernel.
// From batch 128 up one block per row already fills the machine, hence requiring at
// least four blocks per row rather than merely more than one; on captured GLM-5.2
// decode logits at 20 rows the split is worth 2.1-2.8x over one-block.
//
// `row_len_hint` is the score buffer's row stride, not the true sequence length, which
// lives on the device. That is deliberate: the decision has to be identical on capture
// and on replay for a graph-captured decode, and the stride is the only length-like
// quantity the host can see. The cost of that is a wide buffer holding short rows,
// which is the 32768/b1 line above -- 18% on a case the kernel already wins by 2x.
int row_split(int batch, int64_t row_len_hint) {
  // Escape hatch for benchmarking the two paths against each other; 0 forces the
  // one-block kernel. An empty value means unset, not zero.
  if (const char* e = std::getenv("SGL_DSA_TOPK_ROW_SPLIT"); e != nullptr && *e != '\0') {
    const int v = std::atoi(e);
    if (v >= 0) {
      return v;
    }
  }
  int dev = 0;
  (void)hipGetDevice(&dev);
  static thread_local int cached_dev = -1;
  static thread_local int cached_cu = 0;
  if (cached_dev != dev) {
    hipDeviceProp_t p{};
    (void)hipGetDeviceProperties(&p, dev);
    cached_dev = dev;
    cached_cu = p.multiProcessorCount;
  }

  constexpr int64_t kMinRowLen = 65536;
  constexpr int kMinSplit = 4;
  // Holds the slice at >= 2048 elements given kMinRowLen, below which a block cannot
  // amortise the global atomics of its own histogram flush.
  constexpr int kMaxSplit = 32;

  if (row_len_hint < kMinRowLen) {
    return 0;
  }
  const int g = std::min(cached_cu / std::max(batch, 1), kMaxSplit);
  return g < kMinSplit ? 0 : g;
}

void launch(
    const float* input,
    int32_t* out_idx,
    const int32_t* row_starts,
    const int32_t* lengths,
    const OutMap& map,
    int batch,
    int64_t stride,
    at::Device device,
    hipStream_t stream) {
  const int g = row_split(batch, stride);

  if (g == 0) {
    CoopParams<kTopK> p{};
    p.input = input;
    p.out_idx = out_idx;
    p.row_starts = row_starts;
    p.lengths = lengths;
    p.map = map;
    p.stride = stride;
    coop_topk_kernel<kTopK, kHistBits, kTieCap, kBlock><<<batch, kBlock, 0, stream>>>(p);
    return;
  }

  // The row-split path's scratch. Taken from the caching allocator rather than a
  // process-level buffer grown with hipMalloc: the address of such a buffer is baked
  // into every HIP graph captured while it was current, both as a kernel argument and
  // as the destination of the memset below, so freeing it leaves those graphs writing
  // to address space the allocator no longer owns.
  const size_t ws_bytes = Ws::bytes(static_cast<size_t>(batch));
  at::Tensor ws = at::empty({static_cast<int64_t>(ws_bytes)}, at::TensorOptions().dtype(at::kByte).device(device));

  CoopMbParams<kTopK> p{};
  p.input = input;
  p.out_idx = out_idx;
  p.row_starts = row_starts;
  p.lengths = lengths;
  p.map = map;
  p.ws = ws.data_ptr();
  p.stride = stride;
  p.batch = static_cast<uint32_t>(batch);

  // Histograms, state and counters must start at zero; the workspace groups them into
  // one contiguous prefix so this is a single launch regardless of batch.
  C10_HIP_CHECK(hipMemsetAsync(p.ws, 0, Ws::zero_bytes(static_cast<size_t>(batch)), stream));

  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(g), 1);
  coop_mb_hist0<kTopK, kHistBits, kTieCap, kBlock><<<grid, kBlock, 0, stream>>>(p, static_cast<uint32_t>(g));
  coop_mb_hist1<kTopK, kHistBits, kTieCap, kBlock><<<grid, kBlock, 0, stream>>>(p, static_cast<uint32_t>(g));
  coop_mb_scatter<kTopK, kHistBits, kTieCap, kBlock><<<grid, kBlock, 0, stream>>>(p, static_cast<uint32_t>(g));
  coop_mb_refine<kTopK, kHistBits, kTieCap, kBlock><<<batch, kBlock, 0, stream>>>(p);
}

#define CHECK_HIP(x) TORCH_CHECK(x.is_cuda(), #x " must be a HIP tensor")
#define CHECK_SAME_DEVICE(x, ref) TORCH_CHECK(x.device() == ref.device(), #x " must be on the same device as " #ref)

struct Common {
  const float* input;
  const int32_t* lengths;
  const int32_t* row_starts;
  int64_t batch;
  int64_t stride;
};

Common
check_common(const at::Tensor& score, const at::Tensor& lengths, const std::optional<at::Tensor>& row_starts_opt) {
  CHECK_HIP(score);
  CHECK_HIP(lengths);
  CHECK_SAME_DEVICE(lengths, score);
  TORCH_CHECK(score.dim() == 2 && score.scalar_type() == at::kFloat, "score must be a float32 [B, L] tensor");
  TORCH_CHECK(score.stride(1) == 1, "score must be contiguous along the last dim");
  const auto B = score.size(0);
  TORCH_CHECK(
      lengths.dim() == 1 && lengths.is_contiguous() && lengths.scalar_type() == at::kInt,
      "lengths must be a contiguous int32 [B] tensor");
  TORCH_CHECK(lengths.size(0) == B, "lengths must have one entry per score row");

  const int32_t* row_starts = nullptr;
  if (row_starts_opt.has_value()) {
    const auto& rs = row_starts_opt.value();
    CHECK_HIP(rs);
    CHECK_SAME_DEVICE(rs, score);
    TORCH_CHECK(
        rs.dim() == 1 && rs.is_contiguous() && rs.scalar_type() == at::kInt,
        "row_starts must be a contiguous int32 [B] tensor");
    TORCH_CHECK(rs.size(0) == B, "row_starts must have one entry per score row");
    row_starts = rs.data_ptr<int32_t>();
  }

  return Common{score.data_ptr<float>(), lengths.data_ptr<int32_t>(), row_starts, B, score.stride(0)};
}

void check_output(const at::Tensor& out, int64_t B) {
  TORCH_CHECK(
      out.dim() == 2 && out.is_contiguous() && out.scalar_type() == at::kInt,
      "output must be a contiguous int32 [B, topk] tensor");
  TORCH_CHECK(out.size(0) == B, "output must have one row per score row");
  TORCH_CHECK(out.size(1) == static_cast<int64_t>(kTopK), "these ops are instantiated for topk=", kTopK, " only");
}

}  // namespace

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  const auto c = check_common(score, lengths, row_starts_opt);
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(score));
  CHECK_HIP(indices);
  CHECK_SAME_DEVICE(indices, score);
  check_output(indices, c.batch);
  if (c.batch == 0) {
    return;
  }

  launch(
      c.input,
      indices.data_ptr<int32_t>(),
      c.row_starts,
      c.lengths,
      OutMap{},
      static_cast<int>(c.batch),
      c.stride,
      score.device(),
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream());
  C10_HIP_CHECK(hipGetLastError());
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  const auto c = check_common(score, lengths, row_starts_opt);
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(score));
  CHECK_HIP(dst_page_table);
  CHECK_HIP(src_page_table);
  CHECK_HIP(cu_seqlens_q);
  CHECK_SAME_DEVICE(dst_page_table, score);
  CHECK_SAME_DEVICE(src_page_table, score);
  CHECK_SAME_DEVICE(cu_seqlens_q, score);
  check_output(dst_page_table, c.batch);
  TORCH_CHECK(
      src_page_table.dim() == 2 && src_page_table.stride(1) == 1 && src_page_table.scalar_type() == at::kInt,
      "src_page_table must be an int32 [prefill_bs, num_slots] tensor, contiguous along the last dim");
  TORCH_CHECK(
      cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous() && cu_seqlens_q.scalar_type() == at::kInt,
      "cu_seqlens_q must be a contiguous int32 tensor");
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(src_page_table.size(0) == prefill_bs, "src_page_table must have one row per sequence");
  TORCH_CHECK(prefill_bs <= c.batch, "cu_seqlens_q describes more sequences than there are score rows");
  if (c.batch == 0) {
    return;
  }

  OutMap map{};
  map.page_table = src_page_table.data_ptr<int32_t>();
  map.pt_stride = src_page_table.stride(0);
  // Decode is one query token per sequence, so the page-table row is the logits row
  // and the per-block search over cu_seqlens_q is dead work. Extend, draft-extend and
  // target-verify all expand several rows per sequence and need the map.
  const bool is_decode = !row_starts_opt.has_value() && prefill_bs == c.batch;
  if (!is_decode) {
    map.cu_seqlens_q = cu_seqlens_q.data_ptr<int32_t>();
    map.prefill_bs = static_cast<uint32_t>(prefill_bs);
  }

  launch(
      c.input,
      dst_page_table.data_ptr<int32_t>(),
      c.row_starts,
      c.lengths,
      map,
      static_cast<int>(c.batch),
      c.stride,
      score.device(),
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream());
  C10_HIP_CHECK(hipGetLastError());
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  const auto c = check_common(score, lengths, row_starts_opt);
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(score));
  CHECK_HIP(topk_indices_ragged);
  CHECK_HIP(topk_indices_offset);
  CHECK_SAME_DEVICE(topk_indices_ragged, score);
  CHECK_SAME_DEVICE(topk_indices_offset, score);
  check_output(topk_indices_ragged, c.batch);
  TORCH_CHECK(
      topk_indices_offset.dim() == 1 && topk_indices_offset.is_contiguous() &&
          topk_indices_offset.scalar_type() == at::kInt,
      "topk_indices_offset must be a contiguous int32 [B] tensor");
  TORCH_CHECK(topk_indices_offset.size(0) == c.batch, "topk_indices_offset must have one entry per score row");
  if (c.batch == 0) {
    return;
  }

  OutMap map{};
  map.offsets = topk_indices_offset.data_ptr<int32_t>();

  launch(
      c.input,
      topk_indices_ragged.data_ptr<int32_t>(),
      c.row_starts,
      c.lengths,
      map,
      static_cast<int>(c.batch),
      c.stride,
      score.device(),
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream());
  C10_HIP_CHECK(hipGetLastError());
}
