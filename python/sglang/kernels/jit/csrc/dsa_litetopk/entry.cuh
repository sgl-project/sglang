// TVM-FFI launchers for the LiteTopk fused DSA indexer top-k (SM100).
//
// Three primitives (seed_prep / scan / select) stream KV in tiles and fuse
// fp8 MQA scoring (tcgen05 UMMA) + an online bucketed gate + compact top-k,
// so the [num_q, seq_len] logit matrix is never materialized.
//
// The kernels themselves are vendored 1:1 from vLLM PR #48726 (LiteTopk,
// itself ported from the author's FlashInfer TVM-FFI launcher):
//   * dsa_indexer_kernels.cuh -- warp-specialized scan kernel
//   * dsa_indexer.cuh         -- seed_prep / select kernels, TMA helpers,
//                                shape config (GLM DSA: H=32, D=128)
// This file is the only sglang-facing layer; it mirrors the argument order of
// the upstream torch-stable launcher (csrc/.../dsa_litetopk.cu) to keep future
// syncs mechanical. sglang deviations are marked SGLANG DEVIATION in the
// vendored files (-1 index padding) and below (real SM count for the KV-split
// heuristic instead of the hardcoded 148).

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST

#include <sgl_kernel/runtime.cuh>  // For get_sm_count, get_cc_major
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, CHECK_CUDA, fp8_e4m3_t aliases

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "dsa_indexer.cuh"  // vendored kernels + NUM_HEADS/BLOCK_* config

namespace {

using tvm::ffi::TensorView;

// ---------------------------------------------------------------------------
// seed_prep: per-row bucket-space calibration from sample logits.
//   slog [Q, head] fp32 (row stride passed through, unit inner stride).
//   With emit_limit == 0 the sample only calibrates (origin / inv_delta /
//   th_bucket) and zeroes bcount; no seed candidates are emitted.
// ---------------------------------------------------------------------------
void dsa_litetopk_seed_prep(
    TensorView slog,
    int64_t num_buckets,
    int64_t topk,
    int64_t cand_cap,
    int64_t emit_limit,
    double headroom,
    int64_t probe_stride_tok,
    int64_t hist_stride,
    TensorView origin,
    TensorView inv_delta,
    TensorView th_bucket,
    TensorView bcount,
    TensorView cand_val,
    TensorView cand_idx,
    TensorView cand_cnt) {
  using namespace host;
  auto Q = SymbolicSize{"num_q"};
  auto HEAD = SymbolicSize{"sample_len"};
  auto NB = SymbolicSize{"num_buckets"};
  auto CAP = SymbolicSize{"cand_cap"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({Q, HEAD}).with_dtype<fp32_t>().with_device(device_).verify(slog);
  TensorMatcher({Q}).with_dtype<fp32_t>().with_device(device_).verify(origin).verify(inv_delta);
  TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(th_bucket).verify(cand_cnt);
  TensorMatcher({Q, NB}).with_dtype<int32_t>().with_device(device_).verify(bcount);
  TensorMatcher({Q, CAP}).with_dtype<fp32_t>().with_device(device_).verify(cand_val);
  TensorMatcher({Q, CAP}).with_dtype<int32_t>().with_device(device_).verify(cand_idx);

  const int q_rows = static_cast<int>(Q.unwrap());
  const int head = static_cast<int>(HEAD.unwrap());
  // The kernel reads each row with 16B float4 loads whenever head >= 4, so
  // with more than one row the row stride (== head for contiguous slog) must
  // keep every row base 16B-aligned; a misaligned base is a device-side
  // fault, so fail loudly here instead (callers pad the width with -inf).
  CHECK_HOST(q_rows <= 1 || head < 4 || head % 4 == 0)
      << "sample width " << head << " misaligns float4 row loads; pad sample logits to a multiple of 4";
  const int nb = static_cast<int>(num_buckets);
  const int cap = static_cast<int>(cand_cap);
  CHECK_HOST(nb >= 2 && nb <= 4096) << "num_buckets out of range: " << nb;
  CHECK_HOST(static_cast<int64_t>(NB.unwrap()) == nb) << "bcount width != num_buckets";
  CHECK_HOST(topk >= 1 && cap >= topk) << "need cand_cap >= topk >= 1";
  CHECK_HOST(static_cast<int64_t>(CAP.unwrap()) == cap) << "cand width != cand_cap";
  const DLDevice device = device_.unwrap();

  const int seed_smem = 4 * nb * static_cast<int>(sizeof(int));
  if (seed_smem > 48 * 1024) {
    static bool attr_set = false;
    if (!attr_set) {
      CHECK_CUDA(cudaFuncSetAttribute(
          (const void*)seed_prep_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          4 * 4096 * static_cast<int>(sizeof(int))));
      attr_set = true;
    }
  }
  const int emit_lim = emit_limit == 0 ? 0 : (emit_limit > 0 ? static_cast<int>(emit_limit) : head);
  const int pst = static_cast<int>(probe_stride_tok);
  const int hst = hist_stride > 1 ? static_cast<int>(hist_stride) : 1;

  LaunchKernel(q_rows, 1024, device, seed_smem)(
      seed_prep_kernel,
      static_cast<const float*>(slog.data_ptr()),
      static_cast<int64_t>(slog.stride(0)),
      head,
      nb,
      static_cast<int>(topk),
      cap,
      emit_lim,
      pst,
      hst,
      static_cast<float>(headroom),
      static_cast<float*>(origin.data_ptr()),
      static_cast<float*>(inv_delta.data_ptr()),
      static_cast<int32_t*>(th_bucket.data_ptr()),
      static_cast<int32_t*>(bcount.data_ptr()),
      static_cast<float*>(cand_val.data_ptr()),
      static_cast<int32_t*>(cand_idx.data_ptr()),
      static_cast<int32_t*>(cand_cnt.data_ptr()));
}

// ---------------------------------------------------------------------------
// scan: the fused UMMA scoring + gate + candidate-emit pass over all KV.
//   q          [num_q, 32, 128]  fp8_e4m3 (contiguous)
//   kv         [seq_len_kv, 128] fp8_e4m3 (gathered, contiguous)
//   kv_scales  [seq_len_kv]      fp32 (allocation padded to a multiple of 4:
//              the TMA descriptor rounds the global dim up to 16B)
//   weights    [num_q, 32]       fp32 (q-scale and softmax scale folded in)
//   cu_start / cu_end [num_q]    int32 per-row causal KV range into `kv`
//   refresh_every < 0 selects the external-refresh mode (daemon off, one
//   refresh kernel after the scan); >= 0 runs the in-scan daemon.
// ---------------------------------------------------------------------------
void dsa_litetopk_scan(
    TensorView q,
    TensorView kv,
    TensorView kv_scales,
    TensorView weights,
    TensorView cu_start,
    TensorView cu_end,
    TensorView origin,
    TensorView inv_delta,
    TensorView th_bucket,
    TensorView cand_val,
    TensorView cand_idx,
    TensorView cand_cnt,
    TensorView bcount,
    int64_t num_buckets,
    int64_t topk,
    int64_t refresh_every,
    int64_t num_kv_splits_override,
    int64_t probe_group,
    int64_t probe_add_max) {
  using namespace host;
  auto Q = SymbolicSize{"num_q"};
  auto SKV = SymbolicSize{"seq_len_kv"};
  auto NB = SymbolicSize{"num_buckets"};
  auto CAP = SymbolicSize{"cand_cap"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  auto SPAD = SymbolicSize{"kv_scales_padded_len"};
  TensorMatcher({Q, NUM_HEADS, HEAD_DIM}).with_dtype<fp8_e4m3_t>().with_device(device_).verify(q);
  TensorMatcher({SKV, HEAD_DIM}).with_dtype<fp8_e4m3_t>().with_device(device_).verify(kv);
  TensorMatcher({SPAD}).with_dtype<fp32_t>().with_device(device_).verify(kv_scales);
  TensorMatcher({Q, NUM_HEADS}).with_dtype<fp32_t>().with_device(device_).verify(weights);
  TensorMatcher({Q})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(cu_start)
      .verify(cu_end)
      .verify(th_bucket)
      .verify(cand_cnt);
  TensorMatcher({Q}).with_dtype<fp32_t>().with_device(device_).verify(origin).verify(inv_delta);
  TensorMatcher({Q, CAP}).with_dtype<fp32_t>().with_device(device_).verify(cand_val);
  TensorMatcher({Q, CAP}).with_dtype<int32_t>().with_device(device_).verify(cand_idx);
  TensorMatcher({Q, NB}).with_dtype<int32_t>().with_device(device_).verify(bcount);

  const int seq_len = static_cast<int>(Q.unwrap());
  const int seq_len_kv = static_cast<int>(SKV.unwrap());
  const int cand_cap = static_cast<int>(CAP.unwrap());
  const int nb = static_cast<int>(num_buckets);
  const DLDevice device = device_.unwrap();
  CHECK_HOST(static_cast<int64_t>(NB.unwrap()) == nb) << "bcount width != num_buckets";
  CHECK_HOST(runtime::get_cc_major(device.device_id) == 10) << "dsa_litetopk_scan requires SM100 (Blackwell)";
  // The TMA descriptor for kv_scales rounds the global inner dim up to 16B;
  // the tail elements must be readable, so require a padded allocation.
  const int64_t scales_len = static_cast<int64_t>(SPAD.unwrap());
  CHECK_HOST(scales_len % 4 == 0) << "kv_scales must be padded to a multiple of 4 floats, got " << scales_len;
  CHECK_HOST(scales_len >= seq_len_kv) << "kv_scales shorter than kv";

  const bool external_refresh = (refresh_every < 0);
  const int refresh_every_i = external_refresh ? 0x7fffffff : static_cast<int>(refresh_every);

  const int esz_f32 = 4;
  const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
  auto tm_q = make_2d(
      const_cast<void*>(q.data_ptr()),
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      1,
      HEAD_DIM,
      seq_len * NUM_HEADS,
      HEAD_DIM,
      BLOCK_Q * NUM_HEADS,
      HEAD_DIM,
      HEAD_DIM);
  auto tm_kv = make_2d(
      const_cast<void*>(kv.data_ptr()),
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      1,
      HEAD_DIM,
      seq_len_kv,
      HEAD_DIM,
      BLOCK_KV,
      HEAD_DIM,
      HEAD_DIM);
  auto tm_ks = make_2d(
      const_cast<void*>(kv_scales.data_ptr()),
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      esz_f32,
      ks_aligned,
      1,
      BLOCK_KV,
      1,
      0,
      0);
  auto tm_w = make_2d(
      const_cast<void*>(weights.data_ptr()),
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      esz_f32,
      NUM_HEADS,
      seq_len,
      NUM_HEADS,
      BLOCK_Q,
      NUM_HEADS,
      0);

  const int smem = compute_smem_bytes();
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
      NUM_HEADS,
      HEAD_DIM,
      BLOCK_Q,
      BLOCK_KV,
      NUM_Q_STAGES,
      NUM_KV_STAGES,
      NUM_SMS,
      SPEC_THREADS,
      MATH_THREADS>;
  static bool scan_attr_set = false;
  if (!scan_attr_set) {
    CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    scan_attr_set = true;
  }

  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
  int num_kv_splits;
  if (num_kv_splits_override > 0) {
    num_kv_splits = static_cast<int>(num_kv_splits_override);
  } else {
    // SGLANG DEVIATION: use the real SM count for the split heuristic instead
    // of the hardcoded 148 (kernel template keeps NUM_SMS for 1:1 vendoring).
    static const int num_sms = runtime::get_sm_count(device.device_id);
    constexpr int kWaves = 4;
    const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
    num_kv_splits = (kWaves * num_sms + qb - 1) / qb;
    const int max_useful_splits = total_kv_blocks > 0 ? (total_kv_blocks + 1) / 2 : 1;
    if (num_kv_splits > max_useful_splits) num_kv_splits = max_useful_splits;
  }
  if (num_kv_splits < 1) num_kv_splits = 1;
  if (num_kv_splits > total_kv_blocks) num_kv_splits = total_kv_blocks > 0 ? total_kv_blocks : 1;

  const dim3 grid(static_cast<unsigned>(num_q_blocks), static_cast<unsigned>(num_kv_splits), 1);
  const uint64_t probe_magic =
      probe_group > 0 ? (((1ULL << 42) + static_cast<uint64_t>(probe_group) - 1) / static_cast<uint64_t>(probe_group))
                      : 0ULL;
  LaunchKernel(grid, SPEC_THREADS + MATH_THREADS, device, smem)(
      kernel,
      static_cast<uint32_t>(seq_len),
      static_cast<uint32_t>(seq_len_kv),
      static_cast<uint32_t*>(cu_start.data_ptr()),
      static_cast<uint32_t*>(cu_end.data_ptr()),
      static_cast<const float*>(origin.data_ptr()),
      static_cast<const float*>(inv_delta.data_ptr()),
      static_cast<int32_t*>(th_bucket.data_ptr()),
      static_cast<int32_t*>(bcount.data_ptr()),
      static_cast<uint32_t>(nb),
      static_cast<uint32_t>(topk),
      static_cast<uint32_t>(refresh_every_i),
      static_cast<uint32_t>(num_kv_splits),
      static_cast<uint32_t>(probe_group),
      probe_magic,
      static_cast<uint32_t>(probe_add_max),
      static_cast<float*>(cand_val.data_ptr()),
      static_cast<int32_t*>(cand_idx.data_ptr()),
      static_cast<int32_t*>(cand_cnt.data_ptr()),
      static_cast<uint32_t>(cand_cap),
      tm_q,
      tm_kv,
      tm_ks,
      tm_w);

  if (external_refresh) {
    const int block = 128;
    const int grid_r = (seq_len + block - 1) / block;
    LaunchKernel(grid_r, block, device)(
        refresh_threshold_from_bcount_kernel,
        static_cast<int32_t*>(th_bucket.data_ptr()),
        static_cast<const int32_t*>(bcount.data_ptr()),
        seq_len,
        nb,
        static_cast<int>(topk));
  }
}

// ---------------------------------------------------------------------------
// select: exact top-k over the candidate buffer (radix on float bits).
//   Values are in bucket space build-wide, so callers pass the identity
//   affine (origin=0, inv_delta=1). Output indices are -1 padded (SGLANG
//   DEVIATION vs upstream, which pads with 0).
// ---------------------------------------------------------------------------
void dsa_litetopk_select(
    TensorView cand_val,
    TensorView cand_idx,
    TensorView cand_cnt,
    TensorView origin,
    TensorView inv_delta,
    TensorView th_bucket,
    int64_t num_buckets,
    int64_t topk,
    TensorView out_val,
    TensorView out_idx) {
  using namespace host;
  auto R = SymbolicSize{"num_rows"};
  auto CAP = SymbolicSize{"cand_cap"};
  auto K = SymbolicSize{"topk"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({R, CAP}).with_dtype<fp32_t>().with_device(device_).verify(cand_val);
  TensorMatcher({R, CAP}).with_dtype<int32_t>().with_device(device_).verify(cand_idx);
  TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(cand_cnt).verify(th_bucket);
  TensorMatcher({R}).with_dtype<fp32_t>().with_device(device_).verify(origin).verify(inv_delta);
  TensorMatcher({R, K}).with_dtype<fp32_t>().with_device(device_).verify(out_val);
  TensorMatcher({R, K}).with_dtype<int32_t>().with_device(device_).verify(out_idx);

  const int rows = static_cast<int>(R.unwrap());
  const int cap = static_cast<int>(CAP.unwrap());
  const int k = static_cast<int>(topk);
  const int nb = static_cast<int>(num_buckets);
  CHECK_HOST(k >= 1 && k <= cap) << "topk must be in [1, cand_cap], got " << k;
  CHECK_HOST(static_cast<int64_t>(K.unwrap()) == k) << "out width != topk";
  CHECK_HOST(nb >= 2 && nb <= 4096) << "num_buckets out of range: " << nb;
  const DLDevice device = device_.unwrap();

  LaunchKernel(rows, 256, device)(
      compact_topk_min_thr_litetopk_kernel,
      static_cast<const float*>(cand_val.data_ptr()),
      static_cast<const int32_t*>(cand_idx.data_ptr()),
      static_cast<const int32_t*>(cand_cnt.data_ptr()),
      static_cast<const float*>(origin.data_ptr()),
      static_cast<const float*>(inv_delta.data_ptr()),
      static_cast<const int32_t*>(th_bucket.data_ptr()),
      rows,
      cap,
      k,
      nb,
      static_cast<float*>(out_val.data_ptr()),
      static_cast<int32_t*>(out_idx.data_ptr()),
      /*probe_group=*/0u,
      /*probe_magic=*/0ULL,
      /*probe_add_max=*/0u,
      /*seed_base=*/static_cast<const int32_t*>(nullptr));
}

}  // namespace
