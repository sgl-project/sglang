#pragma once

#include "cutlass/bfloat16.h"

enum class ModelType { V32, MODEL1 };

struct __align__(4 * 8) DecodingSchedMeta {
  int begin_req_idx, end_req_idx;      // Both inclusive
  int begin_block_idx, end_block_idx;  // Inclusive, exclusive
  int begin_split_idx;
  int is_first_req_splitted, is_last_req_splitted;
  int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

struct DenseAttnDecodeParams {  // TODO Change name to DenseAttnDecodeParams
  using index_t = int64_t;

  int b;  // batch size
  int s_q;
  int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
  int d, d_v;         // K/V dimension
  int h_q, h_k;       // The number of Q/K heads
  int num_blocks;     // Number of blocks in total
  int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
  bool is_causal;
  float scale_softmax, scale_softmax_log2;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ o_ptr;
  float* __restrict__ softmax_lse_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t o_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t o_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t o_head_stride;

  int* __restrict__ block_table;
  index_t block_table_batch_stride;
  int page_block_size;
  int* __restrict__ seqlens_k_ptr;

  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
  int num_sm_parts;
  int* __restrict__ num_splits_ptr;

  int total_num_splits;
  float* __restrict__ softmax_lseaccum_ptr;
  float* __restrict__ oaccum_ptr;

  cudaStream_t stream;
};

struct SparseAttnDecodeParams {
  int b, s_q;
  int h_q, h_kv;
  int d_qk, d_v;
  float sm_scale, sm_scale_div_log2;
  int num_blocks, page_block_size, topk;
  ModelType model_type;

  cutlass::bfloat16_t* __restrict__ q;  // [b, s_q, h_q, d_qk]
  // Optional original-Q pointer used when q carries folded NoPE for packed
  // orig blocks but extra native blocks still need unfurled/original Q.
  cutlass::bfloat16_t* __restrict__ extra_q = nullptr;  // [b, s_q, h_q, d_qk]
  cutlass::bfloat16_t* __restrict__ kv;                 // [num_blocks, page_block_size, d_qk]
  int* __restrict__ indices;                            // [b, s_q, topk]
  int* __restrict__ topk_length;                        // [b], may be nullptr
  float* __restrict__ attn_sink;                        // [h_q], may be nullptr

  float* __restrict__ lse;                // [b, s_q, h_q]
  cutlass::bfloat16_t* __restrict__ out;  // [b, s_q, h_q, d_v]

  int extra_num_blocks, extra_page_block_size, extra_topk;
  cutlass::bfloat16_t* __restrict__ extra_kv;  // [extra_num_blocks, extra_page_block_size, d_qk]
  int* __restrict__ extra_indices;             // [b, s_q, extra_topk]
  int* __restrict__ extra_topk_length;         // [b], may be nullptr

  int stride_q_b, stride_q_s_q, stride_q_h_q;
  int stride_extra_q_b = 0, stride_extra_q_s_q = 0, stride_extra_q_h_q = 0;
  int stride_kv_block, stride_kv_row;
  int stride_indices_b, stride_indices_s_q;
  int stride_lse_b, stride_lse_s_q;
  int stride_o_b, stride_o_s_q, stride_o_h_q;
  int stride_extra_kv_block, stride_extra_kv_row;
  int stride_extra_indices_b, stride_extra_indices_s_q;

  cudaStream_t stream;

  // SplitKV-related parameters
  float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
  float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
  int stride_lse_accum_split, stride_lse_accum_s_q;
  int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;  // [num_sm_parts, ], contiguous
  int* __restrict__ num_splits_ptr;                             // [batch_size+1, ], contiguous
  int num_sm_parts;

  // ------------------------------------------------------------------
  // [M3.c.4 Stage-1a wiring] packed-FP8 rotated-quant KV cache pointers
  // for the sparse decode path (mirrors DecodingParams_fp8 in
  // csrc/extension/sm90/dense_fp8/flash_mla.h).
  //
  // These six pointers + meta are the **device-side handles** to the
  // rotated low-precision KV cache (INT2/3/4 affine quant after a
  // dense orthogonal rotation) for the sparse path. They are written
  // by the host entry `sparse_attn_decode_interface` when all six
  // caller tensors are non-None. The current sparse kernels do
  // **not yet read these fields**; the next fork commit will fuse
  // INT-N unpack + R @ x + ×scale + zero -> FP8 inside the K-tile
  // load, removing the host-side shadow buffer entirely. Default
  // values are nullptr / 0 so the pre-existing sparse path is
  // byte-identical to before.
  //
  // Layout convention (matches python/sglang/srt/mem_cache/
  // rotated_quant_dsv4_memory_pool.py wall-storage layout):
  //   * packed_kcache : uint8 [num_pages * page_size, row_bytes_nope]
  //                     (only the nope half; rope half kept BF16
  //                      contiguous after nope bytes)
  //   * scale_kcache  : float32 [num_pages * page_size, qk_nope_head_dim]
  //                     per-element dequant scale
  //   * R_matrix      : float32 [qk_nope_head_dim, qk_nope_head_dim]
  //                     dense orthogonal rotation
  //   * zero_point    : float32 [qk_nope_head_dim] per-element zero
  // dim_of_bit / bitpos_in_dim: per-config bit-packing metadata
  // (length = row_bits = sum(bits[d] for d in 0..qk_nope-1)),
  // per-layer constants.
  // ------------------------------------------------------------------
  void* __restrict__ packed_kcache_ptr = nullptr;
  float* __restrict__ scale_kcache_ptr = nullptr;
  float* __restrict__ R_matrix_ptr = nullptr;
  // [step3r] BF16-prestored R for the uniform-bit (bit_uniform>0) wgmma
  //   fill_sR path. The kernel already truncates R to bf16 before the
  //   gemm, so prestoring bf16 is value-identical (RNE) while halving the
  //   L2 load width and removing the per-element fp32->bf16 conversion.
  //   Set only when R_matrix is passed as bf16 (bit_uniform>0); the legacy
  //   variable-bit float4 R@x path (bit_uniform==0) keeps R_matrix_ptr.
  void* __restrict__ R_matrix_bf16_ptr = nullptr;
  float* __restrict__ zero_point_ptr = nullptr;
  int* __restrict__ dim_of_bit_ptr = nullptr;
  int* __restrict__ bitpos_in_dim_ptr = nullptr;
  int64_t packed_kv_block_stride = 0;
  int packed_row_bytes = 0;
  int qk_nope_head_dim = 0;
  int row_bits = 0;

  // [c4c128-packed] Extra (c4/c128 sink) packed KV cache pointer + block
  // stride. c4/c128 share the SAME calib cfg as SWA (build_synthetic_
  // dsv4_calibration builds one R/scale/zero/bit_uniform for all layers),
  // so scale_kcache_ptr / R_matrix(_bf16)_ptr / zero_point_ptr /
  // dim_of_bit_ptr / bitpos_in_dim_ptr / packed_row_bytes / row_bits /
  // qk_nope_head_dim / bit_uniform are REUSED verbatim. Only the packed
  // byte buffer and its per-page stride differ per pool. When
  // extra_packed_kcache_ptr is non-null the IS_EXTRA_BLOCK branch reads
  // packed rows (bit-unpack + R@x fused dequant) instead of the dense
  // FP8 shadow path. Default nullptr keeps the pre-existing dense extra
  // path byte-identical.
  void* __restrict__ extra_packed_kcache_ptr = nullptr;
  int64_t extra_packed_kv_block_stride = 0;

  // ------------------------------------------------------------------
  // Uniform-bit packed-row layout.
  //
  // When bit_uniform == 0 (default), the kernel reads the variable-bit
  // layout described above (dim_of_bit/bitpos_in_dim + global scale/zp
  // arrays). When bit_uniform > 0 (e.g. 3 or 4), every nope dim uses
  // bit_uniform contiguous bits. The Direct INT4 specialization stores
  // one E4M3 absmax/7 step per group32 in the packed-row header. This
  // lets the kernel inner loop decode each signed nibble directly and
  // avoid global scale/zero-point loads.
  //
  // bit_uniform == 0 -> legacy path (byte-identical to pre-step-5).
  // ------------------------------------------------------------------
  int bit_uniform = 0;
  int uniform_header_bytes = 0;
  int uniform_group_size = 32;
  int uniform_num_groups = 0;
  int q_nope_is_folded = 0;
  int identity_tail_bypass = 0;
  int debug_u32_packed_load = 0;
};

struct CombineParams {
  int b, s_q, h_q, d_v;

  float* __restrict__ lse;  // [b, s_q, h_q]
  void* __restrict__ out;   // [b, s_q, h_q, d_v]
  int stride_lse_b, stride_lse_s_q;
  int stride_o_b, stride_o_s_q, stride_o_h_q;

  float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
  float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
  int stride_lse_accum_split, stride_lse_accum_s_q;
  int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;  // [num_sm_parts, ], contiguous
  int* __restrict__ num_splits_ptr;                             // [batch_size+1, ], contiguous
  int num_sm_parts;

  float* attn_sink;  // [h_q], may be nullptr

  cudaStream_t stream;
};

struct GetDecodeSchedMetaParams {
  int b;  // batch size
  int s_q;
  int block_size_n;
  int fixed_overhead_num_blocks;

  int topk, extra_topk;  // -1 if sparse attention (or extra topk) is disabled
  int* __restrict__ topk_length, * __restrict__ extra_topk_length;

  int* __restrict__ seqlens_k_ptr;  // Only necessary for dense attention

  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
  int* __restrict__ num_splits_ptr;
  int num_sm_parts;

  cudaStream_t stream;
};

struct SparseAttnFwdParams {
  int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
  float sm_scale, sm_scale_div_log2;

  // Input tensors
  cutlass::bfloat16_t* __restrict__ q;   // [s_q, h_q, d_qk]
  cutlass::bfloat16_t* __restrict__ kv;  // [s_kv, h_kv, d_qk]
  int* __restrict__ indices;             // [s_q, h_kv, topk]
  float* __restrict__ attn_sink;         // [h_q], may be nullptr
  int* __restrict__ topk_length;         // [s_q], may be nullptr

  // Strides
  int stride_q_s_q;
  int stride_q_h_q;
  int stride_kv_s_kv;
  int stride_kv_h_kv;
  int stride_indices_s_q;
  int stride_indices_h_kv;

  // Output tensors
  cutlass::bfloat16_t* __restrict__ out;  // [s_q, h_q, d_v]
  float* __restrict__ max_logits;         // [s_q, h_q]
  float* __restrict__ lse;                // [s_q, h_q]

  int num_sm;
  cudaStream_t stream;
};

// We have some kernels that implement both prefill and decode modes in a single kernel (with different template
// instantiations). The following enum helps to distinguish the modes.
enum class SparseAttnFwdMode {
  Prefill,            // Normal prefill mode
  DecodeWithSplitKV,  // To trigger decoding mode for kernels that support both prefill and decode
};

template <SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_decode_v = std::bool_constant<FWD_MODE == SparseAttnFwdMode::DecodeWithSplitKV>::value;

template <SparseAttnFwdMode FWD_MODE>
using SparseFwdArgT = std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams, SparseAttnFwdParams>;
