// SPDX-License-Identifier: Apache-2.0
// C++ nanobind bridge for paged attention Metal kernels.
//
// Dispatches reshape_and_cache and paged_attention_v1 through MLX's own
// Metal command encoder, eliminating the PyTorch MPS bridge.
//
// Uses nb::handle + nb::inst_ptr<array>() to extract the C++ array from
// the Python mlx.core.array object, bypassing nanobind's cross-module
// RTTI matching which fails due to hidden symbol visibility in libmlx.

#include <algorithm>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
using namespace mlx::core;

#ifndef VLLM_METAL_PARTITION_SIZE
#define VLLM_METAL_PARTITION_SIZE 512
#endif

// ---------------------------------------------------------------------------
// Library caching
// ---------------------------------------------------------------------------

static std::string reshape_cache_source_;
static std::string paged_attention_source_;
static std::string v2_paged_attention_source_;
constexpr int kPartitionSize = VLLM_METAL_PARTITION_SIZE;

void init_libraries(
    const std::string& reshape_src,
    const std::string& paged_attn_src) {
  reshape_cache_source_ = reshape_src;
  paged_attention_source_ = paged_attn_src;

  auto& d = metal::device(Device::gpu);
  d.get_library(
      "paged_reshape_cache",
      [&]() { return reshape_cache_source_; });
  d.get_library(
      "paged_attention_kern",
      [&]() { return paged_attention_source_; });
}

void init_v2_library(const std::string& v2_src) {
  v2_paged_attention_source_ = v2_src;
  auto& d = metal::device(Device::gpu);
  d.get_library(
      "paged_attention_v2_kern",
      [&]() { return v2_paged_attention_source_; });
}

// ---------------------------------------------------------------------------
// Helper: dtype → Metal type string
// ---------------------------------------------------------------------------

static std::string dtype_to_metal(Dtype dt) {
  switch (dt) {
    case float16:   return "half";
    case bfloat16:  return "bfloat16_t";
    case float32:   return "float";
    default:
      throw std::runtime_error(
          "Unsupported dtype for paged attention kernel");
  }
}

// ---------------------------------------------------------------------------
// reshape_and_cache — dispatch helper + eager binding
// ---------------------------------------------------------------------------

// When called from a primitive's eval_gpu, from_primitive should be true
// to skip ALL add_temporary calls.  add_temporary removes buffer pointers
// from the encoder's input/output tracking, defeating fence-based
// synchronisation across command buffer boundaries.  Inside a primitive,
// MLX's evaluator already manages array lifetimes via the completion
// handler.  In the eager path, add_temporary is needed to keep
// Python-owned arrays alive until the command buffer completes.
static void dispatch_reshape_and_cache(
    const array& key, const array& value,
    array& key_cache, array& value_cache,
    const array& slot_mapping, Stream s,
    bool from_primitive = false) {
  auto& d = metal::device(s.device);

  int num_tokens  = static_cast<int>(key.shape(0));
  int num_heads   = static_cast<int>(key.shape(1));
  int head_size   = static_cast<int>(key.shape(2));
  int block_size  = static_cast<int>(key_cache.shape(1));

  int32_t key_stride   = static_cast<int32_t>(num_heads * head_size);
  int32_t value_stride = static_cast<int32_t>(num_heads * head_size);
  int32_t num_heads_i  = static_cast<int32_t>(num_heads);
  int32_t head_size_i  = static_cast<int32_t>(head_size);
  int32_t block_size_i = static_cast<int32_t>(block_size);

  auto dt = dtype_to_metal(key.dtype());
  std::string kname = "reshape_and_cache_kv_" + dt + "_cache_" + dt;

  auto* lib = d.get_library("paged_reshape_cache");
  bool use_fp8 = false;
  auto* kernel = d.get_kernel(
      kname, lib, kname,
      {{&use_fp8, MTL::DataType::DataTypeBool, NS::UInteger(10)}});

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(key, 0);
  enc.set_input_array(value, 1);
  enc.set_output_array(key_cache, 2);
  enc.set_output_array(value_cache, 3);
  enc.set_input_array(slot_mapping, 4);
  enc.set_bytes(key_stride,   7);
  enc.set_bytes(value_stride, 8);
  enc.set_bytes(num_heads_i,  9);
  enc.set_bytes(head_size_i,  10);
  enc.set_bytes(block_size_i, 11);

  int tpg = std::min(512, num_heads * head_size);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_tokens, 1, 1),
      MTL::Size::Make(tpg, 1, 1));

  if (!from_primitive) {
    d.add_temporary(key, s.index);
    d.add_temporary(value, s.index);
    d.add_temporary(key_cache, s.index);
    d.add_temporary(value_cache, s.index);
    d.add_temporary(slot_mapping, s.index);
  }
}

void reshape_and_cache_impl(
    nb::handle key_h, nb::handle value_h,
    nb::handle key_cache_h, nb::handle value_cache_h,
    nb::handle slot_mapping_h) {
  dispatch_reshape_and_cache(
      *nb::inst_ptr<array>(key_h), *nb::inst_ptr<array>(value_h),
      *nb::inst_ptr<array>(key_cache_h), *nb::inst_ptr<array>(value_cache_h),
      *nb::inst_ptr<array>(slot_mapping_h), default_stream(Device::gpu));
}

// ---------------------------------------------------------------------------
// paged_attention_v1
// ---------------------------------------------------------------------------

void paged_attention_v1_impl(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    int block_size,
    int max_seq_len
) {
  auto& out          = *nb::inst_ptr<array>(out_h);
  auto& query        = *nb::inst_ptr<array>(query_h);
  auto& key_cache    = *nb::inst_ptr<array>(key_cache_h);
  auto& value_cache  = *nb::inst_ptr<array>(value_cache_h);
  auto& block_tables = *nb::inst_ptr<array>(block_tables_h);
  auto& seq_lens     = *nb::inst_ptr<array>(seq_lens_h);

  auto s = default_stream(Device::gpu);
  auto& d = metal::device(Device::gpu);

  int num_seqs   = static_cast<int>(query.shape(0));
  int num_heads  = static_cast<int>(query.shape(1));
  int head_size  = static_cast<int>(query.shape(2));
  int max_blocks = static_cast<int>(block_tables.shape(1));

  // Kernel name
  auto dt = dtype_to_metal(query.dtype());
  std::string kname =
      "paged_attention_" + dt + "_cache_" + dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps0";

  // Function constants
  bool use_partitioning = false;
  bool use_alibi        = false;
  bool use_fp8          = false;
  bool use_sinks        = false;

  auto* lib = d.get_library("paged_attention_kern");
  auto* kernel = d.get_kernel(
      kname, lib, kname,
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

  // Threadgroup shared memory
  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  int padded_ctx = ((max_seq_len + block_size - 1) / block_size) * block_size;
  int logits_bytes  = padded_ctx * static_cast<int>(sizeof(float));
  int outputs_bytes = (NUM_THREADS / NUM_SIMD_LANES / 2)
                      * head_size * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(std::max(logits_bytes, outputs_bytes));

  // Dispatch
  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  // Buffer bindings (match paged_attention.metal signature)
  // 0: exp_sums   — skipped (v1, no partitioning)
  // 1: max_logits — skipped
  enc.set_output_array(out,         2);
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);
  // 6: k_scale    — skipped (no FP8)
  // 7: v_scale    — skipped

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  enc.set_bytes(scale, 9);
  float softcapping = 1.0f;
  enc.set_bytes(softcapping, 10);

  enc.set_input_array(block_tables, 11);
  enc.set_input_array(seq_lens,     12);

  int32_t max_blocks_i = static_cast<int32_t>(max_blocks);
  enc.set_bytes(max_blocks_i, 13);
  // 14: alibi_slopes — skipped

  // Strides (contiguous row-major)
  // Cache layout: [num_blocks, block_size, num_kv_heads, head_size]
  int32_t q_stride        = static_cast<int32_t>(num_heads * head_size);
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  int32_t kv_head_stride  = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride,        15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride,  17);

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, num_seqs, 1),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  // Keep ALL referenced arrays alive until the command buffer completes
  d.add_temporary(out, s.index);
  d.add_temporary(query, s.index);
  d.add_temporary(key_cache, s.index);
  d.add_temporary(value_cache, s.index);
  d.add_temporary(block_tables, s.index);
  d.add_temporary(seq_lens, s.index);
}

// ---------------------------------------------------------------------------
// paged_attention_v2_online — dispatch helper + eager wrappers
// ---------------------------------------------------------------------------

static void dispatch_paged_attention_v2_online(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window, Stream s,
    bool from_primitive = false) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int num_heads  = static_cast<int>(query.shape(1));
  int head_size  = static_cast<int>(query.shape(2));
  int max_blocks = static_cast<int>(block_tables.shape(1));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;

  auto dt = dtype_to_metal(query.dtype());
  std::string kname =
      "paged_attention_" + dt + "_cache_" + dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps0";

  bool use_partitioning = false;
  bool use_alibi        = false;
  bool use_fp8          = false;
  bool use_sinks        = false;

  auto* lib = d.get_library("paged_attention_v2_kern");
  auto* kernel = d.get_kernel(
      kname, lib, kname + "_v2",
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_WARPS      = NUM_THREADS / NUM_SIMD_LANES;
  int warp_scores_bytes = NUM_WARPS * block_size
                          * static_cast<int>(sizeof(float));
  int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size)
                    * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(std::max(warp_scores_bytes, merge_bytes));

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  enc.set_output_array(out, 2);
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  enc.set_bytes(scale, 9);
  float softcapping = softcap;
  enc.set_bytes(softcapping, 10);

  enc.set_input_array(block_tables, 11);
  enc.set_input_array(seq_lens,     12);

  int32_t max_blocks_i = static_cast<int32_t>(max_blocks);
  enc.set_bytes(max_blocks_i, 13);

  int32_t q_stride        = static_cast<int32_t>(num_heads * head_size);
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  int32_t kv_head_stride  = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride,        15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride,  17);

  enc.set_input_array(cu_seqlens_q, 19);
  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  enc.set_bytes(num_seqs_i, 20);
  int32_t sliding_window_i = static_cast<int32_t>(sliding_window);
  enc.set_bytes(sliding_window_i, 21);

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, 1),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  if (!from_primitive) {
    d.add_temporary(out, s.index);
    d.add_temporary(query, s.index);
    d.add_temporary(key_cache, s.index);
    d.add_temporary(value_cache, s.index);
    d.add_temporary(block_tables, s.index);
    d.add_temporary(seq_lens, s.index);
    d.add_temporary(cu_seqlens_q, s.index);
  }
}

// Eager wrapper — keeps the old handle-based API for metal_unified_attention.
// Non-partitioned case delegates to the dispatch helper above;
// partitioned case is handled inline (same as original code on main).
void paged_attention_v2_online_impl_common(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    float softcap,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    nb::handle cu_seqlens_q_h,
    int block_size,
    int max_seq_len,
    int sliding_window,
    array* exp_sums,
    array* max_logits,
    array* tmp_out,
    array* sinks
) {
  auto& out          = *nb::inst_ptr<array>(out_h);
  auto& query        = *nb::inst_ptr<array>(query_h);
  auto& key_cache    = *nb::inst_ptr<array>(key_cache_h);
  auto& value_cache  = *nb::inst_ptr<array>(value_cache_h);
  auto& block_tables = *nb::inst_ptr<array>(block_tables_h);
  auto& seq_lens     = *nb::inst_ptr<array>(seq_lens_h);
  auto& cu_seqlens_q = *nb::inst_ptr<array>(cu_seqlens_q_h);

  // Non-partitioned case: delegate to the shared dispatch helper
  bool needs_partitioning =
      exp_sums != nullptr && max_logits != nullptr && tmp_out != nullptr;
  if (!needs_partitioning) {
    dispatch_paged_attention_v2_online(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, softcap,
        block_tables, seq_lens, cu_seqlens_q,
        block_size, max_seq_len, sliding_window,
        default_stream(Device::gpu));
    return;
  }

  // Partitioned path (unchanged from main)
  auto s = default_stream(Device::gpu);
  auto& d = metal::device(Device::gpu);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int num_heads  = static_cast<int>(query.shape(1));
  int head_size  = static_cast<int>(query.shape(2));
  int max_blocks = static_cast<int>(block_tables.shape(1));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  int max_num_partitions =
      std::max(1, (max_seq_len + kPartitionSize - 1) / kPartitionSize);
  bool use_partitioning =
      kPartitionSize % block_size == 0 && max_num_partitions > 1;

  auto dt = dtype_to_metal(query.dtype());
  std::string kname =
      "paged_attention_" + dt + "_cache_" + dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps" +
      std::to_string(use_partitioning ? kPartitionSize : 0);

  bool use_alibi        = false;
  bool use_fp8          = false;
  bool use_sinks        = sinks != nullptr;

  auto* lib = d.get_library("paged_attention_v2_kern");
  auto* kernel = d.get_kernel(
      kname, lib, kname + "_v2",
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_WARPS      = NUM_THREADS / NUM_SIMD_LANES;
  int warp_scores_bytes = NUM_WARPS * block_size
                          * static_cast<int>(sizeof(float));
  int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size)
                    * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(std::max(warp_scores_bytes, merge_bytes));

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  if (use_partitioning) {
    enc.set_output_array(*exp_sums, 0);
    enc.set_output_array(*max_logits, 1);
    enc.set_output_array(*tmp_out, 2);
  } else {
    enc.set_output_array(out, 2);
  }
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  enc.set_bytes(scale, 9);
  float softcapping = softcap;
  enc.set_bytes(softcapping, 10);

  enc.set_input_array(block_tables, 11);
  enc.set_input_array(seq_lens,     12);

  int32_t max_blocks_i = static_cast<int32_t>(max_blocks);
  enc.set_bytes(max_blocks_i, 13);

  int32_t q_stride        = static_cast<int32_t>(num_heads * head_size);
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  int32_t kv_head_stride  = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride,        15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride,  17);
  if (use_sinks) {
    enc.set_input_array(*sinks, 18);
  }

  enc.set_input_array(cu_seqlens_q, 19);
  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  enc.set_bytes(num_seqs_i, 20);
  int32_t sliding_window_i = static_cast<int32_t>(sliding_window);
  enc.set_bytes(sliding_window_i, 21);

  const int32_t grid_z =
      static_cast<int32_t>(use_partitioning ? max_num_partitions : 1);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, grid_z),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  if (use_partitioning) {
    std::string reduce_kname =
        "paged_attention_v2_reduce_" + dt +
        "_hs" + std::to_string(head_size) +
        "_nt256_nsl32_ps" + std::to_string(kPartitionSize);
    auto* reduce_kernel = d.get_kernel(
        reduce_kname,
        lib,
        reduce_kname + "_v2_reduce",
        {{&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)}});
    size_t reduce_shmem =
        static_cast<size_t>(2 * max_num_partitions * sizeof(float));
    enc.set_compute_pipeline_state(reduce_kernel);
    enc.set_threadgroup_memory_length(reduce_shmem, 0);

    enc.set_output_array(out,         0);
    enc.set_input_array(*exp_sums,    1);
    enc.set_input_array(*max_logits,  2);
    enc.set_input_array(*tmp_out,     3);
    enc.set_input_array(seq_lens,     4);
    int32_t max_num_partitions_i = static_cast<int32_t>(max_num_partitions);
    enc.set_bytes(max_num_partitions_i, 5);
    if (use_sinks) {
      enc.set_input_array(*sinks, 6);
    }
    enc.set_input_array(cu_seqlens_q, 7);
    enc.set_bytes(num_seqs_i, 8);
    enc.dispatch_threadgroups(
        MTL::Size::Make(num_heads, total_q_tokens, 1),
        MTL::Size::Make(NUM_THREADS, 1, 1));
  }

  d.add_temporary(out, s.index);
  d.add_temporary(query, s.index);
  d.add_temporary(key_cache, s.index);
  d.add_temporary(value_cache, s.index);
  d.add_temporary(block_tables, s.index);
  d.add_temporary(seq_lens, s.index);
  d.add_temporary(cu_seqlens_q, s.index);
  if (use_partitioning) {
    d.add_temporary(*exp_sums, s.index);
    d.add_temporary(*max_logits, s.index);
    d.add_temporary(*tmp_out, s.index);
  }
  if (use_sinks) {
    d.add_temporary(*sinks, s.index);
  }
}

// ---------------------------------------------------------------------------
// Paged attention primitive (read-only): paged_attention_v2_online only.
//
// Single output: attention result.  The KV cache is read-only — cache
// writes are handled upstream by MLX-native scatter (pure functional).
// This is a clean pure function: inputs → output, no side effects.
// ---------------------------------------------------------------------------

class PagedAttentionPrimitive : public UnaryPrimitive {
 public:
  PagedAttentionPrimitive(
      Stream stream, int num_kv_heads, float scale, float softcap,
      int block_size, int max_seq_len, int sliding_window)
      : UnaryPrimitive(stream),
        num_kv_heads_(num_kv_heads), scale_(scale), softcap_(softcap),
        block_size_(block_size), max_seq_len_(max_seq_len),
        sliding_window_(sliding_window) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error(
        "PagedAttentionPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    // inputs: [query, key_cache, value_cache, block_tables, seq_lens,
    //          cu_seqlens_q]
    out.set_data(allocator::malloc(out.nbytes()));
    dispatch_paged_attention_v2_online(
        out,
        inputs[0],               // query
        inputs[1], inputs[2],    // key_cache, value_cache
        num_kv_heads_, scale_, softcap_,
        inputs[3], inputs[4], inputs[5],  // block_tables, seq_lens, cu_seqlens_q
        block_size_, max_seq_len_, sliding_window_,
        stream(),
        /*from_primitive=*/true);
  }

  const char* name() const override { return "PagedAttention"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const PagedAttentionPrimitive*>(&other);
    return rhs && rhs->num_kv_heads_ == num_kv_heads_
        && rhs->scale_ == scale_ && rhs->softcap_ == softcap_
        && rhs->block_size_ == block_size_
        && rhs->max_seq_len_ == max_seq_len_
        && rhs->sliding_window_ == sliding_window_;
  }

 private:
  int num_kv_heads_;
  float scale_;
  float softcap_;
  int block_size_;
  int max_seq_len_;
  int sliding_window_;
};

static array paged_attention_primitive_fn(
    const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window) {
  auto prim = std::make_shared<PagedAttentionPrimitive>(
      default_stream(Device::gpu),
      num_kv_heads, scale, softcap,
      block_size, max_seq_len, sliding_window);
  return array(
      query.shape(), query.dtype(), std::move(prim),
      {query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q});
}

void paged_attention_v2_online_impl(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    float softcap,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    nb::handle cu_seqlens_q_h,
    int block_size,
    int max_seq_len,
    int sliding_window
) {
  paged_attention_v2_online_impl_common(
      out_h,
      query_h,
      key_cache_h,
      value_cache_h,
      num_kv_heads,
      scale,
      softcap,
      block_tables_h,
      seq_lens_h,
      cu_seqlens_q_h,
      block_size,
      max_seq_len,
      sliding_window,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

void paged_attention_v2_online_partitioned_impl(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    float softcap,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    nb::handle cu_seqlens_q_h,
    int block_size,
    int max_seq_len,
    int sliding_window,
    nb::handle exp_sums_h,
    nb::handle max_logits_h,
    nb::handle tmp_out_h
) {
  auto& exp_sums = *nb::inst_ptr<array>(exp_sums_h);
  auto& max_logits = *nb::inst_ptr<array>(max_logits_h);
  auto& tmp_out = *nb::inst_ptr<array>(tmp_out_h);
  paged_attention_v2_online_impl_common(
      out_h,
      query_h,
      key_cache_h,
      value_cache_h,
      num_kv_heads,
      scale,
      softcap,
      block_tables_h,
      seq_lens_h,
      cu_seqlens_q_h,
      block_size,
      max_seq_len,
      sliding_window,
      &exp_sums,
      &max_logits,
      &tmp_out,
      nullptr);
}

// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(_paged_ops, m) {
  m.attr("PARTITION_SIZE") = nb::int_(kPartitionSize);

  m.def("init_libraries", &init_libraries,
        nb::arg("reshape_src"), nb::arg("paged_attn_src"),
        "JIT-compile the vendored Metal shaders.");

  m.def("init_v2_library", &init_v2_library,
        nb::arg("v2_src"),
        "JIT-compile the v2 online-softmax Metal shader.");

  m.def("reshape_and_cache", &reshape_and_cache_impl,
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
        "Write projected K/V into the paged cache.");

  m.def("paged_attention_v1", &paged_attention_v1_impl,
        nb::arg("out"), nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        "Zero-copy paged attention (v1, no partitioning).");

  m.def("paged_attention_v2_online", &paged_attention_v2_online_impl,
        nb::arg("out"), nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("softcap"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("sliding_window"),
        "Online-softmax varlen paged attention (v2, unified prefill+decode).");

  m.def("paged_attention_v2_online_partitioned",
        &paged_attention_v2_online_partitioned_impl,
        nb::arg("out"), nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("softcap"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("sliding_window"),
        nb::arg("exp_sums"), nb::arg("max_logits"), nb::arg("tmp_out"),
        "Online-softmax varlen paged attention (v2) with caller-provided "
        "partition scratch buffers.");

  // Paged attention primitive (read-only): dispatches paged_attention_v2_online.
  // Cache writes are handled by MLX-native scatter upstream.
  // Uses overwrite_descriptor to bypass cross-module nanobind RTTI.
  m.def("paged_attention_primitive",
        [](nb::handle query_h,
           nb::handle key_cache_h, nb::handle value_cache_h,
           int num_kv_heads, float scale, float softcap,
           nb::handle block_tables_h, nb::handle seq_lens_h,
           nb::handle cu_seqlens_q_h,
           int block_size, int max_seq_len, int sliding_window,
           nb::handle out_h) {
          auto result = paged_attention_primitive_fn(
              *nb::inst_ptr<array>(query_h),
              *nb::inst_ptr<array>(key_cache_h),
              *nb::inst_ptr<array>(value_cache_h),
              num_kv_heads, scale, softcap,
              *nb::inst_ptr<array>(block_tables_h),
              *nb::inst_ptr<array>(seq_lens_h),
              *nb::inst_ptr<array>(cu_seqlens_q_h),
              block_size, max_seq_len, sliding_window);
          nb::inst_ptr<array>(out_h)->overwrite_descriptor(result);
        },
        nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"), nb::arg("softcap"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("sliding_window"),
        nb::arg("out"),
        "Paged attention primitive (read-only). Cache writes are handled "
        "by MLX-native scatter upstream.");
}