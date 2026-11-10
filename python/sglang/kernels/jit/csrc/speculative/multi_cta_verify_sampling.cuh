/*
 * A multi-CTA form of the EAGLE verify's target-only tree sampling (SGLANG_OPT_MULTI_CTA_VERIFY_TREE_SAMPLING)
 * that performs every floating-point operation of the stock kernel, with the same operands in the same order,
 * so its outputs are bitwise equal.
 *
 * Stock: sglang::speculative_sampling::TreeSpeculativeSamplingTargetOnly (1024 threads, FlashInfer
 * primitives). The chain walk runs once per request (pv_tree_walk_kernel, the stock code); each 1024 * VEC
 * chunk of the residual relu(q - p) is reduced by its own CTA with the same BlockReduce the stock loops apply
 * to that chunk (pv_tree_chunk_kernel); one thread accumulates the chunk sums in the stock sequential order;
 * and only the chunk where the stock second loop crosses u runs the stock DeviceSamplingFromProb, with the
 * same aggregate (pv_tree_final_kernel).
 *
 * The fallback when no token is selected is FlashInfer 0.7.3's: the largest valid index over the chunks the
 * stock loop scanned. Each thread of the crossing chunk starts its last_valid_index from the earlier chunks'
 * maximum, so FlashInfer's block reduction over the threads gives that same index.
 */
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <flashinfer/sampling.cuh>

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace sglang {
namespace multi_cta_verify_sampling {

using namespace flashinfer;
using namespace flashinfer::sampling;
using namespace cub;

// ------------------------------------------------------------------------------------------- tree sampling
constexpr uint32_t kTreeThreads = 1024;  // the stock launcher's block_threads
using TreeTempStorage = SamplingTempStorage<kTreeThreads, SCAN_ALGO, REDUCE_ALGO>;

// The walk's results per request: [0] cur_prob_offset (the stock uint32 value), [1] num_accepted_tokens,
// [2] last_accepted_retrive_idx.
constexpr int kMetaWords = 3;
// Chunks of 1024 * VEC residual values per request staged in shared memory by the final pass (V <= 512K).
constexpr uint32_t kMaxChunks = 512;

// The stock kernel's chain walk, once per request (every stock thread runs it redundantly with the same
// result and the same writes).
template <typename DType, typename IdType, typename IdType2>
__global__ void pv_tree_walk_kernel(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,
    const IdType2* candidates,
    const IdType2* retrive_index,
    const IdType2* retrive_next_token,
    const IdType2* retrive_next_sibling,
    const DType* uniform_samples,
    const DType* target_probs,
    DType* draft_probs,
    int64_t* meta,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc) {
  const uint32_t bx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bx >= batch_size) return;
  DType prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  DType coin = uniform_samples[bx * num_draft_tokens];
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      DType target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      const bool has_target_mass = target_prob_single > DType(0);
      if (has_target_mass && (coin < prob_acc / threshold_acc || target_prob_single >= threshold_single)) {
        // accept token
        prob_acc = 0.;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        draft_probs[cur_prob_offset + draft_token_id] = target_probs[cur_prob_offset + draft_token_id];
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  meta[bx * kMetaWords + 0] = static_cast<int64_t>(cur_prob_offset);
  meta[bx * kMetaWords + 1] = static_cast<int64_t>(num_accepted_tokens);
  meta[bx * kMetaWords + 2] = static_cast<int64_t>(last_accepted_retrive_idx);
  ::sglang::device::PDLTriggerSecondary<true>();
}

// The residual's values for chunk i of a request, as both stock loops load them.
template <uint32_t VEC_SIZE, typename DType>
__device__ __forceinline__ vec_t<DType, VEC_SIZE> load_relu(
    const DType* target_probs,
    const DType* draft_probs,
    uint32_t cur_prob_offset,
    bool load_p,
    uint32_t i,
    uint32_t d) {
  const uint32_t tx = threadIdx.x;
  vec_t<DType, VEC_SIZE> q_vec, p_vec, relu;
  q_vec.fill(DType(0));
  p_vec.fill(DType(0));
  if ((i * kTreeThreads + tx) * VEC_SIZE < d) {
    q_vec.load(target_probs + cur_prob_offset + i * kTreeThreads * VEC_SIZE + tx * VEC_SIZE);
    if (load_p) {
      // there is no draft_probs for the bonus token
      p_vec.load(draft_probs + cur_prob_offset + i * kTreeThreads * VEC_SIZE + tx * VEC_SIZE);
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    relu[j] = max(q_vec[j] - p_vec[j], DType(0));
  }
  return relu;
}

// Per chunk: the stock first loop's BlockReduce sum of relu(q - p), the second loop's BlockReduce sum of
// (x > 0 ? x : 0) over the same values, and its BlockReduce max of the valid indices
// (DeviceSamplingFromProb's three block reductions, minus the scan).
template <uint32_t VEC_SIZE, typename DType>
__global__ void __launch_bounds__(kTreeThreads) pv_tree_chunk_kernel(
    const DType* target_probs,
    const DType* draft_probs,
    const int64_t* meta,
    float* chunk_sums,     // [2, batch, nchunks]: relu sums, then thresholded sums
    int* chunk_max_valid,  // [batch, nchunks]
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t d) {
  extern __shared__ __align__(alignof(TreeTempStorage)) uint8_t smem_chunk[];
  auto& temp_storage = reinterpret_cast<TreeTempStorage&>(smem_chunk);
  ::sglang::device::PDLWaitPrimary<true>();
  const uint32_t i = blockIdx.x, bx = blockIdx.y, tx = threadIdx.x, nchunks = gridDim.x;
  const uint32_t cur_prob_offset = static_cast<uint32_t>(meta[bx * kMetaWords + 0]);
  const uint32_t num_accepted_tokens = static_cast<uint32_t>(meta[bx * kMetaWords + 1]);
  const vec_t<DType, VEC_SIZE> relu = load_relu<VEC_SIZE, DType>(
      target_probs, draft_probs, cur_prob_offset, num_accepted_tokens != num_speculative_tokens - 1, i, d);
  DType relu_q_minus_p[VEC_SIZE];
  DType prob_greater_than_threshold[VEC_SIZE];
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    relu_q_minus_p[j] = relu[j];
    prob_greater_than_threshold[j] = (relu[j] > 0) ? relu[j] : 0;
    const bool valid = (relu[j] > 0) && (i * kTreeThreads + tx) * VEC_SIZE + j < d;
    valid_index[j] = valid ? static_cast<int>((i * kTreeThreads + tx) * VEC_SIZE + j) : -1;
  }
  const DType relu_sum = BlockReduce<DType, kTreeThreads, REDUCE_ALGO>(temp_storage.block_prim.reduce)
                             .template Sum<VEC_SIZE>(relu_q_minus_p);
  __syncthreads();
  const DType thresholded_sum = BlockReduce<float, kTreeThreads, REDUCE_ALGO>(temp_storage.block_prim.reduce)
                                    .template Sum<VEC_SIZE>(prob_greater_than_threshold);
  __syncthreads();
  const int max_valid_index = BlockReduce<int, kTreeThreads, REDUCE_ALGO>(temp_storage.block_prim.reduce_int)
                                  .Reduce(valid_index, MaxReduceOp{});
  if (tx == 0) {
    chunk_sums[bx * nchunks + i] = relu_sum;
    chunk_sums[(batch_size + bx) * nchunks + i] = thresholded_sum;
    chunk_max_valid[bx * nchunks + i] = max_valid_index;
  }
  ::sglang::device::PDLTriggerSecondary<true>();
}

// Per request: the first loop's sequential sum of the chunk sums and u; the second loop's sequential aggregate
// up to the first chunk where aggregate + chunk sum > u (the stock loop scans that chunk and breaks after it)
// and its last valid index over the chunks before; then that chunk through the stock DeviceSamplingFromProb.
template <uint32_t VEC_SIZE, bool DETERMINISTIC, typename DType, typename IdType>
__global__ void __launch_bounds__(kTreeThreads) pv_tree_final_kernel(
    IdType* predicts,
    const DType* uniform_samples_for_final_sampling,
    const DType* target_probs,
    const DType* draft_probs,
    const int64_t* meta,
    const float* chunk_sums,
    const int* chunk_max_valid,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t d,
    uint32_t nchunks) {
  extern __shared__ __align__(alignof(TreeTempStorage)) uint8_t smem_final[];
  auto& temp_storage = reinterpret_cast<TreeTempStorage&>(smem_final);
  __shared__ DType s_u, s_aggregate;
  __shared__ int s_chunk;
  ::sglang::device::PDLWaitPrimary<true>();
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t cur_prob_offset = static_cast<uint32_t>(meta[bx * kMetaWords + 0]);
  const uint32_t num_accepted_tokens = static_cast<uint32_t>(meta[bx * kMetaWords + 1]);
  const int64_t last_accepted_retrive_idx = meta[bx * kMetaWords + 2];
  __shared__ float relu_sums[kMaxChunks], thresholded_sums[kMaxChunks];
  __shared__ int max_valid[kMaxChunks];
  for (uint32_t i = tx; i < nchunks; i += kTreeThreads) {
    relu_sums[i] = chunk_sums[bx * nchunks + i];
    thresholded_sums[i] = chunk_sums[(batch_size + bx) * nchunks + i];
    max_valid[i] = chunk_max_valid[bx * nchunks + i];
  }
  __syncthreads();
  if (tx == 0) {
    DType sum_relu_q_minus_p(0);
    for (uint32_t i = 0; i < nchunks; ++i) {
      sum_relu_q_minus_p += relu_sums[i];
    }
    const DType coin = uniform_samples_for_final_sampling[bx];
    const DType u = coin * sum_relu_q_minus_p;
    DType aggregate(0);
    int chunk = -1;
    int last_valid_id = -1;
    for (uint32_t i = 0; i < nchunks; ++i) {
      const DType aggregate_local = thresholded_sums[i];
      if (aggregate + aggregate_local > u) {
        chunk = static_cast<int>(i);
        break;
      }
      const int max_valid_index = max_valid[i];
      if (max_valid_index != -1 && max_valid_index < (int)d) {
        last_valid_id = max_valid_index;
      }
      aggregate += aggregate_local;
    }
    temp_storage.sampled_id = d;
    temp_storage.last_valid_id = last_valid_id;
    s_u = u;
    s_aggregate = aggregate;
    s_chunk = chunk;
  }
  __syncthreads();
  const int chunk = s_chunk;
  if (chunk >= 0) {
    const vec_t<DType, VEC_SIZE> relu_q_minus_p_vec = load_relu<VEC_SIZE, DType>(
        target_probs,
        draft_probs,
        cur_prob_offset,
        num_accepted_tokens != num_speculative_tokens - 1,
        static_cast<uint32_t>(chunk),
        d);
    DType aggregate_relu_q_minus_p = s_aggregate;
    int last_valid_index = temp_storage.last_valid_id;  // the earlier chunks' maximum
    DeviceSamplingFromProb<VEC_SIZE, kTreeThreads, SCAN_ALGO, REDUCE_ALGO, DETERMINISTIC>(
        static_cast<uint32_t>(chunk),
        d,
        [&](DType x) { return x > 0; },
        s_u,
        relu_q_minus_p_vec,
        aggregate_relu_q_minus_p,
        last_valid_index,
        &temp_storage);
  }
  __syncthreads();
  int sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    if (temp_storage.last_valid_id == -1) {
      sampled_id = d - 1;
    } else {
      sampled_id = temp_storage.last_valid_id;
    }
  }
  if (tx == 0) {
    predicts[last_accepted_retrive_idx] = sampled_id;
  }
}

}  // namespace multi_cta_verify_sampling

// -------------------------------------------------------------------------------------------------- host
/// \brief tree_speculative_sampling_target_only with the same arguments, outputs and side effects, bitwise,
/// in three multi-CTA passes. meta [batch, 3] int64, chunk_sums [2, batch, nchunks] fp32 and chunk_max_valid
/// [batch, nchunks] int32 are scratch (nchunks = ceil(vocab / (1024 * VecSize))).
template <uint32_t VecSize, bool Deterministic>
struct MultiCtaVerifyTree {
  static void
  run(tvm::ffi::TensorView predicts,
      tvm::ffi::TensorView accept_index,
      tvm::ffi::TensorView accept_token_num,
      tvm::ffi::TensorView candidates,
      tvm::ffi::TensorView retrive_index,
      tvm::ffi::TensorView retrive_next_token,
      tvm::ffi::TensorView retrive_next_sibling,
      tvm::ffi::TensorView uniform_samples,
      tvm::ffi::TensorView uniform_samples_for_final_sampling,
      tvm::ffi::TensorView target_probs,
      tvm::ffi::TensorView draft_probs,
      tvm::ffi::TensorView meta,
      tvm::ffi::TensorView chunk_sums,
      tvm::ffi::TensorView chunk_max_valid,
      double threshold_single,
      double threshold_acc) {
    using namespace host;
    using namespace multi_cta_verify_sampling;
    static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
    SymbolicSize batch_size{"batch_size"}, draft_tokens{"draft_tokens"}, spec_tokens{"spec_tokens"},
        vocab_size{"vocab_size"}, nchunks{"nchunks"};
    SymbolicDevice device;
    TensorMatcher({batch_size, draft_tokens})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(candidates)
        .verify(retrive_index)
        .verify(retrive_next_token)
        .verify(retrive_next_sibling);
    TensorMatcher({batch_size, spec_tokens}).with_dtype<int32_t>().with_device(device).verify(accept_index);
    TensorMatcher({batch_size}).with_dtype<int32_t>().with_device(device).verify(accept_token_num);
    TensorMatcher({batch_size.unwrap() * draft_tokens.unwrap()})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(predicts);
    TensorMatcher({batch_size, draft_tokens}).with_dtype<float>().with_device(device).verify(uniform_samples);
    TensorMatcher({batch_size}).with_dtype<float>().with_device(device).verify(uniform_samples_for_final_sampling);
    TensorMatcher({batch_size, draft_tokens, vocab_size})
        .with_dtype<float>()
        .with_device(device)
        .verify(target_probs)
        .verify(draft_probs);
    TensorMatcher({batch_size, kMetaWords}).with_dtype<int64_t>().with_device(device).verify(meta);
    TensorMatcher({2, batch_size, nchunks}).with_dtype<float>().with_device(device).verify(chunk_sums);
    TensorMatcher({batch_size, nchunks}).with_dtype<int32_t>().with_device(device).verify(chunk_max_valid);
    CHECK_HOST(draft_tokens.unwrap() > 0 && spec_tokens.unwrap() > 0 && vocab_size.unwrap() > 0);
    CHECK_HOST(std::gcd(int64_t{4}, vocab_size.unwrap()) == VecSize);
    CHECK_HOST(nchunks.unwrap() == (vocab_size.unwrap() + kTreeThreads * VecSize - 1) / (kTreeThreads * VecSize));
    CHECK_HOST(nchunks.unwrap() <= kMaxChunks);
    CHECK_HOST(threshold_single >= 0 && threshold_single <= 1);
    CHECK_HOST(threshold_acc >= 0 && threshold_acc <= 1);
    // the stock kernel's uint32 row offsets
    CHECK_HOST(batch_size.unwrap() * draft_tokens.unwrap() * vocab_size.unwrap() < (int64_t{1} << 32));
    if (batch_size.unwrap() == 0) return;

    const auto dev = device.unwrap();
    const auto bs = static_cast<uint32_t>(batch_size.unwrap());
    const auto n_chunks = static_cast<uint32_t>(nchunks.unwrap());
    const auto spec = static_cast<uint32_t>(spec_tokens.unwrap());
    const auto draft = static_cast<uint32_t>(draft_tokens.unwrap());
    const auto d = static_cast<uint32_t>(vocab_size.unwrap());
    constexpr size_t smem_size = sizeof(TreeTempStorage);
    constexpr auto chunk_kernel = pv_tree_chunk_kernel<VecSize, float>;
    constexpr auto final_kernel = pv_tree_final_kernel<VecSize, Deterministic, float, int32_t>;
    runtime::set_max_dynamic_smem_per_device<chunk_kernel>(dev.device_id, smem_size);
    runtime::set_max_dynamic_smem_per_device<final_kernel>(dev.device_id, smem_size);
    auto* meta_ptr = static_cast<int64_t*>(meta.data_ptr());
    auto* sums_ptr = static_cast<float*>(chunk_sums.data_ptr());
    auto* valid_ptr = static_cast<int*>(chunk_max_valid.data_ptr());
    const auto* target = static_cast<const float*>(target_probs.data_ptr());

    constexpr uint32_t walk_threads = 32;
    LaunchKernel((bs + walk_threads - 1) / walk_threads, walk_threads, dev)(
        pv_tree_walk_kernel<float, int32_t, int64_t>,
        static_cast<int32_t*>(predicts.data_ptr()),
        static_cast<int32_t*>(accept_index.data_ptr()),
        static_cast<int32_t*>(accept_token_num.data_ptr()),
        static_cast<const int64_t*>(candidates.data_ptr()),
        static_cast<const int64_t*>(retrive_index.data_ptr()),
        static_cast<const int64_t*>(retrive_next_token.data_ptr()),
        static_cast<const int64_t*>(retrive_next_sibling.data_ptr()),
        static_cast<const float*>(uniform_samples.data_ptr()),
        target,
        static_cast<float*>(draft_probs.data_ptr()),
        meta_ptr,
        bs,
        spec,
        draft,
        d,
        static_cast<float>(threshold_single),
        std::max(static_cast<float>(threshold_acc), 1e-9f));
    LaunchKernel(dim3(n_chunks, bs), kTreeThreads, dev, smem_size)
        .enable_pdl(true)(
            chunk_kernel,
            target,
            static_cast<const float*>(draft_probs.data_ptr()),
            static_cast<const int64_t*>(meta_ptr),
            sums_ptr,
            valid_ptr,
            bs,
            spec,
            d);
    LaunchKernel(bs, kTreeThreads, dev, smem_size)
        .enable_pdl(true)(
            final_kernel,
            static_cast<int32_t*>(predicts.data_ptr()),
            static_cast<const float*>(uniform_samples_for_final_sampling.data_ptr()),
            target,
            static_cast<const float*>(draft_probs.data_ptr()),
            static_cast<const int64_t*>(meta_ptr),
            static_cast<const float*>(sums_ptr),
            static_cast<const int*>(valid_ptr),
            bs,
            spec,
            d,
            n_chunks);
  }
};

}  // namespace sglang
