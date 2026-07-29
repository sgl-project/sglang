/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd("Moore Threads").
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/Utils.h>
#include <ATen/core/Generator.h>
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/UnpackRaw.muh"
#include <flashinfer/sampling.muh>
#include <mutex>

#include "musa.h"
#include "musa/dispatch_utils.h"
#include "pytorch_extension_utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace musa {
namespace sampling {

#define kRmemEles 16        // should not too large, otherwise it will cause register spilling
// reuse code of flashinfer
using namespace flashinfer;
using namespace flashinfer::sampling;

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProbWithOffset(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage,
    int offset = 0) {
  const uint32_t tx = threadIdx.x;
  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
    valid[j] = pred(prob_vec[j]) && (offset + (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d);
  }
  float aggregate_local =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .template Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (tx == 0) {
    temp_storage->block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
      BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .template InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

      __syncthreads();
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
    }

    bool greater_than_u_diff[VEC_SIZE];
#ifdef FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u_diff, BoolDiffOp());
#else
    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .template FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
#endif
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u_diff[j]) {
        atomicMin(&(temp_storage->sampled_id), offset + (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
      }
    }
    __syncthreads();
  }

  // update the last valid index
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    if (valid[j]) {
      valid_index[j] = offset + (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    } else {
      valid_index[j] = -1;
    }
  }
  int max_valid_index =
      BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
          .Reduce(valid_index, MaxReduceOp{});
  if (tx == 0 && max_valid_index != -1) {
    temp_storage->last_valid_id = max_valid_index;
  }
  __syncthreads();
  aggregate += aggregate_local;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKTopPSamplingFromProbKernel(DType* probs, IdType* top_k_arr, float* top_p_arr,
                                               IdType* output, IdType* indices, IdType top_k_val,
                                               float top_p_val, uint32_t d, uint64_t philox_seed,
                                               uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  murandStatePhilox4_32_10_t state;
  murand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
  const float p = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];
  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

  vec_t<float, kRmemEles> rprobs;  // persistent
  vec_t<float, VEC_SIZE> rprobs_d;
  // elements of one row will be split into three stages by the address it will be stored
  // [rprobs, probs_vec]
  // rprobs will be persistent to reuse those elements instead of reload
  int real_r_size = d > (BLOCK_THREADS * kRmemEles) ? BLOCK_THREADS * kRmemEles : d;
  int real_rd_size = d > (BLOCK_THREADS * kRmemEles) ? d - BLOCK_THREADS * kRmemEles : 0;

  probs = probs + row_idx * d;
  DType* global_rprobs = probs;
  DType* global_rprobs_d = probs + real_r_size;

  // fisrtly, we load some elements to register
  rprobs.fill(0);
  if (tx * kRmemEles < real_r_size) {
    rprobs.cast_load(global_rprobs + tx * kRmemEles);
    int valid_size = real_r_size - tx * kRmemEles;
    // mask invalid data
    for (int i = valid_size; i < kRmemEles; ++i) {
      rprobs[i] = 0.0;
    }
  }

  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;

  // sample and check
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = murand_uniform(&state) * q;
    aggregate = 0;

    // fisrtly, sample from rprobs
    DeviceSamplingFromProbWithOffset<kRmemEles, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
        0, d, [&](float x) { return x > low; }, u, rprobs, aggregate, &temp_storage);

    if (aggregate <= u) {
      // secondly, sample from rprobs_d
      for (uint32_t i = 0; i < ceil_div(real_rd_size, BLOCK_THREADS * VEC_SIZE); ++i) {
        rprobs_d.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE + real_r_size < d) {
          rprobs_d.cast_load(global_rprobs_d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }

        DeviceSamplingFromProbWithOffset<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
            i, d, [&](float x) { return x > low; }, u, rprobs_d, aggregate, &temp_storage, real_r_size);
        if (aggregate > u) {
          break;
        }
      }
    }

    // id is sampled
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    // check in rprobs
    ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      probs_gt_pivot_0[i] = {0, 0};
      probs_gt_pivot_1[i] = {0, 0};
    }

    // init to 0
    for (uint32_t j = 0; j < kRmemEles; j += VEC_SIZE) {
#pragma unroll
      for (uint32_t k = 0; k < VEC_SIZE; ++k) {
        probs_gt_pivot_0[k] +=
            {(rprobs[j + k] > pivot_0) ? rprobs[j + k] : 0, (rprobs[j + k] > pivot_0 && (tx)*kRmemEles + j + k < d)};
        probs_gt_pivot_1[k] +=
            {(rprobs[j + k] > pivot_1) ? rprobs[j + k] : 0, (rprobs[j + k] > pivot_1 && (tx)*kRmemEles + j + k < d)};
      }
    }

    // check in rprobs_d
    for (uint32_t i = 0; i < ceil_div(real_rd_size, BLOCK_THREADS * VEC_SIZE); ++i) {
      rprobs_d.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE + real_r_size < d) {
        rprobs_d.cast_load(global_rprobs_d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] +=
            {(rprobs_d[j] > pivot_0) ? rprobs_d[j] : 0,
             (rprobs_d[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j + real_r_size < d)};
        probs_gt_pivot_1[j] +=
            {(rprobs_d[j] > pivot_1) ? rprobs_d[j] : 0,
             (rprobs_d[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j + real_r_size < d)};
      }
    }

    aggregate_gt_pivot_0 = BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
                               .template Sum<VEC_SIZE>(probs_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

    aggregate_gt_pivot_1 = BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
                               .template Sum<VEC_SIZE>(probs_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;

    if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0.value;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}
}  // namespace sampling
}  // namespace musa

template <typename T, typename IdType>
musaError_t MusaTopKTopPSamplingFromProb(T* probs, IdType* top_k_arr, T* top_p_arr, IdType* output,
                                     IdType* indices, uint32_t batch_size, IdType top_k_val,
                                     T top_p_val, uint32_t d, bool deterministic,
                                     uint64_t philox_seed, uint64_t philox_offset,
                                     musaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
  using namespace flashinfer;
  using namespace flashinfer::sampling;
  auto compute_capacity = GetCudaComputeCapability();

  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {
        &probs, &top_k_arr, &top_p_arr, &output, &indices, &top_k_val, &top_p_val, &d, &philox_seed, &philox_offset};
    // fall back to flashinfer implementation
    if (d < BLOCK_THREADS * kRmemEles) {
      DISPATCH_ALIGNED_VEC_SIZE(
          vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
            auto kernel = TopKTopPSamplingFromProbKernel<
                BLOCK_THREADS,
                SCAN_ALGO,
                REDUCE_ALGO,
                VEC_SIZE,
                DETERMINISTIC,
                T,
                IdType>;
            FLASHINFER_CUDA_CALL(musaFuncSetAttribute(kernel, musaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            FLASHINFER_CUDA_CALL(musaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          })});
    } else {
      DISPATCH_ALIGNED_VEC_SIZE(
          vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
            auto kernel = musa::sampling::TopKTopPSamplingFromProbKernel<
                BLOCK_THREADS,
                SCAN_ALGO,
                REDUCE_ALGO,
                VEC_SIZE,
                DETERMINISTIC,
                T,
                IdType>;
            FLASHINFER_CUDA_CALL(musaFuncSetAttribute(kernel, musaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            FLASHINFER_CUDA_CALL(musaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          })});
    }
    return musaSuccess;
  });
}

void musa_top_k_top_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_top_k_arr,
    double top_k_val,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    bool deterministic,
    std::optional<at::Generator> gen_) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  auto device = probs.device();
  CHECK_EQ(output.device(), device);
  CHECK_EQ(probs.dtype(), torch::kFloat);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::MUSAGeneratorImpl>(gen_, at::musa::detail::getDefaultMUSAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxMusaState rng_engine_inputs = gen->philox_musa_state(32 * batch_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::musa::OptionalMUSAGuard device_guard(device);
  auto stream = at::musa::getCurrentMUSAStream();
  musaError_t status = MusaTopKTopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size,
      top_k_val,
      top_p_val,
      vocab_size,
      deterministic,
      philox_seed,
      philox_offset,
      stream);
  TORCH_CHECK(
      status == musaSuccess,
      "MusaTopKTopPSamplingFromProb failed with error code " + std::string(musaGetErrorString(status)));
}
