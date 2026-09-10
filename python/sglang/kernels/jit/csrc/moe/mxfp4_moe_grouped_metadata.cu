/* Copyright 2026 SGLang Team. All Rights Reserved.

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

/**
 * \brief Grouped-GEMM metadata and the row permutation for the MXFP4 MoE layer, in one CTA.
 *
 * Replaces torch.sort + compute_src2dst + compute_problem_sizes + compute_expert_offsets: four
 * launches plus a CUB radix-sort temp allocation, ~56 us per layer at decode shapes where the
 * whole layer costs 700 us. A single CTA histograms the expert ids in shared memory, scans the
 * counts, and reuses the histogram as per-expert write cursors, so the row permutation costs one
 * extra pass over topk_ids and no global memory traffic beyond its output.
 *
 * The permutation is not the stable sort's: rows of one expert land in an order fixed by atomic
 * arrival. Only the pre/post reorder kernels read src2dst, and both treat it as an opaque
 * bijection, so grouping by expert is the whole contract.
 */

#include <cub/cub.cuh>

#include "tvm_ffi_utils.h"
#include <cuda_runtime.h>
#include <limits>

namespace sglang {

// Every expert gets one thread in the scan, so the CTA also caps the expert count.
static constexpr int kMetadataBlock = 1024;

__global__ void mxfp4_moe_grouped_metadata_kernel(
    int32_t const* __restrict__ topk_ids,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ src2dst,
    int32_t num_experts,
    int32_t numel,
    int32_t n,
    int32_t k) {
  // [num_experts]: the histogram, then the write cursors.
  extern __shared__ int32_t bucket[];

  for (int e = threadIdx.x; e < num_experts; e += kMetadataBlock) {
    bucket[e] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < numel; i += kMetadataBlock) {
    uint32_t const e = static_cast<uint32_t>(topk_ids[i]);
    if (e < static_cast<uint32_t>(num_experts)) {
      atomicAdd(&bucket[e], 1);
    }
  }
  __syncthreads();

  using Scan = cub::BlockScan<int32_t, kMetadataBlock>;
  __shared__ typename Scan::TempStorage scan_storage;
  int32_t const count = threadIdx.x < num_experts ? bucket[threadIdx.x] : 0;
  int32_t offset = 0;
  int32_t total = 0;
  Scan(scan_storage).ExclusiveSum(count, offset, total);

  if (threadIdx.x < num_experts) {
    int const e = threadIdx.x;
    expert_offsets[e] = offset;
    problem_sizes1[3 * e + 0] = 2 * n;
    problem_sizes1[3 * e + 1] = count;
    problem_sizes1[3 * e + 2] = k;
    problem_sizes2[3 * e + 0] = k;
    problem_sizes2[3 * e + 1] = count;
    problem_sizes2[3 * e + 2] = n;
    bucket[e] = offset;
  }
  if (threadIdx.x == 0) {
    expert_offsets[num_experts] = total;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < numel; i += kMetadataBlock) {
    uint32_t const e = static_cast<uint32_t>(topk_ids[i]);
    src2dst[i] = e < static_cast<uint32_t>(num_experts) ? atomicAdd(&bucket[e], 1) : -1;
  }
}

/**
 * \brief Fills the grouped-GEMM metadata and the expert-grouped row permutation.
 *
 * \param topk_ids [num_tokens, topk] int32 expert ids; ids outside [0, num_experts) map to -1.
 * \param expert_offsets [num_experts + 1] int32, exclusive prefix sum of the per-expert row counts.
 * \param problem_sizes1 [num_experts, 3] int32, per-expert (2 * n, rows, k) for the gate/up GEMM.
 * \param problem_sizes2 [num_experts, 3] int32, per-expert (k, rows, n) for the down GEMM.
 * \param src2dst [num_tokens * topk] int32, flat topk index -> expert-sorted row.
 * \param n Per-rank intermediate size.
 * \param k Hidden size.
 */
void mxfp4_moe_grouped_metadata(
    TensorView topk_ids,
    TensorView expert_offsets,
    TensorView problem_sizes1,
    TensorView problem_sizes2,
    TensorView src2dst,
    int64_t n,
    int64_t k) {
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(expert_offsets, dl_int32);
  CHECK_INPUT_AND_TYPE(problem_sizes1, dl_int32);
  CHECK_INPUT_AND_TYPE(problem_sizes2, dl_int32);
  CHECK_INPUT_AND_TYPE(src2dst, dl_int32);
  CHECK_DEVICE(topk_ids, expert_offsets);
  CHECK_DEVICE(topk_ids, problem_sizes1);
  CHECK_DEVICE(topk_ids, problem_sizes2);
  CHECK_DEVICE(topk_ids, src2dst);
  CHECK_DIM(1, expert_offsets);
  CHECK_DIM(2, problem_sizes1);
  CHECK_DIM(2, problem_sizes2);
  CHECK_DIM(1, src2dst);

  int64_t const num_experts = expert_offsets.size(0) - 1;
  TVM_FFI_ICHECK_GT(num_experts, 0);
  TVM_FFI_ICHECK_LE(num_experts, kMetadataBlock)
      << "the fused metadata kernel scans one expert per thread, so it caps at " << kMetadataBlock;
  TVM_FFI_ICHECK_EQ(problem_sizes1.size(0), num_experts);
  TVM_FFI_ICHECK_EQ(problem_sizes2.size(0), num_experts);
  TVM_FFI_ICHECK_EQ(problem_sizes1.size(1), 3);
  TVM_FFI_ICHECK_EQ(problem_sizes2.size(1), 3);
  int64_t numel = 1;
  for (int d = 0; d < topk_ids.dim(); ++d) {
    numel *= topk_ids.size(d);
  }
  TVM_FFI_ICHECK_EQ(src2dst.size(0), numel);
  TVM_FFI_ICHECK_LE(numel, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_GT(n, 0);
  TVM_FFI_ICHECK_GT(k, 0);

  cudaStream_t stream = get_stream(topk_ids.device());
  size_t const smem_bytes = sizeof(int32_t) * static_cast<size_t>(num_experts);
  mxfp4_moe_grouped_metadata_kernel<<<1, kMetadataBlock, smem_bytes, stream>>>(
      static_cast<int32_t const*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      static_cast<int32_t*>(src2dst.data_ptr()),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(numel),
      static_cast<int32_t>(n),
      static_cast<int32_t>(k));

  cudaError_t const err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "mxfp4_moe_grouped_metadata launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp4_moe_grouped_metadata, mxfp4_moe_grouped_metadata);

}  // namespace sglang
