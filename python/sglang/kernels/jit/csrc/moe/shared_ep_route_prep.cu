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

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

#if defined(USE_ROCM)
constexpr auto kSharedEpDeviceType = kDLROCM;
constexpr int kRouteThreadGroupSize = 64;
#else
constexpr auto kSharedEpDeviceType = kDLCUDA;
constexpr int kRouteThreadGroupSize = 32;
#endif

SGL_DEVICE uint32_t shared_ep_load_acquire_system(const uint32_t* address) {
#if defined(USE_ROCM)
  return __hip_atomic_load(address, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
#endif
}

#if defined(USE_ROCM)
SGL_DEVICE void shared_ep_store_release_system(uint32_t* address, uint32_t value) {
  __hip_atomic_store(address, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void shared_ep_publish_epoch_kernel(
    uint8_t* global_signals, int32_t* epoch_ptr, int64_t rank_stride_words, int32_t rank, int32_t world_size) {
  __shared__ uint32_t next_epoch;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  if (tid == 0) {
    next_epoch = static_cast<uint32_t>(*epoch_ptr) + 1u;
    *epoch_ptr = static_cast<int32_t>(next_epoch);
  }
  __syncthreads();
  if (tid < world_size) {
    auto* signal_words = reinterpret_cast<uint32_t*>(global_signals);
    shared_ep_store_release_system(signal_words + tid * rank_stride_words + rank, next_epoch);
  }
}

__global__ void shared_ep_publish_pointer_table_epoch_kernel(
    const uint64_t* peer_signal_bases, int32_t* epoch_ptr, int32_t rank, int32_t world_size) {
  __shared__ uint32_t next_epoch;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  if (tid == 0) {
    next_epoch = static_cast<uint32_t>(*epoch_ptr) + 1u;
    *epoch_ptr = static_cast<int32_t>(next_epoch);
  }
  __syncthreads();
  if (tid < world_size) {
    auto* peer_signals = reinterpret_cast<uint32_t*>(peer_signal_bases[tid]);
    shared_ep_store_release_system(peer_signals + rank, next_epoch);
  }
}

__global__ void
shared_ep_wait_epoch_kernel(const uint8_t* local_signals, const int32_t* epoch_ptr, int32_t world_size) {
  __shared__ uint32_t expected_epoch;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  if (tid == 0) expected_epoch = static_cast<uint32_t>(*epoch_ptr);
  __syncthreads();
  if (tid < world_size) {
    const auto* signal_words = reinterpret_cast<const uint32_t*>(local_signals);
    while (shared_ep_load_acquire_system(signal_words + tid) != expected_epoch) {
    }
  }
}

struct SharedEpEpochKernel {
  static void publish(
      tvm::ffi::TensorView global_signals,
      tvm::ffi::TensorView epoch,
      int64_t rank_stride_words,
      int64_t rank,
      int64_t world_size) {
    using namespace host;
    RuntimeCheck(global_signals.device().device_type == kDLROCM, "SharedEP epoch signals must be ROCm");
    RuntimeCheck(epoch.device().device_type == kDLROCM, "SharedEP epoch counter must be ROCm");
    RuntimeCheck(
        global_signals.device().device_id == epoch.device().device_id,
        "SharedEP epoch tensors must use the same ROCm device");
    RuntimeCheck(is_type<uint8_t>(global_signals.dtype()), "SharedEP epoch signals must use uint8 storage");
    RuntimeCheck(is_type<int32_t>(epoch.dtype()) && epoch.numel() == 1, "SharedEP epoch must be one int32 value");
    RuntimeCheck(rank_stride_words > 0, "SharedEP epoch rank stride must be positive");
    RuntimeCheck(world_size > 0 && world_size <= 1024, "SharedEP epoch world size must be in [1, 1024]");
    RuntimeCheck(rank >= 0 && rank < world_size, "SharedEP epoch rank is outside the world");
    const int64_t required_words = (world_size - 1) * rank_stride_words + rank + 1;
    RuntimeCheck(
        required_words <= global_signals.numel() / static_cast<int64_t>(sizeof(uint32_t)),
        "SharedEP global epoch storage is too small");

    const auto device = global_signals.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);
    LaunchKernel(dim3(1), dim3(static_cast<uint32_t>(world_size)), stream)(
        shared_ep_publish_epoch_kernel,
        static_cast<uint8_t*>(global_signals.data_ptr()),
        static_cast<int32_t*>(epoch.data_ptr()),
        rank_stride_words,
        static_cast<int32_t>(rank),
        static_cast<int32_t>(world_size));
  }

  static void publish_pointer_table(
      tvm::ffi::TensorView peer_signal_bases, tvm::ffi::TensorView epoch, int64_t rank, int64_t world_size) {
    using namespace host;
    RuntimeCheck(peer_signal_bases.device().device_type == kDLROCM, "SharedEP peer pointer table must be ROCm");
    RuntimeCheck(epoch.device().device_type == kDLROCM, "SharedEP epoch counter must be ROCm");
    RuntimeCheck(
        peer_signal_bases.device().device_id == epoch.device().device_id,
        "SharedEP peer pointer table and epoch must use the same ROCm device");
    RuntimeCheck(is_type<uint64_t>(peer_signal_bases.dtype()), "SharedEP peer pointer table must use uint64");
    RuntimeCheck(is_type<int32_t>(epoch.dtype()) && epoch.numel() == 1, "SharedEP epoch must be one int32 value");
    RuntimeCheck(world_size > 0 && world_size <= 1024, "SharedEP epoch world size must be in [1, 1024]");
    RuntimeCheck(rank >= 0 && rank < world_size, "SharedEP epoch rank is outside the world");
    RuntimeCheck(
        peer_signal_bases.numel() == world_size, "SharedEP peer pointer table must contain one pointer per rank");

    const auto device = peer_signal_bases.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);
    LaunchKernel(dim3(1), dim3(static_cast<uint32_t>(world_size)), stream)(
        shared_ep_publish_pointer_table_epoch_kernel,
        static_cast<const uint64_t*>(peer_signal_bases.data_ptr()),
        static_cast<int32_t*>(epoch.data_ptr()),
        static_cast<int32_t>(rank),
        static_cast<int32_t>(world_size));
  }

  static void wait(tvm::ffi::TensorView local_signals, tvm::ffi::TensorView epoch, int64_t world_size) {
    using namespace host;
    RuntimeCheck(local_signals.device().device_type == kDLROCM, "SharedEP epoch signals must be ROCm");
    RuntimeCheck(epoch.device().device_type == kDLROCM, "SharedEP epoch counter must be ROCm");
    RuntimeCheck(
        local_signals.device().device_id == epoch.device().device_id,
        "SharedEP epoch tensors must use the same ROCm device");
    RuntimeCheck(is_type<uint8_t>(local_signals.dtype()), "SharedEP epoch signals must use uint8 storage");
    RuntimeCheck(is_type<int32_t>(epoch.dtype()) && epoch.numel() == 1, "SharedEP epoch must be one int32 value");
    RuntimeCheck(world_size > 0 && world_size <= 1024, "SharedEP epoch world size must be in [1, 1024]");
    RuntimeCheck(
        world_size <= local_signals.numel() / static_cast<int64_t>(sizeof(uint32_t)),
        "SharedEP local epoch storage is too small");

    const auto device = local_signals.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);
    LaunchKernel(dim3(1), dim3(static_cast<uint32_t>(world_size)), stream)(
        shared_ep_wait_epoch_kernel,
        static_cast<const uint8_t*>(local_signals.data_ptr()),
        static_cast<const int32_t*>(epoch.data_ptr()),
        static_cast<int32_t>(world_size));
  }
};
#endif

#if defined(SHARED_EP_OWNERS) || defined(SHARED_EP_MAX_TOKENS) || defined(SHARED_EP_TOP_K) || \
    defined(SHARED_EP_LOCAL_EXPERTS) || defined(SHARED_EP_BLOCK_M) || defined(SHARED_EP_THREADS)
#if !defined(SHARED_EP_OWNERS) || !defined(SHARED_EP_MAX_TOKENS) || !defined(SHARED_EP_TOP_K) || \
    !defined(SHARED_EP_LOCAL_EXPERTS) || !defined(SHARED_EP_BLOCK_M) || !defined(SHARED_EP_THREADS)
#error "SharedEP route-prep specialization is incomplete"
#endif

constexpr int kOwners = SHARED_EP_OWNERS;
constexpr int kMaxTokens = SHARED_EP_MAX_TOKENS;
constexpr int kTopK = SHARED_EP_TOP_K;
constexpr int kNumel = kOwners * kMaxTokens * kTopK;
constexpr int kLocalExperts = SHARED_EP_LOCAL_EXPERTS;
constexpr int kBlockM = SHARED_EP_BLOCK_M;
constexpr int kThreads = SHARED_EP_THREADS;
constexpr int kMaxSorted = kNumel + kLocalExperts * (kBlockM - 1);
constexpr int kMaxBlocks = (kMaxSorted + kBlockM - 1) / kBlockM;

static_assert(kOwners > 0 && kMaxTokens > 0 && kTopK > 0);
static_assert(kLocalExperts > 0 && kLocalExperts <= kThreads);
static_assert(kBlockM > 0);
static_assert(kThreads > 0 && kThreads <= 1024 && kThreads % kRouteThreadGroupSize == 0);

__global__ void shared_ep_route_prep_kernel(
    const int32_t* __restrict__ global_ids,
    const float* __restrict__ global_weights,
    const uint32_t* __restrict__ ready_signals,
    const int32_t* __restrict__ ready_epoch,
    int64_t ids_owner_stride,
    int64_t ids_token_stride,
    int64_t weights_owner_stride,
    int64_t weights_token_stride,
    int32_t* __restrict__ local_ids,
    float* __restrict__ local_weights,
    int32_t* __restrict__ sorted_ids,
    int32_t max_sorted,
    int32_t* __restrict__ expert_ids,
    int32_t max_blocks,
    int32_t* __restrict__ total_padded,
    int32_t local_expert_start) {
  __shared__ int32_t counts[kLocalExperts];
  __shared__ int32_t starts[kLocalExperts + 1];
  __shared__ int32_t cursors[kLocalExperts];

  const int tid = threadIdx.x;
  const uint32_t expected_epoch = static_cast<uint32_t>(*ready_epoch);
  if (tid < kOwners) {
    uint32_t observed_epoch;
    do {
      observed_epoch = shared_ep_load_acquire_system(ready_signals + tid);
    } while (observed_epoch != expected_epoch);
  }
  __syncthreads();

  if (tid < kLocalExperts) counts[tid] = 0;
  __syncthreads();

  for (int index = tid; index < kNumel; index += blockDim.x) {
    const int owner = index / (kMaxTokens * kTopK);
    const int owner_route = index - owner * kMaxTokens * kTopK;
    const int token = owner_route / kTopK;
    const int slot = owner_route - token * kTopK;
    const int32_t global_expert = global_ids[owner * ids_owner_stride + token * ids_token_stride + slot];
    const int32_t local_expert = global_expert - local_expert_start;
    const bool valid = local_expert >= 0 && local_expert < kLocalExperts;
    local_ids[index] = valid ? local_expert : -1;
    local_weights[index] = global_weights[owner * weights_owner_stride + token * weights_token_stride + slot];
    if (valid) atomicAdd(&counts[local_expert], 1);
  }
  __syncthreads();

  if (tid == 0) {
    int32_t cursor = 0;
    for (int expert = 0; expert < kLocalExperts; ++expert) {
      starts[expert] = cursor;
      cursors[expert] = cursor;
      cursor += ((counts[expert] + kBlockM - 1) / kBlockM) * kBlockM;
    }
    starts[kLocalExperts] = cursor;
    *total_padded = cursor;
  }
  __syncthreads();

  const int32_t padded = starts[kLocalExperts];
  for (int index = tid; index < padded; index += blockDim.x) {
    sorted_ids[index] = kNumel;
  }
  for (int index = tid; index < max_blocks; index += blockDim.x) {
    expert_ids[index] = -1;
  }
  __syncthreads();

  if (tid < kLocalExperts) {
    for (int offset = starts[tid]; offset < starts[tid + 1]; offset += kBlockM) {
      expert_ids[offset / kBlockM] = tid;
    }
  }
  __syncthreads();

  for (int index = tid; index < kNumel; index += blockDim.x) {
    const int32_t expert = local_ids[index];
    if (expert >= 0) {
      const int32_t position = atomicAdd(&cursors[expert], 1);
      sorted_ids[position] = index;
    }
  }
}

struct SharedEpRoutePrepKernel {
  static void
  run(tvm::ffi::TensorView global_ids,
      tvm::ffi::TensorView global_weights,
      tvm::ffi::TensorView ready_signals,
      tvm::ffi::TensorView ready_epoch,
      tvm::ffi::TensorView local_ids,
      tvm::ffi::TensorView local_weights,
      tvm::ffi::TensorView sorted_ids,
      tvm::ffi::TensorView expert_ids,
      tvm::ffi::TensorView total_padded,
      int64_t local_expert_start) {
    using namespace host;

    RuntimeCheck(
        global_ids.device().device_type == kSharedEpDeviceType, "SharedEP route ids must use the active GPU platform");
    RuntimeCheck(
        global_weights.device().device_type == kSharedEpDeviceType,
        "SharedEP route weights must use the active GPU platform");
    RuntimeCheck(
        ready_signals.device().device_type == kSharedEpDeviceType,
        "SharedEP ready signals must use the active GPU platform");
    RuntimeCheck(
        ready_epoch.device().device_type == kSharedEpDeviceType,
        "SharedEP ready epoch must use the active GPU platform");
    RuntimeCheck(
        global_ids.device().device_id == global_weights.device().device_id,
        "SharedEP route ids and weights must use the same GPU device");
    RuntimeCheck(is_type<int32_t>(global_ids.dtype()), "SharedEP route ids must be int32");
    RuntimeCheck(is_type<float>(global_weights.dtype()), "SharedEP route weights must be float32");
    RuntimeCheck(is_type<uint8_t>(ready_signals.dtype()), "SharedEP ready signals must be uint8 storage");
    RuntimeCheck(is_type<int32_t>(ready_epoch.dtype()), "SharedEP ready epoch must be int32");
    RuntimeCheck(is_type<int32_t>(local_ids.dtype()), "SharedEP local route ids must be int32");
    RuntimeCheck(is_type<float>(local_weights.dtype()), "SharedEP local route weights must be float32");
    RuntimeCheck(is_type<int32_t>(sorted_ids.dtype()), "SharedEP sorted route ids must be int32");
    RuntimeCheck(is_type<int32_t>(expert_ids.dtype()), "SharedEP expert ids must be int32");
    RuntimeCheck(is_type<int32_t>(total_padded.dtype()), "SharedEP padded route count must be int32");
    const int device_id = global_ids.device().device_id;
    RuntimeCheck(
        ready_signals.device().device_id == device_id && ready_epoch.device().device_id == device_id &&
            local_ids.device().device_type == kSharedEpDeviceType &&
            local_weights.device().device_type == kSharedEpDeviceType &&
            sorted_ids.device().device_type == kSharedEpDeviceType &&
            expert_ids.device().device_type == kSharedEpDeviceType &&
            total_padded.device().device_type == kSharedEpDeviceType && local_ids.device().device_id == device_id &&
            local_weights.device().device_id == device_id && sorted_ids.device().device_id == device_id &&
            expert_ids.device().device_id == device_id && total_padded.device().device_id == device_id,
        "SharedEP route-prep tensors must use the same GPU device");
    RuntimeCheck(global_ids.dim() == 3, "SharedEP route ids must be three-dimensional");
    RuntimeCheck(global_weights.dim() == 3, "SharedEP route weights must be three-dimensional");
    RuntimeCheck(
        global_ids.size(0) == kOwners && global_ids.size(1) == kMaxTokens && global_ids.size(2) == kTopK,
        "SharedEP route id shape does not match the JIT specialization");
    RuntimeCheck(
        global_weights.size(0) == kOwners && global_weights.size(1) == kMaxTokens && global_weights.size(2) == kTopK,
        "SharedEP route weight shape does not match the JIT specialization");
    RuntimeCheck(global_ids.stride(2) == 1, "SharedEP route id columns must be contiguous");
    RuntimeCheck(global_weights.stride(2) == 1, "SharedEP route weight columns must be contiguous");
    RuntimeCheck(
        local_ids.numel() == kNumel && local_weights.numel() == kNumel, "invalid SharedEP local route output shape");
    RuntimeCheck(sorted_ids.numel() == kMaxSorted, "invalid SharedEP sorted route output shape");
    RuntimeCheck(expert_ids.numel() == kMaxBlocks, "invalid SharedEP expert output shape");
    RuntimeCheck(total_padded.numel() == 1, "total_padded must have one element");
    RuntimeCheck(ready_signals.numel() >= kOwners * sizeof(uint32_t), "SharedEP ready signal storage is too small");
    RuntimeCheck(ready_epoch.numel() == 1, "SharedEP ready epoch must have one element");

    auto device = global_ids.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);
    LaunchKernel(dim3(1), dim3(kThreads), stream)(
        shared_ep_route_prep_kernel,
        static_cast<const int32_t*>(global_ids.data_ptr()),
        static_cast<const float*>(global_weights.data_ptr()),
        static_cast<const uint32_t*>(ready_signals.data_ptr()),
        static_cast<const int32_t*>(ready_epoch.data_ptr()),
        global_ids.stride(0),
        global_ids.stride(1),
        global_weights.stride(0),
        global_weights.stride(1),
        static_cast<int32_t*>(local_ids.data_ptr()),
        static_cast<float*>(local_weights.data_ptr()),
        static_cast<int32_t*>(sorted_ids.data_ptr()),
        static_cast<int32_t>(sorted_ids.numel()),
        static_cast<int32_t*>(expert_ids.data_ptr()),
        static_cast<int32_t>(expert_ids.numel()),
        static_cast<int32_t*>(total_padded.data_ptr()),
        static_cast<int32_t>(local_expert_start));
  }
};

#endif

}  // namespace
