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

// reference: https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.14/cpp/tensorrt_llm/kernels/customAllReduceKernels.h

#include <c10/cuda/CUDAStream.h>

#include <cassert>

#include "trt_reduce_internal.cuh"
#include "utils.h"

using namespace trt_llm;

using fptr_t = int64_t;
using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;

class AllReduceMeta {
 public:
  AllReduceMeta(
      int64_t rank_id,
      int64_t world_size,
      torch::Tensor& rank_data,
      const std::vector<fptr_t>& buffers,
      const std::vector<fptr_t>& tmp_result_buffers,
      const std::vector<fptr_t>& barrier_in,
      const std::vector<fptr_t>& barrier_out) {
    this->rank_id = (int)rank_id;
    this->world_size = (int)world_size;
    this->barrier_in = barrier_in;
    this->barrier_out = barrier_out;
    this->tmp_result_buffers = tmp_result_buffers;

    this->rank_data_base = reinterpret_cast<RankData*>(rank_data.data_ptr());
    RankData data;
    for (int i = 0; i < world_size; i++) {
      data.ptrs[i] = (void*)buffers[i];
    }
    auto d_data = this->rank_data_base++;
    CHECK_CUDA_SUCCESS(cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    this->buffers = d_data;
  }

  ~AllReduceMeta() {
    for (auto [_, ptr] : ipc_handles_) {
      CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(ptr));
    }
  }

 public:
  int world_size;
  int rank_id;
  std::vector<fptr_t> barrier_in;
  std::vector<fptr_t> barrier_out;
  std::vector<fptr_t> tmp_result_buffers;
  int barrier_flag = 1;
  RankData* buffers;
  RankData* rank_data_base;
  std::vector<void*> graph_unreg_buffers;
  std::map<IPC_KEY, char*> ipc_handles_;
};

// Get the number of bits for a given data type.
inline int get_bits(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return 32;
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
      return 16;
    default:
      assert(false && "Unsupported data type");
  }
}

// Check if customized all-reduce kernels can be applied.
inline bool CanApplyCustomAllReduce(int64_t num_elements, at::ScalarType dtype) {
  // The customized all-reduce kernel has the following requirement(s).
  return num_elements % (16 / ((get_bits(dtype) + 7) / 8)) == 0;
}

fptr_t init_custom_ar(
    int64_t rank_id,
    int64_t world_size,
    torch::Tensor& rank_data,
    const std::vector<fptr_t>& buffers,
    const std::vector<fptr_t>& tmp_result_buffers,
    const std::vector<fptr_t>& barrier_in,
    const std::vector<fptr_t>& barrier_out) {
  auto m = new AllReduceMeta(rank_id, world_size, rank_data, buffers, tmp_result_buffers, barrier_in, barrier_out);
  return (fptr_t)m;
}

void dispose(fptr_t _fa) {
  auto fa = reinterpret_cast<AllReduceMeta*>(_fa);
  delete fa;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  AllReduceMeta* m = reinterpret_cast<AllReduceMeta*>(_fa);
  auto num_buffers = m->graph_unreg_buffers.size();
  auto handle_sz = sizeof(cudaIpcMemHandle_t);
  std::string handles(handle_sz * num_buffers, static_cast<char>(0));
  std::vector<int64_t> offsets(num_buffers);
  for (int i = 0; i < num_buffers; i++) {
    auto ptr = m->graph_unreg_buffers[i];
    void* base_ptr;
    // note: must share the base address of each allocation, or we get wrong
    // address
    if (cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS) {
      assert(false && "failed to get pointer attr");
    }

    CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
    offsets[i] = ((char*)ptr) - ((char*)base_ptr);
  }
  std::vector<int64_t> bytes(handles.begin(), handles.end());
  return std::make_pair(bytes, offsets);
}

char* open_ipc_handle(AllReduceMeta* meta, const void* ipc_handle) {
  auto [it, new_handle] = meta->ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
  if (new_handle) {
    char* ipc_ptr;
    CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(
        (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)ipc_handle), cudaIpcMemLazyEnablePeerAccess));
    it->second = ipc_ptr;
  }
  return it->second;
}

// Note: when registering graph buffers, we intentionally choose to not
// deduplicate the addresses. That means if the allocator reuses some
// addresses, they will be registered again. This is to account for the remote
// possibility of different allocation patterns between ranks. For example,
// rank 1 may get the same input address for the second allreduce, but rank 2
// got a different address. IPC handles have internal reference counting
// mechanism so overhead should be small.
void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  AllReduceMeta* m = reinterpret_cast<AllReduceMeta*>(_fa);
  std::vector<std::string> handle_bytes;
  handle_bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    handle_bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  auto num_buffers = m->graph_unreg_buffers.size();
  std::vector<RankData> rank_data(num_buffers);
  for (int i = 0; i < num_buffers; i++) {
    auto self_ptr = m->graph_unreg_buffers[i];
    auto& rd = rank_data[i];
    for (int j = 0; j < m->world_size; j++) {
      if (j != m->rank_id) {
        char* handle = open_ipc_handle(m, &handle_bytes[j][i * sizeof(cudaIpcMemHandle_t)]);
        handle += offsets[j][i];
        rd.ptrs[j] = handle;
      } else {
        rd.ptrs[j] = self_ptr;
      }
    }
  }
  CHECK_CUDA_SUCCESS(
      cudaMemcpy(m->rank_data_base, rank_data.data(), sizeof(RankData) * num_buffers, cudaMemcpyHostToDevice));
  m->rank_data_base += num_buffers;
  m->graph_unreg_buffers.clear();
}

void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out) {
  AllReduceMeta* m = reinterpret_cast<AllReduceMeta*>(_fa);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto num_elements = inp.numel();
  auto dtype = inp.scalar_type();
  AllReduceStrategyType strategy = SelectImplementation(num_elements * ((get_bits(dtype) + 7) / 8), m->world_size);

  // should be gurantee in python code
  assert(strategy == AllReduceStrategyType::ONESHOT || strategy == AllReduceStrategyType::TWOSHOT);
  assert(CanApplyCustomAllReduce(num_elements, dtype));

  // Initialize the all-reduce kernel arguments.
  int world_size = m->world_size;

  AllReduceParams params;
  params.ranks_per_node = world_size;
  params.rank = m->rank_id;
  params.local_rank = m->rank_id;
  params.local_input_buffer_ptr = inp.data_ptr();
  params.local_output_buffer_ptr = out.data_ptr();
  params.elts_total = inp.numel();
  params.elts_size = inp.element_size();
  params.barrier_flag = ++(m->barrier_flag);

  cudaStreamCaptureStatus status;
  CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));
  params.is_capturing = (status == cudaStreamCaptureStatusActive);
  if (params.is_capturing) {
    params.peer_comm_buffer_ptrs = m->rank_data_base + m->graph_unreg_buffers.size();
    m->graph_unreg_buffers.push_back(params.local_input_buffer_ptr);
  } else {
    params.peer_comm_buffer_ptrs = m->buffers;
  }

  for (int i = 0; i < world_size; ++i) {
    params.tmp_result_buffers[i] = reinterpret_cast<uint32_t*>(m->tmp_result_buffers[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(m->barrier_in[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(m->barrier_out[i]);
  }

  auto data_type = out.scalar_type();
  trtCustomAllReduce(params, data_type, strategy, stream);
}
