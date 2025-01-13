// reference: https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.14/cpp/tensorrt_llm/kernels/customAllReduceKernels.h

#include <c10/cuda/CUDAStream.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "trt_reduce_internal.cuh"

using namespace trt_llm;

using fptr_t = int64_t;

class AllReduceMeta {
 public:
  AllReduceMeta(int64_t rank_id, int64_t world_size, const std::vector<fptr_t>& buffers,
                const std::vector<fptr_t>& barrier_in, const std::vector<fptr_t>& barrier_out) {
    this->rank_id = (int)rank_id;
    this->world_size = (int)world_size;
    this->buffers = buffers;
    this->barrier_in = barrier_in;
    this->barrier_out = barrier_out;
  }

 public:
  int world_size;
  int rank_id;
  std::vector<fptr_t> buffers;
  std::vector<fptr_t> barrier_in;
  std::vector<fptr_t> barrier_out;
  int barrier_flag = 1;
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

fptr_t init_custom_ar(int64_t rank_id, int64_t world_size, const std::vector<fptr_t>& buffers,
                      const std::vector<fptr_t>& barrier_in, const std::vector<fptr_t>& barrier_out) {
  auto m = new AllReduceMeta(rank_id, world_size, buffers, barrier_in, barrier_out);
  return (fptr_t)m;
}

void dispose(fptr_t _fa) {
  auto fa = reinterpret_cast<AllReduceMeta*>(_fa);
  delete fa;
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

  for (int i = 0; i < world_size; ++i) {
    params.peer_comm_buffer_ptrs[i] = reinterpret_cast<void*>(m->buffers[i]);
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
