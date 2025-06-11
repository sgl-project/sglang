#include <ATen/record_function.h>
#include <torch/all.h>

#include "shm.h"

// Communication settings
static int world_rank = -1;
static int world_size = -1;

static bool is_initialized = false;

static bool all_ranks_local_p = false;

void initialize(int64_t size, int64_t rank) {
  if (is_initialized) {
    return;
  }

  // Check whether all ranks is on the same physical machine.
  // If true, we will use an SHM based low latency allreduce

  auto ls_string = std::getenv("LOCAL_SIZE");
  int ls = 0;
  if (ls_string != NULL) {
    ls = std::stoi(std::getenv("LOCAL_SIZE"));
  }

  if (size >= 1 && size == ls) {
    all_ranks_local_p = true;
  }

  world_size = size;
  world_rank = rank;
  is_initialized = true;

  const char* addr_string = std::getenv("MASTER_ADDR");
  if (addr_string == NULL) {
    addr_string = "";
  }
  const char* port_string = std::getenv("MASTER_PORT");
  if (port_string == NULL) {
    port_string = "";
  }

  if (all_ranks_local_p) {
    shm_initialize(size, rank, addr_string, port_string);
  }
}

void shm_allreduce(
    torch::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group, c10::intrusive_ptr<c10d::ReduceOp> op) {
  RECORD_FUNCTION("sgl-kernel::shm_allreduce", std::vector<c10::IValue>({data}));

  TORCH_CHECK(op == c10d::ReduceOp::SUM, "Only torch.distributed.ReduceOp.SUM is supported");

  auto numel = data.numel();

  int data_size = 0;
  bool data_type_fallback = false;

  switch (data.scalar_type()) {
    case c10::ScalarType::BFloat16:
      data_size = numel * 2;
      break;
    case c10::ScalarType::Float:
      data_size = numel * 4;
      break;
    default:
      data_type_fallback = true;
  }

  if (data_type_fallback || !all_ranks_local_p) {
    // Fallback to torch distributed allreduce
    std::vector<torch::Tensor> tensors = {data};
    process_group->allreduce(tensors)->wait();
  } else {
    all_reduce_outer_loop(data, numel, data_size);
  }

  return;
}

torch::Tensor shm_allgather(torch::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group, int64_t dim) {
  RECORD_FUNCTION("sgl-kernel::shm_allgather", std::vector<c10::IValue>({data}));

  auto numel = data.numel();

  int data_size = 0;
  bool data_type_fallback = false;

  switch (data.scalar_type()) {
    case c10::ScalarType::BFloat16:
      data_size = numel * 2;
      break;
    case c10::ScalarType::Float:
      data_size = numel * 4;
      break;
    default:
      data_type_fallback = true;
  }
  if (dim < 0) {
    dim += data.dim();
  }
  if (data_type_fallback || !all_ranks_local_p) {
    // Fallback to torch distributed allreduce
    std::vector<std::vector<torch::Tensor>> output_tensors(1);
    auto world_size = process_group->getSize();
    for (int i = 0; i < world_size; i++) {
      output_tensors[0].push_back(torch::empty_like(data));
    }
    std::vector<torch::Tensor> input_tensors = {data};
    process_group->allgather(output_tensors, input_tensors)->wait();
    return torch::cat(output_tensors[0], dim).contiguous();
  }
  std::vector<int64_t> result_shape = data.sizes().vec();
  result_shape[dim] *= world_size;
  torch::Tensor result_tensor = torch::empty(result_shape, data.options());
  return all_gather(result_tensor, data, dim, numel, data_size);
}
