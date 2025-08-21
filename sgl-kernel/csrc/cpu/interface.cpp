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

void shm_allreduce(torch::Tensor& data, int64_t op) {
  RECORD_FUNCTION("sgl-kernel::shm_allreduce", std::vector<c10::IValue>({data}));

  TORCH_CHECK(op == c10d::ReduceOp::SUM, "Only torch.distributed.ReduceOp.SUM is supported");

  auto numel = data.numel();
  int data_size = numel * data.element_size();
  all_reduce_outer_loop(data, numel, data_size);

  return;
}

torch::Tensor shm_allgather(torch::Tensor& data, int64_t dim) {
  RECORD_FUNCTION("sgl-kernel::shm_allgather", std::vector<c10::IValue>({data}));

  auto numel = data.numel();
  int data_size = numel * data.element_size();
  if (dim < 0) {
    dim += data.dim();
  }
  std::vector<int64_t> result_shape = data.sizes().vec();
  result_shape[dim] *= world_size;
  torch::Tensor result_tensor = torch::empty(result_shape, data.options());
  return all_gather(result_tensor, data, dim, numel, data_size);
}
